"""Training semantics and descriptive development metrics, never release gates."""
from collections import Counter
import math
import torch
from .famu_contract_model import MODEL_CLASSES

TARGETS = {'qa':'question_answer','risk_status':'risk_established_status',
           'severity':'final_severity','excerpt_severity':'excerpt_severity'}
MAIN_TASKS = ('qa','risk_status','severity')


def audit_rows(rows):
    ids, groups, hashes = set(), {}, {}
    for row in rows:
        sid=row['sample_id']
        if sid in ids: raise ValueError('Duplicate sample id')
        ids.add(sid)
        for key, owners in [('contract_id',groups),('source_contract_sha256',hashes)]:
            value=row.get(key)
            if value:
                owners.setdefault(value,set()).add(row['fold_id'])
        for task,target in TARGETS.items():
            if row[target] is not None and row[target] not in MODEL_CLASSES[task]:
                raise ValueError('Invalid semantic target: '+task)
        for mask,target in [('qa_loss_mask','question_answer'),('severity_loss_mask','final_severity')]:
            if mask in row and row[mask] != int(row[target] is not None):
                raise ValueError('Loss mask disagrees with observed label')
        if row['excerpt_severity'] is not None and any(row[TARGETS[t]] is not None for t in MAIN_TASKS):
            raise ValueError('Excerpt labels must not become question labels')
    if any(len(v)!=1 for v in list(groups.values())+list(hashes.values())):
        raise ValueError('Source or source hash crosses folds')
    return {'rows':len(rows),'source_groups':len(groups),'label_maps':{
        t:{'id2label':list(c),'label2id':{label:i for i,label in enumerate(c)}} for t,c in MODEL_CLASSES.items()},
        'folds':{str(f):label_support([r for r in rows if r['fold_id']!=f]) for f in sorted({r['fold_id'] for r in rows})}}


def label_support(rows):
    return {t:{'effective_labels':sum(r[k] is not None for r in rows),
        'masked_labels':sum(r[k] is None for r in rows),
        'classes':{c:{'records':sum(r[k]==c for r in rows),
                      'source_groups':len({r['contract_id'] for r in rows if r[k]==c})} for c in MODEL_CLASSES[t]},
        'unsupported_classes':[c for c in MODEL_CLASSES[t] if not any(r[k]==c for r in rows)]}
        for t,k in TARGETS.items()}


def make_task_weights(rows):
    # Import lazily to keep reporting usable without running training.
    from scripts.train_question_evidence_models import base_weights
    from .semantic_risk import balanced_weights
    result={}
    for task,target in TARGETS.items():
        selected=[r for r in rows if r[target] is not None]
        raw=balanced_weights([r[target] for r in selected],base_weights(selected)) if selected else []
        total=sum(raw)
        result[task]={r['sample_id']:w/total for r,w in zip(selected,raw)}
    return result


def record_losses(logits,row,weights,trained_classes,tasks):
    """Each task has unit mass over valid records, irrespective of window count."""
    result={}
    for task in tasks:
        label=row[TARGETS[task]]
        if label is None: continue
        classes=MODEL_CLASSES[task]
        if label not in trained_classes[task]: raise ValueError('Target lacks training support')
        values=logits[task]
        mask=torch.tensor([c in trained_classes[task] for c in classes],device=values.device)
        target=torch.tensor([classes.index(label)],device=values.device)
        result[task]=torch.nn.functional.cross_entropy(values.masked_fill(~mask,-torch.inf)[None,:],target)*weights[task][row['sample_id']]
    return result


def gradient_geometry(main,aux,parameters):
    """No optimizer mutation; unused/initially-zero LoRA gradients are valid."""
    a=torch.autograd.grad(main,parameters,retain_graph=True,allow_unused=True)
    b=torch.autograd.grad(aux,parameters,retain_graph=True,allow_unused=True)
    dot=norm_a=norm_b=0.0
    for x,y in zip(a,b):
        if x is not None: norm_a+=float(x.detach().double().square().sum())
        if y is not None: norm_b+=float(y.detach().double().square().sum())
        if x is not None and y is not None: dot+=float((x.detach().double()*y.detach().double()).sum())
    denominator=math.sqrt(norm_a*norm_b)
    return {'main_l2':math.sqrt(norm_a),'auxiliary_l2':math.sqrt(norm_b),
            'cosine':dot/denominator if denominator else None,
            'conflict':dot<0 if denominator else None,'scope':'shared_LoRA_only'}


def fold_majority(fit,test,task):
    target=TARGETS[task]; counts=Counter(r[target] for r in fit if r[target] is not None)
    if not counts: return []
    # Training-fold counts only; deterministic semantic-order tie breaking.
    label=max(MODEL_CLASSES[task],key=lambda c:counts[c])
    return [{'sample_id':r['sample_id'],'group_id':r['contract_id'],'fold_id':r['fold_id'],
             'actual':r[target],'predicted':label,'score':1.0} for r in test if r[target] is not None]


def selective_errors(rows,coverages=(1.0,.9,.75,.5)):
    ordered=sorted(rows,key=lambda r:(-r['score'],r['sample_id']))
    output=[]
    for coverage in coverages:
        count=math.ceil(len(rows)*coverage); kept=ordered[:count]
        errors=sum(r['actual']!=r['predicted'] for r in kept)
        output.append({'requested_coverage':coverage,'kept':count,'reviewed':len(rows)-count,
            'actual_coverage':count/len(rows) if rows else None,'remaining_errors':errors,
            'retained_error_rate':errors/count if count else None})
    return {'scope':'offline_equal_review_budget_uncalibrated_ranking_not_runtime_automation',
            'tie_break':'sample_id, never correctness','curve':output}


def classification_report(rows,task):
    from scripts.train_confirmed_reconciliation import score
    classes=MODEL_CLASSES[task]
    if not rows: return {'count':0,'status':'inactive_or_no_observed_labels'}
    if any(r['actual'] not in classes or r['predicted'] not in classes for r in rows):
        raise ValueError('Prediction labels do not match task semantics')
    metrics=score([r['actual'] for r in rows],[r['predicted'] for r in rows],classes)
    matrix={a:{p:sum(r['actual']==a and r['predicted']==p for r in rows) for p in classes} for a in classes}
    groups={g:[r for r in rows if r['group_id']==g] for g in sorted({r['group_id'] for r in rows})}
    per_source={g:{'count':len(v),'errors':sum(r['actual']!=r['predicted'] for r in v),
                   'error_sample_ids':[r['sample_id'] for r in v if r['actual']!=r['predicted']]} for g,v in groups.items()}
    metrics.update(confusion_matrix=matrix,per_source=per_source,
        source_balanced_accuracy=sum(1-v['errors']/v['count'] for v in per_source.values())/len(groups),
        prediction_counts=dict(Counter(r['predicted'] for r in rows)),selective_prediction=selective_errors(rows))
    def rate(subset,predicate):
        numerator=sum(predicate(r) for r in subset)
        return {'count':numerator,'denominator':len(subset),'rate':numerator/len(subset) if subset else None}
    if task in ('severity','excerpt_severity'):
        metrics['critical_to_high']=rate([r for r in rows if r['actual']=='critical'],lambda r:r['predicted']=='high')
        metrics['high_critical_to_low_medium']=rate([r for r in rows if r['actual'] in ('high','critical')],lambda r:r['predicted'] in ('low','medium'))
    if task=='qa': metrics['insufficient_material_proposed_determinate']=rate([r for r in rows if r['actual']=='U'],lambda r:r['predicted']!='U')
    if task=='risk_status': metrics['nonrisk_false_alarm']=rate([r for r in rows if r['actual']=='not_established'],lambda r:r['predicted']=='established')
    return metrics


def error_overlap(first,second):
    a={r['sample_id']:r for r in first}; b={r['sample_id']:r for r in second}
    if len(a)!=len(first) or len(b)!=len(second) or set(a)!=set(b): raise ValueError('Paired comparison needs identical unique cases')
    result={k:[] for k in ('both_correct','only_first_correct','only_second_correct','both_wrong')}
    for sid,x in a.items():
        y=b[sid]
        if (x['actual'],x['group_id'])!=(y['actual'],y['group_id']): raise ValueError('Paired targets/sources mismatch')
        correct=x['predicted']==x['actual'],y['predicted']==y['actual']
        key={(True,True):'both_correct',(True,False):'only_first_correct',(False,True):'only_second_correct',(False,False):'both_wrong'}[correct]
        result[key].append(sid)
    return {key:{'count':len(ids),'sample_ids':ids} for key,ids in result.items()}
