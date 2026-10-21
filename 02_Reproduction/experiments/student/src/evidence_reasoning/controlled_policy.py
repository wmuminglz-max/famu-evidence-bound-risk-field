"""Fixed-input head experiments. Costs are research diagnostics, not legal policy."""
from __future__ import annotations
import copy
import math
import random
import statistics
import torch
from .bandit_policy import distribution, clipped_objective
from .famu_contract_model import MODEL_CLASSES
from .self_improvement import action_rewards
from .training_diagnostics import TARGETS, classification_report


def cost_spec(version='historical_C0'):
    if version not in {'historical_C0', 'symmetric_ordinal'}:
        raise ValueError('A business cost matrix needs separate confirmation; no implicit defaults')
    tables={}
    for task,labels in MODEL_CLASSES.items():
        raw=[[1-r for r in action_rewards(task,a)] for a in labels]
        if version=='symmetric_ordinal' and task in {'severity','excerpt_severity'}:
            raw=[[float(abs(i-j)) for j in range(len(labels))] for i in range(len(labels))]
        scale=max(max(row) for row in raw)
        tables[task]={'labels':list(labels),'raw':raw,'scale':scale,
                      'normalized':[[v/scale for v in row] for row in raw]}
    return {'version':version,'business_confirmed':False,
            'normalization':'divide each complete task matrix by its maximum; fixed before scores; range [0,1]',
            'row_semantics':'actual','column_semantics':'predicted','tasks':tables}


def validate_cost_spec(spec):
    if set(spec['tasks'])!=set(MODEL_CLASSES):raise ValueError('Four task matrices required')
    for task,labels in MODEL_CLASSES.items():
        table=spec['tasks'][task];n=len(labels)
        if table['labels']!=list(labels):raise ValueError('Cost label order mismatch')
        raw=table['raw'];scale=table['scale']
        if len(raw)!=n or any(len(r)!=n for r in raw):raise ValueError('Non-square cost matrix')
        if any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for r in raw for v in r):
            raise ValueError('Costs must be finite and nonnegative')
        if any(raw[i][i]!=0 for i in range(n)) or scale!=max(map(max,raw)) or scale<=0:
            raise ValueError('Invalid diagonal or frozen scale')
        if table['normalized']!=[[v/scale for v in row] for row in raw]:raise ValueError('Normalization mismatch')
    return spec


def effective_weights(rows, task):
    from scripts.train_question_evidence_models import base_weights
    from .semantic_risk import balanced_weights
    selected=sorted((r for r in rows if r[TARGETS[task]] is not None),key=lambda r:r['sample_id'])
    if not selected:return [],torch.empty(0),[]
    base=base_weights(selected);raw=balanced_weights([r[TARGETS[task]] for r in selected],base)
    weights=torch.tensor(raw,dtype=torch.float32);weights/=weights.sum()
    audit=[{'sample_id':r['sample_id'],'group_id':r['contract_id'],'cluster_id':r['risk_cluster_id'],
            'actual':r[TARGETS[task]],'base_group_cluster_weight':b,'class_balanced_weight':c,
            'effective_task_weight':float(w),'effective_total_objective_weight':float(w)/4}
           for r,b,c,w in zip(selected,base,raw,weights)]
    return selected,weights,audit


def weighted_cost_audit(records,table):
    labels=table['labels'];mass={c:sum(r['effective_task_weight'] for r in records if r['actual']==c) for c in labels}
    return {'class_mass':mass,'weight_total':sum(mass.values()),
            'constant_raw_cost':{p:sum(mass[a]*table['raw'][i][j] for i,a in enumerate(labels)) for j,p in enumerate(labels)},
            'constant_raw_reward':{p:sum(mass[a]*(1-table['raw'][i][j]) for i,a in enumerate(labels)) for j,p in enumerate(labels)},
            'constant_normalized_cost':{p:sum(mass[a]*table['normalized'][i][j] for i,a in enumerate(labels)) for j,p in enumerate(labels)}}


def build_fixed_tasks(rows,features,classes,spec):
    validate_cost_spec(spec);tasks={};audit={}
    for task,labels in MODEL_CLASSES.items():
        selected,weights,records=effective_weights(rows,task)
        if not selected:raise ValueError('Fixed experiment requires an observed target for every task')
        if any(r[TARGETS[task]] not in classes[task] for r in selected):raise ValueError('Untrained target')
        targets=torch.tensor([labels.index(r[TARGETS[task]]) for r in selected])
        tasks[task]={'features':torch.stack([features[r['sample_id']][0].detach() for r in selected]),
                     'targets':targets,'weights':weights,'supported':[c in classes[task] for c in labels],
                     'costs':torch.tensor(spec['tasks'][task]['normalized'])[targets],
                     'sample_ids':[r['sample_id'] for r in selected]}
        if tasks[task]['features'].ndim!=2:raise ValueError('Only original feature version zero is allowed')
        audit[task]={'records':records,**weighted_cost_audit(records,spec['tasks'][task])}
    return tasks,audit


def exact_expected_cost(probabilities,costs,weights):
    if probabilities.shape!=costs.shape or weights.shape!=(len(probabilities),):raise ValueError('Cost shape mismatch')
    return (weights*(probabilities*costs).sum(-1)).sum()


def sampled_ppo_loss(logp,oldlog,advantage,weights,clip=.2):
    if oldlog.requires_grad or advantage.requires_grad:raise ValueError('Old policy and advantages must be detached')
    return clipped_objective(logp,oldlog,advantage,weights,clip)


def fit_fixed_heads(initial,tasks,method,parameter_mode='full',seed=20260930,rollouts=15,updates_per_rollout=4):
    if method not in {'sft','expected_cost','ppo'} or parameter_mode not in {'full','bias'}:
        raise ValueError('Invalid controlled arm')
    if not 1<=rollouts<=15 or not 1<=updates_per_rollout<=4:raise ValueError('Bounded schedule required')
    heads=copy.deepcopy(initial).eval();reference=copy.deepcopy(initial).eval()
    for p in reference.parameters():p.requires_grad_(False)
    for name,p in heads.named_parameters():p.requires_grad_(parameter_mode=='full' or name.endswith('.bias'))
    parameters=[p for p in heads.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=.0003,weight_decay=.01,foreach=False)
    action_rng=torch.Generator(device='cpu').manual_seed(seed)
    history=[]
    with torch.no_grad():
        refs={t:distribution(reference[t],d['features'],d['supported']).detach() for t,d in tasks.items()}
    for rollout in range(rollouts):
        batches={}
        with torch.no_grad():
            for t,d in tasks.items():
                old=distribution(heads[t],d['features'],d['supported']).detach()
                if method=='ppo':
                    actions=torch.multinomial(old,4,replacement=True,generator=action_rng)
                    reward=-d['costs']
                    advantage=(reward.gather(1,actions)-(old*reward).sum(-1,keepdim=True)).detach()
                    batches[t]=(old,actions,old.clamp_min(1e-12).log().gather(1,actions).detach(),advantage)
        for step in range(updates_per_rollout):
            optimizer.zero_grad();loss=0;details={}
            for t,d in tasks.items():
                p=distribution(heads[t],d['features'],d['supported']);logp=p.clamp_min(1e-12).log();w=d['weights']
                ce=-(w*logp.gather(1,d['targets'][:,None]).squeeze(1)).sum()
                kl=(w*(p*(logp-refs[t].clamp_min(1e-12).log())).sum(-1)).sum()
                entropy=-(w*(p*logp).sum(-1)).sum()
                expected=exact_expected_cost(p,d['costs'],w)
                objective=0*ce;clip_fraction=0.
                if method=='expected_cost':objective=expected
                if method=='ppo':
                    old,actions,oldlog,advantage=batches[t]
                    objective,clipped=sampled_ppo_loss(logp.gather(1,actions),oldlog,advantage,w[:,None]/4)
                    clip_fraction=float(clipped)
                loss+=(objective+.5*ce+.05*kl-.01*entropy)/len(tasks)
                details[t]={'CE':float(ce.detach()),'reference_KL':float(kl.detach()),'entropy':float(entropy.detach()),
                            'expected_cost':float(expected.detach()),'clip_fraction_sampled_before_update':clip_fraction}
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward()
            norms={t:math.sqrt(sum(float(p.grad.square().sum()) for p in heads[t].parameters() if p.grad is not None)) for t in tasks}
            norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True));optimizer.step()
            with torch.no_grad():
                for t,d in tasks.items():
                    p=distribution(heads[t],d['features'],d['supported']);support=torch.tensor(d['supported'])
                    details[t]['post_reference_KL']=float((d['weights']*(p*(p.clamp_min(1e-12).log()-refs[t].clamp_min(1e-12).log())).sum(-1)).sum())
                    if method=='ppo':
                        ratio=p[:,support]/batches[t][0][:,support].clamp_min(1e-12)
                        details[t].update(post_all_action_ratio_min=float(ratio.min()),post_all_action_ratio_max=float(ratio.max()),
                                          post_ratio_outside_clip_fraction=float(((ratio<.8)|(ratio>1.2)).float().mean()))
            history.append({'update':len(history)+1,'rollout':rollout+1,'inner_update':step+1,'loss':float(loss.detach()),
                            'global_gradient_norm':norm,'global_clip_scale':min(1.,1/(norm+1e-6)),
                            'task_gradient_norms':norms,'tasks':details})
    if parameter_mode=='bias' and any(not torch.equal(v,initial.state_dict()[k]) for k,v in heads.state_dict().items() if k.endswith('.weight')):
        raise ValueError('Bias probe changed a weight matrix')
    return heads,{'method':method,'parameter_mode':parameter_mode,'seed':seed,'trainable_parameters':sum(p.numel() for p in parameters),
                  'optimizer_initial_state_empty':True,'updates':len(history),'history':history,
                  'encoder_LoRA_fusion_updated':False,'action_generator_separate':True,
                  'data_order':'canonical sample_id, full batch every update; no data RNG',
                  'unsupported_rows_have_no_target_gradient':'AdamW decay may still move their stored values; masks unchanged'}


def evaluation(rows,task):
    result=classification_report(rows,task);labels=MODEL_CLASSES[task]
    if not rows:return result
    def rate(subset,predicate):
        n=sum(predicate(r) for r in subset)
        return {'count':n,'denominator':len(subset),'rate':n/len(subset) if subset else None}
    if task in {'severity','excerpt_severity'}:
        result['noncritical_false_critical']=rate([r for r in rows if r['actual']!='critical'],lambda r:r['predicted']=='critical')
        result['high_critical_any_underestimate']=rate([r for r in rows if r['actual'] in {'high','critical'}],lambda r:labels.index(r['predicted'])<labels.index(r['actual']))
    if task=='risk_status':result['unknown_false_established']=rate([r for r in rows if r['actual']=='unknown'],lambda r:r['predicted']=='established')
    result['costs']={}
    for name in ('historical_C0','symmetric_ordinal'):
        table=cost_spec(name)['tasks'][task]
        vals=[table['raw'][labels.index(r['actual'])][labels.index(r['predicted'])] for r in rows]
        groups=sorted({r['group_id'] for r in rows})
        result['costs'][name]={'raw_mean':statistics.mean(vals),'normalized_mean':statistics.mean(vals)/table['scale'],
            'source_balanced_raw_mean':statistics.mean(statistics.mean(v for r,v in zip(rows,vals) if r['group_id']==g) for g in groups)}
    for point in result['selective_prediction']['curve']:
        kept=sorted(rows,key=lambda r:(-r['score'],r['sample_id']))[:point['kept']]
        point['remaining_high_critical_to_low_medium']=sum(r['actual'] in {'high','critical'} and r['predicted'] in {'low','medium'} for r in kept)
        point['remaining_critical_to_high']=sum(r['actual']=='critical' and r['predicted']=='high' for r in kept)
        point['remaining_U_determinate']=sum(r['actual']=='U' and r['predicted']!='U' for r in kept)
    result['probabilities_calibrated']=False
    return result


def paired_source_bootstrap(first,second,task,seed=20260930,repetitions=1000):
    """Average seeds within each row BEFORE resampling source groups."""
    if len(first)!=len(second) or not first:raise ValueError('Paired seeds required')
    ids=[r['sample_id'] for r in first[0]];lookup=[]
    for left,right in zip(first,second):
        a={r['sample_id']:r for r in left};b={r['sample_id']:r for r in right}
        if len(a)!=len(left) or len(b)!=len(right) or set(a)!=set(ids) or set(b)!=set(ids):raise ValueError('Unpaired cases')
        if any((a[s]['actual'],a[s]['group_id'])!=(b[s]['actual'],b[s]['group_id']) for s in ids):raise ValueError('Unpaired sources/targets')
        lookup.append((a,b))
    table=cost_spec()['tasks'][task];labels=table['labels'];group_values={}
    for sid in ids:
        group=lookup[0][0][sid]['group_id'];values=[]
        for a,b in lookup:
            if (a[sid]['actual'],a[sid]['group_id'])!=(lookup[0][0][sid]['actual'],group):raise ValueError('Seed target drift')
            y=labels.index(a[sid]['actual'])
            values.append(table['raw'][y][labels.index(a[sid]['predicted'])]-table['raw'][y][labels.index(b[sid]['predicted'])])
        group_values.setdefault(group,[]).append(statistics.mean(values))
    per_source={g:statistics.mean(v) for g,v in group_values.items()};groups=sorted(per_source);rng=random.Random(seed)
    draws=sorted(statistics.mean(per_source[g] for g in rng.choices(groups,k=len(groups))) for _ in range(repetitions))
    return {'metric':'first minus second, source-balanced raw C0 cost; negative favors first',
            'mean':statistics.mean(per_source.values()),'percentile_95':[draws[int(.025*repetitions)],draws[int(.975*repetitions)-1]],
            'source_groups':len(groups),'per_source':per_source,'repetitions':repetitions,
            'interpretation':'descriptive development sensitivity, not independent legal acceptance; seeds are not extra sources'}
