"""Human evidence/condition annotations kept separate from observable model input.

These validators attest format and lineage, not annotator identity or legal truth.
No pending annotation is silently converted into a training target.
"""
from __future__ import annotations
import copy
import torch
from src.risk_grading import validate_severity_basis
from .contract_retrieval import validate_request, sha, prepare_features
from .schema import validate_input, validate_record, model_input, digest
from .logic import solve
from .decision import select_decision
from .training_diagnostics import TARGETS, MODEL_CLASSES

ROLES={'support','refute','exception','definition','irrelevant'}


def require_confirmation(value):
    if value.get('status')!='confirmed' or not all(isinstance(value.get(k),str) and value[k].strip() for k in ('reviewer_id','confirmed_at')):
        raise ValueError('New supervision requires a recorded human confirmation; drafts are not targets')


def validate_annotation(entry):
    request=validate_request(entry['request']); annotation=entry['annotation']
    require_confirmation(annotation)
    if entry.get('request_sha256')!=digest(entry['request']): raise ValueError('Annotated observable input changed')
    if not all(entry.get(k) for k in ('sample_id','group_id','root_id')) or type(entry.get('fold_id')) is not int:
        raise ValueError('Sample, parent-source group and fold are required')
    for t,value in entry['targets'].items():
        if t not in TARGETS or value is not None and value not in MODEL_CLASSES[t]: raise ValueError('Invalid target semantics')
    if entry['targets'].get('severity') is not None and entry['targets'].get('risk_status')!='established':
        raise ValueError('A conditional question grade requires a separately established risk')
    if type(annotation.get('evidence_complete')) is not bool: raise ValueError('Evidence completeness must be explicitly reviewed')
    sources={s['id']:s for s in request['sources']}; spans=annotation.get('spans',[]); ids=set()
    for span in spans:
        if not span.get('id') or span['id'] in ids: raise ValueError('Unique span IDs required')
        ids.add(span['id']); source=sources.get(span.get('source_id'))
        if source is None or not source.get('available',True) or source['id'] in request['withdrawn_sources']:
            raise ValueError('Evidence must exist in available supplied material')
        start,end=span.get('start'),span.get('end')
        if type(start) is not int or type(end) is not int or not 0<=start<end<=len(source['text']) or source['text'][start:end]!=span.get('quote'):
            raise ValueError('Span offsets/quote do not match the original source')
        if span.get('role') not in ROLES or not span.get('scope_id'): raise ValueError('Span needs explicit role and scope')
        if span.get('condition_relation') not in {None,'support','refute','irrelevant'}:
            raise ValueError('Condition pair relation must be explicitly support/refute/irrelevant or missing')
        if span['role'] in {'support','refute','irrelevant'} and span.get('condition_relation') is not None and span['condition_relation']!=span['role']:
            raise ValueError('Evidence role and condition-pair annotation disagree')
    dependency_ids=set()
    for dependency in annotation.get('source_dependencies',[]):
        if not dependency.get('id') or dependency['id'] in dependency_ids:
            raise ValueError('Unique dependency identifiers required')
        dependency_ids.add(dependency['id'])
        if dependency.get('referenced_by_span_id') not in ids or not dependency.get('description') or type(dependency.get('available')) is not bool:
            raise ValueError('Dependency requires its actual referring span and reviewed availability')
        supplied=dependency.get('source_ids')
        if not isinstance(supplied,list) or bool(supplied)!=dependency['available']:
            raise ValueError('Missing dependencies must not be replaced with fabricated sources')
        if any(sid not in sources or not sources[sid].get('available',True) or sid in request['withdrawn_sources'] for sid in supplied):
            raise ValueError('Available dependency must refer to a visible supplied source')
    if annotation.get('conditions_status')=='confirmed':
        payload=structured_input(entry,validate=False)
        validate_input(payload)
        states=annotation.get('condition_states',{})
        conditions={c['id']:c for c in annotation['conditions']}
        if set(states)!=set(conditions) or any(v not in {'T','F','U'} for v in states.values()):
            raise ValueError('Every condition needs an explicit semantic state, including U')
        for span in spans:
            if span.get('condition_id') not in conditions: raise ValueError('Evidence role must name its condition')
            if span['scope_id']!=conditions[span['condition_id']].get('scope_id'):
                raise ValueError('Evidence scope must match the annotated target condition')
            if span['role']=='exception':
                blocked=span.get('blocks_condition_id')
                if blocked not in conditions or conditions[blocked].get('scope_id')!=span['scope_id']:
                    raise ValueError('Exception must identify a condition in its own scope')
        observed={o['condition'] for o in payload['observations']}
        if observed!={cid for cid,state in states.items() if state!='U'}:
            raise ValueError('Each determinate condition needs explicit evidence groups; U has none')
        if len(observed)!=len(payload['observations']): raise ValueError('Use alternative evidence groups, not duplicated condition observations')
        by_id={s['id']:s for s in spans}
        for claim in payload['observations']:
            if any(by_id[eid]['condition_id']!=claim['condition'] for group in claim['evidence_groups'] for eid in group):
                raise ValueError('Condition observations cannot borrow another condition\'s annotated evidence')
    elif annotation.get('conditions_status')!='pending':
        raise ValueError('Conditions must be pending or separately confirmed')
    basis=annotation.get('severity_basis')
    grade=entry['targets'].get('severity') or entry['targets'].get('excerpt_severity')
    if grade=='critical' and basis is None:
        raise ValueError('New critical-grade supervision needs explicit immediate-harm evidence')
    if basis is not None:
        severity=grade
        if severity is None: raise ValueError('Severity basis requires a supplied severity label')
        validate_severity_basis(severity,basis)
        if basis.get('trigger_state') not in {'T','F','U'} or type(basis.get('performance_material_needed')) is not bool:
            raise ValueError('Record trigger state and the need for performance materials independently of the grade')
        refs=basis.get('evidence_span_ids',[])
        if not refs or not set(refs)<=ids: raise ValueError('Grade basis needs original-source evidence spans')
        if severity=='critical' and not set(basis.get('immediacy_span_ids',[]))<=ids:
            raise ValueError('Immediate harm evidence references unknown spans')
        if severity=='critical' and not basis.get('immediacy_span_ids'):
            raise ValueError('Critical grade basis must locate immediacy evidence, never derive it from the grade')
        if severity=='critical':
            from .research_review_protocol import critical_basis_check
            if critical_basis_check(request,basis)['missing']:
                raise ValueError('New critical supervision needs source-bound review scope, time, trigger, immediacy and blockers')
    return copy.deepcopy(entry)


def structured_input(entry,*,validate=True,observations=None):
    if validate: validate_annotation(entry)
    ann=entry['annotation']; request=validate_request(entry['request'])
    if ann.get('conditions_status')!='confirmed': raise ValueError('Confirmed rule/conditions required')
    claims=[]
    for cid,groups in ann.get('condition_evidence_groups',{}).items():
        state=ann['condition_states'].get(cid)
        if state not in {'T','F'}: raise ValueError('Unlabeled/unknown conditions cannot create observations')
        claims.append({'condition':cid,'value':state=='T','evidence_groups':groups,'origin':'provided'})
    payload={'proposition':request['question'],'proposition_type':request['proposition_type'],
        'review_perspective':request['review_perspective'],'task_mode':'structured','rule_status':'resolved',
        'structure_origin':'given_rule','rule':ann['rule'],'constraints':ann.get('constraints',[]),
        'conditions':copy.deepcopy(ann['conditions']),
        'sources':[{'id':s['id'],'available':s.get('available',True),'sha256':sha(s['text'])} for s in request['sources']],
        'evidence':[{'id':s['id'],'text':s['quote'],'start':s['start'],'end':s['end'],'scope_id':s['scope_id'],
                     'scope_kind':'exception' if s['role']=='exception' else 'clause','supports':[[s['source_id']]]} for s in ann['spans']],
        'withdrawn_sources':request['withdrawn_sources'],'observations':claims if observations is None else observations}
    return validate_input(payload)


def oracle_retrieval(entry,automatic):
    validate_annotation(entry)
    if entry['annotation']['evidence_complete'] is not True: raise ValueError('Oracle diagnostic needs explicitly reviewed complete key evidence')
    # Merge overlapping intervals so repeated/overlapping gold spans add no weight.
    request=validate_request(entry['request']); sources={s['id']:s for s in request['sources']}
    intervals={}
    for span in entry['annotation']['spans']:
        if span['role']=='irrelevant': continue
        intervals.setdefault(span['source_id'],[]).append((span['start'],span['end']))
    evidence=[]
    for sid,pieces in sorted(intervals.items()):
        merged=[]
        for start,end in sorted(pieces):
            if merged and start<=merged[-1][1]: merged[-1][1]=max(end,merged[-1][1])
            else: merged.append([start,end])
        for start,end in merged:
            text=sources[sid]['text'][start:end]
            evidence.append({'source_id':sid,'start':start,'end':end,'text':text,'text_sha256':sha(text),
                'source_text_sha256':sha(sources[sid]['text']),'unit_id':f'{sid}:{start}:{end}'})
    return {**copy.deepcopy(automatic),'evidence':evidence,'anchor_ids':[e['unit_id'] for e in evidence],
        'context_budget_exceeded':False,'status':'human_evidence_diagnostic_only',
        'evidence_selection':'confirmed_human_key_spans_not_automatic_retrieval',
        'original_document_completeness_verified':False,'diagnostic_only':True}


def retrieval_recall(entry,retrieval):
    validate_annotation(entry)
    result={}
    for role in ('support','refute','exception','definition'):
        spans=list({(s['source_id'],s['start'],s['end']):s for s in entry['annotation']['spans'] if s['role']==role}.values())
        hits=[]
        for span in spans:
            intervals=[]
            for e in retrieval['evidence']:
                for occurrence in [{'source_id':e['source_id'],'start':e['start'],'end':e['end']}]+e.get('duplicate_occurrences',[]):
                    if occurrence['source_id']==span['source_id']:
                        intervals.append((occurrence['start'],occurrence['end']))
            end=span['start']
            for a,b in sorted(intervals):
                if a<=end: end=max(end,b)
            hits.append(end>=span['end'])
        result[role]={'recalled':sum(hits),'total':len(hits),'recall':sum(hits)/len(hits) if hits else None}
    return result


def composition_diagnostic(entry,extractor=None,learned_composer=None):
    gold=structured_input(entry)
    exact=solve(gold)
    result={'gold_conditions_exact':exact,
        'gold_conditions_match_target':exact['proposition_label']==entry['targets'].get('qa'),
        'scope':'diagnostic_given_human_rule_conditions_and_spans_not_end_to_end',
        'risk_established_status':'unlabeled','severity':None,'decision_authority':False}
    predicted=None
    if extractor is not None:
        extracted=extractor.extract(model_input(gold))
        predicted=validate_input({**model_input(gold),'observations':extracted['input']['observations']})
        logic=solve(predicted)
        result['predicted_conditions_exact']={**select_decision(predicted,logic,uncertain_conditions=extracted.get('uncertain_conditions',[])),
                                              'conditional_logic':logic}
    else: result['predicted_conditions_exact']={'status':'not_run_no_condition_model'}
    for name,payload in [('gold_conditions_learned',gold),('predicted_conditions_learned',predicted)]:
        if learned_composer is None or payload is None:
            result[name]={'status':'not_run_no_learned_composer_or_conditions'}
        else:
            value=learned_composer(copy.deepcopy(payload))
            if value not in {'T','F','U'}: raise ValueError('Composer must return a semantic proposition candidate')
            result[name]={'proposition_candidate':value}
    return result


def validate_interventions(entries):
    by_id={r['sample_id']:r for r in entries}
    if len(by_id)!=len(entries): raise ValueError('Duplicate sample IDs')
    for row in entries:
        variant=row.get('intervention')
        if variant is None: continue
        require_confirmation(variant)
        parent=by_id.get(variant.get('parent_sample_id'))
        if parent is None or parent is row: raise ValueError('Variant requires its parent record')
        if parent.get('intervention') is not None: raise ValueError('Variants must point directly to an original root')
        for item in (parent,row):
            validate_request(item['request'])
            if item.get('request_sha256')!=digest(item['request']): raise ValueError('Intervention input fingerprint changed')
        if any(row[k]!=parent[k] for k in ('group_id','fold_id','root_id')):
            raise ValueError('All counterfactual variants must remain in the parent source group/fold')
        tasks=variant.get('task_relations',{})
        if not tasks or not set(tasks)<=set(TARGETS): raise ValueError('Specify reviewed per-task invariant/change relations')
        for task,relation in tasks.items():
            a,b=parent['targets'].get(task),row['targets'].get(task)
            if a not in MODEL_CLASSES[task] or b not in MODEL_CLASSES[task] or relation not in {'invariant','changed'}:
                raise ValueError('Both sides require observed reviewed labels')
            if (a==b)!=(relation=='invariant'): raise ValueError('Constraint disagrees with reviewed labels')
            if relation=='changed' and row['request_sha256']==parent['request_sha256']:
                raise ValueError('Identical observable input cannot receive contradictory intervention labels')
        if not variant.get('changed_factor'): raise ValueError('Record the single changed factor')
    return True


def intervention_loss(parent_logits,variant_logits,parent,variant,task):
    validate_interventions([parent,variant])
    relationship=variant['intervention']['task_relations'].get(task)
    if relationship is None: raise ValueError('Unreviewed task cannot receive intervention loss')
    classes=MODEL_CLASSES[task]
    if parent_logits.shape!=(len(classes),) or variant_logits.shape!=(len(classes),):
        raise ValueError('Logits must follow the documented semantic label order')
    if not torch.isfinite(parent_logits).all() or not torch.isfinite(variant_logits).all(): raise ValueError('Nonfinite logits')
    if relationship=='invariant':
        a=torch.log_softmax(parent_logits,dim=-1); b=torch.log_softmax(variant_logits,dim=-1)
        return .5*((a.exp()*(a-b)).sum()+(b.exp()*(b-a)).sum())
    # Changed examples receive their own confirmed labels, never consistency KL.
    return .5*sum(torch.nn.functional.cross_entropy(logits[None,:],torch.tensor([classes.index(row['targets'][task])],device=logits.device))
                  for logits,row in [(parent_logits,parent),(variant_logits,variant)])


def relation_training_pairs(entry):
    """Explicit labels only. A refuting clause remains retrieval-relevant."""
    validate_annotation(entry)
    ann=entry['annotation']
    if ann.get('conditions_status')!='confirmed': return []
    conditions={c['id']:c for c in ann['conditions']}
    return [{'query':conditions[s['condition_id']]['text'],'text':s['quote'],
        'relation':s['role'],'retrieval_relevant':s['role']!='irrelevant',
        'sample_id':entry['sample_id'],'group_id':entry['group_id'],'fold_id':entry['fold_id'],
        'source_id':s['source_id'],'start':s['start'],'end':s['end']} for s in ann['spans']]


def export_condition_records(entries,held_out_fold):
    """Bridge confirmed annotations to the existing condition extractor protocol.

    Missing relation labels stay masked. In particular, a rhetorical exception
    role does not automatically mean a refutation of its condition.
    """
    validate_interventions(entries)
    groups={}; roots={}; hashes={}; result=[]
    if type(held_out_fold) is not int or held_out_fold not in {e['fold_id'] for e in entries}:
        raise ValueError('Choose an existing source-held-out fold')
    for entry in entries:
        validate_annotation(entry)
        for value,owners in [(entry['group_id'],groups),(entry['root_id'],roots)]:
            owners.setdefault(value,set()).add(entry['fold_id'])
        for source in entry['request']['sources']:
            if source['text'].strip(): hashes.setdefault(sha(source['text']),set()).add(entry['fold_id'])
        payload=structured_input(entry); ann=entry['annotation']; spans=[]
        for span in ann['spans']:
            relation=span.get('condition_relation')
            if relation is None and span['role'] in {'support','refute','irrelevant'}: relation=span['role']
            if relation is None: continue
            spans.append({'condition':span['condition_id'],'evidence_id':span['id'],
                'start':0,'end':len(span['quote']),'quote':span['quote'],'scope_id':span['scope_id'],'relation':relation})
        grade=entry['targets'].get('severity')
        record={'input':model_input(payload),'targets':{'label':entry['targets'].get('qa'),
            'conditions':copy.deepcopy(ann['condition_states']),'spans':spans,'evidence':{},
            'severity':MODEL_CLASSES['severity'].index(grade) if grade is not None else None,
            'risk_established_status':entry['targets'].get('risk_status') or 'unlabeled',
            'severity_basis':copy.deepcopy(ann.get('severity_basis'))},
            'meta':{'sample_id':entry['sample_id'],'root_id':entry['root_id'],'group_id':entry['group_id'],
                'split':'test' if entry['fold_id']==held_out_fold else 'train','fold_id':entry['fold_id'],
                'label_status':'human_confirmed','reviewer_ids':[ann['reviewer_id']],
                'confirmed_at':ann['confirmed_at'],'annotation_sha256':digest(ann),
                'protocol_version':'contract-evidence-supervision-v1','human_gold':False,
                'identity_independently_verified':False,'independent_double_review':False,
                'evaluation_role':'known_development','independent_test':False}}
        result.append(validate_record(record))
    if any(len(v)!=1 for v in list(groups.values())+list(roots.values())+list(hashes.values())):
        raise ValueError('Condition-training source groups, roots or identical sources cross folds')
    if not any(r['meta']['split']=='train' for r in result): raise ValueError('No training source groups remain')
    return result
