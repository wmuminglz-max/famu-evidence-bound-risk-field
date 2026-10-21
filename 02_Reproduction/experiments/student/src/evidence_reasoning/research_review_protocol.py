"""Source-bound review protocols; model consensus never creates legal gold."""
from __future__ import annotations
import copy
import math
import random
from src.lawyer_review_dataset import digest
from src.risk_grading import validate_severity_basis
from .contract_retrieval import validate_request
from .famu_contract_model import MODEL_CLASSES
from .self_improvement import validate_teacher_feedback

VERSION='2026.09.30-evidence-review-v2'


def validate_quote(quote,request):
    x=validate_request(request)
    sources={s['id']:s['text'] for s in x['sources'] if s.get('available',True) and s['id'] not in x['withdrawn_sources']}
    text=sources.get(quote.get('source_id'));a,b=quote.get('start'),quote.get('end')
    if text is None or type(a) is not int or type(b) is not int or not 0<=a<b<=len(text) or text[a:b]!=quote.get('quote'):
        raise ValueError('Evidence must be an exact quotation of a visible source')


def critical_basis_check(request,basis):
    """Completeness check only. Human reasons remain claims, not verified legal truth."""
    missing=[]
    try:validate_severity_basis('critical',basis)
    except (ValueError,TypeError):missing.append('legacy_severity_basis_incomplete')
    if not isinstance(basis,dict):basis={}
    for key in ['review_object','reference_time','trigger','immediacy_basis','magnitude','exceptions_remedies_blockers']:
        if not isinstance(basis.get(key),str) or not basis[key].strip():missing.append(key)
    if basis.get('trigger_state') not in {'T','F','U'}:missing.append('trigger_state')
    if basis.get('trigger_state')=='U':missing.append('trigger_state_unknown_requires_review')
    if basis.get('trigger_state')=='F' and not basis.get('prospective_action'):
        missing.append('prospective_action_required_when_trigger_not_yet_met')
    evidence=basis.get('source_evidence',{})
    if not isinstance(evidence,dict):evidence={}
    for key in ['review_scope','trigger','immediacy','major_consequence','exceptions_remedies_blockers']:
        refs=evidence.get(key)
        if not isinstance(refs,list) or not refs:missing.append(key+'_source_evidence');continue
        try:
            for ref in refs:validate_quote(ref,request)
        except (ValueError,TypeError,AttributeError):missing.append(key+'_invalid_source_evidence')
    return {'status':'requires_review' if missing else 'fields_complete_not_legal_validation','missing':sorted(set(missing)),
            'deterministic_grade_authorized':False,'automatic_downgrade':False,'standard_version':VERSION}


def validate_opinion(value,request,task):
    x=validate_request(request);v=copy.deepcopy(value)
    if v.get('input_sha256')!=digest(x) or v.get('task')!=task:raise ValueError('Opinion input/task mismatch')
    status=v.get('supervision_status');label=v.get('task_label')
    if status not in {'answered','abstained','technical_failure'}:raise ValueError('Separate supervision status required')
    if status=='answered' and label not in MODEL_CLASSES[task]:raise ValueError('Answered opinion requires a valid task label')
    if status!='answered' and label is not None:raise ValueError('Abstention/failure must not manufacture U or another label')
    scope=v.get('scope')
    if not isinstance(scope,dict) or not scope.get('review_object') or not isinstance(scope.get('source_completeness'),dict):
        raise ValueError('Review object and supplied material scope required')
    expected={s['id']:s.get('completeness','unknown') for s in x['sources'] if s.get('available',True) and s['id'] not in x['withdrawn_sources']}
    if scope['source_completeness']!=expected:raise ValueError('Opinion scope differs from supplied sources')
    evidence=v.get('evidence')
    if not isinstance(evidence,list) or (status=='answered' and not evidence):raise ValueError('Answered opinion requires located evidence')
    for e in evidence:
        validate_quote(e,x)
        if e.get('role') not in {'support','refute','exception','definition','scope','unspecified_legacy'}:raise ValueError('Explicit evidence role required')
    if type(v.get('needs_human_review')) is not bool or not isinstance(v.get('missing_materials'),list):
        raise ValueError('Review flag and missing material list required')
    v.update(protocol_version=VERSION,legal_gold_eligible=False,reward_bonus=0)
    return v


def migrate_legacy_opinion(feedback,request,task):
    fields={'task','input_sha256','preferred_label','evidence','reason','needs_human_review'}
    f=validate_teacher_feedback({k:feedback[k] for k in fields},request,task);x=validate_request(request)
    return validate_opinion({'task':task,'input_sha256':digest(x),'task_label':f['preferred_label'],
        'supervision_status':'abstained' if f['preferred_label'] is None else 'answered',
        'evidence':[{**e,'role':'unspecified_legacy'} for e in f['evidence']],
        'scope':{'review_object':'current_supplied_text','source_completeness':{s['id']:s.get('completeness','unknown') for s in x['sources'] if s.get('available',True) and s['id'] not in x['withdrawn_sources']}},
        'missing_materials':[],'missing_materials_status':'not_collected_in_legacy_schema',
        'needs_human_review':f['needs_human_review'],'reason':f['reason'],
        'teacher_id':feedback.get('teacher_id'),'teacher_family':feedback.get('teacher_family'),
        'independent_blind_review_verified':False,'migration_is_structural_not_new_supervision':True},x,task)


def compare_opinions(opinions,human_label=None):
    if len(opinions)!=2 or not all(o.get('teacher_id') for o in opinions) or len({o['teacher_id'] for o in opinions})!=2:
        raise ValueError('Two distinct identified supervisors required')
    statuses=[o['supervision_status'] for o in opinions];valid=all(s=='answered' for s in statuses)
    agree=valid and opinions[0]['task_label']==opinions[1]['task_label']
    fingerprint=lambda o:digest(sorted([(e['source_id'],e['start'],e['end'],e['quote'],e['role']) for e in o['evidence']]))
    evidence_same=fingerprint(opinions[0])==fingerprint(opinions[1])
    scope_same=opinions[0]['scope']==opinions[1]['scope']
    reasons=[]
    if not valid:reasons.append('abstention_or_technical_failure')
    elif not agree:reasons.append('label_disagreement')
    if not evidence_same:reasons.append('evidence_difference')
    if not scope_same:reasons.append('scope_difference')
    if any(o.get('missing_materials') for o in opinions):reasons.append('missing_materials')
    if any(o.get('missing_materials_status')=='not_collected_in_legacy_schema' for o in opinions):reasons.append('legacy_evidence_completeness_unknown')
    if any(o['needs_human_review'] for o in opinions):reasons.append('supervisor_requests_review')
    if agree and human_label is not None and opinions[0]['task_label']!=human_label:reasons.append('disagrees_with_existing_human_target')
    return {'valid_label_pair':valid,'labels_agree':agree,'evidence_quotes_and_roles_equal':evidence_same,'scope_equal':scope_same,
        'route':'human_review' if reasons else 'agreement_quality_audit' if human_label is not None else 'weak_candidate_only',
        'review_reasons':reasons,'legal_gold_eligible':False,'reward_bonus':0}


def agreement_sample(records,seed=20260930,fraction=.25):
    if not 0<=fraction<=1:raise ValueError('Invalid audit fraction')
    eligible=sorted(r['job_id'] for r in records if r['comparison']['labels_agree'])
    if len(set(eligible))!=len(eligible):raise ValueError('Duplicate review job')
    return sorted(random.Random(seed).sample(eligible,math.ceil(len(eligible)*fraction)))


def audit_call_events(events,ledger):
    reserved={r['job_id'] for r in ledger['calls']};attempts={};local=[]
    for event in events:
        attempt=event.get('attempt_id') or event['job_id']
        if attempt in attempts or attempt not in reserved:
            local.append(event);continue
        attempts[attempt]=event
    failures=sum('feedback' not in r for r in attempts.values())
    return {'live_record_events':len(events),'reserved_API_calls':len(reserved),'live_attempts':len(attempts),
            'local_rejected_or_duplicate_events':len(local),'technical_or_format_failures':failures,
            'technical_or_format_failure_rate':failures/len(attempts) if attempts else None,
            'reserved_calls_without_result':len(reserved-set(attempts))}


def audit_opinion_snapshot(snapshot):
    jobs={j['job_id']:j for j in snapshot['jobs']}
    if len(jobs)!=len(snapshot['jobs']):raise ValueError('Duplicate teacher jobs')
    records=[];seen=set()
    for record in snapshot['records']:
        jid=record['job_id']
        if jid not in jobs or jid in seen:raise ValueError('Unknown or duplicate teacher record')
        seen.add(jid);job=jobs[jid]
        opinions=[validate_opinion(o,job['input'],job['task']) for o in record['opinions']]
        label=record.get('existing_human_label')
        if label is not None and label not in MODEL_CLASSES[job['task']]:raise ValueError('Invalid existing human target')
        records.append({'job_id':jid,'opinions':opinions,'comparison':compare_opinions(opinions,label)})
    selected=agreement_sample(records)
    return {'records':records,'random_agreement_audit_ids':selected,'legal_gold_eligible':False,'new_training_labels':0,
            'snapshot_sha256':digest(snapshot),'unanswered_jobs':sorted(set(jobs)-seen)}


def validate_source_partitions(records):
    owners={k:{} for k in ('original_contract_id','template_group_id','derivation_root_id','source_sha256')}
    for row in records:
        if row.get('purpose') not in {'development','calibration','independent_evaluation'}:raise ValueError('Freeze material purpose')
        partition=(row['purpose'],row.get('fold_id'))
        for k,values in owners.items():
            if not row.get(k):raise ValueError('Three-level source lineage and source hash required')
            values.setdefault(row[k],set()).add(partition)
        if row['purpose']!='development':
            if row.get('template_independence_verified') is not True:raise ValueError('Independent template verification required')
            reviewers=row.get('blind_reviewer_ids',[])
            if len(set(reviewers))<2 or any(not isinstance(r,str) or not r.strip() for r in reviewers):raise ValueError('Two blind human reviewers required')
            if row.get('disagreement') not in {True,False}:raise ValueError('Record adjudication state')
            if row['disagreement'] and (not row.get('adjudicator_id') or row['adjudicator_id'] in reviewers):raise ValueError('Independent third adjudicator required')
    if any(len(parts)>1 for values in owners.values() for parts in values.values()):raise ValueError('Original/template/derived/hash lineage crosses partitions')
    return {'records':len(records),'cross_partition_overlap':0,'provenance_verification':'recorded assertions, not independently authenticated'}


def zero_error_upper_bound(independent_units,alpha=.05):
    if type(independent_units) is not int or independent_units<=0 or not 0<alpha<1:raise ValueError('Positive independent metric denominator required')
    return 1-alpha**(1/independent_units)
