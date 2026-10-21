"""Verifiable challenges, bounded rewards and release gates for contract policies.

The finite label policy is a contextual bandit, not a generative self-play LLM.
Synthetic tasks verify literal contract content only, never legal validity.
"""
from __future__ import annotations
import copy
import math
import random
from collections import Counter
from .contract_retrieval import validate_request,sha
from .question_evidence_models import TASK_CLASSES

VERSION='2026.09.30-contract-bandit-ppo-v1'
REWARD_VERSION='human-label-asymmetric-cost-v1'


def action_rewards(task,actual):
    task='severity' if task=='excerpt_severity' else task
    if task not in TASK_CLASSES or actual not in TASK_CLASSES[task]:
        raise ValueError('Reward requires an explicit supported target; blank is not U')
    labels=TASK_CLASSES[task]
    rewards=[]
    for predicted in labels:
        if predicted==actual:value=1.0
        elif task=='qa':value=-2.0 if actual=='U' else -1.0
        elif task=='risk_status':
            value=-3.0 if actual=='unknown' and predicted=='established' else -1.0
            if actual=='not_established' and predicted=='established':value=-2.0
        else:
            a,p=labels.index(actual),labels.index(predicted)
            value=-1.0-.5*abs(a-p)
            if actual in {'high','critical'} and p<a:value-=2.0
            if predicted in {'high','critical'} and actual=='low':value-=2.0
    # Each reward depends on externally recorded target and action, never on
    # the policy's own confidence or a majority vote by its copies.
        rewards.append(value)
    return rewards


def invariant_variants(payload):
    x=validate_request(payload)
    prefixed=copy.deepcopy(x)
    prefixed['question']='请依据已提供的材料回答以下问题：'+x['question']
    validate_request(prefixed)
    return [{'kind':'original','input':x,'target_policy':'inherit_existing_target'},
            {'kind':'instruction_prefix','input':prefixed,'target_policy':'inherit_existing_target',
             'scope':'Question content and all supplied source text unchanged; derived development case, not a new independent label'}]


def workflow_challenges(payload):
    x=validate_request(payload)
    duplicate=copy.deepcopy(x)
    if duplicate['sources'] and len(duplicate['sources'])<30:
        used={s['id'] for s in duplicate['sources']};name='challenge-copy'
        while name in used:name+='-copy'
        duplicate['sources'].append({**duplicate['sources'][0],'id':name})
    withdrawn=copy.deepcopy(x);withdrawn['withdrawn_sources']=[s['id'] for s in x['sources']]
    return [{'kind':'duplicate_source','input':duplicate,'legal_targets':None,'expected_workflow_property':'same_model_scores'},
            {'kind':'withdraw_sources','input':withdrawn,'legal_targets':None,'expected_workflow_property':'no_evidence_no_grade'}]


def synthetic_literal_cases(seed=20260930,count=36):
    """Balanced generated worlds; gold comes from the stated finite grammar."""
    if count<3 or count>84 or count%3:raise ValueError('Synthetic count must be a multiple of three in 3..84')
    rng=random.Random(seed)
    rows=[]
    for i,day in enumerate(rng.sample(range(1,29),count//3)):
        other=day%28+1
        for label in ('T','F','U'):
            actual_day=None if label=='U' else day if label=='T' else other
            text='第一条 本协议约定的付款日为每月'+str(actual_day)+'日。' if actual_day else '第一条 本协议未约定付款日期。'
            question=f'所给协议是否约定每月{day}日为付款日？'
            payload={'question':question,'proposition_type':'contract_content','domain':'other','review_perspective':'合同相对方',
                'sources':[{'id':'contract','kind':'contract','text':text,'sha256':sha(text),'completeness':'full_supplied_text'}]}
            case={'sample_id':'SYNTH-'+sha(f'{seed}|{i}|{label}')[:18],'input':payload,'actual':label,
                  'proof':{'kind':'literal_payment_day_v1','asked_day':day,'stated_day':actual_day},
                  'origin':'programmatically_verified_synthetic','independent_legal_example':False,'risk_targets':None}
            # U here means a date value cannot be established in the generated
            # world. With an explicit "未约定" sentence the literal proposition
            # is false, so use non-date text instead to represent unprovided scope.
            if label=='U':
                source=case['input']['sources'][0]
                source['text']='第一条 双方采用书面方式联系。'
                source['sha256']=sha(source['text'])
                source['completeness']='excerpt'
            verify_synthetic_case(case)
            rows.append(case)
    rng.shuffle(rows)
    return rows


def verify_synthetic_case(case):
    x=validate_request(case['input']);proof=case['proof']
    if proof.get('kind')!='literal_payment_day_v1':raise ValueError('Unsupported synthetic proof grammar')
    day,actual=proof['asked_day'],proof['stated_day']
    if type(day)!=int or not 1<=day<=28 or (actual is not None and (type(actual)!=int or not 1<=actual<=28)):
        raise ValueError('Invalid synthetic date')
    expected_question=f'所给协议是否约定每月{day}日为付款日？'
    text=f'第一条 本协议约定的付款日为每月{actual}日。' if actual is not None else '第一条 双方采用书面方式联系。'
    label='U' if actual is None else 'T' if actual==day else 'F'
    source=x['sources'][0] if len(x['sources'])==1 else {}
    if x['question']!=expected_question or source.get('text')!=text or case['actual']!=label or case.get('risk_targets') is not None:
        raise ValueError('Synthetic text/target does not match its proof')
    if actual is None and source.get('completeness')!='excerpt':raise ValueError('Missing-date U requires declared partial scope')
    return label


def validate_teacher_feedback(feedback,payload,task):
    """Validate provenance and quotes; a valid quote is not proof of legal truth."""
    x=validate_request(payload)
    required={'task','input_sha256','preferred_label','evidence','reason','needs_human_review'}
    if not isinstance(feedback,dict) or set(feedback)!=required:raise ValueError('Unexpected teacher response schema')
    from src.lawyer_review_dataset import digest
    if feedback['input_sha256']!=digest(x) or feedback['task']!=task:raise ValueError('Teacher response is bound to another input/task')
    target_task='severity' if task=='excerpt_severity' else task
    if feedback['preferred_label'] is not None and feedback['preferred_label'] not in TASK_CLASSES[target_task]:raise ValueError('Invalid teacher label')
    if type(feedback['needs_human_review'])!=bool or not isinstance(feedback['reason'],str) or not 1<=len(feedback['reason'])<=2400:raise ValueError('Invalid teacher review fields')
    if not isinstance(feedback['evidence'],list) or len(feedback['evidence'])>8:raise ValueError('Invalid teacher evidence list')
    visible={s['id']:s['text'] for s in x['sources'] if s.get('available',True) and s['id'] not in x['withdrawn_sources']}
    for evidence in feedback['evidence']:
        if not isinstance(evidence,dict) or set(evidence)!={'source_id','start','end','quote'}:raise ValueError('Invalid quote schema')
        source=visible.get(evidence['source_id']);start,end=evidence['start'],evidence['end']
        if source is None or type(start)!=int or type(end)!=int or not 0<=start<end<=len(source) or source[start:end]!=evidence['quote']:
            raise ValueError('Teacher quote does not match a visible source')
    if feedback['preferred_label'] is not None and not feedback['evidence']:raise ValueError('A proposed label needs at least one exact source quote')
    return {**feedback,'validation_scope':'schema_and_quote_only_not_legal_truth','legal_gold_eligible':False}


def teacher_reward_bonus(feedbacks,task,human_target,input_sha256=None):
    """At most .1 shaping; never relabel data, override gold or reward disputes."""
    task_labels=TASK_CLASSES['severity' if task=='excerpt_severity' else task]
    result=[0.0]*len(task_labels)
    accepted=[f for f in feedbacks if f.get('task')==task and f.get('validation_scope')=='schema_and_quote_only_not_legal_truth'
        and not f['needs_human_review'] and (input_sha256 is None or f.get('input_sha256')==input_sha256)]
    if human_target not in task_labels or not accepted or len(accepted)!=len(feedbacks):return result
    # Two configured model families are a diversity heuristic, not a claim of
    # independent annotation. Same-family repeated votes have no extra weight.
    families={f.get('teacher_family') for f in accepted if f.get('teacher_family')}
    if len(families)<2 or any(f['preferred_label']!=human_target for f in accepted):return result
    result[task_labels.index(human_target)]=.1
    return result


def release_gate(comparisons,independent_evaluation=None):
    reasons=[task+':'+reason for task,result in comparisons.items() for reason in result['regressions']]
    for task,result in comparisons.items():
        if result['observed_classes_never_predicted']:reasons.append(task+':observed_classes_never_predicted')
    # Research loop cannot manufacture release provenance by toggling a flag.
    reasons.append('independent_release_evaluation_not_implemented_in_this_research_runner')
    return {'candidate_research_only':True,'promotion_allowed':False,'reasons':sorted(set(reasons)),
            'external_evaluation_attestation_supplied':independent_evaluation is not None}
