"""Auditable process supervision, exact composition and information acquisition.

Condition schemas/rules are supplied and confirmed, not inferred legal rules.
Runtime condition states and evidence relations come from the trained model.
"""
from __future__ import annotations
import copy
import itertools
import random
from collections import Counter

import torch
from .expert_data import fingerprint, validate_rows, TASKS
from .expert_model import PROCESS_CLASSES
from .contract_retrieval import ContractRetriever, prepare_features, sha, validate_request
from .famu_contract_model import tokenize_case
from .question_evidence_models import context_vector
from .logic import evaluate
from .schema import expression_variables
from .expert_semantics import completeness, explicitly_complete, validate_scope, condition_request, input_identity

ABSENCE_REASONS = {'retrieval_not_found', 'material_missing', 'full_scope_checked_absent', 'not_applicable'}


def confirmed(value):
    if value.get('status') != 'confirmed' or not all(isinstance(value.get(k), str) and value[k].strip() for k in ('reviewer_id', 'confirmed_at')):
        raise ValueError('Process labels require their own human confirmation; final grades are insufficient')


def validate_expression(expr, ids, depth=0):
    if depth > 24:
        raise ValueError('Rule nesting exceeds budget')
    if type(expr) is bool:
        return
    if isinstance(expr, str) and expr in ids:
        return
    if not isinstance(expr, dict) or len(expr) != 1:
        raise ValueError('Only confirmed boolean condition rules are supported')
    op, value = next(iter(expr.items()))
    if op == 'not':
        validate_expression(value, ids, depth + 1)
    elif op in {'all', 'any'} and isinstance(value, list) and value:
        for child in value:
            validate_expression(child, ids, depth + 1)
    else:
        raise ValueError('Unsupported rule expression')


def validate_process(row):
    p = row.get('process')
    if not isinstance(p, dict):
        raise ValueError('No confirmed process annotation')
    confirmed(p)
    if p.get('request_sha256') != row['request_sha256'] or fingerprint(row['request']) != row['request_sha256']:
        raise ValueError('Process input changed after confirmation')
    conditions = p.get('conditions', [])
    ids = {c['id'] for c in conditions}
    if not conditions or len(ids) != len(conditions) or len(ids) > 12:
        raise ValueError('1..12 unique, explicit conditions required')
    for c in conditions:
        if not c.get('id') or not c.get('text') or not c.get('scope_id') or len(c['text']) > 250:
            raise ValueError('Each condition requires a concise statement and scope')
        if c.get('state') not in {None, 'T', 'F', 'U'}:
            raise ValueError('Invalid condition state')
        if c.get('absence_reason') not in ABSENCE_REASONS:
            raise ValueError('Retrieval miss, missing materials and reviewed absence must be separate')
    if p.get('schema_version') == 3:
        for field in ('material_scope_completeness', 'citation_set_sufficiency'):
            completeness(p.get(field), row['request_sha256'])
        for c in conditions:
            validate_scope(c, row['request'])
    elif type(p.get('evidence_complete')) is not bool:
        raise ValueError('Legacy evidence completeness field required for replay')
    sources = {s['id']: s for s in row['request']['sources']}
    pairs, pair_ids = p.get('pairs', []), set()
    for pair in pairs:
        if not pair.get('id') or pair['id'] in pair_ids or pair.get('condition_id') not in ids:
            raise ValueError('Unique evidence pair IDs and known conditions required')
        pair_ids.add(pair['id'])
        if not pair.get('evidence'):
            raise ValueError('Pairs need actual evidence, including jointly supporting groups')
        for span in pair['evidence']:
            source = sources.get(span.get('source_id'))
            if source is None or not source.get('available', True) or source['id'] in row['request'].get('withdrawn_sources', []):
                raise ValueError('Unavailable evidence cannot supervise the model')
            a, b = span.get('start'), span.get('end')
            if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(source['text']) or source['text'][a:b] != span.get('quote'):
                raise ValueError('Evidence quote does not match source offsets')
        for task, key in [('relation', 'relation'), ('scope', 'scope'), ('relevance', 'relevance')]:
            if pair.get(key) is not None and pair[key] not in PROCESS_CLASSES[task]:
                raise ValueError('Invalid explicit evidence target')
        roles = pair.get('roles')
        if roles is not None and (not isinstance(roles, dict) or not roles or set(roles)-set(PROCESS_CLASSES['roles']) or any(type(v) is not bool for v in roles.values())):
            raise ValueError('Roles need partial binary labels; unlabeled roles are masked')
        if roles and roles.get('exception'):
            blocked = pair.get('blocks_condition_id')
            if blocked not in ids:
                raise ValueError('Exception needs a local target condition')
            scope = next(c['scope_id'] for c in conditions if c['id'] == pair['condition_id'])
            if next(c['scope_id'] for c in conditions if c['id'] == blocked) != scope:
                raise ValueError('Exception cannot silently apply outside its condition scope')
    for c in conditions:
        evidence = c.get('evidence_pair_ids', [])
        if not set(evidence) <= pair_ids or any(next(a for a in pairs if a['id'] == k)['condition_id'] != c['id'] for k in evidence):
            raise ValueError('Condition borrows another condition evidence')
        if c['state'] in {'T', 'F'} and not evidence and c['absence_reason'] != 'full_scope_checked_absent':
            raise ValueError('Determinate condition needs evidence or explicitly reviewed complete absence')
        if c['absence_reason'] == 'full_scope_checked_absent' and not explicitly_complete(p, 'material_scope_completeness'):
            raise ValueError('Absence requires separately confirmed complete material scope')
    if p.get('rule_status') == 'confirmed':
        validate_expression(p['rule'], ids)
    elif p.get('rule_status') != 'pending':
        raise ValueError('Separate rule confirmation required')
    return p


def attach_process(rows, annotations):
    """Imports only separately confirmed process annotations, preserving final labels."""
    result = copy.deepcopy(rows)
    for row in result:
        if row.get('process') is not None:
            validate_process(row)
    by_id = {r['sample_id']: r for r in result}; seen = set()
    for entry in annotations:
        sid = entry['sample_id']
        if sid not in by_id or sid in seen:
            raise ValueError('Unknown/duplicate process record')
        seen.add(sid); row = by_id[sid]
        if entry['request_sha256'] != row['request_sha256'] or entry['group_id'] != row['group_id']:
            raise ValueError('Process annotation lineage mismatch')
        if entry.get('process', {}).get('status') != 'confirmed':
            continue
        row['process'] = copy.deepcopy(entry['process'])
        validate_process(row)
    validate_rows(result)
    return result


def prepare_process(row, tokenizer):
    """Gold state never enters text. Condition-state input uses AUTOMATIC evidence."""
    p = validate_process(row); items = []; retriever = ContractRetriever(top_k=5, max_context_chars=7000)
    for c in p['conditions']:
        request = condition_request(row['request'], c); query = request['question']
        retrieved = retriever.retrieve(request); features = prepare_features(request, retrieved)
        pieces, audit = tokenize_case(tokenizer, query, features['evidence'])
        item = {'id': c['id'], 'kind': 'condition', 'pieces': pieces, 'context': context_vector(features),
            'query': query, 'text': features['evidence'], 'labels': {'condition': c.get('state')}, 'audit': audit,
            'input_semantics': 'structured_v3' if 'condition_type' in c else 'legacy_replay',
            'condition_type': c.get('condition_type'), 'visible_proposition_type': request['proposition_type']}
        item.update(input_identity(item)); items.append(item)
        for pair in p['pairs']:
            if pair['condition_id'] != c['id']:
                continue
            text = '\n'.join(e['quote'] for e in pair['evidence'])
            pieces, audit = tokenize_case(tokenizer, query, text)
            items.append({'id': pair['id'], 'kind': 'pair', 'pieces': pieces, 'context': context_vector(features),
                'query': query, 'text': text, 'labels': {k: pair.get(k) for k in ('relation', 'roles', 'scope', 'relevance')}, 'audit': audit})
    return items


def execute_states(rule, states, *, rule_confirmed, uncertain=()):
    """Finite-world evaluation with unknowns; uncertainty is an abstention, not U."""
    if not rule_confirmed:
        return {'proposition_candidate': None, 'workflow_action': 'review_rule', 'decision_authority': False}
    if not states or len(states) > 12 or any(s not in {'T', 'F', 'U'} for s in states.values()):
        raise ValueError('Invalid condition states')
    validate_expression(rule, set(states))
    observed = {cid: state == 'T' for cid, state in states.items() if state != 'U'}
    unknown = [cid for cid, state in states.items() if state == 'U']; witnesses = {}
    for values in itertools.product((False, True), repeat=len(unknown)):
        world = observed | dict(zip(unknown, values))
        witnesses.setdefault(evaluate(rule, world), world)
    conditional = 'U' if len(witnesses) == 2 else 'T' if True in witnesses else 'F'
    abstain = bool(set(uncertain) & expression_variables(rule))
    return {'proposition_candidate': None if abstain else conditional, 'conditional_logic': conditional,
        'condition_states': states, 'certificate': {str(k): v for k, v in witnesses.items()},
        'workflow_action': 'review_extraction' if abstain else 'request_materials' if conditional == 'U' else 'review_candidate',
        'risk_grade': None, 'risk_status': None, 'decision_authority': False}


def hard_example_order(rows, reviewed_hard_ids, seed, fraction=.25):
    """Only already reviewed TRAINING-pool hard cases; explicit exposure accounting."""
    if not 0 <= fraction < 1:
        raise ValueError('Invalid hard-example proportion')
    by_id = {r['sample_id']: r for r in rows}
    if not set(reviewed_hard_ids) <= set(by_id):
        raise ValueError('Hard cases must belong to the current training fold')
    rng = random.Random(seed); regular = list(rows); rng.shuffle(regular)
    if not reviewed_hard_ids:
        return regular
    hard_count = round(len(regular) * fraction)
    additions = [by_id[rng.choice(sorted(reviewed_hard_ids))] for _ in range(hard_count)]
    mixed = regular[:len(regular)-hard_count] + additions; rng.shuffle(mixed)
    return mixed


def information_action(observable_state):
    """Simple baseline uses only current visible state, never changes a risk grade."""
    allowed = {'technical_status', 'missing_materials', 'unresolved_conditions', 'rule_confirmed', 'extraction_uncertain'}
    if set(observable_state) - allowed:
        raise ValueError('Action state contains outcomes or non-observable fields')
    if observable_state.get('technical_status', 'ok') != 'ok':
        action, detail = 'technical_retry', None
    elif not observable_state.get('rule_confirmed', False):
        action, detail = 'human_review', 'rule_unconfirmed'
    elif observable_state.get('missing_materials'):
        action, detail = 'request_attachment', observable_state['missing_materials'][0]
    elif observable_state.get('extraction_uncertain'):
        action, detail = 'human_review', 'condition_extraction_uncertain'
    elif observable_state.get('unresolved_conditions'):
        action, detail = 'retrieve_condition', observable_state['unresolved_conditions'][0]
    else:
        action, detail = 'review_candidate', None
    return {'action': action, 'detail': detail, 'risk_grade_override': None, 'policy': 'observable_missing_information_baseline'}


class NeuralConditionRuntime:
    def __init__(self, model, supported, confidence_threshold=.7, aggregation='mean'):
        if not 0 < confidence_threshold <= 1:
            raise ValueError('Invalid abstention threshold')
        self.model, self.supported, self.threshold = model, supported, confidence_threshold
        if aggregation not in {'mean', 'roles'} or aggregation == 'roles' and not supported.get('roles'):
            raise ValueError('Role aggregation requires a trained role head')
        self.aggregation = aggregation

    def predict(self, request, conditions, rule, rule_confirmed=False):
        """Only condition definitions/scope enter; no states, gold spans or relations."""
        request=validate_request(request)
        if request['technical_status']!='ok':
            return {'raw_model':[], 'system':{'proposition_candidate':None,'workflow_action':'technical_retry',
                    'risk_grade':None,'risk_status':None,'decision_authority':False},
                'technical_status':'failed','decision_authority':False}
        states, uncertain, predictions = {}, [], []
        self.model.eval()
        with torch.no_grad():
            for condition in conditions:
                if set(condition) not in ({'id', 'text', 'scope_id'}, {'id', 'text', 'scope_id', 'condition_type', 'scope'}):
                    raise ValueError('Runtime conditions must not include target states or evidence')
                x = condition_request(request, condition); query = x['question']
                result = ContractRetriever(top_k=5, max_context_chars=7000).retrieve(x)
                features = prepare_features(x, result)
                pieces, _ = tokenize_case(self.model.tokenizer, query, features['evidence'])
                logits = self.model.process_logits(pieces, context_vector(features), self.aggregation)['condition']
                support = self.supported.get('condition', [])
                if not support:
                    raise ValueError('Condition head has not been trained')
                mask = torch.tensor([c in support for c in PROCESS_CLASSES['condition']])
                probs = torch.softmax(logits.masked_fill(~mask, -torch.inf), -1)
                label = PROCESS_CLASSES['condition'][int(probs.argmax())]
                states[condition['id']] = label
                if float(probs.max()) < self.threshold or set(support) != set(PROCESS_CLASSES['condition']):
                    uncertain.append(condition['id'])
                pairs = []
                for evidence in result['evidence']:
                    tokens, _ = tokenize_case(self.model.tokenizer, query, evidence['text'])
                    pred = self.model.process_logits(tokens, context_vector(features))
                    values = {}
                    for task in ('relation', 'roles', 'scope', 'relevance'):
                        supported = self.supported.get(task, [])
                        if not supported:
                            values[task] = None; continue
                        if task == 'roles':
                            values[task] = {c: float(torch.sigmoid(pred[task][i])) for i, c in enumerate(PROCESS_CLASSES[task]) if c in supported}
                        else:
                            m = torch.tensor([c in supported for c in PROCESS_CLASSES[task]])
                            p = torch.softmax(pred[task].masked_fill(~m, -torch.inf), -1)
                            values[task] = dict(zip(PROCESS_CLASSES[task], p.tolist()))
                    pairs.append({'source_id': evidence['source_id'], 'start': evidence['start'], 'end': evidence['end'],
                                  'text': evidence['text'], 'predictions': values})
                predictions.append({'condition_id': condition['id'], 'state': label, 'score': float(probs.max()),
                    'probabilities': dict(zip(PROCESS_CLASSES['condition'], probs.tolist())),
                    'raw_logits': dict(zip(PROCESS_CLASSES['condition'], logits.tolist())),
                    'raw_probabilities': dict(zip(PROCESS_CLASSES['condition'], torch.softmax(logits, -1).tolist())),
                    'evidence': pairs})
        return {'raw_model': predictions, 'system': execute_states(rule, states, rule_confirmed=rule_confirmed, uncertain=uncertain),
                'probabilities_calibrated': False, 'rule_origin': 'caller_confirmed' if rule_confirmed else 'unconfirmed',
                'trained_condition_classes':self.supported.get('condition',[]),
                'unsupported_condition_classes':[c for c in PROCESS_CLASSES['condition'] if c not in self.supported.get('condition',[])],
                'scores_conditioned_on_seen_training_classes':True,
                'confidence_threshold':self.threshold,'threshold_validated_for_legal_safety':False,
                'requires_manual_review_for_unsupported_classes':set(self.supported.get('condition',[])) != set(PROCESS_CLASSES['condition']),
                'condition_definitions_supplied_by_caller':True,'decision_authority': False}
