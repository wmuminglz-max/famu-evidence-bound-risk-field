"""Versioned process semantics. Migration proposals are never human confirmation."""
from __future__ import annotations

import copy
import json

from .expert_data import fingerprint
from .contract_retrieval import sha

CONDITION_TYPES = {
    'text_content': 'contract_content',
    'performance_fact': 'case_application',
    'normative_applicability': 'normative_sufficiency',
    'composite_final_proposition': 'case_application',
}
SCOPE_KEYS = {'subject', 'time', 'object', 'contract_version', 'source_ranges', 'original_question'}


def completeness(value, request_sha256):
    if not isinstance(value, dict) or value.get('state') not in {'complete', 'incomplete', 'unknown'}:
        raise ValueError('Explicit complete/incomplete/unknown completeness required')
    if value.get('request_sha256') != request_sha256 or not value.get('provenance'):
        raise ValueError('Completeness must be bound to its input and provenance')
    if value['state'] != 'unknown' and (value.get('status') != 'confirmed' or
            not all(value.get(k) for k in ('reviewer_id', 'confirmed_at', 'basis'))):
        raise ValueError('Material or citation completeness requires separate human confirmation')
    return value['state']


def explicitly_complete(process, field):
    value = process.get(field)
    return bool(value and completeness(value, process['request_sha256']) == 'complete')


def validate_scope(condition, request):
    if condition.get('condition_type') not in CONDITION_TYPES:
        raise ValueError('Explicit condition_type required')
    scope = condition.get('scope')
    if not isinstance(scope, dict) or set(scope) != SCOPE_KEYS:
        raise ValueError('Scope accepts only visible subject/time/object/version/source ranges/original question')
    for key in SCOPE_KEYS - {'source_ranges'}:
        if scope[key] is not None and (not isinstance(scope[key], str) or not scope[key].strip()):
            raise ValueError('Scope fields are nonempty strings or explicitly unknown')
    sources = {s['id']: s for s in request['sources'] if s.get('available', True)
               and s['id'] not in request.get('withdrawn_sources', [])}
    if not isinstance(scope['source_ranges'], list) or not scope['source_ranges']:
        raise ValueError('Scope must identify visible source ranges')
    for span in scope['source_ranges']:
        if not isinstance(span, dict) or set(span) != {'source_id', 'start', 'end'}:
            raise ValueError('Scope source range contains unsupported fields')
        s = sources.get(span['source_id']); a, b = span['start'], span['end']
        if s is None or type(a) is not int or type(b) is not int or not 0 <= a < b <= len(s['text']):
            raise ValueError('Scope range outside visible source')
    return scope


def condition_request(request, condition):
    """Legacy replay is explicit; v3 never inherits the parent proposition type."""
    result = copy.deepcopy(request)
    if 'condition_type' not in condition and 'scope' not in condition:
        query = condition['text'] + '；审查范围：' + condition['scope_id']
    else:
        scope = validate_scope(condition, request)
        # IDs, annotation status, target states, rationales and provenance stay out.
        names = {'subject': '主体', 'time': '时间', 'object': '对象',
                 'contract_version': '合同版本', 'original_question': '审查问题'}
        query = condition['text'] + '；' + '；'.join(
            names[k] + '：' + scope[k] for k in names if scope[k] is not None)
        result['proposition_type'] = CONDITION_TYPES[condition['condition_type']]
        # Keep original offsets while removing text outside the declared ranges.
        # Otherwise two different paragraph scopes would silently share input.
        for source in result['sources']:
            ranges = [s for s in scope['source_ranges'] if s['source_id'] == source['id']]
            if not ranges:
                source['available'] = False
                continue
            original = source['text']; visible = [ch if ch in '\r\n' else ' ' for ch in original]
            for span in ranges:visible[span['start']:span['end']] = original[span['start']:span['end']]
            source['text'] = ''.join(visible); source['sha256'] = sha(source['text'])
            if source['text'] != original:source['completeness'] = 'excerpt'
    result['question'] = query
    return result


def require_formal_semantics(row):
    process = row['process']
    if process.get('schema_version') != 3:
        raise ValueError('Formal process training requires v3 reviewed semantics; legacy labels are diagnostic only')
    for c in process['conditions']:
        scope = validate_scope(c, row['request'])
        review = c.get('semantic_review', {})
        if review.get('status') != 'confirmed' or not all(review.get(k) for k in
                ('reviewer_id', 'confirmed_at', 'required_evidence', 'state_boundaries')):
            raise ValueError('Condition semantics and T/F/U boundaries await human review')
        if set(review['state_boundaries']) != {'T', 'F', 'U'} or not all(review['state_boundaries'].values()):
            raise ValueError('All T/F/U boundaries must be explicit')
        if any(scope[k] is None or '原问题' in scope[k] for k in ('subject', 'time', 'object', 'contract_version')):
            raise ValueError('Formal scope needs actual subject, time, object and contract version')
        if c['condition_type'] == 'composite_final_proposition':
            raise ValueError('Composite legal conclusions need separately confirmed atomic versions')
        if c['condition_type'] == 'normative_applicability':
            rule = c.get('legal_rule_review', {})
            if rule.get('status') != 'confirmed' or not all(rule.get(k) for k in
                    ('reviewer_id', 'confirmed_at', 'rule_version', 'validity', 'applicability', 'source_evidence')):
                raise ValueError('Normative conditions require reviewed law/version/applicability')


def structured_definition_text(conditions, rule, rule_confirmed):
    """Identical observable definitions for direct and decomposed main-task arms."""
    definitions = [{'id': c['id'], 'text': c['text'], 'condition_type': c['condition_type'],
                    'scope': {k: c['scope'][k] for k in sorted(SCOPE_KEYS)}}
                   for c in conditions]
    return '\n已提供条件定义：' + json.dumps(definitions, ensure_ascii=False, sort_keys=True) + (
        '\n已确认组合规则：' + json.dumps(rule, ensure_ascii=False, sort_keys=True) if rule_confirmed else '')


def prepare_main_definitions(rows, prepared, tokenizer):
    from .famu_contract_model import tokenize_case
    audits = {}
    for row in rows:
        p = row.get('process')
        if not p:continue
        x = prepared[row['sample_id']]
        text = structured_definition_text(p['conditions'], p['rule'], p['rule_status'] == 'confirmed')
        x['features']['evidence'] += text
        x['pieces'], audit = tokenize_case(tokenizer, x['features']['question'], x['features']['evidence'])
        audits[row['sample_id']] = {'definition_sha256': fingerprint(text), 'tokenization': audit}
    return audits


def input_identity(item):
    tokens = fingerprint(item['pieces'])
    return {'token_sha256': tokens, 'token_context_sha256': fingerprint(
        {'pieces': item['pieces'], 'context': item['context']})}


def collision_audit(prepared):
    buckets = {name: {} for name in ('token_sha256', 'token_context_sha256')}
    for sid, items in prepared.items():
        for x in items:
            if x['kind'] != 'condition':
                continue
            for name, digest in input_identity(x).items():
                buckets[name].setdefault(digest, []).append(
                    {'sample_id': sid, 'condition_id': x['id'], 'state': x['labels']['condition']})
    return {name: {'unique_inputs': len(groups),
        'duplicates': [{'sha256': digest, 'items': items, 'conflicting_labels':
                       len({x['state'] for x in items if x['state'] is not None}) > 1}
                      for digest, items in groups.items() if len(items) > 1]}
            for name, groups in buckets.items()}
