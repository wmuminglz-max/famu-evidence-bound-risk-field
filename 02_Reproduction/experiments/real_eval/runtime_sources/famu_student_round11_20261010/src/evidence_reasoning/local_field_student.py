"""CPU-only field/evidence student. Runtime uses the standard library, not an LLM.

This is a bounded, opt-in research candidate. Final decisions have no direct
text-classification bypass: only grounded field states enter the existing solver.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import zlib

from .risk_field_v5 import certified_decision, verify_anchor
from .schema import expression_variables
from .text import numeric_facts

VERSION = 'local-field-student-20261009-v2'
FAMILY = 'famu_v8_simulated_contract_fields'
DIM = 8192
LABELS = ('irrelevant', 'support', 'refute')
VISIBLE = {'visible', 'redacted', 'not_provided', 'ocr_failed', 'retrieval_miss'}
DEFAULT_BUNDLE = Path(__file__).resolve().parents[2] / 'config/local_field_student'


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def visible_input(row):
    """Explicit research-data adapter; never copy targets, gold bindings or metadata."""
    return {'task_family': FAMILY, **{key: copy.deepcopy(row[key])
            for key in ('scope', 'fields', 'rule', 'evidence')}}


def validate(payload):
    allowed = {'task_family', 'scope', 'fields', 'rule', 'evidence'}
    if not isinstance(payload, dict) or set(payload) != allowed:
        raise ValueError('Only task_family/scope/fields/rule/evidence are inference inputs')
    if payload['task_family'] != FAMILY:
        raise ValueError('Unsupported task family; this candidate is not a general legal reviewer')
    fields, scope, evidence = payload['fields'], payload['scope'], payload['evidence']
    if not isinstance(fields, dict) or not 1 <= len(fields) <= 4:
        raise ValueError('One to four declared fields are required')
    if any(not isinstance(k, str) or not 1 <= len(k) <= 32 or
           not isinstance(v, str) or not 1 <= len(v) <= 400 for k, v in fields.items()):
        raise ValueError('Invalid field identifier or proposition')
    if not isinstance(scope, dict) or len(scope) > 8 or any(
            not isinstance(k, str) or len(k) > 40 or not isinstance(v, str) or len(v) > 600
            for k, v in scope.items()):
        raise ValueError('Invalid declared scope')
    if not isinstance(evidence, list) or len(evidence) > 128:
        raise ValueError('At most 128 evidence paragraphs; reject oversize inputs, never truncate')
    ids = set()
    for item in evidence:
        if not isinstance(item, dict) or not {'id', 'text'} <= item.keys() or set(item) - {'id', 'text', 'visibility'}:
            raise ValueError('Evidence accepts only id/text/visibility')
        if not isinstance(item['id'], str) or not 1 <= len(item['id']) <= 64 or item['id'] in ids:
            raise ValueError('Duplicate or invalid evidence identifier')
        ids.add(item['id'])
        if not isinstance(item['text'], str) or len(item['text']) > 2048:
            raise ValueError('Evidence paragraph exceeds 2048 characters')
        if item.get('visibility', 'visible') not in VISIBLE:
            raise ValueError('Unknown visibility; verified blank must be stated in source text')
    if len(json.dumps(payload, ensure_ascii=False)) > 150000:
        raise ValueError('Request exceeds local inference budget')
    # Bound total syntax as well as depth before the shared interpreter sees it.
    pending, nodes = [(payload['rule'], 0)], 0
    while pending:
        expr, depth = pending.pop(); nodes += 1
        if nodes > 64 or depth > 8:
            raise ValueError('Rule exceeds local execution budget')
        if isinstance(expr, dict):
            for args in expr.values():
                pending.extend((a, depth + 1) for a in (args if isinstance(args, list) else [args]))
    if expression_variables(payload['rule']) - fields.keys():
        raise ValueError('Rule references an undeclared field')
    return copy.deepcopy(payload)


def normalize(text, scope):
    for key in ('actor', 'event'):
        if scope.get(key):
            text = text.replace(scope[key], f'<{key}>')
    text = re.sub(r'\d{4}[年/-]\d{1,2}[月/-]\d{1,2}日?', '<date>', text)
    text = re.sub(r'\d+', '#', text)
    return re.sub(r'\s+', '', text.lower())


def features(query, text, scope):
    """Same bounded sparse features in offline fitting and stdlib serving.

    Scope matches and date comparisons are observable features, not hardcoded
    risk verdicts. Exact dates are compared before anonymizing surface numbers.
    """
    q, e = normalize(query, scope), normalize(text, scope)
    signature = str(zlib.crc32(q.encode('utf-8')))
    values = Counter()
    def add(token, value=1.):
        values[zlib.crc32(token.encode('utf-8')) % DIM] += value
    for prefix, value in (('q', q), ('e', e)):
        for n in (2, 3):
            for token in set(value[i:i+n] for i in range(max(0, len(value)-n+1))):
                add(prefix + ':' + token)
                if prefix == 'e':
                    add('joint:' + signature + ':' + token)
                    if token in q:
                        add('overlap:' + token)
    for key in ('actor', 'event', 'time'):
        value = scope.get(key)
        if value:
            add(f'scope:{key}:{value in text}', 3.)
    dates = numeric_facts(text)
    deadline = numeric_facts(scope.get('assessment_deadline', ''))['dates']
    if deadline and deadline[0]['value']:
        for date in dates['dates']:
            if date['value']:
                relation = 'after' if date['value'] > deadline[0]['value'] else 'before' if date['value'] < deadline[0]['value'] else 'same'
                add('date:' + signature + ':' + relation, 3.)
    if dates['errors']:
        add('invalid_date', 3.)
    for amount in dates['amounts']:
        add('amount:' + signature + ':' + ('zero' if float(amount['value_yuan']) == 0 else 'positive'))
    norm = math.sqrt(sum(v*v for v in values.values())) or 1.
    return {k: v/norm for k, v in values.items()}


class LocalFieldStudent:
    def __init__(self, model):
        if model.get('version') != VERSION or model.get('dimension') != DIM or model.get('labels') != list(LABELS):
            raise ValueError('Unsupported student model format')
        coefficients = model.get('coefficients', [])
        intercept = model.get('intercept', [])
        if len(coefficients) != 3 or any(len(row) != DIM for row in coefficients) or len(intercept) != 3:
            raise ValueError('Invalid student coefficient shapes')
        if not all(isinstance(v, (int, float)) and math.isfinite(v)
                   for row in coefficients + [intercept] for v in row):
            raise ValueError('Nonfinite student parameters')
        if not .5 <= model.get('threshold', 0) <= .99:
            raise ValueError('Invalid acceptance threshold')
        self.model = model

    @classmethod
    def load(cls, bundle=None):
        folder = Path(bundle or DEFAULT_BUNDLE)
        manifest = json.loads((folder / 'manifest.json').read_text('utf-8'))
        path = folder / 'student.json'
        if path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError('Student artifact exceeds 4 MiB deployment budget')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest['student_sha256']:
            raise ValueError('Student artifact hash mismatch')
        return cls(json.loads(data))

    def predict_pair(self, query, text, scope):
        x = features(query, text, scope)
        scores = [b + sum(row[i]*v for i, v in x.items())
                  for row, b in zip(self.model['coefficients'], self.model['intercept'])]
        pivot = max(scores)
        exp = [math.exp(v-pivot) for v in scores]
        total = sum(exp)
        return [v/total for v in exp]

    def review(self, payload):
        x = validate(payload)
        threshold = self.model['threshold']
        bindings, trace, ambiguous = {}, [], set()
        texts = {e['id']: e['text'] for e in x['evidence']}
        for field, query in x['fields'].items():
            bindings[field] = {'s': [], 'r': []}
            for e in x['evidence']:
                if e.get('visibility', 'visible') != 'visible' or not e['text']:
                    continue
                probabilities = self.predict_pair(query, e['text'], x['scope'])
                label = max(range(3), key=probabilities.__getitem__)
                accepted = label != 0 and probabilities[label] >= threshold
                # An uncertain "irrelevant" prediction cannot prove that the
                # paragraph has no bearing on this field either.
                if probabilities[label] < threshold:
                    ambiguous.add(field)
                if accepted:
                    bindings[field]['s' if label == 1 else 'r'].append(e['id'])
                    anchor = {'evidence_id': e['id'], 'start': 0, 'end': len(e['text']), 'text': e['text']}
                    assert verify_anchor(texts, anchor)
                    trace.append({'field': field, 'relation': LABELS[label], 'anchor': anchor,
                                  'model_score': probabilities[label], 'score_is_calibrated_probability': False})
        states = {f: 'conflict' if b['s'] and b['r'] else 'supported' if b['s'] else
                  'refuted' if b['r'] else 'unknown' for f, b in bindings.items()}
        # Relax ambiguous fields; only ambiguity able to change the answer blocks it.
        reliable = {f: ('unknown' if f in ambiguous else s) for f, s in states.items()}
        anchors = {f: [t['anchor'] for t in trace if t['field'] == f] for f in states}
        decision = certified_decision(x['rule'], list(reliable), list(reliable.values()), anchors)
        blocked = bool(ambiguous) and decision['state'] in {'unknown', 'conflict'}
        return {'version': VERSION, 'mode': 'shadow', 'decision_authority': False,
                'task_family': FAMILY, 'input_sha256': digest(x),
                'fields': states, 'effective_fields': reliable, 'bindings': bindings,
                'evidence_trace': trace, 'conditional_decision': decision,
                'prediction': None if blocked else decision['state'],
                'workflow_action': 'extraction_review' if blocked else decision['decision_action'],
                'ambiguous_fields': sorted(ambiguous),
                'unknown_scope': 'no accepted support/refutation within supplied evidence, not proof of absent original material',
                'certificate_scope': 'declared rule and learned fields only; not a legal truth guarantee',
                'deployment': {'teacher_calls': 0, 'llm_loaded': False, 'max_evidence': 128}}


def reviewed_teacher_pair(record, requests, heldout_groups):
    """Admission gate for offline teacher records; scores never become soft targets.

    A human review record is required. Verbatim quote matching alone is not
    semantic verification. Missing pair annotations remain masked, not negative.
    """
    if not isinstance(record, dict) or not isinstance(record.get('input_sha256'), str):
        raise ValueError('Teacher record must contain a string input hash')
    request = requests.get(record.get('input_sha256'))
    if request is None or digest(request['input']) != record.get('input_sha256'):
        raise ValueError('Teacher input hash mismatch')
    if request['split'] != 'train' or request['group_id'] in heldout_groups:
        raise ValueError('Held-out supervision must not enter student fitting')
    review = record.get('review', {})
    if not isinstance(review, dict) or review.get('status') != 'accepted' or not review.get('reviewer') or not review.get('record_ref'):
        raise ValueError('Teacher supervision awaits recorded semantic/scope review')
    if not record.get('teacher_id') or not record.get('teacher_version'):
        raise ValueError('Teacher provenance missing')
    x = validate(request['input'])
    if record.get('field') not in x['fields'] or record.get('relation') not in LABELS:
        raise ValueError('Missing or unsupported pair annotation; leave loss masked')
    texts = {e['id']: e['text'] for e in x['evidence'] if e.get('visibility', 'visible') == 'visible'}
    if not isinstance(record.get('anchor'), dict) or not verify_anchor(texts, record['anchor']):
        raise ValueError('Teacher quote/offset does not match visible original')
    # A relation for a short quote cannot be silently widened to its paragraph.
    anchor = record['anchor']
    if anchor['start'] != 0 or anchor['end'] != len(texts[anchor['evidence_id']]):
        raise ValueError('This paragraph student requires whole-paragraph reviewed relation')
    gold = request.get('confirmed_pairs', {}).get(record['field'], {}).get(anchor['evidence_id'])
    if gold is not None and gold != record['relation']:
        raise ValueError('Teacher conflicts with confirmed gold; send for review, never overwrite')
    return {'query': x['fields'][record['field']], 'text': anchor['text'], 'scope': x['scope'],
            'label': record['relation'], 'group_id': request['group_id'],
            'provenance': 'reviewed_teacher_hard_label', 'teacher_id': record['teacher_id'],
            'teacher_version': record['teacher_version'], 'review_record': review['record_ref']}
