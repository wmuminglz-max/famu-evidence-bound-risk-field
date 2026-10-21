"""Small, versioned review task IR. Audit metadata is never a model input."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

VERSION = 'famu-review-ir-1'
VERSION_V2 = 'famu-review-ir-2'
CLASSES = {
    'qa': ('T', 'F', 'U'),
    'condition': ('T', 'F', 'U'),
    'risk_status': ('established', 'not_established', 'unknown'),
    'conditional_question_grade': ('low', 'medium', 'high', 'critical'),
    'review_item_grade': ('low', 'medium', 'high', 'critical'),
    'relation': ('support', 'refute', 'irrelevant'),
}
TASK_TEXT = {
    'qa': '判断给定问题的命题状态',
    'condition': '判断给定条件在指定材料中的状态',
    'risk_status': '判断给定事项的风险成立状态',
    'conditional_question_grade': '判断给定问题中已指明风险的条件性严重度；与命题真假分开',
    'review_item_grade': '判断给定审查事项的严重度；此任务不确认现实风险已经发生',
    'relation': '判断给定证据组合与条件的关系；判断整个组合，不把关系分摊到每条证据',
}


def digest(value):
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text('utf-8-sig').splitlines() if s.strip()]


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), 'utf-8')


def write_jsonl(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(x, ensure_ascii=False, allow_nan=False) + '\n' for x in data), 'utf-8')


def validate_visible_input(task, value):
    if task not in CLASSES:
        raise ValueError('Unknown review task')
    required = {'question', 'original_question', 'domain', 'review_perspective', 'evidence'}
    if set(value) not in (required, required | {'proposition'}):
        raise ValueError('Only the visible input allowlist may reach the model')
    if 'proposition' in value:
        p = value['proposition']
        if set(p) != {'type', 'parent_question_type', 'actor', 'object', 'time', 'scope', 'interpretation'}:
            raise ValueError('Unexpected proposition field')
        types = {None, 'contract_content', 'normative_sufficiency', 'case_application'}
        if p['type'] not in types or p['parent_question_type'] not in types:
            raise ValueError('Unknown proposition type')
        if p['interpretation'] != 'question_literals_are_not_established_facts':
            raise ValueError('Scope literals must not assert facts')
        for name in ('actor', 'object', 'time', 'scope'):
            if p[name] is not None:
                for span in p[name]:
                    if set(span) != {'text', 'start', 'end'} or value['question'][span['start']:span['end']] != span['text']:
                        raise ValueError('Scope must be an exact question span')
    if not isinstance(value['question'], str) or not value['question'].strip():
        raise ValueError('Question required')
    if not isinstance(value['evidence'], list):
        raise ValueError('Evidence list required')
    for piece in value['evidence']:
        if set(piece) != {'id', 'source', 'kind', 'text'}:
            raise ValueError('Unexpected evidence field')
    return value


def validate_example(row):
    if row['schema_version'] not in {VERSION, VERSION_V2}:
        raise ValueError('Unknown review schema')
    if ('proposition' in row['input']) != (row['schema_version'] == VERSION_V2):
        raise ValueError('Representation version mismatch')
    validate_visible_input(row['task'], row['input'])
    if row['target']['status'] != 'confirmed' or not row['target'].get('provenance'):
        raise ValueError('Only separately confirmed fields may be supervised')
    if row['target']['value'] not in CLASSES[row['task']]:
        raise ValueError('Missing/unknown targets are not negative labels')
    if row['input_sha256'] != digest(row['input']):
        raise ValueError('Visible input changed')
    if not row['group_id'] or not row['parent_id']:
        raise ValueError('Family and parent guards required')
    return row
