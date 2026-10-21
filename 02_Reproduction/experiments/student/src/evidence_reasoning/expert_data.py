"""Verified 34/51/196 intake and source-safe supervision for the expert experiment.

The archive is data, never an instruction source. Final grades do not authorize
inventing condition, risk-status, evidence-relation or real-world fact labels.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from .contract_retrieval import validate_request, sha
from .famu_contract_model import MODEL_CLASSES

TASKS = ('qa', 'risk_status', 'severity')


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text('utf-8').splitlines() if line.strip()]


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), 'utf-8')
    temporary.replace(path)


def dump_lines(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows), 'utf-8')


def fingerprint(value):
    return sha(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')))


def archive_rows(archive, snapshot):
    """Validate every listed byte before writing an immutable local snapshot."""
    snapshot = Path(snapshot).resolve()
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(set(names)) != len(names) or sum(i.file_size for i in z.infolist()) > 100_000_000:
            raise ValueError('Duplicate members or oversized dataset archive')
        for name in names:
            p = PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
                raise ValueError('Unsafe archive path')
        manifests = [n for n in names if n.endswith('/文件清单_SHA256.json')]
        if len(manifests) != 1:
            raise ValueError('Exactly one dataset manifest required')
        root = manifests[0].rsplit('/', 1)[0]
        manifest = json.loads(z.read(manifests[0]))
        checked = {}
        for item in manifest['files']:
            member = root + '/' + item['archive_path']
            if member in checked:
                raise ValueError('Duplicate manifest entry')
            data = z.read(member)
            if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
                raise ValueError('Dataset checksum mismatch: ' + member)
            checked[member] = data
        # Only manifest-listed files are imported; other material cannot inject labels.
        for member, data in checked.items():
            target = (snapshot / member[len(root) + 1:]).resolve()
            if not target.is_relative_to(snapshot):
                raise ValueError('Snapshot path escape')
            if target.exists() and target.read_bytes() != data:
                raise ValueError('Existing snapshot differs; use a new output directory')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        dump(snapshot / 'archive_verification.json', {'archive_sha256': hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
            'verified_files': len(checked), 'counts': manifest['counts'], 'all_manifest_bytes_match': True})
    return snapshot


def build_dataset(snapshot):
    snapshot = Path(snapshot)
    old = read_jsonl(snapshot / '01_之前的34条/records.jsonl')
    new = read_jsonl(snapshot / '02_本次的51条/records.jsonl')
    contracts = read_jsonl(snapshot / '02_本次的51条/contracts.jsonl')
    excerpts = json.loads((snapshot / '03_196条摘录/records_reviewed.json').read_text('utf-8'))
    if (len(old), len(new), len(excerpts)) != (34, 51, 196):
        raise ValueError('Unexpected primary dataset sizes')
    contract_by_hash = {c['source_sha256']: c for c in contracts}
    rows = []
    for batch, raw_rows in [('confirmed_34', old), ('finalized_51', new)]:
        for raw in raw_rows:
            if batch == 'confirmed_34':
                if not raw.get('formal_imported') or raw['annotation_status'] != 'user_confirmed_reconciliation':
                    raise ValueError('Existing question lacks its confirmation lineage')
                group = 'FAMILY-' + raw['contract_id']
                fulltext = raw['clause_text']
                targets = {'qa': raw['question_answer'], 'risk_status': raw['risk_established_status'], 'severity': raw['final_severity']}
                proposition = raw['proposition_type']
                label_scope = 'conditional_question_grade'
            else:
                if raw.get('authorize_training') is not True or raw['annotation_status'] != 'user_attested_lawyer_finalized':
                    raise ValueError('New review lacks its confirmation lineage')
                group = raw['source_family_id']
                if raw['split_guard_group_id'] != group:
                    raise ValueError('Source family disagrees with split guard')
                fulltext = contract_by_hash[raw['source_contract_sha256']]['views']['current']
                targets = {'qa': None, 'risk_status': None, 'severity': raw['final_severity']}
                proposition = 'normative_sufficiency'
                label_scope = 'review_item_grade_without_separate_risk_status'
                if raw['qa_loss_mask'] or raw['risk_loss_mask'] or raw['evidence_relation_loss_mask']:
                    raise ValueError('51-item batch must not synthesize missing tasks')
            request = validate_request({'question': raw['issue'], 'proposition_type': proposition,
                'domain': raw['domain'], 'review_perspective': raw['review_perspective'],
                'sources': [{'id': 'contract', 'text': fulltext, 'kind': 'contract', 'available': True,
                             'sha256': sha(fulltext), 'completeness': 'full_supplied_text'}]})
            rows.append({'sample_id': raw['sample_id'], 'root_id': raw['sample_id'], 'group_id': group,
                'contract_id': group, 'original_contract_id': raw['contract_id'], 'cluster_id': raw['risk_cluster_id'],
                'batch': batch, 'label_scope': label_scope, 'request': request, 'request_sha256': fingerprint(request),
                'targets': targets, 'source_contract_sha256': raw['source_contract_sha256'],
                'source_record_sha256': fingerprint(raw), 'process': None, 'intervention': None,
                'annotation_authority': raw['annotation_status']})
    groups = sorted({r['group_id'] for r in rows})
    for row in rows:
        row['fold_id'] = groups.index(row['group_id'])
    validate_rows(rows)
    return rows, {'primary_records': len(rows), 'families': groups, 'excerpt_records_preserved': len(excerpts),
        'excerpt_training_weight': 0, 'label_support': support(rows), 'process_confirmed_records': 0,
        'critical_grade_scope': '14 lawyer review-item grades; immediate-harm condition supervision remains unverified',
        'independent_test_sources': 0, 'independent_calibration_sources': 0,
        'fulltext_policy': 'same automatic retrieval on full supplied text; reviewer-selected links and rationales excluded'}


def validate_rows(rows):
    ids, guards = set(), defaultdict(set)
    for r in rows:
        if r['sample_id'] in ids:
            raise ValueError('Duplicate record')
        ids.add(r['sample_id'])
        validate_request(r['request'])
        if fingerprint(r['request']) != r['request_sha256']:
            raise ValueError('Observable input fingerprint changed')
        for key, value in [('family', r['group_id']), ('root', r['root_id']), ('source', r['source_contract_sha256'])]:
            guards[(key, value)].add(r['fold_id'])
        for source in r['request']['sources']:
            if source['text'].strip():
                guards[('text', sha(source['text']))].add(r['fold_id'])
        if set(r['targets']) != set(TASKS) or any(v is not None and v not in MODEL_CLASSES[k] for k, v in r['targets'].items()):
            raise ValueError('Invalid task labels')
    if any(len(folds) != 1 for folds in guards.values()):
        raise ValueError('A source, text, or derivative crosses evaluation folds')


def support(rows):
    return {task: {label: {'records': sum(r['targets'][task] == label for r in rows),
        'families': len({r['group_id'] for r in rows if r['targets'][task] == label})}
        for label in MODEL_CLASSES[task]} for task in TASKS}


def condition_coverage(rows, required_classes=('T', 'F', 'U')):
    """Count source families and training-fold coverage, never treating conditions as contracts."""
    result = {}
    for label in required_classes:
        selected = [(r, c) for r in rows for c in (r.get('process') or {}).get('conditions', []) if c.get('state') == label]
        result[label] = {'conditions': len(selected), 'families': sorted({r['group_id'] for r, _ in selected})}
    folds = {}
    for fold in sorted({r['fold_id'] for r in rows}):
        seen = {c.get('state') for r in rows if r['fold_id'] != fold
                for c in (r.get('process') or {}).get('conditions', [])}
        folds[str(fold)] = {'training_support': [c for c in required_classes if c in seen],
                            'missing_training_classes': [c for c in required_classes if c not in seen]}
    return {'per_class': result, 'outer_folds': folds,
            'all_training_folds_cover_targets': all(not f['missing_training_classes'] for f in folds.values()) and bool(folds),
            'each_class_at_least_four_families': all(len(v['families']) >= 4 for v in result.values())}


def rule_structure_report(rows):
    from .schema import expression_variables
    groups = {'single_condition': [], 'composed_conditions': [], 'constant_rule': [], 'unconfirmed_rule': []}
    for r in rows:
        p = r.get('process')
        if not p:continue
        if p.get('rule_status') != 'confirmed':kind = 'unconfirmed_rule'
        else:
            count = len(expression_variables(p['rule']))
            kind = 'constant_rule' if count == 0 else 'single_condition' if count == 1 else 'composed_conditions'
        groups[kind].append(r['sample_id'])
    return {k: {'records': len(v), 'sample_ids': v} for k,v in groups.items()}


def task_weights(rows):
    """One fixed family/cluster scheme; no class inverse frequency or resampling."""
    result = {}
    for task in TASKS:
        selected = [r for r in rows if r['targets'][task] is not None]
        counts = Counter((r['group_id'], r['cluster_id']) for r in selected)
        clusters = defaultdict(set)
        for r in selected:
            clusters[r['group_id']].add(r['cluster_id'])
        result[task] = {r['sample_id']: 1 / len(clusters) / len(clusters[r['group_id']]) /
            counts[r['group_id'], r['cluster_id']] for r in selected}
    return result


def annotation_worklist(rows, snapshot):
    """Populate exact candidate spans and conditional phrases, without gold promotion."""
    original = {}
    for folder in ('01_之前的34条', '02_本次的51条'):
        original.update({r['sample_id']: r for r in read_jsonl(Path(snapshot) / folder / 'records.jsonl')})
    result = []
    for row in rows:
        raw = original[row['sample_id']]
        text = row['request']['sources'][0]['text']
        quotes = [c.get('quote', c.get('text', '')) for c in raw.get('evidence_candidates', raw.get('evidence_blocks', []))]
        spans = []
        for quote in dict.fromkeys(quotes):
            if not quote:
                continue
            for found in re.finditer(re.escape(quote), text):
                spans.append({'source_id': 'contract', 'start': found.start(), 'end': found.end(), 'quote': quote,
                    'role': None, 'relation': None, 'condition_id': None, 'status': 'candidate_not_human_gold'})
        conditional = []
        for m in re.finditer(r'[^\n。；]*(?:若|如|除非|否则|但|前提|条件)[^\n。；]*[。；]?', text):
            if len(m[0]) > 8:
                conditional.append({'source_id': 'contract', 'start': m.start(), 'end': m.end(), 'quote': m[0],
                    'role': 'unconfirmed_conditional_phrase', 'state': None})
        result.append({'sample_id': row['sample_id'], 'group_id': row['group_id'], 'fold_id': row['fold_id'],
            'request_sha256': row['request_sha256'], 'question': row['request']['question'],
            'existing_final_targets': row['targets'], 'final_target_authority': row['annotation_authority'],
            'candidate_spans': spans, 'conditional_phrases': conditional,
            'review_rationale_reference': raw.get('review_rationale', raw.get('review_text_as_written', '')),
            'process': {'status': 'pending', 'reviewer_id': None, 'confirmed_at': None,
                'request_sha256': row['request_sha256'],
                'conditions': [], 'pairs': [], 'rule': None, 'rule_status': 'pending',
                'evidence_complete': None, 'missing_materials': []},
            'training_eligible_process_labels': False})
    return result
