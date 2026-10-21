"""Freeze a mechanism pilot; generated supervision is never legal gold."""
import hashlib
import itertools
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evidence_reasoning.risk_field_v5 import STATES, DECISIONS, bounds, refinement_pair
from src.evidence_reasoning.logic import evaluate

OUT = ROOT / 'training_data/famu_v5_20261004'
OLD = ROOT / 'training_data/famu_v4_20261003'
FIELDS = ['A', 'B', 'E', 'D']
RULES = [
    {'all': ['A', 'B', {'not': 'E'}]},
    {'any': [{'all': ['A', 'B', {'not': 'E'}]}, 'D']},
    {'all': ['A', {'any': ['B', 'D']}, {'not': 'E'}]},
    {'all': [{'any': ['A', 'D']}, 'B', {'not': 'E'}]},
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return [json.loads(x) for x in p.read_text('utf-8').splitlines()]


def write(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), 'utf-8')


def synthetic():
    rows, pairs = [], []
    rng = random.Random(20261004)
    order = list(range(96)); rng.shuffle(order)
    splits = {n: 'train' if i < 64 else 'dev' if i < 80 else 'test' for i, n in enumerate(order)}
    for n in range(96):
        family = f'SIM-{n:03d}'
        actor = f'甲方{n+101}号机构'
        other = f'乙方{n+701}号机构'
        subject = ['课程退款', '岗位调整', '材料交付', '服务停办', '验收通知', '费用结算'][n % 6]
        event = f'{subject}事项{n+1}'
        descriptions = [f'{actor}的{event}属于约定适用范围', f'{actor}在{event}中发生约定触发事实',
                        f'{actor}的{event}满足书面豁免条件', f'{actor}的{event}满足第二条独立触发路径']
        rule = RULES[n % len(RULES)]
        base = [rng.choice(STATES[:2]) for _ in FIELDS]
        candidates = []
        for size in range(1, 5):
            for subset in itertools.combinations(range(4), size):
                masked = ['unknown' if i in subset else s for i, s in enumerate(base)]
                if bounds(rule, FIELDS, masked)['state'] == 'unknown':
                    candidates.append(masked)
            if candidates:
                break
        hidden = rng.choice(candidates)
        fact = base.copy(); fact[1] = STATES[1-STATES.index(base[1])]
        exception = base.copy(); exception[2] = STATES[1-STATES.index(base[2])]
        conflict = base.copy(); conflict[1] = 'conflict'
        wrong_actor = base.copy(); wrong_actor[1] = 'unknown'
        variants = [('base', base), ('fact', fact), ('exception', exception), ('hidden', hidden),
                    ('paraphrase', base), ('irrelevant', base), ('conflict', conflict), ('actor_swap', wrong_actor)]
        for kind, states in variants:
            evidence, anchors = [], {}
            for i, (field, description, state) in enumerate(zip(FIELDS, descriptions, states)):
                texts = []
                if kind == 'actor_swap' and i == 1:
                    texts = [f'核查记录：{description.replace(actor, other)}，已确认。与{actor}相应的资料未提供。']
                elif state == 'unknown':
                    texts = [f'关于“{description}”的原始记录未提供；当前副本此处为脱敏占位，不能据此认定原件空白。']
                elif state == 'conflict':
                    texts = [f'记录一明确记载：{description}，属实。', f'记录二明确记载：{description}，不属实。两份记录的效力尚未裁决。']
                elif kind == 'paraphrase':
                    texts = [f'就“{description}”这一陈述，核查结论为' + ('与事实相符。' if state == 'supported' else '与事实不符。')]
                else:
                    texts = [f'核查记录：{description}，' + ('已确认。' if state == 'supported' else '经核实并未发生。')]
                anchors[field] = []
                for j, text in enumerate(texts):
                    eid = f'V{i+1}-{j+1}'
                    evidence.append({'id': eid, 'text': text, 'type': 'simulated_record'})
                    if state != 'unknown':
                        anchors[field].append({'evidence_id': eid, 'start': 0, 'end': len(text), 'text': text})
            if kind == 'irrelevant':
                evidence.append({'id': 'V9', 'text': f'{other}在另外一件事项中提出申请；本事项附件使用蓝色封面。', 'type': 'irrelevant_record'})
            rng.shuffle(evidence)
            rows.append({'id': family+'-'+kind, 'family': family, 'split': splits[n], 'domain': 'synthetic_mechanism',
                         'task': 'risk_status', 'label_authority': 'program_generated_given_rule_not_legal_gold',
                         'visible': {'question': '仅按给定模拟规则判断约定风险是否成立；这些规则不代表真实法律。',
                                     'actor': actor, 'event': event, 'proposition_type': 'simulated_fact',
                                     'fields': dict(zip(FIELDS, descriptions)), 'rule': rule, 'evidence': evidence},
                         'states': states, 'state_mask': [True]*4,
                         'target': bounds(rule, FIELDS, states)['state'], 'anchors': anchors})
        for other_kind in ['fact', 'exception']:
            pairs.append({'before': family+'-base', 'after': family+'-'+other_kind, 'kind': 'fact', 'split': splits[n]})
        for other_kind in ['paraphrase', 'irrelevant']:
            pairs.append({'before': family+'-base', 'after': family+'-'+other_kind, 'kind': 'equivalent', 'split': splits[n]})
        for other_kind in ['hidden', 'actor_swap']:
            pairs.append({'before': family+'-'+other_kind, 'after': family+'-base', 'kind': 'information', 'split': splits[n]})
    return rows, pairs


def single_atom(row, content, value, domain, split, ptype='as_supplied', visible=None):
    mapping = {'T': 'supported', 'F': 'refuted', 'U': 'unknown',
               'established': 'supported', 'not_established': 'refuted', 'unknown': 'unknown'}
    state = mapping[value]
    x = dict(visible or {})
    x.update({'question': '仅依据可见材料，判断指定命题。F表示该命题被反驳，不是任意风险都不存在。',
              'proposition_type': ptype, 'fields': {'A': content}, 'rule': 'A'})
    return {'id': row['id'], 'family': row['group_id'], 'split': split, 'domain': domain,
            'task': row['task'], 'label_authority': row.get('label_authority', 'original_v4_internal_dev_unchanged'), 'visible': x,
            'states': [state], 'state_mask': [True], 'target': DECISIONS[STATES.index(state)],
            'original_target': value,
            'annotations_scope': 'original_whole_proposition_only_not_invented_multifield_gold'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, pairs = synthetic()
    for r in read(OLD/'labor_auxiliary.jsonl'):
        if r['task'] != 'evidence_entailment':
            continue
        value = json.loads(r['messages'][-1]['content'])['label']
        text = r['messages'][1]['content']
        proposition = text.split('命题：', 1)[1].split('\n\n依据资料：', 1)[0]
        evidence = text.split('\n\n依据资料：', 1)[1]
        rows.append(single_atom(r, proposition, value, 'labor_atom', 'train',
                    visible={'evidence': [{'id': 'source', 'text': evidence, 'type': 'supplied_labor_material'}]}))
    for name, split in [('gold_train.jsonl', 'train'), ('internal_dev.jsonl', 'regression')]:
        for r in read(OLD/name):
            if r['task'] not in {'condition', 'qa', 'risk_status'}:
                continue
            x = r['input']
            rows.append(single_atom(r, x['proposition'], r['target']['value'], 'v4_legal', split,
                                    x['proposition_type'], {k: v for k, v in x.items() if k != 'question'}))
    ids = [r['id'] for r in rows]
    assert len(ids) == len(set(ids))
    byid = {r['id']: r for r in rows}
    for p in pairs:
        a, b = byid[p['before']], byid[p['after']]
        assert a['family'] == b['family'] and a['split'] == b['split'] == p['split']
        assert a['visible']['rule'] == b['visible']['rule']
        if p['kind'] == 'information':
            assert refinement_pair(a['states'], b['states'])
        if p['kind'] == 'equivalent':
            assert a['states'] == b['states'] and a['target'] == b['target']
    for name, data in [('examples.jsonl', rows), ('pairs.jsonl', pairs)]:
        (OUT/name).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in data), 'utf-8')
    manifest = {'version': 'v5-mechanism-pilot-1', 'source_v4_hashes': {n: sha(OLD/n) for n in
                ['gold_train.jsonl', 'internal_dev.jsonl', 'labor_auxiliary.jsonl', 'manifest.json']},
                'files': {n: sha(OUT/n) for n in ['examples.jsonl', 'pairs.jsonl']},
                'counts': dict(Counter(r['domain']+'/'+r['split'] for r in rows)),
                'targets': dict(Counter(r['domain']+'/'+r['split']+'/'+r['target'] for r in rows)),
                'synthetic_families': {'train': 64, 'dev': 16, 'test': 16}, 'split_seed': 20261004,
                'shared_synthetic_templates_across_splits': True,
                'all_1657_already_used_in_v4': True, 'labor_atom_task_subset': 390,
                'not_new_legal_gold': True, 'lab9_is_previously_seen_regression_only': True,
                'human_multifield_and_anchor_gold_available': False,
                'excluded_tasks_here': 'relation and grade have different semantics; original V4 files unchanged'}
    write(OUT/'manifest.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
