"""Post-freeze exploratory shift test. Never used to update the fifteen pilot heads."""
import itertools
import json
import sys
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_famu_v5 import RULES, FIELDS
from src.evidence_reasoning.risk_field_v5 import bounds


def main():
    out = ROOT/'training_data/famu_v5_20261004_stress'
    out.mkdir(exist_ok=True)
    rows = []
    values = list(itertools.product(['supported', 'refuted'], repeat=4))
    for n in range(32):
        actor, other = f'辰星{n+1}机构', f'北辰{n+1}机构'
        event = ['交付教材', '退还材料', '结清服务款', '发出结算通知'][n % 4]
        fields = {'A': f'该事项的完整原始协议载明{actor}应当在2026年5月10日前{event}',
                  'B': f'{actor}实际未在2026年5月10日前{event}',
                  'E': f'该事项协议明文允许免除本次逾期处理',
                  'D': f'{actor}的本次结算单存在尚未支付的费用'}
        states = list(values[n % 16])
        rule = RULES[(n//8) % 4]
        source_text = {
            'A': {'supported': f'经核对，这是本事项完整原始协议。其第六条写道：“{actor}须在2026年5月10日前{event}。”',
                  'refuted': f'已逐页核对本事项全部原始协议及附件。其中没有约定{actor}{event}的期限；无缺页或脱敏。'},
            'B': {'supported': f'实际履行记录及双方回执一致显示：{actor}直到2026年5月15日才完成{event}。',
                  'refuted': f'实际履行记录及双方回执一致显示：{actor}已在2026年5月5日完成{event}。'},
            'E': {'supported': '本事项协议第九条原文：“就本次事项，免予逾期处理。”该条没有附加前提。',
                  'refuted': '本事项完整协议第九条原文：“本次事项不适用任何逾期处理豁免。”无其他豁免条款。'},
            'D': {'supported': f'{actor}本次双方确认的结算单记载：应付300元，已付0元，尚欠300元。',
                  'refuted': f'{actor}本次双方确认的结算单记载：应付300元，已付300元，尚欠0元。'},
        }
        for kind in ['base', 'template_only', 'redacted_original', 'wrong_actor', 'wrong_time', 'paraphrase', 'irrelevant', 'conflict']:
            ss = states.copy()
            text = {k: source_text[k][s] for k, s in zip(FIELDS, ss)}
            if kind == 'template_only':
                ss[1] = 'unknown'
                text['B'] = f'模板写明：“{actor}承诺按期{event}”。本事项实际履行日期和回执均未提供。'
            if kind == 'redacted_original':
                ss[0] = 'unknown'
                text['A'] = '本次只有脱敏副本；第六条显示为“[履行期限已隐去]”。原件和脱敏前文本均不可得。'
            if kind == 'wrong_actor':
                ss[1] = 'unknown'
                text['B'] = source_text['B'][states[1]].replace(actor, other)+f'该记录属于{other}；没有{actor}的实际履行记录。'
            if kind == 'wrong_time':
                ss[1] = 'unknown'
                text['B'] = source_text['B'][states[1]].replace('2026年5月', '2025年5月')+'这份记录属于上一年度的独立事项；本次事项实际履行记录缺失。'
            if kind == 'paraphrase':
                text = {k: '卷宗核对摘录：'+v.replace('实际履行记录及双方回执一致显示：', '两方签认的办理时间为：').replace('尚欠0元', '余额为零').replace('尚欠300元', '余额尚有三百元') for k, v in text.items()}
            if kind == 'irrelevant':
                text['unrelated'] = f'{other}在2024年度有另一份交付记录。档案袋为绿色，页码从1开始。'
            if kind == 'conflict':
                ss[1] = 'conflict'
                text['B'] = '相同主体和事项有两份同等效力的回执，矛盾未裁决。回执甲：'+source_text['B']['supported']+'回执乙：'+source_text['B']['refuted']
            rows.append({'id': f'SHIFT-{n:03d}-{kind}', 'family': f'SHIFT-{n:03d}', 'split': 'shift_test',
                         'domain': 'synthetic_shift', 'task': 'risk_status',
                         'label_authority': 'post_freeze_program_generated_not_legal_gold',
                         'visible': {'question': '依照本题明示模拟规则判断是否触发约定处理；模拟规则不代表真实法律。',
                                     'actor': actor, 'event': event, 'proposition_type': 'mixed_text_actual_permission',
                                     'fields': fields, 'rule': rule,
                                     'evidence': [{'id': f'S{i+1}', 'text': t, 'type': 'simulated_record'} for i, t in enumerate(text.values())]},
                         'states': ss, 'state_mask': [True]*4, 'target': bounds(rule, FIELDS, ss)['state'], 'variant': kind})
    path = out/'examples.jsonl'
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), 'utf-8')
    manifest = {'examples_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'rows': len(rows), 'families': 32,
                'created_after_initial_pilot_results': True, 'new_training_updates': 0,
                'not_independent_human_legal_validation': True, 'purpose': 'fixed-head exploratory distribution shift stress test'}
    (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), 'utf-8')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
