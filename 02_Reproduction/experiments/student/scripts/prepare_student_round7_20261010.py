# Deidentified copy. String literals containing machine locations were replaced.
"""Freeze a bounded language-context x contrast-training ablation (fictional)."""
import copy, hashlib, json, shutil, sys, zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import prepare_student_round3_20261010 as r3
from src.evidence_reasoning.local_field_student import validate, visible_input, digest
OUT = ROOT / 'outputs/famu_student_round7_20261010'
PREV = ROOT / 'outputs/famu_student_round6_20261010'
QUESTION = {k: list(r3.QUERY[k][:4]) + v for k, v in {'A': ['双方有效签约材料是否给{a}{e}设定{d}这个日期要求', '对{a}{e}而言，{d}这一履行时限是否有完整有效的约定作为依据'], 'B': ['实际完成凭据能否证实{a}{e}已经晚于{d}', '经证实的{a}{e}全部结束之日是否落在{d}以后'], 'C': ['此次迟延是否已有对方生效决定允许{a}免于处理', '{a}针对本轮逾期是否取得了已经生效的免除处理决定'], 'D': ['核实本期到期应付和到账情况后，{a}是否还有未付金额', '在本次到期结算中，{a}是否留有尚未支付的余款']}.items()}
TEXT = {'A': {'T': ['{m}已逐页核实所有有效签约资料，资料没有缺失；其中明确约定{a}于{d}前{e}，并非没有日期约定。', '{m}没有缺少有效签署材料。双方确已写明{a}最迟在{d}{e}，不是仅仅打算约定。', '{m}全部签约原页清晰可查；不能说双方未定期限，因为有效正文明确要求{a}{d}之前{e}。', '{m}有效文件不存在遗漏，{a}{e}的期限不是空白，而是双方已签认的{d}。', '{m}经逐本核查，签署文件未缺一本；{a}{e}的日期并未留待商议，有效约定为{d}之前。', '{m}已查过生效主文与每份附件，材料均完整；不是未约定{a}{e}的时间，双方签定的最晚时间正是{d}。'], 'F': ['{m}已逐页核实所有有效签约资料，资料没有缺失；其中未约定{a}{e}的日期，并非没有查到资料。', '{m}没有缺少有效签署材料。双方确实没给{a}{e}写下任何期限，不是仅仅没看到原页。', '{m}全部签约原页清晰可查；不能说双方定了期限，因为正文及全部补充协议均未规定{a}{e}日期。', '{m}有效文件不存在遗漏，{a}{e}的期限不是被遮掉，而是签署原文从来没有约定。', '{m}经逐本核查，签署文件未缺一本；双方并未给{a}{e}订下具体时间，这是完整原文的内容。', '{m}已查过生效主文与每份附件，材料均完整；不是未拿到{a}{e}的时间页，而是所有有效文本根本未约定期限。'], 'U': ['{m}未逐页核实所有有效签约资料，资料有缺失；不能确认{a}{e}是否约定日期，并非确认没有日期约定。', '{m}缺少有效签署材料。没看到{a}{e}的期限，但不是已经确认双方未写期限。', '{m}签约原页并非全部清晰可查；不能说双方未定期限，也不能说已定，{a}{e}的原文仍未提供。', '{m}有效文件仍存在遗漏，{a}{e}的期限不是已被证实空白，而是当前无法看到。', '{m}尚未逐本核查，签署文件缺了一部分；未看到{a}{e}的时间不能说明双方并未订下时间。', '{m}未查到生效主文和若干附件，材料并不完整；不是已确认未约定{a}{e}的时间，而是有没有约定都无法查明。']}, 'B': {'T': ['{m}核实的实际回单表明{a}直到{late}才全部{e}；没有将计划当成结果，此前确未完成。', '{m}并非未获得实际回执：全部回执一致确认{a}于{late}完成{e}，不是计划。', '{m}不能说{a}已在截止前{e}，因为实绩核验确认全部办结之日为{late}。', '{m}没有遗漏的交接记录，{a}{e}不是在{early}完成，而是在{late}才全部完成。', '{m}已取得完整履行凭证，{a}{e}直到{late}才结束；这个实绩不是估计。', '{m}所查全部实际凭证已核实，{a}把{e}做完是在{late}；此时已经过了期限，所说日期绝非预约。'], 'F': ['{m}核实的实际回单表明{a}早在{early}就全部{e}；没有将计划当成结果，此后确无待办。', '{m}并非未获得实际回执：全部回执一致确认{a}于{early}完成{e}，不是计划。', '{m}不能说{a}超过截止才{e}，因为实绩核验确认全部办结之日为{early}。', '{m}没有遗漏的交接记录，{a}{e}不是尚待完成，而是在{early}已全部完成。', '{m}已取得完整履行凭证，{a}{e}早在{early}就结束；这个实绩不是估计。', '{m}所查全部实际凭证已核实，{a}把{e}做完是在{early}；此时还没到期限，所说日期绝非预约。'], 'U': ['{m}没有取得核实的实际回单，{a}说将在{late}{e}；不能将计划当成结果，何时完成不明。', '{m}并非已获得实际回执：仅有{a}于{late}{e}的计划，是否完成尚未查明。', '{m}不能说{a}截止前已{e}，也不能说已超过截止；实绩未核验，只见{late}的安排。', '{m}缺少交接记录，{a}{e}不是已被确认完成或没完成，{late}仅是预计日期。', '{m}未取得完整履行凭证，{a}{e}估计在{late}结束；这个安排不能当成实绩。', '{m}所缺的是实际履行凭证；{a}预约{late}{e}，不代表那天已经做完，早晚与完成与否均没有查实。']}, 'C': {'T': ['{m}本次{a}逾期的完整审批档案已核实，最终决定是批准免责，不是拒绝，且已无条件生效。', '{m}对方不是尚未回复，而是已有效批准{a}本次迟延免责，不必再等待条件。', '{m}全部审批原件显示，不能说已拒绝{a}本次免责；有效定案为准予免责。', '{m}本次{a}的免责并非申请中的设想，而是已经有效签署的同意，无待办条件。', '{m}全部批复已经核准，{a}这一次逾期并未被拒绝免责，最终书面决定为无条件批准且已生效。', '{m}此番迟延的审批原文已全数核明，对方并不是没有作答；已签署生效决定批准{a}本次免责，无须等条件成就。'], 'F': ['{m}本次{a}逾期的完整审批档案已核实，最终决定是拒绝免责，不是批准，且已经生效。', '{m}对方不是尚未回复，而是已有效拒绝{a}本次迟延免责，没有其他相反批准。', '{m}全部审批原件显示，不能说已批准{a}本次免责；有效定案为不予免责。', '{m}本次{a}的免责并非仍在申请中，而是已有有效签署的拒绝，无其他批准。', '{m}全部批复已经核准，{a}这一次逾期并未被批准免责，最终书面决定为拒绝且已生效。', '{m}此番迟延的审批原文已全数核明，对方并不是没有作答；已签署生效决定拒绝{a}本次免责，无其他同意决定。'], 'U': ['{m}本次{a}逾期的审批档案未核实，尚无最终决定；没有批准不等于拒绝，没有拒绝也不等于批准。', '{m}对方尚未回复，不是已有效拒绝{a}本次迟延免责，也不是已批准。', '{m}审批原件尚缺，不能说已拒绝或已批准{a}本次免责，结果仍待核实。', '{m}本次{a}的免责仍在申请中，并非已经有效签署的同意或拒绝。', '{m}全部批复尚未取得，{a}这一次免责的最终决定不明，不能把没看到批准当成拒绝。', '{m}此番迟延的审批原文还没取得；不能认定对方已同意，也不能认定已拒绝{a}本次免责，未见决定不等于拒绝决定。']}, 'D': {'T': ['{m}本期所有账款已核实，{a}应付{total}元到账{paid}元，欠款{owed}元未清；不是未查到账目。', '{m}账目并非没有核清，{a}到期应付{total}元只收{paid}元，已确认还欠{owed}元。', '{m}不能说本期{a}没有欠款：全部到期账核实后，应付{total}元而实收{paid}元，仍欠{owed}元。', '{m}本期{a}并非已结清，应付{total}元、实收{paid}元、尚欠{owed}元均已核实。', '{m}所有到期款项与银行收入均已比对，{a}尚欠{owed}元，应付{total}元只收{paid}元，并非推测。', '{m}当期到期对账已经做完，{a}应交{total}元却仅到账{paid}元；未清偿的{owed}元是查实的欠款，不是未完成核账。'], 'F': ['{m}本期所有账款已核实，{a}应付{total}元到账{total}元，欠款0元；不是未查到账目。', '{m}账目并非没有核清，{a}到期应付{total}元全数收到，已确认不再欠款。', '{m}不能说本期{a}尚有欠款：全部到期账核实后，应付{total}元与实收{total}元相等，余额为0元。', '{m}本期{a}并非尚欠款，应付{total}元、实收{total}元、余额0元均已核实。', '{m}所有到期款项与银行收入均已比对，{a}欠款为0元，应付{total}元已全部收到，并非推测。', '{m}当期到期对账已经做完，{a}应交{total}元且已到账{total}元；没有欠款是查实的结算结果，不是未查到欠款资料。'], 'U': ['{m}本期账款没有全部核实，{a}预算{total}元，欠款与否未明；不是已经查实没有欠款。', '{m}账目尚未核清，{a}计划付{total}元，不能确认还欠款或不再欠款。', '{m}不能说本期{a}没有欠款，也不能说有：到期与到账资料均缺，{total}元只是计划。', '{m}本期{a}并非已被确认结清或欠款，{total}元仅是预算，应付与实收未核实。', '{m}到期款项与银行收入没有完成比对，{a}预算{total}元，欠款不能查明，不能由此认定欠款为零。', '{m}当期到期对账还没有做完，{a}只列预算{total}元；未见欠款资料不是已证实没有欠款，应付和到账记录仍缺。']}}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf-8')

def scenes():
    q, t = (r3.QUERY, r3.TEXT)
    try:
        r3.QUERY, r3.TEXT = (QUESTION, TEXT)
        rows = r3.generate('train', 18, [0, 1, 2, 3]) + r3.generate('dev', 6, [4]) + r3.generate('test', 8, [5])
    finally:
        r3.QUERY, r3.TEXT = (q, t)
    for r in rows:
        for k in ['id', 'family', 'pair_group']:
            r[k] = r[k].replace('R3-', 'R7-')
        s = json.dumps(r, ensure_ascii=False).replace('溪序', '惠青').replace('峦序', '靖川').replace('汀序', '遥序')
        r.clear()
        r.update(json.loads(s))
        validate(visible_input(r))
    return rows

def contrast_pairs(rows):
    train = [r for r in rows if r['split'] == 'train']
    questions = {}
    lookup = {}
    for r in train:
        for i, (f, q) in enumerate(r['fields'].items()):
            questions.setdefault((r['family'], f), set()).add(q)
            lookup[r['family'], q] = i
    expanded = []
    for r in train:
        for j in range(4):
            x = copy.deepcopy(r)
            x['fields'] = {f: sorted(questions[r['family'], f])[j] for f in 'ABCD'}
            expanded.append(x)
    pairs = r3.pairs(expanded)
    for p in pairs:
        p['query_kind'] = lookup[p['group'], p['query']]
        p['source'] = 'R7_train_contrast_simulation'
    return pairs

def main():
    if (OUT / 'package').exists():
        raise RuntimeError('Existing R7 preserved')
    package = OUT / 'package'
    package.mkdir(parents=True)
    manifest = json.loads((PREV / 'package/input_manifest.json').read_text('utf-8'))
    for rel, e in manifest.items():
        if rel.startswith(('src/', 'scripts/')):
            src = PREV / 'package' / rel
            assert sha(src) == e['sha256']
            dst = package / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
    for rel in ['src/evidence_reasoning/clause_field_student.py', 'scripts/student_round7_model.py', 'scripts/prepare_student_round7_20261010.py', 'scripts/run_student_round7_20261010.py']:
        shutil.copyfile(ROOT / rel, package / rel)
    (package / 'data').mkdir()
    for name in ['fit', 'historical', 'r2_regression', 'r3_regression', 'r4_regression', 'r5_regression']:
        shutil.copyfile(PREV / f'package/data/{name}.json', package / f'data/{name}.json')
    write(package / 'data/r6_regression.json', [r for r in json.loads((PREV / 'package/data/new_scenes.json').read_text('utf-8')) if r['split'] == 'test'])
    rows = scenes()
    write(package / 'data/new_scenes.json', rows)
    pairs = contrast_pairs(rows)
    write(package / 'data/contrast_pairs.json', pairs)
    fit = json.loads((package / 'data/fit.json').read_text('utf-8'))
    allfit = sum(fit.values(), []) + pairs
    held = [r for r in rows if r['split'] != 'train']
    assert not {p['group'] for p in allfit} & {r['family'] for r in held}
    assert not {digest({k: p[k] for k in ['query', 'text', 'scope']}) for p in allfit} & {p['id'] for p in r3.pairs(held)}
    for seed in [17, 29, 43]:
        src = PREV / f'result_bundle/outputs'
        dst = package / f'baseline_s{seed}'
        dst.mkdir()
        for p in (src / f'LANG_HARD_s{seed}/model').iterdir():
            shutil.copyfile(p, dst / p.name)
        for name in ['initialization.json', 'training_log.jsonl']:
            shutil.copyfile(src / f'LANG_s{seed}' / name, dst / name)
        m = json.loads((dst / 'manifest.json').read_text('utf-8'))
        for n, h in m.items():
            assert sha(dst / n) == h
    protocol = {'id': 'famu_student_round7_20261010', 'seeds': [17, 29, 43], 'training_arms': ['CLAUSE_BASE', 'BAG_CONTRAST', 'CLAUSE_CONTRAST'], 'evaluation_arms': ['BAG_BASE', 'CLAUSE_BASE', 'BAG_CONTRAST', 'CLAUSE_CONTRAST'], 'max_total_updates': 8640, 'max_total_exposures': 276480, 'auxiliary_query_labels': 276480, 'updates_per_run': 960, 'epochs': 12, 'steps_per_epoch': 80, 'batch_size': 32, 'wall_hours_max': 2, 'training': 'BASE uses exact R6 fit/order. CONTRAST replaces 8/32 recombined exposures with training-only paired language examples (16 original,8 recombined,8 contrast); class16/8/8 and family-uniform. New fictional language supervision, not teacher distillation or independent legal facts.', 'new_fit_pairs': len(pairs), 'new_train_scenes': 864, 'new_dev': 72, 'new_test': 96, 'representation': 'BAG is frozen R6 lexical bag; CLAUSE adds punctuation-boundary and ordered negation/condition cue windows hashed into same4096 dimensions. Same parameter architecture and initialization; not a syntax parser or hard-coded verdict.', 'baseline': 'BAG_BASE reuses all3 frozen R6 LANG weights, no updates. All4 arms hard predicted query type; no oracle metadata. BASE initialization/order must match R6; paired CONTRAST arms must match each other.', 'optimizer': 'FP32 TF32off, AdamW .01 wd.01, warmup96 linear960, clip1; complete epoch6/12 optimizer/scheduler/RNG/exposure checkpoints.', 'selection': 'All12 arms selected only historical dev+R7 newdev; threshold .5/.6/.7/.8/.9 and gate0/.4/.6; minimize unknown strong, maximize macroF1/citationF1, lower gate/threshold. Lock all before tests. Also3 frozen R6 original thresholds.', 'research_signal': 'CLAUSE_CONTRAST vs BAG_BASE: ALL3 seeds strictly improve newtest correct and citationF1, fieldF recall nondecrease, unknown strong no increase, and correct/citationF1 nondecrease on every historical/R2/R3/R4/R5/R6 regression. Report factor ablations and all seeds. No promotion.', 'holdout': 'New dev/test wording banks and named families; still program-generated with shared structures/distractors, not blind expert/legal validation. Previously examined tests are regression only. No feature/hyperparameter edits after freezing.', 'release': 'Shadow only; preserve input validation, original quote anchors, unknown/conflict and rules. Formal77/LAB9/actual4core8GiB hard/concurrent website acceptance not shown. Stop this finite round, no automatic same-recipe repetition.', 'references': ['httpREPRO_ROOT/', 'httpREPRO_ROOT/', 'httpREPRO_ROOT/']}
    write(package / 'protocol.json', protocol)
    entries = {p.relative_to(package).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in package.rglob('*') if p.is_file()}
    write(package / 'input_manifest.json', entries)
    with zipfile.ZipFile(OUT / 'inputs.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(package.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(package).as_posix())
    receipt = {'input_zip_sha256': sha(OUT / 'inputs.zip'), 'input_zip_bytes': (OUT / 'inputs.zip').stat().st_size, 'protocol_sha256': sha(package / 'protocol.json'), 'new_scenes_sha256': sha(package / 'data/new_scenes.json'), 'contrast_pairs_sha256': sha(package / 'data/contrast_pairs.json'), 'files': len(entries), 'new_fit_pairs': len(pairs)}
    write(OUT / 'preparation.json', receipt)
    print(json.dumps(receipt))
if __name__ == '__main__':
    main()
