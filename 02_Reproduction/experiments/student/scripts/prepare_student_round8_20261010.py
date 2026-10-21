# Deidentified copy. String literals containing machine locations were replaced.
"""Freeze R8: identical existing training pools; new fictional language holdouts."""
import copy, json, hashlib, shutil, sys, zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import prepare_student_round3_20261010 as r3
from src.evidence_reasoning.local_field_student import visible_input, validate, digest
OUT = ROOT / 'outputs/famu_student_round8_20261010'
PREV = ROOT / 'outputs/famu_student_round7_20261010'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf-8')
Q = {'A': ['完整有效文书是否约束{a}在{d}之前{e}', '{a}{e}须以{d}为最终期限，这一点在双方有效文件中有无订明'], 'B': ['经核真的{a}{e}办结实绩是不是迟于{d}', '把{a}{e}的已证实完结时间与{d}相比，是否已越过期限'], 'C': ['对方的有效最终意见是否已让{a}免去这一次逾期处理', '{a}此番迟延有没有已经生效而不再待批的免责许可'], 'D': ['当前到期结算是否证实{a}还有应付未付的款项', '对于{a}本期已到期款项，查清实收之后是否仍余欠额']}
TEXT = {'A': {'T': ['{m}先前有人说“{a}{e}没定时间”。核对双方签署原件后更正：最迟{d}前完成；原件及附件均齐备，以上引述作废。', '{m}的签署原本及全部增补已经核验：{a}必须不晚于{d}{e}。原始约定就是如此；“没有定期”的早期转述系误记，应予撤销。'], 'F': ['{m}先前有人说“{a}{e}有时间约定”。核对双方全部有效原件及附件后更正：无任何日期条款；不是缺少原页，以上引述作废。', '{m}的签署原本及全部增补已经核验：根本未给{a}{e}设定日期。所有页码均已查全；“已定期”的早期转述系误记，应予撤销。'], 'U': ['{m}先前有人说“{a}{e}有时间约定”。原件及附件至今没齐，核验尚不能进行；引述没有被证实，既不能证明约定了日期，也不能证明未约定。', '{m}现在掌握的只是转述：有人称{a}{e}已经定期。签署原本缺页，增补也未收全，无法判断这句话真伪，更不能据此认定没有日期约定。']}, 'B': {'T': ['{m}旧摘要写“{a}可能在{early}{e}”。最终实际交接回执纠正了它：直到{late}才全部办结，之前有未办部分，旧摘要无效。', '{m}经查双方实际签收流水，{a}把{e}全部落实在{late}，此前尚有工作未完。这是已发生的结果；曾传出的{early}只是猜测并已撤回。'], 'F': ['{m}旧摘要写“{a}可能在{late}{e}”。最终实际交接回执纠正了它：{early}已全部办结，此后没有未办部分，旧摘要无效。', '{m}经查双方实际签收流水，{a}把{e}全部落实在{early}，之后再无工作未完。这是已发生的结果；曾传出的{late}只是猜测并已撤回。'], 'U': ['{m}旧摘要写“{a}可能在{late}{e}”。这一预计从未被实际回执确认，交接凭证仍缺，摘要不能证明已办结或尚未办结。', '{m}尚无双方实际签收流水。关于{a}{e}曾传出{late}这个时间，但仅是将来安排，履行结果与真正完结时间都无法查清。']}, 'C': {'T': ['{m}早先对{a}的本次免责请求曾有拒绝草稿，草稿未生效。最终已签署生效的决定是同意免责，且没有尚待满足的条件。', '{m}全部有效审批资料相互一致：对方已经批准{a}这次逾期免受处理，无条件生效。先前那份写着“拒绝”的工作稿没有获签，已经废止。'], 'F': ['{m}早先对{a}的本次免责请求曾有同意草稿，草稿未生效。最终已签署生效的决定是拒绝免责，相关资料全齐，无其他相反批准。', '{m}全部有效审批资料相互一致：对方已经驳回{a}这次逾期免受处理的请求，决定生效。先前那份写着“同意”的工作稿没有获签，已经废止。'], 'U': ['{m}对{a}本次免责的同意与拒绝各有一份草稿，但均未签署生效。最终决定至今未出，两份草稿不能证明任何有效结果。', '{m}针对{a}这次逾期的材料还停留在拟稿：一份倾向批准，另一份倾向拒绝。两份都未获得签署，也没有最终审批，不能当成已经生效的决定。']}, 'D': {'T': ['{m}关于{a}本期“已经付清”的旧说法经对账被纠正：实际到期应付{total}元、到账{paid}元、仍差{owed}元，原银行凭据已核真。', '{m}实际应付及到账均完成核对：{a}本期应交{total}元，银行只实收{paid}元，未付{owed}元。早先的“余款为零”来自未经核实的摘要，现已更正。'], 'F': ['{m}关于{a}本期“还有欠款”的旧说法经对账被纠正：实际到期应付{total}元、到账{total}元、差额0元，无其他到期款，原银行凭据已核真。', '{m}实际应付及到账均完成核对：{a}本期应交{total}元，银行已实收{total}元，未付0元且没有其他到期款。早先的“尚有欠额”来自未经核实的摘要，现已更正。'], 'U': ['{m}关于{a}本期“已经付清”的旧说法尚未得到核验：{total}元只是预算，应付及银行到账原始凭据仍缺，不能计算真实差额。', '{m}实际应付及到账都未完成核对：{a}本期只列{total}元预算。所谓“尚有欠额”来自未经核实的摘要，真实应付和银行实收仍无依据，不能确认欠款或结清。']}}
SHORT = {'A': {'T': ['{m}原件全齐。租约已签，{a}须在{d}前{e}。', '{m}有效合同查全了。双方定了{a}{d}前{e}，有签字。', '{m}签本和补充条款都查全；{a}{e}最晚{d}，双方已签定。'], 'F': ['{m}签过的租约和附件全齐。没有给{a}{e}定日期。', '{m}所有有效合同查全了。{a}{e}没约时间，不是合同没拿到。', '{m}签本和补充条款都查全；{a}{e}没订时限，无缺页。'], 'U': ['{m}租约缺页。{a}{e}定没定日期，还不知道。', '{m}有效合同没拿齐。看不出{a}{e}约没约时间。', '{m}缺签本和补充页；{a}{e}时限有无约定，无法核定。']}, 'B': {'T': ['{m}实际交接单核实了：{a}{late}才全部{e}，此前没做完。', '{m}真回单已核对，{a}一直到{late}才办完{e}，不是计划。', '{m}全部实绩凭证确认：{a}{late}才办妥{e}，此前未全部完成。'], 'F': ['{m}实际交接单核实了：{a}{early}已全部{e}，没有待办。', '{m}真回单已核对，{a}{early}就办完{e}，不是计划。', '{m}全部实绩凭证确认：{a}{early}便办妥{e}，之后无剩余事项。'], 'U': ['{m}{a}打算{late}{e}。没实际交接单，做没做完不清楚。', '{m}只有{a}{late}{e}的安排。缺真回单，没法确认结果。', '{m}{a}{late}{e}是预约；实绩凭证未取得，是否办妥不明。']}, 'C': {'T': ['{m}这次{a}晚交，房东已签字同意免处理。已生效，不用再等条件。', '{m}房东已正式批准{a}本轮迟延免责，生效了，没附加条件。', '{m}{a}此番迟延的豁免已获对方有效签认，无待成就条件。'], 'F': ['{m}这次{a}晚交，房东已签字说不免处理。拒绝已生效，无别的批准。', '{m}房东已正式驳回{a}本轮迟延免责，生效了，没其他同意。', '{m}{a}此番迟延的豁免已被对方有效驳回，无任何相反许可。'], 'U': ['{m}这次{a}晚交，只问了房东能不能免处理。房东没答，不等于批准或拒绝。', '{m}{a}刚申请本轮迟延免责，房东还没批，也没驳回。', '{m}{a}此番迟延的豁免申请未有最终回函；批准和拒绝均无从认定。']}, 'D': {'T': ['{m}{a}本期交租该付{total}元，只到账{paid}元，还差{owed}元。到期账核实了。', '{m}{a}这期到期房租{total}元，实收{paid}元，欠{owed}元，对账无误。', '{m}本期到期租金查核无误：{a}应交{total}元，入账{paid}元，余欠{owed}元。'], 'F': ['{m}{a}本期交租该付{total}元，已全到账，还差0元。全部到期账核实了。', '{m}{a}这期到期房租{total}元，实收{total}元，没别的欠款，对账无误。', '{m}本期全部到期租金查核无误：{a}应交{total}元，入账{total}元，余欠0元。'], 'U': ['{m}{a}想拿{total}元交租。实际该付多少、到了多少都没凭据，欠没欠不知道。', '{m}{a}这期只做了{total}元房租预算。到期账和收款没核实，欠款不清楚。', '{m}本期{a}仅有{total}元租金预算；应付和实收凭证均缺，余欠无法核定。']}}
TRAIN_Q = {k: list(r3.QUERY[k][:2]) + v for k, v in {'A': ['租约有没有写明{a}得在{d}前{e}', '租房的有效合同里，给{a}{e}定了{d}这个时间吗'], 'B': ['{a}实际办完{e}是不是晚过{d}', '{a}{e}真正做完时，过了{d}没有'], 'C': ['房东已有效同意免掉{a}这次晚交的处理了吗', '{a}这回迟延，已经拿到有效的免责许可吗'], 'D': ['{a}这期到期房租，查实还有没交的吗', '核过账后，{a}本期到期的租金还差钱吗']}.items()}

def scenes():
    oq, ot = (r3.QUERY, r3.TEXT)
    try:
        r3.QUERY = {k: TRAIN_Q[k] + v + [v[1]] for k, v in Q.items()}
        r3.TEXT = {k: {s: list(ot[k][s][:2]) + SHORT[k][s][:2] + v + [SHORT[k][s][2]] for s, v in states.items()} for k, states in TEXT.items()}
        rows = r3.generate('train', 18, [0, 1, 2, 3]) + r3.generate('dev', 6, [4]) + r3.generate('test', 8, [5, 6])
    finally:
        r3.QUERY, r3.TEXT = (oq, ot)
    for r in rows:
        for k in ['id', 'family', 'pair_group']:
            r[k] = r[k].replace('R3-', 'R8-')
        s = json.dumps(r, ensure_ascii=False).replace('溪序', '蓉砺').replace('峦序', '澜砺').replace('汀序', '棠砺')
        for a, b in [('提交安装清单', '返还承租房屋'), ('返还计量装置', '移交租赁钥匙'), ('移交检验报告', '交付出租房屋'), ('交付维护组件', '退租交房')]:
            s = s.replace(a, b)
        domain = ['房屋租赁', '租房', '租赁房屋', '房屋租住'][r['style_id'] % 4]
        s = s.replace('本次核验事项', domain + '本次核验事项')
        r.clear()
        r.update(json.loads(s))
        if r['style_id'] == 6:
            r['split'] = 'test_short'
        prefix = ['房屋租赁记录：', '租房记录：', '房子租住资料：', '住房出租材料：', '承租住房的档案：', '租住这套房的记录：', '住房承租资料：'][r['style_id']]
        for e in r['evidence']:
            e['text'] = prefix + e['text']
        validate(visible_input(r))
    return rows

def main():
    if (OUT / 'package').exists():
        raise RuntimeError('Existing R8 preserved')
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
    for rel in ['src/evidence_reasoning/semantic_reward_student.py', 'scripts/student_round8_model.py', 'scripts/prepare_student_round8_20261010.py', 'scripts/run_student_round8_20261010.py']:
        shutil.copyfile(ROOT / rel, package / rel)
    (package / 'data').mkdir()
    for n in ['fit', 'historical', 'r2_regression', 'r3_regression', 'r4_regression', 'r5_regression', 'r6_regression', 'contrast_pairs']:
        shutil.copyfile(PREV / f'package/data/{n}.json', package / f'data/{n}.json')
    write(package / 'data/r7_regression.json', [r for r in json.loads((PREV / 'package/data/new_scenes.json').read_text('utf-8')) if r['split'] == 'test'])
    rows = scenes()
    write(package / 'data/new_scenes.json', rows)
    from scripts.prepare_student_round7_20261010 import contrast_pairs
    lp = contrast_pairs(rows)
    for p in lp:
        p['source'] = 'R8_train_lease_synonyms_and_semantics_preserving_shortening'
    write(package / 'data/language_pairs.json', lp)
    fit = sum(json.loads((package / 'data/fit.json').read_text('utf-8')).values(), []) + json.loads((package / 'data/contrast_pairs.json').read_text('utf-8')) + lp
    held = [r for r in rows if r['split'] != 'train']
    assert not {p['group'] for p in fit} & {r['family'] for r in held}
    assert not {p['id'] for p in fit} & {p['id'] for p in r3.pairs(held)}
    r2 = ROOT / 'outputs/famu_student_round2_20261010/package'
    r2p = json.loads((r2 / 'protocol.json').read_text('utf-8'))
    src = r2 / 'data/teacher_fit_scores.jsonl'
    assert sha(src) == r2p['teacher_targets_sha256']
    shutil.copyfile(src, package / 'data/teacher_fit_scores.jsonl')
    for seed in [17, 29, 43]:
        src = PREV / f'result_bundle/outputs/BAG_CONTRAST_s{seed}/model'
        dst = package / f'query_source_s{seed}'
        dst.mkdir()
        m = json.loads((src / 'manifest.json').read_text('utf-8'))
        for n, h in m.items():
            assert sha(src / n) == h
            shutil.copyfile(src / n, dst / n)
        shutil.copyfile(src / 'manifest.json', dst / 'manifest.json')
    enc = ROOT / 'outputs/famu_student_gpu_20261009/package/encoder'
    (package / 'encoder').mkdir()
    for n, h in r2p['encoder_sha256'].items():
        assert sha(enc / n) == h
        shutil.copyfile(enc / n, package / 'encoder' / n)
    protocol = {'id': 'famu_student_round8_20261010', 'seeds': [17, 29, 43], 'arms': ['LEX_SFT', 'JOINT_SFT', 'EVID_SFT', 'EVID_VRL', 'EVID_MRL'], 'max_total_updates': 11520, 'max_total_exposures': 368640, 'updates_per_effective_path': 960, 'warm_updates': 480, 'continuation_updates': 480, 'batch_size': 32, 'wall_hours_max': 2, 'training': 'Four pools:16original/8recombined/4R7contrast/4newR8lease synonym+shortening pairs per batch; class16/8/8, eachsource family uniform. 864 new authored training scenes, repeated rephrased facts NOT independent legal data. Frozen pretrained encoder. Continue identical auxiliary queryCE in all arms from per-seed R7 BAG_CONTRAST question head; train common4618->64->12 relation head. No encoder optimization. Actual auxiliary query exposures equal368640 maximum.', 'representations': 'LEX zero512semantic features; JOINT frozen Chinese4layer512 mean normalized embedding of observable scope+evidence+question; EVID same encoder sees scope+evidence only. All retain4096lexical/10rawnumeric. Same head capacity, paired initialization and orders.', 'branching': 'LEX/JOINT independently960 updates. EVID shared480 SFT then3 cloned model+optimizer+scheduler+RNG continuations480; actual cost3840 updates/seed; each effective path960. Save240/480 warm,480/960 controls,720/960 continuations.', 'optimizer': 'Head FP32 AdamW .01 wd.01 clip1,96warmup linear960, TF32off. Frozen encoder FP32 eager/no dropout. No sampling generation; exact expected reward over3actions.', 'objective': 'SFT=CE; VRL=CE-.5 E_pi[R]+.02 KL(pi||frozen warm policy); MRL=VRL-.2 E_pi[T]. R rows gold0/1/2=[[1,-2,-2],[-.5,1,-2],[-.5,-2,1]]. T=softmax(teacher mean_scores/.25)-1/3, NOT calibrated probability.', 'teacher_gate': 'Only cached2500 original training records with accepted and mean/sum argmax==gold and mean_score_range>=.05. Global>=384 and eachclass>=32. Zero teacher reward elsewhere; if global fails mark MRL not admitted and skip its480updates per seed; never relabel. No new9B inference.', 'teacher_targets_sha256': r2p['teacher_targets_sha256'], 'encoder_sha256': r2p['encoder_sha256'], 'selection': 'All15 endpoints threshold.5/.6/.7/.8/.9 and query_gate0/.4/.6 selected on historicaldev+new72dev only, minimize unknown strong then maxmacroF1/citationF1 thenlowergate/threshold. Lock ALL before test predictions.', 'research_signal': 'ALL3 seeds EVID_SFT must improve BOTH new96long and96short tests correct/citation vsLEX_SFT, nondecrease fieldF and each oldregression correct/citation with no unknownstrong increase. Teacher reward separately requires same comparison EVID_MRL vsEVID_VRL. ReportJOINT control and long/short paired both-correct. No production promotion.', 'new_holdout': '72dev96longtest96shorttest fictional lease cases including retracted summaries/current verified records/unsigned drafts/plans; separate train/dev/test families and text banks. Long/short tests share facts; not independent of each other. Shortening preserves polarity/scope/time/completeness/actual vs plan. Shared generator/distractors, NOT real legal gold. AllR2-R7 tests regression only.', 'runtime': 'Small~24M parameter encoder only, original scope validation/evidence anchors/unknown/conflict/explicitrules unchanged. Bounded perrequest embedding reuse. CPUshort/resource tests do not certify4core8GiB website concurrency.', 'release': 'Shadow only. Formal77/LAB9/fullwebsite tests outstanding. Finish one bounded round; no auto repetition.', 'references': ['httpREPRO_ROOT/2212.08073', 'httpREPRO_ROOT/2210.10760', 'httpREPRO_ROOT/rl_intro3.html']}
    write(package / 'protocol.json', protocol)
    entries = {p.relative_to(package).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in package.rglob('*') if p.is_file()}
    write(package / 'input_manifest.json', entries)
    with zipfile.ZipFile(OUT / 'inputs.zip', 'x', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for p in sorted(package.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(package).as_posix())
    receipt = {'input_zip_sha256': sha(OUT / 'inputs.zip'), 'input_zip_bytes': (OUT / 'inputs.zip').stat().st_size, 'protocol_sha256': sha(package / 'protocol.json'), 'new_scenes_sha256': sha(package / 'data/new_scenes.json'), 'files': len(entries), 'fit_pairs': len(fit)}
    write(OUT / 'preparation.json', receipt)
    print(json.dumps(receipt))
if __name__ == '__main__':
    main()
