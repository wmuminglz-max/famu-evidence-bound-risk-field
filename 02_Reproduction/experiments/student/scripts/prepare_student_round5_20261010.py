"""Freeze typed-supervision R5, with only training-derived type labels."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import prepare_student_round3_20261010 as r3
from scripts.prepare_student_round4_20261010 import QUESTION as R4Q,TEXT as R4T
from src.evidence_reasoning.local_field_student import digest,visible_input,validate
OUT=ROOT/'outputs/famu_student_round5_20261010';R4=ROOT/'outputs/famu_student_round4_20261010'
QUESTION={
'A':['本事项双方签订的完整有效文本，有没有要求{a}到{d}为止办完{e}', '全部生效签署原件是否足以证明，本事项{a}{e}的约定期限为{d}'],
'B':['本事项{a}真正完成{e}的那天，是不是已经迟过{d}', '本事项{a}实际办完{e}时是否已晚于指定的{d}'],
'C':['关于本事项这一次逾期，{a}是否已经拿到生效的不予追究许可', '本事项此次迟延是否已有有效决定准予{a}免受处理'],
'D':['{a}本事项本期到期应付款中，是否尚有没付清的余款', '本事项本期应付款已到期，{a}是否还有余额未向对方支付']}
TEXT={
'A':{'T':['{m}全部原始签署资料都已核验。生效正文写明{a}应最晚在{d}办完{e}，双方签字和附件完整且未修改。','{m}原始文书经逐份核真并确认齐备，其中正在生效的签约条文要求{a}在{d}前{e}，双方均已签认。'],
'F':['{m}全部原始签署资料及补充文件均在场，均已核验且文字清晰。双方从未对{a}{e}订明最迟时间，日期处本来就是空白，并非缺失。','{m}原始文书及全部有效附件经核真齐备，{a}{e}没有被约定任何具体日期，所有期限处原本未填，亦无另订期限。'],
'U':['{m}交来的原件少了涉及{a}{e}日期的部分，相关补充文件也未取得，不能判断双方有没有订明最迟时间。','{m}只有摘录，载有{a}{e}期限的签署原页未展示且无法核对，不能确认完整合同如何写定日期。']},
'B':{'T':['{m}真实交接结果已由双方查验签名：{a}到{late}才将{e}全部结束，此前没有办完，这不是日程预告。','{m}已核真的实际交接清单表明{a}在{late}才完成所有{e}，之前一直未全部结束，记录并非计划。'],
'F':['{m}真实交接结果已查验并签名，{a}在{early}就已全部办完{e}，此后没有遗留待办部分。','核实{m}全部履行回执后确认，{a}在{early}已把{e}全数办完，再无剩余事项。'],
'U':['{m}提到{a}准备在{late}{e}，但文件仅是预告，实际交接结果未提交，无法查出真正结束日。','{m}材料把{late}列为{a}{e}的目标日期，却未附真实交接回执，实际何时结束尚无凭据。']},
'C':{'T':['{m}此次迟延的审批手续已经办妥，对方正式签字许可{a}此次不受处理，决定已生效且不附条件。','在{m}此轮逾期的正式批文中，对方已经无条件准许{a}免受处理，批文已有效送达，不待其他条件成就。'],
'F':['{m}此次迟延的审批手续已经结束，生效的最终结果不允许{a}免责，所有相关文件已核对，没有其他同意意见。','在{m}此轮逾期的最终有效批文中，对方明确不予{a}豁免，全部批准记录均查清且无相反许可。'],
'U':['{m}此次迟延尚在等待审批，{a}虽已提出免于处理的愿望，但对方既未同意也未拒绝。','对于{m}此轮逾期，{a}仅递交了豁免请求，对方尚未作出具有约束力的回复，是否获准不得而知。']},
'D':{'T':['{m}本期应收账与实收账已核准，{a}到期应付{total}元、实际收款{paid}元，余款{owed}元还没有付清。','{m}本期全部到期款已对清，{a}需付{total}元而已收{paid}元，双方核定仍欠{owed}元未交。'],
'F':['{m}本期应收账与实收账逐项相符，{a}到期应付{total}元已经全收，剩余欠付0元，所有应付款均清偿完毕。','{m}本期所有到期款核对结果一致：{a}应付的{total}元已足额实收，双方确认未付余额为零且没有其他欠付。'],
'U':['{m}本期{a}仅列出了{total}元预算，已到期应付账与实收账都还没取得，不能核准是否还差钱。','{m}本期可见的{total}元只是{a}预先估算，实际到期应付及收款均无资料，是否留有欠付无法核对。']}}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
def new_scenes():
    q,t=r3.QUERY,r3.TEXT
    try:
        r3.QUERY={k:q[k]+R4Q[k]+QUESTION[k] for k in q}
        r3.TEXT={k:{s:t[k][s]+R4T[k][s]+TEXT[k][s] for s in 'TFU'} for k in t}
        rows=r3.generate('dev',6,[8])+r3.generate('test',8,[9])
    finally:r3.QUERY,r3.TEXT=q,t
    for r in rows:
        for k in ['id','family','pair_group']:r[k]=r[k].replace('R3-','R5-')
        s=json.dumps(r,ensure_ascii=False).replace('峦序','秋涧').replace('汀序','霁原');r.clear();r.update(json.loads(s));validate(visible_input(r))
    return rows

def add_training_types(fit,train):
    lookup={}
    for r in train:
        if r['split']!='train':raise ValueError('Only training scenes may provide type supervision')
        for i,(f,q) in enumerate(r['fields'].items()):
            assert f=='ABCD'[i]
            key=r['family'],q
            if key in lookup:assert lookup[key]==i
            lookup[key]=i
    for pool in fit.values():
        for p in pool:p['query_kind']=lookup[p['group'],p['query']]
    return fit

def main():
    if (OUT/'package').exists():raise RuntimeError('Existing R5 inputs preserved')
    assert json.loads((OUT/'prior_diagnostic.json').read_text())['optimizer_updates']==0
    package=OUT/'package';package.mkdir(parents=True)
    manifest=json.loads((R4/'package/input_manifest.json').read_text('utf-8'))
    for rel,e in manifest.items():
        if rel.startswith('src/') or rel.startswith('scripts/'):
            p=R4/'package'/rel;assert sha(p)==e['sha256'];dst=package/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
    for rel in ['src/evidence_reasoning/typed_field_student.py','scripts/student_round5_model.py','scripts/run_student_round5_20261010.py','scripts/prepare_student_round5_20261010.py']:
        dst=package/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dst)
    allfit=json.loads((R4/'package/data/fit.json').read_text('utf-8'));fit={k:allfit[k] for k in ['original','recombined']}
    old=json.loads((R4/'package/data/historical.json').read_text('utf-8'))
    aug=json.loads((ROOT/'outputs/famu_student_round3_20261010/package/data/new_scenes.json').read_text('utf-8'))
    fit=add_training_types(fit,[r for r in old+aug if r['split']=='train']);write(package/'data/fit.json',fit)
    rows=new_scenes();assert len(rows)==168
    assert not {p['group'] for pool in fit.values() for p in pool}&{r['family'] for r in rows}
    assert not {digest({k:p[k] for k in ['query','text','scope']}) for pool in fit.values() for p in pool}&{p['id'] for p in r3.pairs(rows)}
    write(package/'data/new_scenes.json',rows)
    for name in ['historical.json','r2_regression.json','r3_regression.json']:shutil.copyfile(R4/'package/data'/name,package/'data'/name)
    write(package/'data/r4_regression.json',[r for r in json.loads((R4/'package/data/new_scenes.json').read_text('utf-8')) if r['split']=='test'])
    shutil.copytree(R4/'package/baseline_stdlib',package/'baseline_stdlib')
    for seed in [17,29,43]:
        source=R4/f'result_bundle/outputs/F_RECOMB_s{seed}/model';m=json.loads((source/'manifest.json').read_text())
        for n,h in m.items():assert sha(source/n)==h
        shutil.copytree(source,package/f'baseline_F_s{seed}')
    protocol={'id':'famu_student_round5_20261010','arms':['T_LINEAR','T_MLP'],'seeds':[17,29,43],
        'authorization':'User continues strengthening after R4; new finite typed decomposition, no deployment',
        'wall_hours_max':2,'epochs':12,'steps_per_epoch':80,'updates_per_run':960,'max_total_updates':5760,'max_total_exposures':184320,
        'auxiliary_question_label_exposures':184320,'batch_size':32,'class_counts':[16,8,8],'source_counts':[16,16],
        'sampling':'Same R4 RECOMB pools, source/class/family balancing. Same seed across two new arms has the same pair order.',
        'loss':'CE(question 4-type logits, training-derived type) + CE(evidence 3-way logits for gold training type, relation)',
        'runtime':'No field ID or supplied type as model input. Predict 4 query types from text; independent evidence 4x3 relations, mixture by predicted type distribution; same existing quote/unknown/rule path.',
        'T_LINEAR':'Zero initialized linear query4 and evidence4x3 heads. Evidence uses bag4096+10 observable scope/date/amount features; no query overlap. AdamW .05.',
        'T_MLP':'Same zero initialized query4 head; evidence4106->32 ReLU->12 initialized by seed. AdamW .01.',
        'optimizer':'AdamW wd .01, clip1, 10% warmup linear decay; FP32, TF32 off; fixed960 updates.',
        'checkpoint_epochs':[6,12],'teacher_calls':0,'new_independent_training_facts':0,'train_pool_sizes':{k:len(v) for k,v in fit.items()},
        'new_supervision':'Question type derived only from existing train scene declarations; not expert annotation. Additional supervision and architecture differ from R4, not a pure architecture effect.',
        'selection':'Fixed endpoint12; threshold .5/.6/.7/.8/.9 on historical+R5 new dev: minimize two unknown strong counts, maximize macroF1, lower threshold. Test never used.',
        'new_dev':72,'new_test':96,'holdout':'New simulated wording styles8/9, distinct named families. Shared template generator/structure and some distractors; not expert legal gold.',
        'baselines':'Frozen R4 F_RECOMB seeds17/29/43 with original thresholds, zero updates; frozen old stdlib field student. Not complete website baseline.',
        'old_tests':'historical144/R2/R3/R4 are seen regression only, none fitted.',
        'research_signal':'Per arm, all3 seeds must improve both R5 new_test correct and citationF1 over frozen same-seed R4, unknown strong not increase and historical correct not decrease. No best-seed selection.',
        'release':'Only shadow; formal77/LAB9/website4core8GiB concurrency remain unproved. No production promotion.',
        'retry_policy':'No automatic training restart, no extending2h, no endless extra runs if signal fails.'}
    write(package/'protocol.json',protocol)
    entries={p.relative_to(package).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in package.rglob('*') if p.is_file()}
    write(package/'input_manifest.json',entries)
    with zipfile.ZipFile(OUT/'inputs.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(package.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(package).as_posix())
    receipt={'input_zip_sha256':sha(OUT/'inputs.zip'),'protocol_sha256':sha(package/'protocol.json'),'new_scenes_sha256':sha(package/'data/new_scenes.json'),
        'input_zip_bytes':(OUT/'inputs.zip').stat().st_size,'files':len(entries),'train_pool_sizes':protocol['train_pool_sizes'],'new_dev':72,'new_test':96}
    write(OUT/'preparation.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':main()
