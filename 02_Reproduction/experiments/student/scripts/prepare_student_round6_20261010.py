"""Freeze language normalization x routing, with new fictional wording holdouts."""
import hashlib,json,shutil,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import prepare_student_round3_20261010 as r3
from scripts.prepare_student_round4_20261010 import QUESTION as Q4,TEXT as T4
from scripts.prepare_student_round5_20261010 import QUESTION as Q5,TEXT as T5
from src.evidence_reasoning.local_field_student import validate,visible_input,digest
OUT=ROOT/'outputs/famu_student_round6_20261010';PREV=ROOT/'outputs/famu_student_round5_20261010'
QUESTION={
 'A':['本次已签署且有效的文件是否规定{a}{e}必须不迟于{d}', '关于{a}{e}，本事项完整签约原件是否载有{d}这个履行期限'],
 'B':['{a}本事项真正办结{e}时是否已在{d}之后', '本次履行的已核实结果，是否表明{a}{e}超过{d}才结束'],
 'C':['本次迟延是否已经得到对方有效同意，让{a}不用承担该次处理', '本事项这一次逾期，对方是否已生效批准{a}免于追究'],
 'D':['本期核算到期款项后，{a}本事项是否还有尚未结清的欠款', '就本事项当前结算而言，{a}是否还欠对方已经到期而未支付的钱']}
TEXT={
 'A':{'T':['{m}签订原件及附页都已经核实，双方有效条款规定{a}完成{e}不得超过{d}，没有修改。','{m}原合同和所有补充签约文书均经核实且完整有效，对{a}{e}列出的履行期限是{d}。'],
 'F':['{m}全部有效原件已查齐，{a}{e}的时间位置确实没有填写，其他有效文件也未设日期，原页没有缺少或遮挡。','{m}签约全文及有效补充材料俱全且核验清楚，{a}{e}的期限原本未定，双方没有作出任何日期约定。'],
 'U':['{m}原件时间位置的扫描内容缺失，{a}{e}有没有最后日期尚未核明，不能将看不到日期当成原本未写。','{m}所供复印件隐去了{a}{e}时间一栏，又没有其余原件，无法分辨是未约定还是资料没有展示。']},
 'B':{'T':['{m}所有实际交接签收材料一致：{a}最终完成{e}是在{late}，此前尚未完成，该日不是预计时间。','{m}经核实的履行结果显示，到{late}{a}才办结全部{e}，这记载的是完成实绩而非未来安排。'],
 'F':['{m}实际履行资料已核验，{a}早于截止日、在{early}即把全部{e}完成，剩余工作为零。','{m}全部实际签收凭据确认{a}{e}在{early}已经彻底结束，此后没有任何未完事项。'],
 'U':['{m}有关{a}{e}的{late}只是一项预约日期，实绩凭据尚未取得，因此完成时间无法确定。','{m}材料仅说{a}拟于{late}{e}；这项计划未被实际交接证明印证，是否完成与完成日期都未核实。']},
 'C':{'T':['{m}本次逾期的有效签署决定已经准许{a}免受处理，无须再满足条件，手续已经全部完成。','{m}这一回迟延已有最终生效的书面许可，对方批准{a}不受此次处理，并且没有待履行条件。'],
 'F':['{m}这一次逾期的最终决定已有效作出：拒绝对{a}予以豁免，全部审批文件核实完毕，没有另一份批准。','{m}本轮迟延的有效书面定案明确驳回{a}免责请求，所有相关文书已检查完整，不存在同意免责的决定。'],
 'U':['{m}本次逾期虽然有{a}的申请，但尚未获得生效答复；既没有批准也没有拒绝，不能当作已被驳回。','{m}这次迟延是否让{a}免受处理仍在审议，目前没有同意或否决的生效决定，缺少批准不等于最终拒绝。']},
 'D':{'T':['{m}本期已核实对账，{a}到期应付{total}元，实收只有{paid}元，仍有{owed}元的欠款未结清。','{m}当前全部到期账目已经核真，{a}应交{total}元但只付{paid}元，差额{owed}元依然尚未支付。'],
 'F':['{m}本期应付与实收已全部核对，{a}应付{total}元已经全数收到，欠款为0元，账已结清。','{m}全部当期到期债款已核实，{a}的应付{total}元已清偿完毕，未付部分为零，无其他到期欠项。'],
 'U':['{m}本期仅有{a}估计{total}元的资金安排，实际应付与实收凭据未收齐，无法断言已结清或尚欠。','{m}本期的{total}元只是{a}的预算资料，未拿到到期账款和银行实收记录，不能由记录缺少推定存在欠付。']}}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
def new_scenes():
    q,t=r3.QUERY,r3.TEXT
    try:
        r3.QUERY={k:q[k]+Q4[k]+Q5[k]+QUESTION[k] for k in q}
        r3.TEXT={k:{s:t[k][s]+T4[k][s]+T5[k][s]+TEXT[k][s] for s in 'TFU'} for k in t}
        rows=r3.generate('dev',6,[10])+r3.generate('test',8,[11])
    finally:r3.QUERY,r3.TEXT=q,t
    for r in rows:
        for k in ['id','family','pair_group']:r[k]=r[k].replace('R3-','R6-')
        s=json.dumps(r,ensure_ascii=False).replace('峦序','远岑').replace('汀序','晴汐');r.clear();r.update(json.loads(s));validate(visible_input(r))
    return rows
def main():
    if (OUT/'package').exists():raise RuntimeError('Existing R6 preserved')
    assert json.loads((OUT/'prior_diagnostic.json').read_text('utf-8'))['optimizer_updates']==0
    package=OUT/'package';package.mkdir()
    manifest=json.loads((PREV/'package/input_manifest.json').read_text('utf-8'))
    for rel,e in manifest.items():
        if rel.startswith(('src/','scripts/')):
            src=PREV/'package'/rel;assert sha(src)==e['sha256'];dst=package/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    for rel in ['src/evidence_reasoning/language_field_student.py','scripts/student_round6_model.py','scripts/prepare_student_round6_20261010.py','scripts/run_student_round6_20261010.py']:
        shutil.copyfile(ROOT/rel,package/rel)
    (package/'data').mkdir()
    for name in ['fit','historical','r2_regression','r3_regression','r4_regression']:shutil.copyfile(PREV/f'package/data/{name}.json',package/f'data/{name}.json')
    write(package/'data/r5_regression.json',[r for r in json.loads((PREV/'package/data/new_scenes.json').read_text('utf-8')) if r['split']=='test'])
    new=new_scenes();write(package/'data/new_scenes.json',new)
    fit=json.loads((package/'data/fit.json').read_text('utf-8'));assert not {p['group'] for pool in fit.values() for p in pool}&{r['family'] for r in new}
    assert not {digest({k:p[k] for k in ['query','text','scope']}) for pool in fit.values() for p in pool}&{p['id'] for p in r3.pairs(new)}
    for seed in [17,29,43]:
        src=PREV/f'result_bundle/outputs/T_MLP_s{seed}';dst=package/f'baseline_s{seed}';dst.mkdir()
        for p in (src/'model').iterdir():shutil.copyfile(p,dst/p.name)
        for name in ['initialization.json','training_log.jsonl']:shutil.copyfile(src/name,dst/name)
        m=json.loads((dst/'manifest.json').read_text('utf-8'))
        for n,h in m.items():assert sha(dst/n)==h
    protocol={'id':'famu_student_round6_20261010','seeds':[17,29,43],'training_arms':['LANG'],'evaluation_arms':['RAW_SOFT','RAW_HARD','LANG_SOFT','LANG_HARD'],
      'authorization':'User continued strengthening and requested language optimization. Shadow only.',
      'max_total_updates':2880,'max_total_exposures':92160,'auxiliary_query_labels':92160,'updates_per_run':960,'epochs':12,'steps_per_epoch':80,'batch_size':32,'wall_hours_max':2,
      'training':'Exactly R5 original2500+recombined16704, same seeds/initial weights/order/objective/AdamW. Only lexical normalization changes. No new training facts or teacher calls.',
      'optimizer':'FP32 TF32off, AdamW .01 wd .01, warmup96 linear960, clip1; epoch6/12 complete checkpoints.',
      'baseline':'Reuse all3 frozen R5 T_MLP weights (no repeat training); assert matching initialization and all960 batch-order hashes for normalized branch.',
      'language':'Longest one-pass lexical synonym normalization on anonymized bags; original numeric and scope features and all raw quote/unknown/rule paths unchanged. Does not claim semantic equivalence for arbitrary Chinese.',
      'routing':'soft mixture or hard predicted query argmax. No gold type or field ID input; gate below .0/.4/.6 returns uniform relations for review.',
      'selection':'For each of4 modes x3seed, only historical dev+R6 newdev select relation threshold .5/.6/.7/.8/.9 and query gate0/.4/.6: minimize unknown strong, maximize macroF1, citationF1, lower gate, lower relation threshold. Lock all modes before tests. Also report frozen R5 original thresholds separately.',
      'new_dev':72,'new_test':96,'holdout':'Styles10/11, disjoint named families; shared generator/templates/distractors, fictional not expert gold. Authored after language module without further feature edits based on new test.',
      'research_signal':'Each language mode vs matched RAW routing: all3 seeds improve newtest correct and citationF1, fieldF recall not lower, unknown strong not higher; correct and citationF1 nondecrease on every historical/R2/R3/R4/R5 seen regression. All seeds reported; no production.',
      'release':'Formal77 LAB9 actual website4core8GiB concurrency not proven. No promotion. Stop this finite stage; no automatic repeated training.'}
    write(package/'protocol.json',protocol)
    entries={p.relative_to(package).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in package.rglob('*') if p.is_file()};write(package/'input_manifest.json',entries)
    with zipfile.ZipFile(OUT/'inputs.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(package.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(package).as_posix())
    receipt={'input_zip_sha256':sha(OUT/'inputs.zip'),'protocol_sha256':sha(package/'protocol.json'),'new_scenes_sha256':sha(package/'data/new_scenes.json'),'input_zip_bytes':(OUT/'inputs.zip').stat().st_size,'files':len(entries)}
    write(OUT/'preparation.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':main()
