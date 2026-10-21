"""Freeze R4's 2x2 design and new fictional wording holdouts before any fit."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import prepare_student_round3_20261010 as r3
from src.evidence_reasoning.local_field_student import digest, visible_input, validate
OUT=ROOT/'outputs/famu_student_round4_20261010'
R3=ROOT/'outputs/famu_student_round3_20261010'

# Two newly authored wording banks, development and test only. Their facts and
# labels are fictional program supervision. Distractor structure is shared.
QUESTION={
'A':['{a}本事项{e}应在{d}之前完成，是不是已由双方有效签字的完整文件所确定', '双方生效签字材料中，有没有就本事项约定{a}须于{d}前完成{e}'],
'B':['{a}在本事项完成{e}之时，是否已经越过{d}这条时间界线', '本事项{a}{e}的真实结束日是否排在{d}之后'],
'C':['对方是否已经有效放弃对{a}本事项这次迟延作出处理', '本事项对方给予{a}的本次逾期豁免是否已确定生效'],
'D':['{a}本事项本期已到期的款项中，是不是还有未收妥的钱', '截至本期对账，{a}本事项是否仍有已到期债款未清']}
TEXT={
'A':{
'T':['{m}的签字正本连同附件已悉数验真，双方把{a}{e}的最后一天写定为{d}，该约定正在生效。','从{m}已生效的全部签署文书可见，双方为{a}{e}设下{d}这一期限，没有后续修改。'],
'F':['{m}的签字正本和每份附件均已验真且无遗漏。其中没有给{a}{e}写定最后一天；留空是真实原貌，亦无其他期限协议。','{m}所有有效签署文书齐备、清晰且核对结束，双方并未为{a}{e}订下任何日期要求，未另立时间条款。'],
'U':['{m}目前拿到的只是部分复印页，可能记录{a}{e}期限的部分未到，不能知道签字正本是否写有日期。','送审材料没有{m}的完整生效签署文书，涉及{a}{e}日期的字迹不可辨，其余原件也未取得。']},
'B':{
'T':['{m}的交接双方核对实物后签字证实：{a}直到{late}才完成全部{e}，此前仍未办完，所记并非预计日。','按{m}经双方查证的最终履行实录，{a}完成全部{e}的确切日期是{late}，该日之前没有完成。'],
'F':['{m}全部交接单与实物核对完毕，{a}早在{early}已经完成全部{e}，并无剩余待交部分。','{m}经核实的履行实录说明{a}于{early}已结束全部{e}，后来没有任何待履行余项。'],
'U':['{m}表中写着{a}打算{late}{e}，但这仅是日程意向，真正的交接单没有取得，实际结束日查不出。','{m}能够看到的只是{a}{e}的拟定日程，尚缺实际交接的记录，无法断定何时办结。']},
'C':{
'T':['{m}对方已向{a}送达生效的书面处理结果，确定此次迟延不予追究，已经获准且不需再满足任何条件。','针对{m}这一回逾期，书面批准已经生效，{a}无条件免受本次迟延处理，所有手续已完成。'],
'F':['{m}对方已向{a}送达生效的最终驳回书，拒绝免除此次迟延处理，相关文件全部核齐且没有相反批准。','{m}这一回逾期的最终有效批复已否决{a}的豁免请求，相关文书齐全，没有其他免责许可。'],
'U':['{m}仅收到{a}希望免于本次迟延处理的请示，决定者还没有作答，批准与否均无依据。','关于{m}此次逾期，{a}正在等待豁免批复；现有文书只有请求，没有生效的同意或拒绝。']},
'D':{
'T':['核实{m}本期到期账目后，{a}须支付{total}元，却只收妥{paid}元；仍差{owed}元，双方已确认此差额未付。','{m}本期到期应付{total}元与实收{paid}元已经逐笔核对，{a}尚有{owed}元没有交付，双方确认无误。'],
'F':['{m}本期所有到期账目均已核实，{a}须付的{total}元已全额收妥，未收余额是0元，不存在其他未清项。','逐笔核清{m}本期全部到期款项后，{a}应付的{total}元已如数收到，欠付余额为零，账款已清。'],
'U':['{m}本期{a}表中所列{total}元只是预算，真实到期账目与实收均未查得，差额是否存在无从核实。','{m}本期缺少{a}到期应付与银行实收的依据，仅见{total}元预估支出，是否欠付尚不清楚。']}}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')

def new_scenes():
    oldq,oldt=r3.QUERY,r3.TEXT
    try:
        r3.QUERY={k:oldq[k]+QUESTION[k] for k in oldq}
        r3.TEXT={k:{s:oldt[k][s]+TEXT[k][s] for s in 'TFU'} for k in oldt}
        rows=r3.generate('dev',6,[6])+r3.generate('test',8,[7])
    finally:r3.QUERY,r3.TEXT=oldq,oldt
    for r in rows:
        for k in ['id','family','pair_group']:r[k]=r[k].replace('R3-','R4-')
        # Make named families disjoint as visible instances as well as IDs.
        for old,new in [('峦序','澄谷'),('汀序','岚川')]:
            r=json_replace_in_place(r,old,new)
        validate(visible_input(r))
    return rows

def json_replace_in_place(r,old,new):
    value=json.loads(json.dumps(r,ensure_ascii=False).replace(old,new));r.clear();r.update(value);return r

def main():
    if OUT.exists():raise RuntimeError('R4 directory exists; preserve frozen inputs and inspect before proceeding')
    completion=json.loads((R3/'completion_receipt.json').read_text('utf-8'))
    assert completion['all_45_prediction_sets_verified'] and completion['cpu_comparisons_passed']
    package=OUT/'package';package.mkdir(parents=True)
    old_manifest=json.loads((R3/'package/input_manifest.json').read_text('utf-8'))
    for p in (R3/'package/src').rglob('*.py'):
        rel=p.relative_to(R3/'package');assert sha(p)==old_manifest[rel.as_posix()]['sha256']
        dst=package/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
    for rel in ['src/evidence_reasoning/factorized_field_features.py','src/evidence_reasoning/factorized_field_student.py',
                'scripts/student_round4_model.py','scripts/run_student_round4_20261010.py','scripts/prepare_student_round4_20261010.py']:
        dst=package/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dst)
    for name in ['run_gpu_student_20261009.py','run_student_round2_20261010.py','run_student_round3_20261010.py']:
        p=R3/'package/scripts'/name;assert sha(p)==old_manifest['scripts/'+name]['sha256'];shutil.copyfile(p,package/'scripts'/name)
    fit=json.loads((R3/'package/data/fit.json').read_text('utf-8'))
    recombined=ROOT/'outputs/famu_student_next_representation_20261010/training_recombinations.json'
    assert sha(recombined)=='b6f2b6afbbef1db39da034e433ece74266ecce8c826a670d2e5841e49f275d9b'
    fit['recombined']=json.loads(recombined.read_text('utf-8'))
    rows=new_scenes();assert len(rows)==168
    train_hashes={digest({k:p[k] for k in ['query','text','scope']}) for pool in fit.values() for p in pool}
    assert not train_hashes&{p['id'] for p in r3.pairs(rows)}
    train_groups={p['group'] for pool in fit.values() for p in pool}
    assert not train_groups&{r['family'] for r in rows}
    assert len({r['id'] for r in rows})==168
    write(package/'data/fit.json',fit);write(package/'data/new_scenes.json',rows)
    for name in ['historical.json','r2_regression.json']:shutil.copyfile(R3/'package/data'/name,package/'data'/name)
    r3rows=json.loads((R3/'package/data/new_scenes.json').read_text('utf-8'))
    write(package/'data/r3_regression.json',[r for r in r3rows if r['split']=='test'])
    for name in ['student.json','manifest.json']:
        p=package/'baseline_stdlib'/name;p.parent.mkdir(exist_ok=True);shutil.copyfile(ROOT/'config/local_field_student'/name,p)
    protocol={'id':'famu_student_round4_20261010','arms':['J_ORIG','F_ORIG','J_RECOMB','F_RECOMB'],'seeds':[17,29,43],
        'authorization':'User continued strengthening; one finite R4; no production change',
        'epochs':12,'steps_per_epoch':80,'updates_per_run':960,'max_total_updates':11520,'max_total_exposures':368640,'wall_hours_max':2,
        'batch_size':32,'batch_counts':[16,8,8],'original_source_per_batch':16,'new_source_per_batch':16,
        'sampling':'Class/source/family balanced with replacement. Same pair pool and seed use identical sample orders across representations. Different pair pools not identical examples.',
        'J':'Unchanged R3 sparse whole-question joint CRC features, 24579 parameters, zero initialization, lr .05',
        'F':'Separate 4096x16 question and evidence bags; q/e/product/absolute difference/11 observables; 75->32 ReLU->3, 133603 parameters, lr .01',
        'optimizer':'AdamW wd .01, clip1, 10% warmup linear decay; no KD; FP32; TF32 off',
        'representation_confound':'F changes architecture, capacity and observable-feature encoding together. Attribute findings to this representation package, not solely to removal of CRC.',
        'checkpoint_epochs':[6,12],'loss':'fictional hard-label relation CE','teacher_calls':0,
        'selection':'Fixed epoch12. Threshold .5/.6/.7/.8/.9 on historical dev plus R4 new dev: fewer unknown strong, larger macroF1, lower threshold. Freeze before test output.',
        'new_dev_count':72,'new_test_count':96,'new_styles':[6,7],
        'train_pool_sizes':{k:len(v) for k,v in fit.items()},'independent_new_training_facts':0,
        'new_holdout':'Program-authored fictional wording and named families disjoint from fit; shared generator, field semantics and some distractor templates; not independent expert legal gold.',
        'old_tests':'Historical144, R2 challenge72, R3 test72 are seen regression only. No fitting on them.',
        'quality_signal':'On R4 new test, F_RECOMB must improve correct and citation_F1 over J_ORIG for all three paired seeds, without increasing either unknown-strong count or lowering historical correct. Report all arms regardless. Not a production gate.',
        'release_limit':'Formal77/LAB9 and actual4core8GiB concurrent website acceptance outstanding. No deployment.',
        'production_promotion':False}
    write(package/'protocol.json',protocol)
    manifest={p.relative_to(package).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in package.rglob('*') if p.is_file()}
    write(package/'input_manifest.json',manifest)
    with zipfile.ZipFile(OUT/'inputs.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(package.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(package).as_posix())
    receipt={'input_zip_sha256':sha(OUT/'inputs.zip'),'input_zip_bytes':(OUT/'inputs.zip').stat().st_size,
        'protocol_sha256':sha(package/'protocol.json'),'new_scenes_sha256':sha(package/'data/new_scenes.json'),
        'files':len(manifest),'train_pool_sizes':protocol['train_pool_sizes'],'new_dev':72,'new_test':96,'training_started':False}
    write(OUT/'preparation.json',receipt);print(json.dumps(receipt))

if __name__=='__main__':main()
