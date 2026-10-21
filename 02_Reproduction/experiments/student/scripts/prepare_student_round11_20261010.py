"""Freeze exactly one R11 2x2 experiment, preserving completed rounds."""
import json,hashlib,shutil,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.student_round11_data import generate,replay
from src.evidence_reasoning.continuous_span_student import ARMS
OUT=ROOT/'outputs/famu_student_round11_20261010';PREV=ROOT/'outputs/famu_student_round10_20261010'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
def main():
    if (OUT/'package').exists():raise RuntimeError('Existing R11 preserved')
    package=OUT/'package';package.mkdir(parents=True)
    for n,e in json.loads((PREV/'package/input_manifest.json').read_text('utf-8')).items():
        if n.startswith(('src/','scripts/','encoder/','head_source_','data/')):
            src=PREV/'package'/n;assert sha(src)==e['sha256'];dest=package/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
    # Preserve the frozen original R10, but use its audited runtime anchor-schema repair here.
    for rel in ['src/evidence_reasoning/clause_language_student.py','src/evidence_reasoning/continuous_span_student.py','scripts/student_round11_data.py','scripts/prepare_student_round11_20261010.py','scripts/run_student_round11_20261010.py']:
        shutil.copyfile(ROOT/rel,package/rel)
    for seed in [17,29,43]:
        source=PREV/f'result_bundle/outputs/SELECT_s{seed}/model';manifest=json.loads((source/'manifest.json').read_text('utf-8'));dest=package/f'r10_source_s{seed}';dest.mkdir()
        for n,h in manifest.items():assert sha(source/n)==h;shutil.copyfile(source/n,dest/n)
        shutil.copyfile(source/'manifest.json',dest/'manifest.json')
    previous=json.loads((PREV/'package/data/new_scenes.json').read_text('utf-8'))
    for suffix,split in [('', 'test'),('_short','test_short'),('_reordered','test_reordered')]:write(package/f'data/r10_regression{suffix}.json',[r for r in previous if r['split']==split])
    rows,pairs,paragraphs=generate()
    original=json.loads((PREV/'package/data/fit.json').read_text('utf-8'))['original'];old=json.loads((PREV/'package/data/selector_paragraphs.json').read_text('utf-8'));clean=replay(original)
    bank=[p|{'source':n} for n,pp in [('new',paragraphs),('r10',old),('historical',clean)] for p in pp]
    write(package/'data/new_scenes.json',rows);write(package/'data/paragraphs.json',bank)
    protocol={'id':'famu_student_round11_20261010','seeds':[17,29,43],'arms':ARMS,'updates_per_run':960,'max_total_updates':11520,'batch_size':32,'max_paragraph_exposures':368640,'relation_anchor_exposures':0,'wall_hours_max':2,'checkpoint_steps':[480,960],
      'comparison':'2x2 TOKEN versus SPAN and NEW versus MIX; all3seeds, matched initial original locked encoder/rank8alpha16LoRA24modules/five-outputhead. Same paragraph batches within data arm. Additional endpoint supervision and numeric-boundary decoder are part of SPAN representation intervention, not a single-loss-variable claim.',
      'TOKEN':'First3 logits per token inactive/current-known/current-uncertain CE weights .5/1/1; prefix/special/pad ignored. Independent tokens can yield fragmented source spans; remaining2head outputs unused.',
      'SPAN':'First3 body-token logits mean -> paragraph classCE. Last2 start/endCE on nonempty authored source interval; numeric-atom and repeated-normalization boundary masks apply in train and inference. Joint start<=end argmax, exactly one contiguous original substring or none. Dates/amounts cannot be cut inside matched numeric atoms, but wrong/omitted dates remain possible and must be measured.',
      'sampling':'NEW per32:12inactive12known8uncertain new paragraphs. MIX:14new(6/4/4)+14R10training(6/4/4)+4historicaltrainingknownpositives(0/4/0). Family-uniform per source/class. Only historical label1/2 evidence whole paragraphs eligible; no all-label0 invented inactive targets. Counts are paragraph exposures, not relation pairs or independent facts.',
      'optimizer':'FP32 eager no dropout/TF32; AdamW selector1e-3 adapter2e-4 wd.01 clip1,96warmup linear960; sameR8LEXhead/base frozen. No new teacher or RL; all initialization/base/order hashes retained. Longest positive/uncertain backward preflight zero updates.',
      'data':'1152train72dev96test+96same-factshort+96reordered, authored spans. New family entities/wording banks held apart, shared program generator, entirely fictional, not expert legal gold. R2-R10 tests regression only, no fit. R10 training/old training only MIX replay.',
      'selection':'Only span_gate .5,.7,.9 on historicaldev+newdev: minimize unknownboth strong errors, maximize macroF1/citation, prefer lowergate. Original correspondingR8 querygate/threshold frozen. Lock all12 before any test metrics. Baselines R8_FROZEN and R10_SELECT frozen3seeds each, zero updates and no reselection.',
      'signals':'Predefined TOKEN_NEW->SPAN_NEW,TOKEN_MIX->SPAN_MIX,TOKEN_NEW->TOKEN_MIX,SPAN_NEW->SPAN_MIX plus descriptive R10_SELECT comparisons. All3seeds gain correct+citations on newlong/reordered, no correct/citation loss on newshort and each old regression; no fieldF recall loss or unknownstrong increase on new3. Report full span classF1/exact, numeric atom fragmentation and gold current numeric retention, all lengths and train/inference costs. No best seed or production promotion.',
      'limits':'<=512tokens rejection, no truncation; one current interval assumption fits authored data, not arbitrary multi-fact documents.12completeCPUloads/shortsceneGPUagreement/24fullCP+optimizerRNG audits. Not actualwebsite4core8GiBhardlimit/concurrency,not77formal/LAB9acceptance; no automatic extension, no old deletion.',
      'encoder_sha256':json.loads((PREV/'package/protocol.json').read_text('utf-8'))['encoder_sha256']}
    write(package/'protocol.json',protocol)
    files={p.relative_to(package).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(package.rglob('*')) if p.is_file()};write(package/'input_manifest.json',files)
    with zipfile.ZipFile(OUT/'inputs.zip','x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in sorted(package.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(package).as_posix())
    receipt={'input_zip_sha256':sha(OUT/'inputs.zip'),'input_zip_bytes':(OUT/'inputs.zip').stat().st_size,'protocol_sha256':sha(package/'protocol.json'),'input_manifest_sha256':sha(package/'input_manifest.json'),'files':len(files),'split_sizes':{s:sum(r['split']==s for r in rows) for s in sorted({r['split'] for r in rows})},'paragraph_sources':{n:sum(p['source']==n for p in bank) for n in ['new','r10','historical']}}
    write(OUT/'preparation.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':main()
