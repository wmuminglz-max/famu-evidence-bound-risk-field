"""Recompute archived scores. No model loading, network access or training."""
from pathlib import Path
import json,sys,csv,importlib.util,statistics,collections
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
result={'mode':'score_replay_from_deidentified_archived_predictions; not new model inference'}
v5=[]
for p in sorted((ROOT/'experiments/v5/results/stress').glob('*_predictions.jsonl')):
    rr=rows(p);u=[r for r in rr if r['expected']=='unknown']
    v5.append({'run':p.name.removesuffix('_predictions.jsonl'),'n':len(rr),'correct':sum(r['prediction']==r['expected'] for r in rr),'unknown_n':len(u),'unknown_correct':sum(r['prediction']=='unknown' for r in u)})
assert len(v5)==15
for prefix,expected in [('M0_direct',[53,56,56]),('M3_boundary',[172,172,176]),('M4_dual',[176,176,174])]:
    assert [r['correct'] for r in v5 if r['run'].startswith(prefix)]==expected
result['v5_stress']=v5
v8=[]
for p in sorted((ROOT/'experiments/v8/results').glob('*_s*/reports/step_0401/predictions.jsonl')):
    rr=[r for r in rows(p) if r['split']=='test'];assert len(rr)==144
    score={'run':p.parents[2].name,'n':len(rr),'correct':sum(r['prediction']==r['gold'] for r in rr)}
    if score['run'].startswith('E1'):
        pred=set();gold=set()
        for r in rr:
            for f in 'ABCD':
                for d in ['s','r']:
                    pred.update((r['id'],f,d,x) for x in r['bindings'][f][d])
                    gold.update((r['id'],f,d,x) for x in r['gold_bindings'][f][d])
        score.update({'citation_tp':len(pred&gold),'citation_fp':len(pred-gold),'citation_fn':len(gold-pred),'citation_f1':2*len(pred&gold)/(len(pred)+len(gold))})
        assert score['correct']==144 and score['citation_tp']==498 and score['citation_f1']==1
    v8.append(score)
assert len(v8)==6
assert [x['correct'] for x in v8 if x['run'].startswith('D0')]==[142,143,144]
result['v8_test']=v8
spec=importlib.util.spec_from_file_location('release_scorer',ROOT/'experiments/real_eval/score_predictions.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
public=rows(ROOT/'data/real/public_inputs.jsonl');keys=rows(ROOT/'data/real/gold_keys.jsonl')
assert all(mod.input_hash(r)==r['input_sha256'] for r in public)
real=[]
for base in ['runs','student_runs']:
    for p in sorted((ROOT/'experiments/real_eval'/base).glob('*/predictions*.jsonl')):
        report=mod.evaluate(public,keys,rows(p))
        g=next(x for x in report['groups'] if x['split']=='test_reserved' and x['task_type']=='evidence_reasoning')
        assert g['fields']==53 and g['questions']==17
        arm=p.parent.name if p.name=='predictions.jsonl' else p.parent.name+'_'+p.stem.removeprefix('predictions_')
        real.append({'arm':arm,'fields':53,'correct':round(g['field_value_exact']*53),'joint':round(g['field_value_and_evidence']*53)})
expected={'9b_original':(44,38),'9b_review_sft':(47,40),'9b_v8_bound_s17':(45,38),'9b_v8_bound_s29':(48,40),'9b_v8_bound_s43':(46,38),'27b_review_sft':(50,47),'27b_learned_experts':(50,47),'27b_labor_review_sft':(51,46)}
by={r['arm']:r for r in real}
for arm,values in expected.items():assert (by[arm]['correct'],by[arm]['joint'])==values,(arm,by.get(arm))
result['real_main53']=real
student=[]
for p in sorted((ROOT/'experiments/student/results').glob('*/metrics.json')):
    r=js(p)
    if 'metrics' not in r:continue
    student.append({'run':r.get('run',p.parent.name),'datasets':{k:{'n':v.get('n'),'correct':v.get('correct')} for k,v in r['metrics'].items() if isinstance(v,dict) and 'n' in v}})
result['student_archived_metrics']=student
result['checks_passed']=True
out=ROOT/'validation/recomputed_results.json';out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'checks_passed':True,'v5_runs':len(v5),'v8_runs':len(v8),'real_configurations':len(real),'student_configurations':len(student),'output':out.relative_to(ROOT).as_posix()},ensure_ascii=False))
