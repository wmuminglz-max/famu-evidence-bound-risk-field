"""Paired source-family analysis; no field-level pseudo-replication."""
from pathlib import Path
import csv,json,random,statistics,itertools
ROOT=Path(__file__).absolute().parent;S=ROOT/'scores'
def readcsv(n):return list(csv.DictReader((S/n).open(encoding='utf-8-sig')))
def main():
 summ=json.loads((S/'all_summary.json').read_text('utf-8'));qr=readcsv('逐题评分.csv');fr=readcsv('逐字段评分.csv')
 rows=[r for r in qr if r['split']=='test_reserved' and r['task_type']=='evidence_reasoning']
 by={a:{r['question_id']:r for r in rows if r['arm']==a} for a in {r['arm'] for r in rows}}
 comparisons=[]
 if '9b_original' in by:
  for a in sorted(by):
   if not a.startswith(('9b_','27b_')) or 'native' in a or a=='9b_original':continue
   family={}
   for q,r in by[a].items():family.setdefault(r['source_family'],[]).append(float(r['macro_accuracy'])-float(by['9b_original'][q]['macro_accuracy']))
   assert len(family)==5
   diffs=[statistics.mean(v) for v in family.values()];obs=statistics.mean(diffs)
   permutations=[statistics.mean(d*s for d,s in zip(diffs,signs)) for signs in itertools.product([-1,1],repeat=5)]
   p=sum(abs(x)>=abs(obs)-1e-12 for x in permutations)/len(permutations)
   rng=random.Random(20261011);boot=sorted(statistics.mean(rng.choices(diffs,k=5)) for _ in range(10000))
   comparisons.append({'arm':a,'reference':'9b_original','metric':'equally weighted source-family mean of parent-question accuracy','family_count':5,'delta':obs,'bootstrap95_percentile':[boot[249],boot[9749]],'paired_family_sign_flip_p':p,'family_deltas':{k:statistics.mean(v) for k,v in family.items()},'note':'Exploratory with only5families; smallest two-sided exact sign-flip p=0.0625. No significant superiority claim.'})
 seeds=[]
 for prefix in ['9b_v8_bound','9b_v8_native_rules','S0_supervised','S1_gated_KD','R11_SPAN_MIX']:
  modes=[''] if prefix.startswith('9b') else ['_raw','_rules']
  for mode in modes:
   a=[s for s in summ['summary'] if s['split']=='test_reserved' and s['task_type']=='evidence_reasoning' and s['arm'] in [prefix+'_s'+str(seed)+mode for seed in [17,29,43]]]
   if len(a)!=3:continue
   seeds.append({'family':prefix+mode,'seeds':[17,29,43],'state_mean':statistics.mean(s['state_accuracy'] for s in a),'state_min':min(s['state_accuracy'] for s in a),'state_max':max(s['state_accuracy'] for s in a),'field_mean':statistics.mean(s['field_value_exact'] for s in a),'answer_evidence_mean':statistics.mean(s['state_answer_evidence'] for s in a),'unknown_forced_TF_total':sum(s['unknown_forced_TF'] for s in a)})
 cases=[]
 for qid,field in [('B01-Q01','original_table_present'),('B04-Q04','cash_paid'),('C01-Q04','condition_met'),('C09-Q01','applicable_version'),('A05-Q01','same_cutoff'),('A07-Q03','actual_training_cost_500k')]:
  rr=[r for r in fr if r['question_id']==qid and r['field']==field and r['arm'] in ['9b_original','9b_review_sft','9b_labor_review_sft','27b_review_sft','27b_labor_review_sft','27b_learned_experts','9b_v8_native_rules_s17','S1_gated_KD_s17_rules']]
  cases.append({'question_id':qid,'field':field,'rows':rr})
 text_review=[r for r in fr if r['type']=='text' and r['provided']=='True' and r['correct']=='False']
 output={'paired_comparisons':comparisons,'seed_summary':seeds,'targeted_cases':cases,'unmatched_text_answers_for_blind_human_review':text_review,'baseline_all_U_test_state_accuracy':11/26,'state_test_n':26,'test_no_C':True}
 (S/'analysis.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
 print('analyses',len(comparisons),'seed groups',len(seeds),'unmatched text answers',len(text_review))
if __name__=='__main__':main()
