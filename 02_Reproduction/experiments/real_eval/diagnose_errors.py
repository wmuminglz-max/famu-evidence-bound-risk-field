"""Descriptive error changes and state confusion on held-out evidence questions."""
from pathlib import Path
import csv,json,collections
R=Path(__file__).resolve().parent
def writecsv(name,rows):
 with (R/'scores'/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 rows=[r for r in csv.DictReader((R/'scores/逐字段评分.csv').open(encoding='utf-8-sig')) if r['split']=='test_reserved' and r['task_type']=='evidence_reasoning']
 arms=sorted({r['arm'] for r in rows});base={(r['question_id'],r['field']):r for r in rows if r['arm']=='9b_original'}
 breakdown=[];transitions=[];confusion=[]
 for arm in arms:
  rr=[r for r in rows if r['arm']==arm]
  for kind in sorted({r['type'] for r in rr}):
   rs=[r for r in rr if r['type']==kind];n=len(rs);breakdown.append({'arm':arm,'answer_type':kind,'n':n,'provided':sum(r['provided']=='True' for r in rs),'correct':sum(r['correct']=='True' for r in rs),'correct_and_evidence':sum(r['correct_and_evidence']=='True' for r in rs)})
  c=collections.Counter()
  for r in rr:
   if r['type']=='state':
    v=json.loads(r['predicted']);v=v.strip() if isinstance(v,str) else v;v=v if v in ['T','F','U','C'] else 'MISSING/INVALID';c[json.loads(r['gold']),v]+=1
   if arm!='9b_original' and arm.startswith(('9b_','27b_')) and 'native' not in arm:
    old=base[r['question_id'],r['field']]
    if old['correct']!=r['correct']:transitions.append({'arm':arm,'question_id':r['question_id'],'field':r['field'],'type':r['type'],'change':'fixed' if r['correct']=='True' else 'regressed','gold':r['gold'],'base_prediction':old['predicted'],'new_prediction':r['predicted'],'base_reason':old['reason'],'new_reason':r['reason']})
  confusion.extend({'arm':arm,'gold':g,'predicted':p,'count':n} for (g,p),n in sorted(c.items()))
 writecsv('按答案类型统计.csv',breakdown);writecsv('相对原始9B逐项变化.csv',transitions);writecsv('状态混淆.csv',confusion)
 print('diagnostics',len(breakdown),len(transitions),len(confusion))
if __name__=='__main__':main()
