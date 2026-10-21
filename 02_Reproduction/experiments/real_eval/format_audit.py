"""Post-hoc, uniform format-only sensitivity analysis; frozen score untouched."""
from pathlib import Path
import json,re,copy,csv,sys
ROOT=Path(__file__).absolute().parent;BENCH=ROOT.parent/'real_evidence_benchmark_20261011_v2'
sys.path.insert(0,str(BENCH));from score_predictions import read_rows,evaluate
def normalize(v):
 if isinstance(v,str):return re.sub(r'\s+','',v)
 if isinstance(v,list):return [normalize(a) for a in v]
 return v
def restore(raw,row):
 answers={};notes=[];decoder=json.JSONDecoder()
 for q in row['questions']:
  key=q['key'];found=list(re.finditer(r'"'+re.escape(key)+r'"\s*:\s*\{',raw))
  if len(found)!=1:continue
  start=found[0].end();nxt=re.search(r'\n\s*"[^"\n]+"\s*:\s*\{',raw[start:]);end=start+nxt.start() if nxt else len(raw)
  body=raw[start:end]
  try:
   vs=list(re.finditer(r'"value"\s*:\s*',body));es=list(re.finditer(r'"evidence_ids"\s*:\s*',body))
   if len(vs)!=1 or len(es)!=1:continue
   value,vl=decoder.raw_decode(body[vs[0].end():]);ids,il=decoder.raw_decode(body[es[0].end():])
   if not isinstance(ids,list) or not all(isinstance(a,str) for a in ids):continue
   answers[key]={'value':value,'evidence_ids':ids,'brief_reason':''};notes.append({'field':key,'value':value,'evidence_ids':ids,'operation':'read existing value and citation list; reason omitted'})
  except (ValueError,TypeError):continue
 return answers,notes
def main():
 rows=read_rows(ROOT/'public_inputs.jsonl');pub={r['question_id']:r for r in rows};keys=read_rows(BENCH/'02_最终评分/最终评分键_v2.jsonl');keys_norm=copy.deepcopy(keys);out=ROOT/'scores/format_audit';out.mkdir(exist_ok=True)
 for k in keys_norm:
  for f in k['fields']:
   if f['type'] in ['choice','text','set']:
    f['final_value']=normalize(f['final_value']);f['accepted_values']=[normalize(v) for v in f.get('accepted_values',[])]
 summaries=[];logs=[]
 for folder in sorted((ROOT/'runs').glob('*')):
  if not (folder/'complete.json').exists() or 'native' in folder.name:continue
  preds=read_rows(folder/'predictions.jsonl');pmap={p['question_id']:p for p in preds};raw=read_rows(folder/'raw.jsonl')
  for r in raw:
   qid=r['question_id'];p=pmap[qid]
   if r['parse_error']:
    answers,notes=restore(r['raw'],pub[qid]);p['answers']=answers
    logs.append({'arm':folder.name,'question_id':qid,'parse_error':r['parse_error'],'recoveries':notes})
   for f in pub[qid]['questions']:
    a=p['answers'].get(f['key'])
    if isinstance(a,dict) and 'value' in a and f['answer_type'] in ['choice','text','set']:a['value']=normalize(a['value'])
  score=evaluate(rows,keys_norm,preds);score['status']='SUPPLEMENTAL_POST_HOC_FORMAT_AUDIT';score['protocol']='FORMAT_AUDIT_PROTOCOL.md'
  (out/(folder.name+'.json')).write_text(json.dumps(score,ensure_ascii=False,indent=2),encoding='utf-8')
  summaries += [{'arm':folder.name,**g} for g in score['groups']]
 (out/'summary.json').write_text(json.dumps({'summary':summaries,'recovery_log':logs},ensure_ascii=False,indent=2),encoding='utf-8')
 print('format-audited arms',len(summaries)//3)
 for r in summaries:
  if r['split']=='test_reserved' and r['task_type']=='evidence_reasoning':print(r['arm'],r['field_value_exact'],r['field_value_and_evidence'])
if __name__=='__main__':main()
