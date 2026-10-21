"""Existing students, public state questions, fixed threshold, CPU inference only.
No new fitting. Raw learned fields vs the existing uncertainty/rule gate.
"""
import json,sys,time,hashlib,gc,os,traceback
from pathlib import Path
ROOT=Path(__file__).parent
sys.path.insert(0,'/workspace/famu_student_round11_20261010')
from src.evidence_reasoning.local_field_student import LocalFieldStudent,FAMILY
from src.evidence_reasoning.continuous_span_student import ContinuousSpanStudent
import torch
from transformers import AutoTokenizer,AutoModel
from safetensors.torch import load_file
torch.set_num_threads(4)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def pair_text(q,t,s):return '作用域：'+json.dumps(s,ensure_ascii=False,sort_keys=True)+'\n字段：'+q+'\n原文：'+t
class KDStudent(LocalFieldStudent):
 def __init__(self,run):
  folder=run/'model';manifest=json.loads((folder/'manifest.json').read_text())
  for n,h in manifest.items():assert sha(folder/n)==h,n
  self.model={'threshold':json.loads((run/'selection_before_test.json').read_text())['threshold']}
  self.tok=AutoTokenizer.from_pretrained(str(folder),local_files_only=True,trust_remote_code=False)
  self.base=AutoModel.from_pretrained(str(folder),local_files_only=True,trust_remote_code=False).cpu().eval()
  self.head=torch.nn.Linear(512,3);self.head.load_state_dict(load_file(str(folder/'relation_head.safetensors')),strict=True);self.head.eval()
  self.base.requires_grad_(False);self.head.requires_grad_(False)
  self.parameters=sum(p.numel() for m in [self.base,self.head] for p in m.parameters())
  self.manifest=manifest;self.cache={}
 def predict_pair(self,q,t,s):
  text=pair_text(q,t,s)
  if text not in self.cache:
   x=self.tok(text,return_tensors='pt',truncation=False);assert x.input_ids.shape[-1]<=512
   with torch.inference_mode():self.cache[text]=self.head(self.base(**x).last_hidden_state[:,0]).float().softmax(-1)[0].tolist()
  return self.cache[text]

def windows(row,tok):
 # Full scope and question are retained; every raw character survives in ordered chunks.
 scope={'roles':row['roles'],'time_scope':row['time_scope'],'display_scope':row['scope']}
 states=[q for q in row['questions'] if q['answer_type']=='state'];out=[];mapping={}
 for ev in row['evidence']:
  text=ev['text'];start=0;i=0
  while start<len(text):
   end=min(start+180,len(text))
   if end<len(text):
    stops=[text.rfind(ch,start+40,end) for ch in '。；\n'];stop=max(stops)
    if stop>=start+40:end=stop+1
   while max(len(tok(pair_text(q['prompt'],text[start:end],scope),add_special_tokens=True)['input_ids']) for q in states)>500:
    end=start+(end-start)//2
    if end-start<8:raise ValueError('Scope alone exceeds student budget')
   i+=1;eid=ev['evidence_id']+'#'+str(i)
   out.append({'id':eid,'text':text[start:end],'visibility':'visible'})
   mapping[eid]={'original_evidence_id':ev['evidence_id'],'start':start,'end':end,'text':text[start:end]};start=end
  assert ''.join(e['text'] for e in out if mapping[e['id']]['original_evidence_id']==ev['evidence_id'])==text
 assert len(out)<=128
 return scope,out,mapping

def main():
 rows=[json.loads(s) for s in (ROOT/'public_inputs.jsonl').read_text().splitlines()]
 base=Path('/workspace/famu_student_gpu_20261009');r11=Path('/workspace/famu_student_round11_20261010')
 tok=AutoTokenizer.from_pretrained(str(r11/'encoder'),local_files_only=True)
 adapted={r['question_id']:windows(r,tok) for r in rows}
 save(ROOT/'student_input_adapter_audit.json',{'public_input_sha256':sha(ROOT/'public_inputs.jsonl'),'script_sha256':sha(__file__),'all_characters_retained':True,'target_fields':46,'scope':'46 state questions only; other 41 fields unsupported','window_policy':'180 raw characters, sentence-preferred cut, reduce if combined tokens exceed500; identical windows for all students; citations mapped to original evidence IDs','task_family_transport':FAMILY,'task_family_semantics':'explicit out-of-family real-contract transfer probe, not in-distribution qualification','rows':{qid:{'scope':s,'windows':m} for qid,(s,e,m) in adapted.items()}})
 arms=[(f'{a}_s{s}','kd',base/'outputs'/f'{a}_s{s}') for a in ['S0_supervised','S1_gated_KD'] for s in [17,29,43]]+[(f'R11_SPAN_MIX_s{s}','r11',r11/'outputs'/f'SPAN_MIX_s{s}') for s in [17,29,43]]
 save(ROOT/'STUDENT_PROTOCOL_before_run.json',{'arms':[{'id':a,'kind':k,'path':str(p)} for a,k,p in arms],'input_adapter_sha256':sha(ROOT/'student_input_adapter_audit.json'),'selection':'existing saved final checkpoints and thresholds; no selection on current dev/test','runtime':'CPU four threads','raw':'LocalFieldStudent.review fields, accepted relations, existing saved threshold','rules':'same review effective_fields; existing uncertainty relaxation plus certified_decision for identity proposition. No newly authored legal rule or gold-dependent override.','unsupported':'Non-state fields stay missing. Full-task coverage and common-state accuracy both reported.','citation_mapping':'Any predicted chunk cites parent; this preserves original evidence ID granularity but not necessarily sufficient span semantics.'})
 for name,kind,path in arms:
  start=time.monotonic();student=KDStudent(path) if kind=='kd' else ContinuousSpanStudent.load(path/'model',r11/'encoder',device='cpu')
  d=ROOT/'student_runs'/name;d.mkdir(parents=True,exist_ok=True)
  if (d/'raw.jsonl').exists():raise ValueError('No overwrite '+name)
  save(d/'loading.json',{'arm':name,'path':str(path),'load_seconds':time.monotonic()-start,'parameters':getattr(student,'parameters',None),'config':student.model,'model_manifest':json.loads((path/'model/manifest.json').read_text())})
  for ri,row in enumerate(rows):
   rawans={};ruleans={};details=[];start=time.monotonic();scope,evs,mapping=adapted[row['question_id']]
   for q in row['questions']:
    if q['answer_type']!='state':continue
    payload={'task_family':FAMILY,'scope':scope,'fields':{'field':q['prompt']},'rule':'field','evidence':evs}
    try:
     result=student.review(payload);bind=result['bindings']['field'];ids=list(dict.fromkeys(mapping[e]['original_evidence_id'] for e in bind['s']+bind['r']))
     states={'supported':'T','refuted':'F','unknown':'U','conflict':'C'}
     rawans[q['key']]={'value':states[result['fields']['field']],'evidence_ids':ids,'brief_reason':'冻结学生的逐窗支持/反驳关系聚合；详见原始轨迹。'}
     ruleans[q['key']]={'value':states[result['effective_fields']['field']],'evidence_ids':ids,'brief_reason':'同一学生输出经过既有低置信度字段放宽与规则引擎；详见原始轨迹。'}
     details.append({'field':q['key'],'result':result})
    except Exception:details.append({'field':q['key'],'error':traceback.format_exc()})
   elapsed=time.monotonic()-start
   for file,answers in [('predictions_raw.jsonl',rawans),('predictions_rules.jsonl',ruleans)]:
    with (d/file).open('a') as f:f.write(json.dumps({'question_id':row['question_id'],'input_sha256':row['input_sha256'],'answers':answers},ensure_ascii=False)+'\n')
   with (d/'raw.jsonl').open('a') as f:f.write(json.dumps({'question_id':row['question_id'],'input_sha256':row['input_sha256'],'seconds':elapsed,'details':details},ensure_ascii=False)+'\n')
   save(ROOT/'student_status.json',{'arm':name,'phase':'running','done':ri+1,'total':28,'seconds':elapsed})
   print(name,row['question_id'],round(elapsed,3),'provided',len(rawans),flush=True)
  save(d/'complete.json',{'rows':28,'raw_sha256':sha(d/'raw.jsonl')});del student;gc.collect()
 save(ROOT/'student_status.json',{'phase':'completed','arms':[a for a,k,p in arms]})
if __name__=='__main__':main()
