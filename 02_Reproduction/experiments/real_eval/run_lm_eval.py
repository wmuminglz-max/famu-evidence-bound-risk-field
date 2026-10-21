"""Frozen public-input-only inference. No gold keys are imported or uploaded."""
import argparse, json, hashlib, time, sys, os, gc, traceback, platform
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def prompt(tok,row):
 # The complete frozen public row is identical for all generative arms.
 return tok.apply_chat_template([{'role':'user','content':json.dumps(row,ensure_ascii=False)}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
def parse(raw,row):
 text=raw.strip()
 if text.startswith('```'):
  lines=text.splitlines();text='\n'.join(lines[1:-1]).strip() if lines[-1].strip()=='```' else text
 try:
  x=json.loads(text)
  if not isinstance(x,dict) or not isinstance(x.get('answers'),dict):raise ValueError('missing answers object')
  if set(x['answers'])-set(q['key'] for q in row['questions']):raise ValueError('extra fields')
  # Identity is transported by the runner; it is not a legal reasoning score.
  return {'question_id':row['question_id'],'input_sha256':row['input_sha256'],'answers':x['answers']},None
 except Exception as e:return {'question_id':row['question_id'],'input_sha256':row['input_sha256'],'answers':{}},str(e)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);a=ap.parse_args()
 cfg=json.loads(Path(a.config).read_text());root=Path(a.config).parent;sys.path.insert(0,cfg['runtime_root'])
 import torch,transformers,peft
 from src.evidence_reasoning.famu_llm import load_model,load_tokenizer,verify_base
 from src.evidence_reasoning.compact_delta import enable_compact_delta
 from src.evidence_reasoning.chunked_attention import enable_chunked_mlp
 torch.set_num_threads(4);torch.manual_seed(20261011);torch.backends.cudnn.benchmark=False
 torch.use_deterministic_algorithms(True)
 rows=[json.loads(l) for l in (root/'public_inputs.jsonl').read_text().splitlines()]
 tok=load_tokenizer(Path(cfg['base']));lock=verify_base(Path(cfg['base']))
 prompts=[prompt(tok,r) for r in rows];lens=[len(tok(p,add_special_tokens=False)['input_ids']) for p in prompts]
 assert len(rows)==28 and max(lens)+cfg['max_new_tokens']<=32768
 save(root/(cfg['host_label']+'_preflight.json'),{'input_file_sha256':sha(root/'public_inputs.jsonl'),'script_sha256':sha(__file__),'config_sha256':sha(a.config),'base_lock':lock,'prompt_tokens':dict(zip([r['question_id'] for r in rows],lens)),'prompt_sha256':{r['question_id']:hashlib.sha256(p.encode()).hexdigest() for r,p in zip(rows,prompts)},'no_truncation':True,'environment':{'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,'hostname':platform.node(),'gpu':torch.cuda.get_device_name(0)},'decoding':{'do_sample':False,'enable_thinking':False,'max_new_tokens':cfg['max_new_tokens'],'eos_token_id':tok.eos_token_id,'seed':20261011}})
 for arm in cfg['arms']:
  dest=root/'runs'/arm['id'];dest.mkdir(parents=True,exist_ok=True)
  if (dest/'raw.jsonl').exists():raise ValueError('Refusing to overwrite arm '+arm['id'])
  save(root/(cfg['host_label']+'_status.json'),{'arm':arm['id'],'phase':'loading','time':time.time()})
  t=time.monotonic()
  if cfg['host_label']=='27b':
   from src.evidence_reasoning.famu_27b import load_27b_base,new_adapter,restore_adapter
   model,audit=load_27b_base(Path(cfg['base']))
   if arm.get('adapter'):
    model=new_adapter(model,arm.get('architecture','baseline'),checkpoint_offload=False);restore_adapter(model,Path(arm['adapter']))
   model.gradient_checkpointing_disable()
  else:
   model,audit=load_model(Path(cfg['base']),'cuda',quantization='none',attention='sdpa')
   enable_compact_delta(model);enable_chunked_mlp(model,1024)
   if arm.get('adapter'):model=peft.PeftModel.from_pretrained(model,arm['adapter'],is_trainable=False,local_files_only=True)
  model.requires_grad_(False);model.eval();model.config.use_cache=True
  save(dest/'loading.json',{'arm':arm,'loading_seconds':time.monotonic()-t,'audit':audit,'adapter_sha256':sha(Path(arm['adapter'])/'adapter_model.safetensors') if arm.get('adapter') else None,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'quantization':cfg['quantization']})
  for i,(row,p) in enumerate(zip(rows,prompts)):
   torch.cuda.reset_peak_memory_stats();t=time.monotonic();raw='';error=None;nt=0
   try:
    if arm.get('architecture') in ['fixed','learned']:
     from src.evidence_reasoning.field_experts import visible_features,set_field_context
     visible={k:v for k,v in row.items() if k not in ['input_sha256','question_id','source_id','source_family']}
     set_field_context(model,visible_features(model,tok,{'split':'test','input':visible}),'qa')
    inputs=tok(p,add_special_tokens=False,return_tensors='pt').to('cuda')
    with torch.inference_mode():
     out=model.generate(**inputs,max_new_tokens=cfg['max_new_tokens'],do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id,logits_to_keep=1)
    nt=out.shape[-1]-inputs.input_ids.shape[-1];raw=tok.decode(out[0,inputs.input_ids.shape[-1]:],skip_special_tokens=True)
    del out,inputs
   except Exception: error=traceback.format_exc()
   torch.cuda.synchronize();seconds=time.monotonic()-t
   pred,parse_error=parse(raw,row)
   rec={'question_id':row['question_id'],'input_sha256':row['input_sha256'],'raw':raw,'error':error,'parse_error':parse_error,'input_tokens':lens[i],'output_tokens':nt,'hit_token_limit':nt==cfg['max_new_tokens'],'seconds':seconds,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
   for filename,value in [('raw.jsonl',rec),('predictions.jsonl',pred)]:
    with (dest/filename).open('a') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
   save(root/(cfg['host_label']+'_status.json'),{'arm':arm['id'],'phase':'running','done':i+1,'total':len(rows),'last_seconds':seconds,'last_error':error,'time':time.time()})
   print(arm['id'],row['question_id'],round(seconds,2),'tokens',nt,'error',bool(error),flush=True)
  save(dest/'complete.json',{'rows':len(rows),'raw_sha256':sha(dest/'raw.jsonl'),'predictions_sha256':sha(dest/'predictions.jsonl'),'completed_unix':time.time()})
  del model;gc.collect();torch.cuda.empty_cache()
 save(root/(cfg['host_label']+'_status.json'),{'phase':'completed','arms':[x['id'] for x in cfg['arms']],'time':time.time()})
if __name__=='__main__':main()
