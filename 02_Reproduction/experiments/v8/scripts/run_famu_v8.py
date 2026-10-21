"""Finite matched-exposure V8 direct/bound evidence experiment; verified resume."""
import argparse
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
import zipfile
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from peft import PeftModel
from src.evidence_reasoning import famu_v8 as v8, famu_v4 as v4
from src.evidence_reasoning.famu_llm import load_model,load_tokenizer,verify_base,completion_loss
from src.evidence_reasoning.compact_delta import enable_compact_delta
from src.evidence_reasoning.chunked_attention import enable_chunked_mlp
from src.evidence_reasoning.sft_checkpoint import save_checkpoint,load_checkpoint
from src.evidence_reasoning.review_ir import digest

SEEDS=[17,29,43];ENDPOINT=401;CHECKPOINTS=[16,64,200,401]
def read(p):return [json.loads(s) for s in p.read_text('utf-8').splitlines() if s]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),'utf-8');tmp.replace(p)
def append(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a',encoding='utf-8') as f:f.write(json.dumps(v,ensure_ascii=False,allow_nan=False)+'\n')
def status(p,**v):write(p,dict(updated_unix=time.time(),**v))
def deadline_check(a):
    if time.time()>a.deadline:raise TimeoutError('Fixed 48-hour queue budget reached; preserve checkpoints')

def load(a):
    model,audit=load_model(a.base,'cuda',dtype='bfloat16',quantization='none',attention='sdpa')
    audit['compact_delta']=enable_compact_delta(model);audit['chunked_mlp']=enable_chunked_mlp(model,1024)
    model=PeftModel.from_pretrained(model,a.init,is_trainable=True,local_files_only=True)
    assert model.peft_config['default'].r==16
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads();model.config.use_cache=False
    assert all('lora_' in n for n,p in model.named_parameters() if p.requires_grad)
    audit['trainable_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert audit['trainable_parameters']==43278336
    return model,audit

def generate(model,tok,item,max_new):
    ids=torch.tensor([item['input_ids'][:item['prompt_tokens']]],device='cuda')
    assert len(ids[0])+max_new<=16384
    begin=time.monotonic()
    with torch.inference_mode():
        out=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=max_new,
                           do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    return tok.decode(out[0,len(ids[0]):],skip_special_tokens=True),dict(input_tokens=len(ids[0]),
             output_tokens=len(out[0])-len(ids[0]),seconds=time.monotonic()-begin)

def evaluate(model,tok,scenes,replay,regression,arm,out,progress,a):
    model.eval();out.mkdir(parents=True,exist_ok=True)
    predictions=read(out/'predictions.jsonl') if (out/'predictions.jsonl').exists() else []
    assert [r['id'] for r in predictions]==[r['id'] for r in scenes[:len(predictions)]]
    for row in scenes[len(predictions):]:
        deadline_check(a);status(progress,status='evaluating',report=str(out),done=len(predictions),total=len(scenes),example=row['id'])
        item=v8.encode(tok,row,arm);raw,cost=generate(model,tok,item,192 if arm=='E1_bound' else 32)
        result=dict(id=row['id'],split=row['split'],axis=row['axis'],pair_group=row['pair_group'],gold=row['target'],raw=raw,
                    **cost,**v8.decode(raw,row,arm))
        if arm=='E1_bound':result.update(gold_states=row['states'],gold_bindings=row['bindings'])
        else:
            means={};sums={}
            for label in v8.LABELS:
                candidate=v8.encode(tok,row,arm,label)
                with torch.inference_mode():means[label]=-float(completion_loss(model,candidate,'cuda'))
                sums[label]=means[label]*(candidate['tokens']-candidate['prompt_tokens'])
            result.update(candidate_mean_scores=means,candidate_sum_scores=sums,
                          candidate_mean=max(means,key=means.get),candidate_sum=max(sums,key=sums.get),calibrated_probability=False)
        append(out/'predictions.jsonl',result);predictions.append(result)
    metrics=v8.summarize(predictions)
    if arm=='D0_direct':
        metrics['candidate']={mode:v8.summarize([dict(r,prediction=r['candidate_'+mode]) for r in predictions]) for mode in ['mean','sum']}
    metrics['decoding']='raw certificate -> deterministic field aggregation -> rule' if arm=='E1_bound' else 'raw greedy risk JSON'
    write(out/'metrics.json',metrics)
    legal=replay+regression;legal_results=read(out/'legal_generation.jsonl') if (out/'legal_generation.jsonl').exists() else []
    assert [r['id'] for r in legal_results]==[r['id'] for r in legal[:len(legal_results)]]
    for row in legal[len(legal_results):]:
        deadline_check(a);status(progress,status='evaluating_legal',report=str(out),done=len(legal_results),total=len(legal),example=row['id'])
        item=v4.encode(tok,row,16384);raw,cost=generate(model,tok,item,32)
        r=dict(id=row['id'],split=row['split'],task=row['task'],gold=row['target']['value'],
               prediction=v4.parse(raw,row['task']),raw=raw,proposition_type=row['input']['proposition_type'],**cost)
        if row['task']=='risk_status':
            means={};sums={}
            for label in v4.CLASSES['risk_status']:
                c=v4.encode(tok,row,16384,value=label)
                with torch.inference_mode():means[label]=-float(completion_loss(model,c,'cuda'))
                sums[label]=means[label]*(c['tokens']-c['prompt_tokens'])
            r.update(candidate_mean_scores=means,candidate_sum_scores=sums,candidate_mean=max(means,key=means.get),candidate_sum=max(sums,key=sums.get))
        append(out/'legal_generation.jsonl',r);legal_results.append(r)
    write(out/'legal_metrics.json',{s:v4.metrics([r for r in legal_results if r['split']==s]) for s in sorted({r['split'] for r in legal_results})})
    write(out/'report_complete.json',{'scenes':len(predictions),'legal':len(legal_results),'at':time.time()})
    return metrics

def preflight(a,tok,scenes,replay):
    audit=[];encoded={}
    for arm in v8.ARMS:
        for row in scenes:
            e=v8.encode(tok,row,arm)
            assert e['prompt_tokens']+192<=16384
            audit.append(dict(id=row['id'],arm=arm,split=row['split'],tokens=e['tokens'],prompt_tokens=e['prompt_tokens']))
            if row['split']=='train':encoded[(arm,row['id'])]=e
    for row in replay:
        e=v4.encode(tok,row,16384);encoded[('replay',row['id'])]=e
        audit.append(dict(id=row['id'],arm='replay',split='train',tokens=e['tokens'],prompt_tokens=e['prompt_tokens']))
    write(a.out/'tokenization_audit.json',audit)
    summary={'max_tokens':max(r['tokens'] for r in audit),'min_tokens':min(r['tokens'] for r in audit),
             'by_arm':{arm:{'min':min(r['tokens'] for r in audit if r['arm']==arm),'max':max(r['tokens'] for r in audit if r['arm']==arm)} for arm in v8.ARMS+['replay']},
             'truncated':0,'passed':True,'evaluations_reserved_tokens':192,'rows':len(audit)}
    write(a.out/'cpu_preflight.json',summary)
    return encoded

def train(a,tok,scenes,replay,regression,encoded,common,arm,seed):
    name=f'{arm}_s{seed}';out=a.out/name
    if (out/'result.json').exists():return
    training=[r for r in scenes if r['split']=='train']
    entries=[(arm,r['id']) for r in training for _ in range(3)]+[('replay',r['id']) for r in replay for _ in range(4)]
    random.Random(seed).shuffle(entries);plan=[entries[i:i+4] for i in range(0,len(entries),4)]
    assert len(plan)==ENDPOINT and all(len(b)==4 for b in plan)
    # Equal total objective mass for process/direct synthetic and original supervised tasks.
    weights={arm:1604/(2*1296),'replay':1604/(2*308)}
    protocol=dict(common,arm=arm,seed=seed,plan=plan,weights=weights,lr=1e-5,end_lr=1e-6,warmup=16,
                  weight_decay=.01,clip_norm=1.,updates=401,checkpoints=CHECKPOINTS,reports=[0,64,401],
                  init_optimizer='new AdamW; no inherited optimizer',queries=1604,answer_token_normalization='mean full completion+EOS CE',
                  synthetic_authority='program_constructed_fictional; supplied rule; shared sentence templates',
                  component_attribution='two arms differ in intermediate supervision and deterministic composition; no isolation of each component')
    if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text('utf-8'))==json.loads(json.dumps(protocol))
    else:write(out/'protocol.json',protocol)
    ph=digest(protocol);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);random.seed(seed)
    status(out/'status.json',status='loading',step=0,endpoint=401)
    model,audit=load(a);write(out/'loading_audit.json',audit)
    params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=1e-5,weight_decay=.01)
    events=[];exposures=Counter();updates=0
    for p in (out/'checkpoints').glob('step_*'):
        if not (p/'checkpoint.json').exists():p.rename(p.with_name('incomplete_'+p.name+'_'+str(time.time_ns())))
    checkpoints=sorted((out/'checkpoints').glob('step_*/checkpoint.json'))
    if checkpoints:
        state=load_checkpoint(model,opt,checkpoints[-1].parent,ph,'cuda')
        updates=state['updates'];events=state['events'];exposures=Counter(state['exposures'])
        if (out/'training_log.jsonl').exists():(out/'training_log.jsonl').rename(out/f'training_log_before_resume_{time.time_ns()}.jsonl')
        (out/'training_log.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events),'utf-8')
        write(out/'resume_record.json',{'checkpoint':str(checkpoints[-1].parent),'updates':updates,'at':time.time()})
    report64=[r for r in scenes if r['split']=='dev' and r['family']=='V8-dev-000']
    # Same frozen initial adapter is evaluated for both decoders once, outside seed-specific training.
    if updates==64 and not (out/'reports/step_0064/report_complete.json').exists():
        evaluate(model,tok,report64,[],regression,arm,out/'reports/step_0064',out/'status.json',a)
    torch.cuda.reset_peak_memory_stats()
    for index,batch in enumerate(plan):
        if index<updates:continue
        deadline_check(a);model.train();opt.zero_grad(set_to_none=True);begin=time.monotonic();details=[]
        for kind,ident in batch:
            item=encoded[(kind,ident)]
            status(out/'status.json',status='training',step=updates,endpoint=401,example=ident,tokens=item['tokens'])
            loss=completion_loss(model,item,'cuda')
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            (loss*weights[kind]/4).backward()
            exposures[kind+'/'+ident]+=1
            details.append(dict(id=ident,kind=kind,ce=float(loss.detach()),weight=weights[kind],tokens=item['tokens'],
                                answer_tokens=item['tokens']-item['prompt_tokens']))
            del loss
        grad=torch.nn.utils.clip_grad_norm_(params,1.)
        if not torch.isfinite(grad):raise ValueError('Nonfinite gradient')
        lr=1e-5*min((index+1)/16,1) if index<16 else 1e-6+9e-6*(401-index-1)/(401-16)
        for g in opt.param_groups:g['lr']=lr
        opt.step();updates=index+1;torch.cuda.synchronize()
        event={'step':updates,'exposures':sum(exposures.values()),'grad_norm':float(grad),'lr':lr,
               'microbatches':details,'seconds':time.monotonic()-begin,'peak_memory_bytes':torch.cuda.max_memory_allocated(),'updated_unix':time.time()}
        append(out/'training_log.jsonl',event);events.append(event)
        status(out/'status.json',status='training',step=updates,endpoint=401,last_event=event)
        if updates%8==0:print(json.dumps({'run':name,'step':updates,'seconds':event['seconds'],'grad_norm':float(grad)}),flush=True)
        if updates in CHECKPOINTS:save_checkpoint(model,opt,out/f'checkpoints/step_{updates:04d}',
                                  {'updates':updates,'events':events,'exposures':dict(exposures)},ph,'cuda')
        if updates==64:evaluate(model,tok,report64,[],regression,arm,out/'reports/step_0064',out/'status.json',a)
    expected={arm+'/'+r['id']:3 for r in training};expected.update({'replay/'+r['id']:4 for r in replay})
    assert dict(exposures)==expected
    # Fixed endpoint; no checkpoint selection using test scores.
    selected=[r for r in scenes if r['split']!='train' or r['family']=='V8-train-000']
    evaluate(model,tok,selected,replay,regression,arm,out/'reports/step_0401',out/'status.json',a)
    model.save_pretrained(out/'adapter',safe_serialization=True)
    write(out/'result.json',{'status':'completed','updates':401,'exposures':dict(exposures),'queries':1604,
                           'adapter_sha256':sha(out/'adapter/adapter_model.safetensors'),'train_seconds':sum(e['seconds'] for e in events),
                           'train_tokens':sum(x['tokens'] for e in events for x in e['microbatches']),
                           'train_answer_tokens':sum(x['answer_tokens'] for e in events for x in e['microbatches'])})
    status(out/'status.json',status='completed',step=401)
    del model,opt,params;gc.collect();torch.cuda.empty_cache()

def package(a):
    files=[p for p in a.out.rglob('*') if p.is_file() and 'checkpoints' not in p.parts and p.name not in
           {'pipeline_status.json','pipeline.log','MANIFEST.json','famu_v8_results.zip'} and p.suffix not in {'.tmp','.part'}]
    write(a.out/'MANIFEST.json',{p.relative_to(a.out).as_posix():sha(p) for p in files})
    path=a.out/'famu_v8_results.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(a.out).as_posix())
        z.write(a.out/'MANIFEST.json','MANIFEST.json')
    return dict(zip_name=path.name,zip_bytes=path.stat().st_size,zip_sha256=sha(path),files=len(files))

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--init',type=Path,required=True)
    p.add_argument('--data',type=Path,default=ROOT/'training_data/famu_v8_20261005');p.add_argument('--out',type=Path,default=ROOT/'outputs')
    p.add_argument('--preflight-only',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    budget=a.out/'budget.json'
    if budget.exists():a.deadline=json.loads(budget.read_text())['deadline_unix']
    else:a.deadline=time.time()+48*3600;write(budget,{'started_unix':time.time(),'deadline_unix':a.deadline,'max_hours':48})
    try:
        manifest=json.loads((a.data/'manifest.json').read_text('utf-8'))
        for n,h in manifest['files'].items():assert sha(a.data/n)==h,n
        for n,h in json.loads((ROOT/'code_manifest.json').read_text('utf-8')).items():assert sha(ROOT/n)==h,n
        init_hash=sha(a.init/'adapter_model.safetensors')
        expected=json.loads((ROOT/'init_lock.json').read_text('utf-8'))
        assert init_hash==expected['adapter_model.safetensors'] and sha(a.init/'adapter_config.json')==expected['adapter_config.json']
        scenes=read(a.data/'scenes.jsonl');replay=read(a.data/'replay77.jsonl');regression=read(a.data/'regression9.jsonl')
        status(a.out/'pipeline_status.json',phase='preflight',current='initial')
        tok=load_tokenizer(a.base);encoded=preflight(a,tok,scenes,replay)
        common={'data_manifest_sha256':sha(a.data/'manifest.json'),'base_lock':verify_base(a.base),'initial_adapter_sha256':init_hash,
                'code_manifest_sha256':sha(ROOT/'code_manifest.json'),'protocol_doc_sha256':sha(ROOT/'docs/法眸V8_证据绑定风险场训练协议_20261005.md'),
                'seeds':SEEDS,'arms':v8.ARMS,'init':'V4 review_R1_20261001 endpoint80; chosen before V8 data evaluation',
                'max_length':16384,'truncation':False,'production_promotion':False}
        if (a.out/'pipeline_protocol.json').exists():assert json.loads((a.out/'pipeline_protocol.json').read_text('utf-8'))==common
        else:write(a.out/'pipeline_protocol.json',common)
        if a.preflight_only:return
        initial=a.out/'initial'
        if not (initial/'all_initial_complete.json').exists():
            model,audit=load(a);write(initial/'loading_audit.json',audit)
            # Check CE masking/shift and real longest training backward without optimizer update.
            if not (initial/'gpu_preflight.json').exists():
                model.eval();short=min(encoded.values(),key=lambda x:x['tokens'])
                ids=torch.tensor([short['input_ids']],device='cuda');labs=torch.tensor([short['labels']],device='cuda')
                with torch.no_grad():
                    selected=float(completion_loss(model,short,'cuda'))
                    full=float(model(input_ids=ids,attention_mask=torch.ones_like(ids),labels=labs,use_cache=False).loss)
                assert math.isfinite(full) and abs(selected-full)<.005
                del ids,labs
                probes=[]
                for kind in v8.ARMS+['replay']:
                    item=max((e for (k,_),e in encoded.items() if k==kind),key=lambda x:x['tokens'])
                    status(initial/'status.json',status='gpu_backward_preflight',kind=kind,tokens=item['tokens'])
                    model.train();model.zero_grad(set_to_none=True);loss=completion_loss(model,item,'cuda');loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
                    assert torch.isfinite(loss) and torch.isfinite(norm)
                    probes.append({'kind':kind,'id':item['id'],'tokens':item['tokens'],'loss':float(loss),'grad_norm':float(norm)})
                    del loss;model.zero_grad(set_to_none=True)
                write(initial/'gpu_preflight.json',{'passed':True,'mask_abs_diff':abs(selected-full),'backward_probes':probes,'optimizer_updates':0})
            small=[r for r in scenes if r['family']=='V8-dev-000']
            for arm in v8.ARMS:evaluate(model,tok,small,[],regression,arm,initial/arm,initial/'status.json',a)
            write(initial/'all_initial_complete.json',{'at':time.time()})
            del model;gc.collect();torch.cuda.empty_cache()
        complete=[]
        # Bound-first at seed17, alternating arm order; all branches retain the same frozen start.
        for seed in SEEDS:
            for arm in (list(reversed(v8.ARMS)) if seed!=29 else v8.ARMS):
                name=f'{arm}_s{seed}';status(a.out/'pipeline_status.json',phase='running',current=name,completed=complete)
                train(a,tok,scenes,replay,regression,encoded,common,arm,seed);complete.append(name)
        # Evaluate the unchanged initial adapter on the complete frozen endpoint sets.
        if not (initial/'full_complete.json').exists():
            model,_=load(a)
            selected=[r for r in scenes if r['split']!='train' or r['family']=='V8-train-000']
            status(a.out/'pipeline_status.json',phase='running',current='initial',completed=complete)
            for arm in v8.ARMS:evaluate(model,tok,selected,replay,regression,arm,initial/('full_'+arm),initial/'status.json',a)
            write(initial/'full_complete.json',{'at':time.time()});del model;gc.collect();torch.cuda.empty_cache()
        reports={name:json.loads((a.out/name/'reports/step_0401/metrics.json').read_text('utf-8')) for name in complete}
        write(a.out/'all_reports.json',reports)
        status(a.out/'pipeline_status.json',phase='packaging',completed=complete)
        result=package(a)
        status(a.out/'pipeline_status.json',phase='completed',completed=complete,updates=2406,queries=9624,**result)
    except Exception:
        write(a.out/'failure.json',{'traceback':traceback.format_exc(),'at':time.time()})
        status(a.out/'pipeline_status.json',phase='failed');raise

if __name__=='__main__':main()
