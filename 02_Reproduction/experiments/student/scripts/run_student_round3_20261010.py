"""Bounded GPU R3: train-only wording variety, hybrid versus tiny sparse model."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from collections import Counter
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import traceback
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel,AutoTokenizer
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.evidence_reasoning.local_field_student import features,DIM,digest,visible_input,VERSION,LABELS
from scripts.run_gpu_student_20261009 import CachedStudent,scene_metrics,pair_key,sha,write,append
from scripts.run_student_round2_20261010 import Student,make_inputs

def mixture_epoch(pools,seed,epoch,steps):
    rng=np.random.default_rng(seed+epoch*1000);result=[]
    grouped=[]
    for pool in pools:
        groups=sorted({r['group'] for r in pool})
        by={(g,c):[i for i,p in enumerate(pool) if p['group']==g and p['label']==c] for g in groups for c in range(3)}
        if not all(by.values()):raise ValueError('A training family lacks one of the classes')
        grouped.append((groups,by))
    for _ in range(steps):
        batch=[]
        for c,n in enumerate([16,8,8]):
            for source,(groups,by) in enumerate(grouped):
                for _ in range(n//2):
                    g=groups[int(rng.integers(len(groups)))];v=by[g,c];i=v[int(rng.integers(len(v)))]
                    batch.append(i+(len(pools[0]) if source else 0))
        rng.shuffle(batch);result.append(batch)
    return result

class Sparse(nn.Module):
    def __init__(self):
        super().__init__();self.feature_head=nn.Linear(DIM,3)
        nn.init.zeros_(self.feature_head.weight);nn.init.zeros_(self.feature_head.bias)
    def forward(self,batch,numeric,hybrid=True):return self.feature_head(numeric)

def run(args):
    out=ROOT/'outputs';out.mkdir(exist_ok=True)
    fd=os.open(out/'run.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    protocol=json.loads((ROOT/'protocol.json').read_text('utf-8'))
    origin=time.time();deadline=origin+3600*protocol['wall_hours_max']
    write(out/'run_origin.json',{'pid':os.getpid(),'started_unix':origin,'deadline_unix':deadline,'protocol_sha256':sha(ROOT/'protocol.json')})
    def status(phase,**kw):
        if time.time()>deadline:raise TimeoutError('R3 fixed deadline reached; no extension')
        write(out/'status.json',{'phase':phase,'updated_unix':time.time(),'deadline_unix':deadline,'pid':os.getpid(),**kw})
    status('verifying_inputs')
    for name,e in json.loads((ROOT/'input_manifest.json').read_text('utf-8')).items():
        p=(ROOT/name).resolve();assert p.is_relative_to(ROOT.resolve()) and sha(p)==e['sha256']
    for name,h in protocol['encoder_sha256'].items():assert sha(args.encoder/name)==h
    if not torch.cuda.is_available():raise RuntimeError('Formal R3 requires CUDA')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    write(out/'environment.json',{'python':sys.version,'torch':torch.__version__,'transformers':__import__('transformers').__version__,
        'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda})
    fitdata=json.loads((ROOT/'data/fit.json').read_text('utf-8'));pools=[fitdata['original'],fitdata['augmented']];fit=sum(pools,[])
    labels=torch.tensor([p['label'] for p in fit],device='cuda')
    tok=AutoTokenizer.from_pretrained(str(args.encoder),local_files_only=True,trust_remote_code=False)
    tx,nx=make_inputs(tok,fit);nx=nx.cuda()
    historical=json.loads((ROOT/'data/historical.json').read_text('utf-8'))
    new=json.loads((ROOT/'data/new_scenes.json').read_text('utf-8'))
    sets={s:[r for r in historical if r['split']==s] for s in ['dev','test']}
    sets['r2_regression']=json.loads((ROOT/'data/r2_regression.json').read_text('utf-8'))
    sets.update({f'new_{s}':[r for r in new if r['split']==s] for s in ['dev','test']})
    assert not {p['group'] for p in fit}&{r['family'] for rows in sets.values() for r in rows}
    pairs={}
    for rows in sets.values():
        for r in rows:
            for q in r['fields'].values():
                for e in r['evidence']:
                    p={'query':q,'text':e['text'],'scope':r['scope']};pairs[pair_key(p)]=p
    ep=list(pairs.values());ex,en=make_inputs(tok,ep);en=en.cuda()
    write(out/'data_audit.json',{'original_pairs':len(pools[0]),'new_unique_pairs':len(pools[1]),
        'classes':{k:dict(Counter(p['label'] for p in v)) for k,v in zip(['original','new'],pools)},
        'max_fit_tokens':max(len(x['input_ids']) for x in tx),'max_eval_tokens':max(len(x['input_ids']) for x in ex),
        'eval_unique_pairs':len(ep),'splits':{k:len(v) for k,v in sets.items()},'truncated':0,'teacher_calls':0})
    def evaluate(model,folder,fixed_threshold=None):
        model.eval();scores={};begin=time.monotonic()
        with torch.inference_mode():
            for offset in range(0,len(ep),64):
                if time.time()>deadline:raise TimeoutError('Fixed evaluation budget reached')
                batch={k:v.cuda() for k,v in tok.pad(ex[offset:offset+64],return_tensors='pt').items()} if not isinstance(model,Sparse) else None
                probs=model(batch,en[offset:offset+64],True).float().softmax(-1)
                if not torch.isfinite(probs).all():raise ValueError('Nonfinite eval scores')
                scores.update((pair_key(p),s) for p,s in zip(ep[offset:offset+64],probs.cpu().tolist()))
        grid=[]
        if fixed_threshold is None:
            dev=sets['dev']+sets['new_dev']
            for threshold in [.5,.6,.7,.8,.9]:
                runtime=CachedStudent(scores,threshold);pred=[runtime.review(visible_input(r)) for r in dev]
                grid.append({'threshold':threshold,**scene_metrics(dev,pred)})
            selected=max(grid,key=lambda x:(-x['unknown_to_established']-x['unknown_to_not_established'],x['macro_f1'],-x['threshold']))['threshold']
        else:selected=fixed_threshold
        write(folder/'selection_before_tests.json',{'threshold':selected,'grid':grid,'fixed_epoch':12,'tests_used':False,
            'frozen_R2_threshold':fixed_threshold is not None})
        runtime=CachedStudent(scores,selected);metrics={}
        for split,rows in sets.items():
            pred=[runtime.review(visible_input(r)) for r in rows]
            write(folder/f'{split}_predictions.json',[{'id':r['id'],**p} for r,p in zip(rows,pred)])
            metrics[split]=scene_metrics(rows,pred)
            metrics[split]['by_axis']={axis:scene_metrics([r for r in rows if r['axis']==axis],[p for r,p in zip(rows,pred) if r['axis']==axis]) for axis in sorted({r['axis'] for r in rows})}
        return selected,metrics,time.monotonic()-begin
    completed=[];summaries=[]
    # Frozen baseline: no updates, no threshold reselection on the new dev set.
    from safetensors.torch import load_file,save_file
    for seed in protocol['seeds']:
        name=f'R2_H_CE_s{seed}';folder=out/name;folder.mkdir();status('frozen_baseline_evaluation',run=name)
        src=args.baseline_root/f'H_CE_s{seed}'/'model'
        for file,h in protocol['baseline_model_manifests'][str(seed)].items():assert sha(src/file)==h
        cfg=json.loads((src/'student_config.json').read_text());model=Student(src).cuda()
        heads=load_file(str(src/'relation_heads.safetensors'),device='cuda')
        model.head.load_state_dict({k.split('.',1)[1]:v for k,v in heads.items() if k.startswith('head.')})
        model.feature_head.load_state_dict({k.split('.',1)[1]:v for k,v in heads.items() if k.startswith('feature_head.')})
        threshold,metrics,seconds=evaluate(model,folder,cfg['threshold'])
        result={'run':name,'updates':0,'exposures':0,'metrics':metrics,'evaluation_seconds':seconds,'frozen_baseline':True}
        write(folder/'metrics.json',result);summaries.append(result);del model;gc.collect();torch.cuda.empty_cache()
    for arm in protocol['arms']:
      for seed in protocol['seeds']:
        name=f'{arm}_s{seed}';folder=out/name;folder.mkdir();status('initialization',run=name,completed=completed)
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        sparse=arm=='S_AUG';model=(Sparse() if sparse else Student(args.encoder)).cuda()
        init_hash=hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes() for t in model.state_dict().values())).hexdigest()
        if not sparse:assert init_hash==protocol['R2_initialization_hashes'][str(seed)]
        write(folder/'initialization.json',{'sha256':init_hash,'parameters':sum(p.numel() for p in model.parameters()),'seed':seed})
        groups=[{'params':model.feature_head.parameters(),'lr':.05}] if sparse else [
            {'params':model.base.parameters(),'lr':1e-5},{'params':model.head.parameters(),'lr':1e-4},{'params':model.feature_head.parameters(),'lr':.05}]
        opt=torch.optim.AdamW(groups,weight_decay=.01);steps=protocol['updates_per_run'];warmup=math.ceil(.1*steps)
        schedule=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min((s+1)/warmup,max(0,(steps-s)/(steps-warmup))))
        i=max(range(len(tx)),key=lambda k:len(tx[k]['input_ids']));rng=torch.get_rng_state();crng=torch.cuda.get_rng_state_all()
        b=None if sparse else {k:v.cuda() for k,v in tok.pad([tx[i]],return_tensors='pt').items()}
        loss=F.cross_entropy(model(b,nx[i:i+1],True).float(),labels[i:i+1]);loss.backward()
        assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        write(folder/'backward_preflight.json',{'max_input_tokens':len(tx[i]['input_ids']),'loss':float(loss),'finite':True,'optimizer_updates':0})
        opt.zero_grad(set_to_none=True);torch.set_rng_state(rng);torch.cuda.set_rng_state_all(crng)
        step=exposures=tokens=0;train_seconds=0
        for epoch in range(1,13):
            order=mixture_epoch(pools,seed,epoch,80);order_hash=digest(order);model.train()
            for ids in order:
                if time.time()>deadline:raise TimeoutError('Fixed R3 training deadline reached')
                begin=time.monotonic();b=None if sparse else {k:v.cuda() for k,v in tok.pad([tx[i] for i in ids],return_tensors='pt').items()}
                opt.zero_grad(set_to_none=True);loss=F.cross_entropy(model(b,nx[ids],True).float(),labels[ids])
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                opt.step();schedule.step();torch.cuda.synchronize();seconds=time.monotonic()-begin
                n=sum(len(tx[i]['input_ids']) for i in ids);tokens+=n;train_seconds+=seconds;step+=1;exposures+=len(ids)
                append(folder/'training_log.jsonl',{'step':step,'epoch':epoch,'loss':float(loss),'gradient_norm':float(norm),
                    'examples':32,'tokens':n,'seconds':seconds,'order_sha256':order_hash,'original_pairs':16,'augmented_pairs':16})
                if step==1 or step%40==0:status('training',run=name,step=step,total_steps=960,completed=completed)
            if epoch in protocol['checkpoint_epochs']:
                cp=folder/'checkpoints'/f'epoch_{epoch:02d}';cp.mkdir(parents=True)
                torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':schedule.state_dict(),
                    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
                    'step':step,'epoch':epoch,'exposures':exposures,'protocol_sha256':sha(ROOT/'protocol.json'),'order_sha256':order_hash},cp/'checkpoint.pt')
                write(cp/'manifest.json',{'sha256':sha(cp/'checkpoint.pt'),'bytes':(cp/'checkpoint.pt').stat().st_size,'step':step,'exposures':exposures})
        status('evaluation',run=name,completed=completed);threshold,metrics,seconds=evaluate(model,folder)
        dest=folder/'model';dest.mkdir()
        if sparse:
            value={'version':VERSION,'dimension':DIM,'labels':list(LABELS),'threshold':threshold,
                'coefficients':model.feature_head.weight.detach().cpu().tolist(),'intercept':model.feature_head.bias.detach().cpu().tolist(),
                'mode':'shadow','decision_authority':False,'training':'R3 program-supervised sparse classifier; no teacher runtime'}
            write(dest/'student.json',value);write(dest/'manifest.json',{'student_sha256':sha(dest/'student.json')})
        else:
            model.base.save_pretrained(dest,safe_serialization=True);tok.save_pretrained(dest)
            save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items() if not k.startswith('base.')},str(dest/'relation_heads.safetensors'))
            write(dest/'student_config.json',{'schema':'famu-student-r2','hybrid':True,'threshold':threshold,'labels':list(LABELS),
                'preprocessor':'normalized_text','feature_extractor':'local_field_student.features','max_tokens':512,'mode':'shadow','decision_authority':False,'training_round':'R3'})
            write(dest/'manifest.json',{p.name:sha(p) for p in dest.iterdir() if p.is_file()})
        result={'run':name,'updates':step,'exposures':exposures,'train_tokens':tokens,'train_seconds':train_seconds,'evaluation_seconds':seconds,
            'metrics':metrics,'production_promotion':False,'no_teacher_KD':True}
        write(folder/'metrics.json',result);write(folder/'completed.json',{'run':name,'updates':step,'completed_unix':time.time()})
        summaries.append(result);completed.append(name);del model,opt,schedule;gc.collect();torch.cuda.empty_cache()
    write(out/'summary.json',{'runs':summaries,'total_updates':sum(r['updates'] for r in summaries),'total_exposures':sum(r['exposures'] for r in summaries),'production_promotion':False})
    write(out/'completed.json',{'status':'completed','runs':completed,'completed_unix':time.time()});status('completed',completed=completed);(out/'run.lock').unlink()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--encoder',type=Path,required=True);parser.add_argument('--baseline-root',type=Path,required=True);args=parser.parse_args()
    if (ROOT/'outputs/run_origin.json').exists() or (ROOT/'outputs/run.lock').exists():raise RuntimeError('Existing experiment preserved; no implicit restart')
    try:run(args)
    except BaseException as e:
        write(ROOT/'outputs/failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc(),'time':time.time()})
        write(ROOT/'outputs/status.json',{'phase':'failed_evidence_preserved','updated_unix':time.time()});raise

if __name__=='__main__':main()
