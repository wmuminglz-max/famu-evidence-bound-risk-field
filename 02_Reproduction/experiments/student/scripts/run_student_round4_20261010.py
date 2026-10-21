"""Single bounded CUDA R4 run; fixed 2x2 comparison, no production mutation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
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
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.student_round4_model import Joint,Factorized,inputs
from scripts.run_student_round3_20261010 import mixture_epoch
from scripts.run_gpu_student_20261009 import CachedStudent,scene_metrics,pair_key,sha,write,append
from src.evidence_reasoning.local_field_student import LocalFieldStudent,digest,visible_input,VERSION,LABELS
from src.evidence_reasoning.factorized_field_student import VERSION as FVERSION
from src.evidence_reasoning.factorized_field_features import VERSION as FEATURE_VERSION

def datasets():
    old=json.loads((ROOT/'data/historical.json').read_text('utf-8'))
    new=json.loads((ROOT/'data/new_scenes.json').read_text('utf-8'))
    return {s:[r for r in old if r['split']==s] for s in ['dev','test']}|{
        k:json.loads((ROOT/f'data/{k}.json').read_text('utf-8')) for k in ['r2_regression','r3_regression']}|{
        f'new_{s}':[r for r in new if r['split']==s] for s in ['dev','test']}

def finite(value):
    if torch.is_tensor(value):return bool(torch.isfinite(value).all())
    if isinstance(value,dict):return all(finite(v) for v in value.values())
    if isinstance(value,(list,tuple)):return all(finite(v) for v in value)
    return not isinstance(value,float) or math.isfinite(value)

def run():
    out=ROOT/'outputs';out.mkdir(exist_ok=True)
    if (out/'run_origin.json').exists():raise RuntimeError('Existing run preserved; never implicitly restart')
    fd=os.open(out/'run.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    protocol=json.loads((ROOT/'protocol.json').read_text('utf-8'));ph=sha(ROOT/'protocol.json');mh=sha(ROOT/'input_manifest.json')
    origin=time.time();deadline=origin+3600*protocol['wall_hours_max']
    write(out/'run_origin.json',{'pid':os.getpid(),'started_unix':origin,'deadline_unix':deadline,'protocol_sha256':ph,'input_manifest_sha256':mh})
    def status(phase,**kw):
        if time.time()>deadline:raise TimeoutError('R4 fixed two-hour window expired; preserve evidence')
        write(out/'status.json',{'phase':phase,'updated_unix':time.time(),'deadline_unix':deadline,'pid':os.getpid(),**kw})
    status('verifying_inputs')
    for name,e in json.loads((ROOT/'input_manifest.json').read_text('utf-8')).items():
        p=(ROOT/name).resolve();assert p.is_relative_to(ROOT.resolve()) and sha(p)==e['sha256'] and p.stat().st_size==e['bytes']
    if not torch.cuda.is_available():raise RuntimeError('Authorized formal run requires CUDA')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    write(out/'environment.json',{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'gpu':torch.cuda.get_device_name(0),
        'cuda':torch.version.cuda,'runtime_device':'cuda','training_dtype':'float32','teacher_calls':0})
    fitdata=json.loads((ROOT/'data/fit.json').read_text('utf-8'));sets=datasets()
    assert not {p['group'] for pool in fitdata.values() for p in pool}&{r['family'] for rows in sets.values() for r in rows}
    lookup={}
    for rows in sets.values():
        for r in rows:
            for q in r['fields'].values():
                for e in r['evidence']:
                    p={'query':q,'text':e['text'],'scope':r['scope']};lookup[pair_key(p)]=p
    ep=list(lookup.values());completed=[];summaries=[];initial={};orders={}
    write(out/'data_audit.json',{'pool_sizes':{k:len(v) for k,v in fitdata.items()},'sets':{k:len(v) for k,v in sets.items()},
        'eval_pairs':len(ep),'max_fit_input_chars':max(len(p['query'])+len(p['text']) for pool in fitdata.values() for p in pool),
        'max_eval_input_chars':max(len(p['query'])+len(p['text']) for p in ep),'truncation':False,'teacher_calls':0,'new_independent_training_facts':0})
    def report(runtime,folder):
        metrics={}
        for split,rows in sets.items():
            pred=[runtime.review(visible_input(r)) for r in rows]
            write(folder/f'{split}_predictions.json',[{'id':r['id'],**p} for r,p in zip(rows,pred)])
            metrics[split]=scene_metrics(rows,pred)
            metrics[split]['by_axis']={axis:scene_metrics([r for r in rows if r['axis']==axis],[p for r,p in zip(rows,pred) if r['axis']==axis]) for axis in sorted({r['axis'] for r in rows})}
        return metrics
    folder=out/'frozen_stdlib';folder.mkdir();status('frozen_stdlib_evaluation')
    start=time.monotonic();base=LocalFieldStudent.load(ROOT/'baseline_stdlib')
    write(folder/'metrics.json',{'run':'frozen_stdlib','updates':0,'exposures':0,'metrics':report(base,folder),'evaluation_seconds':time.monotonic()-start})
    for arm in protocol['arms']:
        factor=arm.startswith('F_');pairing=arm.split('_')[1]
        pools=[fitdata['original'],fitdata['recombined' if pairing=='RECOMB' else 'augmented']];fit=sum(pools,[])
        status('precomputing_observable_features',arm=arm,completed=completed)
        nx=[t.cuda() for t in inputs(fit,factor)];ex=[t.cuda() for t in inputs(ep,factor)]
        labels=torch.tensor([p['label'] for p in fit],device='cuda')
        for seed in protocol['seeds']:
            name=f'{arm}_s{seed}';folder=out/name;folder.mkdir();status('initialization',run=name,completed=completed)
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            model=(Factorized() if factor else Joint()).cuda()
            ih=hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes() for t in model.state_dict().values())).hexdigest()
            key=(factor,seed)
            if key in initial:assert initial[key]==ih
            initial[key]=ih
            write(folder/'initialization.json',{'sha256':ih,'parameters':sum(p.numel() for p in model.parameters()),'seed':seed,'representation':arm[0]})
            opt=torch.optim.AdamW(model.parameters(),lr=.01 if factor else .05,weight_decay=.01)
            steps=protocol['updates_per_run'];warmup=math.ceil(.1*steps)
            schedule=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min((s+1)/warmup,max(0,(steps-s)/(steps-warmup))))
            longest=max(range(len(fit)),key=lambda i:len(fit[i]['query'])+len(fit[i]['text']))
            loss=F.cross_entropy(model(*(t[longest:longest+1] for t in nx)),labels[longest:longest+1]);loss.backward()
            assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            write(folder/'backward_preflight.json',{'max_chars':len(fit[longest]['query'])+len(fit[longest]['text']),'loss':float(loss.detach()),'finite':True,'optimizer_updates':0})
            opt.zero_grad(set_to_none=True);step=exposures=chars=0;seconds=0.
            for epoch in range(1,13):
                order=mixture_epoch(pools,seed,epoch,80);orderhash=digest(order)
                k=(pairing,seed,epoch)
                if k in orders:assert orders[k]==orderhash
                orders[k]=orderhash;model.train()
                for ids in order:
                    if time.time()>deadline:raise TimeoutError('R4 update budget wall deadline reached')
                    start=time.monotonic();opt.zero_grad(set_to_none=True)
                    loss=F.cross_entropy(model(*(t[ids] for t in nx)),labels[ids]);assert torch.isfinite(loss)
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    opt.step();schedule.step();torch.cuda.synchronize();elapsed=time.monotonic()-start
                    n=sum(len(fit[i]['query'])+len(fit[i]['text']) for i in ids);chars+=n;seconds+=elapsed;step+=1;exposures+=32
                    append(folder/'training_log.jsonl',{'step':step,'epoch':epoch,'loss':float(loss.detach()),'gradient_norm':float(norm),
                        'examples':32,'input_chars':n,'seconds':elapsed,'order_sha256':orderhash,'original_pairs':16,'augmented_pairs':16})
                    if step==1 or step%80==0:status('training',run=name,step=step,total_steps=960,completed=completed)
                if epoch in protocol['checkpoint_epochs']:
                    cp=folder/'checkpoints'/f'epoch_{epoch:02d}';cp.mkdir(parents=True)
                    assert finite(model.state_dict()) and finite(opt.state_dict())
                    torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':schedule.state_dict(),
                        'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
                        'step':step,'epoch':epoch,'exposures':exposures,'protocol_sha256':ph,'input_manifest_sha256':mh,'order_sha256':orderhash},cp/'checkpoint.pt')
                    write(cp/'manifest.json',{'sha256':sha(cp/'checkpoint.pt'),'bytes':(cp/'checkpoint.pt').stat().st_size,'step':step,'exposures':exposures})
            status('evaluation',run=name,completed=completed);model.eval();scores={};start=time.monotonic()
            with torch.inference_mode():
                for offset in range(0,len(ep),256):
                    prob=model(*(t[offset:offset+256] for t in ex)).softmax(-1);assert torch.isfinite(prob).all()
                    scores.update((pair_key(p),s) for p,s in zip(ep[offset:offset+256],prob.cpu().tolist()))
            grid=[];dev=sets['dev']+sets['new_dev']
            for threshold in [.5,.6,.7,.8,.9]:
                runtime=CachedStudent(scores,threshold);grid.append({'threshold':threshold,**scene_metrics(dev,[runtime.review(visible_input(r)) for r in dev])})
            threshold=max(grid,key=lambda r:(-r['unknown_to_established']-r['unknown_to_not_established'],r['macro_f1'],-r['threshold']))['threshold']
            write(folder/'selection_before_tests.json',{'threshold':threshold,'grid':grid,'fixed_epoch':12,'tests_used':False})
            metrics=report(CachedStudent(scores,threshold),folder);eval_seconds=time.monotonic()-start
            dest=folder/'model';dest.mkdir()
            if factor:
                np.savez(dest/'weights.npz',**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})
                write(dest/'student_config.json',{'version':FVERSION,'feature_version':FEATURE_VERSION,'labels':list(LABELS),
                    'threshold':threshold,'mode':'shadow','decision_authority':False})
                write(dest/'manifest.json',{p.name:sha(p) for p in dest.iterdir() if p.is_file()})
            else:
                write(dest/'student.json',{'version':VERSION,'dimension':8192,'labels':list(LABELS),'threshold':threshold,
                    'coefficients':model.feature_head.weight.detach().cpu().tolist(),'intercept':model.feature_head.bias.detach().cpu().tolist(),'mode':'shadow','decision_authority':False})
                write(dest/'manifest.json',{'student_sha256':sha(dest/'student.json')})
            assert sum(p.stat().st_size for p in dest.iterdir())<4*1024**2
            # Fixed diagnostic scores permit independent CPU/GPU tolerance check.
            write(folder/'pair_cpu_reference.json',[{'input':p,'scores':scores[pair_key(p)]} for p in ep[:128]])
            result={'run':name,'updates':step,'exposures':exposures,'train_input_chars':chars,'train_seconds':seconds,
                'evaluation_seconds':eval_seconds,'metrics':metrics,'production_promotion':False,'teacher_calls':0}
            write(folder/'metrics.json',result);write(folder/'completed.json',{'run':name,'updates':step,'completed_unix':time.time()})
            summaries.append(result);completed.append(name);del model,opt,schedule;gc.collect();torch.cuda.empty_cache()
        del nx,ex,labels;gc.collect();torch.cuda.empty_cache()
    status('checkpoint_readonly_cpu_audit',completed=completed);cpresults=[]
    for cp in sorted(out.glob('*/checkpoints/epoch_*/checkpoint.pt')):
        m=json.loads((cp.parent/'manifest.json').read_text());assert sha(cp)==m['sha256']
        c=torch.load(cp,map_location='cpu',weights_only=False)
        assert finite(c['model']) and finite(c['optimizer']) and c['protocol_sha256']==ph and c['input_manifest_sha256']==mh
        assert c['step']==m['step']==c['epoch']*80 and c['exposures']==m['exposures']==c['step']*32
        assert all(k in c for k in ['torch_rng','cuda_rng','python_rng','numpy_rng','scheduler'])
        cpresults.append({'path':cp.relative_to(ROOT).as_posix(),**m,'finite_optimizer_model':True,'rng_scheduler_present':True})
    assert len(cpresults)==24
    write(out/'checkpoint_audit.json',{'all_24_passed':True,'checkpoints':cpresults,'optimizer_updates':0})
    write(out/'summary.json',{'runs':summaries,'total_updates':sum(r['updates'] for r in summaries),'total_exposures':sum(r['exposures'] for r in summaries),'production_promotion':False})
    write(out/'completed.json',{'status':'completed','runs':completed,'completed_unix':time.time()});status('completed',completed=completed);(out/'run.lock').unlink()

if __name__=='__main__':
    try:run()
    except BaseException as e:
        write(ROOT/'outputs/failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc(),'time':time.time()})
        write(ROOT/'outputs/status.json',{'phase':'failed_evidence_preserved','updated_unix':time.time()});raise
