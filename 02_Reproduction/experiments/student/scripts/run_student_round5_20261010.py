"""Single two-hour-bounded R5 typed relation experiment, CUDA training only."""
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
from scripts.student_round5_model import Typed,inputs
from scripts.run_student_round3_20261010 import mixture_epoch
from scripts.run_gpu_student_20261009 import CachedStudent,scene_metrics,pair_key,sha,write,append
from scripts.run_student_round4_20261010 import finite,datasets as previous_sets
from src.evidence_reasoning.local_field_student import LocalFieldStudent,digest,visible_input,LABELS
from src.evidence_reasoning.factorized_field_student import FactorizedFieldStudent
from src.evidence_reasoning.typed_field_student import VERSION,KINDS,FEATURES

def run():
    out=ROOT/'outputs';out.mkdir(exist_ok=True)
    fd=os.open(out/'run.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    protocol=json.loads((ROOT/'protocol.json').read_text('utf-8'));ph=sha(ROOT/'protocol.json');mh=sha(ROOT/'input_manifest.json')
    start=time.time();deadline=start+3600*protocol['wall_hours_max']
    write(out/'run_origin.json',{'pid':os.getpid(),'started_unix':start,'deadline_unix':deadline,'protocol_sha256':ph,'input_manifest_sha256':mh})
    def status(phase,**kw):
        if time.time()>deadline:raise TimeoutError('R5 fixed wall deadline; no extension')
        write(out/'status.json',{'phase':phase,'updated_unix':time.time(),'deadline_unix':deadline,'pid':os.getpid(),**kw})
    status('verifying_inputs')
    for name,e in json.loads((ROOT/'input_manifest.json').read_text('utf-8')).items():
        p=(ROOT/name).resolve();assert p.is_relative_to(ROOT.resolve()) and sha(p)==e['sha256'] and p.stat().st_size==e['bytes']
    assert torch.cuda.is_available();torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    write(out/'environment.json',{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,'FP32':True,'teacher_calls':0})
    data=json.loads((ROOT/'data/fit.json').read_text('utf-8'));pools=[data['original'],data['recombined']];fit=sum(pools,[])
    sets=previous_sets();sets['r4_regression']=json.loads((ROOT/'data/r4_regression.json').read_text('utf-8'))
    assert not {p['group'] for p in fit}&{r['family'] for rows in sets.values() for r in rows}
    epmap={};kinds={};gold={}
    for rows in sets.values():
        for r in rows:
            for kind,(field,q) in enumerate(r['fields'].items()):
                assert field=='ABCD'[kind]
                for e in r['evidence']:
                    p={'query':q,'text':e['text'],'scope':r['scope']};key=pair_key(p)
                    label=1 if e['id'] in r['bindings'][field]['s'] else 2 if e['id'] in r['bindings'][field]['r'] else 0
                    if key in kinds:assert kinds[key]==kind and gold[key]==label
                    epmap[key]=p;kinds[key]=kind;gold[key]=label
    ep=list(epmap.values());status('precomputing_observable_features')
    nx=[v.cuda() for v in inputs(fit)];ex=[v.cuda() for v in inputs(ep)]
    labels=torch.tensor([p['label'] for p in fit],device='cuda');types=torch.tensor([p['query_kind'] for p in fit],device='cuda')
    write(out/'data_audit.json',{'pool_sizes':{k:len(v) for k,v in data.items()},'splits':{k:len(v) for k,v in sets.items()},
        'eval_unique_pairs':len(ep),'max_fit_input_chars':max(len(p['query'])+len(p['text']) for p in fit),
        'max_eval_input_chars':max(len(p['query'])+len(p['text']) for p in ep),'truncated':False,'teacher_calls':0,'type_labels_training_only':True})
    def report(runtime,folder):
        metrics={}
        for split,rows in sets.items():
            pred=[runtime.review(visible_input(r)) for r in rows]
            write(folder/f'{split}_predictions.json',[{'id':r['id'],**p} for r,p in zip(rows,pred)])
            metrics[split]=scene_metrics(rows,pred)
            metrics[split]['by_axis']={axis:scene_metrics([r for r in rows if r['axis']==axis],[p for r,p in zip(rows,pred) if r['axis']==axis]) for axis in sorted({r['axis'] for r in rows})}
        return metrics
    for name,source,cls in [('frozen_stdlib',ROOT/'baseline_stdlib',LocalFieldStudent)]+[(f'R4_F_s{s}',ROOT/f'baseline_F_s{s}',FactorizedFieldStudent) for s in protocol['seeds']]:
        status('frozen_baseline_evaluation',run=name);folder=out/name;folder.mkdir();begin=time.monotonic()
        runtime=cls.load(source);write(folder/'metrics.json',{'run':name,'updates':0,'exposures':0,'metrics':report(runtime,folder),'evaluation_seconds':time.monotonic()-begin,'threshold_frozen':runtime.model['threshold']})
    completed=[];summaries=[];orders={}
    for arm in protocol['arms']:
        architecture='linear' if arm=='T_LINEAR' else 'mlp'
        for seed in protocol['seeds']:
            name=f'{arm}_s{seed}';folder=out/name;folder.mkdir();status('initialization',run=name,completed=completed)
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            model=Typed(architecture).cuda();ih=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
            write(folder/'initialization.json',{'sha256':ih,'parameters':sum(p.numel() for p in model.parameters()),'seed':seed,'architecture':architecture})
            opt=torch.optim.AdamW(model.parameters(),lr=.05 if architecture=='linear' else .01,weight_decay=.01)
            steps=960;warmup=96;schedule=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min((s+1)/warmup,max(0,(steps-s)/(steps-warmup))))
            def objective(ids):
                q,e=model.components(*(t[ids] for t in nx));qce=F.cross_entropy(q,types[ids]);ece=F.cross_entropy(e[torch.arange(len(ids),device='cuda'),types[ids]],labels[ids]);return qce+ece,qce,ece
            longest=max(range(len(fit)),key=lambda i:len(fit[i]['query'])+len(fit[i]['text']))
            loss,_,_=objective([longest]);loss.backward();assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            write(folder/'backward_preflight.json',{'max_input_chars':len(fit[longest]['query'])+len(fit[longest]['text']),'loss':float(loss.detach()),'finite':True,'optimizer_updates':0});opt.zero_grad(set_to_none=True)
            step=exposures=chars=0;seconds=0.
            for epoch in range(1,13):
                order=mixture_epoch(pools,seed,epoch,80);h=digest(order)
                if (seed,epoch) in orders:assert orders[seed,epoch]==h
                orders[seed,epoch]=h;model.train()
                for ids in order:
                    if time.time()>deadline:raise TimeoutError('R5 wall deadline expired')
                    begin=time.monotonic();opt.zero_grad(set_to_none=True);loss,qce,ece=objective(ids);assert torch.isfinite(loss)
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();schedule.step();torch.cuda.synchronize()
                    elapsed=time.monotonic()-begin;n=sum(len(fit[i]['query'])+len(fit[i]['text']) for i in ids);chars+=n;seconds+=elapsed;step+=1;exposures+=32
                    append(folder/'training_log.jsonl',{'step':step,'epoch':epoch,'loss':float(loss.detach()),'query_loss':float(qce.detach()),'evidence_loss':float(ece.detach()),'gradient_norm':float(norm),
                        'examples':32,'auxiliary_query_labels':32,'input_chars':n,'seconds':elapsed,'order_sha256':h,'original_pairs':16,'augmented_pairs':16})
                    if step==1 or step%80==0:status('training',run=name,step=step,total_steps=960,completed=completed)
                if epoch in [6,12]:
                    cp=folder/'checkpoints'/f'epoch_{epoch:02d}';cp.mkdir(parents=True);assert finite(model.state_dict()) and finite(opt.state_dict())
                    torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':schedule.state_dict(),'torch_rng':torch.get_rng_state(),
                        'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'epoch':epoch,'step':step,'exposures':exposures,
                        'protocol_sha256':ph,'input_manifest_sha256':mh,'order_sha256':h},cp/'checkpoint.pt')
                    write(cp/'manifest.json',{'sha256':sha(cp/'checkpoint.pt'),'bytes':(cp/'checkpoint.pt').stat().st_size,'step':step,'exposures':exposures})
            model.eval();status('evaluation',run=name,completed=completed);begin=time.monotonic();scores={};components={}
            with torch.inference_mode():
                for offset in range(0,len(ep),256):
                    q,e=model.components(*(t[offset:offset+256] for t in ex));qp=q.softmax(-1);ee=e.softmax(-1);pp=(qp.unsqueeze(-1)*ee).sum(1);assert torch.isfinite(pp).all()
                    for p,a,b,c in zip(ep[offset:offset+256],pp.cpu().tolist(),qp.cpu().tolist(),ee.cpu().tolist()):scores[pair_key(p)]=a;components[pair_key(p)]={'query':b,'evidence':c}
            dev=sets['dev']+sets['new_dev'];grid=[]
            for threshold in [.5,.6,.7,.8,.9]:
                runtime=CachedStudent(scores,threshold);grid.append({'threshold':threshold,**scene_metrics(dev,[runtime.review(visible_input(r)) for r in dev])})
            threshold=max(grid,key=lambda r:(-r['unknown_to_established']-r['unknown_to_not_established'],r['macro_f1'],-r['threshold']))['threshold']
            write(folder/'selection_before_tests.json',{'threshold':threshold,'grid':grid,'fixed_epoch':12,'tests_used':False})
            metrics=report(CachedStudent(scores,threshold),folder)
            write(folder/'component_scores.json',[{'input_sha256':pair_key(p),'gold_kind':kinds[pair_key(p)],'gold_relation':gold[pair_key(p)],**components[pair_key(p)]} for p in ep])
            dest=folder/'model';dest.mkdir();np.savez(dest/'weights.npz',**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})
            write(dest/'student_config.json',{'version':VERSION,'feature_version':FEATURES,'architecture':architecture,'kinds':list(KINDS),'labels':list(LABELS),'threshold':threshold,'mode':'shadow','decision_authority':False})
            write(dest/'manifest.json',{p.name:sha(p) for p in dest.iterdir() if p.is_file()});assert sum(p.stat().st_size for p in dest.iterdir())<4*1024**2
            write(folder/'pair_cpu_reference.json',[{'input':p,'scores':scores[pair_key(p)]} for p in ep[:128]])
            result={'run':name,'updates':step,'exposures':exposures,'train_input_chars':chars,'train_seconds':seconds,'evaluation_seconds':time.monotonic()-begin,'metrics':metrics,'production_promotion':False,'teacher_calls':0}
            write(folder/'metrics.json',result);write(folder/'completed.json',{'run':name,'updates':step,'completed_unix':time.time()});completed.append(name);summaries.append(result)
            del model,opt,schedule;gc.collect();torch.cuda.empty_cache()
    status('checkpoint_readonly_cpu_audit',completed=completed);checks=[]
    for p in sorted(out.glob('*/checkpoints/epoch_*/checkpoint.pt')):
        m=json.loads((p.parent/'manifest.json').read_text());c=torch.load(p,map_location='cpu',weights_only=False)
        assert sha(p)==m['sha256'] and finite(c['model']) and finite(c['optimizer']) and c['protocol_sha256']==ph and c['input_manifest_sha256']==mh
        assert c['step']==m['step']==c['epoch']*80 and c['exposures']==m['exposures']==c['step']*32
        assert all(c[k] is not None for k in ['torch_rng','cuda_rng','python_rng','numpy_rng','scheduler'])
        checks.append({'path':p.relative_to(ROOT).as_posix(),**m,'all_passed':True})
    assert len(checks)==12;write(out/'checkpoint_audit.json',{'all_12_passed':True,'checkpoints':checks,'optimizer_updates':0})
    write(out/'summary.json',{'runs':summaries,'total_updates':sum(r['updates'] for r in summaries),'total_exposures':sum(r['exposures'] for r in summaries),'production_promotion':False})
    write(out/'completed.json',{'status':'completed','runs':completed,'completed_unix':time.time()});status('completed',completed=completed);(out/'run.lock').unlink()

if __name__=='__main__':
    if (ROOT/'outputs/run_origin.json').exists() or (ROOT/'outputs/run.lock').exists():raise RuntimeError('Existing R5 preserved; no restart')
    try:run()
    except BaseException as e:
        write(ROOT/'outputs/failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc(),'time':time.time()});write(ROOT/'outputs/status.json',{'phase':'failed_evidence_preserved','updated_unix':time.time()});raise
