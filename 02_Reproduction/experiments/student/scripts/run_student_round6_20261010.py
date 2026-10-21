"""Finite R6 GPU language training, frozen-weights routing controls and full traces."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import gc,hashlib,json,random,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.student_round6_model import Typed,inputs
from scripts.student_round5_model import inputs as raw_inputs
from scripts.run_student_round3_20261010 import mixture_epoch
from scripts.run_gpu_student_20261009 import CachedStudent,scene_metrics,pair_key,sha,write,append
from scripts.run_student_round4_20261010 import finite,datasets as base_sets
from src.evidence_reasoning.local_field_student import digest,visible_input,LABELS
from src.evidence_reasoning.typed_field_student import KINDS
from src.evidence_reasoning.language_field_student import VERSION

def run():
    out=ROOT/'outputs';out.mkdir(exist_ok=True)
    fd=os.open(out/'run.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    protocol=json.loads((ROOT/'protocol.json').read_text('utf-8'));ph=sha(ROOT/'protocol.json');mh=sha(ROOT/'input_manifest.json')
    start=time.time();deadline=start+7200
    write(out/'run_origin.json',{'pid':os.getpid(),'started_unix':start,'deadline_unix':deadline,'protocol_sha256':ph,'input_manifest_sha256':mh})
    def status(phase,**kw):
        if time.time()>deadline:raise TimeoutError('R6 fixed2h deadline; no extension')
        write(out/'status.json',{'phase':phase,'updated_unix':time.time(),'deadline_unix':deadline,'pid':os.getpid(),**kw})
    status('verifying_inputs')
    for n,e in json.loads((ROOT/'input_manifest.json').read_text('utf-8')).items():
        p=(ROOT/n).resolve();assert p.is_relative_to(ROOT.resolve()) and sha(p)==e['sha256'] and p.stat().st_size==e['bytes']
    assert torch.cuda.is_available();torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    write(out/'environment.json',{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,'FP32':True,'teacher_calls':0})
    data=json.loads((ROOT/'data/fit.json').read_text('utf-8'));pools=[data['original'],data['recombined']];fit=sum(pools,[])
    sets=base_sets()|{k:json.loads((ROOT/f'data/{k}.json').read_text('utf-8')) for k in ['r4_regression','r5_regression']}
    assert not {p['group'] for p in fit}&{r['family'] for rows in sets.values() for r in rows}
    epmap={}
    for rows in sets.values():
        for r in rows:
            for q in r['fields'].values():
                for e in r['evidence']:
                    p={'query':q,'text':e['text'],'scope':r['scope']};epmap[pair_key(p)]=p
    ep=list(epmap.values());status('observable_features')
    nx=[v.cuda() for v in inputs(fit)];ex={rep:[v.cuda() for v in fn(ep)] for rep,fn in [('language',inputs),('raw',raw_inputs)]}
    labels=torch.tensor([p['label'] for p in fit],device='cuda');types=torch.tensor([p['query_kind'] for p in fit],device='cuda')
    write(out/'data_audit.json',{'pool_sizes':{k:len(v) for k,v in data.items()},'splits':{k:len(v) for k,v in sets.items()},'eval_unique_pairs':len(ep),
      'max_fit_input_chars':max(len(p['query'])+len(p['text']) for p in fit),'max_eval_input_chars':max(len(p['query'])+len(p['text']) for p in ep),'truncated':False,'teacher_calls':0})
    summaries=[]
    for seed in protocol['seeds']:
        name=f'LANG_s{seed}';folder=out/name;folder.mkdir();status('initialization',run=name)
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);model=Typed('mlp').cuda()
        ih=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
        assert ih==json.loads((ROOT/f'baseline_s{seed}/initialization.json').read_text())['sha256']
        reference=[json.loads(s) for s in (ROOT/f'baseline_s{seed}/training_log.jsonl').read_text().splitlines()]
        write(folder/'initialization.json',{'sha256':ih,'parameters':sum(p.numel() for p in model.parameters()),'seed':seed,'matched_frozen_R5':True})
        opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.01)
        schedule=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min((s+1)/96,max(0,(960-s)/864)))
        def objective(ids):
            q,e=model.components(*(t[ids] for t in nx));qc=F.cross_entropy(q,types[ids]);ec=F.cross_entropy(e[torch.arange(len(ids),device='cuda'),types[ids]],labels[ids]);return qc+ec,qc,ec
        longest=max(range(len(fit)),key=lambda i:len(fit[i]['query'])+len(fit[i]['text']));loss,_,_=objective([longest]);loss.backward()
        assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        write(folder/'backward_preflight.json',{'finite':True,'optimizer_updates':0,'max_input_chars':len(fit[longest]['query'])+len(fit[longest]['text']),'loss':float(loss.detach())});opt.zero_grad(set_to_none=True)
        step=chars=0;seconds=0.
        for epoch in range(1,13):
            order=mixture_epoch(pools,seed,epoch,80);h=digest(order)
            for ids in order:
                if time.time()>deadline:raise TimeoutError('R6 fixed deadline')
                assert h==reference[step]['order_sha256'];begin=time.monotonic();opt.zero_grad(set_to_none=True);loss,qc,ec=objective(ids);assert torch.isfinite(loss)
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();schedule.step();torch.cuda.synchronize()
                elapsed=time.monotonic()-begin;n=sum(len(fit[i]['query'])+len(fit[i]['text']) for i in ids);chars+=n;seconds+=elapsed;step+=1
                append(folder/'training_log.jsonl',{'step':step,'epoch':epoch,'loss':float(loss.detach()),'query_loss':float(qc.detach()),'evidence_loss':float(ec.detach()),'gradient_norm':float(norm),'examples':32,'auxiliary_query_labels':32,'input_chars':n,'seconds':elapsed,'order_sha256':h,'original_pairs':16,'augmented_pairs':16})
                if step%80==0:status('training',run=name,step=step,total_steps=960)
            if epoch in [6,12]:
                cp=folder/'checkpoints'/f'epoch_{epoch:02d}';cp.mkdir(parents=True);assert finite(model.state_dict()) and finite(opt.state_dict())
                torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':schedule.state_dict(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                  'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'epoch':epoch,'step':step,'exposures':32*step,'protocol_sha256':ph,'input_manifest_sha256':mh,'order_sha256':h},cp/'checkpoint.pt')
                write(cp/'manifest.json',{'sha256':sha(cp/'checkpoint.pt'),'bytes':(cp/'checkpoint.pt').stat().st_size,'step':step,'exposures':32*step})
        np.savez(folder/'weights.npz',**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})
        write(folder/'completed.json',{'run':name,'updates':step,'exposures':step*32,'train_seconds':seconds,'train_input_chars':chars})
        summaries.append(json.loads((folder/'completed.json').read_text()));del model,opt,schedule;gc.collect();torch.cuda.empty_cache()
    # Select all12 modes only on dev before producing any test predictions.
    mode_info={};component={};arrays_by={};dev=sets['dev']+sets['new_dev']
    def scores_for(q,e,routing,gate):
        pp=e[np.arange(len(q)),q.argmax(-1)] if routing=='hard' else (q[:,:,None]*e).sum(1)
        pp=pp.copy();pp[q.max(-1)<gate]=1/3
        return {pair_key(p):v.tolist() for p,v in zip(ep,pp)}
    for seed in protocol['seeds']:
        for rep,tag,source in [('raw','RAW',ROOT/f'baseline_s{seed}/weights.npz'),('language','LANG',out/f'LANG_s{seed}/weights.npz')]:
            status('development_selection',run=f'{tag}_s{seed}')
            with np.load(source,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
            arrays_by[tag,seed]=arrays;model=Typed('mlp').cuda();model.load_state_dict({k:torch.from_numpy(v).cuda() for k,v in arrays.items()});model.eval();qs=[];es=[]
            with torch.inference_mode():
                for off in range(0,len(ep),256):
                    q,e=model.components(*(v[off:off+256] for v in ex[rep]));qs.append(q.softmax(-1).cpu().numpy());es.append(e.softmax(-1).cpu().numpy())
            q=np.concatenate(qs);e=np.concatenate(es);assert np.isfinite(q).all() and np.isfinite(e).all();component[tag,seed]=(q,e)
            for routing in ['soft','hard']:
                name=f'{tag}_{routing.upper()}_s{seed}';folder=out/name;folder.mkdir();grid=[]
                for gate in [0.,.4,.6]:
                    scores=scores_for(q,e,routing,gate)
                    for threshold in [.5,.6,.7,.8,.9]:
                        runtime=CachedStudent(scores,threshold);grid.append({'query_gate':gate,'threshold':threshold,**scene_metrics(dev,[runtime.review(visible_input(r)) for r in dev])})
                best=max(grid,key=lambda v:(-v['unknown_to_established']-v['unknown_to_not_established'],v['macro_f1'],v['citation_f1'],-v['query_gate'],-v['threshold']))
                info={'version':VERSION,'architecture':'mlp','kinds':list(KINDS),'labels':list(LABELS),'representation':rep,'routing':routing,'query_gate':best['query_gate'],'threshold':best['threshold'],'mode':'shadow','decision_authority':False}
                write(folder/'selection_before_tests.json',{'selected':info,'grid':grid,'tests_used':False,'fixed_epoch':12});mode_info[name]=info
            frozen=json.loads((ROOT/f'baseline_s{seed}/student_config.json').read_text())
            if tag=='RAW':mode_info[f'FROZEN_s{seed}']={'version':VERSION,'architecture':'mlp','kinds':list(KINDS),'labels':list(LABELS),'representation':'raw','routing':'soft','query_gate':0.,'threshold':frozen['threshold'],'mode':'shadow','decision_authority':False}
            del model;gc.collect();torch.cuda.empty_cache()
    write(out/'all_selections_locked_before_tests.json',{'modes':mode_info,'time':time.time(),'tests_used':False})
    for name,info in mode_info.items():
        status('fixed_evaluation',run=name);begin=time.monotonic();seed=int(name.rsplit('_s',1)[1]);tag='RAW' if info['representation']=='raw' else 'LANG';q,e=component[tag,seed]
        scores=scores_for(q,e,info['routing'],info['query_gate']);runtime=CachedStudent(scores,info['threshold']);folder=out/name;folder.mkdir(exist_ok=True);metrics={}
        for split,rows in sets.items():
            pred=[runtime.review(visible_input(r)) for r in rows];write(folder/(split+'_predictions.json'),[{'id':r['id'],**p} for r,p in zip(rows,pred)])
            metrics[split]=scene_metrics(rows,pred);metrics[split]['by_axis']={axis:scene_metrics([r for r in rows if r['axis']==axis],[p for r,p in zip(rows,pred) if r['axis']==axis]) for axis in sorted({r['axis'] for r in rows})}
        dest=folder/'model';dest.mkdir();np.savez(dest/'weights.npz',**arrays_by[tag,seed]);write(dest/'student_config.json',info);write(dest/'manifest.json',{p.name:sha(p) for p in dest.iterdir()})
        write(folder/'pair_cpu_reference.json',[{'input':p,'scores':scores[pair_key(p)]} for p in ep[:128]])
        write(folder/'metrics.json',{'run':name,'metrics':metrics,'evaluation_seconds':time.monotonic()-begin,'optimizer_updates_for_variant':0})
    status('checkpoint_readonly_cpu_audit');checks=[]
    for p in sorted(out.glob('*/checkpoints/epoch_*/checkpoint.pt')):
        m=json.loads((p.parent/'manifest.json').read_text());c=torch.load(p,map_location='cpu',weights_only=False)
        assert sha(p)==m['sha256'] and finite(c['model']) and finite(c['optimizer']) and c['protocol_sha256']==ph and c['input_manifest_sha256']==mh
        assert c['step']==m['step']==c['epoch']*80 and c['exposures']==m['exposures']==32*c['step']
        assert all(c[k] is not None for k in ['torch_rng','cuda_rng','python_rng','numpy_rng','scheduler']);checks.append({'path':p.relative_to(ROOT).as_posix(),**m,'all_passed':True})
    assert len(checks)==6;write(out/'checkpoint_audit.json',{'all_6_passed':True,'checkpoints':checks,'optimizer_updates':0})
    write(out/'summary.json',{'runs':summaries,'total_updates':2880,'total_exposures':92160,'evaluation_variants':15,'prediction_sets':120,'production_promotion':False})
    write(out/'completed.json',{'status':'completed','runs':list(mode_info),'completed_unix':time.time()});status('completed');(out/'run.lock').unlink()
if __name__=='__main__':
    if (ROOT/'outputs/run_origin.json').exists() or (ROOT/'outputs/run.lock').exists():raise RuntimeError('Existing R6 preserved; no restart')
    try:run()
    except BaseException as e:
        write(ROOT/'outputs/failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc(),'time':time.time()});write(ROOT/'outputs/status.json',{'phase':'failed_evidence_preserved','updated_unix':time.time()});raise
