"""Bounded R8 CUDA head training and exact three-action policy optimization."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import copy,gc,hashlib,json,random,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.student_round8_model import Head,objective,teacher_reward
from scripts.run_gpu_student_20261009 import CachedStudent,scene_metrics,pair_key,sha,write,append
from scripts.run_student_round4_20261010 import finite,datasets as base_sets
from src.evidence_reasoning.local_field_student import digest,visible_input,LABELS
from src.evidence_reasoning.semantic_reward_student import FrozenEncoder,encoder_text,lexical,VERSION

def order_epoch(pools,seed,epoch):
    rng=np.random.default_rng(seed+1000*epoch);grouped=[];offset=0
    for pool in pools:
        groups=sorted({p['group'] for p in pool});by={(g,c):[i+offset for i,p in enumerate(pool) if p['group']==g and p['label']==c] for g in groups for c in range(3)}
        assert all(by.values());grouped.append((groups,by));offset+=len(pool)
    batches=[]
    for _ in range(80):
        b=[]
        for source,counts in enumerate([[8,4,4],[4,2,2],[2,1,1],[2,1,1]]):
            groups,by=grouped[source]
            for c,n in enumerate(counts):
                for _ in range(n):
                    g=groups[int(rng.integers(len(groups)))];v=by[g,c];b.append(v[int(rng.integers(len(v)))])
        rng.shuffle(b);assert len(b)==32;batches.append(b)
    return batches

def datasets():
    sets=base_sets()|{k:json.loads((ROOT/f'data/{k}.json').read_text('utf-8')) for k in ['r4_regression','r5_regression','r6_regression','r7_regression']}
    rows=json.loads((ROOT/'data/new_scenes.json').read_text('utf-8'))
    return sets|{'new_test_short':[r for r in rows if r['split']=='test_short']}

def run():
    out=ROOT/'outputs';out.mkdir(exist_ok=True)
    fd=os.open(out/'run.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    p=json.loads((ROOT/'protocol.json').read_text('utf-8'));ph=sha(ROOT/'protocol.json');mh=sha(ROOT/'input_manifest.json');start=time.time();deadline=start+7200
    write(out/'run_origin.json',{'pid':os.getpid(),'started_unix':start,'deadline_unix':deadline,'protocol_sha256':ph,'input_manifest_sha256':mh})
    def status(phase,**kw):
        if time.time()>deadline:raise TimeoutError('R8 fixed2h deadline; no extension')
        write(out/'status.json',{'phase':phase,'updated_unix':time.time(),'deadline_unix':deadline,'pid':os.getpid(),**kw})
    status('verifying_inputs')
    for n,e in json.loads((ROOT/'input_manifest.json').read_text('utf-8')).items():
        f=(ROOT/n).resolve();assert f.is_relative_to(ROOT.resolve()) and sha(f)==e['sha256'] and f.stat().st_size==e['bytes']
    assert torch.cuda.is_available();torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    write(out/'environment.json',{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'transformers':__import__('transformers').__version__,'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,'FP32':True,'new_teacher_calls':0,'encoder_updates':0})
    old=json.loads((ROOT/'data/fit.json').read_text('utf-8'));pools=[old['original'],old['recombined']]+[json.loads((ROOT/f'data/{n}.json').read_text('utf-8')) for n in ['contrast_pairs','language_pairs']];fit=sum(pools,[]);sets=datasets()
    assert not {x['group'] for x in fit}&{r['family'] for rows in sets.values() for r in rows}
    epmap={}
    for rows in sets.values():
        for r in rows:
            for q in r['fields'].values():
                for e in r['evidence']:
                    x={'query':q,'text':e['text'],'scope':r['scope']};epmap[pair_key(x)]=x
    ep=list(epmap.values());allpairs=fit+ep
    assert sha(ROOT/'data/teacher_fit_scores.jsonl')==p['teacher_targets_sha256']
    records={r['id']:r for r in [json.loads(s) for s in (ROOT/'data/teacher_fit_scores.jsonl').read_text('utf-8').splitlines()]}
    assert set(records)=={x['id'] for x in old['original']}
    tr,mask,gate=teacher_reward(records,fit);write(out/'teacher_reward_gate.json',gate)
    arrays_q,arrays_x=lexical(allpairs);q=torch.from_numpy(arrays_q).cuda();lex=torch.from_numpy(arrays_x).cuda();del arrays_q,arrays_x
    status('frozen_encoder_loading');enc=FrozenEncoder(ROOT/'encoder',p['encoder_sha256'],'cuda');features={};encoding_stats={}
    for rep in ['evidence','joint']:
        status('frozen_language_encoding',representation=rep,pairs=len(allpairs))
        embedding=enc.encode([encoder_text(x,rep) for x in allpairs]);features[rep]=lex.clone();features[rep][:,4106:]=torch.from_numpy(embedding).cuda();encoding_stats[rep]=enc.last_stats
    del enc;gc.collect();torch.cuda.empty_cache();features['lexical']=lex
    labels=torch.tensor([x['label'] for x in fit],device='cuda');types=torch.tensor([x['query_kind'] for x in fit],device='cuda');teacher=torch.from_numpy(tr).cuda();mask=torch.from_numpy(mask).cuda()
    write(out/'data_audit.json',{'pool_sizes':list(map(len,pools)),'split_sizes':{k:len(v) for k,v in sets.items()},'eval_unique_pairs':len(ep),'encoding':encoding_stats,'training_examples_are_repeated_synthetic_facts':True,'no_truncation':True,'encoder_optimizer_updates':0})
    trained=[];weights={};refs={};actual_updates=0
    def model_hash(m):return hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes() for t in m.state_dict().values())).hexdigest()
    def fresh(seed):
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        with np.load(ROOT/f'query_source_s{seed}/weights.npz',allow_pickle=False) as z:query={k:z[k].copy() for k in z.files if k.startswith('question.')}
        m=Head(query).cuda();opt=torch.optim.AdamW(m.parameters(),lr=.01,weight_decay=.01)
        sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min((s+1)/96,max(0,(960-s)/864)))
        return m,opt,sch
    def train(name,seed,rep,arm,first,last,initial=None):
        nonlocal actual_updates
        folder=out/name;folder.mkdir();status('training_initialization',run=name);m,opt,sch=fresh(seed)
        if initial:
            m.load_state_dict(initial['model']);opt.load_state_dict(copy.deepcopy(initial['optimizer']));sch.load_state_dict(initial['scheduler']);torch.set_rng_state(initial['torch_rng']);torch.cuda.set_rng_state_all(initial['cuda_rng']);random.setstate(initial['python_rng']);np.random.set_state(initial['numpy_rng'])
        ih=model_hash(m);write(folder/'initialization.json',{'sha256':ih,'warm_parent_sha256':initial['model_sha256'] if initial else None,'seed':seed,'parameters':sum(t.numel() for t in m.parameters()),'encoder_frozen':True})
        if arm not in ['EVID_VRL','EVID_MRL']:ref=None
        else:ref=refs[seed]
        def loss_fn(ids):
            qc,ec=m.components(q[ids],features[rep][ids]);logits=ec[torch.arange(len(ids),device='cuda'),types[ids]]
            loss,stats=objective(logits,labels[ids],arm,None if ref is None else ref[ids],teacher[ids]);ql=F.cross_entropy(qc,types[ids]);return loss+ql,stats|{'query_ce':float(ql.detach())}
        longest=max(range(len(fit)),key=lambda i:len(fit[i]['query'])+len(fit[i]['text']));loss,stats=loss_fn([longest]);loss.backward();assert torch.isfinite(loss) and all(t.grad is None or torch.isfinite(t.grad).all() for t in m.parameters());opt.zero_grad(set_to_none=True)
        write(folder/'backward_preflight.json',{'finite':True,'optimizer_updates':0,'input_chars':len(fit[longest]['query'])+len(fit[longest]['text']),'loss':float(loss.detach()),'objective':stats})
        seconds=0.;step=first*80;initial_step=step;snapshot=None
        for epoch in range(first+1,last+1):
            order=order_epoch(pools,seed,epoch);oh=digest(order)
            for ids in order:
                if time.time()>deadline:raise TimeoutError('R8 deadline')
                begin=time.monotonic();opt.zero_grad(set_to_none=True);loss,stats=loss_fn(ids);assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step();sch.step();torch.cuda.synchronize();elapsed=time.monotonic()-begin;seconds+=elapsed;step+=1;actual_updates+=1
                append(folder/'training_log.jsonl',{'step':step,'epoch':epoch,'loss':float(loss.detach()),'gradient_norm':float(norm),'examples':32,'auxiliary_query_labels':32,'seconds':elapsed,'order_sha256':oh,'teacher_admitted_pairs':int(mask[ids].sum()) if arm=='EVID_MRL' else 0,**stats})
            status('training',run=name,step=step,total_steps=last*80,actual_total_updates=actual_updates)
            if epoch in [(first+last)//2,last]:
                cp=folder/'checkpoints'/f'epoch_{epoch:02d}';cp.mkdir(parents=True)
                assert finite(m.state_dict()) and finite(opt.state_dict())
                snapshot={'model':{k:v.detach().cpu().clone() for k,v in m.state_dict().items()},'optimizer':copy.deepcopy(opt.state_dict()),'scheduler':copy.deepcopy(sch.state_dict()),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'epoch':epoch,'step':step,'exposures':32*step,'new_updates_this_stage':step-initial_step,'parent_warm_checkpoint':None if initial is None else initial['checkpoint_path'],'model_sha256':model_hash(m),'frozen_encoder_files':p['encoder_sha256'],'protocol_sha256':ph,'input_manifest_sha256':mh,'order_sha256':oh}
                torch.save(snapshot,cp/'checkpoint.pt');write(cp/'manifest.json',{'sha256':sha(cp/'checkpoint.pt'),'bytes':(cp/'checkpoint.pt').stat().st_size,'step':step,'exposures':32*step});snapshot['checkpoint_path']=cp.relative_to(ROOT).as_posix()
        if arm=='WARM':
            with torch.inference_mode():
                refs[seed]=m.components(q[:len(fit)],features[rep][:len(fit)])[1][torch.arange(len(fit),device='cuda'),types].detach()
        else:weights[name]={k:v.detach().cpu().numpy().copy() for k,v in m.state_dict().items()}
        row={'run':name,'effective_step':step,'new_updates':step-initial_step,'new_exposures':32*(step-initial_step),'train_seconds':seconds,'warm_parent':initial is not None};write(folder/'completed.json',row);trained.append(row);del m,opt,sch;gc.collect();torch.cuda.empty_cache();return snapshot
    for seed in p['seeds']:
        for arm,rep in [('LEX_SFT','lexical'),('JOINT_SFT','joint')]:train(f'{arm}_s{seed}',seed,rep,arm,0,12)
        warm=train(f'WARM_s{seed}',seed,'evidence','WARM',0,6)
        for arm in ['EVID_SFT','EVID_VRL','EVID_MRL']:
            if arm=='EVID_MRL' and not gate['eligible']:continue
            train(f'{arm}_s{seed}',seed,'evidence',arm,6,12,warm)
        del warm;gc.collect();torch.cuda.empty_cache()
    assert actual_updates<=p['max_total_updates']
    mode_info={};scores_by={};dev=sets['dev']+sets['new_dev']
    for name,a in weights.items():
        status('development_selection',run=name);seed=int(name.rsplit('_s',1)[1]);rep='lexical' if name.startswith('LEX') else 'joint' if name.startswith('JOINT') else 'evidence';m,_,_=fresh(seed);m.load_state_dict({k:torch.from_numpy(v) for k,v in a.items()});m.eval()
        with torch.inference_mode():
            qq,ee=m.components(q[len(fit):],features[rep][len(fit):]);qq=qq.softmax(-1).cpu().numpy();ee=ee.softmax(-1).cpu().numpy();pp=ee[np.arange(len(ep)),qq.argmax(-1)]
        grid=[]
        for g in [0.,.4,.6]:
            z=pp.copy();z[qq.max(-1)<g]=1/3;scores={pair_key(x):v.tolist() for x,v in zip(ep,z)}
            for th in [.5,.6,.7,.8,.9]:
                rt=CachedStudent(scores,th);grid.append({'query_gate':g,'threshold':th,**scene_metrics(dev,[rt.review(visible_input(r)) for r in dev])})
        best=max(grid,key=lambda v:(-v['unknown_to_established']-v['unknown_to_not_established'],v['macro_f1'],v['citation_f1'],-v['query_gate'],-v['threshold']))
        info={'version':VERSION,'representation':rep,'query_gate':best['query_gate'],'threshold':best['threshold'],'mode':'shadow','decision_authority':False,'encoder_sha256':p['encoder_sha256'],'encoder_shared_relative_path':'encoder','arm':name.rsplit('_s',1)[0]}
        pp[qq.max(-1)<best['query_gate']]=1/3;scores_by[name]={pair_key(x):v.tolist() for x,v in zip(ep,pp)};mode_info[name]=info
        write(out/name/'selection_before_tests.json',{'selected':info,'grid':grid,'tests_used':False,'fixed_effective_step':960});del m
    write(out/'all_selections_locked_before_tests.json',{'modes':mode_info,'time':time.time(),'tests_used':False})
    for name,info in mode_info.items():
        status('fixed_evaluation',run=name);begin=time.monotonic();scores=scores_by[name];runtime=CachedStudent(scores,info['threshold']);folder=out/name;metrics={}
        for split,rows in sets.items():
            pred=[runtime.review(visible_input(r)) for r in rows];write(folder/(split+'_predictions.json'),[{'id':r['id'],**v} for r,v in zip(rows,pred)])
            metrics[split]=scene_metrics(rows,pred);metrics[split]['by_axis']={a:scene_metrics([r for r in rows if r['axis']==a],[v for r,v in zip(rows,pred) if r['axis']==a]) for a in sorted({r['axis'] for r in rows})}
        dest=folder/'model';dest.mkdir();np.savez(dest/'weights.npz',**weights[name]);write(dest/'student_config.json',info);write(dest/'manifest.json',{x.name:sha(x) for x in dest.iterdir()})
        # Include new short, long, historical and each relation; no quality fit.
        reference_pairs=[]
        for split in ['new_dev','new_test','new_test_short','test']:
            row=sets[split][0]
            for query in row['fields'].values():
                for ev in row['evidence']:reference_pairs.append({'query':query,'text':ev['text'],'scope':row['scope']})
        write(folder/'pair_cpu_reference.json',[{'input':x,'scores':scores[pair_key(x)]} for x in reference_pairs]);write(folder/'metrics.json',{'run':name,'metrics':metrics,'evaluation_seconds':time.monotonic()-begin})
    status('checkpoint_readonly_cpu_audit');checks=[]
    for cp in sorted(out.glob('*/checkpoints/epoch_*/checkpoint.pt')):
        m=json.loads((cp.parent/'manifest.json').read_text());c=torch.load(cp,map_location='cpu',weights_only=False)
        assert sha(cp)==m['sha256'] and finite(c['model']) and finite(c['optimizer']) and c['protocol_sha256']==ph and c['input_manifest_sha256']==mh
        assert c['step']==m['step']==c['epoch']*80 and c['exposures']==m['exposures']==32*c['step']
        assert all(c[k] is not None for k in ['torch_rng','cuda_rng','python_rng','numpy_rng','scheduler']);checks.append({'path':cp.relative_to(ROOT).as_posix(),**m,'all_passed':True})
    expected=36 if gate['eligible'] else 30;assert len(checks)==expected
    write(out/'checkpoint_audit.json',{'all_passed':True,'checkpoints':checks,'optimizer_updates':0,'shared_encoder':p['encoder_sha256']})
    write(out/'summary.json',{'runs':trained,'total_updates':actual_updates,'total_exposures':actual_updates*32,'auxiliary_query_labels':actual_updates*32,'evaluation_variants':len(mode_info),'prediction_sets':len(mode_info)*len(sets),'teacher_reward_admitted':gate['eligible'],'production_promotion':False})
    write(out/'completed.json',{'status':'completed','runs':list(mode_info),'completed_unix':time.time()});status('completed');(out/'run.lock').unlink()
if __name__=='__main__':
    if (ROOT/'outputs/run_origin.json').exists() or (ROOT/'outputs/run.lock').exists():raise RuntimeError('Existing R8 preserved')
    try:run()
    except BaseException as e:
        write(ROOT/'outputs/failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc(),'time':time.time()});write(ROOT/'outputs/status.json',{'phase':'failed_evidence_preserved','updated_unix':time.time()});raise
