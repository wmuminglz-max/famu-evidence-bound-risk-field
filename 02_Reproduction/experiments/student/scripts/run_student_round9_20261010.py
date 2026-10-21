"""Fixed R9 CUDA language-adapter/contrast comparison; never deploys."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import copy,gc,hashlib,json,random,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.student_round8_model import Head
from scripts.run_gpu_student_20261009 import CachedStudent,scene_metrics,pair_key,sha,write,append
from scripts.run_student_round4_20261010 import finite
from src.evidence_reasoning.semantic_reward_student import FrozenEncoder,SemanticRewardStudent,encoder_text,lexical
from src.evidence_reasoning.adapted_language_student import install_lora,adapter_state,load_adapter,VERSION
from src.evidence_reasoning.local_field_student import digest,visible_input

REGRESSIONS=['r2_regression','r3_regression','r4_regression','r5_regression','r6_regression','r7_regression','r8_regression','r8_regression_short']
def datasets(root=ROOT):
    old=json.loads((root/'data/historical.json').read_text('utf-8'));new=json.loads((root/'data/new_scenes.json').read_text('utf-8'))
    return {s:[r for r in old if r['split']==s] for s in ['dev','test']}|{s:json.loads((root/f'data/{s}.json').read_text('utf-8')) for s in REGRESSIONS}|{'new_'+s:[r for r in new if r['split']==s] for s in ['dev','test','test_short','test_reordered']}

def orders(pools,seed):
    rng=np.random.default_rng(seed);by=[];offset=0
    for pool in pools:
        gs=sorted({p['group'] for p in pool});ix={(g,c):[i+offset for i,p in enumerate(pool) if p['group']==g and p['label']==c] for g in gs for c in range(3)}
        assert all(ix.values());by.append((gs,ix));offset+=len(pool)
    result=[]
    for _ in range(960):
        batch=[]
        for (gs,ix),counts in zip(by,[[4,2,2],[2,1,1],[1,1,0],[1,0,1],[8,4,4]]):
            for c,n in enumerate(counts):
                for _ in range(n):
                    g=gs[int(rng.integers(len(gs)))];v=ix[g,c];batch.append(v[int(rng.integers(len(v)))])
        rng.shuffle(batch);result.append(batch)
    return result

def contrast_loss(a,p,n,mask):
    losses=1-(a*p).sum(-1)+F.relu(.2+(a*n).sum(-1)-(a*p).sum(-1))
    return (losses*mask).sum()/mask.sum().clamp_min(1)

def run():
    out=ROOT/'outputs';out.mkdir(exist_ok=True);fd=os.open(out/'run.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    protocol=json.loads((ROOT/'protocol.json').read_text('utf-8'));ph=sha(ROOT/'protocol.json');mh=sha(ROOT/'input_manifest.json');start=time.time();deadline=start+7200
    write(out/'run_origin.json',{'pid':os.getpid(),'started_unix':start,'deadline_unix':deadline,'protocol_sha256':ph,'input_manifest_sha256':mh})
    def status(phase,**kw):
        if time.time()>deadline:raise TimeoutError('Fixed R9 2h deadline reached; no extension')
        write(out/'status.json',{'phase':phase,'pid':os.getpid(),'updated_unix':time.time(),'deadline_unix':deadline,**kw})
    for rel,e in json.loads((ROOT/'input_manifest.json').read_text('utf-8')).items():
        p=(ROOT/rel).resolve();assert p.is_relative_to(ROOT.resolve()) and p.stat().st_size==e['bytes'] and sha(p)==e['sha256']
    assert torch.cuda.is_available();torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    write(out/'environment.json',{'torch':torch.__version__,'python':sys.version,'numpy':np.__version__,'transformers':__import__('transformers').__version__,'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,'fp32':True,'teacher_calls':0,'encoder_adapters_actually_train':True})
    status('preparing_features');fitdata=json.loads((ROOT/'data/fit.json').read_text('utf-8'))
    pools=[fitdata['original'],fitdata['recombined']]+[json.loads((ROOT/f'data/{n}.json').read_text('utf-8')) for n in ['contrast_pairs','language_pairs','long_pairs']];fit=sum(pools,[]);sets=datasets()
    assert not {p['group'] for p in fit}&{r['family'] for rr in sets.values() for r in rr}
    # Feature/text table contains train and evaluation inputs but no test labels.
    pairmap={}
    def register(p):
        x={k:p[k] for k in ['query','text','scope']};key=pair_key(x);pairmap.setdefault(key,x);return key
    anchors=[register(p) for p in fit];positives=[register(p.get('positive',p)) for p in fit];negatives=[register(p.get('negative',p)) for p in fit]
    for rows in sets.values():
        for r in rows:
            for q in r['fields'].values():
                for e in r['evidence']:register({'query':q,'text':e['text'],'scope':r['scope']})
    pairlist=list(pairmap.values());keys=list(pairmap);keyindex={k:i for i,k in enumerate(keys)}
    aidx=np.asarray([keyindex[k] for k in anchors]);pidx=np.asarray([keyindex[k] for k in positives]);nidx=np.asarray([keyindex[k] for k in negatives]);del pairmap
    qarr,xarr=lexical(pairlist);q=torch.from_numpy(qarr).cuda();x=torch.from_numpy(xarr).cuda();del qarr,xarr
    token_enc=FrozenEncoder(ROOT/'encoder',protocol['encoder_sha256'],'cpu');tokenizer=token_enc.tokenizer
    texts=list(dict.fromkeys(encoder_text(p,'evidence') for p in pairlist));textindex={t:i for i,t in enumerate(texts)};pairtext=[textindex[encoder_text(p,'evidence')] for p in pairlist]
    tok=tokenizer(texts,truncation=False,padding=False);lengths=[len(v) for v in tok['input_ids']];assert min(lengths)>0 and max(lengths)<=512
    tokens=[{k:v[i] for k,v in tok.items()} for i in range(len(texts))];del tok,token_enc;gc.collect()
    labels=torch.tensor([p['label'] for p in fit],device='cuda');kinds=torch.tensor([p['query_kind'] for p in fit],device='cuda');mask=torch.tensor([bool(p.get('contrast_mask')) for p in fit],device='cuda');neglabels=torch.tensor([p.get('negative_label',p['label']) for p in fit],device='cuda')
    write(out/'data_audit.json',{'pool_sizes':list(map(len,pools)),'split_sizes':{k:len(v) for k,v in sets.items()},'unique_pairs':len(pairlist),'unique_texts':len(texts),'max_encoder_tokens':max(lengths),'no_truncation':True,'pairing_is_same_authored_fact':True,'all_holdouts_synthetic':True})
    def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    def fresh(seed,arm):
        seed_all(seed);enc=FrozenEncoder(ROOT/'encoder',protocol['encoder_sha256'],'cuda');targets=install_lora(enc.model)
        enc.model.eval()
        for n,t in enc.model.named_parameters():t.requires_grad_(arm!='FROZEN' and n.endswith(('lora_A','lora_B')))
        with np.load(ROOT/f'head_source_s{seed}/weights.npz',allow_pickle=False) as z:headstate={k:z[k].copy() for k in z.files}
        head=Head(headstate).cuda();head.load_state_dict({k:torch.from_numpy(v) for k,v in headstate.items()})
        groups=[{'params':list(head.parameters()),'lr':.001}]
        if arm!='FROZEN':groups.append({'params':[v for v in enc.model.parameters() if v.requires_grad],'lr':.0002})
        opt=torch.optim.AdamW(groups,weight_decay=.01);sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min((s+1)/96,max(0,(960-s)/864)))
        return enc,head,opt,sch,targets
    def statehash(head,enc):
        return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in list(head.state_dict().values())+list(adapter_state(enc.model).values()))).hexdigest()
    def embed(enc,pairindices,grad):
        tids=[pairtext[int(i)] for i in pairindices];unique=list(dict.fromkeys(tids));lookup={t:i for i,t in enumerate(unique)}
        batch={k:v.cuda() for k,v in tokenizer.pad([tokens[t] for t in unique],return_tensors='pt').items()}
        with torch.set_grad_enabled(grad):
            hidden=enc.model(**batch).last_hidden_state.float();m=batch['attention_mask'].unsqueeze(-1);z=F.normalize((hidden*m).sum(1)/m.sum(1),dim=-1)
        return z[torch.tensor([lookup[t] for t in tids],device='cuda')],{'encoded_sequences':len(unique),'encoder_tokens':sum(lengths[t] for t in unique),'padded_tokens':int(batch['input_ids'].numel())}
    def step_loss(enc,head,arm,ids):
        ids=np.asarray(ids);valid=mask[ids].cpu().numpy();indices=np.concatenate([aidx[ids],pidx[ids],nidx[ids][valid]]);N=len(ids);K=int(valid.sum())
        em,stats=embed(enc,indices,arm!='FROZEN');xx=x[indices].clone();xx[:,4106:]=em
        qlog,e=head.components(q[indices],xx);typ=torch.cat([kinds[ids],kinds[ids],kinds[ids][valid]])
        ys=torch.cat([labels[ids],labels[ids],neglabels[ids][valid]]);logits=e[torch.arange(len(indices),device='cuda'),typ]
        ce=F.cross_entropy(logits,ys);ql=F.cross_entropy(qlog[:N],kinds[ids]);con=ce*0
        if K:con=contrast_loss(em[:N][valid],em[N:2*N][valid],em[2*N:],torch.ones(K,device='cuda'))
        loss=ce+ql+(.2*con if arm=='LANG_CONTRAST' else 0)
        return loss,stats|{'ce':float(ce.detach()),'query_ce':float(ql.detach()),'contrast':float(con.detach()),'CE_positions':2*N+K,'same_fact_compact_positions':N,'opposite_positions':K,'auxiliary_query_labels':N}
    completed=[];actual=0
    for seed in protocol['seeds']:
      order=orders(pools,seed);orderhash=digest(order)
      for arm in protocol['arms']:
        name=f'{arm}_s{seed}';folder=out/name;folder.mkdir();status('initialization',run=name)
        enc,head,opt,sch,targets=fresh(seed,arm);initialhash=statehash(head,enc);initial_adapter=adapter_state(enc.model)
        base_digest=hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes() for n,t in enc.model.named_parameters() if not n.endswith(('lora_A','lora_B')))).hexdigest()
        write(folder/'initialization.json',{'sha256':initialhash,'seed':seed,'order_sha256':orderhash,'base_sha256':base_digest,'head_parameters':sum(v.numel() for v in head.parameters()),'encoder_base_parameters':sum(v.numel() for n,v in enc.model.named_parameters() if not n.endswith(('lora_A','lora_B'))),'adapter_parameters':sum(v.numel() for v in initial_adapter.values()),'adapter_modules':targets,'encoder_adapters_trainable':arm!='FROZEN'})
        longest=max(range(len(fit)),key=lambda i:max(lengths[pairtext[aidx[i]]],lengths[pairtext[pidx[i]]],lengths[pairtext[nidx[i]]]))
        contrast_example=next(i for i,p in enumerate(fit) if p.get('contrast_mask'))
        loss,st=step_loss(enc,head,arm,[longest,contrast_example]);loss.backward();assert torch.isfinite(loss) and all(t.grad is None or torch.isfinite(t.grad).all() for t in list(head.parameters())+list(enc.model.parameters()))
        adapter_grad=sum(float(t.grad.abs().sum()) for n,t in enc.model.named_parameters() if n.endswith(('lora_A','lora_B')) and t.grad is not None)
        assert (adapter_grad==0) if arm=='FROZEN' else (adapter_grad>0)
        opt.zero_grad(set_to_none=True)
        write(folder/'backward_preflight.json',{'finite':True,'zero_optimizer_updates':True,'loss':float(loss.detach()),'adapter_gradient_abs_sum':adapter_grad,'longest_training_tokens':max(lengths[pairtext[j]] for j in [aidx[longest],pidx[longest],nidx[longest]]),**st})
        totals={'CE_positions':0,'encoder_tokens':0,'padded_tokens':0,'encoded_sequences':0,'opposite_positions':0};seconds=0.
        for step,ids in enumerate(order,1):
            if time.time()>deadline:raise TimeoutError('R9 fixed deadline')
            begin=time.monotonic();opt.zero_grad(set_to_none=True);loss,stats=step_loss(enc,head,arm,ids);assert torch.isfinite(loss);loss.backward()
            pars=[v for g in opt.param_groups for v in g['params']];gn=torch.nn.utils.clip_grad_norm_(pars,1.,error_if_nonfinite=True);opt.step();sch.step();torch.cuda.synchronize();elapsed=time.monotonic()-begin;seconds+=elapsed;actual+=1
            for k in totals:totals[k]+=stats[k]
            append(folder/'training_log.jsonl',{'step':step,'loss':float(loss.detach()),'gradient_norm':float(gn),'anchor_exposures':32,'seconds':elapsed,'batch_sha256':digest(ids),'order_sha256':orderhash,**stats})
            if step%40==0:status('training',run=name,step=step,total_steps=960,actual_total_updates=actual)
            if step in protocol['checkpoint_steps']:
                cp=folder/'checkpoints'/f'step_{step:04d}';cp.mkdir(parents=True);hs={k:v.detach().cpu().clone() for k,v in head.state_dict().items()};ad=adapter_state(enc.model)
                assert finite(hs) and finite(ad) and finite(opt.state_dict())
                snapshot={'head':hs,'adapter':ad,'optimizer':opt.state_dict(),'scheduler':sch.state_dict(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'step':step,'exposures':step*32,'totals':dict(totals),'protocol_sha256':ph,'input_manifest_sha256':mh,'base_encoder_files':protocol['encoder_sha256'],'order_sha256':orderhash,'arm':arm}
                torch.save(snapshot,cp/'checkpoint.pt');write(cp/'manifest.json',{'bytes':(cp/'checkpoint.pt').stat().st_size,'sha256':sha(cp/'checkpoint.pt'),'step':step,'exposures':step*32});del snapshot
        final_base=hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes() for n,t in enc.model.named_parameters() if not n.endswith(('lora_A','lora_B')))).hexdigest();assert final_base==base_digest
        ad=adapter_state(enc.model);difference={k:float((ad[k]-initial_adapter[k]).abs().max()) for k in ad};assert (max(difference.values())==0) if arm=='FROZEN' else (max(difference.values())>0)
        dest=folder/'model';dest.mkdir();np.savez(dest/'weights.npz',**{k:v.detach().cpu().numpy() for k,v in head.state_dict().items()});np.savez(dest/'adapter.npz',**{k:v.numpy() for k,v in ad.items()})
        row={'run':name,'updates':960,'anchor_exposures':30720,'train_seconds':seconds,'encoder_base_unchanged':True,'adapter_maxabs_change':max(difference.values()),**totals};write(folder/'completed.json',row);completed.append(row)
        del enc,head,opt,sch,initial_adapter,ad;gc.collect();torch.cuda.empty_cache()
    assert actual==8640 and sum(r['CE_positions'] for r in completed)<=protocol['max_CE_positions']
    # Endpoints fixed. No test predictions influence selection.
    evalkeys={pair_key({'query':qv,'text':ev['text'],'scope':r['scope']}) for rr in sets.values() for r in rr for qv in r['fields'].values() for ev in r['evidence']};evalindices=[i for i,k in enumerate(keys) if k in evalkeys]
    ep=[pairlist[i] for i in evalindices];dev=sets['dev']+sets['new_dev'];configs={};scorebank={};evalcost={}
    for name in [r['run'] for r in completed]:
        status('development_selection',run=name);begin=time.monotonic();seed=int(name.rsplit('_s',1)[1]);enc,head,opt,sch,_=fresh(seed,'FROZEN');folder=out/name
        with np.load(folder/'model/weights.npz',allow_pickle=False) as z:head.load_state_dict({k:torch.from_numpy(z[k].copy()) for k in z.files})
        with np.load(folder/'model/adapter.npz',allow_pickle=False) as z:load_adapter(enc.model,{k:torch.from_numpy(z[k].copy()) for k in z.files})
        enc.model.requires_grad_(False);head.eval();embedding=enc.encode([encoder_text(p,'evidence') for p in ep]);xx=x[evalindices].clone();xx[:,4106:]=torch.from_numpy(embedding).cuda()
        with torch.inference_mode():qq,ee=head.components(q[evalindices],xx);qp=qq.softmax(-1).cpu().numpy();ee=ee.softmax(-1).cpu().numpy();scores=ee[np.arange(len(ep)),qp.argmax(-1)]
        grid=[]
        for gate in [0.,.4,.6]:
            ss=scores.copy();ss[qp.max(-1)<gate]=1/3;dd={pair_key(p):s.tolist() for p,s in zip(ep,ss)}
            for th in [.5,.6,.7,.8,.9]:rt=CachedStudent(dd,th);grid.append({'query_gate':gate,'threshold':th,**scene_metrics(dev,[rt.review(visible_input(r)) for r in dev])})
        best=max(grid,key=lambda v:(-v['unknown_to_established']-v['unknown_to_not_established'],v['macro_f1'],v['citation_f1'],-v['query_gate'],-v['threshold']))
        cfg={'version':VERSION,'representation':'evidence','query_gate':best['query_gate'],'threshold':best['threshold'],'encoder_sha256':protocol['encoder_sha256'],'lora_rank':8,'lora_alpha':16,'arm':name.rsplit('_s',1)[0],'mode':'shadow','decision_authority':False}
        configs[name]=cfg;scores[qp.max(-1)<best['query_gate']]=1/3;scorebank[name]={pair_key(p):s.tolist() for p,s in zip(ep,scores)};evalcost[name]={'encoder':enc.last_stats,'encoding_and_selection_seconds':time.monotonic()-begin}
        write(folder/'selection_before_tests.json',{'selected':cfg,'grid':grid,'tests_used':False,'fixed_step':960});del enc,head,opt,sch,xx;gc.collect();torch.cuda.empty_cache()
    write(out/'all_selections_locked_before_tests.json',{'modes':configs,'time':time.time(),'tests_used':False})
    def report(name,rt,extra):
        folder=out/name;metrics={};begin=time.monotonic()
        for split,rows in sets.items():
            preds=[rt.review(visible_input(r)) for r in rows];write(folder/(split+'_predictions.json'),[{'id':r['id'],**p} for r,p in zip(rows,preds)]);metrics[split]=scene_metrics(rows,preds)
            metrics[split]['by_axis']={a:scene_metrics([r for r in rows if r['axis']==a],[p for r,p in zip(rows,preds) if r['axis']==a]) for a in sorted({r['axis'] for r in rows})}
        write(folder/'metrics.json',{'run':name,'metrics':metrics,'rule_evaluation_seconds':time.monotonic()-begin,**extra})
    for name,cfg in configs.items():
        status('fixed_evaluation',run=name);folder=out/name;scores=scorebank[name];report(name,CachedStudent(scores,cfg['threshold']),evalcost[name]);dest=folder/'model';write(dest/'student_config.json',cfg);write(dest/'manifest.json',{p.name:sha(p) for p in dest.iterdir()})
        refs=[]
        for split in ['new_dev','new_test','new_test_short','new_test_reordered','test']:
            r=sets[split][0]
            for qv in r['fields'].values():
                for e in r['evidence']:
                    inp={'query':qv,'text':e['text'],'scope':r['scope']};refs.append({'input':inp,'scores':scores[pair_key(inp)]})
        write(folder/'pair_cpu_reference.json',refs)
    for seed in protocol['seeds']:
        name=f'R8_FROZEN_s{seed}';(out/name).mkdir();status('frozen_R8_regression',run=name);base=SemanticRewardStudent.load(ROOT/f'head_source_s{seed}',ROOT/'encoder');report(name,base,{'updates':0,'threshold_reselected':False})
    status('checkpoint_audit');cps=[]
    for cp in sorted(out.glob('*/checkpoints/step_*/checkpoint.pt')):
        m=json.loads((cp.parent/'manifest.json').read_text());c=torch.load(cp,map_location='cpu',weights_only=False)
        assert cp.stat().st_size==m['bytes'] and sha(cp)==m['sha256'] and finite(c['head']) and finite(c['adapter']) and finite(c['optimizer'])
        assert c['step']==m['step'] and c['exposures']==m['exposures']==32*c['step'];assert c['protocol_sha256']==ph and c['input_manifest_sha256']==mh and c['base_encoder_files']==protocol['encoder_sha256']
        assert all(c[k] is not None for k in ['scheduler','python_rng','numpy_rng','torch_rng','cuda_rng']);cps.append({'path':cp.relative_to(ROOT).as_posix(),**m,'all_passed':True});del c
    assert len(cps)==18;write(out/'checkpoint_audit.json',{'all_passed':True,'entries':cps,'zero_optimizer_updates':True})
    write(out/'summary.json',{'runs':completed,'total_updates':actual,'total_exposures':actual*32,'CE_positions':sum(r['CE_positions'] for r in completed),'auxiliary_query_labels':actual*32,'evaluation_variants':12,'trained_models':9,'prediction_sets':12*len(sets),'production_promotion':False})
    write(out/'completed.json',{'status':'completed','runs':list(configs),'completed_unix':time.time()});status('completed');(out/'run.lock').unlink()
if __name__=='__main__':
    if (ROOT/'outputs/run_origin.json').exists():raise RuntimeError('Existing R9 preserved; no automatic restart')
    try:run()
    except BaseException as e:write(ROOT/'outputs/failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc(),'time':time.time()});write(ROOT/'outputs/status.json',{'phase':'failed_evidence_preserved','time':time.time()});raise
