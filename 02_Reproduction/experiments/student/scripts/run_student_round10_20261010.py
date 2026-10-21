"""Bounded R10 pooling comparison and separate current-clause selector training."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import gc,hashlib,json,random,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.student_round8_model import Head
from scripts.run_gpu_student_20261009 import CachedStudent,scene_metrics,pair_key,sha,write,append
from scripts.run_student_round4_20261010 import finite
from src.evidence_reasoning.semantic_reward_student import SemanticRewardStudent,encoder_text,lexical
from src.evidence_reasoning.adapted_language_student import install_lora,adapter_state,load_adapter
from src.evidence_reasoning.clause_language_student import ClauseEncoder,aligned_text,token_raw_offsets,token_targets,decode_spans,VERSION
from src.evidence_reasoning.local_field_student import digest,visible_input,verify_anchor
from src.evidence_reasoning.typed_field_student import softmax

REGRESSIONS=['r2_regression','r3_regression','r4_regression','r5_regression','r6_regression','r7_regression','r8_regression','r8_regression_short','r9_regression','r9_regression_short','r9_regression_reordered']
def datasets(root=ROOT):
    old=json.loads((root/'data/historical.json').read_text('utf-8'));new=json.loads((root/'data/new_scenes.json').read_text('utf-8'))
    return {s:[r for r in old if r['split']==s] for s in ['dev','test']}|{s:json.loads((root/f'data/{s}.json').read_text('utf-8')) for s in REGRESSIONS}|{'new_'+s:[r for r in new if r['split']==s] for s in ['dev','test','test_short','test_reordered']}

def relation_orders(pools,seed):
    rng=np.random.default_rng(seed);banks=[];offset=0
    for pool in pools:
        groups=sorted({p['group'] for p in pool});ix={(g,c):[i+offset for i,p in enumerate(pool) if p['group']==g and p['label']==c] for g in groups for c in range(3)}
        assert all(ix.values());banks.append((groups,ix));offset+=len(pool)
    result=[]
    for _ in range(960):
        ids=[]
        for (groups,ix),counts in zip(banks,[[4,2,2],[2,1,1],[1,1,0],[1,0,1],[2,1,1],[6,3,3]]):
            for c,n in enumerate(counts):
                for _ in range(n):
                    g=groups[int(rng.integers(len(groups)))];v=ix[g,c];ids.append(v[int(rng.integers(len(v)))])
        rng.shuffle(ids);result.append(ids)
    return result

def selector_orders(paragraphs,seed):
    rng=np.random.default_rng(seed);groups=sorted({p['group'] for p in paragraphs})
    kind=lambda p:p['spans'][0]['class'] if p['spans'] else 0
    banks={(g,c):[i for i,p in enumerate(paragraphs) if p['group']==g and kind(p)==c] for g in groups for c in range(3)}
    assert all(banks.values());result=[]
    for _ in range(960):
        ids=[]
        for c,n in enumerate([12,12,8]):
            for _ in range(n):
                g=groups[int(rng.integers(len(groups)))];v=banks[g,c];ids.append(v[int(rng.integers(len(v)))])
        rng.shuffle(ids);result.append(ids)
    return result

def selection_scores(pairs,selected,arrays,query_gate):
    changed=[p|{'text':'。'.join(s['text'] for s in selected[digest({'text':p['text'],'scope':p['scope']})])} for p in pairs]
    q,x=lexical(changed);qp=softmax(q@arrays['question.weight'].T+arrays['question.bias'])
    h=np.maximum(x@arrays['hidden.weight'].T+arrays['hidden.bias'],0);ep=softmax((h@arrays['evidence.weight'].T+arrays['evidence.bias']).reshape(-1,4,3))
    scores=ep[np.arange(len(pairs)),qp.argmax(-1)].copy();scores[qp.max(-1)<query_gate]=1/3
    for i,p in enumerate(changed):
        if not p['text']:scores[i]=[1,0,0]
    return scores

class SelectionCached(CachedStudent):
    def __init__(self,scores,threshold,selected):super().__init__(scores,threshold);self.selected=selected
    def review(self,payload):
        result=super().review(payload)
        if self.selected is not None:
            result['selected_evidence']={e['id']:self.selected.get(digest({'text':e['text'],'scope':payload['scope']}),[]) for e in payload['evidence']}
            for e in payload['evidence']:
                for s in result['selected_evidence'][e['id']]:assert verify_anchor({e['id']:e['text']},{'evidence_id':e['id'],**s})
        return result

def span_metrics(rows,preds):
    tp=fp=fn=exact=total=0
    for row,p in zip(rows,preds):
        for e in row['evidence']:
            gold={(i,s['class']) for s in row['span_gold'][e['id']] for i in range(s['start'],s['end'])}
            got={(i,s['class']) for s in p['selected_evidence'][e['id']] for i in range(s['start'],s['end'])}
            tp+=len(gold&got);fp+=len(got-gold);fn+=len(gold-got);exact+=int(gold==got);total+=1
    return {'character_tp':tp,'character_fp':fp,'character_fn':fn,'character_class_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,'paragraph_exact':exact,'paragraphs':total,'repeated_scene_evidence_not_independent':True}

def run():
    out=ROOT/'outputs';out.mkdir(exist_ok=True);fd=os.open(out/'run.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    protocol=json.loads((ROOT/'protocol.json').read_text('utf-8'));ph=sha(ROOT/'protocol.json');mh=sha(ROOT/'input_manifest.json');start=time.time();deadline=start+7200
    write(out/'run_origin.json',{'pid':os.getpid(),'started_unix':start,'deadline_unix':deadline,'protocol_sha256':ph,'input_manifest_sha256':mh})
    def status(phase,**kw):
        if time.time()>deadline:raise TimeoutError('Fixed R10 2h deadline; no extension')
        write(out/'status.json',{'phase':phase,'pid':os.getpid(),'updated_unix':time.time(),'deadline_unix':deadline,**kw})
    for n,e in json.loads((ROOT/'input_manifest.json').read_text('utf-8')).items():
        p=(ROOT/n).resolve();assert p.is_relative_to(ROOT.resolve()) and p.stat().st_size==e['bytes'] and sha(p)==e['sha256']
    assert torch.cuda.is_available();torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    write(out/'environment.json',{'torch':torch.__version__,'python':sys.version,'numpy':np.__version__,'transformers':__import__('transformers').__version__,'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,'fp32':True,'teacher_calls':0})
    status('preparing_features');fitdata=json.loads((ROOT/'data/fit.json').read_text('utf-8'));sets=datasets()
    pools=[fitdata['original'],fitdata['recombined']]+[json.loads((ROOT/f'data/{n}.json').read_text('utf-8')) for n in ['contrast_pairs','language_pairs','r9_training_pairs','long_pairs']];fit=sum(pools,[])
    paragraphs=json.loads((ROOT/'data/selector_paragraphs.json').read_text('utf-8'))
    assert not ({p['group'] for p in fit}|{p['group'] for p in paragraphs})&{r['family'] for rr in sets.values() for r in rr}
    pairmap={}
    def register(p):
        x={k:p[k] for k in ['query','text','scope']};key=pair_key(x);pairmap.setdefault(key,x);return key
    anchors=[register(p) for p in fit];positive=[register(p.get('positive',p)) for p in fit];negative=[register(p.get('negative',p)) for p in fit]
    for rr in sets.values():
        for r in rr:
            for q in r['fields'].values():
                for e in r['evidence']:register({'query':q,'text':e['text'],'scope':r['scope']})
    pairlist=list(pairmap.values());keys=list(pairmap);ix={k:i for i,k in enumerate(keys)};aidx=np.asarray([ix[k] for k in anchors]);pidx=np.asarray([ix[k] for k in positive]);nidx=np.asarray([ix[k] for k in negative]);del pairmap
    qa,xa=lexical(pairlist);q=torch.from_numpy(qa).cuda();x=torch.from_numpy(xa).cuda();del qa,xa
    enc0=ClauseEncoder(ROOT/'encoder',protocol['encoder_sha256'],'cpu');tokenizer=enc0.tokenizer
    texts=list(dict.fromkeys(encoder_text(p,'evidence') for p in pairlist));textix={t:i for i,t in enumerate(texts)};pt=[textix[encoder_text(p,'evidence')] for p in pairlist]
    tok=tokenizer(texts,truncation=False,padding=False);lengths=[len(t) for t in tok['input_ids']];assert 0<max(lengths)<=512;tokens=[{k:v[i] for k,v in tok.items()} for i in range(len(texts))]
    stok=tokenizer([aligned_text(p)[0] for p in paragraphs],truncation=False,padding=False,return_offsets_mapping=True);sraw=[token_raw_offsets(p,o) for p,o in zip(paragraphs,stok.pop('offset_mapping'))]
    slabels=[token_targets(o,p['spans']) for o,p in zip(sraw,paragraphs)];stokens=[{k:v[i] for k,v in stok.items()} for i in range(len(paragraphs))];assert max(len(v) for v in slabels)<=512;del tok,stok,enc0;gc.collect()
    labels=torch.tensor([p['label'] for p in fit],device='cuda');kinds=torch.tensor([p['query_kind'] for p in fit],device='cuda');mask=np.asarray([bool(p.get('contrast_mask')) for p in fit]);neglabels=torch.tensor([p.get('negative_label',p['label']) for p in fit],device='cuda')
    write(out/'data_audit.json',{'pool_sizes':list(map(len,pools)),'selector_paragraphs':len(paragraphs),'split_sizes':{k:len(v) for k,v in sets.items()},'unique_pairs':len(pairlist),'unique_texts':len(texts),'max_encoder_tokens':max(lengths),'max_selector_tokens':max(map(len,slabels)),'no_truncation':True,'all_holdouts_synthetic':True})
    def fresh(seed,arm):
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        enc=ClauseEncoder(ROOT/'encoder',protocol['encoder_sha256'],'cuda');enc.pooling='mean' if arm=='MEAN' else 'cls';targets=install_lora(enc.model);enc.model.eval()
        with np.load(ROOT/f'head_source_s{seed}/weights.npz',allow_pickle=False) as z:values={k:z[k].copy() for k in z.files}
        head=Head(values).cuda();head.load_state_dict({k:torch.from_numpy(v) for k,v in values.items()});selector=torch.nn.Linear(512,3).cuda()
        head.requires_grad_(arm!='SELECT');selector.requires_grad_(arm=='SELECT')
        pars=list(selector.parameters()) if arm=='SELECT' else list(head.parameters())
        opt=torch.optim.AdamW([{'params':pars,'lr':.001},{'params':[v for v in enc.model.parameters() if v.requires_grad],'lr':.0002}],weight_decay=.01)
        sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min((s+1)/96,max(0,(960-s)/864)))
        return enc,head,selector,opt,sch,targets
    def statehash(head,enc,selector):return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in list(head.state_dict().values())+list(adapter_state(enc.model).values())+list(selector.state_dict().values()))).hexdigest()
    def loss_for(enc,head,selector,arm,ids):
        if arm=='SELECT':
            rows=[stokens[i] for i in ids];batch={k:v.cuda() for k,v in tokenizer.pad(rows,return_tensors='pt').items()};ys=torch.full_like(batch['input_ids'],-100)
            for i,j in enumerate(ids):ys[i,:len(slabels[j])]=torch.tensor(slabels[j],device='cuda')
            logits=selector(enc.model(**batch).last_hidden_state.float());loss=F.cross_entropy(logits.reshape(-1,3),ys.reshape(-1),ignore_index=-100,weight=logits.new_tensor([.5,1,1]))
            return loss,{'ce':float(loss.detach()),'query_ce':0.,'CE_positions':0,'paragraph_exposures':len(ids),'relation_anchor_exposures':0,'span_token_positions':int((ys!=-100).sum()),'encoder_tokens':sum(len(v['input_ids']) for v in rows),'padded_tokens':int(batch['input_ids'].numel()),'encoded_sequences':len(rows),'auxiliary_query_labels':0}
        ids=np.asarray(ids);valid=mask[ids];indices=np.concatenate([aidx[ids],pidx[ids],nidx[ids][valid]]);N=len(ids);K=int(valid.sum())
        tids=[pt[int(i)] for i in indices];unique=list(dict.fromkeys(tids));lookup={t:i for i,t in enumerate(unique)};batch={k:v.cuda() for k,v in tokenizer.pad([tokens[t] for t in unique],return_tensors='pt').items()}
        hidden=enc.model(**batch).last_hidden_state.float();m=batch['attention_mask'].unsqueeze(-1);pooled=hidden[:,0] if arm=='CLS' else (hidden*m).sum(1)/m.sum(1)
        em=F.normalize(pooled,dim=-1)[torch.tensor([lookup[t] for t in tids],device='cuda')];xx=x[indices].clone();xx[:,4106:]=em
        qlog,e=head.components(q[indices],xx);typ=torch.cat([kinds[ids],kinds[ids],kinds[ids][valid]]);ys=torch.cat([labels[ids],labels[ids],neglabels[ids][valid]])
        ce=F.cross_entropy(e[torch.arange(len(indices),device='cuda'),typ],ys);ql=F.cross_entropy(qlog[:N],kinds[ids]);loss=ce+ql
        return loss,{'ce':float(ce.detach()),'query_ce':float(ql.detach()),'CE_positions':2*N+K,'paragraph_exposures':0,'relation_anchor_exposures':N,'span_token_positions':0,'encoder_tokens':sum(lengths[t] for t in unique),'padded_tokens':int(batch['input_ids'].numel()),'encoded_sequences':len(unique),'auxiliary_query_labels':N}
    completed=[];actual=0
    for seed in protocol['seeds']:
      for arm in protocol['arms']:
        order=selector_orders(paragraphs,seed) if arm=='SELECT' else relation_orders(pools,seed);orderhash=digest(order);name=f'{arm}_s{seed}';folder=out/name;folder.mkdir();status('initialization',run=name)
        enc,head,selector,opt,sch,targets=fresh(seed,arm);initial=statehash(head,enc,selector);iad=adapter_state(enc.model)
        basehash=lambda:hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for n,v in enc.model.named_parameters() if not n.endswith(('lora_A','lora_B')))).hexdigest()
        bh=basehash();write(folder/'initialization.json',{'sha256':initial,'seed':seed,'order_sha256':orderhash,'base_sha256':bh,'arm':arm,'head_parameters':sum(v.numel() for v in head.parameters()),'adapter_parameters':sum(v.numel() for v in iad.values()),'selector_parameters':sum(v.numel() for v in selector.parameters()),'adapter_modules':targets})
        longest=max(range(len(paragraphs)),key=lambda i:len(slabels[i])) if arm=='SELECT' else max(range(len(fit)),key=lambda i:max(lengths[pt[aidx[i]]],lengths[pt[pidx[i]]],lengths[pt[nidx[i]]]))
        loss,stats=loss_for(enc,head,selector,arm,[longest]+order[0][:3]);loss.backward();assert torch.isfinite(loss) and all(v.grad is None or torch.isfinite(v.grad).all() for g in opt.param_groups for v in g['params'])
        ag=sum(float(v.grad.abs().sum()) for n,v in enc.model.named_parameters() if n.endswith(('lora_A','lora_B')) and v.grad is not None);assert ag>0;opt.zero_grad(set_to_none=True)
        write(folder/'backward_preflight.json',{'finite':True,'zero_optimizer_updates':True,'adapter_gradient_abs_sum':ag,'longest_input_index':longest,**stats})
        totals={k:0 for k in ['CE_positions','relation_anchor_exposures','paragraph_exposures','span_token_positions','encoder_tokens','padded_tokens','encoded_sequences','auxiliary_query_labels']};seconds=0.
        for step,ids in enumerate(order,1):
            if time.time()>deadline:raise TimeoutError('R10 deadline')
            begin=time.monotonic();opt.zero_grad(set_to_none=True);loss,stats=loss_for(enc,head,selector,arm,ids);assert torch.isfinite(loss);loss.backward();pars=[v for g in opt.param_groups for v in g['params']];gn=torch.nn.utils.clip_grad_norm_(pars,1.,error_if_nonfinite=True);opt.step();sch.step();torch.cuda.synchronize();elapsed=time.monotonic()-begin;seconds+=elapsed;actual+=1
            for k in totals:totals[k]+=stats[k]
            append(folder/'training_log.jsonl',{'step':step,'loss':float(loss.detach()),'gradient_norm':float(gn),'seconds':elapsed,'batch_sha256':digest(ids),'order_sha256':orderhash,**stats})
            if step%40==0:status('training',run=name,step=step,total_steps=960,actual_total_updates=actual)
            if step in [480,960]:
                cp=folder/'checkpoints'/f'step_{step:04d}';cp.mkdir(parents=True)
                snapshot={'head':{k:v.detach().cpu().clone() for k,v in head.state_dict().items()},'selector':{k:v.detach().cpu().clone() for k,v in selector.state_dict().items()},'adapter':adapter_state(enc.model),'optimizer':opt.state_dict(),'scheduler':sch.state_dict(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'step':step,'exposures':step*32,'totals':dict(totals),'protocol_sha256':ph,'input_manifest_sha256':mh,'base_encoder_files':protocol['encoder_sha256'],'order_sha256':orderhash,'arm':arm}
                assert all(finite(snapshot[k]) for k in ['head','selector','adapter','optimizer']);torch.save(snapshot,cp/'checkpoint.pt');write(cp/'manifest.json',{'bytes':(cp/'checkpoint.pt').stat().st_size,'sha256':sha(cp/'checkpoint.pt'),'step':step,'exposures':step*32});del snapshot
        assert basehash()==bh;ad=adapter_state(enc.model);delta=max(float((ad[k]-iad[k]).abs().max()) for k in ad);assert delta>0
        dest=folder/'model';dest.mkdir()
        for n,m in [('weights',head),('selector',selector)]:np.savez(dest/(n+'.npz'),**{k:v.detach().cpu().numpy() for k,v in m.state_dict().items()})
        np.savez(dest/'adapter.npz',**{k:v.numpy() for k,v in ad.items()})
        row={'run':name,'arm':arm,'updates':960,'train_seconds':seconds,'encoder_base_unchanged':True,'adapter_maxabs_change':delta,**totals};write(folder/'completed.json',row);completed.append(row);del enc,head,selector,opt,sch,iad,ad;gc.collect();torch.cuda.empty_cache()
    assert actual==8640
    evalkeys={pair_key({'query':qv,'text':ev['text'],'scope':r['scope']}) for rr in sets.values() for r in rr for qv in r['fields'].values() for ev in r['evidence']};ei=[i for i,k in enumerate(keys) if k in evalkeys];ep=[pairlist[i] for i in ei]
    dev=sets['dev']+sets['new_dev'];configs={};bank={};costs={};selections={}
    for row in completed:
        name=row['run'];arm=row['arm'];seed=int(name.rsplit('_s',1)[1]);folder=out/name;status('development_selection',run=name);begin=time.monotonic();enc,head,selector,opt,sch,_=fresh(seed,arm)
        with np.load(folder/'model/weights.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files};head.load_state_dict({k:torch.from_numpy(v) for k,v in arrays.items()})
        with np.load(folder/'model/adapter.npz',allow_pickle=False) as z:load_adapter(enc.model,{k:torch.from_numpy(z[k].copy()) for k in z.files})
        with np.load(folder/'model/selector.npz',allow_pickle=False) as z:sw={k:z[k].copy() for k in z.files}
        enc.model.requires_grad_(False);head.eval();grid=[];candidates={}
        if arm=='SELECT':
            # Original R8 relation weights/threshold/query gate remain unchanged.
            oldcfg=json.loads((ROOT/f'head_source_s{seed}/student_config.json').read_text('utf-8'));th=oldcfg['threshold'];gate=oldcfg['query_gate']
            unique={digest({'text':p['text'],'scope':p['scope']}):p for p in ep};pr=enc.selection_probabilities(list(unique.values()),sw)
            for sg in [.5,.7,.9]:
                selected={k:decode_spans(p['text'],v['offsets'],v['probabilities'],sg) for (k,p),v in zip(unique.items(),pr)};ss=selection_scores(ep,selected,arrays,gate);dd={pair_key(p):s.tolist() for p,s in zip(ep,ss)}
                candidates[sg]=(dd,selected);rt=SelectionCached(dd,th,selected);grid.append({'query_gate':gate,'threshold':th,'span_gate':sg,**scene_metrics(dev,[rt.review(visible_input(r)) for r in dev])})
            best=max(grid,key=lambda v:(-v['unknown_to_established']-v['unknown_to_not_established'],v['macro_f1'],v['citation_f1'],-v['span_gate']))
            scores,selected=candidates[best['span_gate']]
        else:
            emb=enc.encode([encoder_text(p,'evidence') for p in ep]);xx=x[ei].clone();xx[:,4106:]=torch.from_numpy(emb).cuda()
            with torch.inference_mode():qq,ee=head.components(q[ei],xx);qp=qq.softmax(-1).cpu().numpy();ee=ee.softmax(-1).cpu().numpy();base_scores=ee[np.arange(len(ep)),qp.argmax(-1)]
            for gate in [0.,.4,.6]:
                ss=base_scores.copy();ss[qp.max(-1)<gate]=1/3;dd={pair_key(p):s.tolist() for p,s in zip(ep,ss)}
                for th in [.5,.6,.7,.8,.9]:rt=CachedStudent(dd,th);grid.append({'query_gate':gate,'threshold':th,'span_gate':None,**scene_metrics(dev,[rt.review(visible_input(r)) for r in dev])})
            best=max(grid,key=lambda v:(-v['unknown_to_established']-v['unknown_to_not_established'],v['macro_f1'],v['citation_f1'],-v['query_gate'],-v['threshold']))
            ss=base_scores.copy();ss[qp.max(-1)<best['query_gate']]=1/3;scores={pair_key(p):s.tolist() for p,s in zip(ep,ss)};selected=None;del xx
        cfg={'version':VERSION,'representation':'lexical' if arm=='SELECT' else 'evidence','query_gate':best['query_gate'],'threshold':best['threshold'],'span_gate':best['span_gate'],'encoder_sha256':protocol['encoder_sha256'],'pooling':enc.pooling,'arm':arm,'mode':'shadow','decision_authority':False}
        configs[name]=cfg;bank[name]=scores;selections[name]=selected;costs[name]={'encoding_and_selection_seconds':time.monotonic()-begin}
        write(folder/'selection_before_tests.json',{'selected':cfg,'grid':grid,'tests_used':False,'fixed_step':960});dest=folder/'model';write(dest/'student_config.json',cfg);write(dest/'manifest.json',{p.name:sha(p) for p in dest.iterdir()})
        del enc,head,selector,opt,sch,candidates;gc.collect();torch.cuda.empty_cache()
    write(out/'all_selections_locked_before_tests.json',{'modes':configs,'time':time.time(),'tests_used':False})
    def report(name,rt,extra):
        folder=out/name;metrics={};spans={};begin=time.monotonic()
        for split,rr in sets.items():
            preds=[rt.review(visible_input(r)) for r in rr];write(folder/(split+'_predictions.json'),[{'id':r['id'],**p} for r,p in zip(rr,preds)]);metrics[split]=scene_metrics(rr,preds)
            metrics[split]['by_axis']={a:scene_metrics([r for r in rr if r['axis']==a],[p for r,p in zip(rr,preds) if r['axis']==a]) for a in sorted({r['axis'] for r in rr})}
            if name.startswith('SELECT') and split.startswith('new_'):spans[split]=span_metrics(rr,preds)
        write(folder/'metrics.json',{'run':name,'metrics':metrics,'span_metrics':spans,'rule_evaluation_seconds':time.monotonic()-begin,**extra})
    for name,cfg in configs.items():
        status('fixed_evaluation',run=name);report(name,SelectionCached(bank[name],cfg['threshold'],selections[name]),costs[name]);refs=[]
        for split in ['new_dev','new_test','new_test_short','new_test_reordered','test']:
            r=sets[split][0]
            for qv in r['fields'].values():
                for e in r['evidence']:
                    inp={'query':qv,'text':e['text'],'scope':r['scope']};refs.append({'input':inp,'scores':bank[name][pair_key(inp)]})
        write(out/name/'pair_cpu_reference.json',refs)
    for seed in [17,29,43]:
        name=f'R8_FROZEN_s{seed}';(out/name).mkdir();status('frozen_R8_regression',run=name);rt=SemanticRewardStudent.load(ROOT/f'head_source_s{seed}',ROOT/'encoder');report(name,rt,{'updates':0,'threshold_reselected':False})
    status('checkpoint_audit');cps=[]
    for cp in sorted(out.glob('*/checkpoints/step_*/checkpoint.pt')):
        m=json.loads((cp.parent/'manifest.json').read_text());c=torch.load(cp,map_location='cpu',weights_only=False)
        assert cp.stat().st_size==m['bytes'] and sha(cp)==m['sha256'] and all(finite(c[k]) for k in ['head','selector','adapter','optimizer'])
        assert c['step']==m['step'] and c['exposures']==m['exposures']==32*c['step'];assert c['protocol_sha256']==ph and c['input_manifest_sha256']==mh and c['base_encoder_files']==protocol['encoder_sha256'];assert all(c[k] is not None for k in ['scheduler','python_rng','numpy_rng','torch_rng','cuda_rng']);cps.append({'path':cp.relative_to(ROOT).as_posix(),**m,'all_passed':True});del c
    assert len(cps)==18;write(out/'checkpoint_audit.json',{'all_passed':True,'entries':cps,'zero_optimizer_updates':True})
    summary={'runs':completed,'total_updates':actual,'trained_models':9,'prediction_sets':12*len(sets),'production_promotion':False}
    for k in ['CE_positions','relation_anchor_exposures','paragraph_exposures','span_token_positions','encoder_tokens','padded_tokens','encoded_sequences','auxiliary_query_labels']:summary[k]=sum(r[k] for r in completed)
    write(out/'summary.json',summary);write(out/'completed.json',{'status':'completed','runs':list(configs),'completed_unix':time.time()});status('completed');(out/'run.lock').unlink()

if __name__=='__main__':
    if (ROOT/'outputs/run_origin.json').exists():raise RuntimeError('Existing R10 preserved; no automatic restart')
    try:run()
    except BaseException as e:write(ROOT/'outputs/failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc(),'time':time.time()});write(ROOT/'outputs/status.json',{'phase':'failed_evidence_preserved','time':time.time()});raise
