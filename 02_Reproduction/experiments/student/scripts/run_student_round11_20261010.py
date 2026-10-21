"""Bounded R11 2x2 paragraph representation and training-only replay experiment."""
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
from src.evidence_reasoning.clause_language_student import aligned_text,token_raw_offsets,token_targets
from src.evidence_reasoning.local_field_student import digest,visible_input,verify_anchor
from src.evidence_reasoning.typed_field_student import softmax

from src.evidence_reasoning.continuous_span_student import SpanEncoder,selection_scores,decode,span_target,boundary_masks,VERSION,ARMS

REGRESSIONS=['r10_regression','r10_regression_short','r10_regression_reordered','r2_regression','r3_regression','r4_regression','r5_regression','r6_regression','r7_regression','r8_regression','r8_regression_short','r9_regression','r9_regression_short','r9_regression_reordered']
def datasets(root=ROOT):
    old=json.loads((root/'data/historical.json').read_text('utf-8'));new=json.loads((root/'data/new_scenes.json').read_text('utf-8'))
    return {s:[r for r in old if r['split']==s] for s in ['dev','test']}|{s:json.loads((root/f'data/{s}.json').read_text('utf-8')) for s in REGRESSIONS}|{'new_'+s:[r for r in new if r['split']==s] for s in ['dev','test','test_short','test_reordered']}

def paragraph_orders(paragraphs,seed,mode):
    rng=np.random.default_rng(seed);banks={}
    for source in ['new','r10','historical']:
        for c in range(3):
            groups={}
            for i,p in enumerate(paragraphs):
                kind=p['spans'][0]['class'] if p['spans'] else 0
                if p['source']==source and kind==c:groups.setdefault(p['group'],[]).append(i)
            if groups:banks[source,c]=groups
    plan=[('new',[12,12,8])] if mode=='NEW' else [('new',[6,4,4]),('r10',[6,4,4]),('historical',[0,4,0])]
    result=[]
    for _ in range(960):
        ids=[]
        for source,counts in plan:
            for c,n in enumerate(counts):
                if not n:continue
                bank=banks[source,c];groups=sorted(bank)
                for _ in range(n):
                    g=groups[int(rng.integers(len(groups)))];v=bank[g];ids.append(v[int(rng.integers(len(v)))])
        rng.shuffle(ids);assert len(ids)==32;result.append(ids)
    return result

class SelectionCached(CachedStudent):
    def __init__(self,scores,threshold,selected):super().__init__(scores,threshold);self.selected=selected
    def review(self,payload):
        result=super().review(payload)
        if self.selected is not None:
            result['selected_evidence']={e['id']:self.selected.get(digest({'text':e['text'],'scope':payload['scope']}),[]) for e in payload['evidence']}
            for e in payload['evidence']:
                for s in result['selected_evidence'][e['id']]:assert verify_anchor({e['id']:e['text']},{'evidence_id':e['id'],**{k:s[k] for k in ['start','end','text']}})
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
        if time.time()>deadline:raise TimeoutError('Fixed R11 2h deadline; no extension')
        write(out/'status.json',{'phase':phase,'pid':os.getpid(),'updated_unix':time.time(),'deadline_unix':deadline,**kw})
    for n,e in json.loads((ROOT/'input_manifest.json').read_text('utf-8')).items():
        p=(ROOT/n).resolve();assert p.is_relative_to(ROOT.resolve()) and p.stat().st_size==e['bytes'] and sha(p)==e['sha256']
    assert torch.cuda.is_available();torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    write(out/'environment.json',{'torch':torch.__version__,'python':sys.version,'numpy':np.__version__,'transformers':__import__('transformers').__version__,'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,'fp32':True,'teacher_calls':0})
    status('preparing_features');sets=datasets();paragraphs=json.loads((ROOT/'data/paragraphs.json').read_text('utf-8'))
    assert not {p['group'] for p in paragraphs}&{r['family'] for rr in sets.values() for r in rr}
    pairmap={}
    for rr in sets.values():
        for r in rr:
            for qv in r['fields'].values():
                for ev in r['evidence']:
                    inp={'query':qv,'text':ev['text'],'scope':r['scope']};pairmap.setdefault(pair_key(inp),inp)
    ep=list(pairmap.values());del pairmap
    enc0=SpanEncoder(ROOT/'encoder',protocol['encoder_sha256'],'cpu');tokenizer=enc0.tokenizer
    tok=tokenizer([aligned_text(p)[0] for p in paragraphs],truncation=False,padding=False,return_offsets_mapping=True)
    raw=[token_raw_offsets(p,o) for p,o in zip(paragraphs,tok.pop('offset_mapping'))]
    labels=[token_targets(o,p['spans']) for o,p in zip(raw,paragraphs)];targets=[span_target(p,o) for p,o in zip(paragraphs,raw)]
    masks=[boundary_masks(p['text'],o) for p,o in zip(paragraphs,raw)];tokens=[{k:v[i] for k,v in tok.items()} for i in range(len(paragraphs))]
    assert max(map(len,labels))<=512;del tok,enc0;gc.collect()
    write(out/'data_audit.json',{'paragraphs':len(paragraphs),'source_counts':{x:sum(p['source']==x for p in paragraphs) for x in ['new','r10','historical']},'split_sizes':{k:len(v) for k,v in sets.items()},'max_tokens':max(map(len,labels)),'all_gold_boundaries_representable':True,'no_truncation':True,'all_holdouts_synthetic':True})
    def fresh(seed,arm):
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        enc=SpanEncoder(ROOT/'encoder',protocol['encoder_sha256'],'cuda');modules=install_lora(enc.model);enc.model.eval()
        with np.load(ROOT/f'head_source_s{seed}/weights.npz',allow_pickle=False) as z:values={k:z[k].copy() for k in z.files}
        head=Head(values).cuda();head.load_state_dict({k:torch.from_numpy(v) for k,v in values.items()});head.requires_grad_(False)
        selector=torch.nn.Linear(512,5).cuda()
        opt=torch.optim.AdamW([{'params':list(selector.parameters()),'lr':.001},{'params':[v for v in enc.model.parameters() if v.requires_grad],'lr':.0002}],weight_decay=.01)
        sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min((s+1)/96,max(0,(960-s)/864)))
        return enc,head,selector,opt,sch,modules
    def statehash(head,enc,selector):return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in list(head.state_dict().values())+list(adapter_state(enc.model).values())+list(selector.state_dict().values()))).hexdigest()
    def loss_for(enc,head,selector,arm,ids):
        rows=[tokens[i] for i in ids];batch={k:v.cuda() for k,v in tokenizer.pad(rows,return_tensors='pt').items()};ys=torch.full_like(batch['input_ids'],-100)
        for i,j in enumerate(ids):ys[i,:len(labels[j])]=torch.tensor(labels[j],device='cuda')
        logits=selector(enc.model(**batch).last_hidden_state.float());body=ys!=-100
        stats={'CE_positions':0,'paragraph_exposures':len(ids),'relation_anchor_exposures':0,'span_token_positions':0,'class_positions':0,'boundary_positions':0,'encoder_tokens':sum(len(v['input_ids']) for v in rows),'padded_tokens':int(batch['input_ids'].numel()),'encoded_sequences':len(rows),'auxiliary_query_labels':0}
        if arm.startswith('TOKEN'):
            loss=F.cross_entropy(logits[:,:,:3].reshape(-1,3),ys.reshape(-1),ignore_index=-100,weight=logits.new_tensor([.5,1,1]));stats['span_token_positions']=int(body.sum())
        else:
            classes=torch.tensor([targets[j][0] for j in ids],device='cuda');pooled=(logits[:,:,:3]*body.unsqueeze(-1)).sum(1)/body.sum(1,keepdim=True)
            loss=F.cross_entropy(pooled,classes);stats['class_positions']=len(ids)
            valid=[i for i,c in enumerate(classes.tolist()) if c!=0]
            if valid:
                for k,col in [(0,3),(1,4)]:
                    allowed=torch.zeros_like(body)
                    for i,j in enumerate(ids):allowed[i,:len(masks[j][k])]=torch.as_tensor(masks[j][k],device='cuda')
                    boundary=logits[:,:,col].masked_fill(~allowed,-1e9);target=torch.tensor([targets[ids[i]][k+1] for i in valid],device='cuda')
                    loss=loss+F.cross_entropy(boundary[valid],target)
                stats['boundary_positions']=2*len(valid)
        return loss,{'ce':float(loss.detach()),'query_ce':0.,**stats}
    completed=[];actual=0
    for seed in protocol['seeds']:
      for arm in protocol['arms']:
        order=paragraph_orders(paragraphs,seed,arm.rsplit('_',1)[1]);orderhash=digest(order);name=f'{arm}_s{seed}';folder=out/name;folder.mkdir();status('initialization',run=name)
        enc,head,selector,opt,sch,targets=fresh(seed,arm);initial=statehash(head,enc,selector);iad=adapter_state(enc.model)
        basehash=lambda:hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for n,v in enc.model.named_parameters() if not n.endswith(('lora_A','lora_B')))).hexdigest()
        bh=basehash();write(folder/'initialization.json',{'sha256':initial,'seed':seed,'order_sha256':orderhash,'base_sha256':bh,'arm':arm,'head_parameters':sum(v.numel() for v in head.parameters()),'adapter_parameters':sum(v.numel() for v in iad.values()),'selector_parameters':sum(v.numel() for v in selector.parameters()),'adapter_modules':targets})
        eligible=sorted({i for ids in order for i in ids});longest=max(eligible,key=lambda i:len(labels[i]))
        loss,stats=loss_for(enc,head,selector,arm,[longest]+[max((i for i in eligible if targets[i][0]==c),key=lambda i:len(labels[i])) for c in [1,2]]+order[0][:1]);loss.backward();assert torch.isfinite(loss) and all(v.grad is None or torch.isfinite(v.grad).all() for g in opt.param_groups for v in g['params'])
        ag=sum(float(v.grad.abs().sum()) for n,v in enc.model.named_parameters() if n.endswith(('lora_A','lora_B')) and v.grad is not None);assert ag>0;opt.zero_grad(set_to_none=True)
        write(folder/'backward_preflight.json',{'finite':True,'zero_optimizer_updates':True,'adapter_gradient_abs_sum':ag,'longest_input_index':longest,**stats})
        totals={k:0 for k in ['CE_positions','relation_anchor_exposures','paragraph_exposures','span_token_positions','class_positions','boundary_positions','encoder_tokens','padded_tokens','encoded_sequences','auxiliary_query_labels']};seconds=0.
        for step,ids in enumerate(order,1):
            if time.time()>deadline:raise TimeoutError('R11 deadline')
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
    assert actual==11520
    dev=sets['dev']+sets['new_dev'];configs={};bank={};costs={};selections={}
    for row in completed:
        name=row['run'];arm=row['arm'];seed=int(name.rsplit('_s',1)[1]);folder=out/name;status('development_selection',run=name);begin=time.monotonic();enc,head,selector,opt,sch,_=fresh(seed,arm)
        with np.load(folder/'model/weights.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        with np.load(folder/'model/adapter.npz',allow_pickle=False) as z:load_adapter(enc.model,{k:torch.from_numpy(z[k].copy()) for k in z.files})
        with np.load(folder/'model/selector.npz',allow_pickle=False) as z:sw={k:z[k].copy() for k in z.files}
        enc.model.requires_grad_(False);grid=[];candidates={};oldcfg=json.loads((ROOT/f'head_source_s{seed}/student_config.json').read_text('utf-8'));th=oldcfg['threshold'];gate=oldcfg['query_gate']
        unique={digest({'text':p['text'],'scope':p['scope']}):p for p in ep};pr=enc.selection_logits(list(unique.values()),sw)
        for sg in [.5,.7,.9]:
            selected={k:decode(p['text'],v,arm,sg) for (k,p),v in zip(unique.items(),pr)};ss=selection_scores(ep,selected,arrays,gate);dd={pair_key(p):v.tolist() for p,v in zip(ep,ss)}
            candidates[sg]=(dd,selected);rt=SelectionCached(dd,th,selected);grid.append({'query_gate':gate,'threshold':th,'span_gate':sg,**scene_metrics(dev,[rt.review(visible_input(r)) for r in dev])})
        best=max(grid,key=lambda v:(-v['unknown_to_established']-v['unknown_to_not_established'],v['macro_f1'],v['citation_f1'],-v['span_gate']))
        scores,selected=candidates[best['span_gate']]
        cfg={'version':VERSION,'representation':'lexical','query_gate':gate,'threshold':th,'span_gate':best['span_gate'],'encoder_sha256':protocol['encoder_sha256'],'arm':arm,'mode':'shadow','decision_authority':False}
        configs[name]=cfg;bank[name]=scores;selections[name]=selected;costs[name]={'encoding_and_selection_seconds':time.monotonic()-begin}
        write(folder/'selection_before_tests.json',{'selected':cfg,'grid':grid,'tests_used':False,'fixed_step':960});dest=folder/'model';write(dest/'student_config.json',cfg);write(dest/'manifest.json',{p.name:sha(p) for p in dest.iterdir()})
        del enc,head,selector,opt,sch,candidates,pr;gc.collect();torch.cuda.empty_cache()
    write(out/'all_selections_locked_before_tests.json',{'modes':configs,'time':time.time(),'tests_used':False})
    def report(name,rt,extra):
        folder=out/name;metrics={};spans={};begin=time.monotonic()
        for split,rr in sets.items():
            preds=[rt.review(visible_input(r)) for r in rr];write(folder/(split+'_predictions.json'),[{'id':r['id'],**p} for r,p in zip(rr,preds)]);metrics[split]=scene_metrics(rr,preds)
            metrics[split]['by_axis']={a:scene_metrics([r for r in rr if r['axis']==a],[p for r,p in zip(rr,preds) if r['axis']==a]) for a in sorted({r['axis'] for r in rr})}
            if name in configs and split.startswith('new_'):spans[split]=span_metrics(rr,preds)
        write(folder/'metrics.json',{'run':name,'metrics':metrics,'span_metrics':spans,'rule_evaluation_seconds':time.monotonic()-begin,**extra})
    for name,cfg in configs.items():
        status('fixed_evaluation',run=name);report(name,SelectionCached(bank[name],cfg['threshold'],selections[name]),costs[name]);refs=[]
        for split in ['new_dev','new_test','new_test_short','new_test_reordered','test']:
            r=sets[split][0]
            for qv in r['fields'].values():
                for e in r['evidence']:
                    inp={'query':qv,'text':e['text'],'scope':r['scope']};refs.append({'input':inp,'scores':bank[name][pair_key(inp)]})
        write(out/name/'pair_cpu_reference.json',refs)
    from src.evidence_reasoning.clause_language_student import ClauseLanguageStudent
    for seed in [17,29,43]:
        for label,loader,path in [('R8_FROZEN',SemanticRewardStudent,f'head_source_s{seed}'),('R10_SELECT',ClauseLanguageStudent,f'r10_source_s{seed}')]:
            name=f'{label}_s{seed}';(out/name).mkdir();status('frozen_baseline',run=name);rt=loader.load(ROOT/path,ROOT/'encoder',device='cuda');begin=time.monotonic()
            scores=rt.predict_pairs(ep);selected=rt.selected if label=='R10_SELECT' else None;cached=SelectionCached({pair_key(p):v for p,v in zip(ep,scores)},rt.config['threshold'],selected)
            report(name,cached,{'updates':0,'threshold_reselected':False,'encoding_and_selection_seconds':time.monotonic()-begin});del rt,cached,scores;gc.collect();torch.cuda.empty_cache()
    status('checkpoint_audit');cps=[]
    for cp in sorted(out.glob('*/checkpoints/step_*/checkpoint.pt')):
        m=json.loads((cp.parent/'manifest.json').read_text());c=torch.load(cp,map_location='cpu',weights_only=False)
        assert cp.stat().st_size==m['bytes'] and sha(cp)==m['sha256'] and all(finite(c[k]) for k in ['head','selector','adapter','optimizer'])
        assert c['step']==m['step'] and c['exposures']==m['exposures']==32*c['step'];assert c['protocol_sha256']==ph and c['input_manifest_sha256']==mh and c['base_encoder_files']==protocol['encoder_sha256'];assert all(c[k] is not None for k in ['scheduler','python_rng','numpy_rng','torch_rng','cuda_rng']);cps.append({'path':cp.relative_to(ROOT).as_posix(),**m,'all_passed':True});del c
    assert len(cps)==24;write(out/'checkpoint_audit.json',{'all_passed':True,'entries':cps,'zero_optimizer_updates':True})
    summary={'runs':completed,'total_updates':actual,'trained_models':12,'prediction_sets':18*len(sets),'production_promotion':False}
    for k in ['CE_positions','relation_anchor_exposures','paragraph_exposures','span_token_positions','class_positions','boundary_positions','encoder_tokens','padded_tokens','encoded_sequences','auxiliary_query_labels']:summary[k]=sum(r[k] for r in completed)
    write(out/'summary.json',summary);write(out/'completed.json',{'status':'completed','runs':list(configs),'completed_unix':time.time()});status('completed');(out/'run.lock').unlink()

if __name__=='__main__':
    if (ROOT/'outputs/run_origin.json').exists():raise RuntimeError('Existing R11 preserved; no automatic restart')
    try:run()
    except BaseException as e:write(ROOT/'outputs/failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc(),'time':time.time()});write(ROOT/'outputs/status.json',{'phase':'failed_evidence_preserved','time':time.time()});raise
