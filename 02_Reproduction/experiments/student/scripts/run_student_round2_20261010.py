"""Finite 4-arm paired study: balanced semantic training and bounded teacher targets."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
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
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evidence_reasoning.local_field_student import features, DIM, digest, visible_input
from scripts.run_gpu_student_20261009 import CachedStudent, scene_metrics, pair_key, sha, write, append


def normalized_text(pair):
    # Exact observable substitutions only; dates and amounts are preserved.
    scope = pair['scope']
    def normalize(value):
        for key, marker in [('actor', '【目标主体】'), ('event', '【目标行为】'), ('matter', '【目标事项】')]:
            if scope.get(key):
                value = value.replace(scope[key], marker)
        return value
    s = {k: normalize(v) for k, v in scope.items()}
    return '作用域：'+json.dumps(s,ensure_ascii=False,sort_keys=True)+'\n字段：'+normalize(pair['query'])+'\n原文：'+normalize(pair['text'])


def balanced_epoch(fit, seed, epoch, steps):
    rng = np.random.default_rng(seed + epoch*1000)
    groups = sorted({p['group'] for p in fit})
    by = {(g,c):[i for i,p in enumerate(fit) if p['group']==g and p['label']==c] for g in groups for c in range(3)}
    if not all(by.values()):
        raise ValueError('Every fitting family must contain all three labels')
    result=[]
    for _ in range(steps):
        batch=[]
        for c,n in enumerate([16,8,8]):
            for _ in range(n):
                group=groups[int(rng.integers(len(groups)))];choices=by[group,c]
                batch.append(choices[int(rng.integers(len(choices)))])
        rng.shuffle(batch);result.append(batch)
    return result


def teacher_target(record, label):
    # Labels are complete synthetic supervision. Never admit disagreement.
    accepted=bool(record['accepted'])
    if accepted and not record['mean_prediction']==record['sum_prediction']==record['gold']==label:
        raise ValueError('Teacher admission inconsistency')
    q=torch.tensor(record['mean_scores'],dtype=torch.float64).div(.25).softmax(-1).float()
    if not torch.isfinite(q).all():
        raise ValueError('Nonfinite teacher target')
    anchor=F.one_hot(torch.tensor(label),3).float()
    return .9*anchor+.1*q, float(accepted)


def objective(logits, labels, target, mask, kd):
    ce=F.cross_entropy(logits.float(),labels)
    if not kd:
        return ce
    kl=F.kl_div(F.log_softmax(logits.float(),-1),target,reduction='none').sum(-1)
    return ce+.1*(mask*kl).mean()


class Student(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.base=AutoModel.from_pretrained(str(encoder),local_files_only=True,trust_remote_code=False)
        assert self.base.config.num_hidden_layers==4 and self.base.config.hidden_size==512
        self.head=nn.Linear(512,3)
        # Same initialization and architecture across arms; feature contribution
        # is enabled only for the declared hybrid arms.
        self.feature_head=nn.Linear(DIM,3)
        nn.init.zeros_(self.feature_head.weight);nn.init.zeros_(self.feature_head.bias)

    def forward(self, batch, numeric, hybrid):
        logits=self.head(self.base(**batch).last_hidden_state[:,0])
        return logits+self.feature_head(numeric) if hybrid else logits


def make_inputs(tok, pairs):
    encoded=[];numeric=torch.zeros((len(pairs),DIM))
    for i,p in enumerate(pairs):
        item=tok(normalized_text(p),truncation=False)
        if len(item['input_ids'])>512:
            raise ValueError('Full input exceeds token bound; truncation forbidden')
        encoded.append(item)
        for k,v in features(p['query'],p['text'],p['scope']).items():numeric[i,k]=v
    return encoded,numeric


def run(args):
    out=ROOT/'outputs';out.mkdir(exist_ok=True)
    lock=out/'run.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    protocol=json.loads((ROOT/'protocol.json').read_text('utf-8'))
    start_time=time.time();deadline=start_time+protocol['wall_hours_max']*3600
    write(out/'run_origin.json',{'pid':os.getpid(),'started_unix':start_time,'deadline_unix':deadline,'protocol_sha256':sha(ROOT/'protocol.json')})
    def status(phase,**kw):
        if time.time()>deadline:raise TimeoutError('Round-two fixed wall deadline reached')
        write(out/'status.json',{'phase':phase,'pid':os.getpid(),'updated_unix':time.time(),'deadline_unix':deadline,**kw})
    status('verifying_inputs')
    for name,entry in json.loads((ROOT/'input_manifest.json').read_text('utf-8')).items():
        p=(ROOT/name).resolve();assert p.is_relative_to(ROOT.resolve()) and sha(p)==entry['sha256']
    encoder=Path(args.encoder)
    for name,h in protocol['encoder_sha256'].items():assert sha(encoder/name)==h
    if not torch.cuda.is_available():raise RuntimeError('Formal training requires CUDA')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    write(out/'environment.json',{'python':sys.version,'torch':torch.__version__,'transformers':__import__('transformers').__version__,
        'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda})
    fit=json.loads((ROOT/'data/pairs.json').read_text('utf-8'))['fit']
    assert len(fit)==2500 and len({p['id'] for p in fit})==2500
    teacher={r['id']:r for r in [json.loads(s) for s in (ROOT/'data/teacher_fit_scores.jsonl').read_text('utf-8').splitlines()]}
    assert set(teacher)=={p['id'] for p in fit}
    targets,masks=zip(*(teacher_target(teacher[p['id']],p['label']) for p in fit))
    target=torch.stack(targets).cuda();mask=torch.tensor(masks).cuda();labels=torch.tensor([p['label'] for p in fit]).cuda()
    tok=AutoTokenizer.from_pretrained(str(encoder),local_files_only=True,trust_remote_code=False)
    tx,nx=make_inputs(tok,fit);nx=nx.cuda()
    scenes=json.loads((ROOT/'data/scenes.json').read_text('utf-8'))
    challenge=json.loads((ROOT/'data/challenge.json').read_text('utf-8'))
    assert not {p['group'] for p in fit}&{r['family'] for r in challenge}
    eval_rows=[r for r in scenes if r['split'] in ['dev','test']]+challenge
    ep={}
    for row in eval_rows:
        for q in row['fields'].values():
            for e in row['evidence']:
                p={'query':q,'text':e['text'],'scope':row['scope']};ep[pair_key(p)]=p
    ep=list(ep.values());ex,en=make_inputs(tok,ep);en=en.cuda()
    write(out/'data_audit.json',{'fit_n':len(fit),'fit_classes':dict(Counter(p['label'] for p in fit)),
        'teacher_accepted':int(mask.sum()),'gold_target_min_on_accepted':min(float(target[i,p['label']]) for i,p in enumerate(fit) if masks[i]),
        'fit_max_tokens':max(len(t['input_ids']) for t in tx),'eval_max_tokens':max(len(t['input_ids']) for t in ex),
        'eval_unique_pairs':len(ep),'challenge_scenes':len(challenge),'truncated':0,'all_labels_synthetic':True})
    completed=[]
    summaries=[]
    for arm in protocol['arms']:
        hybrid=arm.startswith('H');kd=arm.endswith('KD')
        for seed in protocol['seeds']:
            run_name=f'{arm}_s{seed}';folder=out/run_name;folder.mkdir()
            status('student_initialization',run=run_name,completed=completed)
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            model=Student(encoder).cuda()
            init_hash=hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes() for t in model.state_dict().values())).hexdigest()
            if arm!=protocol['arms'][0]:
                assert init_hash==json.loads((out/f'{protocol["arms"][0]}_s{seed}'/'initialization.json').read_text())['sha256']
            write(folder/'initialization.json',{'sha256':init_hash,'parameters':sum(p.numel() for p in model.parameters()),'seed':seed,'hybrid':hybrid,'KD':kd})
            opt=torch.optim.AdamW([{'params':model.base.parameters(),'lr':1e-5},{'params':model.head.parameters(),'lr':1e-4},
                                   {'params':model.feature_head.parameters(),'lr':.05}],weight_decay=.01)
            steps=protocol['epochs']*protocol['steps_per_epoch'];warmup=math.ceil(steps*.1)
            scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min((s+1)/warmup,max(0,(steps-s)/(steps-warmup))))
            rng=torch.get_rng_state();crng=torch.cuda.get_rng_state_all();i=max(range(len(tx)),key=lambda i:len(tx[i]['input_ids']))
            b={k:v.cuda() for k,v in tok.pad([tx[i]],return_tensors='pt').items()}
            loss=objective(model(b,nx[i:i+1],hybrid),labels[i:i+1],target[i:i+1],mask[i:i+1],kd);loss.backward()
            assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            write(folder/'backward_preflight.json',{'tokens':len(tx[i]['input_ids']),'loss':float(loss),'finite':True,'optimizer_updates':0})
            opt.zero_grad(set_to_none=True);torch.set_rng_state(rng);torch.cuda.set_rng_state_all(crng)
            step=exposures=tokens=0;train_seconds=0
            for epoch in range(1,protocol['epochs']+1):
                order=balanced_epoch(fit,seed,epoch,protocol['steps_per_epoch']);model.train()
                order_hash=digest(order)
                for ids in order:
                    if time.time()>deadline:raise TimeoutError('Fixed wall budget reached')
                    begin=time.monotonic();batch={k:v.cuda() for k,v in tok.pad([tx[i] for i in ids],return_tensors='pt').items()}
                    opt.zero_grad(set_to_none=True)
                    loss=objective(model(batch,nx[ids],hybrid),labels[ids],target[ids],mask[ids],kd)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    opt.step();scheduler.step();torch.cuda.synchronize()
                    seconds=time.monotonic()-begin;train_seconds+=seconds;step+=1;exposures+=len(ids)
                    n=sum(len(tx[i]['input_ids']) for i in ids);tokens+=n
                    append(folder/'training_log.jsonl',{'step':step,'epoch':epoch,'loss':float(loss),'gradient_norm':float(norm),'examples':len(ids),
                        'tokens':n,'seconds':seconds,'kd_pairs':int(mask[ids].sum()) if kd else 0,'order_sha256':order_hash})
                    if step==1 or step%40==0:status('training',run=run_name,step=step,total_steps=steps,loss=float(loss),completed=completed)
                if epoch in protocol['checkpoint_epochs']:
                    cp=folder/'checkpoints'/f'epoch_{epoch:02d}';cp.mkdir(parents=True)
                    torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'torch_rng':torch.get_rng_state(),
                        'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
                        'step':step,'epoch':epoch,'exposures':exposures,'protocol_sha256':sha(ROOT/'protocol.json'),'order_sha256':order_hash},cp/'checkpoint.pt')
                    write(cp/'manifest.json',{'sha256':sha(cp/'checkpoint.pt'),'bytes':(cp/'checkpoint.pt').stat().st_size,'step':step,'exposures':exposures})
            status('evaluation',run=run_name,completed=completed);model.eval();scores={};begin=time.monotonic()
            with torch.inference_mode():
                for offset in range(0,len(ep),64):
                    if time.time()>deadline:raise TimeoutError('Fixed evaluation budget reached')
                    batch={k:v.cuda() for k,v in tok.pad(ex[offset:offset+64],return_tensors='pt').items()}
                    probs=model(batch,en[offset:offset+64],hybrid).float().softmax(-1).cpu().tolist()
                    scores.update((pair_key(p),s) for p,s in zip(ep[offset:offset+64],probs))
            dev=[r for r in scenes if r['split']=='dev'];grid=[]
            for threshold in protocol['threshold_grid']:
                runtime=CachedStudent(scores,threshold);pred=[runtime.review(visible_input(r)) for r in dev]
                grid.append({'threshold':threshold,**scene_metrics(dev,pred)})
            best=max(grid,key=lambda x:(-x['unknown_to_established']-x['unknown_to_not_established'],x['macro_f1'],-x['threshold']))
            write(folder/'selection_before_tests.json',{'fixed_epoch':protocol['epochs'],'threshold':best['threshold'],'grid':grid,'tests_used':False})
            runtime=CachedStudent(scores,best['threshold']);metrics={}
            for split in ['dev','test','challenge']:
                rows=challenge if split=='challenge' else [r for r in scenes if r['split']==split]
                preds=[runtime.review(visible_input(r)) for r in rows]
                write(folder/f'{split}_predictions.json',[{'id':r['id'],**p} for r,p in zip(rows,preds)])
                metrics[split]=scene_metrics(rows,preds)
                metrics[split]['by_axis']={axis:scene_metrics([r for r in rows if r['axis']==axis],[p for r,p in zip(rows,preds) if r['axis']==axis]) for axis in sorted({r['axis'] for r in rows})}
            fm=folder/'model';fm.mkdir();model.base.save_pretrained(fm,safe_serialization=True);tok.save_pretrained(fm)
            from safetensors.torch import save_file
            save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items() if not k.startswith('base.')},str(fm/'relation_heads.safetensors'))
            write(fm/'student_config.json',{'schema':'famu-student-r2','hybrid':hybrid,'threshold':best['threshold'],'labels':['irrelevant','support','refute'],
                'preprocessor':'normalized_text','feature_extractor':'local_field_student.features','max_tokens':512,'mode':'shadow','decision_authority':False})
            write(fm/'manifest.json',{p.name:sha(p) for p in fm.iterdir() if p.is_file()})
            summary={'run':run_name,'updates':step,'exposures':exposures,'train_seconds':train_seconds,'train_tokens':tokens,
                'evaluation_seconds':time.monotonic()-begin,'metrics':metrics,'production_promotion':False,'independent_legal_test':False}
            write(folder/'metrics.json',summary);write(folder/'completed.json',{'run':run_name,'updates':step,'completed_unix':time.time()})
            summaries.append(summary);completed.append(run_name)
            del model,opt,scheduler;gc.collect();torch.cuda.empty_cache()
    write(out/'summary.json',{'runs':summaries,'total_updates':sum(r['updates'] for r in summaries),'total_exposures':sum(r['exposures'] for r in summaries),
        'no_teacher_inference_or_new_labels':True,'production_promotion':False})
    write(out/'completed.json',{'runs':completed,'status':'completed','completed_unix':time.time(),'production_promotion':False})
    status('completed',completed=completed);lock.unlink()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--encoder',type=Path,required=True);args=parser.parse_args()
    if (ROOT/'outputs/run_origin.json').exists() or (ROOT/'outputs/run.lock').exists():raise RuntimeError('Existing experiment preserved; no implicit restart')
    try:run(args)
    except BaseException as e:
        write(ROOT/'outputs/failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc(),'time':time.time()})
        write(ROOT/'outputs/status.json',{'phase':'failed_evidence_preserved','updated_unix':time.time()});raise


if __name__=='__main__':main()
