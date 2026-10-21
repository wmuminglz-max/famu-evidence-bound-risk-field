"""R11: token filtering or one raw continuous interval; numeric boundaries protected."""
import json,re
from pathlib import Path
import numpy as np
from .clause_language_student import ClauseEncoder,aligned_text,token_raw_offsets,decode_spans
from .semantic_reward_student import SemanticRewardStudent,lexical,file_sha,VERSION as PREVIOUS
from .adapted_language_student import install_lora,load_adapter
from .local_field_student import digest,verify_anchor
from .typed_field_student import softmax
VERSION='continuous-span-student-r11-v1'
ARMS=['TOKEN_NEW','SPAN_NEW','TOKEN_MIX','SPAN_MIX']
ATOM=re.compile(r'\d{4}年\d{1,2}月(?:\d{1,2}日)?|\d+(?:\.\d+)?(?:元|万元|日|月|年)?')

def boundary_masks(text,offsets):
    atoms=[m.span() for m in ATOM.finditer(text)];starts=[];ends=[]
    for i,o in enumerate(offsets):
        starts.append(o is not None and (i==0 or offsets[i-1]!=o) and not any(a<o[0]<b for a,b in atoms))
        ends.append(o is not None and (i==len(offsets)-1 or offsets[i+1]!=o) and not any(a<o[1]<b for a,b in atoms))
    return np.asarray(starts),np.asarray(ends)

def span_target(pair,offsets):
    spans=pair['spans'];assert len(spans)<=1
    if not spans:return (0,-100,-100)
    s=spans[0];sm,em=boundary_masks(pair['text'],offsets)
    a=[i for i,o in enumerate(offsets) if o and o[0]==s['start'] and sm[i]]
    b=[i for i,o in enumerate(offsets) if o and o[1]==s['end'] and em[i]]
    assert a and b and a[0]<=b[-1],(pair['text'],s,a,b)
    return s['class'],a[0],b[-1]

def decode_interval(text,offsets,logits,gate):
    logits=np.asarray(logits);body=np.asarray([o is not None for o in offsets]);assert body.any()
    p=softmax(logits[body,:3].mean(0,keepdims=True))[0];c=int(p.argmax())
    if c==0 or p[c]<gate:return []
    starts,ends=boundary_masks(text,offsets);best_start=None;best_value=-np.inf;pair=None;score=-np.inf
    # The normalizers of start/end softmax are constants for joint argmax.
    for j in range(len(offsets)):
        if starts[j] and logits[j,3]>best_value:best_start=j;best_value=logits[j,3]
        if ends[j] and best_start is not None and best_value+logits[j,4]>score:
            pair=(best_start,j);score=best_value+logits[j,4]
    assert pair is not None
    a,b=offsets[pair[0]][0],offsets[pair[1]][1]
    return [{'start':int(a),'end':int(b),'class':c,'text':text[a:b]}]

def decode(text,result,arm,gate):
    if arm.startswith('SPAN'):return decode_interval(text,result['offsets'],result['logits'],gate)
    return decode_spans(text,result['offsets'],softmax(np.asarray(result['logits'])[:,:3]),gate)

def selection_scores(pairs,selected,arrays,query_gate):
    changed=[{'query':p['query'],'scope':p['scope'],'text':'。'.join(s['text'] for s in selected[digest({'text':p['text'],'scope':p['scope']})])} for p in pairs]
    q,x=lexical(changed);qp=softmax(q@arrays['question.weight'].T+arrays['question.bias'])
    h=np.maximum(x@arrays['hidden.weight'].T+arrays['hidden.bias'],0);ep=softmax((h@arrays['evidence.weight'].T+arrays['evidence.bias']).reshape(-1,4,3))
    scores=ep[np.arange(len(pairs)),qp.argmax(-1)].copy();scores[qp.max(-1)<query_gate]=1/3
    for i,p in enumerate(changed):
        if not p['text']:scores[i]=[1,0,0]
    return scores

class SpanEncoder(ClauseEncoder):
    def selection_logits(self,pairs,weights,batch_size=64):
        texts=[aligned_text(p)[0] for p in pairs];encoded=self.tokenizer(texts,truncation=False,padding=False,return_offsets_mapping=True)
        raw=[token_raw_offsets(p,o) for p,o in zip(pairs,encoded.pop('offset_mapping'))];lengths=list(map(len,encoded['input_ids']))
        if not lengths or max(lengths)>512:raise ValueError('Empty or >512-token input; no truncation')
        w=self.torch.as_tensor(weights['weight'],device=self.device);b=self.torch.as_tensor(weights['bias'],device=self.device);values=[]
        with self.torch.inference_mode():
            for off in range(0,len(pairs),batch_size):
                rows=[{k:v[i] for k,v in encoded.items()} for i in range(off,min(off+batch_size,len(pairs)))];batch={k:v.to(self.device) for k,v in self.tokenizer.pad(rows,return_tensors='pt').items()}
                logits=(self.model(**batch).last_hidden_state.float()@w.T+b).cpu().numpy()
                values.extend(v[:n].tolist() for v,n in zip(logits,lengths[off:off+len(rows)]))
        return [{'offsets':o,'logits':p} for o,p in zip(raw,values)]

class ContinuousSpanStudent(SemanticRewardStudent):
    @classmethod
    def load(cls,folder,encoder_path,device='cpu'):
        import torch
        p=Path(folder);m=json.loads((p/'manifest.json').read_text('utf-8'))
        if set(m)!={'weights.npz','adapter.npz','selector.npz','student_config.json'}:raise ValueError('R11 manifest')
        for n,h in m.items():
            if (p/n).stat().st_size>8*1024**2 or file_sha(p/n)!=h:raise ValueError('R11 hash/size')
        cfg=json.loads((p/'student_config.json').read_text('utf-8'))
        if cfg['version']!=VERSION or cfg['arm'] not in ARMS:raise ValueError('R11 config')
        enc=SpanEncoder(encoder_path,cfg['encoder_sha256'],device);install_lora(enc.model)
        with np.load(p/'adapter.npz',allow_pickle=False) as z:load_adapter(enc.model,{k:torch.from_numpy(z[k].copy()) for k in z.files})
        enc.model.requires_grad_(False);enc.model.eval()
        with np.load(p/'weights.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        obj=cls(cfg|{'version':PREVIOUS,'representation':'lexical'},a,enc)
        with np.load(p/'selector.npz',allow_pickle=False) as z:obj.selector={k:z[k].copy() for k in z.files}
        if set(obj.selector)!={'weight','bias'} or obj.selector['weight'].shape!=(5,512) or obj.selector['bias'].shape!=(5,) or any(not np.isfinite(v).all() for v in obj.selector.values()):raise ValueError('R11 selector')
        obj.options=cfg;obj.selected={};return obj

    def predict_pairs(self,pairs):
        unique={digest({'text':p['text'],'scope':p['scope']}):p for p in pairs};pr=self.encoder.selection_logits(list(unique.values()),self.selector)
        self.selected={k:decode(p['text'],v,self.options['arm'],self.options['span_gate']) for (k,p),v in zip(unique.items(),pr)}
        return selection_scores(pairs,self.selected,self.arrays,self.options['query_gate']).tolist()

    def review(self,payload):
        result=super().review(payload)|{'version':VERSION}
        result['selected_evidence']={e['id']:self.selected.get(digest({'text':e['text'],'scope':payload['scope']}),[]) for e in payload['evidence']}
        for e in payload['evidence']:
            for s in result['selected_evidence'][e['id']]:assert s['class'] in [1,2] and verify_anchor({e['id']:e['text']},{'evidence_id':e['id'],**{k:s[k] for k in ['start','end','text']}})
        return result
