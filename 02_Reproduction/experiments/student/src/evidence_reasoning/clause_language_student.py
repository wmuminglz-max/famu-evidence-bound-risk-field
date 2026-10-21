"""R10 shadow runtimes: pooling control and learned current-clause selection."""
import json
from pathlib import Path
import numpy as np
from .semantic_reward_student import FrozenEncoder, SemanticRewardStudent, encoder_text, lexical, file_sha
from .adapted_language_student import install_lora, load_adapter
from .typed_field_student import softmax
from .local_field_student import digest, verify_anchor

VERSION='clause-language-student-r10-v1'

def aligned_text(pair):
    """Exact normalization with every normalized character mapped to raw source."""
    text=pair['text'];maps=[(i,i+1) for i in range(len(text))]
    for key,mark in [('actor','【目标主体】'),('event','【目标行为】'),('matter','【目标事项】')]:
        needle=pair['scope'].get(key)
        if not needle:continue
        pieces=[];mapped=[];offset=0
        while True:
            start=text.find(needle,offset)
            if start<0:pieces.append(text[offset:]);mapped+=maps[offset:];break
            pieces.append(text[offset:start]);mapped+=maps[offset:start]
            span=(maps[start][0],maps[start+len(needle)-1][1])
            pieces.append(mark);mapped += [span]*len(mark);offset=start+len(needle)
        text=''.join(pieces);maps=mapped
    encoded=encoder_text(pair,'evidence');prefix=encoded[:-len(text)] if text else encoded
    assert encoded==prefix+text
    return encoded,[None]*len(prefix)+maps

def token_raw_offsets(pair,offsets):
    text,mapping=aligned_text(pair);result=[]
    for a,b in offsets:
        span=[v for v in mapping[a:b] if v is not None]
        result.append((min(v[0] for v in span),max(v[1] for v in span)) if span else None)
    return result

def token_targets(raw_offsets,spans):
    labels=[]
    for span in raw_offsets:
        if span is None:labels.append(-100);continue
        hits=[s['class'] for s in spans if span[0]>=s['start'] and span[1]<=s['end']]
        assert len(hits)<=1
        labels.append(hits[0] if hits else 0)
    return labels

def decode_spans(text,offsets,probabilities,gate):
    spans=[]
    for span,p in zip(offsets,probabilities):
        cls=int(np.argmax(p))
        if span is None or cls==0 or float(p[cls])<gate:continue
        a,b=span
        if spans and spans[-1]['class']==cls and a<=spans[-1]['end']:
            spans[-1]['end']=max(spans[-1]['end'],b)
        else:spans.append({'start':a,'end':b,'class':cls})
    for s in spans:s['text']=text[s['start']:s['end']]
    return spans

class ClauseEncoder(FrozenEncoder):
    def encode(self,texts,batch_size=64):
        import time
        start=time.monotonic();unique=list(dict.fromkeys(texts));enc=self.tokenizer(unique,truncation=False,padding=False)
        lengths=list(map(len,enc['input_ids']))
        if not lengths or max(lengths)>512:raise ValueError('Empty or >512-token input; no truncation')
        values=[]
        with self.torch.inference_mode():
            for off in range(0,len(unique),batch_size):
                rows=[{k:v[i] for k,v in enc.items()} for i in range(off,min(off+batch_size,len(unique)))]
                batch={k:v.to(self.device) for k,v in self.tokenizer.pad(rows,return_tensors='pt').items()}
                h=self.model(**batch).last_hidden_state.float();mask=batch['attention_mask'].unsqueeze(-1)
                pooled=h[:,0] if self.pooling=='cls' else (h*mask).sum(1)/mask.sum(1)
                values.append(self.torch.nn.functional.normalize(pooled,dim=-1).cpu().numpy())
        arr=np.concatenate(values);assert arr.shape==(len(unique),512) and np.isfinite(arr).all()
        ix={t:i for i,t in enumerate(unique)}
        self.last_stats={'unique_inputs':len(unique),'requested_pairs':len(texts),'encoded_tokens':sum(lengths),
                         'max_tokens':max(lengths),'seconds':time.monotonic()-start,'truncated':False,'pooling':self.pooling}
        return arr[[ix[t] for t in texts]]

    def selection_probabilities(self,pairs,weights,batch_size=64):
        # Metadata such as labels/spans is deliberately not read here.
        texts=[aligned_text(p)[0] for p in pairs];encoded=self.tokenizer(texts,truncation=False,padding=False,return_offsets_mapping=True)
        raw=[token_raw_offsets(p,o) for p,o in zip(pairs,encoded.pop('offset_mapping'))]
        lengths=list(map(len,encoded['input_ids']))
        if not lengths or max(lengths)>512:raise ValueError('Empty or >512-token input; no truncation')
        w=self.torch.as_tensor(weights['weight'],device=self.device);b=self.torch.as_tensor(weights['bias'],device=self.device);values=[]
        with self.torch.inference_mode():
            for off in range(0,len(pairs),batch_size):
                rows=[{k:v[i] for k,v in encoded.items()} for i in range(off,min(off+batch_size,len(pairs)))]
                batch={k:v.to(self.device) for k,v in self.tokenizer.pad(rows,return_tensors='pt').items()}
                h=self.model(**batch).last_hidden_state.float();probs=(h@w.T+b).softmax(-1).cpu().numpy()
                values.extend(p[:n] for p,n in zip(probs,lengths[off:off+len(rows)]))
        return [{'offsets':o,'probabilities':p.tolist()} for o,p in zip(raw,values)]

class ClauseLanguageStudent(SemanticRewardStudent):
    @classmethod
    def load(cls,folder,encoder_path,device='cpu'):
        import torch
        p=Path(folder);m=json.loads((p/'manifest.json').read_text('utf-8'))
        if set(m)!={'weights.npz','adapter.npz','selector.npz','student_config.json'}:raise ValueError('R10 manifest')
        for n,h in m.items():
            if (p/n).stat().st_size>8*1024**2 or file_sha(p/n)!=h:raise ValueError('R10 hash/size')
        cfg=json.loads((p/'student_config.json').read_text('utf-8'))
        if cfg['version']!=VERSION or cfg['arm'] not in ['MEAN','CLS','SELECT']:raise ValueError('R10 config')
        enc=ClauseEncoder(encoder_path,cfg['encoder_sha256'],device);enc.pooling=cfg['pooling'];install_lora(enc.model)
        with np.load(p/'adapter.npz',allow_pickle=False) as z:load_adapter(enc.model,{k:torch.from_numpy(z[k].copy()) for k in z.files})
        enc.model.requires_grad_(False);enc.model.eval()
        with np.load(p/'weights.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        from .semantic_reward_student import VERSION as PREVIOUS
        obj=cls(cfg|{'version':PREVIOUS,'representation':'lexical' if cfg['arm']=='SELECT' else 'evidence'},a,enc)
        with np.load(p/'selector.npz',allow_pickle=False) as z:obj.selector={k:z[k].copy() for k in z.files}
        if set(obj.selector)!={'weight','bias'} or obj.selector['weight'].shape!=(3,512) or obj.selector['bias'].shape!=(3,) or any(not np.isfinite(v).all() for v in obj.selector.values()):raise ValueError('R10 selector')
        obj.options=cfg;obj.selected={};obj.version=VERSION
        return obj

    def predict_pairs(self,pairs):
        if self.options['arm']!='SELECT':return super().predict_pairs(pairs)
        unique={digest({'text':p['text'],'scope':p['scope']}):p for p in pairs}
        probs=self.encoder.selection_probabilities(list(unique.values()),self.selector)
        self.selected={k:decode_spans(p['text'],v['offsets'],v['probabilities'],self.options['span_gate']) for (k,p),v in zip(unique.items(),probs)}
        changed=[p|{'text':'。'.join(s['text'] for s in self.selected[digest({'text':p['text'],'scope':p['scope']})])} for p in pairs]
        q,x=lexical(changed);a=self.arrays;qp=softmax(q@a['question.weight'].T+a['question.bias'])
        h=np.maximum(x@a['hidden.weight'].T+a['hidden.bias'],0);ep=softmax((h@a['evidence.weight'].T+a['evidence.bias']).reshape(-1,4,3))
        scores=ep[np.arange(len(pairs)),qp.argmax(-1)].copy();scores[qp.max(-1)<self.options['query_gate']]=1/3
        for i,p in enumerate(changed):
            if not p['text']:scores[i]=[1,0,0]
        return scores.tolist()

    def review(self,payload):
        result=super().review(payload)|{'version':VERSION}
        if self.options['arm']=='SELECT':
            result['selected_evidence']={e['id']:self.selected.get(digest({'text':e['text'],'scope':payload['scope']}),[]) for e in payload['evidence']}
            for e in payload['evidence']:
                for s in result['selected_evidence'][e['id']]:assert verify_anchor({e['id']:e['text']},{'evidence_id':e['id'],**{k:s[k] for k in ['start','end','text']}})
            result['selection_scope']='Learned spans; original full paragraph remains the evidence anchor. Selection is not expert fact verification.'
        return result
