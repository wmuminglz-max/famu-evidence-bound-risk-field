"""R8 shadow model: frozen Chinese encoder, typed NumPy relation head.

No training labels, field identifiers or rule outcome enter representation.
Evidence representation is reusable across queries only in evidence mode.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from .local_field_student import LocalFieldStudent, validate, digest
from .language_field_student import language_inputs
from .typed_field_student import softmax

VERSION='semantic-reward-student-r8-v1'
WIDTH=4096+10+512

def encoder_text(pair, mode):
    scope=pair['scope']
    def norm(value):
        for key,mark in [('actor','【目标主体】'),('event','【目标行为】'),('matter','【目标事项】')]:
            if scope.get(key):value=value.replace(scope[key],mark)
        return value
    text='作用域：'+json.dumps({k:norm(v) for k,v in scope.items()},ensure_ascii=False,sort_keys=True)+'\n原文：'+norm(pair['text'])
    if mode=='joint':text+='\n问题：'+norm(pair['query'])
    elif mode!='evidence':raise ValueError('Unknown encoder mode')
    return text

def lexical(pairs):
    q=np.zeros((len(pairs),4096),np.float32);x=np.zeros((len(pairs),WIDTH),np.float32)
    for row,p in enumerate(pairs):
        f=language_inputs(p['query'],p['text'],p['scope'])
        for k,v in f['question'].items():q[row,k]=v
        for k,v in f['evidence'].items():x[row,k]=v
        x[row,4096:4106]=f['numeric'][:10]
    return q,x

def file_sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

class FrozenEncoder:
    def __init__(self,path,expected,device='cpu'):
        import torch
        from transformers import AutoModel,AutoTokenizer
        self.device=device;self.torch=torch
        for name,h in expected.items():
            if file_sha(Path(path)/name)!=h:raise ValueError('Frozen encoder hash mismatch: '+name)
        self.tokenizer=AutoTokenizer.from_pretrained(str(path),local_files_only=True,trust_remote_code=False)
        self.model=AutoModel.from_pretrained(str(path),local_files_only=True,trust_remote_code=False,attn_implementation='eager').to(device).eval()
        self.model.requires_grad_(False)
        if self.model.config.num_hidden_layers!=4 or self.model.config.hidden_size!=512:raise ValueError('Encoder shape')
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):raise ValueError('Nonfinite encoder')
        self.last_stats={}

    def encode(self,texts,batch_size=64):
        import time
        start=time.monotonic();unique=list(dict.fromkeys(texts));encoded=self.tokenizer(unique,truncation=False,padding=False)
        lengths=list(map(len,encoded['input_ids']))
        if not lengths or max(lengths)>512:raise ValueError('Empty or >512-token input; no truncation')
        rows=[{k:v[i] for k,v in encoded.items()} for i in range(len(unique))];values=[]
        with self.torch.inference_mode():
            for off in range(0,len(rows),batch_size):
                batch={k:v.to(self.device) for k,v in self.tokenizer.pad(rows[off:off+batch_size],return_tensors='pt').items()}
                h=self.model(**batch).last_hidden_state.float();mask=batch['attention_mask'].unsqueeze(-1)
                pooled=(h*mask).sum(1)/mask.sum(1);pooled=self.torch.nn.functional.normalize(pooled,dim=-1)
                values.append(pooled.cpu().numpy())
        arr=np.concatenate(values);assert arr.shape==(len(unique),512) and np.isfinite(arr).all()
        ix={t:i for i,t in enumerate(unique)}
        self.last_stats={'unique_inputs':len(unique),'requested_pairs':len(texts),'encoded_tokens':sum(lengths),'max_tokens':max(lengths),'seconds':time.monotonic()-start,'truncated':False}
        return arr[[ix[t] for t in texts]]

class SemanticRewardStudent(LocalFieldStudent):
    def __init__(self,config,arrays,encoder):
        if config.get('version')!=VERSION or config.get('representation') not in ['lexical','joint','evidence']:raise ValueError('Invalid R8 model')
        shapes={'question.weight':(4,4096),'question.bias':(4,),'hidden.weight':(64,WIDTH),'hidden.bias':(64,),'evidence.weight':(12,64),'evidence.bias':(12,)}
        if set(arrays)!=set(shapes) or any(arrays[k].shape!=s or arrays[k].dtype!=np.float32 or not np.isfinite(arrays[k]).all() for k,s in shapes.items()):raise ValueError('Head shape/dtype/finite check')
        self.options=config;self.arrays=arrays;self.encoder=encoder;self.threshold=float(config['threshold']);self._cache=None
        # Parent review uses predict_pair, threshold and these bookkeeping fields.
        self.config=config;self.version=VERSION;self.model={'threshold':self.threshold}

    @classmethod
    def load(cls,folder,encoder_path,device='cpu'):
        p=Path(folder);m=json.loads((p/'manifest.json').read_text('utf-8'))
        if set(m)!={'weights.npz','student_config.json'}:raise ValueError('Model manifest')
        for n,h in m.items():
            if (p/n).stat().st_size>4*1024**2 or file_sha(p/n)!=h:raise ValueError('Head hash/size')
        with np.load(p/'weights.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        c=json.loads((p/'student_config.json').read_text('utf-8'))
        enc=None if c['representation']=='lexical' else FrozenEncoder(encoder_path,c['encoder_sha256'],device)
        return cls(c,a,enc)

    def predict_pairs(self,pairs):
        q,x=lexical(pairs);a=self.arrays
        if self.options['representation']!='lexical':x[:,4106:]=self.encoder.encode([encoder_text(p,self.options['representation']) for p in pairs])
        qp=softmax(q@a['question.weight'].T+a['question.bias'])
        h=np.maximum(x@a['hidden.weight'].T+a['hidden.bias'],0)
        logits=(h@a['evidence.weight'].T+a['evidence.bias']).reshape(-1,4,3)
        ep=softmax(logits);scores=ep[np.arange(len(pairs)),qp.argmax(-1)].copy()
        scores[qp.max(-1)<self.options['query_gate']]=1/3
        return scores.tolist()

    def predict_pair(self,query,text,scope):
        p={'query':query,'text':text,'scope':scope};key=digest(p)
        return self._cache[key] if self._cache is not None else self.predict_pairs([p])[0]

    def review(self,payload):
        validate(payload)
        pairs=[{'query':q,'text':e['text'],'scope':payload['scope']} for q in payload['fields'].values() for e in payload['evidence'] if e.get('visibility','visible')=='visible']
        pairs=list({digest(p):p for p in pairs}.values())
        self._cache={digest(p):s for p,s in zip(pairs,self.predict_pairs(pairs))} if pairs else {}
        try:return super().review(payload)|{'version':VERSION,'mode':'shadow'}
        finally:self._cache=None
