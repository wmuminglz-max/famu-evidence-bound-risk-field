"""Small typed relation extractor; type is learned from the question, not its ID."""
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from .factorized_field_features import pair_inputs, DIM, VERSION as FEATURES
from .local_field_student import LocalFieldStudent, LABELS
VERSION='typed-field-student-r5-v1'
KINDS=('signed_deadline','actual_late_completion','effective_exemption','unpaid_due_balance')

def shapes(architecture):
    base={'question.weight':(4,DIM),'question.bias':(4,)}
    if architecture=='linear':return base|{'evidence.weight':(12,DIM+10),'evidence.bias':(12,)}
    if architecture=='mlp':return base|{'hidden.weight':(32,DIM+10),'hidden.bias':(32,),'evidence.weight':(12,32),'evidence.bias':(12,)}
    raise ValueError('Unknown typed architecture')

def softmax(x):
    x=x.astype(np.float64);v=np.exp(x-x.max(axis=-1,keepdims=True));return v/v.sum(axis=-1,keepdims=True)

class TypedFieldStudent(LocalFieldStudent):
    def __init__(self,config,arrays):
        if config.get('version')!=VERSION or config.get('feature_version')!=FEATURES or config.get('labels')!=list(LABELS) or config.get('kinds')!=list(KINDS) or not .5<=config.get('threshold',0)<=.99:
            raise ValueError('Invalid typed model config')
        expected=shapes(config.get('architecture'))
        if set(expected)!=set(arrays):raise ValueError('Unexpected parameter names')
        for key,shape in expected.items():
            if arrays[key].shape!=shape or arrays[key].dtype!=np.float32 or not np.isfinite(arrays[key]).all():raise ValueError('Invalid model tensor: '+key)
        self.model=config;self.arrays=arrays

    @classmethod
    def load(cls,bundle):
        p=Path(bundle);manifest=json.loads((p/'manifest.json').read_text('utf-8'))
        if set(manifest)!={'student_config.json','weights.npz'}:raise ValueError('Invalid manifest')
        for name,h in manifest.items():
            raw=(p/name).read_bytes()
            if len(raw)>4*1024**2 or hashlib.sha256(raw).hexdigest()!=h:raise ValueError('Size/hash mismatch')
        with zipfile.ZipFile(p/'weights.npz') as z:
            if sum(i.file_size for i in z.infolist())>4*1024**2:raise ValueError('Expanded size limit')
        with np.load(p/'weights.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        return cls(json.loads((p/'student_config.json').read_text('utf-8')),arrays)

    def components(self,query,text,scope):
        value=pair_inputs(query,text,scope);a=self.arrays
        def project(weight,values):
            ids=sorted(values);v=np.asarray([values[i] for i in ids],np.float32)
            return (weight[:,ids]*v).sum(axis=1)
        q=project(a['question.weight'],value['question'])+a['question.bias']
        # No question overlap or field ID enters the evidence branch.
        e=dict(value['evidence']);e.update({DIM+i:v for i,v in enumerate(value['numeric'][:10])})
        if self.model['architecture']=='mlp':
            h=np.maximum(project(a['hidden.weight'],e)+a['hidden.bias'],0)
            scores=a['evidence.weight']@h+a['evidence.bias']
        else:scores=project(a['evidence.weight'],e)+a['evidence.bias']
        return softmax(q),softmax(scores.reshape(4,3))

    def predict_pair(self,query,text,scope):
        q,e=self.components(query,text,scope)
        return (q[:,None]*e).sum(axis=0).tolist()

    def review(self,payload):
        result=super().review(payload);result['version']=VERSION;return result
