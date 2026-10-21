"""R6 lexical normalization candidate; original evidence and rule path stay intact."""
import json
import re
from pathlib import Path
import numpy as np
from .factorized_field_features import pair_inputs,bag,surface
from .typed_field_student import TypedFieldStudent,softmax,VERSION as TVERSION,FEATURES

VERSION='language-field-student-r6-v1'
# Longest, one-pass phrase replacement. No source/target/field metadata is used.
# Deliberately do not erase negation, missing-material markers or conditions.
GROUPS={
 '签署':['签字','签章','签认','落款'],
 '有效':['生效'],
 '全部':['齐备','齐全','悉数','全数'],
 '核实':['查验','核验','核真','验真','核对','核齐','核查','查清','核清'],
 '期限':['截止日期','最后期限','最迟日期','最晚日期','截止日','最迟时间','最晚时间'],
 '完成':['办妥','办完','办结','结束'],
 '实际':['真实','实绩','确切'],
 '计划':['预计','预定','拟定','日程意向','日程预告','目标日期'],
 '批准':['许可','准许','获准','同意','准予'],
 '免责':['免受处理','免于处理','不予追究','不追究','豁免','不受处理'],
 '拒绝':['否决','驳回'],
 '结清':['清偿完毕','悉数清偿','足额清偿','全额支付'],
 '欠款':['欠付','欠账','未付余额','剩余欠款','应付未付','未清偿款项']}
MAP={term:canonical for canonical,terms in GROUPS.items() for term in terms}
PATTERN=re.compile('|'.join(re.escape(t) for t in sorted(MAP,key=lambda t:(-len(t),t))))
def normalize_language(text):return PATTERN.sub(lambda m:MAP[m.group()],text)
def language_inputs(query,text,scope):
    result=pair_inputs(query,text,scope)
    # Keep numeric/scope parsing on the unmodified source, and preserve originals
    # in review/anchors. Only lexical bags receive the one-pass normalization.
    result['question']=bag(normalize_language(surface(query,scope)))
    result['evidence']=bag(normalize_language(surface(text,scope)))
    return result

class LanguageFieldStudent(TypedFieldStudent):
    def __init__(self,config,arrays):
        if config.get('version')!=VERSION or config.get('representation') not in ['raw','language'] or config.get('routing') not in ['soft','hard'] or config.get('query_gate') not in [0.,.4,.6] or config.get('architecture')!='mlp':raise ValueError('Invalid R6 options')
        super().__init__(config|{'version':TVERSION,'feature_version':FEATURES},arrays)
        self.options=config

    @classmethod
    def load(cls,bundle):
        p=Path(bundle)
        import hashlib
        import zipfile
        m=json.loads((p/'manifest.json').read_text('utf-8'))
        if set(m)!={'student_config.json','weights.npz'}:raise ValueError('R6 manifest names')
        for name,h in m.items():
            f=p/name
            if f.stat().st_size>4*1024**2 or hashlib.sha256(f.read_bytes()).hexdigest()!=h:raise ValueError('R6 model hash/size mismatch')
        with zipfile.ZipFile(p/'weights.npz') as z:
            if sum(i.file_size for i in z.infolist())>4*1024**2:raise ValueError('Expanded size limit')
        with np.load(p/'weights.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        return cls(json.loads((p/'student_config.json').read_text('utf-8')),arrays)

    def components(self,query,text,scope):
        if self.options['representation']=='raw':return super().components(query,text,scope)
        x=language_inputs(query,text,scope);a=self.arrays
        def project(w,values):
            ids=sorted(values);v=np.asarray([values[i] for i in ids],np.float32);return (w[:,ids]*v).sum(axis=1)
        q=project(a['question.weight'],x['question'])+a['question.bias'];e=dict(x['evidence'])
        e.update({4096+i:v for i,v in enumerate(x['numeric'][:10])})
        h=np.maximum(project(a['hidden.weight'],e)+a['hidden.bias'],0)
        return softmax(q),softmax((a['evidence.weight']@h+a['evidence.bias']).reshape(4,3))

    def predict_pair(self,query,text,scope):
        q,e=self.components(query,text,scope)
        if max(q)<self.options['query_gate']:return [1/3]*3
        return e[int(q.argmax())].tolist() if self.options['routing']=='hard' else (q[:,None]*e).sum(0).tolist()

    def review(self,payload):
        result=super().review(payload);result['version']=VERSION;return result
