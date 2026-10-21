"""R7 shadow extractor: observable clause/cue context, no programmed verdicts."""
import math
import re
import zlib
from collections import Counter
import numpy as np
from .language_field_student import LanguageFieldStudent,language_inputs,normalize_language,VERSION as R6_VERSION
from .factorized_field_features import surface,DIM
from .typed_field_student import softmax

VERSION='clause-field-student-r7-v1'
CLAUSES=re.compile(r'[，,。；;：:！？!?\n]+')
CUES=re.compile(r'并非|不是|不能|没有|尚未|未曾|无法|不再|未|不|无|仅|只|如果|若|待|后才')

def context_tokens(text):
    """Punctuation is a boundary heuristic, NOT a verified syntactic scope parser."""
    tokens=set()
    for clause in CLAUSES.split(text):
        if not clause:continue
        for width in (3,6):
            tokens.add('start:'+clause[:width]);tokens.add('end:'+clause[-width:])
        cues=list(CUES.finditer(clause))
        for cue in cues:
            for width in (2,4,8):
                tokens.add('left:'+clause[max(0,cue.start()-width):cue.start()]+'|'+cue.group())
                tokens.add('right:'+cue.group()+'|'+clause[cue.end():cue.end()+width])
            # Nearby pairs retain order and multiplicity without canceling negation.
            for other in cues:
                if 0<other.start()-cue.start()<=12:
                    tokens.add('cue_pair:'+clause[cue.start():other.end()])
    return tokens

def clause_inputs(query,text,scope):
    x=language_inputs(query,text,scope);value=normalize_language(surface(text,scope))
    extra=Counter(zlib.crc32(('R7:'+t).encode('utf-8'))%DIM for t in context_tokens(value))
    norm=math.sqrt(sum(v*v for v in extra.values())) or 1.
    combined=Counter(x['evidence'])
    for k,v in extra.items():combined[k]+=v/norm
    total=math.sqrt(sum(v*v for v in combined.values())) or 1.
    x['evidence']={k:v/total for k,v in combined.items()}
    return x

class ClauseFieldStudent(LanguageFieldStudent):
    def __init__(self,config,arrays):
        if config.get('version')!=VERSION or config.get('representation') not in ['bag','clause'] or config.get('routing')!='hard':raise ValueError('Invalid R7 options')
        super().__init__(config|{'version':R6_VERSION,'representation':'language'},arrays)
        self.options=config

    def components(self,query,text,scope):
        fn=clause_inputs if self.options['representation']=='clause' else language_inputs
        x=fn(query,text,scope);a=self.arrays
        def project(w,v):
            ids=sorted(v);vals=np.asarray([v[i] for i in ids],np.float32);return (w[:,ids]*vals).sum(axis=1)
        q=project(a['question.weight'],x['question'])+a['question.bias'];e=dict(x['evidence'])
        e.update({DIM+i:v for i,v in enumerate(x['numeric'][:10])})
        h=np.maximum(project(a['hidden.weight'],e)+a['hidden.bias'],0)
        return softmax(q),softmax((a['evidence.weight']@h+a['evidence.bias']).reshape(4,3))

    def review(self,payload):
        result=super().review(payload);result['version']=VERSION;return result
