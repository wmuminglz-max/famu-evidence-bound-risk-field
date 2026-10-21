"""Untrained next-round representation: separate question/evidence lexical bags.

No whole-question hash chooses an evidence feature bank. A future learned
low-rank interaction must be fitted and validated before this can be served.
This module does not infer field states or risk decisions.
"""
from collections import Counter
import math
import re
import zlib
from .text import numeric_facts

VERSION='factorized-field-features-v1-untrained'
DIM=4096
NUMERIC_NAMES=('actor_present','event_present','matter_present','period_present',
    'date_before','date_same','date_after','date_error','amount_zero','amount_positive','lexical_overlap')

def surface(value,scope):
    for key in ('actor','event','matter'):
        if scope.get(key):value=value.replace(scope[key],f'<{key}>')
    value=re.sub(r'\d{4}[年/-]\d{1,2}[月/-]\d{1,2}日?','<date>',value)
    value=re.sub(r'\d+(?:\.\d+)?','#',value)
    return re.sub(r'\s+','',value.lower())

def bag(text):
    values=Counter()
    for n in (1,2,3):
        for token in set(text[i:i+n] for i in range(max(0,len(text)-n+1))):
            values[zlib.crc32(token.encode('utf-8'))%DIM]+=1.
    norm=math.sqrt(sum(v*v for v in values.values())) or 1.
    return {k:v/norm for k,v in values.items()}

def pair_inputs(query,text,scope):
    """Only observable inputs; dates compared before surface anonymization."""
    q=surface(query,scope);e=surface(text,scope)
    numeric=[float(bool(scope.get(k)) and scope[k] in text) for k in ('actor','event','matter','time')]
    observed=numeric_facts(text);deadline=numeric_facts(scope.get('assessment_deadline',''))['dates']
    d=deadline[0]['value'] if deadline else None
    dates=[r['value'] for r in observed['dates'] if r['value']]
    numeric += [float(bool(d) and any(x<d for x in dates)),float(bool(d) and d in dates),float(bool(d) and any(x>d for x in dates)),float(bool(observed['errors']))]
    amounts=[float(r['value_yuan']) for r in observed['amounts']]
    numeric += [float(any(x==0 for x in amounts)),float(any(x>0 for x in amounts)),len(set(q)&set(e))/max(1,len(set(q)))]
    assert len(numeric)==len(NUMERIC_NAMES) and all(math.isfinite(x) for x in numeric)
    return {'question':bag(q),'evidence':bag(e),'numeric':numeric}
