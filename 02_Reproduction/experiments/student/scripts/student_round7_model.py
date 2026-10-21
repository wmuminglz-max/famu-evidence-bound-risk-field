"""Unchanged R6 parameter capacity; only observable feature encoding varies."""
import numpy as np
import torch
from scripts.student_round6_model import Typed
from src.evidence_reasoning.clause_field_student import clause_inputs
from src.evidence_reasoning.language_field_student import language_inputs
from src.evidence_reasoning.factorized_field_features import DIM

def inputs(pairs,representation='clause'):
    fn={'clause':clause_inputs,'bag':language_inputs}[representation]
    q=np.zeros((len(pairs),DIM),np.float32);e=np.zeros((len(pairs),DIM+10),np.float32)
    for row,p in enumerate(pairs):
        x=fn(p['query'],p['text'],p['scope'])
        for i,v in x['question'].items():q[row,i]=v
        for i,v in x['evidence'].items():e[row,i]=v
        e[row,DIM:]=x['numeric'][:10]
    return [torch.from_numpy(v) for v in (q,e)]
