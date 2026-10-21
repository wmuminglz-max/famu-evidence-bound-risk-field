"""Training-only typed question and independent evidence heads."""
import numpy as np
import torch
from torch import nn
from src.evidence_reasoning.factorized_field_features import pair_inputs,DIM

class Typed(nn.Module):
    def __init__(self,architecture):
        super().__init__();self.architecture=architecture;self.question=nn.Linear(DIM,4)
        nn.init.zeros_(self.question.weight);nn.init.zeros_(self.question.bias)
        if architecture=='mlp':self.hidden=nn.Linear(DIM+10,32)
        self.evidence=nn.Linear(32 if architecture=='mlp' else DIM+10,12)
        if architecture=='linear':nn.init.zeros_(self.evidence.weight);nn.init.zeros_(self.evidence.bias)
    def components(self,q,e):
        return self.question(q),self.evidence(torch.relu(self.hidden(e)) if self.architecture=='mlp' else e).reshape(-1,4,3)
    def forward(self,q,e):
        qp,ep=self.components(q,e)
        return (qp.softmax(-1).unsqueeze(-1)*ep.softmax(-1)).sum(1)

def inputs(pairs):
    q=np.zeros((len(pairs),DIM),np.float32);e=np.zeros((len(pairs),DIM+10),np.float32)
    for row,p in enumerate(pairs):
        x=pair_inputs(p['query'],p['text'],p['scope'])
        for i,v in x['question'].items():q[row,i]=v
        for i,v in x['evidence'].items():e[row,i]=v
        e[row,DIM:]=x['numeric'][:10]
    return [torch.from_numpy(v) for v in (q,e)]
