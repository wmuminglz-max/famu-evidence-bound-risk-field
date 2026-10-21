"""Training-only Torch implementation of the frozen R4 architectures."""
import numpy as np
import torch
from torch import nn
from src.evidence_reasoning.factorized_field_features import pair_inputs, DIM
from src.evidence_reasoning.local_field_student import features


class Factorized(nn.Module):
    def __init__(self):
        super().__init__()
        self.question = nn.Parameter(torch.empty(DIM, 16))
        self.evidence = nn.Parameter(torch.empty(DIM, 16))
        nn.init.normal_(self.question, std=.1)
        nn.init.normal_(self.evidence, std=.1)
        self.hidden = nn.Linear(75, 32)
        self.output = nn.Linear(32, 3)

    def forward(self, q, e, numeric):
        q = q @ self.question
        e = e @ self.evidence
        return self.output(torch.relu(self.hidden(torch.cat((q, e, q * e, (q - e).abs(), numeric), dim=-1))))


class Joint(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_head = nn.Linear(8192, 3)
        nn.init.zeros_(self.feature_head.weight)
        nn.init.zeros_(self.feature_head.bias)

    def forward(self, x):
        return self.feature_head(x)


def inputs(pairs, factorized):
    if factorized:
        q = np.zeros((len(pairs), DIM), np.float32)
        e = np.zeros_like(q)
        n = np.zeros((len(pairs), 11), np.float32)
        for row, p in enumerate(pairs):
            value = pair_inputs(p['query'], p['text'], p['scope'])
            for k, dst in [('question', q), ('evidence', e)]:
                for i, v in value[k].items(): dst[row, i] = v
            n[row] = value['numeric']
        return [torch.from_numpy(v) for v in (q, e, n)]
    x = np.zeros((len(pairs), 8192), np.float32)
    for row, p in enumerate(pairs):
        for i, v in features(p['query'], p['text'], p['scope']).items(): x[row, i] = v
    return [torch.from_numpy(x)]
