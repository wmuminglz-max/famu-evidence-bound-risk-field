"""Sparse token-level low-rank experts inside frozen FFN down projections.

This is an adaptation MoE, not four copies of the 9B model or a classifier
ensemble. The base FFN runs once. Only selected low-rank residuals are evaluated.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

TYPES = ['text_presence', 'actual_event', 'normative_permission', 'financial_balance']
LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]


def public_expert(row):
    kind = row['proposition_type']
    if kind == 'actual_fact': kind = 'actual_event'
    return TYPES.index(kind) if kind in TYPES else 3


class AuxiliaryGradient(torch.autograd.Function):
    """Attach a separately weighted auxiliary objective through checkpointing.

The caller supplies the complete microbatch and layer/token normalization.
No AMP loss scaler is used. Forward values remain unchanged. This avoids
retaining side-channel checkpoint graphs or accidentally counting recomputes.
"""
    @staticmethod
    def forward(ctx, output, auxiliary):
        ctx.save_for_backward(auxiliary)
        return output

    @staticmethod
    def backward(ctx, gradient):
        auxiliary, = ctx.saved_tensors
        return gradient, torch.ones_like(auxiliary)


class LowRank(nn.Module):
    def __init__(self, in_features, out_features, rank, device):
        super().__init__()
        self.a = nn.Parameter(torch.empty(rank, in_features, device=device, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(out_features, rank, device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x):
        # FP32 master parameters; BF16 matrix products on the GPU base model.
        return F.linear(F.linear(x, self.a.to(x.dtype)), self.b.to(x.dtype)) * 2


class RoutedDown(nn.Module):
    def __init__(self, base, kind='token', rank=4, experts=4):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.kind = kind
        self.experts = nn.ModuleList([LowRank(base.in_features, base.out_features, rank, base.weight.device)
                                     for _ in range(experts)])
        self.router = nn.Linear(base.in_features, experts, bias=False, device=base.weight.device, dtype=torch.float32)
        nn.init.normal_(self.router.weight, std=.002)
        self.primary = 0
        self.prompt_vector = None
        self.forced = None
        self.aux_scale = 0.
        self.total_tokens = 1
        self.collect = False
        self.route_records = []

    def routing(self, flat):
        n, e = len(flat), len(self.experts)
        if self.forced is not None or self.kind == 'specialize':
            expert = self.forced if self.forced is not None else self.primary
            chosen = torch.full((n, 1), expert, dtype=torch.long, device=flat.device)
            return chosen, flat.new_ones((n, 1)), None
        if self.kind == 'hard':
            second = 3 if self.primary != 3 else 0
            chosen = torch.tensor([self.primary, second], device=flat.device).expand(n, 2)
            return chosen, flat.new_full((n, 2), .5), None
        if self.kind == 'prompt':
            if self.prompt_vector is None: raise ValueError('Prompt-only routing context required')
            inputs = self.prompt_vector.float().reshape(1, -1).expand(n, -1)
        elif self.kind == 'token':
            inputs = flat.float()
        else: raise ValueError(self.kind)
        # RMS normalization is per token, hence does not expose later tokens.
        inputs = inputs * torch.rsqrt(inputs.square().mean(-1, keepdim=True) + 1e-6)
        logits = self.router(inputs)
        probabilities = logits.softmax(-1)
        weights, chosen = probabilities.topk(2, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        frequency = F.one_hot(chosen, e).float().mean((0, 1)).detach()
        balance = e * (frequency * probabilities.mean(0)).sum()
        z = logits.logsumexp(-1).square().mean()
        return chosen, weights, (.01 * balance + .001 * z)

    def forward(self, x):
        if x.ndim != 3 or x.shape[0] != 1:
            raise ValueError('Audited path requires one unpadded sequence per microbatch')
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        chosen, weights, auxiliary = self.routing(flat)
        delta = torch.zeros((len(flat), self.base.out_features), device=x.device, dtype=x.dtype)
        for i, expert in enumerate(self.experts):
            positions, slots = torch.where(chosen == i)
            if positions.numel():
                value = expert(flat.index_select(0, positions)) * weights[positions, slots, None].to(x.dtype)
                # Each token occurs at most once within an expert: deterministic scatter.
                delta = delta.index_copy(0, positions, delta.index_select(0, positions) + value)
        output = self.base(x) + delta.reshape(*shape[:-1], self.base.out_features)
        if self.training and torch.is_grad_enabled() and auxiliary is not None and self.aux_scale:
            output = AuxiliaryGradient.apply(output, auxiliary * self.aux_scale * len(flat) / self.total_tokens)
        if self.collect:
            with torch.no_grad():
                counts = torch.bincount(chosen.flatten(), minlength=len(self.experts)).cpu().tolist()
                signatures = torch.unique(chosen.sort(-1).values, dim=0).shape[0]
                self.route_records.append({'tokens':len(flat), 'dispatches':counts,
                                           'distinct_token_expert_sets':signatures, 'dropped':0})
        return output


class DenseDown(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        base.requires_grad_(False)
        self.delta = LowRank(base.in_features, base.out_features, rank, base.weight.device)

    def forward(self, x):
        return self.base(x) + self.delta(x)


def install(model, kind, layers=LAYERS):
    model.requires_grad_(False)
    found = {}
    for name, module in list(model.named_modules()):
        if not name.endswith('.mlp') or not hasattr(module, 'down_proj'): continue
        parts = name.split('.')
        index = int(parts[parts.index('layers')+1])
        if index not in layers: continue
        base = module.down_proj
        if kind.startswith('dense'):
            module.down_proj = DenseDown(base, int(kind[5:]))
        else:
            module.down_proj = RoutedDown(base, kind=kind)
        found[index] = module.down_proj
    if set(found) != set(layers): raise ValueError(f'Missing FFN layers: {set(layers)-set(found)}')
    model._famu_v7_modules = [found[i] for i in layers]
    return {'layers':layers, 'kind':kind, 'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'base_weights_frozen':True, 'rank_per_expert':4 if not kind.startswith('dense') else int(kind[5:])}


def context(model, row, prompt_ids, total_tokens, micro_scale=0., forced=None, collect=False):
    modules = model._famu_v7_modules
    vector = None
    if any(isinstance(m, RoutedDown) and m.kind == 'prompt' for m in modules):
        # Only prompt IDs; the answer and EOS are explicitly excluded by caller.
        with torch.no_grad():
            vector = model.get_input_embeddings()(prompt_ids).float().mean((0,1))
            width = modules[0].base.in_features
            if width % vector.numel(): raise ValueError('Embedding and MLP dimensions incompatible')
            vector = vector.repeat(width // vector.numel())
    for module in modules:
        if isinstance(module, RoutedDown):
            module.primary = public_expert(row)
            module.prompt_vector = vector
            module.forced = forced
            module.aux_scale = micro_scale / len(modules)
            module.total_tokens = total_tokens
            module.collect = collect
            module.route_records = []


def adapter_state(model):
    return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}


def load_adapter(model, state):
    parameters = {n:p for n,p in model.named_parameters() if p.requires_grad}
    if set(parameters) != set(state): raise ValueError('Adapter structure differs')
    with torch.no_grad():
        for name, value in state.items(): parameters[name].copy_(value)


def route_summary(model):
    return {str(i):m.route_records for i,m in zip(LAYERS, model._famu_v7_modules) if isinstance(m, RoutedDown)}
