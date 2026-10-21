"""Risk-field conditioned low-rank experts inside PEFT's LoRA B projection.

The ordinary rank-r update is partitioned into E rank-r/E subspaces. A
token/field router selects top-k subspaces, preserving the exact ordinary LoRA
update when routing is uniform. Experts have no presumed legal semantics.
Context is derived only from visible query/evidence, never targets or IDs.
"""
import json
import math
import torch
from torch import nn
from torch.nn import functional as F

CONTEXT_DIM = 64
EXPERTS = 4
TASK_ORDER = ('qa', 'condition', 'relation', 'risk_status', 'conditional_grade', 'matter_grade')


class FieldExpertProjection(nn.Module):
    def __init__(self, linear, mode='learned', top_k=2):
        super().__init__()
        if linear.in_features % EXPERTS:
            raise ValueError('LoRA rank must be divisible by four')
        self.weight = linear.weight
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.mode, self.top_k = mode, top_k
        self.token_router = nn.Linear(self.in_features, EXPERTS, bias=False,
                                      device=self.weight.device, dtype=self.weight.dtype)
        self.field_router = nn.Linear(CONTEXT_DIM, EXPERTS, bias=False,
                                      device=self.weight.device, dtype=self.weight.dtype)
        nn.init.normal_(self.token_router.weight, std=0.02)
        nn.init.normal_(self.field_router.weight, std=0.02)
        if mode != 'learned':
            self.token_router.requires_grad_(False)
            self.field_router.requires_grad_(False)
        self.register_buffer('context', torch.zeros(CONTEXT_DIM, device=self.weight.device,
                                                   dtype=self.weight.dtype), persistent=False)
        self.fixed_index = 0
        self.override = None
        self.collect_usage = False
        self.register_buffer('usage_mass', torch.zeros(EXPERTS, device=self.weight.device), persistent=False)
        self.register_buffer('usage_tokens', torch.zeros((), device=self.weight.device), persistent=False)

    def forward(self, z):
        mode = self.override or self.mode
        if mode == 'uniform':
            return F.linear(z, self.weight)
        if mode == 'fixed':
            gates = z.new_zeros(EXPERTS)
            gates[self.fixed_index] = 0.5
            gates[(self.fixed_index + 1) % EXPERTS] = 0.5
        else:
            logits = self.token_router(z) + self.field_router(self.context)
            if self.top_k < EXPERTS:
                keep = logits.topk(self.top_k, dim=-1).indices
                mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, keep, True)
                logits = logits.masked_fill(~mask, float('-inf'))
            gates = logits.float().softmax(-1).to(z.dtype)
        if self.collect_usage:
            with torch.no_grad():
                if gates.ndim == 1:
                    n = z.numel() // z.shape[-1]
                    self.usage_mass.add_(gates.float() * n)
                else:
                    flat = gates.reshape(-1, EXPERTS)
                    n = flat.shape[0]
                    self.usage_mass.add_(flat.float().sum(0))
                self.usage_tokens.add_(n)
        width = self.in_features // EXPERTS
        output = None
        for i in range(EXPERTS):
            # Top-k activates subspaces; batched GEMMs remain dense across tokens.
            part = F.linear(z[..., i*width:(i+1)*width], self.weight[:, i*width:(i+1)*width])
            part = part * (EXPERTS * gates[..., i:i+1])
            output = part if output is None else output + part
        return output


def attach_field_experts(model, mode='learned'):
    if mode not in {'learned', 'fixed'}:
        raise ValueError(mode)
    count = 0
    # Keep the other attention/FFN input projections ordinary LoRA. Routing
    # operates on attention output and FFN output residuals inside every layer.
    for name, module in list(model.named_modules()):
        if name.split('.')[-1] not in {'o_proj', 'out_proj', 'down_proj'}:
            continue
        if not hasattr(module, 'lora_B') or 'default' not in module.lora_B:
            continue
        module.lora_B['default'] = FieldExpertProjection(module.lora_B['default'], mode)
        count += 1
    if count == 0:
        raise ValueError('No compatible LoRA projections')
    return count


def routed_modules(model):
    return [m for m in model.modules() if isinstance(m, FieldExpertProjection)]


def set_field_context(model, features, task='qa'):
    index = (TASK_ORDER.index(task) if task in TASK_ORDER else 0) % EXPERTS
    for module in routed_modules(model):
        module.context.copy_(torch.as_tensor(features, device=module.context.device, dtype=module.context.dtype))
        module.fixed_index = index


def visible_features(model, tokenizer, row):
    """Two fixed random projections of frozen token-embedding means (32+32).

    This lightweight field encoder is an explicit hypothesis, not a legal
    entailment oracle. All evidence is included; no label-directed retrieval.
    """
    if row['split'] == 'aux_train':
        query = '\n'.join(m['content'] for m in row['messages'][:-1])
        evidence = query
    else:
        x = row['input']
        query = json.dumps({k:x[k] for k in sorted(x) if k != 'evidence'}, ensure_ascii=False)
        evidence = '\n'.join(e['text'] for e in x['evidence'])
    embedding = model.get_input_embeddings()
    hidden = embedding.weight.shape[1]
    generator = torch.Generator(device='cpu').manual_seed(20261004)
    projection = torch.randn(hidden, CONTEXT_DIM//2, generator=generator, dtype=torch.float32) / math.sqrt(hidden)
    projection = projection.to(embedding.weight.device)
    vectors = []
    with torch.no_grad():
        for text in [query, evidence]:
            ids = tokenizer(text, add_special_tokens=False)['input_ids']
            if not ids:
                vectors.append(torch.zeros(CONTEXT_DIM//2, device=projection.device))
                continue
            total = torch.zeros(hidden, dtype=torch.float32, device=projection.device)
            for start in range(0, len(ids), 1024):
                values = embedding(torch.tensor(ids[start:start+1024], device=projection.device))
                total.add_(values.float().sum(0))
            vector = (total / len(ids)) @ projection
            vectors.append(F.normalize(vector, dim=0))
    return torch.cat(vectors).cpu().tolist()


def routing_usage(model, reset=False):
    result = []
    for name, module in model.named_modules():
        if not isinstance(module, FieldExpertProjection):
            continue
        mass = module.usage_mass.detach().cpu()
        n = float(module.usage_tokens)
        fractions = mass / max(1., n)
        entropy = float(-(fractions * fractions.clamp_min(1e-12).log()).sum())
        result.append(dict(module=name,token_evaluations=int(n),expert_mass_fraction=fractions.tolist(),entropy=entropy))
        if reset:
            module.usage_mass.zero_(); module.usage_tokens.zero_()
    return result
