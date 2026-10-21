"""Token-forward BGE/LoRA process heads with record-normalized masked losses."""
from __future__ import annotations

import hashlib
import math
import torch
from torch import nn
from safetensors.torch import load_file, save_file

from .famu_contract_model import FamuContractModel, MODEL_CLASSES
from .expert_data import TASKS

PROCESS_CLASSES = {'condition': ('T', 'F', 'U'), 'relation': ('support', 'refute', 'irrelevant'),
                   'roles': ('support', 'refute', 'exception', 'definition', 'irrelevant'),
                   'scope': ('outside', 'applies'), 'relevance': ('irrelevant', 'relevant')}


def is_lora(name):
    return name.startswith('encoder.') and (name.endswith('.a.weight') or name.endswith('.b.weight'))


def is_adapter(name):
    return not name.startswith('encoder.') or is_lora(name)


class ExpertContractModel(FamuContractModel):
    def __init__(self, encoder_folder, rank=4, alpha=8):
        super().__init__(encoder_folder, rank, alpha)
        self.process_heads = nn.ModuleDict({k: nn.Linear(128, len(v)) for k, v in PROCESS_CLASSES.items()})

    def token_features(self, pieces):
        vectors = []
        for offset in range(0, len(pieces), 8):
            tokens = self.pad_tokens(pieces[offset:offset+8])
            vectors.append(nn.functional.normalize(self.encoder(**tokens).last_hidden_state[:, 0], dim=-1))
        return nn.functional.normalize(torch.cat(vectors).mean(dim=0), dim=-1)

    def fuse_features(self, pooled, context):
        text = self.text_projection(pooled)
        visible = self.context_projection(torch.as_tensor(context, dtype=pooled.dtype, device=pooled.device))
        gate = torch.sigmoid(self.fusion_gate(torch.cat([text, visible])))
        return self.fusion_norm(text + gate * visible)

    def encode_case(self, pieces, context):
        return self.fuse_features(self.token_features(pieces), context)

    def process_logits(self, pieces, context, aggregation='mean', representation='fused'):
        if representation == 'text':
            if aggregation != 'mean':
                raise ValueError('Text-only diagnostic uses mean pooling')
            features = self.fusion_norm(self.text_projection(self.token_features(pieces)))
        elif representation != 'fused':
            raise ValueError('Unknown process representation')
        elif aggregation == 'mean':
            features = self.encode_case(pieces, context)
        elif aggregation == 'roles':
            # Window importance comes from PREDICTED roles, never gold roles.
            windows = []
            for offset in range(0, len(pieces), 8):
                tokens = self.pad_tokens(pieces[offset:offset+8])
                windows.append(nn.functional.normalize(self.encoder(**tokens).last_hidden_state[:, 0], dim=-1))
            vectors = torch.cat(windows)
            role_logits = torch.stack([self.process_heads['roles'](self.fuse_features(v, context)) for v in vectors])
            relevance = torch.sigmoid(role_logits[:, :3]).mean(-1)
            weights = torch.softmax(relevance, dim=0)
            pooled = nn.functional.normalize((weights[:, None] * vectors).sum(0), dim=-1)
            features = self.fuse_features(pooled, context)
        else:
            raise ValueError('Unknown evidence aggregation')
        return {k: head(features) for k, head in self.process_heads.items()}

    def adapter_state(self):
        # Frozen LoRA and inactive heads MUST remain in saved adapters.
        return {n: p.detach().cpu().contiguous() for n, p in self.named_parameters() if is_adapter(n)}

    def save_adapter(self, path):
        save_file(self.adapter_state(), str(path))

    def load_adapter(self, path):
        state = load_file(str(path), device='cpu')
        expected = self.adapter_state()
        if set(state) != set(expected) or any(state[k].shape != expected[k].shape or not torch.isfinite(state[k]).all() for k in state):
            raise ValueError('Invalid expert model checkpoint')
        self.load_state_dict(state, strict=False)


def parameter_digest(model, predicate):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if predicate(name):
            h.update(name.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def optimizer_groups(model, update_lora, active_process=()):
    buckets = {}
    for name, parameter in model.named_parameters():
        enabled = is_adapter(name)
        if is_lora(name):
            enabled = update_lora
            kind, lr = 'lora', 2e-5
        elif name.startswith('process_heads.'):
            enabled = name.split('.')[1] in active_process
            kind, lr = 'new_process', 3e-4
        else:
            kind, lr = 'downstream', 1e-4
        if name.startswith('heads.excerpt_severity.'):
            enabled = False
        parameter.requires_grad_(enabled)
        if not enabled:
            continue
        decay = 0.0 if name.endswith('.bias') or 'norm' in name.lower() else .01
        bucket = buckets.setdefault((kind, decay), {'params': [], 'names': [], 'lr': lr,
            'base_lr': lr, 'weight_decay': decay, 'group_name': kind})
        bucket['params'].append(parameter); bucket['names'].append(name)
    groups = list(buckets.values())
    optimized = [id(p) for g in groups for p in g['params']]
    if len(set(optimized)) != len(optimized) or set(optimized) != {id(p) for p in model.parameters() if p.requires_grad}:
        raise ValueError('Optimizer membership mismatch')
    return groups


def learning_rate_factor(step, total):
    warmup = max(1, math.ceil(total * .1))
    if step < warmup:
        return (step + 1) / warmup
    return max(0., (total - step) / max(1, total - warmup))


def main_record_loss(logits, row, weights, trained_classes):
    losses = {}
    for task in TASKS:
        label = row['targets'][task]
        if label is None:
            continue
        classes = MODEL_CLASSES[task]
        if label not in trained_classes[task]:
            raise ValueError('Target absent from training support')
        raw = logits[task]
        mask = torch.tensor([c in trained_classes[task] for c in classes], device=raw.device)
        target = torch.tensor([classes.index(label)], device=raw.device)
        losses[task] = nn.functional.cross_entropy(raw.masked_fill(~mask, -torch.inf)[None], target) * weights[task][row['sample_id']]
    return losses


def process_record_loss(predictions, labels, supported=None):
    """Average within each task in a record; missing labels never become negatives."""
    parts = {}
    for task in PROCESS_CLASSES:
        items = []
        for prediction, target in zip(predictions, labels, strict=True):
            label = target.get(task)
            if label is None:
                continue
            logits = prediction[task]
            if task == 'roles':
                if not isinstance(label, dict) or not label or set(label) - set(PROCESS_CLASSES[task]):
                    raise ValueError('Roles need explicit per-role binary labels; unspecified roles stay masked')
                indices = [PROCESS_CLASSES[task].index(k) for k in label]
                values = torch.tensor(list(label.values()), dtype=logits.dtype, device=logits.device)
                if not all(type(v) is bool for v in label.values()):
                    raise ValueError('Role labels must be boolean')
                items.append(nn.functional.binary_cross_entropy_with_logits(logits[indices], values))
            else:
                if label not in PROCESS_CLASSES[task]:
                    raise ValueError('Invalid process target')
                if supported is not None:
                    if label not in supported.get(task, []):
                        raise ValueError('Process target absent from declared training support')
                    mask = torch.tensor([c in supported[task] for c in PROCESS_CLASSES[task]], device=logits.device)
                    logits = logits.masked_fill(~mask, -torch.inf)
                items.append(nn.functional.cross_entropy(logits[None], torch.tensor([PROCESS_CLASSES[task].index(label)], device=logits.device)))
        if items:
            parts[task] = torch.stack(items).mean()
    return parts


def gradient_audit(groups):
    """Distinguish no gradient, a zero gradient and an effective update."""
    result = {}
    for group in groups:
        grads = [p.grad.detach() for p in group['params'] if p.grad is not None]
        result[group['group_name'] + '/' + str(group['weight_decay'])] = {
            'parameters_with_grad': len(grads),
            'norm': float(torch.sqrt(torch.stack([g.square().sum() for g in grads]).sum())) if grads else 0.0}
    return result
