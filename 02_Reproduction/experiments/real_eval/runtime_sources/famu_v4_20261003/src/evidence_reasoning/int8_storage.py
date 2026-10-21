"""Compiler-free, per-output-row INT8 storage for frozen linear weights.

Matmul remains in the activation dtype. Weights are deterministically quantized
on CPU; each linear dequantizes for computation. This is NOT LLM.int8 or NF4.
Embeddings and the tied language-model head retain their original precision.
"""
import hashlib
import torch
from torch.nn import functional as F


class FrozenInt8Linear(torch.nn.Linear):
    @classmethod
    def from_linear(cls, layer):
        device = layer.weight.device
        original = layer.weight.detach().cpu().float()
        if not torch.isfinite(original).all():
            raise ValueError('Cannot quantize non-finite weights')
        scales = original.abs().amax(dim=1, keepdim=True) / 127
        scales = torch.where(scales == 0, torch.ones_like(scales), scales)
        quantized = torch.round(original / scales).clamp(-127, 127).to(torch.int8)
        fingerprint = hashlib.sha256(quantized.numpy().tobytes() + scales.numpy().tobytes()).hexdigest()
        result = cls(layer.in_features, layer.out_features, bias=layer.bias is not None, device='meta')
        result.weight = torch.nn.Parameter(quantized.to(device), requires_grad=False)
        result.register_buffer('weight_scale', scales.to(device))
        if layer.bias is not None:
            result.bias = torch.nn.Parameter(layer.bias.detach(), requires_grad=False)
        return result, fingerprint

    def forward(self, value):
        weight = (self.weight.float() * self.weight_scale).to(value.dtype)
        return F.linear(value, weight, self.bias)


def quantize_frozen_linears(model):
    audit = {}
    # Snapshot names so replacing children does not invalidate traversal.
    names = [(name, module) for name, module in model.named_modules()
             if type(module) is torch.nn.Linear and name != 'lm_head']
    for name, module in names:
        replacement, fingerprint = FrozenInt8Linear.from_linear(module)
        parent_name, child = name.rsplit('.', 1) if '.' in name else ('', name)
        setattr(model.get_submodule(parent_name), child, replacement)
        audit[name] = fingerprint
    if not audit:
        raise ValueError('No frozen linear weights quantized')
    return {'algorithm': 'per_output_row_symmetric_int8_torch_v1',
            'rounding': 'CPU float32 nearest-even; max_abs/127; zero_row_scale=1',
            'excluded': ['embedding', 'lm_head'], 'module_sha256': audit,
            'module_count': len(audit), 'external_compiler_required': False}
