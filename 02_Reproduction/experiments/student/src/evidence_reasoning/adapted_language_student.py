# Deidentified copy. String literals containing machine locations were replaced.
"""R9 small-language shadow runtime; trainable adapters, unchanged evidence rules."""
import json
from pathlib import Path
import numpy as np
from .semantic_reward_student import SemanticRewardStudent, FrozenEncoder, file_sha
VERSION = 'adapted-language-student-r9-v1'

def install_lora(model, rank=8, alpha=16):
    import torch
    from torch import nn

    class LowRankLinear(nn.Module):

        def __init__(self, base):
            super().__init__()
            self.base = base
            self.base.requires_grad_(False)
            self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
            self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype))
            nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
            self.scale = alpha / rank

        def forward(self, x):
            return self.base(x) + x @ self.lora_A.T @ self.lora_B.T * self.scale
    targets = [(n, m) for n, m in model.named_modules() if n.startswith('encoder.layer.') and isinstance(m, nn.Linear)]
    if len(targets) != 24:
        raise ValueError('Expected 24 BERT layer linear modules')
    for name, base in targets:
        parent, _, leaf = name.rpartition('.')
        setattr(model.get_submodule(parent), leaf, LowRankLinear(base))
    return [n for n, _ in targets]

def adapter_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k.endswith(('lora_A', 'lora_B'))}

def load_adapter(model, values):
    current = model.state_dict()
    keys = {k for k in current if k.endswith(('lora_A', 'lora_B'))}
    if set(values) != keys:
        raise ValueError('Adapter keys differ')
    import torch
    if any((values[k].shape != current[k].shape or not torch.isfinite(values[k]).all() for k in keys)):
        raise ValueError('Adapter shape/finite')
    result = model.load_state_dict(values, strict=False)
    if result.unexpected_keys or set(result.missing_keys) != set(current) - keys:
        raise ValueError('Adapter load mismatch')

class AdaptedLanguageStudent(SemanticRewardStudent):

    @classmethod
    def load(cls, folder, encoder_path, device='cpu'):
        import torch
        p = Path(folder)
        manifest = json.loads((p / 'manifest.json').read_text('utf-8'))
        if set(manifest) != {'weights.npz', 'adapter.npz', 'student_config.json'}:
            raise ValueError('R9 manifest')
        for n, h in manifest.items():
            if (p / n).stat().st_size > 8 * 1024 ** 2 or file_sha(p / n) != h:
                raise ValueError('R9 hash/size')
        cfg = json.loads((p / 'student_config.json').read_text('utf-8'))
        if cfg['version'] != VERSION or cfg['representation'] != 'evidence':
            raise ValueError('R9 config')
        enc = FrozenEncoder(encoder_path, cfg['encoder_sha256'], device)
        install_lora(enc.model, cfg['lora_rank'], cfg['lora_alpha'])
        with np.load(p / 'adapter.npz', allow_pickle=False) as z:
            load_adapter(enc.model, {k: torch.from_numpy(z[k].copy()) for k in z.files})
        enc.model.requires_grad_(False)
        enc.model.eval()
        with np.load(p / 'weights.npz', allow_pickle=False) as z:
            a = {k: z[k].copy() for k in z.files}
        from .semantic_reward_student import VERSION as R8_VERSION
        obj = cls(cfg | {'version': R8_VERSION}, a, enc)
        obj.options = cfg
        obj.config = cfg
        obj.version = VERSION
        return obj

    def review(self, payload):
        return super().review(payload) | {'version': VERSION, 'mode': 'shadow'}
