"""Exact causal attention in query blocks, with recomputation during backward.

Every query still attends to every allowed earlier key. This changes the memory
schedule, not context length or the model's attention pattern. The mask adapter
deliberately rejects packing, sliding windows and arbitrary mask factories.
"""
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

BACKEND = 'famu_chunked_sdpa'


@dataclass(frozen=True)
class CausalMask:
    padding: torch.Tensor | None
    query_offset: int
    query_length: int
    key_length: int


def mask_interface(*, batch_size, q_length, kv_length, q_offset, kv_offset,
                   mask_function, attention_mask, config, use_vmap=False, **kwargs):
    from transformers.masking_utils import causal_mask_function
    if (mask_function is not causal_mask_function or use_vmap or kv_offset != 0
            or not getattr(config, 'is_causal', True)):
        raise ValueError('Chunked attention requires ordinary non-packed causal attention')
    if attention_mask is not None and attention_mask.shape != (batch_size, kv_length):
        raise ValueError('Unexpected padding-mask shape')
    return CausalMask(attention_mask, int(q_offset), q_length, kv_length)


def chunked_attention_forward(module, query, key, value, attention_mask,
                              dropout=0.0, scaling=None, **kwargs):
    if dropout:
        raise ValueError('This deterministic backend requires attention dropout=0')
    batch, heads, q_length, _ = query.shape
    kv_length = key.shape[-2]
    if key.shape != value.shape or heads % key.shape[1] or kv_length < q_length:
        raise ValueError('Unsupported grouped-key/value attention shapes')
    chunk_size = int(getattr(module.config, 'famu_attention_chunk_size', 128))
    if chunk_size < 1:
        raise ValueError('Attention chunk size must be positive')
    if isinstance(attention_mask, CausalMask):
        if (attention_mask.query_length, attention_mask.key_length) != (q_length, kv_length):
            raise ValueError('Causal mask and attention sequence lengths differ')
        offset, padding = attention_mask.query_offset, attention_mask.padding
    elif attention_mask is None:
        offset, padding = kv_length - q_length, None
    else:
        raise ValueError('Unsupported attention mask')
    repeats = heads // key.shape[1]
    outputs = []
    for start in range(0, q_length, chunk_size):
        end = min(q_length, start + chunk_size)
        # Keys beyond this block's last query are causally impossible. Slicing
        # them saves work without dropping any permitted attention connection.
        key_end = min(kv_length, offset + end)
        qs = torch.arange(offset + start, offset + end, device=query.device)
        ks = torch.arange(key_end, device=query.device)
        block_mask = (ks[None, :] <= qs[:, None])[None, None, :, :]
        if padding is not None:
            block_mask = block_mask & padding[:, None, None, :key_end].bool()

        def block(q, k, v, mask):
            # Explicit repeat works on CPU and XPU, independent of GQA kernels.
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                                  dropout_p=0.0, is_causal=False, scale=scaling)

        args = (query[:, :, start:end], key[:, :, :key_end], value[:, :, :key_end], block_mask)
        if torch.is_grad_enabled() and any(t.requires_grad for t in args[:3]):
            output = checkpoint(block, *args, use_reentrant=False)
        else:
            output = block(*args)
        outputs.append(output)
    return torch.cat(outputs, dim=2).transpose(1, 2).contiguous(), None


def register():
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface
    AttentionInterface.register(BACKEND, chunked_attention_forward)
    AttentionMaskInterface.register(BACKEND, mask_interface)
    return BACKEND


def chunked_mlp_forward(module, hidden_states):
    """Token-independent gated MLP, checkpointed in small token blocks."""
    chunks = []
    size = module._famu_mlp_chunk_size
    for start in range(0, hidden_states.shape[-2], size):
        block = hidden_states[..., start:start + size, :]
        if torch.is_grad_enabled() and block.requires_grad:
            chunks.append(checkpoint(module._famu_original_forward, block, use_reentrant=False))
        else:
            chunks.append(module._famu_original_forward(block))
    return torch.cat(chunks, dim=-2)


def enable_chunked_mlp(model, chunk_size=256):
    from types import MethodType
    if chunk_size < 1:
        raise ValueError('MLP block size must be positive')
    count = 0
    for name, module in model.named_modules():
        if name.endswith('.mlp') and all(hasattr(module, x) for x in ('gate_proj', 'up_proj', 'down_proj')):
            if not hasattr(module, '_famu_original_forward'):
                module._famu_original_forward = module.forward
                module.forward = MethodType(chunked_mlp_forward, module)
            module._famu_mlp_chunk_size = chunk_size
            count += 1
    if not count:
        raise ValueError('No compatible gated MLP layers found')
    return count
