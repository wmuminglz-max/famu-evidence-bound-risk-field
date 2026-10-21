"""Explicit NF4 QLoRA loading and custom expert adapter restoration."""
from pathlib import Path
import json

TARGETS = ['q_proj','k_proj','v_proj','o_proj','in_proj_qkv','in_proj_z',
           'in_proj_b','in_proj_a','out_proj','gate_proj','up_proj','down_proj']


def load_27b_base(folder):
    import torch
    import bitsandbytes as bnb
    from peft import prepare_model_for_kbit_training
    from .famu_llm import load_model
    from .compact_delta import enable_compact_delta
    from .chunked_attention import enable_chunked_mlp
    # K-bit preparation keeps residuals/norms in FP32; a full SDPA math
    # fallback can allocate quadratic long-context score tensors. Query
    # blocks preserve all causal keys and bound that allocation.
    model, audit = load_model(folder, 'cuda', quantization='nf4', attention='chunked')
    audit['compact_delta_layers'] = enable_compact_delta(model)
    audit['checkpointed_mlp_layers'] = enable_chunked_mlp(model, 512)
    audit['famu_mlp_token_block'] = 512
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                          gradient_checkpointing_kwargs={'use_reentrant':False})
    modules = [(n,m) for n,m in model.named_modules() if isinstance(m,bnb.nn.Linear4bit)]
    if not modules or not all(m.weight.quant_state is not None for _,m in modules):
        raise ValueError('NF4 quantization state missing')
    audit['nf4_modules'] = len(modules)
    audit['quantization'] = 'NF4 double quantization; BF16 compute; PEFT k-bit preparation'
    audit['stored_parameter_elements_including_packed_nf4'] = sum(p.numel() for p in model.parameters())
    assert not any(p.requires_grad for p in model.parameters())
    assert not any('visual' in n or n.startswith('mtp') for n,_ in model.named_parameters())
    return model, audit


def restore_adapter(model, folder):
    from safetensors.torch import load_file
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    weights = load_file(str(Path(folder)/'adapter_model.safetensors'))
    current = get_peft_model_state_dict(model)
    if set(weights) != set(current):
        raise ValueError('Adapter architecture does not match saved parameters')
    set_peft_model_state_dict(model, weights)
    restored = get_peft_model_state_dict(model)
    import torch
    if not all(torch.equal(v.cpu(), restored[k].detach().cpu()) for k,v in weights.items()):
        raise ValueError('Adapter did not reload exactly')


def offload_checkpoint_inputs(model):
    """Store only decoder checkpoint inputs on CPU, retaining their exact dtype.

    PyTorch's inner non-reentrant checkpoint hook discards intermediates as
    usual. The outer save_on_cpu hook handles the checkpoint's saved inputs;
    model weights and the active layer's computation remain on CUDA.
    """
    import torch
    from functools import wraps
    count = 0
    for module in model.modules():
        if not getattr(module, 'gradient_checkpointing', False):
            continue
        original = getattr(module, '_gradient_checkpointing_func', None)
        if original is None or getattr(original, '_famu_cpu_inputs', False):
            continue
        @wraps(original)
        def checkpoint_on_cpu(*args, _original=original, **kwargs):
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                return _original(*args, **kwargs)
        checkpoint_on_cpu._famu_cpu_inputs = True
        module._gradient_checkpointing_func = checkpoint_on_cpu
        count += 1
    if not count:
        raise ValueError('No decoder checkpoints available for CPU input storage')
    return count


def new_adapter(model, architecture='baseline', checkpoint_offload=True):
    from peft import get_peft_model,LoraConfig
    from .field_experts import attach_field_experts
    model = get_peft_model(model,LoraConfig(r=32,lora_alpha=64,lora_dropout=.05,
                target_modules=TARGETS,bias='none',task_type='CAUSAL_LM'))
    if architecture != 'baseline':
        attach_field_experts(model, architecture)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    if checkpoint_offload:
        offload_checkpoint_inputs(model)
    model.enable_input_require_grads()
    model.config.use_cache=False
    return model
