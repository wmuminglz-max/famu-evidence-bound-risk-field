# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Adapted from Hugging Face Transformers 5.8.0 modeling_qwen3_5.py
# https://github.com/huggingface/transformers (Apache License 2.0).
# Change: solve the unit-lower-triangular system directly instead of retaining
# 63 growing row-update graphs. XPU uses an explicit CPU triangular solve, with
# differentiable device copies; the recurrent state and projections stay on XPU.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy at
# https://example.invalid/source-withheld . Unless required by applicable law
# or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
"""Memory-bounded equivalent of the upstream torch gated delta rule."""
import torch
from torch.nn import functional as F


def compact_gated_delta_rule(query, key, value, g, beta, chunk_size=64,
                           initial_state=None, output_final_state=False,
                           use_qk_l2norm_in_kernel=False):
    dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = query * torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
    q, k, v, beta, g = [x.transpose(1, 2).contiguous().float() for x in (query, key, value, beta, g)]
    batch, heads, length, key_dim = k.shape
    value_dim = v.shape[-1]
    padding = (-length) % chunk_size
    q, k, v = [F.pad(x, (0, 0, 0, padding)) for x in (q, k, v)]
    beta, g = [F.pad(x, (0, padding)) for x in (beta, g)]
    q = q / key_dim ** .5
    kb, vb = k * beta.unsqueeze(-1), v * beta.unsqueeze(-1)
    q, k, v, kb, vb = [x.reshape(batch, heads, -1, chunk_size, x.shape[-1]) for x in (q, k, v, kb, vb)]
    g = g.reshape(batch, heads, -1, chunk_size).cumsum(-1)
    decay = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().tril()
    lower = ((kb @ k.transpose(-1, -2)) * decay).tril(-1)
    identity = torch.eye(chunk_size, device=q.device, dtype=q.dtype)
    system = lower + identity
    # The stock Windows XPU build otherwise implicitly falls back for this op.
    # Make that transfer explicit and keep its autograd connection intact.
    solver = system.cpu() if system.device.type == 'xpu' else system
    inverse = torch.linalg.solve_triangular(solver, torch.eye(chunk_size, device=solver.device,
        dtype=solver.dtype).expand_as(solver), upper=False, unitriangular=True).to(q.device)
    transformed_v = inverse @ vb
    transformed_k = inverse @ (kb * g.exp().unsqueeze(-1))
    state = (torch.zeros(batch, heads, key_dim, value_dim, device=q.device, dtype=q.dtype)
             if initial_state is None else initial_state.to(q))
    outputs = []
    for index in range(q.shape[2]):
        qi, ki = q[:, :, index], k[:, :, index]
        delta_v = transformed_v[:, :, index] - transformed_k[:, :, index] @ state
        within = (qi @ ki.transpose(-1, -2)) * decay[:, :, index]
        carry = (qi * g[:, :, index, :, None].exp()) @ state
        outputs.append(carry + within @ delta_v)
        state = (state * g[:, :, index, -1, None, None].exp() +
                 (ki * (g[:, :, index, -1, None] - g[:, :, index]).exp()[..., None]).transpose(-1, -2) @ delta_v)
    output = torch.cat(outputs, dim=2)[:, :, :length].transpose(1, 2).contiguous().to(dtype)
    return output, state if output_final_state else None


def enable_compact_delta(model):
    count = 0
    for name, module in model.named_modules():
        if name.endswith('.linear_attn') and hasattr(module, 'chunk_gated_delta_rule'):
            module.chunk_gated_delta_rule = compact_gated_delta_rule
            count += 1
    if not count:
        raise ValueError('No gated delta layers found')
    return count
