"""Predeclared exposure/weight experiment; no changes to historical training code."""
import math
import random
import re
import time
from collections import defaultdict

from .famu_llm import source_weights
from .review_ir import digest

TASK_MASS = {'condition': .20, 'qa': .20, 'risk_status': .20,
             'conditional_question_grade': .10, 'review_item_grade': .25, 'relation': .05}
REPORT_STEPS = (66, 264)
GRADIENT_STEPS = (0, 66, 264)
DATA_SHA = 'c6237368511d30eb9dd194937187116fad7cca445fd95776161a1be3eba36380'


def task_weights(rows, scheme):
    weights = source_weights(rows)
    if scheme == 'W0':
        return weights
    if scheme != 'W1':
        raise ValueError('Unknown weight scheme')
    masses = defaultdict(float)
    for row in rows:
        masses[row['task']] += weights[row['id']]
    if set(masses) != set(TASK_MASS):
        raise ValueError('All six tasks are required')
    total = math.fsum(weights.values())
    return {r['id']: total * TASK_MASS[r['task']] * weights[r['id']] / masses[r['task']] for r in rows}


def accumulated_objective(loss, weight, actual_batch_size):
    """Mean of weighted per-example completion-token means (not sum-weight normalization)."""
    if actual_batch_size < 1 or weight <= 0:
        raise ValueError('Positive weight and batch denominator required')
    return loss * weight / actual_batch_size


def absolute_lr(completed_updates):
    if not 0 <= completed_updates < 264:
        raise ValueError('Update outside fixed trajectory')
    return 2e-5 * (1 - completed_updates / 264)


def anchor_rows(rows):
    """Two source groups per task where available, selected without consulting targets."""
    result = []
    for task in sorted(TASK_MASS):
        groups = defaultdict(list)
        for r in rows:
            if r['split'] != 'train':
                raise ValueError('Gradient anchors must be training only')
            if r['task'] == task:
                groups[r['group_id']].append(r)
        selected_groups = sorted(groups, key=lambda g: digest('anchor-v1:' + g))[:2]
        for group in selected_groups:
            result.append(min(groups[group], key=lambda r: digest('anchor-v1:' + r['id'])))
    return result


def attributed_loss(model, encoded, parts, device):
    """Same completion CE as v1, additionally exposing detached label/format/EOS terms."""
    import torch
    ids = torch.tensor([encoded['input_ids']], device=device)
    positions = torch.tensor([p['target_position'] for p in parts], device=device)
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                   logits_to_keep=positions - 1).logits[0]
    ce = torch.nn.functional.cross_entropy(logits.float(), ids[0, positions], reduction='none')
    values = ce.detach().cpu().tolist()
    by_part = {}
    for kind in ('label_value', 'json_format', 'eos'):
        subset = [v for p, v in zip(parts, values) if p['kind'] == kind]
        by_part[kind] = {'sum_ce': sum(subset), 'n': len(subset),
                         'mean_ce': sum(subset) / len(subset) if subset else None}
    return ce.mean(), by_part


def gradient_snapshot(model, encoded, parts, rows, weights, device, step):
    """Train-mode anchor gradients on first/middle/last LoRA layers; restore all RNGs.

    This is a declared parameter slice and a fixed sample estimate, never a claim
    about the full dataset gradient or full-model task conflict.
    """
    import torch
    started = time.perf_counter()
    all_named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    layers = sorted({int(m.group(1)) for n, _ in all_named if (m := re.search(r'\.layers\.(\d+)\.', n))})
    if not layers:
        raise ValueError('Cannot identify shared LoRA layers')
    selected = {layers[0], layers[len(layers) // 2], layers[-1]}
    named = [(n, p) for n, p in all_named if (m := re.search(r'\.layers\.(\d+)\.', n)) and int(m.group(1)) in selected]
    cpu_rng = torch.get_rng_state()
    py_rng = random.getstate()
    device_rng = getattr(torch, device).get_rng_state_all() if device != 'cpu' else []
    prior_training = model.training
    vectors, details = {}, []
    try:
        model.train()
        for task in sorted(TASK_MASS):
            task_rows = [r for r in rows if r['task'] == task]
            raw, weighted = None, None
            for row in task_rows:
                model.zero_grad(set_to_none=True)
                torch.manual_seed(int(digest('gradient:' + row['id'])[:8], 16))
                loss, ce = attributed_loss(model, encoded[row['id']], parts[row['id']], device)
                loss.backward()
                vector = torch.cat([(p.grad.detach().float().cpu().reshape(-1) if p.grad is not None
                                     else torch.zeros(p.numel())) for _, p in named])
                raw = vector.clone() if raw is None else raw + vector
                weighted = vector * weights[row['id']] if weighted is None else weighted + vector * weights[row['id']]
                details.append({'id': row['id'], 'task': task, 'group_id': row['group_id'],
                                'weight': weights[row['id']], 'unweighted_gradient_norm': float(vector.norm()),
                                'weighted_gradient_norm': float(vector.norm()) * weights[row['id']], 'ce_by_part': ce})
            vectors[task] = {'unweighted': raw / len(task_rows), 'weighted': weighted / len(task_rows)}
        norms = {t: {k: float(v.norm()) for k, v in values.items()} for t, values in vectors.items()}
        cosines = []
        tasks = sorted(vectors)
        for i, a in enumerate(tasks):
            for b in tasks[i + 1:]:
                item = {'task_a': a, 'task_b': b}
                for kind in ('unweighted', 'weighted'):
                    x, y = vectors[a][kind], vectors[b][kind]
                    denominator = float(x.norm() * y.norm())
                    item[kind] = float(x.dot(y)) / denominator if denominator else None
                cosines.append(item)
        return {'step': step, 'scope': 'fixed_train_anchor_mean_on_first_middle_last_LoRA_layers',
                'input_tokens': sum(encoded[r['id']]['tokens'] for r in rows),
                'seconds': time.perf_counter() - started,
                'layer_indices': sorted(selected), 'parameter_names': [n for n, _ in named],
                'parameter_count': sum(p.numel() for _, p in named), 'anchors': details,
                'gradient_norms': norms, 'cosines': cosines, 'clipping_applied': False,
                'dropout': 'train_mode_fixed_per_anchor_seed', 'rng_restored': True}
    finally:
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng)
        random.setstate(py_rng)
        if device != 'cpu':
            getattr(torch, device).set_rng_state_all(device_rng)
        model.train(prior_training)
