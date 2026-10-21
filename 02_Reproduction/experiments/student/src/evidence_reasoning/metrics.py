"""Root-balanced metrics. Technical failures stay in the coverage denominator."""
from __future__ import annotations

from collections import Counter
import numpy as np

LABELS = ("T", "F", "U")


def weights(roots, groups=None):
    counts = Counter(roots)
    if groups is None:
        return np.array([1. / counts[root] for root in roots], dtype=float)
    if len(roots) != len(groups):
        raise ValueError('Unaligned source groups')
    membership, group_roots = {}, {}
    for root, group in zip(roots, groups):
        if root in membership and membership[root] != group:
            raise ValueError('One root spans independent source groups')
        membership[root] = group
        group_roots.setdefault(group, set()).add(root)
    return np.array([1. / len(group_roots[group]) / counts[root] for root, group in zip(roots, groups)], dtype=float)


def classification(y, predictions, roots, probabilities=None, groups=None):
    if not (len(y) == len(predictions) == len(roots)) or (groups is not None and len(groups) != len(y)):
        raise ValueError("Unaligned evaluation rows")
    truth, pred = np.asarray(y, dtype=int), np.asarray(predictions, dtype=int)
    eligible = (truth >= 0) & (truth < 3)
    weight = np.zeros(len(truth), dtype=float)
    weight[eligible] = weights(np.asarray(roots)[eligible], np.asarray(groups)[eligible] if groups is not None else None)
    valid = eligible & (pred >= 0) & (pred < 3)
    mass = float(weight[eligible].sum())
    confusion = np.zeros((3, 4))  # Last column = technical failure / no prediction.
    for target, answer, w in zip(truth[eligible], pred[eligible], weight[eligible]):
        confusion[target, answer if 0 <= answer < 3 else 3] += w
    scores = []
    for c in range(3):
        tp = confusion[c, c]
        fp, fn = confusion[:, c].sum() - tp, confusion[c].sum() - tp
        scores.append(float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.)
    def rate(mask, denominator):
        value = float(weight[denominator].sum())
        return float(weight[mask].sum() / value) if value else None
    result = {"n": int(eligible.sum()), "roots": len(set(np.asarray(roots)[eligible])),
              "groups": len(set(np.asarray(groups)[eligible])) if groups is not None else None,
              "weighting": "source_group_root_view_balanced" if groups is not None else "root_balanced",
              "accuracy": float(weight[valid & (truth == pred)].sum() / mass) if mass else None,
              "macro_f1": float(np.mean(scores)) if mass else None,
              "per_class_f1": dict(zip(LABELS, scores)), "confusion": confusion.tolist(),
              "per_class_recall": {label: float(confusion[i, i] / confusion[i].sum()) if confusion[i].sum() else None
                                   for i, label in enumerate(LABELS)},
              "coverage": rate(valid, eligible),
              "determinate_coverage": rate(valid & (pred != 2), eligible),
              "u_subset_coverage": rate(valid & (truth == 2), eligible & (truth == 2)),
              "u_recall": rate(valid & (truth == 2) & (pred == 2), eligible & (truth == 2)),
              "false_certainty_among_accepted_u": rate(valid & (truth == 2) & (pred != 2), valid & (truth == 2)),
              "false_certainty": rate(eligible & (truth == 2) & ((pred == 0) | (pred == 1)), eligible & (truth == 2)),
              "unnecessary_unknown": rate(eligible & (truth != 2) & (pred == 2), eligible & (truth != 2)),
              "determinate_flip": rate(eligible & (truth != 2) & (pred < 2) & (pred >= 0) & (truth != pred), eligible & (truth != 2))}
    if probabilities is not None and eligible.any():
        p = np.asarray(probabilities, dtype=float)[eligible]
        if p.shape != (int(eligible.sum()), 3) or not np.all(np.isfinite(p)) or np.any(p < 0) or not np.allclose(p.sum(1), 1):
            raise ValueError("Probabilities must be finite, normalized and aligned")
        target, w = truth[eligible], weight[eligible]
        result["nll"] = float(np.average(-np.log(np.clip(p[np.arange(len(p)), target], 1e-9, 1)), weights=w))
        result["brier"] = float(np.average(((p - np.eye(3)[target]) ** 2).sum(1), weights=w))
        confidence, correct = p.max(1), p.argmax(1) == target
        ece = 0.
        for b in range(10):
            mask = (confidence >= b / 10) & (confidence <= 1 if b == 9 else confidence < (b + 1) / 10)
            if mask.any():
                ece += w[mask].sum() / w.sum() * abs(np.average(correct[mask], weights=w[mask]) - np.average(confidence[mask], weights=w[mask]))
        result["ece_10_bins"] = float(ece)
    return result


def group_bootstrap_difference(y, a, b, roots, iterations=2000, seed=20260920, groups=None, metric="accuracy"):
    """Paired source-group bootstrap, recomputing the named root-balanced metric."""
    metrics = {'accuracy', 'macro_f1', 'false_certainty', 'unnecessary_unknown', 'determinate_flip',
               'coverage', 'determinate_coverage', 'u_subset_coverage', 'u_recall', 'false_certainty_among_accepted_u'}
    if metric not in metrics or iterations < 1:
        raise ValueError("Unsupported metric or nonpositive bootstrap iterations")
    if not (len(y) == len(a) == len(b) == len(roots)) or (groups is not None and len(groups) != len(y)):
        raise ValueError("Unaligned bootstrap rows")
    y, a, b, roots = map(np.asarray, (y, a, b, roots))
    source_balanced = groups is not None
    groups = np.asarray(groups) if source_balanced else roots
    valid = (y >= 0) & (y < 3)
    y, a, b, roots, groups = [value[valid] for value in (y, a, b, roots, groups)]
    summary = {"metric": metric, "roots": len(set(roots)), "groups": len(set(groups)),
               "iterations": iterations, "unit": "group", "weighting": "source_group_root_view_balanced" if source_balanced else "root_balanced", "seed": seed}
    if not len(y):
        return summary | {"difference": None, "ci95": None}
    for root in set(roots):
        if len(set(groups[roots == root])) != 1:
            raise ValueError("One root spans independent bootstrap groups")
    ordered = sorted(set(groups))
    index = {g: i for i, g in enumerate(ordered)}
    counts = np.zeros((len(ordered), 2, 3, 4))
    for target, pa, pb, group, w in zip(y, a, b, groups, weights(roots, groups if source_balanced else None)):
        for model, pred in enumerate((pa, pb)):
            counts[index[group], model, int(target), int(pred) if 0 <= pred < 3 else 3] += w
    def score(c):
        def ratio(numerator, denominator):
            return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan, dtype=float), where=denominator > 0)
        tp = np.diagonal(c[..., :3], axis1=-2, axis2=-1)
        if metric == "accuracy":
            return tp.sum(-1) / c.sum(axis=(-2, -1))
        if metric == 'false_certainty':
            return ratio(c[..., 2, :2].sum(-1), c[..., 2, :].sum(-1))
        if metric == 'false_certainty_among_accepted_u':
            return ratio(c[..., 2, :2].sum(-1), c[..., 2, :3].sum(-1))
        if metric == 'unnecessary_unknown':
            return ratio(c[..., :2, 2].sum(-1), c[..., :2, :].sum((-2, -1)))
        if metric == 'determinate_flip':
            return ratio(c[..., 0, 1]+c[..., 1, 0], c[..., :2, :].sum((-2, -1)))
        if metric in {'coverage', 'determinate_coverage'}:
            return ratio(c[..., :3 if metric == 'coverage' else 2].sum((-2, -1)), c.sum((-2, -1)))
        if metric == 'u_subset_coverage':
            return ratio(c[..., 2, :3].sum(-1), c[..., 2, :].sum(-1))
        if metric == 'u_recall':
            return ratio(c[..., 2, 2], c[..., 2, :].sum(-1))
        denominator = c.sum(-1) + c[..., :3].sum(-2)
        return np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0).mean(-1)
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(ordered), (iterations, len(ordered)))
    values = score(counts[selected].sum(1))
    point = score(counts.sum(0))
    deltas = values[:, 0] - values[:, 1]
    finite = deltas[np.isfinite(deltas)]
    delta = point[0] - point[1]
    return summary | {"difference": float(delta) if np.isfinite(delta) else None,
                      "valid_iterations": len(finite), "undefined_iterations": iterations-len(finite),
                      "ci95": np.quantile(finite, [.025, .975]).tolist() if len(finite) else None}


def pair_metrics(rows, probabilities):
    index = {r["meta"]["sample_id"]: i for i, r in enumerate(rows)}
    pairs = {tuple(p) for r in rows for p in r["meta"].get("pair_ids", []) if all(s in index for s in p)}
    deltas, correct, roots, groups = [], [], [], []
    for a, b in sorted(pairs):
        ia, ib = index[a], index[b]
        if rows[ia]['meta']['group_id'] != rows[ib]['meta']['group_id'] or rows[ia]['meta']['root_id'] != rows[ib]['meta']['root_id']:
            raise ValueError('Boundary pairs must share the same source group and root')
        roots.append(rows[ia]['meta']['root_id'])
        groups.append(rows[ia]['meta']['group_id'])
        if rows[ia]["targets"].get("label") != "U" or rows[ib]["targets"].get("label") not in ("T", "F"):
            raise ValueError("Boundary pair does not match frozen semantics")
        deltas.append(float(probabilities[ia, 2] - probabilities[ib, 2]))
        correct.append(int(probabilities[ia].argmax()) == 2 and LABELS[int(probabilities[ib].argmax())] == rows[ib]["targets"]["label"])
    w = weights(roots, groups)
    return {"pairs": len(pairs), 'roots': len(set(roots)), 'groups': len(set(groups)),
            'weighting': 'source_group_root_pair_balanced',
            "both_correct": float(np.average(correct, weights=w)) if correct else None,
            "mean_unknown_margin": float(np.average(deltas, weights=w)) if deltas else None}


def complementary(y, model_predictions, roots, groups=None):
    y = np.asarray(y)
    eligible = (y >= 0) & (y < 3)
    matrix = np.stack([np.asarray(p) == y for p in model_predictions], 1)
    w = weights(np.asarray(roots)[eligible], np.asarray(groups)[eligible] if groups is not None else None)
    individual = [float(np.average(matrix[eligible, i], weights=w)) for i in range(matrix.shape[1])] if len(w) else []
    oracle = float(np.average(matrix[eligible].any(1), weights=w)) if len(w) else None
    return {"oracle_accuracy": float(np.average(matrix[eligible].any(1), weights=w)) if len(w) else None,
            'individual_accuracy': individual, 'best_single_accuracy': max(individual) if individual else None,
            'oracle_gain': oracle-max(individual) if individual else None,
            "all_correct": int(matrix[eligible].all(1).sum()),
            "all_wrong": int((~matrix[eligible].any(1)).sum()),
            "exclusive_correct": [int((matrix[eligible, i] & (matrix[eligible].sum(1) == 1)).sum()) for i in range(matrix.shape[1])]}


def grouped_scalar_difference(a, b, roots, groups, iterations=2000, seed=20260929):
    """Paired group interval for boundary accuracy, citations or scope indicators.

    Input vectors must already encode the same eligible denominator; None is not
    silently replaced with zero. Use separate denominators for other tasks.
    """
    if iterations < 1 or not len(a) or not (len(a) == len(b) == len(roots) == len(groups)):
        raise ValueError('Aligned nonempty pairs and positive iterations required')
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Missing scalar observations require an explicit denominator')
    w = weights(roots, groups)
    delta = a-b
    values = np.array([np.sum(w[np.asarray(groups) == g]*delta[np.asarray(groups) == g]) for g in sorted(set(groups))])
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(0, len(values), (iterations, len(values)))].mean(1)
    return {'difference': float(values.mean()), 'ci95': np.quantile(sample, [.025, .975]).tolist(),
            'groups': len(values), 'iterations': iterations, 'seed': seed, 'weighting': 'source_group_root_item_balanced'}


def severity_metrics(truth, predicted, roots, groups=None):
    y, p = np.asarray(truth, int), np.asarray(predicted, int)
    valid = (y >= 0) & (y < 4)
    w = weights(np.asarray(roots)[valid], np.asarray(groups)[valid] if groups is not None else None)
    y, p = y[valid], p[valid]
    known = (p >= 0) & (p < 4)
    def rate(mask, denominator):
        return float(w[mask].sum()/w[denominator].sum()) if w[denominator].sum() else None
    return {'n': len(y), 'coverage': rate(known, np.ones(len(y), bool)),
            'mae_accepted': float(np.average(abs(y[known]-p[known]), weights=w[known])) if known.any() else None,
            'per_level_recall': {str(k): rate((y == k) & (p == k), y == k) for k in range(4)},
            'high_critical_underestimate': rate((y >= 2) & known & (p < y), y >= 2),
            'high_critical_missing': rate((y >= 2) & ~known, y >= 2)}


def calibrate_coverage_threshold(scores, roots, groups, splits, target_coverage):
    """Choose the highest score cutoff meeting prespecified calibration coverage.

    No truth labels enter threshold selection; a test group is never inspected.
    Ties may exceed the target and the actual calibration coverage is reported.
    """
    if not len(scores) or not (len(scores) == len(roots) == len(groups) == len(splits)) or set(splits) != {'calibration'}:
        raise ValueError('Coverage threshold selection requires aligned calibration groups')
    values = np.asarray(scores, float)
    if not np.isfinite(values).all() or not 0 < target_coverage <= 1:
        raise ValueError('Finite scores and a coverage target in (0,1] are required')
    w = weights(roots, groups)
    threshold = max(v for v in sorted(set(values)) if w[values >= v].sum()/w.sum() >= target_coverage)
    return {'threshold': float(threshold), 'target_coverage': target_coverage,
            'calibration_coverage': float(w[values >= threshold].sum()/w.sum()),
            'fit_groups': sorted(set(groups)), 'selection_split': 'calibration',
            'score_scope': 'model score; not a probability of legal correctness'}


def evaluate_at_coverage_threshold(y, probabilities, scores, roots, groups, frozen_threshold, workflow_eligible=None):
    if set(groups) & set(frozen_threshold['fit_groups']):
        raise ValueError('Evaluation source groups overlap threshold fitting')
    p, scores = np.asarray(probabilities, float), np.asarray(scores, float)
    if p.shape != (len(y),3) or len(scores) != len(y) or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1),1):
        raise ValueError('Finite normalized aligned probabilities required')
    allowed = np.ones(len(y), bool) if workflow_eligible is None else np.asarray(workflow_eligible, bool)
    if allowed.shape != (len(y),): raise ValueError('Workflow eligibility must align')
    accepted = allowed & np.isfinite(scores) & (scores >= frozen_threshold['threshold'])
    prediction = np.where(accepted, p.argmax(1), -1)
    return classification(y, prediction, roots, groups=groups) | {'frozen_threshold': frozen_threshold['threshold'],
        'requested_calibration_coverage': frozen_threshold['target_coverage']}
