# Deidentified copy. String literals containing machine locations were replaced.
"""Dedicated supervised severity MLP, isolated from missing severity annotations."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import f1_score, confusion_matrix
from .schema import model_input
from .provenance import visible_evidence
from .data import read_records, write_json, audit_splits
from .metrics import weights
from .graphs import FrozenEncoder

def features(payload, encoder):
    x = model_input(payload)
    evidence = visible_evidence(x)
    by_id = {e['id']: e['text'] for e in evidence}
    texts = [x['proposition'], by_id.get('clause_a', evidence[0]['text'] if evidence else ''), by_id.get('clause_b', evidence[1]['text'] if len(evidence) > 1 else ''), by_id.get('fact_context', '\n'.join((e['text'] for e in evidence[2:])))]
    q, a, b, f = encoder.encode(texts)
    return np.concatenate([q, a, b, f, np.abs(a - b), a * b]).astype(np.float32)

class SeverityMLP(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.body = nn.Sequential(nn.Linear(input_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.15), nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1))
        self.categorical = nn.Linear(64, 4)
        self.ordinal = nn.Linear(64, 3)

    def forward(self, x):
        h = self.body(x)
        return (self.categorical(h), self.ordinal(h))

def _evaluate(rows, p):
    y = np.array([r['targets']['severity'] for r in rows])
    w = weights([r['meta']['root_id'] for r in rows])
    prediction = p.argmax(1)
    return {'samples': len(rows), 'roots': len(set((r['meta']['root_id'] for r in rows))), 'accuracy': float(np.average(y == prediction, weights=w)), 'macro_f1': float(f1_score(y, prediction, labels=range(4), average='macro', sample_weight=w, zero_division=0)), 'mae': float(np.average(np.abs(y - prediction), weights=w)), 'confusion': confusion_matrix(y, prediction, labels=range(4), sample_weight=w).tolist(), 'underestimate_high_or_critical': int(((y >= 2) & (prediction < 2)).sum()), 'label_status': 'weak_pending', 'scope': 'severity conditional on a true positive risk; not end-to-end seven-class accuracy'}

def severity_weights(rows, mode='root_class'):
    """Audit historical, root-only and repaired weighting rules."""
    if mode not in {'legacy', 'root', 'root_class'}:
        raise ValueError('Unknown severity weighting mode')
    if not rows or any((r.get('targets', {}).get('severity') not in range(4) for r in rows)):
        raise ValueError('Severity weights require observed levels 0..3')
    roots = np.asarray([r['meta']['root_id'] for r in rows])
    y = np.asarray([r['targets']['severity'] for r in rows], dtype=int)
    base = weights(roots)
    raw_counts = np.bincount(y, minlength=4)
    root_mass = np.bincount(y, weights=base, minlength=4)
    counts = raw_counts if mode == 'legacy' else root_mass
    multipliers = np.clip(np.sqrt(counts.sum() / (4 * np.maximum(counts, 1e-12))), 0.6, 2.0)
    if mode == 'root':
        multipliers = np.ones(4)
    before = base * multipliers[y]
    final = before.copy()
    if mode == 'legacy':
        for root in set(roots):
            mask = roots == root
            final[mask] /= final[mask].sum()
    else:
        final *= base.sum() / final.sum()
    audit = {'mode': mode, 'rows': len(rows), 'roots': len(set(roots)), 'root_by_level': {root: np.bincount(y[roots == root], minlength=4).tolist() for root in sorted(set(roots))}, 'raw_class_counts': raw_counts.tolist(), 'root_balanced_class_mass': root_mass.tolist(), 'class_multipliers': multipliers.tolist(), 'class_mass_before_final_normalization': np.bincount(y, weights=before, minlength=4).tolist(), 'class_mass_after_final_normalization': np.bincount(y, weights=final, minlength=4).tolist(), 'final_root_mass': {root: float(final[roots == root].sum()) for root in sorted(set(roots))}}
    return (final, audit)

def train_severity(output, weight_mode='root_class', seeds=(20260920, 20260921, 20260922), epochs=100):
    from .fusion import softmax
    import hashlib
    output = Path(output)
    torch.set_num_threads(4)
    manifest_path = output / 'bundle/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    encoder_path = Path(manifest['encoder_path'])
    if not encoder_path.is_absolute():
        encoder_path = output / 'bundle' / encoder_path
    encoder = FrozenEncoder(encoder_path)
    rows = [r for r in read_records(output / 'dataset.json') if r['targets'].get('severity') is not None]
    audit_splits(rows)
    train = [r for r in rows if r['meta']['split'] == 'train']
    val = [r for r in rows if r['meta']['split'] == 'validation']
    test = [r for r in rows if r['meta']['split'] == 'test']
    fit_groups, selection_groups = ({r['meta']['group_id'] for r in train}, {r['meta']['group_id'] for r in val})
    if fit_groups & selection_groups:
        raise ValueError('Severity fitting/selection overlap')
    train_x, val_x, test_x = (torch.tensor(np.stack([features(r['input'], encoder) for r in part])) for part in (train, val, test))
    y = torch.tensor([r['targets']['severity'] for r in train])
    vy = torch.tensor([r['targets']['severity'] for r in val])
    vw = torch.tensor(weights([r['meta']['root_id'] for r in val]), dtype=torch.float32)
    values_w, weight_audit = severity_weights(train, weight_mode)
    weighted = torch.tensor(values_w, dtype=torch.float32)
    write_json(output / 'severity_weight_audit.json', weight_audit)
    report = {'train_risks': len(train), 'validation_risks': len(val), 'test_risks': len(test), 'label_status': 'weak_pending', 'decision_authority': False, 'seeds': [], 'epochs': epochs, 'weighting': weight_audit, 'method': 'dedicated same-text MLP; CE plus ordinal auxiliary; only observed severity labels normalize loss'}
    values, files = ([], [])
    for seed in seeds:
        torch.manual_seed(seed)
        model = SeverityMLP(train_x.shape[1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.02)
        best, best_score, best_epoch = (None, float('inf'), None)
        for epoch in range(epochs):
            model.train()
            logits, ordinal = model(train_x)
            ce = (F.cross_entropy(logits, y, reduction='none') * weighted).sum() / weighted.sum()
            truth = (y[:, None] > torch.arange(3)[None, :]).float()
            ordinal_loss = (F.binary_cross_entropy_with_logits(ordinal, truth, reduction='none').mean(1) * weighted).sum() / weighted.sum()
            loss = ce + 0.3 * ordinal_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            model.eval()
            with torch.no_grad():
                score = float((F.cross_entropy(model(val_x)[0], vy, reduction='none') * vw).sum() / vw.sum())
            if score < best_score:
                best, best_score, best_epoch = (copy.deepcopy(model.state_dict()), score, epoch + 1)
        model.load_state_dict(best)
        model.eval()
        with torch.no_grad():
            logits = model(test_x)[0].numpy()
        values.append(softmax(logits))
        name = f'severity_{seed}.pt'
        torch.save(model.state_dict(), output / 'bundle' / name)
        files.append(name)
        report['seeds'].append({'seed': seed, 'selected_epoch': best_epoch, 'validation_nll': best_score, 'test': _evaluate(test, values[-1])})
    combined = np.mean(values, axis=0)
    report['ensemble'] = _evaluate(test, combined)
    report['predictions'] = [{'sample_id': r['meta']['sample_id'], 'truth': r['targets']['severity'], 'probabilities': p.tolist()} for r, p in zip(test, combined)]
    write_json(output / 'severity_report.json', report)
    write_json(output / 'bundle/severity.json', {'input_dim': int(train_x.shape[1]), 'files': files, 'fit_groups': sorted(fit_groups), 'selection_groups': sorted(selection_groups), 'label_status': 'weak_pending', 'decision_authority': False, 'version': '2026.09.20-er1', 'weight_mode': weight_mode})
    for name in files + ['severity.json']:
        manifest['files_sha256'][name] = hashlib.sha256((output / 'bundle' / name).read_bytes()).hexdigest()
    manifest['severity_adapter'] = 'severity.json'
    write_json(manifest_path, manifest)
    print(json.dumps(report['ensemble'], ensure_ascii=False), flush=True)
    return report

def evaluate_upgrade(output):
    """Evaluate the new severity head behind predicted, never gold, risk gates."""
    from .data import adapt_legacy
    from .neural import load_model, predict
    from .graphs import graph_record
    from .diagnostics import _seven_predictions, seven_metrics
    output = Path(output)
    torch.set_num_threads(4)
    manifest = json.loads((output / 'bundle/manifest.json').read_text(encoding='utf-8'))
    config = json.loads((output / 'bundle/severity.json').read_text(encoding='utf-8'))
    encoder_path = Path(manifest['encoder_path'])
    if not encoder_path.is_absolute():
        encoder_path = output / 'bundle' / encoder_path
    encoder = FrozenEncoder(encoder_path)
    severity_models = []
    for file in config['files']:
        m = SeverityMLP(config['input_dim'])
        m.load_state_dict(torch.load(output / 'bundle' / file, map_location='cpu', weights_only=True))
        m.eval()
        severity_models.append(m)
    neural = {name: [load_model(output / 'bundle' / file)[0] for file in files] for name, files in manifest['models'].items()}
    scopes = {'legacy_test': [r for r in read_records(output / 'dataset.json') if r['meta']['split'] == 'test' and r['meta']['domain'] == 'legacy_regression']}
    old_boundary = Path('REPRO_ROOT/fresh_boundary_probes.json')
    if old_boundary.exists():
        scopes['old_boundary_regression'] = adapt_legacy(read_records(old_boundary))
    report = {'scope': 'previously examined weak legal references; not independent human gold', 'evaluations': {}}
    for scope, rows in scopes.items():
        x = torch.tensor(np.stack([features(r['input'], encoder) for r in rows]))
        with torch.no_grad():
            p = np.mean([m(x)[0].softmax(-1).numpy() for m in severity_models], axis=0)
        severity = p.argmax(1)
        graphs = [graph_record(r, encoder) for r in rows]
        values = {}
        for name, models in neural.items():
            raw = [predict(m, graphs) for m in models]
            original = _seven_predictions(*(np.mean([v[key] for v in raw], axis=0) for key in ('logits', 'related', 'severity')))
            upgraded = [('risk_low', 'risk_medium', 'risk_high', 'risk_critical')[s] if label.startswith('risk_') else label for label, s in zip(original, severity)]
            values[name] = {'before': seven_metrics(rows, original), 'after': seven_metrics(rows, upgraded), 'high_risk_slice': [{'sample_id': r['meta']['sample_id'], 'before': a, 'after': b} for r, a, b in zip(rows, original, upgraded) if r['meta']['original_outcome'] == 'risk_high']}
        ix = [i for i, r in enumerate(rows) if r['targets'].get('severity') is not None]
        values['conditional_severity_only'] = _evaluate([rows[i] for i in ix], p[ix])
        report['evaluations'][scope] = values
    write_json(output / 'severity_upgrade_evaluation.json', report)
    print(json.dumps({scope: {name: {'before': v['before']['macro_f1'], 'after': v['after']['macro_f1']} if name != 'conditional_severity_only' else v['macro_f1'] for name, v in values.items()} for scope, values in report['evaluations'].items()}, ensure_ascii=False), flush=True)
    return report
