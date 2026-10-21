# Deidentified copy. String literals containing machine locations were replaced.
"""Failure slices, graph intervention checks and matched-budget sampling study."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix
from .data import read_records, write_json, adapt_legacy
from .graphs import FrozenEncoder, topology, graph_record
from .neural import load_model, predict, collate, fit, save_model
from .fusion import softmax
from .training import labels, evaluate, DEFAULT_OUTPUT
from .metrics import classification, weights, group_bootstrap_difference
from .boundary import select_boundary_groups
from .logic import solve
OUTCOMES = ('unrelated', 'related_safe', 'risk_low', 'risk_medium', 'risk_high', 'risk_critical', 'abstain')

def _seven_predictions(answer_logits, related_logits, severity_logits):
    answer = answer_logits.argmax(1)
    related = related_logits.argmax(1)
    severity = (np.minimum.accumulate(1 / (1 + np.exp(-severity_logits)), axis=1) > 0.5).sum(1)
    return ['unrelated' if rel == 0 else 'abstain' if ans == 2 else 'related_safe' if ans == 1 else OUTCOMES[2 + sev] for ans, rel, sev in zip(answer, related, severity)]

def seven_metrics(rows, predictions):
    truth = [r['meta']['original_outcome'] for r in rows]
    roots = [r['meta']['root_id'] for r in rows]
    w = weights(roots)
    return {'samples': len(rows), 'roots': len(set(roots)), 'macro_f1': float(f1_score(truth, predictions, labels=OUTCOMES, average='macro', sample_weight=w, zero_division=0)), 'confusion': confusion_matrix(truth, predictions, labels=OUTCOMES, sample_weight=w).tolist(), 'classes': list(OUTCOMES), 'accuracy': float(np.average(np.asarray(truth) == predictions, weights=w))}

def make_portable_bundle(output):
    output = Path(output)
    path = output / 'bundle'
    manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    encoder = manifest.get('encoder_path')
    if encoder and Path(encoder).is_absolute():
        source = Path(encoder)
        target = path / 'encoder'
        target.mkdir(exist_ok=True)
        for name, checksum in manifest['encoder_files_sha256'].items():
            if Path(name).name != name:
                raise ValueError('Invalid encoder artifact name')
            if hashlib.sha256((source / name).read_bytes()).hexdigest() != checksum:
                raise ValueError('Encoder changed since training')
            shutil.copy2(source / name, target / name)
        manifest['encoder_path'] = 'encoder'
        manifest['portable'] = True
        write_json(path / 'manifest.json', manifest)
    return manifest

def run(output=DEFAULT_OUTPUT, sampling=True):
    output = Path(output)
    torch.set_num_threads(4)
    manifest = make_portable_bundle(output)
    rows = read_records(output / 'dataset.json')
    test = [r for r in rows if r['meta']['split'] == 'test']
    old_report = json.loads((output / 'report.json').read_text(encoding='utf-8'))
    encoder = FrozenEncoder(output / 'bundle' / manifest['encoder_path'] if manifest['encoder_path'] else None, dimension=manifest['encoder_dimension'])
    graphs = [graph_record(r, encoder) for r in test]
    models = {name: [load_model(output / 'bundle' / p)[0] for p in files] for name, files in manifest['models'].items()}
    previous_diagnostics = json.loads((output / 'diagnostics.json').read_text(encoding='utf-8')) if (output / 'diagnostics.json').exists() else {}
    results = {'seven_class': {}, 'graph_interventions': {}, 'heads': {}, 'sampling': previous_diagnostics.get('sampling', {}), 'formal_oracle': {'rows': 0, 'mismatches': 0}, 'notes': ['All legal labels remain weak_pending; old boundary probes were previously examined.', 'Graph interventions are inference-time sensitivity, not retrained graph ablations.', 'Input rules are supplied; no claim of automatic legal rule discovery.']}
    for row in rows:
        if row['meta']['domain'] == 'formal_control':
            results['formal_oracle']['rows'] += 1
            results['formal_oracle']['mismatches'] += solve(row['input'])['proposition_label'] != row['targets']['label']
    test_predictions = read_records(output / 'predictions.json')
    y = labels(test)
    prediction_map = {r['sample_id']: r for r in test_predictions}
    a, b = (np.asarray([prediction_map[r['meta']['sample_id']]['probabilities'][name] for r in test]).argmax(1) for name in ('rgcn', 'mlp'))
    cluster_ci = group_bootstrap_difference(y, a, b, [r['meta']['root_id'] for r in test], groups=[r['meta']['group_id'] for r in test])
    results['gnn_minus_mlp_cluster_ci'] = cluster_ci
    old_report['gnn_minus_mlp'] = cluster_ci
    write_json(output / 'report.json', old_report)
    for architecture, ensemble in models.items():
        outputs = [predict(m, graphs) for m in ensemble]
        averaged = {key: np.mean([o[key] for o in outputs], axis=0) for key in outputs[0]}
        legacy_indices = [i for i, r in enumerate(test) if r['meta']['domain'] == 'legacy_regression']
        p7 = _seven_predictions(*(averaged[key][legacy_indices] for key in ('logits', 'related', 'severity')))
        results['seven_class'][architecture] = seven_metrics([test[i] for i in legacy_indices], p7)
        targets, predictions, root_ids = ([], [], [])
        for start in range(0, len(graphs), 32):
            batch = collate(graphs[start:start + 32])
            with torch.no_grad():
                condition_logits = torch.stack([m(batch)['conditions'] for m in ensemble]).mean(0).numpy()
            targets.extend(batch['condition_y'].tolist())
            predictions.extend(condition_logits.argmax(1).tolist())
            for g in graphs[start:start + 32]:
                root_ids.extend([g['meta']['root_id']] * len(g['condition_labels']))
        results['heads'][architecture] = {'condition': classification(targets, predictions, root_ids)}
        for intervention in ('self_edges_only', 'shuffle_relation_types'):
            changed = copy.deepcopy(graphs)
            permutation = np.random.default_rng(20260920).permutation(np.arange(1, 11))
            for graph in changed:
                if intervention == 'self_edges_only':
                    graph['edges'] = graph['edges'][graph['edges'][:, 2] == 0]
                else:
                    e = graph['edges'][:, 2]
                    e[e > 0] = permutation[e[e > 0] - 1]
            logits = np.mean([predict(m, changed)['logits'] for m in ensemble], axis=0)
            p = softmax(logits / old_report['calibration']['base'][architecture]['value'])
            results['graph_interventions'][f'{architecture}_{intervention}'] = evaluate(test, p)
    boundary_path = Path('REPRO_ROOT/fresh_boundary_probes.json')
    if boundary_path.exists():
        boundaries = adapt_legacy(read_records(boundary_path))
        bgraphs = [graph_record(r, encoder) for r in boundaries]
        results['old_boundary_regression'] = {}
        for name, ensemble in models.items():
            outputs = [predict(m, bgraphs) for m in ensemble]
            averaged = {key: np.mean([o[key] for o in outputs], axis=0) for key in outputs[0]}
            p = _seven_predictions(*(averaged[key] for key in ('logits', 'related', 'severity')))
            results['old_boundary_regression'][name] = seven_metrics(boundaries, p)
            results['old_boundary_regression'][name]['high_risk_slice'] = [{'sample_id': r['meta']['sample_id'], 'prediction': answer} for r, answer in zip(boundaries, p) if r['meta']['original_outcome'] == 'risk_high']
    errors = []
    for row, p in zip(test, test_predictions):
        if row['targets'].get('label'):
            for name, probabilities in p['probabilities'].items():
                prediction = ('T', 'F', 'U')[int(np.argmax(probabilities))]
                if prediction != row['targets']['label']:
                    errors.append({'sample_id': row['meta']['sample_id'], 'group_id': row['meta']['group_id'], 'model': name, 'domain': row['meta']['domain'], 'operation': row['meta'].get('operation'), 'truth': row['targets']['label'], 'prediction': prediction, 'input': model_safe_input(row['input']), 'label_status': row['meta']['label_status']})
    write_json(output / 'error_cases.json', errors)
    if sampling:
        results['sampling'] = sampling_experiment(rows, encoder, output)
    write_json(output / 'diagnostics.json', results)
    print(json.dumps({'seven_class': results['seven_class'], 'ci': cluster_ci, 'sampling': {k: v['mean_macro_f1'] for k, v in results['sampling'].items()}}, ensure_ascii=False), flush=True)
    return results

def model_safe_input(payload):
    from .schema import model_input
    return model_input(payload)

def sampling_experiment(rows, encoder, output):
    original = [r for r in rows if r['meta']['split'] == 'train' and r['meta'].get('operation') == 'original']
    train = [r for r in rows if r['meta']['split'] == 'train' and r['meta']['domain'] == 'formal_control']
    by_id = {r['meta']['sample_id']: r for r in train}
    oof = json.loads((output / 'oof_predictions.json').read_text(encoding='utf-8'))
    p = softmax(np.asarray(oof['logits']['mlp']))
    predicted = dict(zip(oof['sample_ids'], p))
    pairs = sorted({tuple(pair) for r in train for pair in r['meta'].get('pair_ids', [])})
    pool = []
    for a, b in pairs:
        if a not in by_id or b not in by_id:
            continue
        ya, yb = (('T', 'F', 'U').index(by_id[a]['targets']['label']), ('T', 'F', 'U').index(by_id[b]['targets']['label']))
        pool.append({'split': 'train', 'root_id': by_id[a]['meta']['root_id'], 'pair': [a, b], 'hardness': float(-np.log(max(predicted[a][ya], 1e-08)) - np.log(max(predicted[b][yb], 1e-08))), 'boundary_value': float(1.0 - predicted[a][2] + predicted[b][2])})
    budget = min(40, len({x['root_id'] for x in pool}))
    protocol = {'atomic_pair_budget': budget, 'additional_views': budget * 2, 'original_roots': len(original), 'scoring_source': 'training-only grouped OOF; no test losses', 'epochs': 16, 'seeds': [20260920, 20260921, 20260922], 'margin_loss': False, 'limitation': 'synthetic proxy for annotation cost; not measured lawyer hours'}
    write_json(output / 'sampling_protocol.json', protocol)
    result = {}
    validation = [r for r in rows if r['meta']['split'] == 'validation' and r['meta']['domain'] == 'formal_control']
    test = [r for r in rows if r['meta']['split'] == 'test' and r['meta']['domain'] == 'formal_control']
    vgraphs, tgraphs = ([graph_record(r, encoder) for r in part] for part in (validation, test))
    steps = max(4, (len(original) + 7) // 8)
    for strategy in ('random', 'hardness', 'boundary'):
        selected = select_boundary_groups(pool, budget, strategy, 20260920)
        chosen = original + [by_id[sid] for item in selected for sid in item['pair']]
        if len({r['meta']['sample_id'] for r in chosen}) != len(chosen):
            raise ValueError('Matched sampling budget contains repeated views')
        graphs = [graph_record(r, encoder) for r in chosen]
        scores = []
        for seed in protocol['seeds']:
            model, meta = fit(graphs, vgraphs, 'mlp', seed, protocol['epochs'], auxiliary=True, boundary=False, grouped=True, hidden=48, steps_per_epoch=steps)
            save_model(output / f'sampling_models/{strategy}_{seed}.pt', model, meta)
            scores.append({'seed': seed, 'evaluation': evaluate(test, softmax(predict(model, tgraphs)['logits']))})
        result[strategy] = {'selection': selected, 'training_rows': len(chosen), 'scores': scores, 'mean_macro_f1': float(np.mean([s['evaluation']['classification']['macro_f1'] for s in scores]))}
        print(f'matched-budget sampling {strategy}: complete', flush=True)
    return result
