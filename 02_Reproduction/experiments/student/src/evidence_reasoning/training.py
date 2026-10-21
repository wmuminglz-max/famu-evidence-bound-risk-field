# Deidentified copy. String literals containing machine locations were replaced.
"""Reproducible offline research pipeline. Test labels never select parameters."""
from __future__ import annotations
import copy
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from . import VERSION
from .schema import digest, decision_route
from .decision import select_decision
from .data import synthetic_records, adapt_legacy, read_records, write_json, audit_splits, group_folds, annotation_queue
from .graphs import FrozenEncoder, graph_record, topology
from .extraction import ConditionExtractor
from .logic import solve, verify_certificate
from .neural import fit, predict, save_model
from .fusion import softmax, temperature, Stacker, fit_convex_weights
from .metrics import classification, pair_metrics, complementary, group_bootstrap_difference, weights
from .consensus import SequentialConsensus, evaluate_policy
DEFAULT_ENCODER = Path('REPRO_ROOT/pretrained_bge_small_zh')
DEFAULT_LEGACY = Path('REPRO_ROOT/risk_graph_canonical.jsonl')
DEFAULT_LIVE = Path('REPRO_ROOT/consensus_live.json')
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / 'outputs/algorithm_strengthening_20260920'

def labels(rows):
    return np.array([('T', 'F', 'U').index(r['targets']['label']) if r['targets'].get('label') in ('T', 'F', 'U') else -1 for r in rows])

def evaluate(rows, probabilities):
    return {'classification': classification(labels(rows), probabilities.argmax(1), [r['meta']['root_id'] for r in rows], probabilities), 'boundary': pair_metrics(rows, probabilities), 'label_statuses': dict(Counter((r['meta']['label_status'] for r in rows)))}

def prepare_data(output, roots=120, seed=20260920, legacy_path=DEFAULT_LEGACY):
    output = Path(output)
    synthetic = synthetic_records(roots, seed)
    legacy = adapt_legacy(read_records(legacy_path)) if legacy_path else []
    groups = sorted({r['meta']['group_id'] for r in legacy if r['meta']['split'] == 'validation'})
    random.Random(seed).shuffle(groups)
    calibration = set(groups[::2])
    for row in legacy:
        row['meta']['original_split'] = row['meta']['split']
        if row['meta']['split'] == 'validation' and row['meta']['group_id'] in calibration:
            row['meta']['split'] = 'calibration'
    rows = synthetic + legacy
    audit = audit_splits(rows)
    write_json(output / 'dataset.json', rows)
    write_json(output / 'split_audit.json', audit)
    write_json(output / 'independent_annotation_queue.json', annotation_queue(legacy))
    write_json(output / 'examples/structured_case.json', next((r['input'] for r in synthetic if r['meta']['operation'] == 'original')))
    protocol = {'version': VERSION, 'seed': seed, 'dataset_sha256': digest(rows), 'label_order': ['T', 'F', 'U'], 'synthetic_roots': roots, 'independent_human_gold_roots': 0, 'legacy_path': str(legacy_path) if legacy_path else None, 'legacy_sha256': hashlib.sha256(Path(legacy_path).read_bytes()).hexdigest() if legacy_path else None, 'test_role': 'formal_controls_and_previously_seen_weak_regression; not independent legal blind test', 'input_whitelist': 'schema.INPUT_FIELDS; observations removed from all learned inputs', 'neural_extractor_features': 'disabled: both neural models use identical raw frozen embeddings and observable topology', 'split_unit': 'group/root/family', 'selection_partition': 'validation', 'calibration_partition': 'calibration', 'deployment_gate': {'independent_human_gold_required': True, 'net_consensus_utility_positive': True, 'shadow_only_until_independent_validation': True}}
    write_json(output / 'protocol.json', protocol)
    return (rows, protocol)

def _extractor_evaluation(extractor, rows):
    predicted, truth, roots, condition_y, condition_p, condition_roots, certificates = ([], [], [], [], [], [], [])
    workflow_predictions = []
    failures = Counter()
    for row in rows:
        if row['meta']['domain'] != 'formal_control':
            continue
        extracted = extractor.extract(row['input'])
        answer = solve(extracted['input'])
        decision = select_decision(row['input'], answer, uncertain_conditions=extracted.get('uncertain_conditions', []))
        workflow_predictions.append(('T', 'F', 'U').index(decision['proposition_label']) if decision['proposition_label'] else -1)
        predicted.append(('T', 'F', 'U').index(answer['proposition_label']) if answer['proposition_label'] else -1)
        truth.append(('T', 'F', 'U').index(row['targets']['label']))
        roots.append(row['meta']['root_id'])
        failures[answer['input_status']] += 1
        if answer['proposition_label'] == 'U':
            certificates.append(verify_certificate(extracted['input'], answer['certificate']))
        for cid, gold in row['targets']['conditions'].items():
            state = answer['condition_states'].get(cid)
            condition_y.append(('T', 'F', 'U').index(gold))
            condition_p.append(('T', 'F', 'U').index(state) if state else -1)
            condition_roots.append(row['meta']['root_id'])
    return {'end_to_end': classification(truth, predicted, roots), 'end_to_end_scope': 'conditional executor given learned observations', 'workflow_classification': classification(truth, workflow_predictions, roots), 'conditions': classification(condition_y, condition_p, condition_roots), 'input_status': dict(failures), 'unknown_certificates': len(certificates), 'valid_certificates': sum(certificates), 'certificate_scope': 'conditional_on_learned_observations; not validation of extractor correctness'}

def run(output=DEFAULT_OUTPUT, roots=120, seeds=(20260920, 20260921, 20260922), epochs=16, encoder_path=DEFAULT_ENCODER, legacy_path=DEFAULT_LEGACY, live_path=DEFAULT_LIVE, ablations=True, progress=print):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows, protocol = prepare_data(output, roots, seeds[0], legacy_path)
    protocol.update(seeds=list(seeds), epochs=epochs, hidden=48, layers=2, oof_folds=3, ablations=ablations, software={'numpy': np.__version__})
    import torch
    torch.set_num_threads(4)
    protocol['software'].update(torch=torch.__version__)
    write_json(output / 'protocol.json', protocol)
    progress(f"data: {len(rows)} rows; {len(set((r['meta']['root_id'] for r in rows)))} roots", flush=True)
    by_split = {s: [r for r in rows if r['meta']['split'] == s] for s in ('train', 'validation', 'calibration', 'test')}
    train, validation, calibration, test = (by_split[s] for s in ('train', 'validation', 'calibration', 'test'))
    extractor = ConditionExtractor().fit(train)
    write_json(output / 'bundle/extractor.json', extractor.payload)
    encoder = FrozenEncoder(encoder_path)
    encoder.prepare((n['text'] for row in rows for n in topology(row['input'])[0]))
    write_json(output / 'tokenization_audit.json', encoder.audit)
    graphs = [graph_record(r, encoder) for r in rows]
    by_graph = {s: [g for g in graphs if g['meta']['split'] == s] for s in by_split}
    progress(f'encoded: {len(encoder.cache)} texts, {len(encoder.audit)} windows', flush=True)
    report = {'version': VERSION, 'dataset_audit': audit_splits(rows), 'protocol_sha256': digest(protocol), 'human_gold_roots': 0, 'extractor': _extractor_evaluation(extractor, test), 'models': {}, 'ablations': {}, 'selection': {}, 'deployment': {'mode': 'shadow', 'decision_authority': False, 'passed': False, 'reason': 'No independent adjudicated legal test set'}}
    raw, paths = ({}, {})
    steps = max(4, (len({r['meta']['root_id'] for r in train}) + 7) // 8)
    for architecture in ('mlp', 'rgcn'):
        logits = defaultdict(list)
        seed_reports = []
        paths[architecture] = []
        for seed in seeds:
            begin = time.perf_counter()
            model, meta = fit(by_graph['train'], by_graph['validation'], architecture, seed, epochs, hidden=48, steps_per_epoch=steps)
            path = output / f'bundle/{architecture}_{seed}.pt'
            save_model(path, model, meta)
            paths[architecture].append(path.name)
            for split in by_split:
                logits[split].append(predict(model, by_graph[split])['logits'])
            seed_reports.append({'seed': seed, 'parameters': meta['parameter_count'], 'seconds': time.perf_counter() - begin, 'test': evaluate(test, softmax(logits['test'][-1]))})
            progress(f'trained {architecture} seed={seed}, {time.perf_counter() - begin:.1f}s', flush=True)
        raw[architecture] = {s: np.mean(values, axis=0) for s, values in logits.items()}
        report['models'][architecture] = {'seeds': seed_reports}
    folds = group_folds(train, 3, seeds[0])
    oof, certificates = ({}, {})
    for architecture in ('mlp', 'rgcn'):
        oof[architecture] = np.zeros((len(train), 3))
        certificates[architecture] = [None] * len(train)
        for fold in range(3):
            train_graphs = [g for i, g in enumerate(by_graph['train']) if folds[i] != fold]
            held = [i for i, f in enumerate(folds) if f == fold]
            held_graphs = [by_graph['train'][i] for i in held]
            values = []
            for seed in seeds:
                model, meta = fit(train_graphs, by_graph['validation'], architecture, seed, epochs, hidden=48, steps_per_epoch=steps)
                values.append(predict(model, held_graphs)['logits'])
                save_model(output / f'oof_models/{architecture}_fold{fold}_{seed}.pt', model, meta)
            oof[architecture][held] = np.mean(values, axis=0)
            for i in held:
                certificates[architecture][i] = {'sample_id': train[i]['meta']['sample_id'], 'fold': fold, 'fit_groups': meta['fit_groups'], 'selection_groups': meta['selection_groups']}
            progress(f'OOF {architecture} fold={fold + 1}/3 complete', flush=True)
    write_json(output / 'oof_predictions.json', {'sample_ids': [r['meta']['sample_id'] for r in train], 'logits': {k: v.tolist() for k, v in oof.items()}, 'certificates': certificates})
    stacker = Stacker().fit([softmax(oof[a]) for a in ('mlp', 'rgcn')], labels(train), train, [certificates[a] for a in ('mlp', 'rgcn')])
    calibration_roots = [r['meta']['root_id'] for r in calibration]
    calibration_splits = ['calibration'] * len(calibration)
    temperatures = {a: temperature(raw[a]['calibration'], labels(calibration), calibration_roots, calibration_splits) for a in raw}
    calibrated = {a: {s: softmax(z / temperatures[a]['value']) for s, z in split.items()} for a, split in raw.items()}
    convex = fit_convex_weights([calibrated[a]['calibration'] for a in ('mlp', 'rgcn')], labels(calibration), calibration_roots, calibration_splits)
    stack_logits = {s: stacker.logits([softmax(raw[a][s]) for a in ('mlp', 'rgcn')]) for s in by_split}
    stack_temperature = temperature(stack_logits['calibration'], labels(calibration), calibration_roots, calibration_splits)
    candidates = {a: calibrated[a] for a in ('mlp', 'rgcn')}
    candidates['mean'] = {s: (calibrated['mlp'][s] + calibrated['rgcn'][s]) / 2 for s in by_split}
    candidates['convex'] = {s: sum((convex['weights'][i] * calibrated[a][s] for i, a in enumerate(('mlp', 'rgcn')))) for s in by_split}
    candidates['stacking'] = {s: softmax(v / stack_temperature['value']) for s, v in stack_logits.items()}
    scores = {k: evaluate(validation, v['validation'])['classification']['nll'] for k, v in candidates.items()}
    selected = min(scores, key=scores.get)
    report['selection'] = {'criterion': 'validation root-weighted NLL', 'scores': scores, 'selected': selected}
    report['calibration'] = {'base': temperatures, 'stacking': stack_temperature, 'convex': convex}
    report['evaluation'] = {}
    for name, p in candidates.items():
        subsets = {'all': list(range(len(test))), 'formal_controls': [i for i, r in enumerate(test) if r['meta']['domain'] == 'formal_control'], 'legacy_weak': [i for i, r in enumerate(test) if r['meta']['domain'] == 'legacy_regression']}
        report['evaluation'][name] = {scope: evaluate([test[i] for i in ix], p['test'][ix]) for scope, ix in subsets.items() if ix}
    report['complementarity'] = complementary(labels(test), [candidates[a]['test'].argmax(1) for a in ('mlp', 'rgcn')], [r['meta']['root_id'] for r in test])
    for metric, key in (('accuracy', 'gnn_minus_mlp'), ('macro_f1', 'gnn_minus_mlp_macro_f1')):
        report[key] = group_bootstrap_difference(labels(test), candidates['rgcn']['test'].argmax(1), candidates['mlp']['test'].argmax(1), [r['meta']['root_id'] for r in test], groups=[r['meta']['group_id'] for r in test], metric=metric)
    hybrid_predictions = candidates[selected]['test'].argmax(1).copy()
    logic_routes = 0
    route_indices = defaultdict(list)
    for i, row in enumerate(test):
        route = decision_route(row['input'])
        route_indices[route['result_origin']].append(i)
        if route['executor_eligible']:
            extracted = extractor.extract(row['input'])
            answer = solve(extracted['input'])
            label = select_decision(row['input'], answer, uncertain_conditions=extracted.get('uncertain_conditions', []))['proposition_label']
            hybrid_predictions[i] = ('T', 'F', 'U').index(label) if label else -1
            logic_routes += 1
    report['structured_hybrid'] = {'classification': classification(labels(test), hybrid_predictions, [r['meta']['root_id'] for r in test]), 'condition_executor_rows': logic_routes, 'route': 'semantic-route-v1: validated structured task -> executor; explicitly declared text_only task -> neural candidate', 'per_route': {name: {'rows': len(ix), 'coverage': len(ix) / len(test), 'classification': classification(labels(test)[ix], hybrid_predictions[ix], [test[i]['meta']['root_id'] for i in ix])} for name, ix in route_indices.items()}, 'scope': 'conditional on supplied rule and learned observations; shadow only; no calibrated proposition confidence'}
    write_json(output / 'predictions.json', [{'sample_id': r['meta']['sample_id'], 'root_id': r['meta']['root_id'], 'domain': r['meta']['domain'], 'label_status': r['meta']['label_status'], 'truth': r['targets'].get('label'), 'probabilities': {name: p['test'][i].tolist() for name, p in candidates.items()}} for i, r in enumerate(test)])
    write_json(output / 'bundle/fusion.json', {'temperatures': temperatures, 'stacker': stacker.payload, 'stack_temperature': stack_temperature, 'convex': convex, 'selected': selected})
    if ablations:
        _ablations(by_split, by_graph, output, report, seeds, epochs, progress)
    if live_path:
        live = read_records(live_path)
        membership = defaultdict(set)
        for row in live:
            for key in ('root_id', 'group_id'):
                membership[key, row[key]].add(row['split'])
        if any((len(s) > 1 for s in membership.values())):
            raise ValueError('Consensus real-log group leakage')
        consensus = SequentialConsensus().fit([r for r in live if r['split'] == 'train'])
        consensus.select_thresholds([r for r in live if r['split'] == 'validation'])
        report['consensus'] = {split: {policy: evaluate_policy(consensus, [r for r in live if r['split'] == split], policy) for policy in ('primary', 'always_secondary', 'agreement_guard', 'learned')} for split in ('validation', 'test', 'external_diagnostic')}
        write_json(output / 'bundle/consensus.json', consensus.payload)
    report['wall_seconds'] = time.perf_counter() - started
    report['frozen_encoder'] = {'path': str(encoder_path) if encoder_path else None, 'dimension': encoder.dimension, 'windows': len(encoder.audit), 'truncated_windows': sum((any((f['truncated'] for f in a['fields'])) for a in encoder.audit))}
    report['oracle_control'] = {'samples': sum((r['meta']['domain'] == 'formal_control' for r in test)), 'description': 'Exact reference with provided observations is the generator oracle, not independently measured legal accuracy'}
    write_json(output / 'report.json', report)
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (output / 'bundle').iterdir() if p.is_file() and p.name != 'manifest.json'}
    encoder_files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(encoder_path).iterdir() if p.is_file()} if encoder_path else {}
    write_json(output / 'bundle/manifest.json', {'version': VERSION, 'mode': 'shadow', 'decision_authority': False, 'models': paths, 'encoder_path': str(encoder_path) if encoder_path else None, 'encoder_dimension': encoder.dimension, 'encoder_files_sha256': encoder_files, 'files_sha256': files, 'protocol_sha256': digest(protocol), 'report': '../report.json'})
    progress(f"complete: {report['wall_seconds']:.1f}s; selected={selected}; mode=shadow", flush=True)
    return report

def _ablations(by_split, by_graph, output, report, seeds, epochs, progress):
    train = [g for g in by_graph['train'] if g['meta']['domain'] == 'formal_control']
    validation = [g for g in by_graph['validation'] if g['meta']['domain'] == 'formal_control']
    test = [g for g in by_graph['test'] if g['meta']['domain'] == 'formal_control']
    test_rows = [r for r in by_split['test'] if r['meta']['domain'] == 'formal_control']
    steps = max(4, (len({g['meta']['root_id'] for g in train}) + 7) // 8)
    for arm, auxiliary, grouped, boundary in (('A', False, False, False), ('B', False, False, False), ('C', True, False, False), ('D', True, True, False), ('E', True, True, True)):
        fitting = [g for g in train if g['meta']['operation'] == 'original'] if arm == 'A' else train
        results = []
        for seed in seeds:
            model, meta = fit(fitting, validation, 'mlp', seed, epochs, auxiliary, boundary, grouped, steps_per_epoch=steps, hidden=48)
            p = softmax(predict(model, test)['logits'])
            results.append({'seed': seed, 'fit_samples': len(fitting), 'optimizer_updates': epochs * steps, 'evaluation': evaluate(test_rows, p)})
            save_model(output / f'ablation_models/{arm}_{seed}.pt', model, meta)
        report['ablations'][arm] = results
        write_json(output / 'ablation_report.partial.json', report['ablations'])
        progress(f'ablation {arm}, {len(seeds)} seeds complete', flush=True)
