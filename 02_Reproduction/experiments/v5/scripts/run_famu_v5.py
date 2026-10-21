"""Finite frozen-9B mechanism pilot: auditable feature extraction and five-arm training."""
import argparse
import hashlib
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from src.evidence_reasoning.risk_field_v5_learning import (encode_queries, tensors, tables, Specialists,
    balanced_weights, pair_objectives, predictions, metrics)
from src.evidence_reasoning.risk_field_v5 import DECISIONS, STATES

ARMS = ['M0_direct', 'M1_cbm', 'M2_cbm_fact', 'M3_boundary', 'M4_dual']
SEEDS = [17, 29, 43]
UPDATES = 240


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, x):
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x, ensure_ascii=False, indent=2), 'utf-8')
    tmp.replace(p)


def read(p):
    return [json.loads(x) for x in p.read_text('utf-8').splitlines()]


def save_cache(path, ids, features, lengths, lock):
    tmp = path.with_suffix('.tmp')
    torch.save({'ids': ids, 'features': torch.stack(features).half(), 'lengths': lengths, 'lock': lock}, tmp)
    tmp.replace(path)


def extract(rows, out, base, data_hash):
    from src.evidence_reasoning.famu_llm import verify_base, load_tokenizer, load_model
    from src.evidence_reasoning.compact_delta import enable_compact_delta
    from src.evidence_reasoning.chunked_attention import enable_chunked_mlp
    path = out/'features.pt'
    lock = {'examples_sha256': data_hash, 'base_lock_sha256': sha(base/'base_lock.json'),
            'encoder_sha256': sha(ROOT/'src/evidence_reasoning/risk_field_v5_learning.py'),
            'mode': 'frozen_bf16_last_hidden_at_target_free_queries', 'revision': 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'}
    ids, features, lengths = [], [], []
    if path.exists():
        old = torch.load(path, map_location='cpu', weights_only=True)
        assert old['lock'] == lock
        ids, features, lengths = old['ids'], list(old['features']), old['lengths']
        assert ids == [r['id'] for r in rows[:len(ids)]]
        if len(ids) == len(rows):
            return
    write(out/'pipeline_status.json', dict(phase='verifying_base', at=stamp(), done=len(ids), total=len(rows)))
    base_lock = verify_base(base)
    assert base_lock['revision'] == lock['revision']
    tokenizer = load_tokenizer(base)
    model, loading = load_model(base, device='cuda', dtype='bfloat16', attention='sdpa')
    enable_compact_delta(model)
    enable_chunked_mlp(model, 1024)
    model.eval().requires_grad_(False)
    write(out/'model_loading.json', {'lock': lock, 'loading': loading, 'backbone_training': False})
    t0 = time.time()
    for row in rows[len(ids):]:
        token_ids, indices = encode_queries(tokenizer, row)
        token_ids = torch.tensor([token_ids], dtype=torch.long, device='cuda')
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        with torch.no_grad():
            result = model.model(input_ids=token_ids, attention_mask=torch.ones_like(token_ids), use_cache=False, return_dict=True)
            vec = result.last_hidden_state[0, indices].detach().float().cpu()
        if not torch.isfinite(vec).all():
            raise ValueError('Nonfinite backbone features: '+row['id'])
        ids.append(row['id']); features.append(vec); lengths.append(token_ids.shape[1])
        del result, token_ids
        if len(ids) % 16 == 0 or len(ids) == len(rows):
            save_cache(path, ids, features, lengths, lock)
        status = dict(phase='extracting_features', at=stamp(), done=len(ids), total=len(rows),
                      last_id=row['id'], last_tokens=lengths[-1], max_tokens=max(lengths),
                      last_seconds=time.time()-started, elapsed_seconds=time.time()-t0,
                      peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
        write(out/'pipeline_status.json', status)
        if len(ids) % 32 == 0:
            print(json.dumps(status), flush=True)
    del model
    torch.cuda.empty_cache()


def make_pairs(rows, pairs):
    idx = {r['id']: i for i, r in enumerate(rows)}
    result = dict(fact=[], information=[], equivalent=[])
    for p in pairs:
        if p['before'] in idx and p['after'] in idx:
            result[p['kind']].append((idx[p['before']], idx[p['after']]))
    return result


def pair_metrics(rows, pairs, values):
    idx = {r['id']: i for i, r in enumerate(rows)}
    totals = {}
    for kind in ['fact', 'information', 'equivalent']:
        edges = [p for p in pairs if p['kind'] == kind and p['before'] in idx and p['after'] in idx]
        both, violations, eligible = 0, 0, 0
        for p in edges:
            i, j = idx[p['before']], idx[p['after']]
            both += (DECISIONS[values[i]] == rows[i]['target'] and DECISIONS[values[j]] == rows[j]['target'])
            if kind == 'equivalent':
                eligible += 1; violations += values[i] != values[j]
            elif kind == 'information' and values[i] != 3 and values[j] != 3:
                lo = {0: 1, 1: 0, 2: 0}; hi = {0: 1, 1: 0, 2: 1}
                eligible += 1
                violations += lo[values[j]] < lo[values[i]] or hi[values[j]] > hi[values[i]]
        totals[kind] = dict(n=len(edges), both_correct=both, both_correct_rate=both/len(edges) if edges else None,
                           eligible_consistency=eligible, violations=violations,
                           note='conflict predictions excluded from interval comparison, still counted in pair accuracy')
    return totals


def train(rows, pairs, out):
    cached = torch.load(out/'features.pt', map_location='cpu', weights_only=True)
    assert cached['ids'] == [r['id'] for r in rows]
    x = F.layer_norm(cached['features'].float(), (cached['features'].shape[-1],)).cuda()
    train_indices = [i for i, r in enumerate(rows) if r['split'] == 'train']
    training = [rows[i] for i in train_indices]
    train_x = x[train_indices]
    train_meta = tensors(training, 'cuda'); all_meta = tensors(rows, 'cuda')
    train_tables = tables(training, 'cuda'); all_tables = tables(rows, 'cuda')
    weights = balanced_weights(training, 'cuda')
    edges = make_pairs(training, pairs)
    reports = []
    for seed in SEEDS:
        for arm in ARMS:
            folder = out/f'{arm}_s{seed}'
            folder.mkdir(exist_ok=True)
            # Resume only a fully completed head run; interrupted small runs restart at seed.
            if (folder/'report.json').exists():
                reports.append(json.loads((folder/'report.json').read_text('utf-8')))
                continue
            random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            model = Specialists(x.shape[-1]).cuda()
            opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
            log = []
            for step in range(1, UPDATES+1):
                model.train(); opt.zero_grad(set_to_none=True)
                field, decision = model(train_x, train_meta, arm, train_tables)
                field_ce = F.cross_entropy(field.flatten(0, 1), train_meta['states'].flatten(), reduction='none').reshape(-1, 4)
                field_loss = ((field_ce*train_meta['masks']).sum(1)/train_meta['masks'].sum(1)*weights).sum()
                status_loss = (-decision.clamp_min(1e-8).log().gather(1, train_meta['labels'][:, None]).squeeze(1)*weights).sum()
                intervention = pair_objectives(decision, field, edges, train_meta['labels'])
                loss = field_loss+status_loss
                if arm in {'M2_cbm_fact', 'M4_dual'}:
                    loss = loss+.2*intervention['fact']
                if arm == 'M4_dual':
                    loss = loss+.2*intervention['information']+.2*intervention['equivalent']
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite loss')
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(grad):
                    raise ValueError('Nonfinite gradients')
                opt.step()
                if step == 1 or step % 24 == 0:
                    record = dict(step=step, loss=float(loss.detach()), field_loss=float(field_loss.detach()),
                                  status_loss=float(status_loss.detach()), grad_norm=float(grad),
                                  pair_losses={k: float(v.detach()) for k, v in intervention.items()})
                    log.append(record)
                    write(folder/'training_log.json', log)
                    write(out/'pipeline_status.json', dict(phase='head_training', at=stamp(), arm=arm,
                                                          seed=seed, step=step, endpoint=UPDATES, completed_runs=len(reports)))
            model.eval()
            values, fields, soft_values = predictions(rows, model, x, all_meta, arm, all_tables)
            report = {'arm': arm, 'seed': seed, 'updates': UPDATES,
                      'all_training_cases_exposures_per_run': UPDATES,
                      'backbone_training': False, 'generated_text_evaluation': False,
                      'trainable_parameters': sum(p.numel() for p in model.parameters() if p.grad is not None),
                      'groups': {}, 'pair_groups': {}, 'task_groups': {}, 'numerics_finite': True,
                      'label_space_note': 'Algebra maps T/F/U to established/not_established/unknown for whole propositions. Only task=risk_status metrics are legal risk states.'}
            for group in sorted({r['domain']+'/'+r['split'] for r in rows}):
                ii = [i for i, r in enumerate(rows) if r['domain']+'/'+r['split'] == group]
                subset = [rows[i] for i in ii]; vv = [values[i] for i in ii]
                report['groups'][group] = metrics(subset, vv, [fields[i] for i in ii])
                if group.startswith('synthetic'):
                    report['pair_groups'][group] = pair_metrics(subset, pairs, vv)
            for task in ['risk_status', 'condition', 'qa']:
                for split in ['train', 'regression']:
                    ii = [i for i, r in enumerate(rows) if r['domain'] == 'v4_legal' and r['task'] == task and r['split'] == split]
                    if ii:
                        report['task_groups'][task+'/'+split] = metrics([rows[i] for i in ii], [values[i] for i in ii], [fields[i] for i in ii])
            torch.save({'model': model.state_dict(), 'arm': arm, 'seed': seed, 'updates': UPDATES,
                        'features_lock': cached['lock'], 'optimizer': opt.state_dict()}, folder/'specialists.pt')
            predictions_file = folder/'predictions.jsonl'
            predictions_file.write_text(''.join(json.dumps({'id': r['id'], 'domain': r['domain'], 'split': r['split'],
                'task': r['task'], 'original_target': r.get('original_target'),
                'expected': r['target'], 'prediction': DECISIONS[v], 'soft_prediction': DECISIONS[sv],
                'field_expected': r['states'], 'field_prediction': [STATES[s] for s in ff[:len(r['states'])]]},
                ensure_ascii=False)+'\n' for r, v, sv, ff in zip(rows, values, soft_values, fields)), 'utf-8')
            report['sha256'] = {n: sha(folder/n) for n in ['specialists.pt', 'predictions.jsonl', 'training_log.json']}
            write(folder/'report.json', report)
            reports.append(report)
            print(json.dumps({'completed': arm, 'seed': seed, 'synthetic_test': report['groups']['synthetic_mechanism/test']['accuracy'],
                              'lab9': report['groups']['v4_legal/regression']['accuracy']}), flush=True)
            del model, opt
    write(out/'all_reports.json', reports)
    final_files = {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()
                   and p.name not in {'pipeline_status.json', 'result_manifest.json'} and not p.name.endswith('.tmp')}
    write(out/'result_manifest.json', {'files': final_files, 'scope': 'frozen9b_specialist_mechanism_pilot'})
    write(out/'pipeline_status.json', dict(phase='completed', at=stamp(), runs=len(reports),
        head_updates=len(reports)*UPDATES, backbone_updates=0, hash_manifest='result_manifest.json'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--data', type=Path, default=ROOT/'training_data/famu_v5_20261004')
    ap.add_argument('--out', type=Path, default=ROOT/'outputs')
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    rows = read(args.data/'examples.jsonl'); pairs = read(args.data/'pairs.jsonl')
    manifest = json.loads((args.data/'manifest.json').read_text('utf-8'))
    for name, expected in manifest['files'].items():
        assert sha(args.data/name) == expected
    protocol = {'arms': ARMS, 'seeds': SEEDS, 'updates_per_run': UPDATES, 'lr': .001, 'weight_decay': .01,
                'gradient_clip': 1., 'field_loss': 1., 'status_loss': 1., 'each_pair_loss': .2,
                'backbone_updates': 0, 'full_batch': True, 'domain_then_family_balanced': True,
                'no_test_selection_or_early_stopping': True, 'data_manifest_sha256': sha(args.data/'manifest.json'),
                'code_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__),
                    ROOT/'src/evidence_reasoning/risk_field_v5.py', ROOT/'src/evidence_reasoning/risk_field_v5_learning.py']}}
    if (args.out/'protocol.json').exists():
        assert json.loads((args.out/'protocol.json').read_text('utf-8')) == protocol
    else:
        write(args.out/'protocol.json', protocol)
    try:
        extract(rows, args.out, args.base, manifest['files']['examples.jsonl'])
        train(rows, pairs, args.out)
    except Exception:
        error = traceback.format_exc()
        write(args.out/'failure.json', dict(at=stamp(), traceback=error))
        write(args.out/'pipeline_status.json', dict(phase='failed', at=stamp(), error=error))
        raise


if __name__ == '__main__':
    main()
