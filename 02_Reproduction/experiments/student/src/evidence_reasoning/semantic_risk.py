"""Frozen local semantic features and staged risk/ordinal severity learning.

No legal rationales, gold spans, or reviewer identifiers are model inputs.
Unsupported severity classes remain explicit, never supplied with invented data.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.lawyer_review_dataset import file_hash, digest
from .contract_retrieval import validate_request, source_units, normalize
from .question_evidence_models import context_vector, TASK_CLASSES

VERSION = "2026.09.29-semantic-risk-v2"
ENCODER_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt")


class LocalSemanticEncoder:
    name = "bm25_bge_reciprocal_rank_fusion"

    def __init__(self, folder, expected_hashes=None):
        self.folder = Path(folder).resolve()
        self.hashes = {name: file_hash(self.folder/name) for name in ENCODER_FILES}
        if expected_hashes is not None:
            if self.hashes != expected_hashes:
                raise ValueError("Frozen semantic encoder hash mismatch")
        else:
            snapshot = json.loads((self.folder/'snapshot_manifest.json').read_text('utf-8'))
            if any(self.hashes[name] != snapshot['files'][name]['sha256'] for name in ENCODER_FILES):
                raise ValueError("Local encoder differs from its recorded snapshot")
        import torch
        torch.set_num_threads(2)
        from .graphs import FrozenEncoder
        self.encoder = FrozenEncoder(self.folder)

    def encode(self, texts):
        return self.encoder.encode(texts)

    def rank(self, question, units, lexical_scores):
        vectors = self.encode([question] + [u['text'] for u in units])
        similarity = vectors[1:] @ vectors[0]
        lexical = sorted([i for i, u in enumerate(units) if lexical_scores.get(u['unit_id'], 0) > 0],
                         key=lambda i: (-lexical_scores[units[i]['unit_id']], units[i]['source_id'], units[i]['start']))
        semantic = sorted(range(len(units)), key=lambda i: (-float(similarity[i]), units[i]['source_id'], units[i]['start']))
        # Fixed RRF ranks, not tuned on the legal labels. Similarity is not a
        # calibrated relevance probability or a support/refutation relation.
        scores = np.zeros(len(units))
        for order in (lexical, semantic):
            for rank, i in enumerate(order[:20], 1):
                if order is semantic and similarity[i] <= 0:
                    continue
                scores[i] += 1.0/(60+rank)
        return scores.tolist()

    def features(self, payload, retrieval):
        x = validate_request(payload)
        texts = list(dict.fromkeys(e['text'] for e in retrieval['evidence']))
        q = self.encode([x['question']])[0]
        if texts:
            evidence = self.encode(texts)
            # Query-weighted pooling is fixed, not supervised with gold spans.
            similarities = evidence @ q
            attention = np.exp((similarities-similarities.max())/.1)
            attention /= attention.sum()
            e = attention @ evidence
            e /= max(np.linalg.norm(e), 1e-12)
        else:
            e = np.zeros_like(q)
        visible = [s for s in x['sources'] if s.get('available', True) and s['id'] not in x['withdrawn_sources']]
        context = context_vector({**x, 'has_performance_source': any(s['kind']=='performance_record' for s in visible)})
        # Keep question, evidence, agreement and discrepancy separate.
        return np.concatenate([q, e, q*e, np.abs(q-e), np.asarray(context, dtype=float)])


def balanced_weights(labels, base_weights):
    mass = Counter()
    for label, weight in zip(labels, base_weights):
        mass[label] += weight
    return [weight/mass[label]*len(labels)/len(mass) for label, weight in zip(labels, base_weights)]


def fit_linear(matrix, labels, weights):
    classes = sorted(set(labels))
    if len(classes) == 1:
        return {'classes': classes, 'constant': classes[0], 'coefficients': [], 'intercept': []}
    model = LogisticRegression(C=1.0, max_iter=800, solver='lbfgs', random_state=20260929)
    model.fit(matrix, labels, sample_weight=balanced_weights(labels, weights))
    return {'classes': model.classes_.tolist(), 'constant': None,
            'coefficients': model.coef_.tolist(), 'intercept': model.intercept_.tolist()}


def linear_probabilities(head, matrix):
    if head['constant'] is not None:
        return np.ones((len(matrix), 1))
    logits = matrix @ np.asarray(head['coefficients']).T + np.asarray(head['intercept'])
    if len(head['classes']) == 2:
        p = 1/(1+np.exp(-np.clip(logits[:, 0], -700, 700)))
        return np.stack([1-p, p], axis=1)
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p/p.sum(axis=1, keepdims=True)


def event_probability(head, matrix, positive):
    p = linear_probabilities(head, matrix)
    return p[:, head['classes'].index(positive)] if positive in head['classes'] else np.zeros(len(matrix))


def project_nonincreasing(values):
    """Least-squares PAVA for ordered exceedance probabilities."""
    blocks = []
    for value in values:
        blocks.append([float(value), 1])
        while len(blocks) > 1 and blocks[-2][0]/blocks[-2][1] < blocks[-1][0]/blocks[-1][1]:
            right, left = blocks.pop(), blocks.pop()
            blocks.append([left[0]+right[0], left[1]+right[1]])
    return np.array([total/n for total, n in blocks for _ in range(n)])


class SemanticRiskHead:
    def __init__(self, task, architecture='staged', payload=None):
        if task not in {'risk_status', 'severity'} or architecture not in {'flat', 'staged'}:
            raise ValueError('Unsupported semantic risk head')
        self.task, self.architecture, self.payload = task, architecture, payload
        if payload is not None and (payload.get('version') != VERSION or payload['task'] != task or payload['architecture'] != architecture):
            raise ValueError('Incompatible semantic risk payload')

    def fit(self, matrix, labels, base_weights, fit_groups):
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or not len(matrix) or len(labels) != len(matrix) or len(base_weights) != len(labels):
            raise ValueError('Mismatched semantic training data')
        if not np.isfinite(matrix).all() or any(not np.isfinite(w) or w <= 0 for w in base_weights):
            raise ValueError('Nonfinite features or invalid weights')
        if any(label not in TASK_CLASSES[self.task] for label in labels):
            raise ValueError('Unexpected risk target')
        # Fold-only scaling. Variance floor limits noise amplification on 3 contracts.
        mean = matrix.mean(axis=0)
        scale = np.maximum(matrix.std(axis=0), 0.05)
        x = (matrix-mean)/scale/np.sqrt(matrix.shape[1])
        heads = {}
        if self.architecture == 'flat':
            heads['flat'] = fit_linear(x, labels, base_weights)
        elif self.task == 'risk_status':
            heads['unknown'] = fit_linear(x, ['unknown' if label == 'unknown' else 'assessable' for label in labels], base_weights)
            known = [i for i, label in enumerate(labels) if label != 'unknown']
            heads['conditional_risk'] = fit_linear(x[known], [labels[i] for i in known], [base_weights[i] for i in known]) if known else None
        else:
            order = TASK_CLASSES['severity']
            for boundary in range(1, len(order)):
                heads[order[boundary]] = fit_linear(x, ['at_or_above' if order.index(label) >= boundary else 'below' for label in labels], base_weights)
        self.payload = {'version': VERSION, 'task': self.task, 'architecture': self.architecture,
            'classes': sorted(set(labels)), 'mean': mean.tolist(), 'scale': scale.tolist(),
            'feature_dimension': matrix.shape[1], 'heads': heads, 'training_rows': len(labels),
            'fit_groups': sorted(set(fit_groups)), 'probabilities_calibrated': False,
            'class_counts': dict(Counter(labels)), 'standardization_fit_on_training_only': True}
        return self

    def predict(self, matrix):
        if self.payload is None:
            raise ValueError('Unfitted semantic head')
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != self.payload['feature_dimension'] or not np.isfinite(matrix).all():
            raise ValueError('Invalid semantic feature matrix')
        if not len(matrix):
            return []
        x = (matrix-np.asarray(self.payload['mean']))/np.asarray(self.payload['scale'])/np.sqrt(matrix.shape[1])
        heads = self.payload['heads']
        classes = list(TASK_CLASSES[self.task])
        stages = [{} for _ in x]
        if self.architecture == 'flat':
            partial = linear_probabilities(heads['flat'], x)
            probs = np.zeros((len(x), len(classes)))
            for i, label in enumerate(heads['flat']['classes']):
                probs[:, classes.index(label)] = partial[:, i]
        elif self.task == 'risk_status':
            u = event_probability(heads['unknown'], x, 'unknown')
            risk = event_probability(heads['conditional_risk'], x, 'established') if heads['conditional_risk'] else np.zeros(len(x))
            probs = np.stack([(1-u)*risk, (1-u)*(1-risk), u], axis=1)
            stages = [{'unknown_score': float(a), 'conditional_established_score': float(b)} for a, b in zip(u, risk)]
        else:
            raw = np.stack([event_probability(heads[label], x, 'at_or_above') for label in classes[1:]], axis=1)
            ordered = np.stack([project_nonincreasing(r) for r in raw])
            probs = np.column_stack([1-ordered[:, 0], ordered[:, 0]-ordered[:, 1], ordered[:, 1]-ordered[:, 2], ordered[:, 2]])
            stages = [{'raw_exceedance': r.tolist(), 'ordered_exceedance': o.tolist()} for r, o in zip(raw, ordered)]
        supported = set(self.payload['classes'])
        for i, label in enumerate(classes):
            if label not in supported:
                probs[:, i] = 0  # No invented support for absent classes.
        totals = probs.sum(axis=1, keepdims=True)
        if np.any(totals <= 0):
            raise ValueError('No supported semantic output class')
        probs /= totals
        return [{'label': classes[int(p.argmax())], 'score': float(p.max()),
                 'probabilities': {label: float(p[i]) for i, label in enumerate(classes) if label in supported},
                 'untrained_classes': sorted(set(classes)-supported), 'probabilities_calibrated': False,
                 'stage_scores': stage} for p, stage in zip(probs, stages)]
