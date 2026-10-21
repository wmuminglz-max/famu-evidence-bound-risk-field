"""Experimental condition attention adapter with an evidence-backed exact gate.

No proposition logits authorize an observation. Conditions and visible support
must agree; unavailable sources and incomplete scopes cannot be selected.
"""
import numpy as np
import torch
from .schema import model_input
from .graphs import graph_record
from .neural import collate
from .provenance import visible_evidence


class AttentionConditionExtractor:
    def __init__(self, model, encoder):
        if model.config['architecture'] != 'rule_tree':
            raise ValueError('Condition adapter requires a rule tree model')
        self.model, self.encoder = model, encoder

    def extract(self, payload, threshold=.70):
        x = model_input(payload)
        row = {'input': x, 'targets': {}, 'meta': {'sample_id': 'runtime', 'root_id': 'runtime',
               'group_id': 'runtime', 'split': 'external', 'label_status': 'unlabeled'}}
        g = graph_record(row, self.encoder, preserve_scope=True)
        candidates = []
        for cid, cindex in g['condition_lookup'].items():
            for path in g['candidate_paths'][cid]:
                if path['usable'] and len(path['indices']) == 1:
                    candidates.append((cid, cindex, path['indices'][0]))
        # Pair enumeration uses observable topology only, including unlabeled pairs.
        g['evidence_pairs'] = [(cindex, eid, -1) for _, cindex, eid in candidates]
        self.model.eval()
        with torch.no_grad():
            out = self.model(collate([g]))
            conditions = out['conditions'].softmax(-1).numpy()
            relations = out['evidence'].softmax(-1).numpy()
        cp = dict(zip(g['condition_ids'], conditions))
        claims, details, uncertain = [], [], set()
        evidence = {e['id']: e for e in visible_evidence(x)}
        candidate_counts = {cid: 0 for cid in cp}
        for (cid, _, i), p in zip(candidates, relations):
            n = g['nodes'][i]
            relation = int(p.argmax()); state = int(cp[cid].argmax())
            accepted = (relation in (1, 2) and state == relation-1 and p[relation] >= threshold and cp[cid][state] >= threshold)
            ambiguous = float(p.max()) < threshold or (relation in (1,2) and not accepted)
            if ambiguous: uncertain.add(cid)
            candidate_counts[cid] += 1
            details.append({'condition': cid, 'evidence_id': n['key'], 'start': n.get('start', 0),
                'end': n.get('end', len(n['text'])), 'text': n['text'], 'accepted': bool(accepted),
                'ambiguous_extraction': ambiguous, 'probabilities': dict(zip(('irrelevant','support','refute'), p.tolist())),
                'source_supports': evidence[n['key']]['live_supports']})
            if accepted:
                claims.append({'condition': cid, 'value': relation == 1, 'evidence_groups': [[n['key']]],
                               'origin': 'attention_condition_and_relation_agreement'})
        for cid, paths in g['candidate_paths'].items():
            if any(not p['usable'] for p in paths) or (candidate_counts[cid] and cp[cid].max() < threshold):
                uncertain.add(cid)
        x['observations'] = claims
        return {'input': x, 'evidence_predictions': details, 'uncertain_conditions': sorted(uncertain),
                'condition_features': {cid: p.tolist() for cid, p in cp.items()},
                'scope': 'experimental; final answer requires execute_extracted',
                'joint_support_scope': 'explicit joint natural-language proofs require their own relation supervision; refine atomic conditions when unavailable'}
