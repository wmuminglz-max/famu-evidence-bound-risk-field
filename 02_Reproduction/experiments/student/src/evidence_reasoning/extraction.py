"""Trainable condition-to-evidence classifier with offsets and explicit conflicts."""
from __future__ import annotations

import copy
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from .provenance import visible_evidence, evidence_signature
from .schema import model_input
from .text import windows
from .spans import annotated_pairs, scoped_spans

PAIR_LABELS = ("irrelevant", "support", "refute")


class ConditionExtractor:
    def __init__(self, payload=None):
        self.dim = 2048
        self.vectorizer = HashingVectorizer(n_features=self.dim, analyzer="char", ngram_range=(1, 3),
                                           alternate_sign=False, norm="l2")
        self.payload = payload

    def features(self, pairs):
        texts = ["条件：" + q + "；原文：" + text for q, text in pairs]
        numeric = []
        for q, text in pairs:
            chars = set(q)
            overlap = len(chars & set(text)) / max(1, len(chars))
            numeric.append([float(q in text), overlap, float("未" in text),
                            float("否认" in text), float("已经" in text or "确认已" in text),
                            float(q in text and ("未" in text or "否认" in text))])
        return hstack([self.vectorizer.transform(texts), csr_matrix(numeric)], format="csr")

    def fit(self, records):
        if any(r["meta"]["split"] != "train" for r in records):
            raise ValueError("Extractor fitting may only use training groups")
        samples, audit = annotated_pairs(records, return_audit=True)
        pairs = [(s['query'], s['text']) for s in samples]
        labels = [s['label'] for s in samples]
        weights = [s['weight'] for s in samples]
        if set(labels) != set(range(3)):
            raise ValueError("Extractor training requires all three independently annotated pair states")
        model = LogisticRegression(C=3., max_iter=300, class_weight="balanced", random_state=20260920)
        model.fit(self.features(pairs), labels, sample_weight=weights)
        self.payload = {"coefficients": model.coef_.tolist(), "intercept": model.intercept_.tolist(),
                        "classes": model.classes_.tolist(), "pair_count": len(pairs),
                        "pair_filter_audit": audit, "pair_weighting": "root_normalized",
                        "fit_groups": sorted({r["meta"]["group_id"] for r in records}),
                        "label_statuses": sorted({r["meta"]["label_status"] for r in records})}
        return self

    def predict_pairs(self, pairs):
        if not self.payload:
            raise ValueError("Extractor has not been fitted")
        if not pairs:
            return np.empty((0, 3))
        logits = np.asarray(self.features(pairs) @ np.asarray(self.payload["coefficients"]).T)
        logits += np.asarray(self.payload["intercept"])
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        output = np.zeros((len(pairs), 3))
        output[:, self.payload["classes"]] = p
        return output

    def extract(self, payload, threshold=.70):
        x = model_input(payload)
        candidates, pairs = [], []
        visible = {e['id']: e for e in visible_evidence(x)}
        seen = set()
        for condition in x["conditions"]:
            for evidence in visible.values():
                signature = (condition['id'], evidence_signature(evidence))
                if signature in seen:
                    continue
                seen.add(signature)
                for span in scoped_spans(evidence["text"]):
                    candidates.append((condition["id"], [evidence["id"]], span, evidence['live_supports']))
                    full_assertion = span['start'] == 0 and span['end'] == len(evidence['text'])
                    pair_text = evidence.get('equivalence', {}).get('canonical_text', span['text']) if full_assertion else span['text']
                    pairs.append((condition["text"], pair_text))
        for group in x.get('candidate_groups', []):
            ids = group['evidence_ids']
            if not set(ids) <= set(visible):
                continue  # hard mask every joint path if any member is unavailable
            text = '\n'.join(visible[eid]['text'] for eid in ids)
            condition = next(c for c in x['conditions'] if c['id'] == group['condition'])
            candidates.append((condition['id'], ids, {'text': text, 'start': None, 'end': None,
                'scope_complete': len(text) <= 180, 'candidate_group': group['id']},
                {eid: visible[eid]['live_supports'] for eid in ids}))
            pairs.append((condition['text'], text))
        predictions = np.full((len(pairs), 3), 1./3.)
        complete = [i for i, candidate in enumerate(candidates) if candidate[2]['scope_complete'] and len(pairs[i][1]) <= 180]
        if complete:
            predictions[complete] = self.predict_pairs([pairs[i] for i in complete])
        claims, details, uncertain = [], [], set()
        states = {c["id"]: [0., 0., 1.] for c in x["conditions"]}
        relevant_candidates = {c['id']: 0 for c in x['conditions']}
        for (cid, eids, span, sources), probabilities in zip(candidates, predictions):
            label = int(probabilities.argmax())
            accepted = label in (1, 2) and probabilities[label] >= threshold and span['scope_complete']
            ambiguous = float(probabilities.max()) < threshold or not span['scope_complete']
            if ambiguous:
                uncertain.add(cid)
            if accepted:
                relevant_candidates[cid] += 1
            details.append({"condition": cid, "evidence_id": eids[0] if len(eids) == 1 else None,
                            'evidence_ids': list(eids), 'source_supports': sources, **span,
                            "probabilities": dict(zip(PAIR_LABELS, probabilities.tolist())), "accepted": bool(accepted),
                            "ambiguous_extraction": ambiguous})
            if accepted:
                claims.append({"condition": cid, "value": label == 1, "evidence_groups": [list(eids)],
                               "origin": "learned_extractor"})
            states[cid][0] = max(states[cid][0], float(probabilities[1]))
            states[cid][1] = max(states[cid][1], float(probabilities[2]))
        for values in states.values():
            values[2] = max(0., 1. - max(values[:2]))
            total = sum(values)
            values[:] = [v / total for v in values]
        x["observations"] = claims
        return {"input": x, "evidence_predictions": details, "condition_features": states,
                "uncertain_conditions": sorted(uncertain),
                "condition_evidence_status": {cid: 'extraction_ambiguous' if cid in uncertain else
                    'relevant_evidence' if relevant_candidates[cid] else 'no_relevant_evidence' for cid in states},
                "joint_support_scope": "only explicitly supplied/predicted candidates; not automatic proof discovery",
                "uncertainty_scope": "model extraction uncertainty; not a legal U target",
                "threshold": threshold, "scope": "learned_facts_not_independent_legal_gold"}
