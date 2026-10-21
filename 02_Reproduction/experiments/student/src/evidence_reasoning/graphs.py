"""Variable typed graphs from observable input; no target-derived edges."""
from __future__ import annotations

import re
import numpy as np
from .schema import model_input, digest
from .provenance import visible_evidence, evidence_signature
from .spans import scoped_spans
from .text import numeric_facts, adaptive_tokenize, windows

NODE_TYPES = ("query", "condition", "evidence", "source", "all", "any", "not", "constant")
RELATIONS = ("self", "candidate", "candidate_reverse", "origin", "origin_reverse",
             "requires", "alternative", "negates", "rule_query", "reverse_rule", "reference")


class FrozenEncoder:
    def __init__(self, model_path=None, dimension=128):
        self.path = str(model_path) if model_path else None
        self.cache = {}
        self.audit = []
        if model_path:
            import torch
            from transformers import AutoModel, AutoTokenizer
            self.torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
            self.model = AutoModel.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
            self.model.eval()
            self.dimension = self.model.config.hidden_size
        else:
            from sklearn.feature_extraction.text import HashingVectorizer
            self.vectorizer = HashingVectorizer(n_features=dimension, analyzer="char", ngram_range=(1, 3),
                                               alternate_sign=False, norm="l2")
            self.dimension = dimension

    def prepare(self, texts):
        unseen = sorted(set(texts) - set(self.cache))
        if self.path:
            # Every character contributes to a window; long clauses are not cut
            # at a fixed per-field budget. Pool frozen window vectors per node.
            pieces = [(text, span) for text in unseen for span in (windows(text, width=360, overlap=48) or [{"text": "", "start": 0, "end": 0}])]
            accumulated = {text: [] for text in unseen}
            for start in range(0, len(pieces), 32):
                batch = pieces[start:start + 32]
                encoded = []
                for text, span in batch:
                    tokens, audit = adaptive_tokenize([("内容", span["text"])], self.tokenizer, max_length=512)
                    if any(f["truncated"] for f in audit["fields"]):
                        raise ValueError("Window exceeded encoder budget; lower window width")
                    encoded.append(tokens)
                    self.audit.append({"text_sha256": digest(text), "start": span["start"], "end": span["end"], **audit})
                tokens = self.tokenizer.pad(encoded, padding=True, return_tensors="pt")
                with self.torch.no_grad():
                    values = self.model(**tokens).last_hidden_state[:, 0]
                    values = self.torch.nn.functional.normalize(values, dim=-1).cpu().numpy()
                for (text, _), vector in zip(batch, values):
                    accumulated[text].append(vector)
            for text, vectors in accumulated.items():
                mean = np.mean(vectors, axis=0)
                self.cache[text] = mean / max(float(np.linalg.norm(mean)), 1e-9)
        else:
            for start in range(0, len(unseen), 32):
                batch = unseen[start:start + 32]
                values = self.vectorizer.transform(batch).toarray().astype(np.float32)
                self.cache.update(zip(batch, values))

    def encode(self, texts):
        self.prepare(texts)
        return np.stack([self.cache[text] for text in texts])


def topology(payload, preserve_scope=False, merge_sources=True):
    x = model_input(payload)
    nodes, edges = [], []
    def node(kind, text, key=None):
        nodes.append({"kind": kind, "text": text, "key": key})
        return len(nodes) - 1
    def edge(a, b, relation):
        edges.append((a, b, RELATIONS.index(relation)))
    query = node("query", x["proposition"])
    conditions = {c["id"]: node("condition", c["text"], c["id"]) for c in x["conditions"]}
    sources, evidence_nodes, canonical = {}, {}, {}
    visible = []
    for e in visible_evidence(x):
        if preserve_scope:
            for span in scoped_spans(e['text']):
                visible.append({**e, **span, 'original_id': e['id'], 'original_length': len(e['text']),
                                'id': f"{e['id']}@{span['start']}"})
        else:
            visible.append(e)
    for evidence in visible:
        # Same source-derived identical information has one neural node. Two
        # independently supported artifacts are NOT collapsed into one source.
        signature = evidence_signature(evidence)
        # An equivalence attestation applies to its complete assertion only.
        full_assertion = not preserve_scope or (evidence['start'] == 0 and evidence['end'] == evidence['original_length'])
        if preserve_scope and not full_assertion:
            signature = (signature, evidence['text'], evidence['start'], evidence['end'])
        if merge_sources and signature in canonical:
            evidence_nodes[evidence["id"]] = canonical[signature]
            continue
        node_text = evidence.get('equivalence', {}).get('canonical_text', evidence['text']) if full_assertion else evidence['text']
        eid = node("evidence", node_text, evidence.get('original_id', evidence["id"]))
        nodes[eid].update({k: evidence[k] for k in ('scope_id', 'scope_kind', 'scope_complete', 'start', 'end', 'original_length', 'live_supports') if k in evidence})
        canonical[signature] = eid
        evidence_nodes[evidence["id"]] = eid
        for source in sorted({s for group in evidence["live_supports"] for s in group}):
            if source not in sources:
                sources[source] = node("source", "可见材料来源")
            edge(sources[source], eid, "origin")
            edge(eid, sources[source], "origin_reverse")
        for cid in conditions.values():
            edge(eid, cid, "candidate")
            edge(cid, eid, "candidate_reverse")
    # Reify AND / OR provenance instead of silently treating every origin edge
    # as an independent corroborating source. Actual lineage groups are visible.
    processed = set()
    for evidence in visible:
        eid = evidence_nodes[evidence["id"]]
        if eid in processed:
            continue
        processed.add(eid)
        if len(evidence["live_supports"]) > 1 or any(len(g) > 1 for g in evidence["live_supports"]):
            root = node("any", "替代来源支持")
            for group in evidence["live_supports"]:
                joint = node("all", "联合来源支持")
                for sid in group:
                    edge(sources[sid], joint, "requires")
                edge(joint, root, "alternative")
            edge(root, eid, "origin")
    def expression(expr):
        if isinstance(expr, str):
            return conditions[expr]
        if isinstance(expr, bool):
            return node("constant", "恒真" if expr else "恒假")
        op, children = next(iter(expr.items()))
        current = node(op, {"all": "全部条件同时满足", "any": "至少一个条件满足", "not": "条件不成立"}[op])
        for child in ([children] if op == "not" else children):
            index = expression(child)
            edge(index, current, {"all": "requires", "any": "alternative", "not": "negates"}[op])
            edge(current, index, "reverse_rule")
        return current
    rule = expression(x["rule"])
    edge(rule, query, "rule_query")
    for constraint in x.get("constraints", []):
        edge(expression(constraint), query, "requires")
    # Only explicit clause numbers visible in text can induce reference edges.
    numbered = {}
    for eid in set(evidence_nodes.values()):
        match = re.match(r"\s*第([一二三四五六七八九十\d]+)条", nodes[eid]["text"])
        if match:
            numbered[match[1]] = eid
    for eid in set(evidence_nodes.values()):
        for reference in re.findall(r"第([一二三四五六七八九十\d]+)条", nodes[eid]["text"]):
            if reference in numbered and numbered[reference] != eid:
                edge(numbered[reference], eid, "reference")
    edges.extend((i, i, 0) for i in range(len(nodes)))
    return nodes, sorted(set(edges)), query, conditions


def graph_record(record, encoder: FrozenEncoder, extractor=None, preserve_scope=False, merge_sources=True):
    x = model_input(record["input"])
    nodes, edges, query, conditions = topology(x, preserve_scope=preserve_scope, merge_sources=merge_sources)
    extracted = extractor.extract(x) if extractor else None
    embeddings = encoder.encode([n["text"] for n in nodes])
    extra = np.zeros((len(nodes), len(NODE_TYPES) + 6), dtype=np.float32)
    for i, n in enumerate(nodes):
        extra[i, NODE_TYPES.index(n["kind"])] = 1.
        if n["kind"] == "condition" and extracted:
            extra[i, len(NODE_TYPES):len(NODE_TYPES) + 3] = extracted["condition_features"][n["key"]]
        facts = numeric_facts(n["text"])
        extra[i, -3:] = [min(len(facts["amounts"]), 5) / 5., min(len(facts["dates"]), 5) / 5.,
                         float("未" in n["text"] or "不" in n["text"])]
    labels = ("T", "F", "U")
    gold = record.get("targets", {})
    live_records = {e['id']:e for e in visible_evidence(x)}
    equivalents = {eid: {other for other, candidate in live_records.items()
                         if evidence_signature(candidate) == evidence_signature(e)} for eid,e in live_records.items()}
    scope_lookup = {c['id']: c.get('scope_id') for c in x['conditions']}
    candidate_paths = {}
    for cid in conditions:
        paths = []
        for i, n in enumerate(nodes):
            if n['kind'] != 'evidence':
                continue
            scope = scope_lookup[cid]
            compatible = not scope or not n.get('scope_id') or scope == n['scope_id']
            paths.append({'indices': [i], 'usable': compatible and n.get('scope_complete', True),
                          'scope_features': [float(bool(scope) and scope == n.get('scope_id')),
                                             float(n.get('scope_kind') == 'exception' or '除外' in n['text']),
                                             float(n.get('scope_complete', True))]})
        for group in x.get('candidate_groups', []):
            if group['condition'] != cid:
                continue
            indices = [i for i, n in enumerate(nodes) if n['kind'] == 'evidence' and
                       equivalents[n['key']] & set(group['evidence_ids'])]
            found = {k for i in indices for k in equivalents[nodes[i]['key']]} & set(group['evidence_ids'])
            if found == set(group['evidence_ids']):
                paths.append({'indices': indices, 'usable': all(nodes[i].get('scope_complete', True) for i in indices),
                              'scope_features': [0., 0., 1.], 'origin': group['origin']})
        candidate_paths[cid] = paths
    def pair_label(cid, n):
        # Whole-evidence labels must not be copied onto every segmented window.
        if 'original_length' in n and n['end']-n['start'] != n['original_length']:
            values = {s['relation'] for s in gold.get('spans', []) if s['condition'] == cid and
                      s['evidence_id'] == n['key'] and s['start'] == n['start'] and s['end'] == n['end']}
            return next(iter(values)) if len(values) == 1 else None
        return gold.get('evidence', {}).get(f"{cid}|{n['key']}")
    return {"features": np.concatenate([embeddings, extra], axis=1), "edges": np.array(edges, dtype=np.int64),
            'candidate_paths': candidate_paths, 'structure_origin': x.get('structure_origin', 'given_rule'),
            "query": query, "condition_indices": list(conditions.values()), "condition_ids": list(conditions),
            "rule_expression": x["rule"], "constraint_expressions": x["constraints"],
            "condition_lookup": conditions,
            "condition_labels": [labels.index(gold.get("conditions", {})[c]) if c in gold.get("conditions", {}) else -1 for c in conditions],
            "label": labels.index(gold["label"]) if gold.get("label") in labels else -1,
            "severity": gold.get("severity") if gold.get("severity") is not None else -1,
            "related": int(gold["related"]) if gold.get("related") is not None else -1,
            "evidence_pairs": [(conditions[cid], evidence_index,
                ("irrelevant", "support", "refute").index(pair_label(cid, n)))
                for evidence_index, n in enumerate(nodes) if n["kind"] == "evidence"
                for cid in conditions if pair_label(cid, n) in ("irrelevant", "support", "refute")],
            "meta": record["meta"], "nodes": nodes}
