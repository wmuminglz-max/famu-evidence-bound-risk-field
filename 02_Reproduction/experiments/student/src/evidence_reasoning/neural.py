"""Same-input node/edge MLP and residual relation-aware variable graph network."""
from __future__ import annotations

from collections import defaultdict
import copy
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .graphs import RELATIONS, NODE_TYPES


def collate(graphs):
    features, edges, batches, conditions, condition_y, pairs, pair_y, query = [], [], [], [], [], [], [], []
    offset = 0
    for i, graph in enumerate(graphs):
        features.append(graph["features"])
        e = graph["edges"].copy()
        e[:, :2] += offset
        edges.append(e)
        batches.extend([i] * len(graph["features"]))
        query.append(graph["query"] + offset)
        conditions.extend(c + offset for c in graph["condition_indices"])
        condition_y.extend(graph["condition_labels"])
        for c, e, y in graph.get("evidence_pairs", []):
            pairs.append([c + offset, e + offset])
            pair_y.append(y)
        offset += len(graph["features"])
    return {"x": torch.tensor(np.concatenate(features), dtype=torch.float32),
            "edges": torch.tensor(np.concatenate(edges), dtype=torch.long),
            "batch": torch.tensor(batches), "count": len(graphs), "query": torch.tensor(query),
            "conditions": torch.tensor(conditions, dtype=torch.long), "condition_y": torch.tensor(condition_y),
            "pairs": torch.tensor(pairs, dtype=torch.long).reshape(-1, 2), "pair_y": torch.tensor(pair_y),
            "y": torch.tensor([g["label"] for g in graphs]),
            "severity": torch.tensor([g["severity"] for g in graphs]),
            "related": torch.tensor([g["related"] for g in graphs]), "graphs": graphs}


def mean_pool(x, batch, count):
    result = x.new_zeros((count, x.shape[-1]))
    result.index_add_(0, batch, x)
    denominator = x.new_zeros(count)
    denominator.index_add_(0, batch, x.new_ones(len(batch)))
    return result / denominator.clamp_min(1).unsqueeze(1)


class TypedLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        # Basis decomposition keeps relation parameters bounded and trainable.
        self.bases = nn.Parameter(torch.empty(4, hidden, hidden))
        self.coefficients = nn.Parameter(torch.empty(len(RELATIONS), 4))
        nn.init.xavier_uniform_(self.bases)
        nn.init.xavier_uniform_(self.coefficients)
        self.update = nn.Linear(hidden * 2, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, edges):
        matrices = torch.einsum("rb,bhi->rhi", self.coefficients, self.bases)
        aggregated = torch.zeros_like(x)
        degree = x.new_zeros(len(x))
        for relation in range(len(RELATIONS)):
            e = edges[edges[:, 2] == relation]
            if len(e):
                messages = x[e[:, 0]] @ matrices[relation]
                aggregated.index_add_(0, e[:, 1], messages)
                degree.index_add_(0, e[:, 1], x.new_ones(len(e)))
        aggregated = aggregated / degree.clamp_min(1).unsqueeze(1)
        return self.norm(x + F.gelu(self.update(torch.cat([x, aggregated], 1))))


class EvidenceNetwork(nn.Module):
    def __init__(self, input_dim, architecture="rgcn", hidden=64, layers=2, dropout=.12,
                 leaf_aggregation="mean", global_pooling=True, use_scope=True):
        super().__init__()
        if architecture not in {"rgcn", "mlp", "rule_tree"}:
            raise ValueError("architecture must be rgcn, mlp or rule_tree")
        if leaf_aggregation not in {'mean', 'attention'} or (leaf_aggregation == 'attention' and architecture != 'rule_tree'):
            raise ValueError('Condition attention is a rule_tree leaf option')
        self.config = dict(input_dim=input_dim, architecture=architecture, hidden=hidden, layers=layers, dropout=dropout,
                           leaf_aggregation=leaf_aggregation, global_pooling=global_pooling, use_scope=use_scope)
        self.project = nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.layers = nn.ModuleList([TypedLayer(hidden) for _ in range(layers)]) if architecture == "rgcn" else nn.ModuleList()
        self.node_mlp = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden))
        self.relation_embedding = nn.Embedding(len(RELATIONS), hidden)
        self.edge_mlp = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.readout = nn.Sequential(nn.Linear(hidden * 4, hidden * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden * 2, hidden), nn.GELU())
        self.answer = nn.Linear(hidden, 3)
        self.condition = nn.Linear(hidden, 3)
        self.evidence = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, 3))
        self.severity = nn.Linear(hidden, 3)  # P(level > k), cumulative ordinal targets.
        self.related = nn.Linear(hidden, 2)
        if architecture == "rule_tree":
            # Non-graph baseline: condition/evidence joint encoding followed by
            # recursive Boolean syntax composition. No gold observations or
            # condition targets are read, and no graph message passing occurs.
            self.leaf_pair = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, hidden))
            self.tree_ops = nn.ModuleDict({op: nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(),
                nn.Linear(hidden * 2, hidden), nn.Tanh()) for op in ("all", "any", "not")})
            self.tree_constants = nn.Embedding(2, hidden)
            if leaf_aggregation == 'attention':
                self.leaf_attention = nn.Sequential(nn.Linear(hidden*3+3, hidden), nn.GELU(), nn.Linear(hidden, 1))
                self.no_evidence = nn.Parameter(torch.zeros(hidden))
                self.no_evidence_score = nn.Linear(hidden, 1)
                self.joint_proof = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))

    def _rule_readout(self, x, batch):
        result, offset = [], 0
        condition_vectors = []
        for graph in batch["graphs"]:
            evidence = [offset + i for i, n in enumerate(graph["nodes"]) if n["kind"] == "evidence"]
            leaves = {}
            for cid, index in graph["condition_lookup"].items():
                leaf = x[offset + index]
                if self.config['leaf_aggregation'] == 'attention':
                    values, scores = [self.no_evidence], [self.no_evidence_score(leaf).squeeze(-1)]
                    for path in graph.get('candidate_paths', {}).get(cid, []):
                        if not path['usable']:
                            continue
                        h = x[[offset+i for i in path['indices']]]
                        # Joint support and alternatives remain separate paths.
                        e = h[0] if len(h) == 1 else self.joint_proof(h.sum(0) / len(h)**.5)
                        joint = torch.cat([leaf, e, leaf*e])
                        scope = leaf.new_tensor(path['scope_features']) if self.config['use_scope'] else leaf.new_zeros(3)
                        scores.append(self.leaf_attention(torch.cat([joint, scope])).squeeze(-1))
                        values.append(self.leaf_pair(joint))
                    attention = torch.stack(scores).softmax(0)
                    leaf = leaf + (torch.stack(values)*attention[:, None]).sum(0)
                elif evidence:
                    e = x[evidence]
                    c = leaf.expand(len(e), -1)
                    leaf = leaf + self.leaf_pair(torch.cat([c, e, c * e], -1)).mean(0)
                leaves[cid] = leaf
                condition_vectors.append(leaf)
            def compose(expr):
                if isinstance(expr, str):
                    return leaves[expr]
                if isinstance(expr, bool):
                    return self.tree_constants.weight[int(expr)]
                op, children = next(iter(expr.items()))
                children = [children] if op == "not" else children
                combined = torch.stack([compose(child) for child in children]).sum(0) / len(children) ** .5
                return self.tree_ops[op](combined)
            root = compose(graph["rule_expression"])
            constraints = graph["constraint_expressions"]
            if constraints:
                root = root + self.tree_ops["all"](torch.stack([compose(c) for c in constraints]).mean(0))
            result.append(root)
            offset += len(graph["features"])
        self._attention_conditions = torch.stack(condition_vectors) if condition_vectors else x.new_empty((0, x.shape[-1]))
        return torch.stack(result)

    def forward(self, batch):
        x = self.project(batch["x"])
        for layer in self.layers:
            x = layer(x, batch["edges"])
        if self.config["architecture"] in {"mlp", "rule_tree"}:
            x = x + self.node_mlp(x)
        src, dst, relation = batch["edges"].unbind(1)
        edge_features = self.edge_mlp(torch.cat([x[src], x[dst], self.relation_embedding(relation)], 1))
        c = batch["conditions"]
        query = x[batch["query"]]
        if self.config["architecture"] == "rule_tree":
            query = query + self._rule_readout(x, batch)
        pooled = torch.cat([query, mean_pool(x, batch["batch"], batch["count"]),
                            mean_pool(x[c], batch["batch"][c], batch["count"]),
                            mean_pool(edge_features, batch["batch"][dst], batch["count"])], 1)
        if not self.config['global_pooling']:
            pooled = torch.cat([query, query.new_zeros((len(query), query.shape[-1]*3))], 1)
        h = self.readout(pooled)
        pair = batch["pairs"]
        pair_x = torch.cat([x[pair[:, 0]], x[pair[:, 1]], x[pair[:, 0]] * x[pair[:, 1]]], 1)
        condition_values = self._attention_conditions if self.config['leaf_aggregation'] == 'attention' else x[c]
        return {"logits": self.answer(h), "conditions": self.condition(condition_values), "evidence": self.evidence(pair_x),
                "severity": self.severity(h), "related": self.related(h)}


def _masked_loss(logits, labels, item_weights):
    keep = labels >= 0
    if not keep.any():
        return logits.sum() * 0
    # Missing labels contribute neither loss nor normalization mass.
    return (F.cross_entropy(logits[keep], labels[keep].long(), reduction="none") * item_weights[keep]).sum() / item_weights[keep].sum().clamp_min(1e-9)


def objective(output, batch, root_weights, auxiliary=True, boundary=False, margin=.2):
    w = root_weights
    answer_loss = _masked_loss(output["logits"], batch["y"], w)
    related_loss = _masked_loss(output["related"], batch["related"], w)
    loss = answer_loss + .15 * related_loss
    mask = batch["severity"] >= 0
    if mask.any():
        ordinal = (batch["severity"][mask, None] > torch.arange(3)[None, :]).float()
        per_row = F.binary_cross_entropy_with_logits(output["severity"][mask], ordinal, reduction="none").mean(1)
        loss = loss + .2 * (per_row * w[mask]).sum() / w[mask].sum().clamp_min(1e-9)
        # Discourage invalid increasing cumulative probabilities.
        s = output["severity"][mask].sigmoid()
        monotonic = F.relu(s[:, 1:] - s[:, :-1]).mean(1)
        loss = loss + .03 * (monotonic * w[mask]).sum() / w[mask].sum().clamp_min(1e-9)
    if auxiliary:
        for key, labels, owners in (("conditions", batch["condition_y"], batch["batch"][batch["conditions"]]),
                                    ("evidence", batch["pair_y"], batch["batch"][batch["pairs"][:, 0]])):
            if len(labels):
                counts = torch.bincount(owners[labels >= 0], minlength=batch["count"]).clamp_min(1)
                item_weights = w[owners] / counts[owners]
                loss = loss + .35 * _masked_loss(output[key], labels, item_weights)
    if boundary:
        index = {g["meta"]["sample_id"]: i for i, g in enumerate(batch["graphs"])}
        pairs = {tuple(p) for g in batch["graphs"] for p in g["meta"].get("pair_ids", []) if all(s in index for s in p)}
        terms, pair_weights = [], []
        u = output["logits"].softmax(-1)[:, 2]
        for a, b in pairs:
            ia, ib = index[a], index[b]
            if batch["y"][ia] != 2 or batch["y"][ib] not in (0, 1):
                raise ValueError("Invalid boundary supervision")
            terms.append(F.relu(margin - u[ia] + u[ib]))
            pair_weights.append((w[ia] + w[ib]) / 2)
        if terms:
            pw = torch.stack(pair_weights)
            loss = loss + .3 * (torch.stack(terms) * pw).sum() / pw.sum()
    return loss


def predict(model, graphs, batch_size=48):
    model.eval()
    outputs = defaultdict(list)
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            result = model(collate(graphs[start:start + batch_size]))
            for key in ("logits", "severity", "related"):
                outputs[key].append(result[key].numpy())
    return {k: np.concatenate(v) for k, v in outputs.items()}


def fit(graphs, validation, architecture="rgcn", seed=20260920, epochs=18,
        auxiliary=True, boundary=True, grouped=True, steps_per_epoch=None, hidden=64, layers=2,
        leaf_aggregation="mean", global_pooling=True, use_scope=True):
    from .data import audit_group_identity
    audit_group_identity(graphs + validation)
    if any(g["meta"]["split"] != "train" for g in graphs):
        raise ValueError("Only training groups may update neural weights")
    if any(g["meta"]["split"] not in {"validation"} for g in validation):
        raise ValueError("Early stopping may only inspect validation groups")
    train_groups = {g["meta"]["group_id"] for g in graphs}
    if train_groups & {g["meta"]["group_id"] for g in validation}:
        raise ValueError("Early stopping group overlaps training")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = EvidenceNetwork(graphs[0]["features"].shape[1], architecture, hidden=hidden, layers=layers,
                            leaf_aggregation=leaf_aggregation, global_pooling=global_pooling, use_scope=use_scope)
    if leaf_aggregation == 'attention' and sum(p.numel() for p in model.parameters() if p.requires_grad) > 500_000:
        raise ValueError('Attention pilot exceeds the 500,000 trainable parameter budget')
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.015)
    groups = defaultdict(list)
    for g in graphs:
        groups[g["meta"]["root_id"]].append(g)
    rng = random.Random(seed)
    # Fixed updates for every A-E arm; root-uniform batches prevent view count dominance.
    steps = steps_per_epoch or max(4, (len(groups) + 7) // 8)
    best, best_loss, trace = None, float("inf"), []
    for epoch in range(epochs):
        model.train()
        losses = []
        for _ in range(steps):
            if grouped:
                selected = rng.sample(sorted(groups), min(6, len(groups)))
                batch_graphs = [g for root in selected for g in groups[root]]
            else:
                selected = [rng.choice(sorted(groups)) for _ in range(48)]
                batch_graphs = [rng.choice(groups[root]) for root in selected]
            batch = collate(batch_graphs)
            w = torch.tensor([1. / len(groups[g["meta"]["root_id"]]) if grouped else 1. for g in batch_graphs])
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model(batch), batch, w, auxiliary, boundary)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
            losses.append(float(loss.detach()))
        logits = predict(model, validation)["logits"]
        labels = np.array([g["label"] for g in validation])
        mask = labels >= 0
        from .metrics import weights
        if mask.any():
            val_weights = weights([g["meta"]["root_id"] for i,g in enumerate(validation) if mask[i]])
            val_loss = float(np.average(F.cross_entropy(torch.tensor(logits[mask]), torch.tensor(labels[mask]), reduction="none").numpy(), weights=val_weights))
        else:
            val_loss = float(np.mean(losses))
        trace.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_nll": val_loss})
        if val_loss < best_loss:
            best, best_loss = copy.deepcopy(model.state_dict()), val_loss
    model.load_state_dict(best)
    return model, {"config": model.config, "seed": seed, "epochs": epochs, "steps_per_epoch": steps,
                   "optimizer_updates": epochs * steps, "auxiliary": auxiliary, "boundary": boundary,
                   "grouped": grouped, "parameter_count": sum(p.numel() for p in model.parameters()),
                   "fit_groups": sorted(train_groups), "selection_groups": sorted({g["meta"]["group_id"] for g in validation}),
                   "label_statuses": sorted({g["meta"]["label_status"] for g in graphs}), "trace": trace}


def save_model(path, model, metadata):
    from pathlib import Path
    from .data import write_json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    write_json(path.with_suffix(".json"), metadata)


def load_model(path):
    import json
    from pathlib import Path
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    model = EvidenceNetwork(**meta["config"])
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model, meta
