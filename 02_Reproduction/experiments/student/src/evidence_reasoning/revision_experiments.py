"""Controlled expert-review experiments, always written outside frozen bundles."""
from __future__ import annotations

import copy
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .data import read_records, write_json, audit_splits
from .schema import digest, model_input
from .logic import solve
from .graphs import FrozenEncoder, graph_record, RELATIONS
from .neural import EvidenceNetwork, collate, objective, fit, predict, save_model
from .metrics import classification, group_bootstrap_difference, weights
from .fusion import softmax
from .training import evaluate, labels


def fresh_output(output):
    path = Path(output).resolve()
    if path.exists() and any(path.iterdir()):
        raise ValueError("Choose a new empty output directory; frozen experiments are never overwritten")
    path.mkdir(parents=True, exist_ok=True)
    return path


def binding_inputs():
    texts = ["完成验收", "支付价款", "交付设备", "发出通知"]
    values = [True, True, False, False]
    a = {"proposition": "按给定规则，条件是否满足？", "rule": {"any": [{"all": ["a", "b"]}, {"all": ["c", "d"]}]},
         "conditions": [{"id": c, "text": t} for c, t in zip("abcd", texts)],
         "sources": [{"id": "s" + c} for c in "abcd"],
         "evidence": [{"id": "e" + c, "text": ("双方确认已" if v else "双方确认未") + t + "。", "supports": [["s" + c]]}
                      for c, t, v in zip("abcd", texts, values)],
         "observations": [{"condition": c, "value": v, "evidence_groups": [["e" + c]]} for c, v in zip("abcd", values)]}
    b = copy.deepcopy(a)
    b["rule"] = {"any": [{"all": ["a", "c"]}, {"all": ["b", "d"]}]}
    return a, b


def _control_record(x, i):
    answer = solve(x)
    return {"input": x, "targets": {"label": answer["proposition_label"]},
            "meta": {"sample_id": str(i), "root_id": str(i), "group_id": str(i), "split": "train",
                     "label_status": "synthetic_verified", "pair_ids": []}}


def _memorize(graphs, architecture, seed, steps=300):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    model = EvidenceNetwork(graphs[0]["features"].shape[1], architecture, hidden=32, dropout=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.005, weight_decay=0.)
    batch = collate(graphs)
    for step in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model(batch), batch, torch.ones(len(graphs)), auxiliary=False, boundary=False)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
        optimizer.step()
        if (step + 1) % 20 == 0:
            prediction = predict(model, graphs)["logits"].argmax(1)
            if np.array_equal(prediction, [g["label"] for g in graphs]) and float(loss.detach()) < .02:
                break
    logits = predict(model, graphs)["logits"]
    return {"seed": seed, "updates": step + 1, "loss": float(loss.detach()),
            "training_fit_only": classification([g["label"] for g in graphs], logits.argmax(1), [g["meta"]["root_id"] for g in graphs]),
            "predictions": logits.argmax(1).tolist()}


def run_controls(output, seeds=(20260928, 20260929, 20260930)):
    output = fresh_output(output)
    encoder = FrozenEncoder(dimension=64)
    inputs = []
    for text in ("完成验收", "支付价款", "交付设备"):
        for state in (True, False, None):
            x = {"proposition": "是否" + text + "？", "rule": "a", "conditions": [{"id": "a", "text": text}],
                 "sources": [{"id": "s"}], "evidence": [], "observations": []}
            if state is not None:
                x["evidence"] = [{"id": "e", "text": ("双方确认已" if state else "双方确认未") + text + "。", "supports": [["s"]]}]
                x["observations"] = [{"condition": "a", "value": state, "evidence_groups": [["e"]]}]
            inputs.append(x)
    rows = [_control_record(x, i) for i, x in enumerate(inputs)]
    graphs = [graph_record(r, encoder) for r in rows]
    collision_rows = [_control_record(x, i) for i, x in enumerate(binding_inputs())]
    collision = [graph_record(r, encoder) for r in collision_rows]
    report = {"scope": "synthetic training-fit diagnostic only; no legal/generalization claim; gold observations stripped from features",
              "class_counts": dict(Counter(r["targets"]["label"] for r in rows)), "balanced_fit": {}, "binding_fit": {}}
    for architecture in ("mlp", "rule_tree", "rgcn"):
        report["balanced_fit"][architecture] = [_memorize(graphs, architecture, seed) for seed in seeds]
        print(f"balanced fit {architecture} complete", flush=True)
    for architecture in ("mlp", "rule_tree"):
        report["binding_fit"][architecture] = [_memorize(collision, architecture, seed) for seed in seeds]
    report["balanced_fit_passed"] = all(r["training_fit_only"]["accuracy"] == 1. for runs in report["balanced_fit"].values() for r in runs)
    report["binding_repair_passed"] = all(r["training_fit_only"]["accuracy"] == 1. for r in report["binding_fit"]["rule_tree"])
    write_json(output / "controls.json", report)
    write_json(output / "balanced_rows.json", rows)
    write_json(output / "binding_rows.json", collision_rows)
    return report


def transform_graph(graph, variant, seed):
    g = copy.deepcopy(graph)
    if variant == "no_source":
        keep = [i for i, n in enumerate(g["nodes"]) if n["kind"] != "source"
                and not (n["kind"] in {"all", "any"} and n["text"] in {"替代来源支持", "联合来源支持"})]
        remap = {old: new for new, old in enumerate(keep)}
        g["features"] = g["features"][keep]
        g["nodes"] = [g["nodes"][i] for i in keep]
        g["edges"] = np.asarray([(remap[a], remap[b], r) for a, b, r in g["edges"] if a in remap and b in remap], dtype=np.int64)
        g["query"] = remap[g["query"]]
        g["condition_indices"] = [remap[i] for i in g["condition_indices"]]
        g["condition_lookup"] = {c: remap[i] for c, i in g["condition_lookup"].items()}
        g["evidence_pairs"] = [(remap[c], remap[e], y) for c, e, y in g["evidence_pairs"]]
    elif variant == "shuffle_relations":
        # Shuffle individual edge labels, not a learnable global renaming of types.
        signature = digest({"features": g["features"].tolist(), "edges": g["edges"].tolist()})
        rng = np.random.default_rng(seed + int(signature[:8], 16))
        mask = g["edges"][:, 2] != RELATIONS.index("self")
        g["edges"][mask, 2] = rng.permutation(g["edges"][mask, 2])
    return g


def compare_models(dataset, output, encoder_path=None, seeds=(20260928,20260929,20260930), epochs=16):
    """Six prespecified arms retrained on the same source split and compute budget."""
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import LogisticRegression
    from .extraction import ConditionExtractor
    output = fresh_output(output)
    rows = read_records(dataset)
    for row in rows:
        # Explicit migration of known proxy records; not an inference from syntax.
        if row["meta"].get("domain") in {"legacy_regression", "labor_excerpt"}:
            row["input"]["task_mode"] = "text_only"
    audit = audit_splits(rows)
    splits = {s: [r for r in rows if r["meta"]["split"] == s] for s in ("train", "validation", "test")}
    if any(not part for part in splits.values()):
        raise ValueError("Comparison needs frozen nonempty train, validation and test splits")
    torch.set_num_threads(4)
    encoder = FrozenEncoder(encoder_path)
    graphs = {s: [graph_record(r, encoder) for r in part] for s, part in splits.items()}
    dim = graphs["train"][0]["features"].shape[1]
    nparams = lambda arch, h: sum(p.numel() for p in EvidenceNetwork(dim, arch, hidden=h).parameters())
    target = nparams("rgcn", 48)
    matched_hidden = min(range(24, 129), key=lambda h: abs(nparams("mlp", h) - target))
    arms = [("mlp", "mlp", "full", 48), ("mlp_matched", "mlp", "full", matched_hidden),
            ("rule_tree", "rule_tree", "full", 48), ("rgcn", "rgcn", "full", 48),
            ("rgcn_no_source", "rgcn", "no_source", 48), ("rgcn_shuffled", "rgcn", "shuffle_relations", 48)]
    steps = max(4, (len({r["meta"]["root_id"] for r in splits["train"]}) + 7) // 8)
    protocol = {"dataset_sha256": digest(rows), "audit": audit, "seeds": list(seeds), "epochs": epochs, "steps_per_epoch": steps,
                "selection": "validation root-balanced NLL only", "encoder": str(encoder_path) if encoder_path else "hash_diagnostic",
                "arms": arms, "auxiliary": True, "boundary": False, "extracted_condition_features": "disabled_no_cross_fit",
                "scope": "existing formal controls and previously seen weak-label regression; not new legal blind test",
                "graph_quality": "supplied provenance/rule plus observable candidate edges; not predicted legal relation accuracy",
                "test_role": "diagnostic; no model deployment selection", "decision_authority": False}
    write_json(output / "protocol.json", protocol)
    report, predictions = {"protocol": protocol, "arms": {}, "baselines": {}}, {}
    for name, architecture, variant, hidden in arms:
        runs, test_values = [], []
        for seed in seeds:
            started = time.perf_counter()
            transformed = {s: [transform_graph(g, variant, seed) for g in part] for s, part in graphs.items()}
            model, meta = fit(transformed["train"], transformed["validation"], architecture, seed, epochs,
                              auxiliary=True, boundary=False, steps_per_epoch=steps, hidden=hidden)
            save_model(output / f"models/{name}_{seed}.pt", model, meta | {"variant": variant})
            p = softmax(predict(model, transformed["test"])["logits"])
            test_values.append(p)
            runs.append({"seed": seed, "parameters": meta["parameter_count"], "seconds": time.perf_counter() - started,
                         "all": evaluate(splits["test"], p), "slices": _slices(splits["test"], p)})
            print(f"{name} seed={seed} complete", flush=True)
        predictions[name] = np.mean(test_values, axis=0)
        report["arms"][name] = {"seeds": runs, "ensemble": evaluate(splits["test"], predictions[name]),
                               "slices": _slices(splits["test"], predictions[name])}
        write_json(output / "comparison.partial.json", report)
    train = [r for r in splits["train"] if r["targets"].get("label") in {"T", "F", "U"}]
    y, roots = labels(train), [r["meta"]["root_id"] for r in train]
    counts = np.bincount(y, weights=weights(roots), minlength=3)
    majority = np.tile(np.eye(3)[int(counts.argmax())], (len(splits["test"]), 1))
    report["baselines"]["majority"] = evaluate(splits["test"], majority)
    def text(row):
        x = model_input(row["input"])
        mapping = {c["id"]: c["text"] for c in x["conditions"]}
        def expression(expr):
            if isinstance(expr, str):
                return mapping[expr]
            if isinstance(expr, bool):
                return str(expr)
            op, children = next(iter(expr.items()))
            return op + "(" + ",".join(expression(c) for c in ([children] if op == "not" else children)) + ")"
        from .provenance import visible_evidence
        return "\n".join([x["proposition"], expression(x["rule"])] + [expression(c) for c in x["constraints"]] + [e["text"] for e in visible_evidence(x)])
    vectorizer = HashingVectorizer(n_features=8192, analyzer="char", ngram_range=(1, 3), alternate_sign=False)
    linear = LogisticRegression(C=3., max_iter=500, random_state=seeds[0])
    linear.fit(vectorizer.transform([text(r) for r in train]), y, sample_weight=weights(roots))
    probabilities = np.zeros((len(splits["test"]), 3))
    probabilities[:, linear.classes_] = linear.predict_proba(vectorizer.transform([text(r) for r in splits["test"]]))
    predictions["character_linear"] = probabilities
    report["baselines"]["character_linear"] = evaluate(splits["test"], probabilities)
    extractor = ConditionExtractor().fit(splits["train"])
    write_json(output / "extractor.json", extractor.payload)
    formal = [r for r in splits["test"] if r["meta"].get("domain") == "formal_control"]
    from .training import _extractor_evaluation
    report["baselines"]["extraction_executor_formal"] = _extractor_evaluation(extractor, formal)
    test_roots = [r["meta"]["root_id"] for r in splits["test"]]
    groups = [r["meta"]["group_id"] for r in splits["test"]]
    report["paired_comparisons"] = {name: {metric: group_bootstrap_difference(labels(splits["test"]), predictions["rgcn"].argmax(1),
        predictions[name].argmax(1), test_roots, groups=groups, metric=metric) for metric in ("accuracy", "macro_f1")}
        for name in ("mlp", "mlp_matched", "rule_tree", "rgcn_no_source", "rgcn_shuffled")}
    write_json(output / "predictions.json", [{"sample_id": r["meta"]["sample_id"], "root_id": r["meta"]["root_id"], "group_id": r["meta"]["group_id"],
        "truth": r["targets"].get("label"), "probabilities": {name: p[i].tolist() for name, p in predictions.items()}}
        for i, r in enumerate(splits["test"])])
    write_json(output / "comparison.json", report)
    return report


def _slices(rows, probabilities):
    result = {}
    for domain in sorted({r["meta"].get("domain", "unspecified") for r in rows}):
        ix = [i for i, r in enumerate(rows) if r["meta"].get("domain", "unspecified") == domain]
        result[domain] = evaluate([rows[i] for i in ix], probabilities[ix])
    return result


def compare_severity(dataset, output, encoder_path, seeds=(20260928,20260929,20260930)):
    from .severity import train_severity
    output = fresh_output(output)
    rows = read_records(dataset)
    audit_splits(rows)
    result = {"scope": "same frozen weak-label positive-risk data; no production replacement", "modes": {}}
    for mode in ("legacy", "root", "root_class"):
        folder = output / mode
        write_json(folder / "dataset.json", rows)
        write_json(folder / "bundle/manifest.json", {"encoder_path": str(Path(encoder_path).resolve()), "files_sha256": {}})
        result["modes"][mode] = train_severity(folder, weight_mode=mode, seeds=seeds)
    write_json(output / "severity_comparison.json", result)
    return result
