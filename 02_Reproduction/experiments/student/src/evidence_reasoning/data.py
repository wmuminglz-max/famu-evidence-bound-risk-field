"""Reproducible synthetic controls and explicit weak-data adapters."""
from __future__ import annotations

import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from .schema import digest, validate_record
from .logic import solve
from .provenance import withdraw, visible_evidence
from .boundary import matched_pairs


def read_records(path) -> list[dict]:
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()] if path.suffix == ".jsonl" else json.loads(text)
    if not isinstance(rows, list):
        raise ValueError("Dataset must be a JSON array or JSONL")
    return rows


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def root_weights(rows, class_weights: dict | None = None):
    """Every root has total mass 1, including after class reweighting."""
    raw, mass = [], Counter()
    for row in rows:
        weight = (class_weights or {}).get(row.get("targets", {}).get("label"), 1.)
        raw.append(weight)
        mass[row["meta"]["root_id"]] += weight
    return [w / mass[row["meta"]["root_id"]] for row, w in zip(rows, raw)]


def audit_splits(rows: list[dict]) -> dict:
    memberships = defaultdict(set)
    samples, duplicates = set(), []
    for row in rows:
        validate_record(row)
        meta = row["meta"]
        if meta["sample_id"] in samples:
            duplicates.append(meta["sample_id"])
        samples.add(meta["sample_id"])
        for kind in ("group_id", "root_id", "family_id", "contract_id", "template_family_id", "original_source_id"):
            if meta.get(kind):
                memberships[(kind, meta[kind])].add(meta["split"])
        for source_hash in meta.get("source_document_sha256", []):
            memberships[("source_document_sha256", source_hash)].add(meta["split"])
        # Complete input duplicates across partitions are invalid, too. Root ids
        # are not used here; observations/targets do not affect this fingerprint.
        from .schema import model_input
        memberships[("input", digest(model_input(row["input"])))].add(meta["split"])
    leaks = [{"kind": k[0], "id": k[1], "splits": sorted(v)} for k, v in memberships.items() if len(v) > 1]
    if leaks or duplicates:
        raise ValueError(f"Dataset isolation failure: {leaks[:4]}, duplicate ids={duplicates[:4]}")
    audit_group_identity(rows)
    return {"records": len(rows), "roots": len({r["meta"]["root_id"] for r in rows}),
            "groups": len({r["meta"]["group_id"] for r in rows}),
            "by_split": dict(Counter(r["meta"]["split"] for r in rows)),
            "label_status": dict(Counter(r["meta"]["label_status"] for r in rows)), "leaks": []}


def audit_group_identity(rows):
    """Related roots/templates must also stay together inside cross-fit folds."""
    identity_groups = defaultdict(set)
    for row in rows:
        meta = row["meta"]
        keys = [(kind, meta[kind]) for kind in ("root_id", "family_id", "contract_id", "template_family_id", "original_source_id")
                if meta.get(kind) and meta[kind] != "not_applicable"]
        keys += [("source_document_sha256", value) for value in meta.get("source_document_sha256", [])]
        for key in keys:
            identity_groups[key].add(meta["group_id"])
    if any(len(groups) > 1 for groups in identity_groups.values()):
        raise ValueError("Dataset isolation failure: related roots/source families assigned to different groups")


def group_folds(rows: list[dict], folds: int = 3, seed: int = 20260920):
    audit_group_identity(rows)
    groups = sorted({r["meta"]["group_id"] for r in rows})
    if len(groups) < folds or folds < 2:
        raise ValueError("Not enough independent groups for cross-fitting")
    random.Random(seed).shuffle(groups)
    assignment = {group: i % folds for i, group in enumerate(groups)}
    return [assignment[r["meta"]["group_id"]] for r in rows]


CONCEPTS = ["完成验收", "支付价款", "交付设备", "提交清单", "取得许可", "发出通知", "完成检验", "交回资料"]


def synthetic_records(roots: int = 120, seed: int = 20260920) -> list[dict]:
    """Formal controls, NOT independent lawyer-labelled contracts."""
    if roots < 20:
        raise ValueError("Use at least 20 roots to exercise all partitions")
    rng = random.Random(seed)
    records = []
    split_order = list(range(roots))
    rng.shuffle(split_order)
    for index in range(roots):
        root = "SYN-" + digest([seed, index])[:16]
        split_index = split_order[index] % 10
        split = ("train" if split_index < 6 else "validation" if split_index < 8 else
                 "calibration" if split_index == 8 else "test")
        chosen = rng.sample(CONCEPTS, 4)
        conditions = [{"id": f"c{i}", "text": text} for i, text in enumerate(chosen)]
        shape = index % 6
        rules = [{"all": ["c0", "c1"]}, {"any": ["c0", "c1"]},
                 {"all": ["c0", "c1", {"not": "c2"}]},
                 {"any": [{"all": ["c0", "c1"]}, "c2"]},
                 {"all": [{"any": ["c0", "c1"]}, {"not": "c2"}]},
                 {"any": [{"all": ["c0", "c1"]}, {"all": ["c2", "c3"]}]}]
        values = [bool(rng.randrange(2)) for _ in conditions]
        x = {"proposition": "在本题明确规则下，是否满足约定条件？", "rule": rules[shape],
             "conditions": conditions, "sources": [], "evidence": [], "observations": []}
        # Position, duplicate support and assert/deny wording vary independently
        # of the final proposition label. Opaque identifiers are not embedded.
        for ci, condition in enumerate(conditions):
            alternatives = []
            for ai in range(2 if ci == 1 and index % 2 == 0 else 1):
                sid, eid = f"s{ci}_{ai}", f"e{ci}_{ai}"
                x["sources"].append({"id": sid})
                assert_text = ["双方确认已", "记录载明已经", "交接记录确认已"][rng.randrange(3)]
                deny_text = ["双方确认未", "记录明确否认已经", "交接记录确认尚未"][rng.randrange(3)]
                text = f"事项{index + 100}：{assert_text if values[ci] else deny_text}{condition['text']}。"
                x["evidence"].append({"id": eid, "text": text, "supports": [[sid]]})
                alternatives.append([eid])
            x["observations"].append({"condition": condition["id"], "value": values[ci],
                                      "evidence_groups": alternatives, "origin": "synthetic_generator"})
        x["sources"].extend([{"id": "noise0"}, {"id": "noise1"}])
        x["evidence"].extend([{"id": f"en{i}", "text": f"事项{index + 100}的材料仅用于归档编号，不约定上述事项。",
                               "supports": [[f"noise{i}"]]} for i in range(2)])
        rng.shuffle(x["evidence"])
        views = [("original", x)]
        # All source-single withdrawals, plus genuine minimal joint boundaries.
        for source in x["sources"]:
            views.append(("withdraw", withdraw(x, [source["id"]])))
        pairs = matched_pairs(x)
        for pair in pairs:
            views.append(("critical", withdraw(x, pair["critical"]["sources"])))
            if pair["control"]:
                views.append(("control", withdraw(x, pair["control"]["sources"])))
        copied = copy.deepcopy(x)
        original_evidence = x["evidence"][0]
        copied["evidence"].append({"id": "copied_view", "text": original_evidence["text"],
                                    "parents": [[original_evidence["id"]]]})
        views.append(("duplicate", copied))
        # Explicit fact negation is distinct from withdrawing observations.
        flipped = copy.deepcopy(x)
        flipped["observations"][0]["value"] = not values[0]
        e0 = next(e for e in flipped["evidence"] if e["id"] == "e0_0")
        e0["text"] = f"事项{index + 100}：双方确认{'未' if values[0] else '已'}{chosen[0]}。"
        views.append(("fact_change", flipped))
        seen, view_ids = set(), {}
        for operation, view in views:
            fingerprint = digest(view)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            answer = solve(view)
            sid = "EX-" + digest([root, fingerprint])[:18]
            view_ids[fingerprint] = sid
            evidence_targets = {}
            live = {e["id"] for e in visible_evidence(view)}
            for condition in conditions:
                for eid in live:
                    truth = "irrelevant"
                    for observation in view["observations"]:
                        reference_id = original_evidence["id"] if eid == "copied_view" else eid
                        if observation["condition"] == condition["id"] and any(reference_id in g and set(g) <= live for g in observation["evidence_groups"]):
                            truth = "support" if observation["value"] else "refute"
                    evidence_targets[f"{condition['id']}|{eid}"] = truth
            records.append({"input": view, "targets": {"label": answer["proposition_label"],
                             "conditions": answer["condition_states"], "evidence": evidence_targets,
                             "severity": None, "related": True},
                            "meta": {"sample_id": sid, "root_id": root, "group_id": root, "family_id": root,
                                     "split": split, "label_status": "synthetic_verified", "operation": operation,
                                     "rule_shape": shape, "human_gold": False, "domain": "formal_control",
                                     "pair_ids": []}})
        root_rows = [r for r in records if r["meta"]["root_id"] == root]
        for pair in pairs:
            if not pair["matched"]:
                continue
            a = view_ids.get(digest(withdraw(x, pair["critical"]["sources"])))
            b = view_ids.get(digest(withdraw(x, pair["control"]["sources"])))
            if a and b:
                for row in root_rows:
                    if row["meta"]["sample_id"] in {a, b}:
                        row["meta"]["pair_ids"].append([a, b])
    audit_splits(records)
    return records


def adapt_legacy(rows: list[dict]) -> list[dict]:
    mapping = {"established": "T", "not_established": "F", "uncertain": "U"}
    severity = {"risk_low": 0, "risk_medium": 1, "risk_high": 2, "risk_critical": 3}
    output = []
    for row in rows:
        inp = row["input"]
        label = mapping.get(row.get("target_judgment"))
        if row.get("target_outcome") == "unrelated":
            label = None
        # A single proposition variable is explicit; no invented expert conditions.
        x = {"proposition": inp["risk_proposition"], "rule": "q", "task_mode": "text_only",
             "conditions": [{"id": "q", "text": inp["risk_proposition"]}],
             "sources": [{"id": "contract"}], "observations": [], "evidence": []}
        for key in ("clause_a", "clause_b", "fact_context"):
            if inp.get(key):
                x["evidence"].append({"id": key, "text": str(inp[key]), "supports": [["contract"]]})
        split = row.get("split", "regression")
        split = {"external_diagnostic": "external", "fresh_behavior_only": "regression"}.get(split, split)
        output.append({"input": x, "targets": {"label": label, "conditions": {}, "evidence": {},
                       "severity": severity.get(row.get("target_outcome")),
                       "related": row.get("target_outcome") != "unrelated"},
                       "meta": {"sample_id": row["sample_id"], "root_id": row["root_id"],
                                "group_id": row["group_id"], "split": split, "label_status": "weak_pending",
                                "domain": "legacy_regression", "human_gold": False,
                                "original_outcome": row.get("target_outcome"), "pair_ids": []}})
    return output


def annotation_queue(rows: list[dict]) -> list[dict]:
    """Blind to model suggestions, existing answers and explanations."""
    selected = {}
    for row in rows:
        root = row["meta"]["root_id"]
        if root in selected:
            continue
        selected[root] = {"root_id": root, "group_id": row["meta"]["group_id"],
                          "input": {k: v for k, v in row["input"].items() if k != "observations"},
                          "independent_annotation_1": None, "independent_annotation_2": None,
                          "adjudication": None, "gold_status": "pending"}
    return list(selected.values())
