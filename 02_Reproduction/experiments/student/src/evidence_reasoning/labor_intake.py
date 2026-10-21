"""Import excerpt annotations without upgrading them to adjudicated legal truth.

Only the supplied clause text and a fixed, disclosed proxy proposition enter
features. Contract titles, purported facts, reasons and labels stay in a sidecar.
This adapter performs no training, legal adjudication or production mutation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import unicodedata

from .schema import digest, validate_record

HEADERS = ["合同文件名", "条款位置", "条款原文", "相关情况", "判断结果（原值）",
           "风险等级（原值）", "理由或依据（法律来源）", "审阅人", "风险说明（Word）",
           "修改建议（Word）", "记录编号", "来源文件", "来源位置", "合并备注"]
PROPOSITION = "该条款存在对劳动者不利的合同风险。"
GRADES = {"提示风险": 0, "提示风险（低风险）": 0, "提示风险(低风险)": 0,
          "低风险": 0, "中风险": 1, "高风险": 2, "重大风险": 3}
GRADE_NAMES = ("提示风险（低风险）", "中风险", "高风险", "重大风险")
JUDGMENTS = {"有风险": "T", "未发现风险": "F", "信息不足": "U"}
LABOR_LAW_URL = "https://example.invalid/source-withheld"
CONTRACT_LAW_URL = "https://example.invalid/source-withheld"


def text(value):
    return "" if value is None else str(value).strip()


def normalized_text(value):
    """Normalize Unicode and whitespace, preserving numbers and punctuation."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text(value)))


def domain_hint(title):
    if "服务协议" in title:
        return "service_training_to_verify"
    if "劳务合同" in title:
        return "labor_service_to_verify"
    if "解除劳动合同" in title:
        return "labor_termination_excerpt"
    return "labor_contract_excerpt"


def citation_flags(reason):
    """Narrow checks for verified law-title/quoted-content mismatches only."""
    reason = normalized_text(reason)
    findings = []
    for article, markers in (("第四十四条", ("百分之一百五十", "延长工作时间")),
                             ("第三十九条", ("企业因生产特点",))):
        if (re.search(r"《(?:中华人民共和国)?劳动合同法》" + article, reason)
                and any(marker in reason for marker in markers)):
            findings.append({"code": "law_title_quote_mismatch", "article": article,
                             "check": f"所引内容属于《劳动法》{article}，不是《劳动合同法》同号条文；仍需另核该条文与本条款的适用关系。",
                             "verified_on": "2026-09-25",
                             "sources": [LABOR_LAW_URL, CONTRACT_LAW_URL]})
    return findings


class _Union:
    def __init__(self, values):
        self.parent = {v: v for v in values}

    def find(self, value):
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parent[max(a, b)] = min(a, b)


def known_text_index(paths):
    """Index observable text only; supports er1, legacy canonical and ml7 JSON."""
    index = defaultdict(list)
    counts = {}
    for path in paths:
        path = Path(path)
        content = path.read_text(encoding="utf-8-sig")
        data = ([json.loads(line) for line in content.splitlines() if line.strip()]
                if path.suffix == ".jsonl" else json.loads(content))
        rows = data if isinstance(data, list) else [r for key in ("risk_graph", "consensus") for r in data.get(key, [])]
        counts[str(path)] = len(rows)
        for row in rows:
            inp, meta = row.get("input", row), row.get("meta", row)
            parts = [(e.get("id", "evidence"), e.get("text")) for e in inp.get("evidence", [])]
            parts.extend((key, inp.get(key)) for key in ("clause_a", "clause_b", "fact_context"))
            for field, value in parts:
                key = normalized_text(value)
                if key:
                    index[key].append({"dataset": str(path), "sample_id": meta.get("sample_id", meta.get("case_id")),
                                       "split": meta.get("split"), "group_id": meta.get("group_id"), "field": field})
    return dict(index), counts


def import_excerpt_rows(raw, *, known_index=None, review_attestation=None, near_threshold=.90):
    """Return original annotations, canonical candidates and an auditable queue.

    A blank judgment is NOT a negative, an unknown, or an implicit positive.
    Even a user-attested single review is not the existing double-review gold.
    Near matches only form conservative split guards, never label inheritance.
    """
    known_index = known_index or {}
    sheet = raw["sheets"]["合并总表"]
    header = next((r["values"] for r in sheet if r["excel_row"] == 5), None)
    if header != HEADERS:
        raise ValueError("Expected the 14-column merged-excerpt schema at row 5")
    candidates, annotations, used = [], [], set()
    for row in sheet:
        if row["excel_row"] < 6:
            continue
        values = row["values"]
        if len(values) != len(HEADERS):
            raise ValueError(f"Wrong width at Excel row {row['excel_row']}")
        fields = dict(zip(HEADERS, values))
        sid, clause = text(values[10]), text(values[2])
        if not sid or sid in used or not clause:
            raise ValueError(f"Missing/duplicate ID or empty clause at row {row['excel_row']}")
        used.add(sid)
        judgment, grade = text(values[4]), text(values[5])
        if (judgment and judgment not in JUDGMENTS) or (grade and grade not in GRADES):
            raise ValueError(f"Unsupported judgment/grade at {sid}; do not guess")
        key = normalized_text(clause)
        root = "EXCERPT-" + digest(key)[:20]
        contract = "CONTRACT-" + digest([text(values[11]), text(values[0])])[:20]
        findings = citation_flags(text(values[6]))
        flags = []
        for flag, missing in (("missing_judgment", not judgment), ("missing_grade", not grade),
                              ("reviewer_identity_missing", not text(values[7]))):
            if missing:
                flags.append(flag)
        if text(values[3]):
            flags.append("context_excluded_as_unverified_annotation")
        domain = domain_hint(text(values[0]))
        if domain.endswith("to_verify"):
            flags.append("domain_needs_confirmation")
        if findings:
            flags.append("law_title_quote_mismatch")
        overlap = known_index.get(key, [])
        if overlap:
            flags.append("known_text_overlap")
        inp = {"proposition": PROPOSITION, "rule": "q", "task_mode": "text_only", "conditions": [{"id": "q", "text": PROPOSITION}],
               "sources": [{"id": "contract"}], "evidence": [{"id": "clause_a", "text": clause, "supports": [["contract"]]}],
               "observations": []}
        record = validate_record({"input": inp,
            "targets": {"label": JUDGMENTS.get(judgment), "severity": GRADES.get(grade), "conditions": {}, "evidence": {}},
            "meta": {"sample_id": sid, "root_id": root, "group_id": contract, "source_contract_id": contract,
                     "split": "external", "label_status": "weak_pending" if judgment or grade else "unlabeled",
                     "source_sha256": raw["sha256"], "excel_row": row["excel_row"], "domain_hint": domain,
                     "reviewer_ids": [text(values[7])] if text(values[7]) else [],
                     "gold_eligible": False, "train_eligible": False, "evaluation_only": True,
                     "proposition_origin": "fixed_proxy_not_annotated", "full_contract_available": False,
                     "quality_flags": flags}})
        candidates.append(record)
        annotations.append({"sample_id": sid, "excel_row": row["excel_row"], "fields": fields,
                            "law_checks": findings, "known_text_matches": overlap})

    by_root = defaultdict(list)
    for r in candidates:
        by_root[r["meta"]["root_id"]].append(r)
    contracts = {r["meta"]["source_contract_id"] for r in candidates}
    union = _Union(contracts)
    duplicates = []
    for root, part in sorted(by_root.items()):
        if len(part) < 2:
            continue
        grades = {r["targets"]["severity"] for r in part} - {None}
        judgments = {r["targets"]["label"] for r in part} - {None}
        conflict = len(grades) > 1
        for r in part:
            union.join(part[0]["meta"]["source_contract_id"], r["meta"]["source_contract_id"])
            r["meta"]["quality_flags"].append("exact_text_duplicate")
            if conflict:
                r["meta"]["quality_flags"].append("grade_conflict_same_text")
            if len(judgments) > 1:
                r["meta"]["quality_flags"].append("judgment_conflict_same_text")
        duplicates.append({"root_id": root, "sample_ids": [r["meta"]["sample_id"] for r in part],
                           "grades": [r["targets"]["severity"] for r in part], "grade_conflict": conflict,
                           "judgment_conflict": len(judgments) > 1,
                           "clause": part[0]["input"]["evidence"][0]["text"]})
    exact_components = len({union.find(c) for c in contracts})
    # Textually close clauses can differ in a decisive amount, negation or date.
    # Keep all rows, flag candidates, and conservatively prevent split leakage.
    unique = list(by_root.values())
    near = []
    for i, a in enumerate(unique):
        ta = normalized_text(a[0]["input"]["evidence"][0]["text"])
        for b in unique[i + 1:]:
            tb = normalized_text(b[0]["input"]["evidence"][0]["text"])
            if min(len(ta), len(tb)) < 20 or min(len(ta), len(tb)) / max(len(ta), len(tb)) < .8:
                continue
            matcher = SequenceMatcher(None, ta, tb, autojunk=False)
            if matcher.quick_ratio() < near_threshold:
                continue
            ratio = matcher.ratio()
            if ratio >= near_threshold:
                near.append({"left_ids": [r["meta"]["sample_id"] for r in a],
                             "right_ids": [r["meta"]["sample_id"] for r in b], "similarity": ratio,
                             "numeric_tokens_differ": re.findall(r"\d+(?:\.\d+)?", ta) != re.findall(r"\d+(?:\.\d+)?", tb),
                             "not_a_semantic_equivalence_decision": True})
                union.join(a[0]["meta"]["source_contract_id"], b[0]["meta"]["source_contract_id"])
                for r in a + b:
                    if "near_duplicate_candidate" not in r["meta"]["quality_flags"]:
                        r["meta"]["quality_flags"].append("near_duplicate_candidate")
    members = defaultdict(list)
    for c in sorted(contracts):
        members[union.find(c)].append(c)
    group_ids = {c: "GUARD-" + digest(group)[:20] for group in members.values() for c in group}
    for r in candidates:
        r["meta"]["group_id"] = group_ids[r["meta"]["source_contract_id"]]

    audit = {"records": len(candidates), "distinct_observable_clauses": len(by_root),
             "source_named_contracts": len(contracts), "verified_independent_contracts": None,
             "exact_duplicate_groups": len(duplicates), "exact_duplicate_rows": sum(len(g["sample_ids"]) for g in duplicates),
             "grade_conflict_groups": sum(g["grade_conflict"] for g in duplicates),
             "exact_duplicate_guard_groups": exact_components, "near_duplicate_pairs": len(near),
             "conservative_split_guard_groups": len(members),
             "source_counts": dict(Counter(text(a["fields"]["来源文件"]) for a in annotations)),
             "grade_counts": dict(Counter(GRADE_NAMES[r["targets"]["severity"]] if r["targets"]["severity"] is not None else "未填写" for r in candidates)),
             "judgment_counts": dict(Counter(r["targets"]["label"] or "unlabeled" for r in candidates)),
             "domain_counts": dict(Counter(r["meta"]["domain_hint"] for r in candidates)),
             "flag_counts": dict(Counter(f for r in candidates for f in r["meta"]["quality_flags"])),
             "gold_records": 0, "training_records": 0, "review_attestation": review_attestation or {},
             "annotation_fields_excluded_from_features": HEADERS[:2] + HEADERS[3:],
             "duplicate_rule": "NFKC plus whitespace removal; punctuation, numbers and negation preserved",
             "near_duplicate_rule": {"threshold": near_threshold, "method": "SequenceMatcher; candidate only"},
             "independence_limit": "Source names and conservative split guards are not verified independent contracts."}
    return {"records": candidates, "annotations": annotations, "audit": audit, "duplicates": duplicates, "near_duplicates": near}


def diagnostic_indices(records, target, *, labor_only=True, exclude_quality=True):
    """Predeclared representative selection; never uses model predictions."""
    bad = {"known_text_overlap", "grade_conflict_same_text", "judgment_conflict_same_text", "law_title_quote_mismatch"}
    selected, roots = [], set()
    for i, r in enumerate(records):
        if target == "severity" and r["meta"].get("severity_origin") == "ai_supplement":
            continue
        if r["targets"].get(target) is None:
            continue
        if labor_only and r["meta"]["domain_hint"].endswith("to_verify"):
            continue
        if exclude_quality and bad.intersection(r["meta"]["quality_flags"]):
            continue
        root = r["meta"]["root_id"]
        if root not in roots:
            selected.append(i)
            roots.add(root)
    return selected


def supplement_missing_grades(records, suggestions):
    """Apply authorized AI proposals to a copy; never overwrite an original grade."""
    result = copy.deepcopy(records)
    by_id = {r["meta"]["sample_id"]: r for r in result}
    seen = set()
    for proposal in suggestions:
        sid, grade = proposal["sample_id"], proposal["severity"]
        if sid not in by_id or sid in seen or type(grade) is not int or grade not in range(4):
            raise ValueError("Invalid or duplicate AI grade proposal")
        seen.add(sid)
        r = by_id[sid]
        if r["targets"].get("severity") is not None:
            raise ValueError(f"Cannot overwrite original severity: {sid}")
        if not proposal.get("reason") or not proposal.get("sources"):
            raise ValueError("A proposed grade needs a reason and traceable sources")
        r["targets"]["severity"] = grade
        r["meta"].update(severity_origin="ai_supplement", label_status="weak_pending", gold_eligible=False,
                         train_eligible=False, evaluation_only=True, severity_suggestion_sha256=digest(proposal))
        r["meta"]["quality_flags"] = [f for f in r["meta"]["quality_flags"] if f != "missing_grade"]
        r["meta"]["quality_flags"].append("ai_grade_supplement")
        validate_record(r)
    return result
