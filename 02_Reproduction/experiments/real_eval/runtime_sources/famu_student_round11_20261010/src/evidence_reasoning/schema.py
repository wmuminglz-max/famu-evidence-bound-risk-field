"""Validated input boundary; annotation fields never become model features."""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping
from src.risk_grading import PROPOSITION_TYPES

INPUT_FIELDS = {"proposition", "rule", "constraints", "conditions", "sources", "evidence",
                "observations", "withdrawn_sources", "rule_status", "technical_status", "task_mode",
                "proposition_type", "review_perspective", "legal_context", "structure_origin", "candidate_groups"}
NODE_LIMIT = 256
CONDITION_LIMIT = 14


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def expression_variables(expr: Any, depth: int = 0) -> set[str]:
    if depth > 32:
        raise ValueError("Rule nesting exceeds 32")
    if isinstance(expr, bool):
        return set()
    if isinstance(expr, str):
        if not expr:
            raise ValueError("Empty condition identifier")
        return {expr}
    if not isinstance(expr, dict) or len(expr) != 1:
        raise ValueError("Rule must be a boolean, condition id, or one of all/any/not")
    op, args = next(iter(expr.items()))
    if op == "not":
        return expression_variables(args, depth + 1)
    if op not in {"all", "any"} or not isinstance(args, list) or not args or len(args) > NODE_LIMIT:
        raise ValueError("all/any requires a nonempty list")
    return set().union(*(expression_variables(a, depth + 1) for a in args))


def normalize_expression(expr: Any) -> Any:
    """Canonicalize supported Boolean syntax without changing open-world semantics.

    Validation precedes reduction, including on branches later removed by constants.
    This is a syntactic normal form, not an unrestricted Boolean theorem prover.
    """
    expression_variables(expr)
    def visit(value):
        if isinstance(value, (str, bool)):
            return value
        op, children = next(iter(value.items()))
        if op == "not":
            child = visit(children)
            if isinstance(child, bool):
                return not child
            if isinstance(child, dict) and "not" in child:
                return child["not"]
            return {"not": child}
        identity = op == "all"
        items = []
        for child in children:
            child = visit(child)
            if isinstance(child, bool):
                if child != identity:
                    return child
                continue
            items.extend(child[op] if isinstance(child, dict) and op in child else [child])
        unique = {json.dumps(c, sort_keys=True, ensure_ascii=False): c for c in items}
        items = [unique[k] for k in sorted(unique)]
        return identity if not items else items[0] if len(items) == 1 else {op: items}
    return visit(expr)


def _ids(items: list, name: str) -> set[str]:
    if not isinstance(items, list) or len(items) > NODE_LIMIT:
        raise ValueError(f"Invalid {name} list (limit {NODE_LIMIT})")
    ids = [x.get("id") for x in items if isinstance(x, dict)]
    if len(ids) != len(items) or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError(f"Every {name} needs a string id")
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate {name} id")
    return set(ids)


def _groups(value: Any, allowed: set[str], field: str) -> None:
    if not isinstance(value, list) or not value or len(value) > 128:
        raise ValueError(f"{field} requires nonempty alternative support groups")
    for group in value:
        if not isinstance(group, list) or not group or len(group) > NODE_LIMIT:
            raise ValueError(f"{field}: each AND group must be nonempty")
        if any(not isinstance(x, str) or x not in allowed for x in group):
            raise ValueError(f"{field}: unknown reference")


def validate_input(payload: Mapping) -> dict:
    if not isinstance(payload, Mapping):
        raise ValueError("Input must be an object")
    unknown = set(payload) - INPUT_FIELDS
    if unknown:
        raise ValueError(f"Fields outside input whitelist: {sorted(unknown)}")
    x = copy.deepcopy(dict(payload))
    if "proposition_type" in x and x["proposition_type"] not in PROPOSITION_TYPES:
        raise ValueError("Invalid proposition_type")
    if "structure_origin" in x and x["structure_origin"] not in {"given_rule", "predicted_rule"}:
        raise ValueError("Invalid structure_origin")
    if 'review_perspective' in x and not isinstance(x['review_perspective'], str):
        raise ValueError('Review perspective must be observable text')
    if "legal_context" in x:
        context = x["legal_context"]
        if not isinstance(context, dict) or set(context) - {"jurisdiction", "signed_at", "performed_at", "reviewed_at", "rule_version", "rule_effective_at"}:
            raise ValueError("Legal context may contain only deployment-time jurisdiction, dates and rule version")
        if any(not isinstance(v, (str, type(None))) for v in context.values()):
            raise ValueError("Invalid legal context value")
    if not isinstance(x.get("proposition"), str) or not x["proposition"].strip():
        raise ValueError("A precise proposition is required")
    cids = _ids(x.setdefault("conditions", []), "condition")
    if len(cids) > CONDITION_LIMIT:
        raise ValueError(f"Exact reference executor supports at most {CONDITION_LIMIT} conditions")
    for condition in x["conditions"]:
        if not isinstance(condition.get("text"), str) or not condition["text"].strip():
            raise ValueError("Conditions require query text")
        if set(condition) - {"id", "text", "scope_id", "statement_kind"}:
            raise ValueError("Condition annotations must be stored outside input")
        if 'scope_id' in condition and (not isinstance(condition['scope_id'], str) or not condition['scope_id']):
            raise ValueError('Condition scope must be a nonempty identifier')
        if 'statement_kind' in condition and condition['statement_kind'] not in {'normative_obligation', 'observed_fact', 'legal_applicability'}:
            raise ValueError('Invalid condition statement kind')
    if "rule" not in x:
        raise ValueError("Explicit rule is required")
    constraints = x.setdefault("constraints", [])
    if not isinstance(constraints, list) or len(constraints) > 64:
        raise ValueError("Invalid constraints")
    variables = expression_variables(x["rule"])
    for expression in constraints:
        variables |= expression_variables(expression)
    def expression_size(expr):
        if isinstance(expr, (str, bool)):
            return 1
        op, value = next(iter(expr.items()))
        return 1 + sum(expression_size(c) for c in ([value] if op == "not" else value))
    if expression_size(x["rule"]) + sum(expression_size(c) for c in constraints) > NODE_LIMIT:
        raise ValueError("Rule expression node budget exceeded")
    if not variables <= cids:
        raise ValueError("Rule references undefined conditions")
    x["rule"] = normalize_expression(x["rule"])
    normalized_constraints = {digest(normalize_expression(c)): normalize_expression(c) for c in constraints}
    x["constraints"] = [normalized_constraints[k] for k in sorted(normalized_constraints)]
    x["conditions"].sort(key=lambda c: c["id"])
    if x.setdefault("task_mode", "structured") not in {"structured", "text_only"}:
        raise ValueError("task_mode must be structured or text_only")
    sids = _ids(x.setdefault("sources", []), "source")
    eids = _ids(x.setdefault("evidence", []), "evidence")
    if len(sids) + len(eids) + len(cids) > NODE_LIMIT:
        raise ValueError("Input node budget exceeded")
    for source in x["sources"]:
        if set(source) - {"id", "available", "name", "sha256", "authenticity", "admissibility", "independence", "independence_basis"}:
            raise ValueError("Unsupported source field")
        if "available" in source and not isinstance(source["available"], bool):
            raise ValueError("Source availability must be boolean")
        for field in ("authenticity", "admissibility", "independence"):
            if field in source and source[field] not in {"unknown", "confirmed", "rejected"}:
                raise ValueError("Trust dimensions need separate explicit states")
        if source.get("independence") == "confirmed" and not source.get("independence_basis"):
            raise ValueError("Source independence requires its substantive basis")
    for evidence in x["evidence"]:
        if set(evidence) - {"id", "text", "supports", "parents", "start", "end", "page", "scope_id", "scope_kind", "equivalence"}:
            raise ValueError("Unsupported evidence field")
        if not isinstance(evidence.get("text"), str) or len(evidence["text"]) > 100_000:
            raise ValueError("Invalid evidence text")
        if 'scope_id' in evidence and (not isinstance(evidence['scope_id'], str) or not evidence['scope_id']):
            raise ValueError('Evidence scope must be a nonempty identifier')
        if 'scope_kind' in evidence and evidence['scope_kind'] not in {'clause', 'exception', 'fact'}:
            raise ValueError('Invalid evidence scope kind')
        if ("supports" in evidence) == ("parents" in evidence):
            raise ValueError("Evidence needs exactly one of source supports or parent groups")
        if "supports" in evidence:
            _groups(evidence["supports"], sids, "supports")
        else:
            _groups(evidence["parents"], eids, "parents")
        eq = evidence.get("equivalence")
        if eq is not None and (not isinstance(eq, dict) or set(eq) != {"assertion_id", "canonical_text", "verification_event"}
                               or any(not isinstance(v, str) or not v.strip() for v in eq.values())):
            raise ValueError("Equivalence requires assertion identity, canonical text and verification event")
    candidates = x.get("candidate_groups", [])
    if not isinstance(candidates, list) or len(candidates) > NODE_LIMIT:
        raise ValueError("Invalid candidate proof groups")
    candidate_ids = set()
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) - {"id", "condition", "evidence_ids", "origin"}:
            raise ValueError("Invalid candidate group")
        if not candidate.get("id") or candidate["id"] in candidate_ids or candidate.get("condition") not in cids:
            raise ValueError("Candidate group needs unique id and defined condition")
        candidate_ids.add(candidate["id"])
        _groups([candidate.get("evidence_ids")], eids, "candidate_groups")
        if candidate.get("origin") not in {"given", "predicted"}:
            raise ValueError("Candidate groups must identify given/predicted structure")
    removed = x.setdefault("withdrawn_sources", [])
    if not isinstance(removed, list) or any(s not in sids for s in removed):
        raise ValueError("Unknown withdrawn source")
    observations = x.setdefault("observations", [])
    if not isinstance(observations, list) or len(observations) > NODE_LIMIT:
        raise ValueError("Invalid observation list")
    for claim in observations:
        if not isinstance(claim, dict) or set(claim) - {"condition", "value", "evidence_groups", "origin"}:
            raise ValueError("Invalid observation schema")
        if claim.get("condition") not in cids or (claim.get("value") is not None and type(claim["value"]) is not bool):
            raise ValueError("Observation needs a defined condition and boolean/null value")
        _groups(claim.get("evidence_groups"), eids, "evidence_groups")
    if x.get("rule_status", "resolved") not in {"resolved", "unresolved"}:
        raise ValueError("Invalid rule_status")
    if x.get("technical_status", "ok") not in {"ok", "failed"}:
        raise ValueError("Invalid technical_status")
    return x


def decision_route(payload: Mapping) -> dict:
    """Choose authority from the declared task contract, never JSON shape/count.

    structured: caller supplies the supported rule and all referenced conditions;
    absent evidence is allowed and may produce U. text_only: the rule is a proxy
    for an unformalized text task and cannot authorize an executor conclusion.
    """
    x = validate_input(payload)
    ready = x.get("rule_status", "resolved") == "resolved" and x.get("technical_status", "ok") == "ok"
    eligible = ready and x["task_mode"] == "structured"
    return {"version": "semantic-route-v1", "task_mode": x["task_mode"],
            "executor_eligible": eligible,
            "result_origin": "condition_executor" if eligible else "neural_candidate" if ready else "review",
            "reason": "validated_explicit_rule_and_conditions" if eligible else "declared_text_only_task" if ready else "unresolved_or_failed_input",
            "normalized_rule": x["rule"], "rule_sha256": digest(x["rule"])}


def model_input(payload: Mapping) -> dict:
    """Do not feed provided/gold observations, source ids or labels as features."""
    x = validate_input(payload)
    x["observations"] = []
    return x


def validate_record(record: Mapping) -> dict:
    if not isinstance(record, Mapping) or set(record) - {"input", "targets", "meta"}:
        raise ValueError("Record requires input, targets and meta namespaces")
    x = copy.deepcopy(dict(record))
    x["input"] = validate_input(x["input"])
    meta = x.setdefault("meta", {})
    for key in ("sample_id", "root_id", "group_id", "split", "label_status"):
        if not isinstance(meta.get(key), str) or not meta[key]:
            raise ValueError(f"Missing metadata {key}")
    if meta["split"] not in {"train", "validation", "calibration", "test", "regression", "external"}:
        raise ValueError("Invalid split")
    if meta["label_status"] not in {"synthetic_verified", "weak_pending", "human_adjudicated", "human_confirmed", "unlabeled"}:
        raise ValueError("Invalid label status")
    targets = x.setdefault("targets", {})
    if not isinstance(targets, dict) or set(targets) - {"label", "conditions", "evidence", "severity", "related", "spans", "proof_groups", "scope", "risk_established_status", "severity_basis"}:
        raise ValueError("Unsupported target schema")
    if targets.get("label") not in {None, "T", "F", "U"}:
        raise ValueError("Invalid proposition target")
    conditions = targets.get("conditions", {})
    if not isinstance(conditions,dict) or not set(conditions) <= {c["id"] for c in x["input"]["conditions"]} or not set(conditions.values()) <= {"T","F","U"}:
        raise ValueError("Invalid condition targets")
    evidence = targets.get("evidence", {})
    pair_ids = {f"{c['id']}|{e['id']}" for c in x["input"]["conditions"] for e in x["input"]["evidence"]}
    if not isinstance(evidence,dict) or not set(evidence) <= pair_ids or not set(evidence.values()) <= {"irrelevant","support","refute"}:
        raise ValueError("Invalid evidence targets")
    from .spans import validate_span_annotations
    validate_span_annotations(x["input"], targets.get("spans", []))
    proof = targets.get("proof_groups", {})
    if (not isinstance(proof, dict) or not set(proof) <= {p["id"] for p in x["input"].get("candidate_groups", [])}
            or not set(proof.values()) <= {"irrelevant", "support", "refute"}):
        raise ValueError("Invalid joint proof targets")
    severity = targets.get("severity")
    if severity is not None and (type(severity) is not int or severity not in range(4)):
        raise ValueError("Invalid ordinal severity target")
    if meta.get('protocol_version') == 'expert-20260929-v2':
        from src.risk_grading import risk_result, LEVELS
        risk_result(targets.get('risk_established_status', 'unlabeled'), LEVELS[severity] if severity is not None else None, targets.get('severity_basis'))
    if targets.get("related") is not None and type(targets["related"]) is not bool:
        raise ValueError("Related target must be boolean or null")
    if meta["label_status"] == "human_adjudicated" and (len(set(meta.get("reviewer_ids", []))) < 2 or not meta.get("annotation_sha256")):
        raise ValueError("Human-reviewed labels need double-review provenance")
    if meta["label_status"] == "human_confirmed" and (not isinstance(meta.get("reviewer_ids"),list) or not meta.get("reviewer_ids") or
            any(not isinstance(v,str) or not v.strip() for v in meta["reviewer_ids"]) or
            not meta.get("annotation_sha256") or not meta.get("confirmed_at")):
        raise ValueError("Single-review confirmation requires actual reviewer/time and annotation provenance")
    if meta["label_status"] == "human_confirmed" and (meta.get("human_gold") or meta.get("independent_double_review")):
        raise ValueError("Single confirmation cannot assert independently double-reviewed gold")
    return x
