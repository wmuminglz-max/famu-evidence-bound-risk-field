"""48-case legal diagnostic intake and evaluation, without invented human labels."""
from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
import numpy as np

from .annotations import import_annotations
from .schema import validate_input, model_input, digest
from .data import write_json, audit_splits
from .logic import solve, evaluate as evaluate_rule
from .metrics import classification, group_bootstrap_difference

CATEGORIES = ("完整材料成立与不成立", "必要材料缺失", "例外否定与作用范围", "独立替代支持", "同源复制摘要派生", "主体时间金额跨条款变化")
ARMS = ("human_rule_human_conditions_executor", "human_rule_predicted_conditions_executor",
        "human_conditions_learned_composition", "specific_proposition_end_to_end", "broad_risk_proposition_end_to_end")


def create_packet(output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Diagnostic packets must use a new empty directory")
    entries = []
    for category in CATEGORIES:
        for i in range(8):
            entries.append({"slot_id": f"D48-{len(entries)+1:02d}", "category": category,
                'protocol_version': 'expert-20260929-v2', 'proposition_type': None,
                "status": "awaiting_real_materials_and_independent_review", "root_id": None, "group_id": None,
                "materials": {"complete_context_checked": False, "contract_id": None, "original_source_id": None,
                              "template_family_id": None, "rule_version": None, "fact_background": None,
                              'jurisdiction': None, 'signed_at': None, 'performed_at': None, 'reviewed_at': None,
                              'missing_inventory': [], 'inspection_scope': None,
                              "documents": []},
                "input": None, "independent_annotation_1": None, "independent_annotation_2": None, "adjudication": None})
    write_json(output / "case_intake.json", entries)
    # Separate blank reviewer forms: neither reviewer sees existing answers/model output.
    for reviewer in (1, 2):
        write_json(output / f"reviewer_{reviewer}.json", [{"slot_id": e["slot_id"], "category": e["category"],
            "annotator_id": None, "blind_to_model_and_other_reviewer": None, "label": None,
            'proposition_type': None, 'event_at': None, 'qualification_scope': None, 'rationale': None,
            'spans': [], 'proof_groups': {}, 'scope': {}, 'source_relations': {}, 'rule_assessment': None,
            'risk_established_status': 'unlabeled',
            "conditions": {}, "evidence": {}, "condition_context": {}, "severity": None,
            "severity_basis": {"risk_object": None, "risk_domain": None, "immediate": None, 'immediacy_basis': None,
                "consequence": None, "trigger": None, "magnitude": None, "remediability": None, "unconfirmed_facts": []},
            "related": None} for e in entries])
    write_json(output / "protocol.json", {"status": "pending_materials", "planned_scenarios": 48,
        'minimum_target_source_groups': 24, 'proposition_types': ['contract_content', 'normative_sufficiency', 'case_application'],
        'include_sufficiency_application_pairs': True, 'paired_questions_are_not_independent_sources': True,
        "actual_independent_roots": 0, "categories": dict(Counter(e["category"] for e in entries)), "arms": list(ARMS),
        "role": "development_diagnostic_not_final_blind_test", "primary_metrics": ["root_balanced_macro_f1", "U_recall", "false_certainty", "boundary_both_correct"],
        "bootstrap_iterations": 2000, "split_unit": "original_source_contract_template_family_with_all_derivatives",
        "severity": "report conditional and full-pipeline outcomes separately",
        "stop_conditions": ["repair task/formalization if human-rule executor disagrees with human final label",
                            "repair extraction if predicted conditions lose performance",
                            "repair composition if supplied human conditions still fail",
                            "do not claim graph innovation if binding-preserving baseline explains gains"],
        "pending_external_work": "complete contract context and referenced documents; two independent blind annotations and arbitration"})
    return {"slots": 48, "actual_cases": 0, "output": str(output.resolve())}


def _validate_entry(entry):
    if entry.get("category") not in CATEGORIES or not entry.get("root_id") or not entry.get("group_id"):
        raise ValueError("A category and verified root/source group are required")
    materials = entry.get("materials") or {}
    if materials.get("complete_context_checked") is not True:
        raise ValueError("Complete context and unavailable-material inventory must be checked")
    for key in ("contract_id", "original_source_id", "template_family_id", "rule_version", "fact_background"):
        if not isinstance(materials.get(key), str) or not materials[key].strip():
            raise ValueError(f"Missing materials.{key}; explicitly record not_applicable where appropriate")
    documents = materials.get("documents", [])
    if not documents:
        raise ValueError("Original material inventory is required")
    for doc in documents:
        if not doc.get("source_id") or not doc.get("location") or not isinstance(doc.get("available"), bool):
            raise ValueError("Every document needs source_id, location and actual availability")
        checksum = doc.get("sha256", "")
        if doc["available"] and (len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum.lower())):
            raise ValueError("Available originals require a SHA-256 identity")
    x = validate_input(entry["input"])
    if entry.get('protocol_version') == 'expert-20260929-v2':
        from src.risk_grading import PROPOSITION_TYPES, RISK_STATES, LEVELS, validate_severity_basis
        if x.get('proposition_type') not in PROPOSITION_TYPES or x.get('proposition_type') != entry.get('proposition_type'):
            raise ValueError('Confirm the same proposition type in question and annotation protocol')
        for key in ('jurisdiction', 'signed_at', 'performed_at', 'reviewed_at', 'inspection_scope'):
            if not materials.get(key):
                raise ValueError('Missing legal time/context: '+key)
        for annotation in (entry.get('independent_annotation_1'), entry.get('independent_annotation_2')):
            if not isinstance(annotation, dict):
                raise ValueError('Independent annotation required')
            for key in ('event_at', 'qualification_scope', 'rationale', 'rule_assessment'):
                if not annotation.get(key):
                    raise ValueError('Missing independent review event field: '+key)
            if annotation.get('proposition_type') != x['proposition_type']:
                raise ValueError('Proposition type changed during annotation')
            if annotation.get('risk_established_status') not in RISK_STATES:
                raise ValueError('Risk status must be explicit')
            if annotation.get('severity') is not None:
                if annotation['risk_established_status'] != 'established':
                    raise ValueError('Severity is conditional on a specified established risk')
                validate_severity_basis(LEVELS[annotation['severity']], annotation.get('severity_basis'))
    if x["task_mode"] != "structured":
        raise ValueError("Diagnostic cases need a specific formalized proposition")
    if {d["source_id"] for d in documents} != {s["id"] for s in x["sources"]}:
        raise ValueError("Material inventory must match all input sources")
    source_availability = {s["id"]: s.get("available", True) for s in x["sources"]}
    if any(source_availability[d["source_id"]] != d["available"] for d in documents):
        raise ValueError("Material availability disagrees with input")
    evidence = {e["id"]: e for e in x["evidence"]}
    for annotation in (entry.get("independent_annotation_1"), entry.get("independent_annotation_2")):
        if not isinstance(annotation, dict) or annotation.get("blind_to_model_and_other_reviewer") is not True:
            raise ValueError("Two blind independent reviews must be explicitly attested")
        context = annotation.get("condition_context", {})
        if set(context) != {c["id"] for c in x["conditions"]}:
            raise ValueError("Each condition needs subject/object/time/statement-kind/span annotations")
        for cid, atom in context.items():
            if any(not atom.get(k) for k in ("subject", "object", "time_scope")) or atom.get("statement_kind") not in {"normative_obligation", "observed_fact", "legal_applicability"}:
                raise ValueError("Do not conflate normative obligations with observed facts")
            spans = atom.get("spans", [])
            if not spans and not atom.get("missing_material"):
                raise ValueError("Provide observable support/refutation spans or identify missing material")
            for span in spans:
                e = evidence.get(span.get("evidence_id"))
                start, end = span.get("start"), span.get("end")
                if not e or type(start) is not int or type(end) is not int or not 0 <= start < end <= len(e["text"]):
                    raise ValueError("Invalid original evidence span")
                if span.get("text") != e["text"][start:end] or span.get("stance") not in {"support", "refute"}:
                    raise ValueError("Span must quote its original text and annotate stance")
        if annotation.get("severity") is not None:
            basis = annotation.get("severity_basis") or {}
            if any(not basis.get(k) for k in ("consequence", "trigger", "magnitude", "remediability")) or "unconfirmed_facts" not in basis:
                raise ValueError("Conditional severity rationale is incomplete")
    return x


def intake_status(entries):
    details = []
    for entry in entries:
        try:
            _validate_entry(entry)
            import_annotations([entry], split="regression")
            details.append({"slot_id": entry["slot_id"], "ready": True})
        except (ValueError, KeyError, TypeError) as exc:
            details.append({"slot_id": entry.get("slot_id"), "ready": False, "reason": str(exc)})
    return {"planned": len(entries), "ready": sum(d["ready"] for d in details), "details": details}


def import_diagnostic(entries, known_records=()):
    for entry in entries:
        _validate_entry(entry)
    rows = import_annotations(entries, split="regression", known_records=known_records)
    for row, entry in zip(rows, entries):
        materials = entry["materials"]
        row["meta"].update({key: materials[key] for key in ("contract_id", "original_source_id", "template_family_id")})
        row["meta"].update(category=entry["category"], domain="legal_development_diagnostic",
            source_document_sha256=[d["sha256"] for d in materials["documents"] if d["available"]],
            diagnostic_slot=entry["slot_id"], materials=copy.deepcopy(materials), independent_annotations=copy.deepcopy([
                entry["independent_annotation_1"], entry["independent_annotation_2"]]), adjudication=copy.deepcopy(entry.get("adjudication")))
    # Distinct roots from a shared original/template must have a common source group.
    memberships = {}
    for row in rows:
        meta = row["meta"]
        for kind in ("contract_id", "original_source_id", "template_family_id"):
            if meta[kind] == "not_applicable":
                continue
            key = (kind, meta[kind])
            if key in memberships and memberships[key] != meta["group_id"]:
                raise ValueError("Related originals/templates must share one source group")
            memberships[key] = meta["group_id"]
    audit_splits(rows)
    return rows


def from_condition_states(payload, states):
    """Diagnostic oracle: supplied condition labels, never injected into learned inputs."""
    import itertools
    x = model_input(payload)
    ids = [c["id"] for c in x["conditions"]]
    if set(states) != set(ids) or not set(states.values()) <= {"T", "F", "U"}:
        raise ValueError("Every condition requires an explicit T/F/U state")
    fixed = {c: value == "T" for c, value in states.items() if value != "U"}
    unknown = [c for c in ids if states[c] == "U"]
    values = set()
    for assignment in itertools.product((False, True), repeat=len(unknown)):
        world = fixed | dict(zip(unknown, assignment))
        if all(evaluate_rule(c, world) for c in x["constraints"]):
            values.add(evaluate_rule(x["rule"], world))
    return "U" if len(values) == 2 else "T" if values == {True} else "F" if values == {False} else None


def evaluate_diagnostic(rows, predictions=None):
    """Score all arms against the same adjudicated proposition, with missingness visible.

    Predictions are keyed by sample_id. Every external arm must attest the exact
    source input fingerprint and disjoint fitting/selection groups. Broad-prompt
    outputs are a diagnostic perturbation, not a different gold task.
    """
    predictions = predictions or {}
    audit_splits(rows)
    if not rows or any(r["meta"].get("domain") != "legal_development_diagnostic" for r in rows):
        raise ValueError("Import independently reviewed diagnostic cases first")
    roots, groups = ([r["meta"][key] for r in rows] for key in ("root_id", "group_id"))
    y = [("T", "F", "U").index(r["targets"]["label"]) for r in rows]
    report = {"scope": "development diagnostic; human-rule oracle is not a learned model", "arms": {}, "predictions": {},
              "annotation_agreement": annotation_agreement(rows)}
    for arm in ARMS:
        answers, missing = [], []
        for row in rows:
            sid = row["meta"]["sample_id"]
            if arm == ARMS[0]:
                label = from_condition_states(row["input"], row["targets"]["conditions"])
            else:
                value = predictions.get(arm, {}).get(sid)
                if value is None:
                    missing.append(sid)
                    answers.append(-1)
                    continue
                if value.get("input_sha256") != digest(model_input(row["input"])):
                    raise ValueError("Prediction input mismatch; all arms require the same material access")
                if "fit_groups" not in value or "selection_groups" not in value or row["meta"]["group_id"] in set(value["fit_groups"]) | set(value["selection_groups"]):
                    raise ValueError("Diagnostic prediction lacks disjoint fit/selection provenance")
                if arm == ARMS[1] and row['meta'].get('protocol_version') == 'expert-20260929-v2':
                    if 'observations' not in value:
                        raise ValueError('Predicted conditions require evidence-backed observations for the exact gate')
                    extracted = model_input(row['input'])
                    extracted['observations'] = value['observations']
                    from .decision import select_decision
                    logic = solve(extracted)
                    label = select_decision(extracted, logic, uncertain_conditions=value.get('uncertain_conditions', []))['proposition_label']
                else:
                    label = from_condition_states(row["input"], value["conditions"]) if arm == ARMS[1] else value.get("label")
            answers.append(("T", "F", "U").index(label) if label in ("T", "F", "U") else -1)
        report["arms"][arm] = {"status": "pending" if len(missing) == len(rows) else "partial" if missing else "complete",
                               "missing_predictions": missing,
                               "metrics": None if len(missing) == len(rows) else classification(y, answers, roots, groups=groups)}
        report["predictions"][arm] = dict(zip([r["meta"]["sample_id"] for r in rows], answers))
    report["formalization_mismatches"] = [r["meta"]["sample_id"] for r, label in zip(rows, report["predictions"][ARMS[0]].values()) if label != ("T", "F", "U").index(r["targets"]["label"])]
    if all(report["arms"][a]["status"] == "complete" for a in ARMS[-2:]):
        report["specific_minus_broad_macro_f1"] = group_bootstrap_difference(y, list(report["predictions"][ARMS[-2]].values()),
            list(report["predictions"][ARMS[-1]].values()), roots, groups=groups, metric="macro_f1")
    return report


def annotation_agreement(rows):
    from sklearn.metrics import cohen_kappa_score
    pairs = [r["meta"]["independent_annotations"] for r in rows]
    a, b = ([pair[i]["label"] for pair in pairs] for i in (0, 1))
    kappa = float(cohen_kappa_score(a, b, labels=["T", "F", "U"])) if len(set(a + b)) > 1 else None
    levels = [(p[0]["severity"], p[1]["severity"]) for p in pairs if p[0].get("severity") is not None and p[1].get("severity") is not None]
    weighted = float(cohen_kappa_score(*zip(*levels), labels=range(4), weights="quadratic")) if levels and len(set(v for pair in levels for v in pair)) > 1 else None
    return {"raw_label_agreement": float(np.mean(np.asarray(a) == np.asarray(b))),
            "cohen_kappa": kappa if kappa is None or np.isfinite(kappa) else None,
            "severity_pairs": len(levels), "severity_quadratic_kappa": weighted if weighted is None or np.isfinite(weighted) else None,
            "scope": "before adjudication; reviewer identities attested, not independently authenticated"}
