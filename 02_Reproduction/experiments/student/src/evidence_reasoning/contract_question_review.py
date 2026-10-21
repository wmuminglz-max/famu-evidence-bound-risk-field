"""Callable question/evidence review pipeline, separate from production routing."""
from __future__ import annotations
import json
from pathlib import Path
from src.lawyer_review_dataset import file_hash, digest
from src.risk_grading import validate_severity_basis
from .contract_retrieval import ContractRetriever, prepare_features, validate_request, sha
from .question_evidence_models import QuestionEvidenceHead, VERSION, TASK_CLASSES


class ContractQuestionReviewer:
    def __init__(self, bundle):
        self.folder = Path(bundle).resolve()
        self.manifest = json.loads((self.folder / "manifest.json").read_text("utf-8"))
        if self.manifest.get("version") != VERSION or self.manifest.get("mode") != "shadow":
            raise ValueError("Unsupported question/evidence bundle")
        required = {task + ".json" for task in TASK_CLASSES}
        files = self.manifest.get("files_sha256", {})
        if not required <= set(files):
            raise ValueError("Missing classifier hash coverage")
        for name, checksum in files.items():
            path = (self.folder / name).resolve()
            if path.parent != self.folder or file_hash(path) != checksum:
                raise ValueError("Bundle file hash or path mismatch")
        self.heads = {task: QuestionEvidenceHead(task, payload=json.loads((self.folder/(task+".json")).read_text("utf-8")))
                      for task in TASK_CLASSES}
        self.retriever = ContractRetriever(**self.manifest["retrieval_parameters"])

    def review(self, payload):
        x = validate_request(payload)
        base = {"version": self.manifest["version"], "mode": "shadow", "decision_authority": False,
                "proposition_label": None, "risk_established_status": "unlabeled", "severity": None,
                "scores_calibrated": False, "automatic_decision_enabled": False}
        if x["technical_status"] == "failed":
            return {**base, "workflow_status": "technical_failure", "review_reasons": ["input_processing_failed"],
                    "risk_established_status": "technical_failure", "evidence": [], "candidates": {}}
        retrieval, features, candidates, details = self._infer(x)
        views = [retrieval] + ([details["qa_retrieval"]] if "qa_retrieval" in details else [])
        reasons = []
        if any(not view["anchor_ids"] for view in views):
            reasons.append("no_retrieval_match_is_not_proof_of_absence")
        if any(not view["full_text_scope_declared"] for view in views):
            reasons.append("supplied_material_scope_incomplete_or_unknown")
        if any(view["unresolved_references"] for view in views):
            reasons.append("unresolved_or_ambiguous_article_reference")
        if any(view["context_budget_exceeded"] for view in views):
            reasons.append("relevant_context_exceeds_budget")
        if retrieval["missing_required_source_kinds"]:
            reasons.append("explicitly_required_source_kind_missing")
        if x["proposition_type"] == "case_application" and not features["has_performance_source"]:
            reasons.append("case_application_has_only_unverified_contract_or_other_text")
        qa, risk, severity = (candidates[k] for k in ("qa", "risk_status", "severity"))
        if qa["label"] == "U":
            reasons.append("semantic_U_is_a_model_proposal_pending_evidence_review")
        if qa["label"] == "U" and risk["label"] == "established":
            reasons.append("QA_and_risk_candidates_require_reconciliation")
        # Conditional severity cannot silently turn risk uncertainty into a grade.
        conditional_severity = severity if risk["label"] == "established" and qa["label"] != "U" else None
        evidence_blocked = (any(not view["anchor_ids"] or view["missing_required_source_kinds"] or
                            view["unresolved_references"] or view["context_budget_exceeded"] for view in views) or
                            (x["proposition_type"] == "case_application" and not features["has_performance_source"]))
        if evidence_blocked:
            conditional_severity = None
            reasons.append("severity_withheld_until_required_evidence_and_context_are_available")
        if details.get("ensemble_disagreement_tasks"):
            conditional_severity = None
            reasons.append("model_disagreement_requires_review_before_grading")
        if conditional_severity is None:
            reasons.append("severity_withheld_until_risk_is_established")
        else:
            try:
                validate_severity_basis(severity["label"], x.get("severity_basis"))
                if severity["label"] == "critical":
                    from .research_review_protocol import critical_basis_check
                    audit = critical_basis_check(x, x.get("severity_basis"))
                    if audit["missing"]:
                        reasons.extend("critical_basis_missing:" + field for field in audit["missing"])
                        raise ValueError("Critical basis lacks source-bound review scope or immediacy")
            except ValueError:
                reasons.append("severity_basis_requires_review")
                if severity["label"] == "critical":
                    conditional_severity = None
                    reasons.append("critical_grade_needs_immediacy_and_major_harm_basis")
        for task, candidate in candidates.items():
            if candidate["untrained_classes"]:
                reasons.append(task + "_has_untrained_classes")
        # This bundle has no independent calibration or release evidence.
        # Raw score thresholds must never masquerade as validated automation.
        reasons.append("independent_calibration_and_release_evaluation_pending")
        action = "request_materials" if (not retrieval["visible_source_count"] or retrieval["missing_required_source_kinds"]) else "human_review"
        return {**base, "workflow_status": action, "review_reasons": sorted(set(reasons)),
                "question_answer_candidate": qa["label"], "risk_status_candidate": risk["label"],
                "severity_candidate": conditional_severity["label"] if conditional_severity else None,
                "candidates": candidates, "evidence": retrieval["evidence"], "retrieval": retrieval,
                "feature_fingerprint": digest(features),
                "semantic_U_and_workflow_abstention_are_separate": True,
                "evidence_relation_status": "retrieved_candidates_not_verified_support_or_refutation", **details}

    def _infer(self, x):
        retrieval = self.retriever.retrieve(x)
        features = prepare_features(x, retrieval)
        candidates = {task: head.predict([features])[0] for task, head in self.heads.items()}
        return retrieval, features, candidates, {}


def load_contract_question_reviewer(bundle):
    """Explicit bundle selection; no automatic switch of the production default."""
    manifest = json.loads((Path(bundle)/"manifest.json").read_text("utf-8"))
    if manifest.get("version") == VERSION:
        return ContractQuestionReviewer(bundle)
    if manifest.get("version") == "2026.09.29-contract-ensemble-v1":
        from .contract_ensemble import ContractEnsembleReviewer
        return ContractEnsembleReviewer(bundle)
    if manifest.get("version") == "2026.09.29-famu-contract-lora-v1":
        from .famu_contract_model import FamuContractReviewer
        return FamuContractReviewer(bundle)
    from .semantic_risk import VERSION as SEMANTIC_VERSION
    if manifest.get("version") == SEMANTIC_VERSION:
        from .semantic_contract_review import SemanticContractReviewer
        return SemanticContractReviewer(bundle)
    raise ValueError("Unsupported contract question bundle version")


def fit_review_threshold(cases, fit_groups, fit_source_hashes, min_groups=4, min_cases=12, max_empirical_error=0.1):
    """Fit a triage-only score cutoff on declared independent calibration data.

    This selects a review workload threshold; it does not calibrate probabilities
    or authorize legal decisions. A separate release/test evaluation is required.
    """
    if not cases:
        raise ValueError("Independent calibration cases required")
    groups = {c["group_id"] for c in cases}
    if groups & set(fit_groups) or any(c["source_hash"] in fit_source_hashes for c in cases):
        raise ValueError("Calibration overlaps training sources")
    if any(c.get("split") != "calibration" or c.get("template_independence_verified") is not True for c in cases):
        raise ValueError("Calibration role and template independence must be recorded")
    if len(groups) < min_groups or len(cases) < min_cases:
        raise ValueError("Insufficient independent calibration coverage")
    if {c["actual"] for c in cases} != set(TASK_CLASSES["qa"]):
        raise ValueError("Calibration must cover T, F and U")
    if len({c["predicted"] for c in cases}) < 2:
        raise ValueError("Collapsed predictions cannot validate discrimination")
    owners = {}
    for case in cases:
        if case["source_hash"] in owners and owners[case["source_hash"]] != case["group_id"]:
            raise ValueError("One source hash cannot masquerade as independent groups")
        owners[case["source_hash"]] = case["group_id"]
    if any(c["actual"] not in TASK_CLASSES["qa"] or c["predicted"] not in TASK_CLASSES["qa"] or
           not isinstance(c.get("score"), (int, float)) or not 0 <= c["score"] <= 1 for c in cases):
        raise ValueError("Invalid calibration prediction or target")
    candidates = []
    for cutoff in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
        kept = [c for c in cases if c["score"] >= cutoff]
        represented = {c["group_id"] for c in kept}
        if len(represented) < min_groups or not kept:
            continue
        error = sum(sum(c["actual"] != c["predicted"] for c in kept if c["group_id"] == group) /
                    sum(c["group_id"] == group for c in kept) for group in represented) / len(represented)
        if error <= max_empirical_error:
            candidates.append({"cutoff": cutoff, "coverage": len(kept)/len(cases), "source_balanced_empirical_error": error})
    chosen = max(candidates, key=lambda c: (c["coverage"], -c["cutoff"])) if candidates else None
    return {"scope": "triage_threshold_only_not_probability_calibration", "chosen": chosen,
            "calibration_groups": sorted(groups), "cases": len(cases), "decision_authority": False,
            "release_evaluation_required": True, "probabilities_calibrated": False,
            "calibration_data_sha256": digest(cases)}
