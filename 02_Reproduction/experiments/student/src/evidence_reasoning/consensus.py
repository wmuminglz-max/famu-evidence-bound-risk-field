"""Sequential call routing and conservative correction, fitted on real log pairs.

Weak reference labels are retained as weak. No live API call happens here.
"""
from __future__ import annotations

from collections import Counter
import numpy as np
from sklearn.linear_model import Ridge
from .text import verify_quotes
from .metrics import classification, weights

STATUSES = ("established", "not_established", "uncertain")
PRE_CALL_FIELDS = ("risk_proposition", "fact_context", "clause_a", "clause_b", "primary_status",
                   "primary_reason", "primary_confidence", "primary_evidence_quotes", "secondary_budget",
                   "policy_state", "gate_status")


def normalized_answer(payload):
    """Malformed or failed answers never acquire a legal unknown label."""
    if not isinstance(payload, dict):
        return {"status": None, "technical_status": "invalid_response"}
    parsed = payload.get("parsed_json", payload)
    if not isinstance(parsed, dict) or payload.get("success") is False:
        return {"status": None, "technical_status": "failed"}
    value = parsed.get("risk_status") or parsed.get("judgment") or parsed.get("status")
    if value not in STATUSES:
        if type(parsed.get("is_risk")) is bool:
            value = "established" if parsed["is_risk"] else "not_established"
        else:
            return {"status": None, "technical_status": "invalid_response"}
    return {"status": value, "technical_status": "ok"}


def pre_call_features(inp):
    # Construct a whitelist before feature extraction; adding secondary fields
    # or target annotations cannot affect call decisions.
    x = {key: inp.get(key) for key in PRE_CALL_FIELDS}
    text = "\n".join(str(x.get(k) or "") for k in ("clause_a", "clause_b", "fact_context"))
    reason = str(x.get("primary_reason") or "")
    quote = verify_quotes(x.get("primary_evidence_quotes") or [], text)
    confidence = x.get("primary_confidence")
    confidence_valid = isinstance(confidence, (int, float)) and not isinstance(confidence, bool) and np.isfinite(confidence) and 0 <= confidence <= 1
    return [float(x.get("primary_status") == s) for s in STATUSES] + [
        float(confidence) if confidence_valid else 0., float(not confidence_valid),
        min(len(text), 4000) / 4000, min(len(reason), 1000) / 1000,
        float(quote["present"]), float(quote["verbatim"]),
        float(any(w in reason for w in ("缺少", "不足", "无法", "未提供", "不明确"))),
        float(any(w in reason for w in ("矛盾", "冲突", "但", "除外"))),
        float(x.get("policy_state") not in (None, "NOT_RUN")),
        float(x.get("gate_status") not in (None, "NOT_RUN"))]


def post_call_features(inp):
    text = "\n".join(str(inp.get(k) or "") for k in ("clause_a", "clause_b", "fact_context"))
    quote = verify_quotes(inp.get("secondary_evidence_quotes") or [], text)
    secondary = inp.get("secondary_status")
    return pre_call_features(inp) + [float(secondary == s) for s in STATUSES] + [
        float(inp.get("primary_status") == secondary), float(quote["present"]), float(quote["verbatim"]),
        min(len(str(inp.get("secondary_reason") or "")), 1000) / 1000]


def utility(row, damage_penalty=2.):
    x, gold = row["input"], row.get("target_judgment")
    a, b = x.get("primary_status"), x.get("secondary_status")
    if gold not in STATUSES or a not in STATUSES or b not in STATUSES:
        return None
    return float(a != gold and b == gold) - damage_penalty * float(a == gold and b != gold)


class SequentialConsensus:
    def __init__(self, payload=None):
        self.payload = payload

    @staticmethod
    def _score(model, features):
        return float(np.dot(features, model["coefficients"]) + model["intercept"])

    def fit(self, rows, call_cost=.02, damage_penalty=2., human_review_cost=0., latency_cost_per_second=0., folds=3, seed=20260928):
        if any(r["split"] != "train" or not r.get("real_api_call") for r in rows):
            raise ValueError("Router fitting requires training-only real API trajectories")
        if any(not np.isfinite(v) or v < 0 for v in (call_cost, damage_penalty, human_review_cost, latency_cost_per_second)):
            raise ValueError("Policy costs must be finite and nonnegative")
        usable = [r for r in rows if r.get("target_judgment") in STATUSES]
        groups = sorted({r["group_id"] for r in usable})
        if folds < 2 or len(groups) < folds:
            raise ValueError("Post-policy routing requires independent groups for cross-fitting")
        for root in {r["root_id"] for r in usable}:
            if len({r["group_id"] for r in usable if r["root_id"] == root}) != 1:
                raise ValueError("One Consensus root spans multiple source groups")
        self.payload = {"call_cost": call_cost, "damage_penalty": damage_penalty,
                        "human_review_cost": human_review_cost, "latency_cost_per_second": latency_cost_per_second,
                        "route_threshold": 0., "correction_threshold": .05,
                        "fit_groups": sorted({r["group_id"] for r in usable}), "rows": len(usable),
                        "label_status": "weak_pending", "decision_authority": False,
                        "pre_call_fields": list(PRE_CALL_FIELDS),
                        "route_target": "oof_post_policy_net_utility", "folds": folds,
                        "cost_unit": "experimental_utility_not_currency",
                        "human_outcome_scope": "proposed_answer_before_human_review; no assumed human correction"}
        def correction(training):
            pairs = [r for r in training if utility(r, damage_penalty) is not None
                     and r["input"].get("secondary_technical_status", "ok") == "ok"]
            if not pairs:
                raise ValueError("No usable correction pairs in a training fold")
            model = Ridge(alpha=2.)
            model.fit(np.asarray([post_call_features(r["input"]) for r in pairs]),
                      [utility(r, damage_penalty) for r in pairs], sample_weight=weights([r["root_id"] for r in pairs]))
            return {"coefficients": model.coef_.tolist(), "intercept": float(model.intercept_)}
        shuffled = np.random.default_rng(seed).permutation(groups).tolist()
        assignment = {g: i % folds for i, g in enumerate(shuffled)}
        targets, certificates = {}, []
        for fold in range(folds):
            training = [r for r in usable if assignment[r["group_id"]] != fold]
            policy = SequentialConsensus(self.payload | {"correct": correction(training)})
            for i, row in enumerate(usable):
                if assignment[row["group_id"]] != fold:
                    continue
                outcome = policy.after_call(row["input"])
                gold, primary, selected = row["target_judgment"], row["input"].get("primary_status"), outcome["selected_status"]
                gain = float(primary != gold and selected == gold) - damage_penalty * float(primary == gold and selected != gold)
                targets[i] = gain - policy.costs(row["input"], outcome["action"])["total"]
                certificates.append({"sample_id": row["sample_id"], "held_group": row["group_id"], "fold": fold,
                    "fit_groups": sorted({r["group_id"] for r in training}), "selected_status": selected,
                    "action": outcome["action"], "target_net_utility": targets[i]})
        route = Ridge(alpha=2.)
        route.fit(np.asarray([pre_call_features(r["input"]) for r in usable]), [targets[i] for i in range(len(usable))],
                  sample_weight=weights([r["root_id"] for r in usable]))
        self.payload.update(route={"coefficients": route.coef_.tolist(), "intercept": float(route.intercept_)},
                            correct=correction(usable), oof_post_policy=certificates)
        return self

    def costs(self, inp, action):
        latency = inp.get("secondary_latency_ms")
        known = isinstance(latency, (int, float)) and not isinstance(latency, bool) and np.isfinite(latency) and latency >= 0
        review = self.payload.get("human_review_cost", 0.) * float(action == "review")
        elapsed = self.payload.get("latency_cost_per_second", 0.) * (latency / 1000 if known else 0.)
        call = self.payload.get("call_cost", .02)
        return {"call": call, "latency": elapsed, "review": review, "total": call + elapsed + review,
                "latency_observed": bool(known)}

    def before_call(self, inp):
        if not self.payload:
            raise ValueError("Consensus router is not fitted")
        try:
            budget = float(inp.get("secondary_budget", 1))
        except (TypeError, ValueError):
            budget = 0.
        if not np.isfinite(budget) or budget < 1:
            return {"call_secondary": False, "reason": "budget_exhausted", "expected_utility": None}
        score = self._score(self.payload["route"], pre_call_features(inp))
        return {"call_secondary": score > self.payload["route_threshold"],
                "reason": "expected_net_benefit" if score > self.payload["route_threshold"] else "retain_primary",
                "expected_utility": score, "decision_authority": False,
                "utility_scope": self.payload.get("route_target", "legacy_forced_secondary_gain"),
                "cost_unit": "experimental_utility_not_currency"}

    def after_call(self, inp):
        primary, secondary = inp.get("primary_status"), inp.get("secondary_status")
        valid_primary = primary in STATUSES
        if secondary not in STATUSES or inp.get("secondary_technical_status", "ok") != "ok":
            return {"selected_status": primary if valid_primary else None, "source": "primary" if valid_primary else None,
                    "action": "review", "technical_status": "secondary_failed", "decision_authority": False}
        if primary == secondary:
            return {"selected_status": primary, "source": "primary", "action": "retain", "technical_status": "ok", "decision_authority": False}
        text = "\n".join(str(inp.get(k) or "") for k in ("clause_a", "clause_b", "fact_context"))
        quote = verify_quotes(inp.get("secondary_evidence_quotes") or [], text)
        score = self._score(self.payload["correct"], post_call_features(inp))
        # Literal support is necessary for a proposed rewrite, never sufficient
        # to certify its legal correctness. Every rewrite remains review-only.
        replace = valid_primary and quote["present"] and quote["verbatim"] and score > self.payload["correction_threshold"]
        return {"selected_status": secondary if replace else primary if valid_primary else None,
                "source": "secondary" if replace else "primary" if valid_primary else None,
                "action": "review" if replace or not valid_primary else "retain", "expected_utility": score,
                "technical_status": "ok" if valid_primary else "primary_failed", "decision_authority": False}

    def select_thresholds(self, validation):
        if any(r["split"] != "validation" for r in validation) or set(self.payload["fit_groups"]) & {r["group_id"] for r in validation}:
            raise ValueError("Threshold selection must use separate validation groups")
        best = None
        net_target = self.payload.get("route_target") == "oof_post_policy_net_utility"
        # Changing the downstream policy would invalidate the learned call target.
        corrections = (self.payload["correction_threshold"],) if net_target else (.05, .1, .2, .4, 1e6)
        for route in (0. if net_target else self.payload["call_cost"], .05, .1, .2, 1e6):
            for correct in corrections:
                self.payload.update(route_threshold=route, correction_threshold=correct)
                report = evaluate_policy(self, validation)
                score = (report["net_utility_per_root"], -report["damage"], -report["secondary_calls"])
                if best is None or score > best[0]:
                    best = (score, route, correct)
        self.payload.update(route_threshold=best[1], correction_threshold=best[2],
                            selection_groups=sorted({r["group_id"] for r in validation}))
        return self


def evaluate_policy(model, rows, policy="learned"):
    y, final, primary_answers, roots = [], [], [], []
    calls, repaired, damaged, overwritten, errors = [], [], [], [], []
    details = []
    costs, call_failures, reviews, missing_latency = [], [], [], []
    for row in rows:
        x, gold = row["input"], row.get("target_judgment")
        primary, secondary = x.get("primary_status"), x.get("secondary_status")
        outcome = None
        if policy == "primary":
            called, selected = False, primary
        elif policy == "always_secondary":
            called = True
            selected = secondary if x.get("secondary_technical_status", "ok") == "ok" else None
        elif policy == "agreement_guard":
            called, selected = True, primary
        else:
            called = model.before_call(x)["call_secondary"]
            outcome = model.after_call(x) if called else None
            selected = outcome["selected_status"] if outcome else primary
        call_failed = called and (secondary not in STATUSES or x.get("secondary_technical_status", "ok") != "ok")
        action = outcome["action"] if outcome else "review" if call_failed else "retain"
        cost = model.costs(x, action) if called else {"total": 0., "latency_observed": True}
        costs.append(cost["total"])
        call_failures.append(call_failed)
        reviews.append(action == "review")
        missing_latency.append(called and not cost["latency_observed"])
        y.append(STATUSES.index(gold) if gold in STATUSES else -1)
        final.append(STATUSES.index(selected) if selected in STATUSES else -1)
        primary_answers.append(STATUSES.index(primary) if primary in STATUSES else -1)
        roots.append(row["root_id"])
        evaluable = gold in STATUSES
        repair = evaluable and primary != gold and selected == gold
        damage = evaluable and primary == gold and selected != gold
        repaired.append(repair)
        damaged.append(damage)
        calls.append(called)
        overwritten.append(selected != primary)
        errors.append(selected not in STATUSES)
        details.append({"sample_id": row["sample_id"], "called": called, "primary": primary,
                        "selected": selected, "truth": gold, "label_status": "weak_pending",
                        "call_failed": bool(call_failed), "action": action, "costs": cost})
    w = weights(roots)
    utility_values = np.asarray(repaired, dtype=float) - model.payload["damage_penalty"] * np.asarray(damaged) - np.asarray(costs)
    return {"policy": policy, "metrics": classification(y, final, roots), "total_rows": len(rows),
            "secondary_calls": int(sum(calls)), "repair": int(sum(repaired)), "damage": int(sum(damaged)),
            "overwrites": int(sum(overwritten)), "technical_failures": int(sum(errors)),
            "failed_secondary_calls": int(sum(call_failures)), "human_reviews": int(sum(reviews)),
            "calls_missing_latency": int(sum(missing_latency)),
            "call_cost": model.payload["call_cost"], "human_review_cost": model.payload.get("human_review_cost", 0.),
            "latency_cost_per_second": model.payload.get("latency_cost_per_second", 0.),
            "net_utility_per_root": float(np.average(utility_values, weights=w)) if len(w) else 0.,
            "cost_unit": "experimental_utility_not_currency; missing latency is uncosted and explicitly counted",
            "details": details}


def summarize_call_costs(call_records):
    """Real-call token/latency/retry/reviewer ledger; unknown costs stay unknown."""
    from .experiment_protocol import audit_telemetry
    return audit_telemetry(call_records)
