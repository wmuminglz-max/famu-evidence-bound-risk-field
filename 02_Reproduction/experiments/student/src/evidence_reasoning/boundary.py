"""Verified source withdrawals and length/position-matched negative controls."""
from __future__ import annotations

import itertools
from .logic import solve, verify_certificate
from .provenance import withdraw, visible_evidence


def withdrawal_boundaries(payload: dict, max_sources: int = 10) -> dict:
    original = solve(payload)
    if original["input_status"] != "valid" or original["proposition_label"] == "U":
        return {"original": original, "minimal_sets": [], "controls": [], "exhaustive": True, "evaluated": 0}
    active = [s["id"] for s in payload["sources"] if s.get("available", True)
              and s["id"] not in payload.get("withdrawn_sources", [])]
    if len(active) > max_sources:
        return {"original": original, "minimal_sets": [], "controls": [], "exhaustive": False,
                "evaluated": 0, "reason": "source_budget_exceeded"}
    critical, controls, count = [], [], 0
    for size in range(1, len(active) + 1):
        for combination in itertools.combinations(sorted(active), size):
            # Pure withdrawal expands the set of worlds. Supersets cannot be minimal.
            if any(set(old["sources"]) <= set(combination) for old in critical):
                continue
            changed = withdraw(payload, combination)
            answer = solve(changed)
            count += 1
            if answer["input_status"] != "valid":
                raise ValueError("Pure withdrawal from consistent input became inconsistent")
            if answer["proposition_label"] == "U":
                if not verify_certificate(changed, answer["certificate"]):
                    raise ValueError("Invalid reference certificate")
                critical.append({"sources": list(combination), "answer": answer})
            elif answer["proposition_label"] != original["proposition_label"]:
                raise ValueError("Pure withdrawal reversed a determinate answer")
            else:
                controls.append({"sources": list(combination), "answer": answer})
    return {"original": original, "minimal_sets": critical, "controls": controls,
            "exhaustive": True, "evaluated": count}


def _removal_profile(payload, sources):
    original = visible_evidence(payload)
    retained = {e["id"] for e in visible_evidence(withdraw(payload, sources))}
    positions = [i for i, e in enumerate(original) if e["id"] not in retained]
    return (len(sources), sum(len(e["text"]) for e in original if e["id"] not in retained),
            sum(positions) / max(1, len(positions)) / max(1, len(original)))


def matched_pairs(payload: dict, max_sources: int = 10) -> list[dict]:
    result = withdrawal_boundaries(payload, max_sources)
    pairs = []
    for critical in result["minimal_sets"]:
        profile = _removal_profile(payload, critical["sources"])
        compatible = [c for c in result["controls"] if len(c["sources"]) == profile[0]]
        if not compatible:
            pairs.append({"critical": critical, "control": None, "matched": False})
            continue
        def distance(control):
            p = _removal_profile(payload, control["sources"])
            return abs(profile[1] - p[1]) / max(1, profile[1], p[1]) + abs(profile[2] - p[2])
        control = min(compatible, key=lambda c: (distance(c), c["sources"]))
        pairs.append({"critical": critical, "control": control, "matched": True,
                      "match_distance": distance(control), "profile": list(profile)})
    return pairs


def select_boundary_groups(groups: list[dict], budget: int, strategy: str = "boundary", seed: int = 0) -> list[dict]:
    """Only training groups may be mined; scores must come from training predictions."""
    import random
    if budget < 0 or strategy not in {"boundary", "hardness", "random"}:
        raise ValueError("Invalid sampling request")
    if any(g.get("split") != "train" for g in groups):
        raise ValueError("Boundary mining cannot inspect development/test groups")
    rng = random.Random(seed)
    ordered = list(groups)
    rng.shuffle(ordered)
    if strategy != "random":
        ordered.sort(key=lambda g: float(g.get("hardness", 0)) +
                     (float(g.get("boundary_value", 0)) if strategy == "boundary" else 0), reverse=True)
    # First pass gives each root one opportunity, then allocate remaining budget.
    selected, seen = [], set()
    for g in ordered:
        if g["root_id"] not in seen and len(selected) < budget:
            selected.append(g)
            seen.add(g["root_id"])
    ids = {id(g) for g in selected}
    selected.extend(g for g in ordered if id(g) not in ids and len(selected) < budget)
    return selected
