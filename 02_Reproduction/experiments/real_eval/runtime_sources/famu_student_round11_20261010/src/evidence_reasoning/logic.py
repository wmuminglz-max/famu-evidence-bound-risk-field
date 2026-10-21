"""Exact finite-world reference semantics; no Python eval, no closed-world default."""
from __future__ import annotations

import itertools
from .schema import validate_input
from .provenance import visible_observations


def evaluate(expression, world: dict[str, bool]) -> bool:
    if isinstance(expression, bool):
        return expression
    if isinstance(expression, str):
        return world[expression]
    op, args = next(iter(expression.items()))
    if op == "not":
        return not evaluate(args, world)
    return (all if op == "all" else any)(evaluate(a, world) for a in args)


def solve(payload: dict) -> dict:
    x = validate_input(payload)
    base = {"proposition_label": None, "input_status": "valid", "certificate": None,
            "condition_states": {}, "world_count": 0, "scope": "given_rule_and_observations"}
    if x.get("technical_status") == "failed":
        return base | {"input_status": "technical_failure"}
    if x.get("rule_status") == "unresolved":
        return base | {"input_status": "rule_unresolved"}
    observed = {}
    for claim in visible_observations(x):
        key, value = claim["condition"], claim["value"]
        if key in observed and observed[key] != value:
            return base | {"input_status": "conflicting", "conflicting_conditions": [key]}
        observed[key] = value
    ids = [c["id"] for c in x["conditions"]]
    missing = [cid for cid in ids if cid not in observed]
    witnesses, domains, count = {}, {cid: set() for cid in ids}, 0
    for assignment in itertools.product((False, True), repeat=len(missing)):
        world = observed | dict(zip(missing, assignment))
        if not all(evaluate(c, world) for c in x["constraints"]):
            continue
        count += 1
        truth = evaluate(x["rule"], world)
        witnesses.setdefault(truth, world)
        for cid in ids:
            domains[cid].add(world[cid])
    if not count:
        return base | {"input_status": "conflicting", "reason": "Unsatisfiable observations and constraints"}
    label = "U" if len(witnesses) == 2 else ("T" if True in witnesses else "F")
    states = {cid: "U" if len(values) == 2 else ("T" if True in values else "F") for cid, values in domains.items()}
    certificate = {"true_world": witnesses[True], "false_world": witnesses[False]} if label == "U" else None
    return base | {"proposition_label": label, "certificate": certificate, "condition_states": states,
                   "world_count": count, "missing_conditions": [c for c, state in states.items() if state == "U"]}


def verify_certificate(payload: dict, certificate: dict | None) -> bool:
    if not isinstance(certificate, dict):
        return False
    x = validate_input(payload)
    if x.get("technical_status", "ok") != "ok" or x.get("rule_status", "resolved") != "resolved":
        return False
    ids = {c["id"] for c in x["conditions"]}
    for key, truth in (("true_world", True), ("false_world", False)):
        world = certificate.get(key)
        if not isinstance(world, dict) or set(world) != ids or any(type(v) is not bool for v in world.values()):
            return False
        if any(world[o["condition"]] != o["value"] for o in visible_observations(x)):
            return False
        if not all(evaluate(c, world) for c in x["constraints"]) or evaluate(x["rule"], world) != truth:
            return False
    return True
