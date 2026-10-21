"""AND/OR source lineage, exact withdrawal and content-addressed recomputation."""
from __future__ import annotations

import copy
import itertools
from .schema import digest, validate_input


def evidence_signature(evidence):
    """Only verified equivalent assertions with the same live origins collapse."""
    eq = evidence.get("equivalence")
    assertion = (eq["assertion_id"], eq["canonical_text"]) if eq else ("literal", evidence["text"])
    paths = tuple(sorted(tuple(sorted(g)) for g in evidence["live_supports"]))
    return assertion, paths, evidence.get("scope_id", ""), evidence.get("scope_kind", "")


def minimal_sets(groups):
    result = []
    for item in sorted({frozenset(g) for g in groups}, key=lambda x: (len(x), sorted(x))):
        if not any(previous <= item for previous in result):
            result.append(item)
    return result


def lineage(payload: dict) -> dict[str, list[frozenset]]:
    x = validate_input(payload)
    by_id = {e["id"]: e for e in x["evidence"]}
    resolved, visiting = {}, set()

    def visit(eid):
        if eid in resolved:
            return resolved[eid]
        if eid in visiting:
            raise ValueError("Cycle in evidence derivation")
        visiting.add(eid)
        evidence = by_id[eid]
        if "supports" in evidence:
            paths = evidence["supports"]
        else:
            paths = []
            for parent_group in evidence["parents"]:
                options = [visit(parent) for parent in parent_group]
                count = 1
                for option in options:
                    count *= len(option)
                if count + len(paths) > 4096:
                    raise ValueError("Lineage expansion budget exceeded")
                paths.extend(set().union(*pieces) for pieces in itertools.product(*options))
        resolved[eid] = minimal_sets(paths)
        visiting.remove(eid)
        return resolved[eid]

    for eid in by_id:
        visit(eid)
    return resolved


def visible_evidence(payload: dict) -> list[dict]:
    x = validate_input(payload)
    removed = set(x["withdrawn_sources"])
    available = {s["id"] for s in x["sources"] if s.get("available", True)
                 and s.get('authenticity') != 'rejected' and s.get('admissibility') != 'rejected'} - removed
    paths = lineage(x)
    output = []
    for evidence in x["evidence"]:
        live = [sorted(p) for p in paths[evidence["id"]] if p <= available]
        if live:
            output.append({**evidence, "live_supports": live})
    return output


def withdraw(payload: dict, sources) -> dict:
    x = validate_input(payload)
    all_sources = {s["id"] for s in x["sources"]}
    if not set(sources) <= all_sources:
        raise ValueError("Withdrawal contains unknown source")
    x["withdrawn_sources"] = sorted(set(x["withdrawn_sources"]) | set(sources))
    return x


def visible_observations(payload: dict) -> list[dict]:
    x = validate_input(payload)
    eids = {e["id"] for e in visible_evidence(x)}
    return [o for o in x["observations"] if o["value"] is not None and
            any(set(g) <= eids for g in o["evidence_groups"])]


def materialize(payload: dict) -> dict:
    """Return only available information; never reuse stale summaries or vectors."""
    x = validate_input(payload)
    evidence = visible_evidence(x)
    live = {s for e in evidence for group in e["live_supports"] for s in group}
    result = copy.deepcopy(x)
    result["sources"] = [s for s in x["sources"] if s["id"] in live]
    result["evidence"] = [{k: v for k, v in e.items() if k not in {"parents", "live_supports", "supports"}}
                          | {"supports": e["live_supports"]} for e in evidence]
    result["observations"] = visible_observations(x)
    result["withdrawn_sources"] = []
    # Remove dead alternatives from otherwise live claim provenance.
    eids = {e["id"] for e in evidence}
    for o in result["observations"]:
        o["evidence_groups"] = [g for g in o["evidence_groups"] if set(g) <= eids]
    return result


def visible_fingerprint(payload: dict) -> str:
    return digest(materialize(payload))
