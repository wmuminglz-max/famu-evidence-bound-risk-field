"""Import externally completed double annotations; never fill in human decisions."""
from __future__ import annotations

import copy
from .schema import validate_input, validate_record, digest, model_input
from .data import audit_splits


def import_annotations(entries, split="external", known_records=()):
    if split not in {"external", "regression"}:
        raise ValueError("New human reviews enter external evaluation or known-data regression")
    known_roots = {r["meta"]["root_id"] for r in known_records}
    known_groups = {r["meta"]["group_id"] for r in known_records}
    known_input = {digest(model_input(r["input"])) for r in known_records}
    result = []
    for entry in entries:
        x = validate_input(entry["input"])
        a,b = entry.get("independent_annotation_1"), entry.get("independent_annotation_2")
        if not isinstance(a,dict) or not isinstance(b,dict):
            raise ValueError("Both independent annotations must be supplied by reviewers")
        people = [str(v.get("annotator_id") or "").strip() for v in (a,b)]
        if not all(people) or people[0] == people[1]:
            raise ValueError("Two distinct reviewer identifiers are required")
        def target(annotation):
            if annotation.get("label") not in {"T","F","U"}:
                raise ValueError("Reviewer must explicitly annotate T/F/U")
            conditions = annotation.get("conditions",{})
            if not isinstance(conditions,dict) or set(conditions) != {c["id"] for c in x["conditions"]} or not set(conditions.values()) <= {"T","F","U"}:
                raise ValueError("Every supplied condition needs an independent T/F/U annotation")
            evidence = annotation.get("evidence",{})
            allowed = {f"{c['id']}|{e['id']}" for c in x["conditions"] for e in x["evidence"]}
            if not isinstance(evidence,dict) or not set(evidence) <= allowed or not set(evidence.values()) <= {"irrelevant","support","refute"}:
                raise ValueError("Invalid evidence annotation")
            severity = annotation.get("severity")
            if severity is not None and (type(severity) is not int or severity not in range(4)):
                raise ValueError("Severity must be null or an ordinal level 0..3")
            result = {"label":annotation["label"],"conditions":conditions,"evidence":evidence,
                      "severity":severity,"related":annotation.get("related")}
            for key in ('spans', 'proof_groups', 'scope'):
                if key in annotation:
                    result[key] = copy.deepcopy(annotation[key])
            if entry.get('protocol_version') == 'expert-20260929-v2':
                result.update(risk_established_status=annotation.get('risk_established_status', 'unlabeled'),
                              severity_basis=copy.deepcopy(annotation.get('severity_basis')))
            return result
        ta,tb = target(a),target(b)
        # Legal structure disagreements count even when final T/F/U matches.
        review_fields = ('proposition_type', 'condition_context', 'source_relations', 'risk_established_status', 'severity_basis', 'rule_assessment')
        structural_disagreement = any(a.get(k) != b.get(k) for k in review_fields)
        if ta != tb or structural_disagreement:
            adjudication = entry.get("adjudication")
            if not isinstance(adjudication,dict) or not adjudication.get("annotator_id") or adjudication["annotator_id"] in people:
                raise ValueError("Disagreement requires a distinct adjudicator and explicit final labels")
            final = target(adjudication)
            people.append(adjudication["annotator_id"])
        else:
            final = ta
        root,group = entry["root_id"],entry["group_id"]
        if split == "external" and (root in known_roots or group in known_groups or digest(model_input(x)) in known_input):
            raise ValueError("Previously used groups cannot become a new independent blind test")
        row = {"input":x,"targets":copy.deepcopy(final),"meta":{
            "sample_id":"HUM-" + digest(entry)[:20],"root_id":root,"group_id":group,
            "split":split,"label_status":"human_adjudicated","domain":"external_human_review",
            "human_gold":True,"reviewer_ids":people,"annotation_sha256":digest(entry),
            "identity_verification":"reviewer_identifiers_supplied_by_importer; not independently authenticated"}}
        row['meta']['annotation_events'] = copy.deepcopy([a, b] + ([entry['adjudication']] if len(people) == 3 else []))
        row['meta']['protocol_version'] = entry.get('protocol_version', 'legacy')
        result.append(validate_record(row))
    audit_splits(result)
    return result
