"""Shared semantic routing and uncertainty policy for runtime and evaluation."""
from .schema import decision_route, expression_variables, model_input


def workflow_status(input_status, label, extraction_abstention=False):
    if input_status == 'technical_failure':
        return 'technical_failure'
    if input_status == 'rule_unresolved':
        return 'rule_review'
    if input_status == 'conflicting':
        return 'conflict_review'
    if extraction_abstention:
        return 'extraction_review'
    return 'request_materials' if label == 'U' else 'accept_conditional' if label in {'T', 'F'} else 'review'


def select_decision(payload, logic, neural_label=None, uncertain_conditions=()):
    x = model_input(payload)
    routing = decision_route(x)
    proposed = logic["proposition_label"] if routing["executor_eligible"] else neural_label
    uncertain = set(uncertain_conditions)
    relevant = expression_variables(x["rule"])
    for constraint in x["constraints"]:
        relevant |= expression_variables(constraint)
    # Unresolved opposing/exception candidates can undermine an apparently
    # determinate observation too. Do not only review uncertainty after U.
    extraction_abstention = routing["executor_eligible"] and bool(uncertain & relevant)
    invalid = logic["input_status"] != "valid" or routing["result_origin"] == "review"
    return {"proposition_label": None if invalid or extraction_abstention else proposed,
            "workflow_status": 'rule_review' if routing['task_mode'] == 'text_only' and logic['input_status'] == 'valid' else workflow_status(logic['input_status'], proposed, extraction_abstention),
            "risk_established": None, "risk_established_status": "technical_failure" if logic['input_status'] == 'technical_failure' else "unlabeled",
            "severity": None,
            "routing": routing, "result_origin": routing["result_origin"],
            "extraction_uncertain_conditions": sorted(uncertain),
            "abstention_reason": "ambiguous_extraction_requires_review" if extraction_abstention else None,
            "unknown_scope": "given_rule_and_extracted_observations; extraction uncertainty is not a material-insufficiency label"}


def execute_extracted(payload, extractor):
    """E2 production-independent route: learned observations, mandatory exact gate."""
    from .logic import solve, verify_certificate
    x = model_input(payload)
    initial = solve(x)
    if initial['input_status'] != 'valid':
        return {**select_decision(x, initial), 'conditional_logic': initial, 'certificate_verified': None}
    if not decision_route(x)['executor_eligible']:
        raise ValueError('Exact extraction route requires structured given/predicted rule input')
    extracted = extractor.extract(x)
    logic = solve(extracted['input'])
    return {**select_decision(x, logic, uncertain_conditions=extracted.get('uncertain_conditions', [])),
            'conditional_logic': logic, 'evidence_predictions': extracted['evidence_predictions'],
            'condition_evidence_status': extracted.get('condition_evidence_status', {}),
            'certificate_verified': verify_certificate(extracted['input'], logic['certificate']) if logic['certificate'] else None,
            'certificate_scope': 'given_rule_and_extracted_observations_only', 'decision_authority': False}
