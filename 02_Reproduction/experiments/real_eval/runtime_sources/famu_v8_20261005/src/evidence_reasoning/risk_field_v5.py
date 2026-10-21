"""Typed evidence bottleneck and finite semantic boundaries. No free-text bypass."""
from itertools import product, combinations
from .logic import evaluate
from .schema import expression_variables

STATES = ('supported', 'refuted', 'unknown', 'conflict')
DECISIONS = ('established', 'not_established', 'unknown', 'conflict')


def evidence_state(value, proposition_type, visibility, coverage_checked=False):
    """Visibility is an observed/verified property, never inferred from retrieval failure."""
    if value not in STATES:
        raise ValueError('Unknown evidence state')
    if visibility in {'redacted', 'not_provided', 'ocr_failed', 'retrieval_miss'}:
        return 'unknown'
    if visibility in {'visible_blank', 'complete_absence'}:
        return 'refuted' if proposition_type == 'text_presence' and coverage_checked else 'unknown'
    if visibility != 'visible':
        raise ValueError('Unknown visibility')
    return value


def legal_worlds(fields, states, constraints=()):
    if len(fields) > 10 or len(fields) != len(set(fields)) or len(fields) != len(states):
        raise ValueError('Invalid field count or duplicate fields')
    if any(s not in STATES for s in states):
        raise ValueError('Unannotated is not an unknown observation')
    if any(expression_variables(c) - set(fields) for c in constraints):
        raise ValueError('Constraint references undeclared field')
    domains = [(True,) if s == 'supported' else (False,) if s == 'refuted'
               else () if s == 'conflict' else (False, True) for s in states]
    return [dict(zip(fields, x)) for x in product(*domains)
            if all(evaluate(c, dict(zip(fields, x))) for c in constraints)]


def bounds(rule, fields, states, constraints=()):
    if expression_variables(rule) - set(fields):
        raise ValueError('Rule references undeclared field')
    worlds = legal_worlds(fields, states, constraints)
    if not worlds:
        return dict(lower=None, upper=None, state='conflict', world_count=0,
                    decision_action='human_review', witnesses=None)
    pos = [w for w in worlds if evaluate(rule, w)]
    neg = [w for w in worlds if not evaluate(rule, w)]
    lo, hi = int(not neg), int(bool(pos))
    witnesses = None
    if pos and neg:
        t, f = min(product(pos, neg), key=lambda pair: sum(pair[0][k] != pair[1][k] for k in fields))
        witnesses = dict(established_world=t, not_established_world=f,
                         differing_fields=[k for k in fields if t[k] != f[k]], hypothesis_only=True)
    return dict(lower=lo, upper=hi, state='unknown' if lo != hi else DECISIONS[0 if lo else 1],
                world_count=len(worlds), decision_action='request_evidence' if lo != hi else 'answer',
                witnesses=witnesses)


def certified_decision(rule, fields, states, anchors=None, constraints=()):
    """Certificates are relative to supplied rule/field states, not legal truth guarantees."""
    answer = bounds(rule, fields, states, constraints)
    answer['certificate'] = None
    if answer['state'] in {'unknown', 'conflict'}:
        return answer
    known = [i for i, s in enumerate(states) if s != 'unknown']
    for size in range(len(known)+1):
        found = False
        for subset in combinations(known, size):
            relaxed = [states[i] if i in subset else 'unknown' for i in range(len(fields))]
            if bounds(rule, fields, relaxed, constraints)['state'] == answer['state']:
                answer['certificate'] = dict(fields=[fields[i] for i in subset],
                    observations={fields[i]: states[i] for i in subset},
                    anchors={fields[i]: (anchors or {}).get(fields[i], []) for i in subset},
                    minimal_cardinality=True, scope='declared_rule_and_extracted_states')
                found = True
                break
        if found:
            break
    return answer


def verify_anchor(text_by_id, anchor):
    return (set(anchor) == {'evidence_id', 'start', 'end', 'text'}
            and anchor['evidence_id'] in text_by_id
            and type(anchor['start']) is int and type(anchor['end']) is int
            and 0 <= anchor['start'] < anchor['end'] <= len(text_by_id[anchor['evidence_id']])
            and text_by_id[anchor['evidence_id']][anchor['start']:anchor['end']] == anchor['text'])


def refinement_pair(before, after):
    """Only consistent information addition; fact revisions must use another objective."""
    if len(before) != len(after) or any(s not in STATES for s in before+after):
        return False
    return all(a == b or a == 'unknown' and b in {'supported', 'refuted'} for a, b in zip(before, after))


def state_table(rule, fields, constraints=()):
    """Discrete concept assignments -> exact semantic class; used for differentiable WMC."""
    assignments = list(product(range(4), repeat=len(fields)))
    labels = [DECISIONS.index(bounds(rule, fields, [STATES[i] for i in row], constraints)['state'])
              for row in assignments]
    return assignments, labels


def soft_semantics(probabilities, assignments, labels):
    """Differentiable weighted model count; neural scores are not calibrated bounds."""
    import torch
    masses = torch.ones((len(probabilities), len(assignments)), device=probabilities.device,
                        dtype=probabilities.dtype)
    for field in range(probabilities.shape[1]):
        masses = masses * probabilities[:, field, assignments[:, field]]
    out = torch.zeros((len(probabilities), 4), device=masses.device, dtype=masses.dtype)
    return out.scatter_add(1, labels.expand(len(masses), -1), masses)


def information_loss(before_probs, after_probs):
    """Consistent information refinement: expected lower rises, upper falls."""
    import torch
    lower_before, lower_after = before_probs[:, 0], after_probs[:, 0]
    upper_before = before_probs[:, 0]+before_probs[:, 2]
    upper_after = after_probs[:, 0]+after_probs[:, 2]
    return (torch.relu(lower_before-lower_after)+torch.relu(upper_after-upper_before)).mean()
