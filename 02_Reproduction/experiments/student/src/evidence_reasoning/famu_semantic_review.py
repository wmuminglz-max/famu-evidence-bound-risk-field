"""Review-only proposition binding and shadow checks; never changes a model label."""
from .review_ir import digest

PROPOSITION_KINDS = {'text_presence', 'actual_fact', 'normative_permission', 'evidence_sufficiency'}
FIELD_STATES = {'filled_visible', 'redacted_present', 'blank_in_original', 'unverified'}


def three_valued(operator, values):
    values = list(values)
    if not values or any(v not in {'T', 'F', 'U'} for v in values):
        raise ValueError('Nonempty T/F/U premises required')
    if operator == 'and':
        return 'F' if 'F' in values else 'U' if 'U' in values else 'T'
    if operator == 'or':
        return 'T' if 'T' in values else 'U' if 'U' in values else 'F'
    raise ValueError('Only explicitly confirmed conjunction/disjunction is supported')


def validate_binding(binding, documents):
    """Exact source/version/span binding; no bracket-placeholder heuristics."""
    required = {'source_id','source_sha256','start','end','quote','field_state','state_provenance',
                'proposition_id','relation','evidence_kind','scope_version'}
    if set(binding) != required:
        raise ValueError('Incomplete evidence binding')
    source = documents[binding['source_id']]
    if digest(source) != binding['source_sha256']:
        raise ValueError('Source version changed')
    a,b=binding['start'],binding['end']
    if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(source) or source[a:b] != binding['quote']:
        raise ValueError('Evidence quote does not match source offsets')
    if binding['field_state'] not in FIELD_STATES or binding['relation'] not in {'support','refute','irrelevant','unverified'}:
        raise ValueError('Unknown field state or relation')
    if binding['field_state'] != 'unverified' and not binding['state_provenance']:
        raise ValueError('Field fill/redaction state needs provenance')
    if not binding['scope_version'] or not binding['proposition_id']:
        raise ValueError('Proposition and scope version must be explicit')
    return binding


def validate_confirmed_proposition(item):
    required = ['canonical_proposition','proposition_kind','subject','predicate','object',
                'quantifier','polarity','review_time','document_scope']
    if item.get('review_status') != 'expert_confirmed' or not item.get('confirmation_provenance'):
        raise ValueError('Expert confirmation required before v3 training admission')
    if item.get('proposition_kind') not in PROPOSITION_KINDS or any(item.get(k) is None for k in required):
        raise ValueError('Unresolved proposition fields cannot become training facts')
    if item.get('task') == 'risk_status' and not item.get('risk_proposition'):
        raise ValueError('An explicit risk proposition is required')
    if item.get('split') != 'train':
        raise ValueError('Development or test review cannot be reassigned to training')
    return item


def shadow_check(raw_prediction, states, rule, scope_id):
    """Only a confirmed rule of the same scope can yield a diagnostic derivation.

    Confidence and U are not interchangeable; none of these actions releases a
    contract. Raw prediction remains untouched even when a conflict is found.
    """
    result = {'raw_prediction':raw_prediction,'derived_truth':None,'action':'record_only',
              'automatic_release':False,'mode':'shadow','reason':None}
    if not rule or rule.get('status') != 'expert_confirmed' or not rule.get('provenance') or rule.get('scope_id') != scope_id:
        return result | {'action':'defer_to_human','reason':'unconfirmed_or_mismatched_rule'}
    premises=rule.get('premises',[])
    if not premises or any(k not in states for k in premises):
        return result | {'action':'defer_to_human','reason':'missing_bound_premise'}
    derived=three_valued(rule['operator'],[states[k] for k in premises])
    result['derived_truth']=derived
    if derived=='U':
        result.update(action='need_more_evidence',reason='no_decisive_premise_under_confirmed_rule')
    elif raw_prediction!=derived:
        result.update(action='defer_to_human',reason='model_rule_conflict')
    return result


def admit_new_source(record, known_families, known_hashes):
    if record.get('source_family_id') in known_families or record.get('source_sha256') in known_hashes:
        raise ValueError('Existing family/source cannot fill an independent-source slot')
    if not record.get('source_family_id') or not record.get('source_sha256') or record.get('ancestry_review') != 'confirmed_independent':
        raise ValueError('Independent source ancestry is not yet confirmed')
    if record.get('pool') not in {'development','calibration','sealed_test'}:
        raise ValueError('Allocation pool must be fixed at admission')
    if record.get('pool') == 'sealed_test' and record.get('used_for_model_selection'):
        raise ValueError('Selection-exposed material cannot be sealed test')
    return record
