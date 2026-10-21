"""Frozen experiment manifest and observable cost/OOF gates, without API calls."""
from pathlib import Path
import hashlib
import math
from .schema import digest
from .data import write_json
from .metrics import complementary

EXPERIMENTS = {
    'E0': 'same-object severity: NB frequency controls / TF-IDF linear / frozen embedding MLP',
    'E1': 'given rule + gold conditions + exact execution (oracle)',
    'E2': 'given rule + predicted conditions + exact execution',
    'E3': 'gold conditions + learned rule composition (oracle)',
    'E4': 'mean / condition attention / scope / source merging / global pooling ablations',
    'E5': 'same-version LLM extraction + exact execution versus direct structured answer',
    'E6': 'raw-document predicted rule/conditions/source structure; separate from given-rule',
    'E7': 'conditional MoE: best single, parameter match, mean, random/constant gate, OOF stacking, learned gate',
}


def protocol():
    return {'version': 'expert-20260929-v2', 'research_title': '不完整合同材料下的命题可确定性：来源依赖与例外作用域的可核验建模和评价',
        'experiments': EXPERIMENTS, 'seeds': [20260929, 20260930, 20261001],
        'primary_comparison': 'E2 versus E4 source-constrained condition attention',
        'primary_metric': 'false_certainty', 'primary_denominator': 'all eligible gold-U source-group-balanced mass',
        'coverage_metrics': ['coverage', 'determinate_coverage', 'u_subset_coverage', 'false_certainty_among_accepted_u'],
        'evaluation_weighting': 'source groups equal; roots equal within group; views equal within root',
        'bootstrap': {'unit': 'paired_source_group', 'iterations': 2000},
        'decision_thresholds': {'status': 'team_decision_required_before_test_freeze', 'coverage_target': None,
            'minimum_effect': None, 'maximum_u_coverage_loss': None, 'maximum_cost_increase': None},
        'attention_budget': {'frozen_encoder': True, 'hidden_dimensions': [32, 64], 'max_configs': 4,
                             'max_trainable_parameters': 500000, 'seeds_per_config': 3},
        'stop_conditions': ['no oracle routing benefit', 'benefit only with gold structure', 'no gain on new source groups',
                            'gain explained by parameter/input budget', 'gain obtained by excessive abstention/U/cost'],
        'test_requirements': ['natural and targeted challenge tests reported separately', 'held-out label custodian',
            'model/prompt/threshold/source groups/evaluator frozen before scoring',
            'known three contracts stay development', 'active sampling excluded from final test'],
        'not_claimed': ['new possible-world logic', 'full defeasible legal reasoning', 'attention proves legal validity',
                        'MoE or graph improves real legal reasoning before external evidence'],
        'human_cost_budget': {'double_label_48_hours_assumption': [32, 64], 'measured_minutes': None,
                              'collection_anonymization_law_checks_adjudication_extra': True}}


def freeze_manifest(output, artifacts, registered_protocol, custodian_id):
    thresholds = registered_protocol.get('decision_thresholds', {})
    for key in ('coverage_target', 'minimum_effect', 'maximum_u_coverage_loss', 'maximum_cost_increase'):
        if thresholds.get(key) is None:
            raise ValueError('Choose deployment/effect/cost targets before freeze: '+key)
    if not 0 < thresholds['coverage_target'] <= 1 or not 0 < thresholds['minimum_effect'] <= 1:
        raise ValueError('Coverage/effect targets must be in (0,1]')
    if not 0 <= thresholds['maximum_u_coverage_loss'] <= 1 or thresholds['maximum_cost_increase'] < 0:
        raise ValueError('Invalid coverage or cost noninferiority bound')
    if not custodian_id or not {'model', 'prompt', 'thresholds', 'source_groups', 'evaluator'} <= set(artifacts):
        raise ValueError('Freeze all model/prompt/threshold/group/evaluator artifacts and name the independent label custodian')
    files = {key: {'path': str(Path(path).resolve()), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}
             for key, path in artifacts.items()}
    result = {'protocol': registered_protocol, 'artifacts': files, 'label_custodian_id': custodian_id,
              'status': 'frozen', 'protocol_sha256': digest(registered_protocol)}
    output = Path(output)
    if output.exists():
        raise ValueError('A freeze manifest is immutable; use a new experiment version')
    write_json(output, result)
    return result


def audit_telemetry(records):
    fields = ('input_tokens', 'output_tokens', 'latency_ms', 'encoding_ms', 'retry_count', 'lawyer_review_minutes', 'cost')
    result = {'calls': len(records), 'failed_calls': sum(r.get('status') != 'ok' for r in records), 'totals': {}, 'missing': {}}
    for key in fields:
        values = [r.get(key) for r in records]
        if any(v is not None and (isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0) for v in values):
            raise ValueError('Invalid nonnegative cost observation: '+key)
        result['missing'][key] = sum(v is None for v in values)
        result['totals'][key] = sum(values) if all(v is not None for v in values) else None
    result['model_versions'] = sorted({r['model_version'] for r in records if r.get('model_version')})
    result['net_utility_weights'] = 'must be preregistered; do not infer unknown costs as zero'
    return result


def routing_gate(y, predictions, rows, certificates, minimum_gain):
    if minimum_gain is None or minimum_gain <= 0:
        raise ValueError('A positive prespecified minimum routing gain is required')
    if len(certificates) != len(predictions):
        raise ValueError('Each expert needs grouped OOF certificates')
    for certs in certificates:
        if len(certs) != len(rows):
            raise ValueError('Unaligned OOF certificates')
        for row, cert in zip(rows, certs):
            group = row['meta']['group_id']
            if (row['meta']['split'] not in {'train', 'validation', 'regression'} or
                cert.get('sample_id') != row['meta']['sample_id'] or
                'fit_groups' not in cert or 'selection_groups' not in cert or
                group in set(cert['fit_groups']) | set(cert['selection_groups'])):
                raise ValueError('Gate diagnosis requires deployment-available group-OOF predictions')
    report = complementary(y, predictions, [r['meta']['root_id'] for r in rows], [r['meta']['group_id'] for r in rows])
    report['continue_gate_research'] = report['oracle_gain'] is not None and report['oracle_gain'] >= minimum_gain
    report['production_promotion_allowed'] = False
    return report
