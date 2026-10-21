"""Comparable development diagnostics; never mistaken for independent release approval."""
from .question_evidence_models import TASK_CLASSES


def compare_development_candidate(candidate, reference, task):
    from scripts.train_question_evidence_models import evaluate
    if task not in TASK_CLASSES:
        raise ValueError('Unsupported acceptance target')
    def keyed(rows):
        result = {}
        for row in rows:
            if row['sample_id'] in result:
                raise ValueError('Repeated prediction ID')
            if row['actual'] not in TASK_CLASSES[task] or row['predicted'] not in TASK_CLASSES[task]:
                raise ValueError('Invalid prediction label')
            result[row['sample_id']] = (row['actual'], row['group_id'])
        return result
    if not candidate or keyed(candidate) != keyed(reference):
        raise ValueError('Candidate and reference must evaluate identical labeled source-group cases')
    current, previous = evaluate(candidate, task), evaluate(reference, task)
    reasons = []
    for metric in ('accuracy', 'macro_f1_fixed_classes'):
        if current['row_metrics'][metric] < previous['row_metrics'][metric]-1e-12:
            reasons.append(metric+'_regressed')
    for label, metrics in current['row_metrics']['per_class'].items():
        if metrics['support'] and metrics['recall'] < previous['row_metrics']['per_class'][label]['recall']-1e-12:
            reasons.append(label+'_recall_regressed')
    if current['health']['single_class_collapse']:
        reasons.append('single_class_collapse')
    if task == 'severity' and current['high_or_critical_underestimate_rate'] is not None:
        previous_rate = previous['high_or_critical_underestimate_rate']
        if previous_rate is not None and current['high_or_critical_underestimate_rate'] > previous_rate+1e-12:
            reasons.append('high_or_critical_underestimate_increased')
    return {'development_nonregression_passed': not reasons, 'regressions': reasons,
            'accuracy_delta': current['row_metrics']['accuracy']-previous['row_metrics']['accuracy'],
            'missing_target_classes': current['health']['unobserved_target_classes'],
            'observed_classes_never_predicted': current['health']['observed_classes_never_predicted'],
            'promotion_allowed': False, 'independent_release_evaluation_required': True,
            'meaning': 'Known development comparison only; passing does not establish statistically reliable improvement or release quality'}
