"""Auditable local span supervision, shared by linear and neural extractors."""
from collections import Counter, defaultdict
import re

PAIR_LABELS = ("irrelevant", "support", "refute")


def validate_span_annotations(payload, annotations):
    if not isinstance(annotations, list):
        raise ValueError("Span annotations must be a list")
    evidence = {e['id']: e for e in payload['evidence']}
    conditions = {c['id'] for c in payload['conditions']}
    for span in annotations:
        if not isinstance(span, dict):
            raise ValueError("Invalid span")
        e = evidence.get(span.get('evidence_id'))
        a, b = span.get('start'), span.get('end')
        if (e is None or span.get('condition') not in conditions or type(a) is not int or type(b) is not int
                or not 0 <= a < b <= len(e['text']) or span.get('quote') != e['text'][a:b]):
            raise ValueError("Span offsets and quote must exactly match original evidence")
        if span.get('relation') not in PAIR_LABELS:
            raise ValueError("Span requires support/refute/irrelevant annotation")
        if span.get('scope_id') is not None and not isinstance(span['scope_id'], str):
            raise ValueError("Invalid scope id")
    return annotations


def scoped_spans(text, width=180):
    """Keep clause/exception sentences intact; incomplete windows require review.

    This is syntactic candidate generation, not a prediction of legal scope.
    Exceptions introduced after a sentence boundary remain with that sentence.
    """
    from .text import windows
    units = []
    for match in re.finditer(r'[^。\n]+[。\n]?|[。\n]', text):
        if units and re.match(r'\s*(?:但|除|除外|但是|然而)', match.group()):
            units[-1][1] = match.end()
        else:
            units.append([match.start(), match.end()])
    result = []
    for a, b in units:
        for span in windows(text[a:b], width=width, overlap=min(32, width-1)):
            result.append({**span, 'start': a+span['start'], 'end': a+span['end'],
                           'scope_start': a, 'scope_end': b, 'scope_complete': b-a <= width})
    return result


def annotated_pairs(rows, return_audit=False):
    from .provenance import visible_evidence
    result, seen, labels_seen = [], set(), {}
    audit = Counter(records=len(rows), candidate_pairs=0, labeled_candidates=0, unavailable_pairs=0,
                    unlabeled_pairs=0, long_without_span=0, same_root_duplicates=0, kept_span=0, kept_whole=0, kept_joint=0)
    for row in rows:
        x, target, meta = row['input'], row.get('targets', {}), row['meta']
        annotations = validate_span_annotations(x, target.get('spans', []))
        visible = {e['id']: e for e in visible_evidence(x)}
        all_evidence = {e['id']: e for e in x['evidence']}
        by_pair = defaultdict(list)
        for span in annotations:
            by_pair[(span['condition'], span['evidence_id'])].append(span)
        def add(condition, evidence, quote, label, start, end, kind):
            audit['labeled_candidates'] += 1
            # A relation classifier sees query/span/scope, not source IDs. The
            # same pair across withdrawal views must not gain training mass.
            pair_key = (meta['root_id'], condition['text'], quote, evidence.get('scope_id'))
            if pair_key in labels_seen and labels_seen[pair_key] != label:
                raise ValueError('Conflicting same-root relation labels require adjudication')
            labels_seen[pair_key] = label
            key = (*pair_key, label)
            if key in seen:
                audit['same_root_duplicates'] += 1
                return
            seen.add(key)
            result.append({'query': condition['text'], 'text': quote, 'label': PAIR_LABELS.index(label),
                'root_id': meta['root_id'], 'group_id': meta['group_id'], 'split': meta['split'],
                'evidence_id': evidence['id'], 'start': start, 'end': end, 'annotation_kind': kind})
            audit['kept_'+kind] += 1
        for c in x['conditions']:
            for eid, e in all_evidence.items():
                audit['candidate_pairs'] += 1
                if eid not in visible:
                    audit['unavailable_pairs'] += 1
                    continue
                e = visible[eid]
                spans = by_pair[(c['id'], eid)]
                if spans:
                    for span in spans:
                        add(c, e, span['quote'], span['relation'], span['start'], span['end'], 'span')
                    continue
                label = target.get('evidence', {}).get(c['id']+'|'+eid)
                if label not in PAIR_LABELS:
                    audit['unlabeled_pairs'] += 1
                elif len(e['text']) > 180:
                    audit['long_without_span'] += 1
                else:
                    add(c, e, e['text'], label, 0, len(e['text']), 'whole')
        for group in x.get('candidate_groups', []):
            label = target.get('proof_groups', {}).get(group['id'])
            if label not in PAIR_LABELS or not set(group['evidence_ids']) <= set(visible):
                continue
            items = [visible[eid] for eid in group['evidence_ids']]
            quote = '\n'.join(e['text'] for e in items)
            c = next(c for c in x['conditions'] if c['id'] == group['condition'])
            # Explicit group targets supervise the whole combination, not each member.
            key = (meta['root_id'], c['text'], tuple(sorted(e['text'] for e in items)), label)
            if key in seen:
                audit['same_root_duplicates'] += 1
                continue
            seen.add(key)
            result.append({'query': c['text'], 'text': quote, 'label': PAIR_LABELS.index(label),
                'root_id': meta['root_id'], 'group_id': meta['group_id'], 'split': meta['split'],
                'annotation_kind': 'joint', 'evidence_ids': list(group['evidence_ids'])})
            audit['kept_joint'] += 1
    counts = Counter(p['root_id'] for p in result)
    for p in result:
        p['weight'] = 1. / counts[p['root_id']]
    audit['kept'] = len(result)
    audit['roots'] = len(counts)
    return (result, dict(audit)) if return_audit else result
