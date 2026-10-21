"""Local evidence and certificate metrics with explicit eligible denominators."""
import numpy as np
from .metrics import weights
from .provenance import visible_evidence, evidence_signature


def evidence_metrics(rows, outputs):
    if len(rows) != len(outputs):
        raise ValueError('Unaligned extraction outputs')
    roots, groups = ([r['meta'][k] for r in rows] for k in ('root_id', 'group_id'))
    w = weights(roots, groups)
    tp = fp = fn = citations = invalid_citations = 0.
    local_gold_rows = span_gold_rows = 0
    relation_counts = np.zeros((3,4))
    relations = ('irrelevant','support','refute')
    for row, output, weight in zip(rows, outputs, w):
        visible_records = {e['id']: e for e in visible_evidence(row['input'])}
        visible = set(visible_records)
        evidence = {e['id']: e for e in row['input']['evidence']}
        predicted = output.get('evidence_predictions', [])
        for p in predicted:
            if p.get('accepted'):
                ids = p.get('evidence_ids') or ([p['evidence_id']] if p.get('evidence_id') else [])
                citations += weight*len(ids)
                invalid_citations += weight*sum(eid not in visible for eid in ids)
        gold = row['targets'].get('spans', [])
        if gold:
            span_gold_rows += 1
            gold_chars, predicted_chars = set(), set()
            for s in gold:
                if s['relation'] != 'irrelevant':
                    gold_chars.update((s['condition'], s['evidence_id'], s['relation'], i) for i in range(s['start'],s['end']))
            for p in predicted:
                if p.get('accepted') and p.get('evidence_id') and isinstance(p.get('start'),int) and isinstance(p.get('end'),int):
                    label = max(p['probabilities'], key=p['probabilities'].get)
                    predicted_chars.update((p['condition'], p['evidence_id'], label, i) for i in range(p['start'],p['end']))
            tp += weight*len(gold_chars & predicted_chars)
            fp += weight*len(predicted_chars-gold_chars)
            fn += weight*len(gold_chars-predicted_chars)
        gold_pairs = row['targets'].get('evidence', {})
        if gold_pairs:
            local_gold_rows += 1
        unique_pairs = {}
        for key, label in gold_pairs.items():
            cid, eid = key.split('|',1)
            if eid not in visible:
                continue
            signature = evidence_signature(visible_records[eid])
            unique = (cid, signature)
            if unique in unique_pairs and unique_pairs[unique][1] != label:
                raise ValueError('Equivalent evidence has conflicting relation targets')
            unique_pairs[unique] = (eid, label)
        for (cid, signature), (eid, label) in unique_pairs.items():
            # Compare only exact annotated evidence; never use a full-text label for arbitrary windows.
            candidates = [p for p in predicted if p['condition'] == cid and p.get('evidence_id') in visible_records
                          and evidence_signature(visible_records[p['evidence_id']]) == signature
                          and p.get('start') == 0 and p.get('end') == len(evidence[p['evidence_id']]['text'])]
            answer = max(candidates[0]['probabilities'], key=candidates[0]['probabilities'].get) if candidates else None
            relation_counts[relations.index(label), relations.index(answer) if answer else 3] += weight/max(1,len(unique_pairs))
    f1 = []
    for i in range(3):
        true = relation_counts[i,i]; denom = relation_counts[i].sum()+relation_counts[:,i].sum()
        f1.append(float(2*true/denom) if denom else 0.)
    return {'local_gold_rows': local_gold_rows, 'span_gold_rows': span_gold_rows,
            'relation_macro_f1': float(np.mean(f1)) if local_gold_rows else None, 'relation_confusion': relation_counts.tolist(),
            'span_character_f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'invalid_source_citation_rate': invalid_citations/citations if citations else None,
            'weighted_citations': citations, 'span_scope': 'only explicitly annotated spans; absent annotations are not negative labels'}


def certificate_metrics(outputs, truth):
    if len(outputs) != len(truth): raise ValueError('Unaligned certificate results')
    certified = [(o, y) for o, y in zip(outputs, truth) if o.get('certificate_verified') is not None]
    return {'certificates_checked': len(certified),
            'formal_validity': sum(o['certificate_verified'] is True for o,_ in certified)/len(certified) if certified else None,
            'label_accuracy_on_certified': sum(o.get('proposition_label') == y for o,y in certified)/len(certified) if certified else None,
            'legal_truth_validated_by_certificate': False}
