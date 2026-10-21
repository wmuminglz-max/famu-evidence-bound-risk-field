"""Strict structured-answer scorer; provisional labels require explicit opt-in.

The built-in checks test software only. They are not model experiment results.
"""
from pathlib import Path
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
import argparse, datetime, hashlib, json

ROOT = Path(__file__).resolve().parent

def no_duplicates(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f'Duplicate JSON key: {k}')
        out[k] = v
    return out

def read_rows(path):
    return [json.loads(line, object_pairs_hook=no_duplicates)
            for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]

def index(rows, field='question_id'):
    out = {}
    for row in rows:
        key = row[field]
        if key in out:
            raise ValueError(f'Duplicate {field}: {key}')
        out[key] = row
    return out

def input_hash(row):
    return hashlib.sha256(json.dumps({k:v for k,v in row.items() if k != 'input_sha256'},
        ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def normalized(value, kind):
    if kind == 'number':
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ValueError('number required')
        # Exact decimal comparison. No undisclosed rounding or acceptance tolerance.
        n = Decimal(str(value).replace(',', '').strip())
        if not n.is_finite():
            raise ValueError('finite number required')
        return n
    if kind == 'set':
        if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
            raise ValueError('string array required')
        values = [v.strip() for v in value]
        if len(values) != len(set(values)):
            raise ValueError('duplicate set item')
        return frozenset(values)
    if not isinstance(value, str):
        raise ValueError('string required')
    value = value.strip()
    if kind == 'state' and value not in {'T','F','U','C'}:
        raise ValueError('T/F/U/C required')
    if kind == 'date':
        if len(value) != 10 or datetime.date.fromisoformat(value).isoformat() != value:
            raise ValueError('YYYY-MM-DD required')
    return value

def matches(value, field, formal=False):
    expected = field['final_value'] if formal else field['proposed_value']
    try:
        value = normalized(value, field['type'])
        return any(value == normalized(v, field['type'])
                   for v in [expected] + field.get('accepted_values', []))
    except (ValueError, TypeError, InvalidOperation):
        return False

def evaluate(public, keys, predictions, allow_provisional=False):
    pub, keymap, pred = index(public), index(keys), index(predictions)
    unknown = set(pred) - set(keymap)
    if unknown:
        raise ValueError(f'Predictions outside selected question set: {sorted(unknown)}')
    formal = all(k.get('formal_evaluation_ready') is True for k in keys)
    if not formal and not allow_provisional:
        raise ValueError('标签未获专家确认。调试可显式使用 --allow-provisional；不能报告正式测试成绩。')
    if formal:
        for k in keys:
            if not k.get('expert_confirmation') or (k['split']=='test_reserved' and not k.get('independent_review')):
                raise ValueError('Formal key is missing human review provenance')
            if any('final_value' not in f for f in k['fields']):
                raise ValueError('Formal key is missing final values')
    results, confusion = [], Counter()
    for qid, k in keymap.items():
        p = pub[qid]
        if p['input_sha256'] != input_hash(p) or k['input_sha256'] != p['input_sha256']:
            raise ValueError(f'{qid}: key/input hash mismatch')
        prediction = pred.get(qid)
        answers = prediction.get('answers', {}) if prediction else {}
        if not isinstance(answers, dict):
            raise ValueError(f'{qid}: answers must be an object')
        if prediction and prediction.get('input_sha256') != p['input_sha256']:
            raise ValueError(f'{qid}: prediction input hash mismatch')
        extra = set(answers) - {f['key'] for f in k['fields']}
        if extra:
            raise ValueError(f'{qid}: unknown fields: {sorted(extra)}')
        visible = {e['evidence_id'] for e in p['evidence']}
        rows = []
        for field in k['fields']:
            answer = answers.get(field['key'])
            provided = isinstance(answer, dict) and 'value' in answer
            answer = answer if isinstance(answer, dict) else {}
            ok = provided and matches(answer.get('value'), field, formal)
            ids = answer.get('evidence_ids', [])
            citations_valid = (isinstance(ids, list) and all(isinstance(v,str) for v in ids)
                               and len(ids)==len(set(ids)) and bool(ids) and set(ids)<=visible)
            sufficient = bool(citations_valid and any(set(option)<=set(ids)
                for option in field['sufficient_evidence_sets']))
            # Extra in-scope citations do not invalidate sufficiency; report their cost separately.
            minimal_count = min((len(set(ids)-set(option)) for option in field['sufficient_evidence_sets']
                                 if citations_valid and set(option)<=set(ids)), default=None)
            reason_present = isinstance(answer.get('brief_reason'),str) and bool(answer['brief_reason'].strip())
            rows.append({'field':field['key'], 'provided':provided, 'value_exact':bool(ok),
                         'citations_valid':bool(citations_valid), 'evidence_sufficient':sufficient,
                         'value_and_evidence':bool(ok and sufficient), 'reason_present':reason_present,
                         'extra_citations_over_sufficient_set':minimal_count})
            if field['type']=='state':
                expected = field['final_value'] if formal else field['proposed_value']
                actual = answer.get('value')
                if isinstance(actual,str):actual=actual.strip()
                actual = actual if isinstance(actual,str) and actual in {'T','F','U','C'} else 'MISSING/INVALID'
                confusion[(expected,actual)] += 1
        results.append({'question_id':qid,'split':k['split'],'task_type':k['task_type'],
                        'source_family':k['source_family'],'fields':rows})

    def aggregate(tasks):
        fields = [f for t in tasks for f in t['fields']]
        n = len(fields)
        return {'questions':len(tasks),'fields':n,
                'coverage':sum(f['provided'] for f in fields)/n,
                'field_value_exact':sum(f['value_exact'] for f in fields)/n,
                'question_macro_value_exact':sum(sum(f['value_exact'] for f in t['fields'])/len(t['fields']) for t in tasks)/len(tasks),
                'question_all_values_correct':sum(all(f['value_exact'] for f in t['fields']) for t in tasks)/len(tasks),
                'field_value_and_evidence':sum(f['value_and_evidence'] for f in fields)/n,
                'reason_coverage_only':sum(f['reason_present'] for f in fields)/n}

    strata = defaultdict(list)
    for t in results:
        strata[(t['split'],t['task_type'])].append(t)
    families = defaultdict(list)
    for t in results:
        families[(t['split'],t['task_type'],t['source_family'])].append(t)
    return {'status':'formal' if formal else 'PROVISIONAL_DEBUG_NOT_PAPER_RESULT',
            'reason_quality':'not automatically scored; needs blinded human review',
            'groups':[{'split':s,'task_type':t,**aggregate(rows)} for (s,t),rows in sorted(strata.items())],
            'families':[{'split':s,'task_type':t,'source_family':f,**aggregate(rows)} for (s,t,f),rows in sorted(families.items())],
            'state_confusion':[{'gold':g,'predicted':p,'count':n} for (g,p),n in sorted(confusion.items())],
            'details':results}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions',type=Path)
    parser.add_argument('--key',type=Path,default=ROOT/'02_最终评分/最终评分键_v2.jsonl')
    parser.add_argument('--inputs',type=Path,default=ROOT/'01_模型输入/模型输入_28题_v2.jsonl')
    parser.add_argument('--split',choices=['development','test_reserved','all'],default='development')
    parser.add_argument('--allow-provisional',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    keys=read_rows(args.key)
    if args.split!='all':keys=[k for k in keys if k['split']==args.split]
    report=evaluate(read_rows(args.inputs),keys,read_rows(args.predictions),args.allow_provisional)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(report['status'])

if __name__=='__main__':main()
