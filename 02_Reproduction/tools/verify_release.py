"""Verify release SHA256, JSON syntax and real-data reference integrity. Standard library only."""
from pathlib import Path
import json,hashlib,ast,re
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
manifest=json.loads((ROOT/'MANIFEST_SHA256.json').read_text(encoding='utf-8'))
bad=[]
expected={row['path'] for row in manifest['files']}
assert len(expected)==len(manifest['files']),'duplicate manifest paths'
actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()}
assert actual==expected|{'MANIFEST_SHA256.json'},('unexpected or missing files',sorted(actual^expected^{'MANIFEST_SHA256.json'}))
for row in manifest['files']:
    p=ROOT/row['path']
    assert p.resolve().is_relative_to(ROOT.resolve()),'path outside release'
    if not p.is_file() or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:bad.append(row['path'])
assert not bad,('missing or changed files',bad)
parsed=0
for p in ROOT.rglob('*'):
    if not p.is_file():continue
    assert p.suffix.lower() not in {'.pt','.npz','.npy','.safetensors','.docx','.pdf','.xlsx','.zip','.gz','.pkl','.bin','.pyc'},('unexpected original/binary',p)
    if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p));parsed+=1
    elif p.suffix=='.json':json.loads(p.read_text(encoding='utf-8-sig'));parsed+=1
    elif p.suffix=='.jsonl':rows(p);parsed+=1
public={r['question_id']:r for r in rows(ROOT/'data/real/public_inputs.jsonl')}
keys=rows(ROOT/'data/real/gold_keys.jsonl')
for k in keys:
    r=public[k['question_id']];ids={e['evidence_id'] for e in r['evidence']}
    assert k['input_sha256']==r['input_sha256']
    for f in k['fields']:
        for option in f['sufficient_evidence_sets']:assert set(option)<=ids
for path in ['experiments/v5/training_data/famu_v5_20261004/examples.jsonl','experiments/v5/training_data/famu_v5_20261004_stress/examples.jsonl']:
    for r in rows(ROOT/path):
        evidence={e['id']:e['text'] for e in r['visible']['evidence']}
        for aa in r.get('anchors',{}).values():
            for a in aa:assert evidence[a['evidence_id']][a['start']:a['end']]==a['text']
audit=json.loads((ROOT/'experiments/real_eval/config/student_input_adapter_audit.json').read_text(encoding='utf-8'))
student_anchors=0
def check_student(x,windows,window=None):
    global student_anchors
    if isinstance(x,list):
        for v in x:check_student(v,windows,window)
    elif isinstance(x,dict):
        current=x.get('evidence_id',window)
        if 'text' in x and 'start' in x and 'end' in x:
            assert current in windows
            assert windows[current]['text'][x['start']:x['end']]==x['text']
            student_anchors+=1
        for k,v in x.items():
            if k=='selected_evidence':
                for wid,spans in v.items():check_student(spans,windows,wid)
            else:check_student(v,windows,current)
for qid,row in audit['rows'].items():
    texts={e['evidence_id']:e['text'] for e in public[qid]['evidence']}
    for win in row['windows'].values():
        assert win['original_evidence_id'] in texts
        if 'start' in win:assert texts[win['original_evidence_id']][win['start']:win['end']]==win['text']
for p in (ROOT/'experiments/real_eval/student_runs').glob('*/raw.jsonl'):
    for row in rows(p):check_student(row,audit['rows'][row['question_id']]['windows'])
provenance=json.loads((ROOT/'manifests/source_fingerprints.json').read_text(encoding='utf-8'))
for item in provenance['files']:
    p=ROOT/item['path']
    assert sha(p)==item['released_sha256'],item['path']
    assert len(item['original_sha256'])==64
print(json.dumps({'hashes_passed':len(manifest['files']),'syntax_files_checked':parsed,'real_questions':len(public),'reference_and_anchor_checks':True,'student_anchors_checked':student_anchors,'source_fingerprints_checked':len(provenance['files']),'weights_or_original_documents_present':False},ensure_ascii=False))
