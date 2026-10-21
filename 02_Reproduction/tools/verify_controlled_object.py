"""Check a separately obtained controlled file against its recorded SHA256."""
from pathlib import Path
import argparse,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('object_id',help='object_id from models/controlled_objects.json')
p.add_argument('file',type=Path,help='separately obtained original file')
a=p.parse_args()
items=json.loads((ROOT/'models/controlled_objects.json').read_text(encoding='utf-8'))['objects']
item=next((x for x in items if x['object_id']==a.object_id),None)
if item is None:raise SystemExit('Unknown object_id')
h=hashlib.sha256()
with a.file.open('rb') as f:
    for b in iter(lambda:f.read(8<<20),b''):h.update(b)
ok=h.hexdigest()==item['sha256'] and a.file.stat().st_size==item['bytes']
print(json.dumps({'object_id':a.object_id,'matches_recorded_original':ok,'sha256':h.hexdigest(),'reference_verification':item['verification']},ensure_ascii=False))
raise SystemExit(0 if ok else 1)
