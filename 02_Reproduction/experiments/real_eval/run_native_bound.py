"""Native V8 binding interface on the same visible evidence and state questions."""
import json,hashlib,sys,time
from pathlib import Path
ROOT=Path(__file__).parent
sys.path.insert(0,'/workspace/famu_v8_20261005')
import run_lm_eval as runner
from src.evidence_reasoning.famu_v8 import messages,decode
from src.evidence_reasoning.risk_field_v5 import certified_decision,STATES

def adapt(row):
 qs=[q for q in row['questions'] if q['answer_type']=='state'];assert 1<=len(qs)<=4
 # The saved native interface requires exactly A/B/C/D. Unused fields repeat A
 # and are discarded; no answer or external premise is inserted.
 mapped={letter:(qs[i] if i<len(qs) else qs[0])['prompt'] for i,letter in enumerate('ABCD')}
 return {'scope':{k:row[k] for k in ['roles','scope','time_scope','original_question']},'fields':mapped,'rule':'A','evidence':[{'id':e['evidence_id'],'text':e['text']} for e in row['evidence']]}
def prompt(tok,row):
 return tok.apply_chat_template(messages(adapt(row),'E1_bound'),tokenize=False,add_generation_prompt=True,enable_thinking=False)
def parse(raw,row):
 raw=raw.strip()
 if raw.startswith('```') and raw.splitlines()[-1].strip()=='```':raw='\n'.join(raw.splitlines()[1:-1])
 obj=decode(raw,adapt(row),'E1_bound');answers={}
 if obj['valid']:
  qs=[q for q in row['questions'] if q['answer_type']=='state']
  for i,q in enumerate(qs):
   state=obj['states'][i];binding=obj['bindings']['ABCD'[i]]
   cert=certified_decision('field',['field'],[STATES['TFUC'.index(state)]])
   translated={'established':'T','not_established':'F','unknown':'U','conflict':'C'}[cert['state']]
   assert translated==state
   answers[q['key']]={'value':translated,'evidence_ids':list(dict.fromkeys(binding['s']+binding['r'])),'brief_reason':'V8原生接口预测支持/反驳证据，经既有状态聚合和规则引擎输出；原始绑定见raw。','rule_certificate':cert}
 return {'question_id':row['question_id'],'input_sha256':row['input_sha256'],'answers':answers},None if obj['valid'] else 'invalid native binding JSON'
if __name__=='__main__':
 rows=[json.loads(s) for s in (ROOT/'public_inputs.jsonl').read_text().splitlines()]
 projection=[{'question_id':r['question_id'],'original_input_sha256':r['input_sha256'],'native_input':adapt(r),'mapping':dict(zip('ABCD',[q['key'] for q in r['questions'] if q['answer_type']=='state']))} for r in rows]
 runner.save(ROOT/'native_projection.json',projection)
 runner.save(ROOT/'NATIVE_PROTOCOL_before_run.json',{'created_unix':time.time(),'script_sha256':runner.sha(__file__),'projection_sha256':runner.sha(ROOT/'native_projection.json'),'purpose':'Native deployed V8 binding and existing symbolic engine; separate from uniform JSON end-to-end task','seeds':[17,29,43],'selection':'All three frozen E1 endpoints, no current-data tuning','shared_input':'Same evidence text, roles, display/time scope and state question wording. Original-question context preserved. Native output instructions replace generic answer protocol.','unscored_padding':'Repeat first real proposition to fill unused B/C/D, then discard padding outputs','scope':'46 state fields;27other evidence-test nonstate fields unsupported; full87 coverage reported','rule':'Each requested state itself is an identity proposition; native bindings s/r map TFUC and existing certified_decision runs on each field. This benchmark does not measure composite AND/OR rule correctness.'})
 runner.prompt=prompt;runner.parse=parse;runner.main()
