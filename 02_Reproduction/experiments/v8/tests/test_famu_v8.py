"""Input boundary, executable certificate and counterfactual data safety checks."""
import copy
import itertools
import json
from pathlib import Path
import unittest
from src.evidence_reasoning import famu_v8 as v

ROOT=Path(__file__).resolve().parents[1]
class V8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.rows=[json.loads(s) for s in (ROOT/'training_data/famu_v8_20261005/scenes.jsonl').read_text('utf-8').splitlines()]
    def test_every_gold_certificate_and_visible_firewall(self):
        for row in self.rows:
            result=v.decode(v.answer(row,'E1_bound'),row,'E1_bound')
            self.assertEqual(result['prediction'],row['target']);self.assertEqual(result['states'],row['states'])
            changed=copy.deepcopy(row);changed['states']=['C']*4;changed['target']='WRONG';changed['bindings']={}
            self.assertEqual(v.messages(row,'E1_bound'),v.messages(changed,'E1_bound'))
            self.assertEqual(v.decode(v.answer(row,'E1_bound'),changed,'E1_bound'),result)
    def test_rule_independent_truth_table(self):
        rule={'all':['A','B',{'not':'C'}]}
        for states in itertools.product('TFU',repeat=4):
            values=[]
            domains=[(True,) if s=='T' else (False,) if s=='F' else (False,True) for s in states]
            for a,b,c,d in itertools.product(*domains):values.append(a and b and not c)
            expected='established' if all(values) else 'not_established' if not any(values) else 'unknown'
            self.assertEqual(v.compose(rule,states),expected)
    def test_withdrawing_support_changes_prediction(self):
        r=next(r for r in self.rows if r['axis']=='event' and r['states'][1]=='T')
        obj=copy.deepcopy(r['bindings']);self.assertEqual(v.decode(json.dumps(obj),r,'E1_bound')['prediction'],'established')
        obj['B']['s']=[]
        self.assertEqual(v.decode(json.dumps(obj),r,'E1_bound')['prediction'],'unknown')
        obj['C']['s']=[r['evidence'][0]['id']];obj['C']['r']=[]
        self.assertEqual(v.decode(json.dumps(obj),r,'E1_bound')['prediction'],'not_established')
    def test_invalid_citation_cannot_be_silently_unknown(self):
        r=self.rows[0];obj=copy.deepcopy(r['bindings']);obj['A']['s']=['P999']
        self.assertFalse(v.decode(json.dumps(obj),r,'E1_bound')['valid'])
        self.assertIsNone(v.decode(json.dumps(obj),r,'E1_bound')['prediction'])
    def test_pair_only_evidence_changes_and_exact_budget(self):
        for split,count in [('train',432),('dev',96),('test',144)]:self.assertEqual(sum(r['split']==split for r in self.rows),count)
        self.assertEqual(432*3+77*4,401*4)
        groups={r['pair_group'] for r in self.rows if r['axis']!='composition'}
        for group in groups:
            rr=[r for r in self.rows if r['pair_group']==group]
            self.assertEqual(len(rr),3)
            self.assertEqual({r['target'] for r in rr},{'established','not_established','unknown'})
            self.assertTrue(all(r['scope']==rr[0]['scope'] and r['fields']==rr[0]['fields'] and r['rule']==rr[0]['rule'] for r in rr))
            for x,y in zip(rr,rr[1:]):
                self.assertEqual([e['id'] for e in x['evidence']],[e['id'] for e in y['evidence']])
                self.assertEqual(sum(a['text']!=b['text'] for a,b in zip(x['evidence'],y['evidence'])),1)
    def test_formal_replay_unchanged(self):
        for old,new in [('gold_train.jsonl','replay77.jsonl'),('internal_dev.jsonl','regression9.jsonl')]:
            self.assertEqual((ROOT/'training_data/famu_v4_20261003'/old).read_bytes(),(ROOT/'training_data/famu_v8_20261005'/new).read_bytes())

if __name__=='__main__':unittest.main()
