import itertools
import pytest
from src.evidence_reasoning.risk_field_v5 import (
    STATES, bounds, certified_decision, evidence_state, refinement_pair, verify_anchor)


def test_visibility_and_task_types():
    for visibility in ['redacted', 'not_provided', 'ocr_failed', 'retrieval_miss']:
        assert evidence_state('refuted', 'text_presence', visibility, True) == 'unknown'
    assert evidence_state('unknown', 'text_presence', 'visible_blank', True) == 'refuted'
    assert evidence_state('unknown', 'actual_event', 'visible_blank', True) == 'unknown'
    assert evidence_state('unknown', 'text_presence', 'complete_absence', False) == 'unknown'
    with pytest.raises(ValueError): evidence_state(None, 'text_presence', 'visible')


def test_exception_multipath_and_conflict():
    fields = ['A', 'B', 'E']
    rule = {'all': ['A', 'B', {'not': 'E'}]}
    assert bounds(rule, fields, ['supported', 'supported', 'unknown'])['state'] == 'unknown'
    assert bounds(rule, fields, ['supported', 'refuted', 'unknown'])['state'] == 'not_established'
    assert bounds(rule, fields, ['conflict', 'unknown', 'unknown'])['state'] == 'conflict'
    assert bounds('A', ['A'], ['unknown'], ['A', {'not': 'A'}])['lower'] is None
    rule = {'any': ['A', 'B']}
    assert certified_decision(rule, ['A', 'B'], ['refuted', 'unknown'])['state'] == 'unknown'
    c = certified_decision(rule, ['A', 'B'], ['refuted', 'refuted'])['certificate']
    assert set(c['fields']) == {'A', 'B'}


def test_minimal_certificate_and_diagnostic_worlds():
    rule = {'any': ['A', {'all': ['B', 'C']}]}
    out = certified_decision(rule, ['A', 'B', 'C'], ['supported', 'refuted', 'unknown'])
    assert out['certificate']['fields'] == ['A']
    out = bounds(rule, ['A', 'B', 'C'], ['unknown', 'refuted', 'unknown'])
    assert out['witnesses']['differing_fields'] == ['A']
    assert out['witnesses']['hypothesis_only'] is True


def test_anchor_and_revision_distinction():
    assert verify_anchor({'e': '甲方日期____'}, {'evidence_id':'e','start':4,'end':8,'text':'____'})
    assert not verify_anchor({'e':'原文'}, {'evidence_id':'e','start':0,'end':2,'text':'伪造'})
    assert refinement_pair(['unknown', 'refuted'], ['supported', 'refuted'])
    assert not refinement_pair(['supported'], ['refuted'])
    assert not refinement_pair(['unknown'], ['conflict'])


def test_all_three_variable_boolean_functions_information_order():
    fields = ['A', 'B', 'C']
    worlds = list(itertools.product([False, True], repeat=3))
    for mask in range(256):
        terms = [{'all':[k if value else {'not':k} for k,value in zip(fields,w)]}
                 for i,w in enumerate(worlds) if mask & (1 << i)]
        rule = {'any':terms} if terms else False
        for state in itertools.product(STATES[:3], repeat=3):
            result = bounds(rule, fields, list(state))
            truth = [int(bool(mask & (1 << i))) for i,w in enumerate(worlds)
                     if all(s == 'unknown' or (s == 'supported') == v for s,v in zip(state,w))]
            assert [result['lower'],result['upper']] == [min(truth),max(truth)]
            for i,s in enumerate(state):
                if s != 'unknown': continue
                for value in STATES[:2]:
                    after = list(state);after[i] = value
                    new = bounds(rule, fields, after)
                    assert new['lower'] >= result['lower'] and new['upper'] <= result['upper']


def test_weighted_model_count_gradient_and_masked_annotations():
    torch = pytest.importorskip('torch')
    from src.evidence_reasoning.risk_field_v5 import soft_semantics,state_table,information_loss
    assignments,labels = state_table({'all':['A',{'not':'E'}]},['A','E'])
    a=torch.tensor(assignments);l=torch.tensor(labels)
    z=torch.randn(2,2,4,requires_grad=True);p=soft_semantics(z.softmax(-1),a,l)
    assert torch.allclose(p.sum(-1),torch.ones(2),atol=1e-6)
    loss=-p[0,0].log()+information_loss(p[:1],p[1:]);loss.backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum()>0
