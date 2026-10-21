import copy
import json
import torch
from scripts.build_famu_v5 import synthetic
from src.evidence_reasoning.risk_field_v5 import DECISIONS, STATES, refinement_pair, verify_anchor
from src.evidence_reasoning.risk_field_v5_learning import (public_prompt, tensors, tables, Specialists,
    balanced_weights, pair_objectives, predictions)


def test_synthetic_split_and_intervention_contract():
    rows, pairs = synthetic()
    byid = {r['id']: r for r in rows}
    families = {s: {r['family'] for r in rows if r['split'] == s} for s in ['train', 'dev', 'test']}
    assert [len(families[s]) for s in families] == [64, 16, 16]
    assert not families['train'] & families['test']
    for p in pairs:
        a, b = byid[p['before']], byid[p['after']]
        assert a['split'] == b['split']
        if p['kind'] == 'information':
            assert refinement_pair(a['states'], b['states'])
        elif p['kind'] == 'equivalent':
            assert a['states'] == b['states']
    for r in rows:
        evidence = {e['id']: e['text'] for e in r['visible']['evidence']}
        assert all(verify_anchor(evidence, a) for aa in r['anchors'].values() for a in aa)


def test_target_firewall():
    rows, _ = synthetic()
    r = rows[0]; original = public_prompt(r)
    modified = copy.deepcopy(r)
    modified.update(target='SENTINEL', states=['SENTINEL'], split='SENTINEL', anchors={'leak': 'SENTINEL'})
    assert public_prompt(modified) == original
    assert 'SENTINEL' not in original
    assert 'label_authority' not in original


def test_all_heads_and_boundary_gradient_paths():
    rows, _ = synthetic(); rows = rows[:8]
    torch.manual_seed(1)
    x = torch.randn(len(rows), 5, 12)
    meta, tt = tensors(rows, 'cpu'), tables(rows, 'cpu')
    assert torch.allclose(balanced_weights(rows, 'cpu').sum(), torch.tensor(1.))
    for arm in ['M0_direct', 'M1_cbm', 'M2_cbm_fact', 'M3_boundary', 'M4_dual']:
        model = Specialists(12)
        field, out = model(x, meta, arm, tt)
        assert torch.allclose(out.sum(1), torch.ones(len(rows)), atol=1e-6)
        loss = -out.clamp_min(1e-8).log().gather(1, meta['labels'][:, None]).mean()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if arm in ['M3_boundary', 'M4_dual']:
            assert model.direct[0].weight.grad is None and model.concepts[0].weight.grad is None
            assert model.field[0].weight.grad is not None
        vv, ss, _ = predictions(rows, model, x, meta, arm, tt)
        assert len(vv) == len(rows)


def test_information_loss_does_not_reward_fact_flip():
    # Supported -> refuted is a factual revision, not an information refinement.
    rows, _ = synthetic()
    assert not refinement_pair(rows[0]['states'], rows[1]['states'])
    decision = torch.tensor([[.1, .1, .8, 0.], [.8, .1, .1, 0.]], requires_grad=True)
    fields = torch.zeros(2, 4, 4, requires_grad=True)
    ll = pair_objectives(decision, fields, {'information': [(0, 1)], 'fact': [], 'equivalent': []}, torch.tensor([2, 0]))
    assert abs(float(ll['information'].detach())) < 1e-6
