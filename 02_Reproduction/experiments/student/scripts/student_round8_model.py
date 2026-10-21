"""Finite-action contextual-bandit objective; not PPO/GRPO generation."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.evidence_reasoning.semantic_reward_student import WIDTH

class Head(nn.Module):
    def __init__(self,question):
        super().__init__();self.question=nn.Linear(4096,4)
        self.question.load_state_dict({k:torch.from_numpy(question['question.'+k]) for k in ['weight','bias']})
        # Continue question semantics with the same auxiliary CE in every arm.
        self.hidden=nn.Linear(WIDTH,64);self.evidence=nn.Linear(64,12)
    def components(self,q,x):return self.question(q),self.evidence(torch.relu(self.hidden(x))).reshape(-1,4,3)

def teacher_reward(records,pairs):
    rewards=np.zeros((len(pairs),3),np.float32);mask=np.zeros(len(pairs),np.float32)
    for i,p in enumerate(pairs):
        r=records.get(p['id'])
        if r is None:continue
        assert r['gold']==p['label']
        v=np.asarray(r['mean_scores'],np.float64);w=np.asarray(r['sum_scores'],np.float64)
        assert v.shape==w.shape==(3,) and np.isfinite(v).all() and np.isfinite(w).all()
        if r['accepted'] and int(v.argmax())==int(w.argmax())==p['label'] and np.ptp(v)>=.05:
            q=np.exp((v-v.max())/.25);q/=q.sum();rewards[i]=q-1/3;mask[i]=1
    counts={str(c):int(sum(mask[i] for i,p in enumerate(pairs) if p['label']==c)) for c in range(3)}
    eligible=sum(counts.values())>=384 and min(counts.values())>=32
    if not eligible:rewards[:]=0;mask[:]=0
    return rewards,mask,{'eligible':eligible,'accepted_by_class_before_global_gate':counts,'candidate_scores_are_calibrated_probabilities':False,'new_teacher_inferences':0,'reused_frozen_teacher_scores':len(records)}

def objective(logits,labels,arm,reference=None,teacher=None):
    ce=F.cross_entropy(logits,labels);p=logits.softmax(-1)
    if arm.endswith('SFT') or arm=='WARM':return ce,{'ce':float(ce.detach()),'expected_reward':0.,'teacher_reward':0.,'kl':0.}
    # Gold0=irrelevant; false strong actions receive -2. Missing positive=-.5.
    table=logits.new_tensor([[1,-2,-2],[-.5,1,-2],[-.5,-2,1]])
    er=(p*table[labels]).sum(-1).mean();kl=(p*(logits.log_softmax(-1)-reference.log_softmax(-1))).sum(-1).mean()
    tr=(p*teacher).sum(-1).mean() if arm=='EVID_MRL' else ce*0
    loss=ce-.5*er+.02*kl-.2*tr
    return loss,{'ce':float(ce.detach()),'expected_reward':float(er.detach()),'teacher_reward':float(tr.detach()),'kl':float(kl.detach())}
