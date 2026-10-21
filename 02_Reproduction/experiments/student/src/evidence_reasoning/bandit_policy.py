"""Clipped PPO for a finite categorical contextual bandit, with a fixed reference.

Only decision heads update. A grammar- or human-grounded reward table supplies
the sampled-action reward and an exact baseline; no self-confidence reward.
"""
import copy
import torch
from torch.nn import functional as F


def distribution(head, features, supported):
    if not supported or not any(supported):raise ValueError('At least one trained action required')
    mask=torch.as_tensor(supported,dtype=torch.bool,device=features.device)
    return torch.softmax(head(features).masked_fill(~mask,-1e9),dim=-1)


def clipped_objective(new_logp,old_logp,advantage,weights,clip=.2):
    ratio=(new_logp-old_logp).exp()
    surrogate=torch.minimum(ratio*advantage,ratio.clamp(1-clip,1+clip)*advantage)
    return -(surrogate*weights).sum(),((ratio-1).abs()>clip).float().mean()


def fit_heads(initial,tasks,method='ppo',rounds=3,rollouts=5,epochs=4,seed=20260930,on_round=None):
    if method not in {'ppo','sft'} or not 1<=rounds<=10 or not 1<=rollouts<=100 or not 1<=epochs<=10:
        raise ValueError('Invalid finite training schedule')
    torch.manual_seed(seed)
    heads=copy.deepcopy(initial);reference=copy.deepcopy(initial).eval()
    for p in reference.parameters():p.requires_grad_(False)
    optimizer=torch.optim.AdamW(heads.parameters(),lr=.0003,weight_decay=.01)
    history=[]
    for round_index in range(rounds):
        records=[]
        for rollout in range(rollouts):
            batches={}
            with torch.no_grad():
                for task,data in tasks.items():
                    # Challenger selects the invariant version with lowest
                    # expected externally grounded reward on TRAINING cases.
                    p=distribution(heads[task],data['features'],data['supported'])
                    worst=(p*data['rewards'][:,None,:]).sum(-1).argmin(1)
                    x=data['features'][torch.arange(len(worst)),worst]
                    old=distribution(heads[task],x,data['supported'])
                    ref=distribution(reference[task],x,data['supported'])
                    action=torch.multinomial(old,4,replacement=True)
                    rewards=data['rewards'].gather(1,action)
                    advantage=rewards-(old*data['rewards']).sum(-1,keepdim=True)
                    batches[task]=(x,old.clamp_min(1e-12).log().gather(1,action),ref,action,advantage,worst)
            for epoch in range(epochs):
                optimizer.zero_grad();loss=0;details=[]
                for task,data in tasks.items():
                    x,oldlog,ref,action,advantage,worst=batches[task]
                    p=distribution(heads[task],x,data['supported']);logp=p.clamp_min(1e-12).log()
                    weights=data['weights'];ce=-(weights*logp.gather(1,data['targets'][:,None]).squeeze(1)).sum()
                    kl=(weights*(p*(logp-ref.clamp_min(1e-12).log())).sum(-1)).sum()
                    entropy=-(weights*(p*logp).sum(-1)).sum()
                    objective,clipped=clipped_objective(logp.gather(1,action),oldlog,advantage,weights[:,None]/action.shape[1])
                    loss=loss+(objective+.5*ce+.05*kl-.01*entropy if method=='ppo' else ce)/len(tasks)
                    details.append({'task':task,'CE':float(ce.detach()),'reference_KL':float(kl.detach()),
                        'expected_reward':float((weights*(p*data['rewards']).sum(-1)).sum().detach()),
                        'clip_fraction':float(clipped),'adversarial_variant_count':int(worst.sum()),
                        'sampled_action_counts':torch.bincount(action.flatten(),minlength=p.shape[1]).tolist()})
                if not torch.isfinite(loss):raise ValueError('Nonfinite policy loss; stop before saving candidate')
                loss.backward();torch.nn.utils.clip_grad_norm_(heads.parameters(),1);optimizer.step()
                records.append({'rollout':rollout+1,'epoch':epoch+1,'loss':float(loss.detach()),'tasks':details})
        history.append({'round':round_index+1,'updates':records})
        if on_round:on_round(round_index+1,heads,history[-1])
    return heads,history
