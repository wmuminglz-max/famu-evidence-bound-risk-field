"""Fixed-weight candidate fusion, with disagreement separate from legal truth."""
import math
import json
from pathlib import Path
from src.lawyer_review_dataset import file_hash,digest
from .contract_question_review import ContractQuestionReviewer,load_contract_question_reviewer
from .question_evidence_models import TASK_CLASSES


def fuse_candidates(candidates,task,weights=None):
    if not candidates or task not in TASK_CLASSES: raise ValueError('Fusion requires supported nonempty candidates')
    weights=[1/len(candidates)]*len(candidates) if weights is None else weights
    if len(weights)!=len(candidates) or any(not math.isfinite(w) or w<0 for w in weights) or abs(sum(weights)-1)>1e-8:
        raise ValueError('Fusion weights must form a finite convex combination')
    classes=TASK_CLASSES[task]
    probabilities={label:0.0 for label in classes}
    supported=set()
    for candidate,weight in zip(candidates,weights):
        p=candidate['probabilities']
        if not p or not set(p)<=set(classes) or any(not math.isfinite(v) or v<0 for v in p.values()) or abs(sum(p.values())-1)>1e-6:
            raise ValueError('Malformed candidate probabilities')
        if candidate['label'] not in p or abs(p[candidate['label']]-max(p.values()))>1e-6:
            raise ValueError('Candidate label and probabilities disagree')
        if weight>0: supported.update(p)
        for label,value in p.items(): probabilities[label]+=weight*value
    probabilities={k:v for k,v in probabilities.items() if k in supported}
    label=max(probabilities,key=probabilities.get)
    return {'label':label,'score':probabilities[label],'probabilities':probabilities,
            'untrained_classes':sorted(set(classes)-supported),'probabilities_calibrated':False,
            'models_disagree':len({c['label'] for c,w in zip(candidates,weights) if w>0})>1,
            'weights':weights,'decision_authority':False}


class ContractEnsembleReviewer(ContractQuestionReviewer):
    def __init__(self,bundle):
        self.folder=Path(bundle).resolve()
        self.manifest=json.loads((self.folder/'manifest.json').read_text('utf8'))
        if self.manifest.get('version')!='2026.09.29-contract-ensemble-v1' or self.manifest.get('mode')!='shadow':
            raise ValueError('Unsupported ensemble bundle')
        if self.manifest.get('weights')!=[.5,.5] or len(self.manifest['members'])!=2:
            raise ValueError('Only the fixed prespecified two-model average is supported')
        self.members=[]
        for member in self.manifest['members']:
            folder=Path(member['bundle'])
            if not folder.is_absolute():folder=(self.folder/folder).resolve()
            if file_hash(folder/'manifest.json')!=member['manifest_sha256']:raise ValueError('Ensemble member manifest changed')
            version=json.loads((folder/'manifest.json').read_text('utf8'))['version']
            if version not in {'2026.09.29-question-evidence-v1','2026.09.29-famu-contract-lora-v1'}:
                raise ValueError('Unsupported ensemble member; recursive ensembles are not allowed')
            self.members.append(load_contract_question_reviewer(folder))

    def _infer(self,x):
        first,second=(member._infer(x) for member in self.members)
        retrieval,features,candidates,details=first
        other_retrieval,other_features,other_candidates,_=second
        # Current members share the lexical source/context selector. Reject a
        # future mismatched configuration until its evidence views are designed.
        if [(e['source_id'],e['start'],e['end']) for e in retrieval['evidence']]!=[(e['source_id'],e['start'],e['end']) for e in other_retrieval['evidence']]:
            raise ValueError('Ensemble evidence scopes differ')
        fused={task:fuse_candidates([candidates[task],other_candidates[task]],task,self.manifest['weights']) for task in TASK_CLASSES}
        return retrieval,{'primary':features,'secondary':other_features,'has_performance_source':features['has_performance_source']},fused,{
            **details,'ensemble_disagreement_tasks':[task for task,pred in fused.items() if pred['models_disagree']],
            'fusion':'fixed_equal_probability_average_with_disagreement_review',
            'member_candidates':[candidates,other_candidates]}
