"""Explicit local inference for expert-trained candidates; no production routing."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch
from .expert_model import ExpertContractModel
from .expert_process import NeuralConditionRuntime
from .contract_retrieval import ContractRetriever, prepare_features, validate_request
from .question_evidence_models import context_vector
from .famu_contract_model import tokenize_case, decode_logits
from .semantic_risk import ENCODER_FILES
from .expert_semantics import structured_definition_text


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ExpertReviewer:
    def __init__(self, bundle, arm='B'):
        self.folder=Path(bundle).resolve()
        self.manifest=json.loads((self.folder/'manifest.json').read_text('utf8'))
        if self.manifest.get('version')!='expert-process-v1' or arm not in self.manifest['arms']:
            raise ValueError('Unsupported expert model bundle/arm')
        if set(self.manifest['arms'])!={'A','B'} or self.manifest.get('mode')!='research_candidate':
            raise ValueError('Incomplete candidate manifest')
        required={'base_encoder/'+name for name in ENCODER_FILES}|{v['adapter'] for v in self.manifest['arms'].values()}
        if not required<=set(self.manifest.get('files_sha256',{})):
            raise ValueError('Candidate manifest omits required checksums')
        self.arm=arm
        for relative,expected in self.manifest['files_sha256'].items():
            path=(self.folder/relative).resolve()
            if not path.is_relative_to(self.folder) or not path.is_file() or sha_file(path)!=expected:
                raise ValueError('Model bundle checksum mismatch')
        info=self.manifest['arms'][arm]
        torch.set_num_threads(2)
        self.model=ExpertContractModel(self.folder/'base_encoder')
        self.model.load_adapter(self.folder/info['adapter'])
        self.model.eval();self.classes=info['trained_classes']
        self.retriever=ContractRetriever(**self.manifest['retrieval'])

    def review(self, request):
        x=validate_request(request)
        if x['technical_status']!='ok':
            return {'technical_status':'failed','model_candidates':None,'workflow_action':'technical_retry',
                    'final_legal_conclusion':None,'decision_authority':False}
        result=self.retriever.retrieve(x); features=prepare_features(x,result)
        pieces,audit=tokenize_case(self.model.tokenizer,features['question'],features['evidence'],**self.manifest['tokenization'])
        with torch.no_grad():
            candidates=decode_logits(self.model(pieces,context_vector(features)),self.classes,include_trace=True,allow_inactive=True)
        candidates.pop('excerpt_severity')
        return {'model_version':'expert-process-v1','arm':self.arm,'model_candidates':candidates,
            'retrieval':result,'tokenization':audit,'workflow_action':'human_review',
            'risk_grade_scope':'conditional question/review item prediction; does not establish risk occurrence',
            'condition_supervision_status':'not_trained_missing_confirmed_process_labels',
            'probabilities_calibrated':False,'final_legal_conclusion':None,'decision_authority':False}


def load_condition_runtime(folder):
    """Load a genuinely trained process checkpoint produced by train_expert_conditions."""
    torch.set_num_threads(2)
    folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_text('utf8'))
    encoder=Path(manifest['encoder'])
    if not encoder.is_absolute(): encoder=(folder/encoder).resolve()
    if sha_file(encoder/'model.safetensors')!=manifest['encoder_sha256'] or sha_file(folder/'adapter.safetensors')!=manifest['adapter_sha256']:
        raise ValueError('Condition checkpoint integrity mismatch')
    if 'encoder_files_sha256' in manifest:
        if set(manifest['encoder_files_sha256']) != set(ENCODER_FILES):
            raise ValueError('Incomplete condition encoder checksums')
        if any(sha_file(encoder/name)!=checksum for name,checksum in manifest['encoder_files_sha256'].items()):
            raise ValueError('Condition tokenizer/config integrity mismatch')
    model=ExpertContractModel(encoder);model.load_adapter(folder/'adapter.safetensors')
    supported={task:values if task in manifest['active_process_tasks'] else []
        for task,values in manifest['process_support'].items()}
    return NeuralConditionRuntime(model,supported,aggregation=manifest.get('aggregation','mean'))


def review_joint_candidate(folder, request, conditions, rule, rule_confirmed=False):
    """Expose joint-model candidates alongside process predictions, never overriding labels."""
    folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_text('utf8'))
    if manifest.get('training_mode')!='joint':
        raise ValueError('Joint review requires a model trained jointly on final and condition targets')
    x=validate_request(request)
    if x['technical_status']!='ok':
        return {'technical_status':'failed','model_candidates':None,'workflow_action':'technical_retry','decision_authority':False}
    runtime=load_condition_runtime(folder)
    process=runtime.predict(x,conditions,rule,rule_confirmed)
    retrieved=ContractRetriever(top_k=5,max_context_chars=7000).retrieve(x)
    features=prepare_features(x,retrieved)
    if manifest.get('main_definition_input_version')==3:
        features['evidence']+=structured_definition_text(conditions,rule,rule_confirmed)
    pieces,audit=tokenize_case(runtime.model.tokenizer,features['question'],features['evidence'])
    with torch.no_grad():
        main=decode_logits(runtime.model(pieces,context_vector(features)),manifest['trained_classes'],include_trace=True,allow_inactive=True)
    main.pop('excerpt_severity',None)
    return {'model_candidates':main,'process':process,'retrieval':retrieved,'tokenization':audit,
        'workflow_action':'human_review','probabilities_calibrated':False,
        'final_legal_conclusion':None,'decision_authority':False}
