"""Runnable shadow reviewer with separate lexical QA and semantic risk evidence."""
import json
from pathlib import Path
from src.lawyer_review_dataset import file_hash
from .contract_question_review import ContractQuestionReviewer
from .contract_retrieval import ContractRetriever, prepare_features
from .question_evidence_models import QuestionEvidenceHead
from .semantic_risk import LocalSemanticEncoder, SemanticRiskHead, VERSION


class SemanticContractReviewer(ContractQuestionReviewer):
    def __init__(self, bundle):
        self.folder = Path(bundle).resolve()
        self.manifest = json.loads((self.folder/'manifest.json').read_text('utf-8'))
        if self.manifest.get('version') != VERSION or self.manifest.get('mode') != 'shadow' or self.manifest.get('architecture') != 'staged':
            raise ValueError('Unsupported semantic review bundle')
        required = {'qa.json', 'risk_status.json', 'severity.json'}
        files = self.manifest.get('files_sha256', {})
        if not required <= set(files):
            raise ValueError('Missing semantic classifier hash coverage')
        for name, checksum in files.items():
            path = (self.folder/name).resolve()
            if path.parent != self.folder or file_hash(path) != checksum:
                raise ValueError('Bundle file hash or path mismatch')
        encoder = self.manifest['encoder']
        self.encoder = LocalSemanticEncoder(encoder['folder'], encoder['files_sha256'])
        self.retriever = ContractRetriever(**self.manifest['retrieval_parameters'], ranker=self.encoder)
        self.qa_retriever = ContractRetriever(**self.manifest['qa_retrieval_parameters'])
        self.qa_head = QuestionEvidenceHead('qa', payload=json.loads((self.folder/'qa.json').read_text('utf-8')))
        self.heads = {task: SemanticRiskHead(task, 'staged', json.loads((self.folder/(task+'.json')).read_text('utf-8')))
                      for task in ('risk_status', 'severity')}

    def _infer(self, x):
        retrieval = self.retriever.retrieve(x)
        qa_retrieval = self.qa_retriever.retrieve(x)
        qa_features = prepare_features(x, qa_retrieval)
        semantic = self.encoder.features(x, retrieval)
        candidates = {task: head.predict([semantic])[0] for task, head in self.heads.items()}
        candidates['qa'] = self.qa_head.predict([qa_features])[0]
        features = {**prepare_features(x, retrieval), 'semantic_vector': semantic.tolist(), 'qa_features': qa_features}
        return retrieval, features, candidates, {'qa_retrieval': qa_retrieval, 'qa_evidence': qa_retrieval['evidence'],
            'evidence_by_task': {'qa': 'qa_evidence', 'risk_status': 'evidence', 'severity': 'evidence'},
            'semantic_encoder_sha256': self.encoder.hashes['model.safetensors'],
            'risk_architecture': 'unknown_then_conditional_risk', 'severity_architecture': 'ordinal_exceedance'}
