"""Locally fine-tuned BGE contract reviewer: shared LoRA encoder and four heads.

Only user-confirmed labels supervise training. Long retrieved evidence is split
into overlapping token windows and fully pooled, never silently truncated.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import torch
from torch import nn
from safetensors.torch import load_file, save_file
from transformers import AutoModel, AutoTokenizer
from src.lawyer_review_dataset import file_hash
from .semantic_risk import ENCODER_FILES
from .question_evidence_models import TASK_CLASSES, context_vector
from .contract_retrieval import ContractRetriever, prepare_features
from .contract_question_review import ContractQuestionReviewer

VERSION = '2026.09.29-famu-contract-lora-v1'
MODEL_CLASSES = {**TASK_CLASSES, 'excerpt_severity':TASK_CLASSES['severity']}


class LoRALinear(nn.Module):
    def __init__(self, base, rank=4, alpha=8):
        super().__init__()
        self.base, self.scaling = base, alpha/rank
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)
        for parameter in self.base.parameters(): parameter.requires_grad_(False)

    def forward(self, x):
        return self.base(x)+self.b(self.a(x))*self.scaling


def tokenize_case(tokenizer, question, evidence, max_length=384, overlap=32):
    if not 64<=max_length<=512 or not 0<=overlap<max_length:
        raise ValueError('Invalid BERT window configuration')
    q = tokenizer.encode(question, add_special_tokens=False,verbose=False)
    e = tokenizer.encode(evidence, add_special_tokens=False,verbose=False)
    width = max_length-len(q)-3
    if width <= overlap:
        raise ValueError('Question exceeds model window budget; shorten or split the question explicitly')
    pieces, ranges = [], []
    start = 0
    while True:
        end = min(start+width,len(e))
        part = e[start:end]
        # Explicit BERT pair format also works with Transformers 5 tokenizers,
        # which no longer expose build_inputs_with_special_tokens.
        ids = [tokenizer.cls_token_id]+q+[tokenizer.sep_token_id]+part+[tokenizer.sep_token_id]
        pieces.append({'input_ids':ids, 'attention_mask':[1]*len(ids),
                       'token_type_ids':[0]*(len(q)+2)+[1]*(len(part)+1)})
        ranges.append({'start':start,'end':end})
        if end == len(e): break
        start = end-overlap
    return pieces, {'question_tokens':len(q),'evidence_tokens':len(e),'windows':len(pieces),
                    'max_length':max_length,'overlap':overlap,'all_retrieved_tokens_included':True,
                    'evidence_token_ranges':ranges,'supervision_unit':'record_after_window_pooling',
                    'scope':'retrieved evidence only; retrieval budget and missing context remain separately reported'}


class FamuContractModel(nn.Module):
    def __init__(self, encoder_folder, rank=4, alpha=8):
        super().__init__()
        self.encoder_folder = Path(encoder_folder)
        self.tokenizer = AutoTokenizer.from_pretrained(str(encoder_folder),local_files_only=True,trust_remote_code=False)
        self.encoder = AutoModel.from_pretrained(str(encoder_folder),local_files_only=True,trust_remote_code=False)
        for parameter in self.encoder.parameters(): parameter.requires_grad_(False)
        self.lora_modules = []
        for name, module in list(self.encoder.named_modules()):
            if name.endswith(('.attention.self.query','.attention.self.value')) and isinstance(module,nn.Linear):
                parent_name, child_name = name.rsplit('.',1)
                setattr(self.encoder.get_submodule(parent_name),child_name,LoRALinear(module,rank,alpha))
                self.lora_modules.append(name)
        if not self.lora_modules:
            raise ValueError('Encoder architecture has no supported BERT query/value layers')
        self.text_projection = nn.Sequential(nn.Linear(self.encoder.config.hidden_size,128),nn.GELU())
        self.context_projection = nn.Sequential(nn.Linear(8,128),nn.GELU())
        self.fusion_gate = nn.Linear(256,128)
        self.fusion_norm = nn.LayerNorm(128)
        self.heads = nn.ModuleDict({task:nn.Linear(128,len(classes)) for task,classes in MODEL_CLASSES.items()})
        self.encoder.eval()  # Fixed dropout behavior for reproducibility experiments.

    def pad_tokens(self, pieces):
        """Keep integer token tensors on the same device as the encoder."""
        tokens = self.tokenizer.pad(pieces, padding=True, return_tensors='pt')
        device = next(self.encoder.parameters()).device
        return {name: value.to(device) for name, value in tokens.items()}

    def encode_case(self, pieces, context):
        pooled = []
        for offset in range(0,len(pieces),8):
            tokens = self.pad_tokens(pieces[offset:offset+8])
            output = self.encoder(**tokens).last_hidden_state[:,0]
            pooled.append(torch.nn.functional.normalize(output,dim=-1))
        pooled = torch.cat(pooled).mean(dim=0)
        pooled = torch.nn.functional.normalize(pooled,dim=-1)
        text = self.text_projection(pooled)
        context = self.context_projection(torch.as_tensor(context,dtype=pooled.dtype,device=pooled.device))
        gate = torch.sigmoid(self.fusion_gate(torch.cat([text,context])))
        # Learned residual context fusion. Gate values are not legal confidence.
        return self.fusion_norm(text+gate*context)

    def forward(self, pieces, context):
        features = self.encode_case(pieces,context)
        return {task:head(features) for task,head in self.heads.items()}

    def trainable_state(self):
        return {name:parameter.detach().cpu().contiguous() for name,parameter in self.named_parameters() if parameter.requires_grad}

    def save_adapter(self,path):
        save_file(self.trainable_state(),str(path))

    def load_adapter(self,path):
        state = load_file(str(path),device='cpu')
        expected = self.trainable_state()
        if set(state)!=set(expected) or any(state[k].shape!=expected[k].shape or not torch.isfinite(state[k]).all() for k in state):
            raise ValueError('Invalid LoRA adapter tensors')
        self.load_state_dict(state,strict=False)

    def predict_case(self,pieces,context,trained_classes,*,include_trace=False,allow_inactive=False):
        self.eval()
        with torch.no_grad(): logits = self(pieces,context)
        return decode_logits(logits,trained_classes,include_trace=include_trace,allow_inactive=allow_inactive)


def decode_logits(logits,trained_classes,*,include_trace=False,allow_inactive=False):
    """Decode by semantic labels and retain the pre-mask stage for diagnostics."""
    if set(logits)!=set(MODEL_CLASSES) or set(trained_classes)!=set(MODEL_CLASSES):
        raise ValueError('Every head requires explicit label support metadata')
    result = {}
    for task,raw in logits.items():
        classes, supported = MODEL_CLASSES[task], trained_classes[task]
        if (raw.shape != (len(classes),) or not torch.isfinite(raw).all() or
                len(set(supported))!=len(supported) or not set(supported)<=set(classes) or
                (not supported and not allow_inactive)):
            raise ValueError('Invalid logits or trained classes')
        values=raw.clone()
        for i,label in enumerate(classes):
            if label not in supported: values[i]=-torch.inf
        probabilities=torch.softmax(values,dim=-1) if supported else None
        index=int(probabilities.argmax()) if supported else None
        result[task]={'label':classes[index] if supported else None,
            'score':float(probabilities[index]) if supported else None,
            'probabilities':{label:float(probabilities[i]) for i,label in enumerate(classes) if label in supported},
            'untrained_classes':sorted(set(classes)-set(supported)),'probabilities_calibrated':False}
        if include_trace:
            raw_probs=torch.softmax(raw,dim=-1)
            result[task]['trace']={'label_order':list(classes),
                'label2id':{label:i for i,label in enumerate(classes)},
                'raw_logits':dict(zip(classes,raw.detach().tolist())),
                'raw_probabilities':dict(zip(classes,raw_probs.detach().tolist())),
                'raw_label':classes[int(raw.argmax())],
                'support_mask':{label:label in supported for label in classes},
                'decoded_label':result[task]['label'],'inactive_task':not bool(supported)}
    return result


class FamuContractReviewer(ContractQuestionReviewer):
    def __init__(self,bundle):
        self.folder = Path(bundle).resolve()
        self.manifest = json.loads((self.folder/'manifest.json').read_text('utf-8'))
        if self.manifest.get('version')!=VERSION or self.manifest.get('mode')!='shadow':
            raise ValueError('Unsupported Famu contract bundle')
        if file_hash(self.folder/'adapter.safetensors')!=self.manifest['adapter_sha256']:
            raise ValueError('Famu adapter hash mismatch')
        encoder = self.manifest['encoder']
        encoder_folder = Path(encoder['folder'])
        if not encoder_folder.is_absolute():
            encoder_folder = (self.folder/encoder_folder).resolve()
            if not encoder_folder.is_relative_to(self.folder):raise ValueError('Base encoder path leaves bundle')
        if {name:file_hash(encoder_folder/name) for name in ENCODER_FILES} != encoder['files_sha256']:
            raise ValueError('Famu base encoder hash mismatch')
        torch.set_num_threads(2)
        self.model = FamuContractModel(encoder_folder,**self.manifest['lora'])
        self.model.load_adapter(self.folder/'adapter.safetensors')
        self.retriever = ContractRetriever(**self.manifest['retrieval_parameters'])

    def _infer(self,x):
        retrieval = self.retriever.retrieve(x)
        features = prepare_features(x,retrieval)
        pieces,audit = tokenize_case(self.model.tokenizer,features['question'],features['evidence'],**self.manifest['tokenization'])
        candidates = self.model.predict_case(pieces,context_vector(features),self.manifest['trained_classes'])
        candidates.pop('excerpt_severity')
        return retrieval,features,candidates,{'model_name':self.manifest.get('name','法眸合同审查模型 LoRA v1'),'tokenization_audit':audit,
            'auxiliary_training_head_not_used_for_question_grade':True,
            'training_method':self.manifest.get('training_method','supervised multitask LoRA fine-tuning of a local pretrained encoder'),
            'generative_legal_opinions_supported':False}
