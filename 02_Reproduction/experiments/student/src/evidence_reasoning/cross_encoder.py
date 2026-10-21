"""Joint query/evidence encoder; fine-tune only the top BERT block and pair head."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import f1_score

from .data import read_records, write_json
from .extraction import ConditionExtractor, PAIR_LABELS
from .provenance import visible_evidence
from .text import adaptive_tokenize
from .metrics import weights
from .spans import annotated_pairs


class PairEncoder(nn.Module):
    def __init__(self, encoder_path):
        super().__init__()
        from transformers import AutoModel
        self.base = AutoModel.from_pretrained(str(encoder_path),local_files_only=True,trust_remote_code=False)
        if not hasattr(self.base,"encoder") or not hasattr(self.base.encoder,"layer"):
            raise ValueError("Cross-encoder currently requires the validated BGE/BERT backbone")
        self.base.requires_grad_(False)
        self.base.encoder.layer[-1].requires_grad_(True)
        self.head = nn.Linear(self.base.config.hidden_size,3)

    def forward(self,batch):
        return self.head(self.base(**batch).last_hidden_state[:,0])

    def adapter_state(self):
        names = {name for name,p in self.named_parameters() if p.requires_grad}
        return {name:weight for name,weight in self.state_dict().items() if name in names}

    def load_adapter(self,state):
        expected = set(self.adapter_state())
        if set(state) != expected:
            raise ValueError("Unexpected cross-encoder adapter tensors")
        result = self.load_state_dict(state,strict=False)
        if result.unexpected_keys or any(name in expected for name in result.missing_keys):
            raise ValueError("Incomplete cross-encoder adapter")


def encode_pairs(pairs,tokenizer):
    result = []
    for query,text in pairs:
        encoded,audit = adaptive_tokenize([("条件",query),("原文",text)],tokenizer,max_length=512,priority=[1.,3.])
        if any(field["truncated"] for field in audit["fields"]):
            raise ValueError("Pair exceeds full-evidence token budget")
        result.append(encoded)
    return result


def probabilities(model,encoded,tokenizer):
    result = []
    model.eval()
    with torch.no_grad():
        for start in range(0,len(encoded),64):
            batch = tokenizer.pad(encoded[start:start+64],padding=True,return_tensors="pt")
            result.append(model(batch).softmax(-1).numpy())
    return np.concatenate(result) if result else np.empty((0,3))


class NeuralConditionExtractor(ConditionExtractor):
    def __init__(self,encoder_path,adapter_paths):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(encoder_path),local_files_only=True,trust_remote_code=False)
        self.models = []
        self.cache = {}
        for path in adapter_paths:
            model = PairEncoder(encoder_path)
            model.load_adapter(torch.load(path,map_location="cpu",weights_only=True))
            model.eval()
            self.models.append(model)
        if not self.models:
            raise ValueError("No neural condition adapters supplied")

    def predict_pairs(self,pairs):
        unseen = sorted(set(pairs) - set(self.cache))
        if unseen:
            encoded = encode_pairs(unseen,self.tokenizer)
            p = np.mean([probabilities(m,encoded,self.tokenizer) for m in self.models],axis=0)
            self.cache.update(zip(unseen,p))
        return np.stack([self.cache[pair] for pair in pairs]) if pairs else np.empty((0,3))


def train_cross_encoder(output,epochs=4):
    from transformers import AutoTokenizer
    from .training import _extractor_evaluation
    output = Path(output)
    torch.set_num_threads(4)
    manifest_path = output / "bundle/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    encoder_path = Path(manifest["encoder_path"])
    if not encoder_path.is_absolute():
        encoder_path = output / "bundle" / encoder_path
    tokenizer = AutoTokenizer.from_pretrained(str(encoder_path),local_files_only=True,trust_remote_code=False)
    rows = read_records(output / "dataset.json")
    train,validation,test = ([r for r in rows if r["meta"]["split"] == split] for split in ("train","validation","test"))
    train_pairs, train_audit = annotated_pairs(train, return_audit=True)
    val_pairs, val_audit = annotated_pairs(validation, return_audit=True)
    fit_groups,selection_groups = {r["group_id"] for r in train_pairs},{r["group_id"] for r in val_pairs}
    if fit_groups & selection_groups:
        raise ValueError("Cross-encoder fitting/selection overlap")
    tx,vx = (encode_pairs([(r["query"],r["text"]) for r in pairs],tokenizer) for pairs in (train_pairs,val_pairs))
    y,vy = (torch.tensor([r["label"] for r in part]) for part in (train_pairs,val_pairs))
    w = torch.tensor(weights([r["root_id"] for r in train_pairs]),dtype=torch.float32)
    counts = torch.bincount(y,minlength=3).float()
    w *= (counts.sum() / (3 * counts.clamp_min(1))).sqrt()[y]
    for root in {r["root_id"] for r in train_pairs}:
        ix = torch.tensor([r["root_id"] == root for r in train_pairs])
        w[ix] /= w[ix].sum()
    vw = weights([r["root_id"] for r in val_pairs])
    report = {"training_pairs":len(tx),"validation_pairs":len(vx),"epochs":epochs,"label_status":"synthetic_verified",
              "filter_denominators": {"train": train_audit, "validation": val_audit},
              "seeds":[],"decision_authority":False,"frozen_blocks":3,"trainable_blocks":1,
              "long_pair_policy":"requires span labels; long unlabeled windows are excluded from fitting"}
    files = []
    for seed in (20260920,20260921,20260922):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = PairEncoder(encoder_path)
        optimizer = torch.optim.AdamW([{"params":model.base.encoder.layer[-1].parameters(),"lr":.0003},
                                       {"params":model.head.parameters(),"lr":.002}],weight_decay=.01)
        best,best_loss,best_epoch = None,float("inf"),None
        for epoch in range(epochs):
            model.train()
            order = rng.permutation(len(tx))
            for start in range(0,len(order),64):
                ix = order[start:start+64]
                batch = tokenizer.pad([tx[int(i)] for i in ix],padding=True,return_tensors="pt")
                loss = (F.cross_entropy(model(batch),y[ix],reduction="none") * w[ix]).sum() / w[ix].sum()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
                optimizer.step()
            p = probabilities(model,vx,tokenizer)
            value = float(np.average(-np.log(np.clip(p[np.arange(len(p)),vy.numpy()],1e-9,1)),weights=vw))
            if value < best_loss:
                best,best_loss,best_epoch = copy.deepcopy(model.adapter_state()),value,epoch+1
        model.load_adapter(best)
        name = f"condition_cross_{seed}.pt"
        torch.save(best,output / "bundle" / name)
        files.append(name)
        report["seeds"].append({"seed":seed,"selected_epoch":best_epoch,"validation_nll":best_loss,
            "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad)})
        print(f"condition cross encoder seed={seed} complete",flush=True)
    neural = NeuralConditionExtractor(encoder_path,[output / "bundle" / f for f in files])
    linear = ConditionExtractor(json.loads((output / "bundle/extractor.json").read_text(encoding="utf-8")))
    linear_val = _extractor_evaluation(linear,validation)
    neural_val = _extractor_evaluation(neural,validation)
    report.update(validation={"linear":linear_val,"neural":neural_val},test=_extractor_evaluation(neural,test))
    # Choose only on development outcomes; equal scores prefer the faster model.
    selected = "neural" if neural_val["end_to_end"]["macro_f1"] > linear_val["end_to_end"]["macro_f1"] else "linear"
    report["selected_extractor"] = selected
    write_json(output / "cross_encoder_report.json",report)
    write_json(output / "bundle/cross_encoder.json",{"files":files,"selected":selected,"fit_groups":sorted(fit_groups),
        "selection_groups":sorted(selection_groups),"label_status":"synthetic_verified","decision_authority":False})
    for name in files + ["cross_encoder.json"]:
        manifest["files_sha256"][name] = hashlib.sha256((output / "bundle" / name).read_bytes()).hexdigest()
    manifest["cross_encoder_adapter"] = "cross_encoder.json"
    write_json(manifest_path,manifest)
    print(json.dumps({"selected":selected,"neural_test_f1":report["test"]["end_to_end"]["macro_f1"]},ensure_ascii=False),flush=True)
    return report
