"""Verified local model bundle with separate truth, uncertainty and workflow fields."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

from . import VERSION
from .schema import model_input, decision_route
from .decision import select_decision, workflow_status, execute_extracted
from .logic import solve, verify_certificate
from .provenance import visible_evidence, visible_fingerprint, withdraw
from .text import numeric_facts
from .extraction import ConditionExtractor
from .graphs import FrozenEncoder, graph_record, topology
from .neural import load_model, predict
from .fusion import softmax, Stacker
from .consensus import SequentialConsensus

DEFAULT_BUNDLE = Path(__file__).resolve().parents[2] / "outputs/algorithm_strengthening_20260920/bundle"


class ReasoningRuntime:
    def __init__(self, bundle=None, extractor_variant="auto"):
        if extractor_variant not in {"auto","linear","neural"}:
            raise ValueError("Extractor variant must be auto, linear or neural")
        self.extractor_variant = extractor_variant
        self.path = Path(bundle or DEFAULT_BUNDLE).resolve()
        self.manifest = self._json("manifest.json")
        if self.manifest.get("version") != VERSION or self.manifest.get("mode") != "shadow" or self.manifest.get("decision_authority") is not False:
            raise ValueError("Unsupported or unapproved reasoning bundle")
        required = {"extractor.json", "fusion.json"}
        for files in self.manifest.get("models", {}).values():
            for name in files:
                required.update({name, Path(name).with_suffix(".json").name})
        if self._file("consensus.json").exists():
            required.add("consensus.json")
        if self.manifest.get("severity_adapter"):
            required.add(self.manifest["severity_adapter"])
        if self.manifest.get("cross_encoder_adapter"):
            required.add(self.manifest["cross_encoder_adapter"])
        if not required <= set(self.manifest.get("files_sha256", {})):
            raise ValueError("Missing hash coverage for a required model artifact")
        for name, checksum in self.manifest["files_sha256"].items():
            file = self._file(name)
            if hashlib.sha256(file.read_bytes()).hexdigest() != checksum:
                raise ValueError(f"Reasoning artifact hash mismatch: {name}")
        self.extractor = ConditionExtractor(self._json("extractor.json"))
        self.fusion = self._json("fusion.json")
        self.stacker = Stacker(self.fusion["stacker"])
        self.consensus = SequentialConsensus(self._json("consensus.json")) if self._file("consensus.json").exists() else None
        self.encoder = None
        self.models = {}
        self.severity_models = None
        self._neural_extractor = None
        if self.manifest.get("severity_adapter"):
            config = self._json(self.manifest["severity_adapter"])
            if not set(config["files"]) <= set(self.manifest["files_sha256"]):
                raise ValueError("Missing hash coverage for severity weights")
        if self.manifest.get("cross_encoder_adapter"):
            config = self._json(self.manifest["cross_encoder_adapter"])
            if not set(config["files"]) <= set(self.manifest["files_sha256"]):
                raise ValueError("Missing hash coverage for neural condition weights")

    def extract_conditions(self,payload):
        config = self._json(self.manifest["cross_encoder_adapter"]) if self.manifest.get("cross_encoder_adapter") else None
        variant = self.extractor_variant if self.extractor_variant != "auto" else config["selected"] if config else "linear"
        if variant == "linear":
            result = self.extractor.extract(payload)
        else:
            if not config:
                raise ValueError("Neural condition adapter is unavailable")
            self._load_neural()
            if self._neural_extractor is None:
                from .cross_encoder import NeuralConditionExtractor
                path = Path(self.manifest["encoder_path"])
                if not path.is_absolute():
                    path = self.path / path
                self._neural_extractor = NeuralConditionExtractor(path,[self._file(name) for name in config["files"]])
            result = self._neural_extractor.extract(payload)
        result["extractor_variant"] = variant
        return result

    def _file(self, name):
        path = (self.path / name).resolve()
        if path.parent != self.path or name not in {path.name}:
            raise ValueError("Bundle artifact must be a direct child file")
        return path

    def _json(self, name):
        return json.loads(self._file(name).read_text(encoding="utf-8"))

    def _load_neural(self):
        if self.encoder is not None:
            return
        import torch
        torch.set_num_threads(min(4, torch.get_num_threads()))
        encoder_path = self.manifest.get("encoder_path")
        if encoder_path:
            folder = Path(encoder_path)
            if not folder.is_absolute():
                folder = self.path / folder
            encoder_path = folder.resolve()
            for name, checksum in self.manifest.get("encoder_files_sha256", {}).items():
                file = (folder / name).resolve()
                if file.parent != folder.resolve() or hashlib.sha256(file.read_bytes()).hexdigest() != checksum:
                    raise ValueError("Frozen encoder hash mismatch")
        encoder = FrozenEncoder(encoder_path, dimension=self.manifest["encoder_dimension"])
        models = {name: [load_model(self._file(file))[0] for file in files] for name, files in self.manifest["models"].items()}
        if set(models) != {"mlp", "rgcn"} or not all(models.values()):
            raise ValueError("Both model ensembles are required")
        self.models, self.encoder = models, encoder

    def predict(self, payload):
        x = model_input(payload)
        routing = decision_route(x)
        initial = solve(x)
        if initial["input_status"] != "valid":
            return {"version": VERSION, "mode": "shadow", "decision_authority": False,
                    "proposition_label": None, "input_status": initial["input_status"],
                    'workflow_status': workflow_status(initial['input_status'], None),
                    'risk_established': None, 'risk_established_status': 'technical_failure' if initial['input_status'] == 'technical_failure' else 'unlabeled', 'severity': None,
                    "workflow_action": "review", "model_confidence": None, "certificate": None, "routing": routing}
        extracted = self.extract_conditions(x)
        logic = solve(extracted["input"])
        self._load_neural()
        row = {"input": x, "targets": {}, "meta": {"sample_id": "runtime", "root_id": "runtime", "group_id": "runtime",
                "split": "external", "label_status": "unlabeled"}}
        graph = graph_record(row, self.encoder)
        raw = {name: np.mean([predict(model, [graph])["logits"] for model in models], axis=0) for name, models in self.models.items()}
        calibrated = {name: softmax(z / self.fusion["temperatures"][name]["value"]) for name, z in raw.items()}
        probabilities = {**calibrated, "mean": (calibrated["mlp"] + calibrated["rgcn"]) / 2,
            "convex": sum(self.fusion["convex"]["weights"][i] * calibrated[a] for i,a in enumerate(("mlp", "rgcn"))),
            "stacking": softmax(self.stacker.logits([softmax(raw[a]) for a in ("mlp", "rgcn")]) / self.fusion["stack_temperature"]["value"])}
        p = probabilities[self.fusion["selected"]][0]
        label = ("T", "F", "U")[int(p.argmax())]
        decision = select_decision(x, logic, label, extracted.get("uncertain_conditions", []))
        return {"version": VERSION, "mode": "shadow", "decision_authority": False,
            **decision, "input_status": logic["input_status"],
            "extractor_variant": extracted["extractor_variant"],
            "proposition_confidence": None,
            "model_proposal": label, "model_confidence": float(p.max()),
            "model_entropy": float(-(p * np.log(np.clip(p, 1e-9, 1))).sum()),
            "selected_model": self.fusion["selected"], "neural_probabilities": dict(zip(("T", "F", "U"), p.tolist())),
            "model_probabilities": {name: dict(zip(("T", "F", "U"), values[0].tolist())) for name, values in probabilities.items()},
            "workflow_action": "review", "neural_logic_disagreement": label != logic["proposition_label"],
            "conditional_logic": logic,
            "certificate_verified": verify_certificate(extracted["input"], logic["certificate"]) if logic["certificate"] else None,
            "certificate_scope": "given_rule_and_learned_observations_only",
            "evidence_predictions": extracted["evidence_predictions"],
            "visible_evidence_ids": [e["id"] for e in visible_evidence(x)],
            "visible_fingerprint": visible_fingerprint(x),
            "graph": {"nodes": len(graph["nodes"]), "edges": len(graph["edges"])},
            "numeric_facts": {e["id"]: numeric_facts(e["text"]) for e in visible_evidence(x)}}

    def predict_exact(self, payload):
        """Extraction and exact execution without loading proposition ensembles."""
        runtime = self
        class Adapter:
            def extract(self, x):
                return runtime.extract_conditions(x)
        return execute_extracted(payload, Adapter())

    def withdraw_and_predict(self, payload, sources):
        changed = withdraw(payload, sources)
        result = self.predict(changed)
        result["withdrawn_sources"] = changed["withdrawn_sources"]
        result["previous_visible_fingerprint"] = visible_fingerprint(model_input(payload))
        return result

    def route_consensus(self, observable_input):
        if self.consensus is None:
            raise ValueError("Consensus trajectory model is unavailable")
        return self.consensus.before_call(observable_input)

    def correct_consensus(self, observable_input):
        if self.consensus is None:
            raise ValueError("Consensus trajectory model is unavailable")
        return self.consensus.after_call(observable_input)

    def predict_risk_severity(self, payload, risk_established=None):
        """Severity has a separate applicability gate; generic condition T is not a risk."""
        if risk_established is not None and type(risk_established) is not bool:
            raise ValueError('Risk establishment must be boolean or unlabeled None')
        if risk_established is not True:
            return {"severity": None, "reason": "risk_not_established" if risk_established is False else "risk_unlabeled",
                    'risk_established': risk_established, 'risk_established_status': 'not_established' if risk_established is False else 'unlabeled',
                    "mode": "shadow", "decision_authority": False}
        if not self.manifest.get("severity_adapter"):
            raise ValueError("Dedicated severity adapter is unavailable")
        from .severity import SeverityMLP, features
        import torch
        self._load_neural()
        if self.severity_models is None:
            config = self._json(self.manifest["severity_adapter"])
            models = []
            for file in config["files"]:
                model = SeverityMLP(config["input_dim"])
                model.load_state_dict(torch.load(self._file(file),map_location="cpu",weights_only=True))
                model.eval()
                models.append(model)
            self.severity_models = models
        x = torch.tensor(features(payload,self.encoder)[None,:])
        with torch.no_grad():
            p = np.mean([model(x)[0].softmax(-1).numpy()[0] for model in self.severity_models],axis=0)
        names = ("low","medium","high","critical")
        return {"severity": names[int(p.argmax())], "probabilities":dict(zip(names,p.tolist())),
                "calibrated": False, "mode":"shadow", "decision_authority":False,
                "scope":"risk_already_established_by_caller", "workflow_action":"review"}
