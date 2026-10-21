"""Serializable sparse classifiers with separate question/evidence channels.

The heads predict QA, risk establishment and conditional severity independently.
Their uncalibrated scores are proposals, not probabilities of legal correctness.
"""
from __future__ import annotations
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

VERSION = "2026.09.29-question-evidence-v1"
ARMS = ("flat_fulltext", "question_only", "evidence_only", "question_evidence")
TASK_CLASSES = {"qa": ("T", "F", "U"),
                "risk_status": ("established", "not_established", "unknown"),
                "severity": ("low", "medium", "high", "critical")}
FEATURE_KEYS = {"question", "evidence", "fulltext", "domain", "review_perspective", "proposition_type",
                "has_performance_source", "retrieved", "context_incomplete"}


def context_vector(row):
    return [float(row["domain"] == "education"), float(row["domain"] == "labor"),
            float(row["review_perspective"] == "学员"), float(row["review_perspective"] == "劳动者"),
            *[float(row["proposition_type"] == t) for t in ("contract_content", "normative_sufficiency", "case_application")],
            float(row["has_performance_source"])]


class QuestionEvidenceHead:
    def __init__(self, task, arm="question_evidence", payload=None):
        if task not in TASK_CLASSES or arm not in ARMS:
            raise ValueError("Unknown classification task or feature arm")
        self.task, self.arm = task, arm
        self.payload = payload
        self.vectorizer = None
        if payload is not None:
            if payload.get("version") != VERSION or payload["task"] != task or payload["arm"] != arm:
                raise ValueError("Incompatible question-evidence weights")
            self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), lowercase=True,
                sublinear_tf=True, norm="l2", vocabulary=payload["vocabulary"])
            self.vectorizer.idf_ = np.asarray(payload["idf"], dtype=float)

    def _corpus(self, rows):
        if self.arm == "flat_fulltext":
            return [r["question"] + "\n" + r["fulltext"] for r in rows]
        if self.arm == "question_only":
            return [r["question"] for r in rows]
        if self.arm == "evidence_only":
            return [r["evidence"] or "无检索匹配" for r in rows]
        return [r[field] or "无检索匹配" for r in rows for field in ("question", "evidence")]

    def features(self, rows):
        if any(set(r) != FEATURE_KEYS for r in rows):
            raise ValueError("Feature input must exactly match the observable whitelist")
        if self.vectorizer is None:
            raise ValueError("Unfitted vectorizer")
        context = csr_matrix(np.asarray([context_vector(r) for r in rows], dtype=float))
        if self.arm == "question_evidence":
            q = self.vectorizer.transform([r["question"] for r in rows])
            e = self.vectorizer.transform([r["evidence"] for r in rows])
            # Shared vocabulary allows explicit query/evidence overlap, while
            # channel normalization prevents raw contract length dominating.
            interaction = normalize(q.multiply(e), norm="l2", copy=True)
            return hstack([q, e, interaction, context], format="csr")
        return hstack([self.vectorizer.transform(self._corpus(rows)), context], format="csr")

    def fit(self, rows, labels, base_weights, fit_groups):
        if len(rows) != len(labels) or len(rows) != len(base_weights) or not rows:
            raise ValueError("Nonempty matched training arrays required")
        if any(label not in TASK_CLASSES[self.task] for label in labels):
            raise ValueError("Unexpected target label")
        if any(not np.isfinite(w) or w <= 0 for w in base_weights):
            raise ValueError("Invalid training weight")
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=6000,
            sublinear_tf=True, norm="l2", lowercase=True)
        # No held-out source contributes to the vocabulary or IDF.
        self.vectorizer.fit(list(dict.fromkeys(self._corpus(rows))))
        matrix = self.features(rows)
        classes = sorted(set(labels))
        mass = Counter()
        for label, weight in zip(labels, base_weights):
            mass[label] += weight
        sample_weight = [w / mass[label] * len(rows) / len(classes) for label, w in zip(labels, base_weights)]
        if len(classes) == 1:
            coefficients, intercept, constant = [], [], classes[0]
        else:
            estimator = LogisticRegression(C=1.0, max_iter=800, solver="lbfgs", random_state=20260929)
            estimator.fit(matrix, labels, sample_weight=sample_weight)
            classes = estimator.classes_.tolist()
            coefficients, intercept, constant = estimator.coef_.tolist(), estimator.intercept_.tolist(), None
        self.payload = {"version": VERSION, "task": self.task, "arm": self.arm, "classes": classes,
            "vocabulary": {k: int(v) for k, v in self.vectorizer.vocabulary_.items()}, "idf": self.vectorizer.idf_.tolist(),
            "coefficients": coefficients, "intercept": intercept, "constant_class": constant,
            "feature_dimension": matrix.shape[1], "training_rows": len(rows),
            "fit_groups": sorted(set(fit_groups)), "C": 1.0,
            "weights": "contract/cluster base weights; balance weighted class mass within training fold; normalize total to row count",
            "base_class_mass": dict(mass), "effective_class_mass": dict(Counter({label: len(rows)/len(classes) for label in classes})),
            "probabilities_calibrated": False}
        return self

    def predict(self, rows):
        if self.payload is None:
            raise ValueError("Classifier has not been fitted")
        if not rows:
            return []
        matrix = self.features(rows)
        if matrix.shape[1] != self.payload["feature_dimension"]:
            raise ValueError("Feature shape differs from saved weights")
        classes = self.payload["classes"]
        if self.payload["constant_class"]:
            probs = np.ones((len(rows), 1))
        else:
            logits = np.asarray(matrix @ np.asarray(self.payload["coefficients"]).T) + np.asarray(self.payload["intercept"])
            if len(classes) == 2:
                p = 1.0 / (1.0 + np.exp(-np.clip(logits[:, 0], -700, 700)))
                probs = np.stack([1-p, p], axis=1)
            else:
                logits -= logits.max(axis=1, keepdims=True)
                probs = np.exp(logits)
                probs /= probs.sum(axis=1, keepdims=True)
        return [{"label": classes[int(p.argmax())], "score": float(p.max()),
                 "probabilities": dict(zip(classes, p.tolist())), "probabilities_calibrated": False,
                 "untrained_classes": sorted(set(TASK_CLASSES[self.task])-set(classes))} for p in probs]
