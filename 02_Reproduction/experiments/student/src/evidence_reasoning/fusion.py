"""Group-out-of-fold stacking and calibration on a separate group partition."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from .metrics import weights


def softmax(logits):
    x = np.asarray(logits, dtype=float)
    x = x - x.max(axis=-1, keepdims=True)
    value = np.exp(x)
    return value / value.sum(axis=-1, keepdims=True)


def temperature(logits, labels, roots, splits):
    if set(splits) != {"calibration"}:
        raise ValueError("Temperature fitting requires the calibration partition")
    labels = np.asarray(labels)
    mask = labels >= 0
    if not mask.any():
        raise ValueError("Calibration labels missing")
    logits, labels = np.asarray(logits)[mask], labels[mask]
    w = weights(np.asarray(roots)[mask])
    def nll(t):
        p = softmax(logits / t)
        return np.average(-np.log(np.clip(p[np.arange(len(p)), labels], 1e-9, 1)), weights=w)
    grid = np.geomspace(.4, 6., 61)
    t = float(min(grid, key=nll))
    return {"value": t, "nll_before": float(nll(1.)), "nll_after": float(nll(t)),
            "fit_roots": sorted(set(np.asarray(roots)[mask])), "split": "calibration"}


class Stacker:
    def __init__(self, payload=None):
        self.payload = payload

    @staticmethod
    def features(probabilities):
        p = np.stack(probabilities, 1)
        if p.ndim != 3 or p.shape[-1] != 3 or not np.all(np.isfinite(p)) or not np.allclose(p.sum(2), 1):
            raise ValueError("Stacking needs aligned three-class probabilities")
        return np.log(np.clip(p, 1e-6, 1)).reshape(len(p), -1)

    def fit(self, probabilities, labels, rows, certificates):
        if len(certificates) != len(probabilities):
            raise ValueError("Every base model must supply OOF isolation certificates")
        if any(row["meta"]["split"] != "train" for row in rows):
            raise ValueError("Stacking is fitted only on training OOF rows")
        for model_certificates in certificates:
            if len(model_certificates) != len(rows):
                raise ValueError("Missing OOF certificate")
            for row, cert in zip(rows, model_certificates):
                group = row["meta"]["group_id"]
                if cert.get("sample_id") != row["meta"]["sample_id"] or group in set(cert["fit_groups"]) | set(cert.get("selection_groups", [])):
                    raise ValueError("OOF prediction saw its own group during training or model selection")
        labels = np.asarray(labels)
        mask = labels >= 0
        if set(labels[mask]) != {0, 1, 2}:
            raise ValueError("Stacking needs all three answer classes")
        model = LogisticRegression(C=.2, max_iter=500, random_state=20260920)
        w = weights([r["meta"]["root_id"] for r in rows])
        model.fit(self.features(probabilities)[mask], labels[mask], sample_weight=w[mask])
        self.payload = {"coefficients": model.coef_.tolist(), "intercept": model.intercept_.tolist(),
                        "classes": model.classes_.tolist(), "models": len(probabilities),
                        "fit_groups": sorted({r["meta"]["group_id"] for r in rows}),
                        "method": "group_out_of_fold_logistic", "certified_rows": len(rows)}
        return self

    def logits(self, probabilities):
        if not self.payload or len(probabilities) != self.payload["models"]:
            raise ValueError("Stacker model count differs from fitted protocol")
        return self.features(probabilities) @ np.asarray(self.payload["coefficients"]).T + np.asarray(self.payload["intercept"])

    def predict(self, probabilities):
        return softmax(self.logits(probabilities))


def fit_convex_weights(probabilities, labels, roots, splits):
    if set(splits) != {"calibration"} or len(probabilities) != 2:
        raise ValueError("Two-model convex weights require independent calibration groups")
    y = np.asarray(labels)
    mask = y >= 0
    if not mask.any():
        raise ValueError("Missing calibration labels")
    w = weights(np.asarray(roots)[mask])
    def loss(alpha):
        p = alpha * probabilities[0] + (1 - alpha) * probabilities[1]
        return float(np.average(-np.log(np.clip(p[mask, y[mask]], 1e-9, 1)), weights=w))
    alpha = float(min(np.linspace(0, 1, 21), key=loss))
    return {"weights": [alpha, 1 - alpha], "calibration_nll": loss(alpha), "fit_roots": sorted(set(roots))}


def diagnose_routing(*args, **kwargs):
    from .experiment_protocol import routing_gate
    return routing_gate(*args, **kwargs)
