"""Opt-in NumPy relation student; inherited evidence validation and rule path.

This runtime never loads Torch or a teacher. It does not select a production model.
"""
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from .factorized_field_features import pair_inputs, DIM, NUMERIC_NAMES, VERSION as FEATURES
from .local_field_student import LocalFieldStudent, LABELS

VERSION = 'factorized-field-student-r4-v1'
SHAPES = {'question': (DIM, 16), 'evidence': (DIM, 16),
          'hidden.weight': (32, 64 + len(NUMERIC_NAMES)), 'hidden.bias': (32,),
          'output.weight': (3, 32), 'output.bias': (3,)}


class FactorizedFieldStudent(LocalFieldStudent):
    def __init__(self, config, arrays):
        if (config.get('version') != VERSION or config.get('feature_version') != FEATURES
                or config.get('labels') != list(LABELS) or not .5 <= config.get('threshold', 0) <= .99):
            raise ValueError('Unsupported factorized model or threshold')
        if set(arrays) != set(SHAPES):
            raise ValueError('Unexpected parameter names')
        for key, shape in SHAPES.items():
            if arrays[key].shape != shape or arrays[key].dtype != np.float32 or not np.isfinite(arrays[key]).all():
                raise ValueError('Invalid parameter shape, dtype or finite values: ' + key)
        self.model = config
        self.arrays = arrays

    @classmethod
    def load(cls, bundle):
        folder = Path(bundle)
        manifest = json.loads((folder / 'manifest.json').read_text('utf-8'))
        if set(manifest) != {'student_config.json', 'weights.npz'}:
            raise ValueError('Unexpected model manifest')
        for name, h in manifest.items():
            p = folder / name
            if p.stat().st_size > 4 * 1024**2 or hashlib.sha256(p.read_bytes()).hexdigest() != h:
                raise ValueError('Model file size or SHA256 mismatch')
        with zipfile.ZipFile(folder / 'weights.npz') as z:
            if sum(i.file_size for i in z.infolist()) > 4 * 1024**2:
                raise ValueError('Expanded parameters exceed deployment budget')
        with np.load(folder / 'weights.npz', allow_pickle=False) as loaded:
            arrays = {k: loaded[k].copy() for k in loaded.files}
        return cls(json.loads((folder / 'student_config.json').read_text('utf-8')), arrays)

    def logits(self, query, text, scope):
        x = pair_inputs(query, text, scope)
        pooled = []
        for key in ('question', 'evidence'):
            indices = sorted(x[key])
            values = np.asarray([x[key][i] for i in indices], dtype=np.float32)
            pooled.append((self.arrays[key][indices] * values[:, None]).sum(axis=0))
        q, e = pooled
        v = np.concatenate((q, e, q * e, np.abs(q - e), np.asarray(x['numeric'], dtype=np.float32)))
        h = np.maximum(self.arrays['hidden.weight'] @ v + self.arrays['hidden.bias'], 0)
        return self.arrays['output.weight'] @ h + self.arrays['output.bias']

    def predict_pair(self, query, text, scope):
        logits = self.logits(query, text, scope).astype(np.float64)
        scores = np.exp(logits - logits.max())
        return (scores / scores.sum()).tolist()

    def review(self, payload):
        result = super().review(payload)
        result['version'] = VERSION
        return result
