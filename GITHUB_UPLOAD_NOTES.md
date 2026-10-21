# File integrity verification

[中文](GITHUB_UPLOAD_NOTES.zh-CN.md) · [Repository overview](README.en.md)

## Checksum manifests

The root `MANIFEST_SHA256.json` lists relative paths, byte sizes, and SHA256 hashes for repository files, excluding itself. The manifests in `01_Dataset/` and `02_Reproduction/` each use their own subdirectory as the root.

From the repository root, use Python 3.10 or later to check every file in the root manifest:

```python
from pathlib import Path
import hashlib
import json

root = Path.cwd()
manifest = json.loads((root / "MANIFEST_SHA256.json").read_text(encoding="utf-8"))
for item in manifest["files"]:
    path = root / item["path"]
    data = path.read_bytes()
    assert len(data) == item["bytes"], item["path"]
    assert hashlib.sha256(data).hexdigest() == item["sha256"], item["path"]
print(len(manifest["files"]))
```

The output is the number of verified files. A size or hash mismatch raises an assertion identifying the file.

## Data and reference verification

From the repository root:

```bash
cd 02_Reproduction
python -B tools/verify_release.py
```

This tool checks the reproduction directory's file set, hashes, Python/JSON/JSONL syntax, question references, evidence character positions, and source fingerprints. Use documentation and manifests from the same version. Historical hashes in experimental archives identify historical objects; the relevant directory's `MANIFEST_SHA256.json` describes current repository files.

Score recomputation is separate from integrity verification. `tools/recompute_results.py` scores archived predictions against the scoring keys and writes `validation/recomputed_results.json`. See the [object inventory](02_Reproduction/docs/S2.8_objects_and_reproduction_scope.en.md) for model weights, feature caches, and controlled original inputs.
