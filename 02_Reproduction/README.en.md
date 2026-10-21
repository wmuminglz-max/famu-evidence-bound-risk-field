# Reproduction guide

[中文](README_先读.md) · [Repository overview](../README.en.md)

This directory provides experimental code, protocols, de-identified inputs, scoring keys, and archived predictions for inspecting the method and recomputing results.

## Quick start

From this directory, use Python 3.10 or later:

```bash
python -B tools/verify_release.py
python -B tools/recompute_results.py
```

Both tools use only the Python standard library. Integrity verification checks file SHA256 hashes, data formats, question references, and evidence positions. Score recomputation reads archived predictions and scoring keys and writes `validation/recomputed_results.json`. Verify integrity first; recompute in a directory copy if you need to preserve the manifest's checksum state.

## Contents

| Path | Contents |
|---|---|
| [Objects and reproduction scope](docs/S2.8_objects_and_reproduction_scope.en.md) | Required objects, version locks, fingerprints, and controlled materials. |
| [De-identification](docs/deidentification.en.md) | Data protection, evidence positions, and implications for reproduction. |
| [Execution and environment](docs/execution_and_environment.en.md) | Score recomputation, model inference, and training entry points. |
| `experiments/v5/` | Experiment 1: conditional judgment code, data, M0–M4 results across seeds, and feature extraction settings. |
| `experiments/v8/` | Experiment 2: evidence binding code, 432/96/144 synthetic examples, 77 replay tasks, and results from six runs across two branches. |
| `experiments/real_eval/` | Full-answer generation, evidence interfaces, student inference, scoring code, and archived predictions. |
| `data/real/` | 28 question inputs and 87 scoring items; the main evaluation has 17 questions and 53 items. |
| `data/professional/` | 77 professional tasks, 9 internal development tasks, and 1,657 auxiliary labor-domain tasks. |
| `experiments/27b/` | The 27B architecture, training implementation, and protocols and results for three reference configurations. |
| `experiments/student/` | R11 data, code, four training configurations, and results for long, short, reordered, and historical regression inputs. |
| [R11 results](experiments/student/R11_results.en.md) | Complete results and limitations for contiguous spans and training-data replay. |
| `models/` | The 9B version lock, 27B architecture metadata, and inventories of controlled weights and caches. |
| `results/real/` | Field-level scoring, system scores, and source-group statistics. |
| `manifests/` | Data splits, condition ordering, source fingerprints, and de-identification records. |
| `validation/` | Integrity checks and score recomputation records. |

## Reproduction scope

1. **Recompute archived scores:** use included predictions and scoring keys without model weights or a GPU.
2. **Train or run inference on de-identified inputs:** prepare compatible models and environments, and report results as new experiments on derived data.
3. **Exactly rerun original experiments:** obtain controlled original weights, caches, unchanged inputs, and the corresponding environment.

V5 corresponds to Experiment 1 and V8 to Experiment 2. Use each version's frozen code and protocol. Preserve sample groups, seeds, and task denominators during evaluation. Interpret full-answer and state-only metrics separately.

## Material availability

The repository provides de-identified derived materials. Identity mappings, original contract attachments, original expert sign-offs, original weights, and feature caches are controlled objects. Their inventories provide roles, sizes, SHA256 hashes, and verification sources. Third-party materials remain subject to their existing licenses and usage conditions.
