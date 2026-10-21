# Execution and environment

[中文](运行与环境.md) · [Reproduction guide](../README.en.md)

## Score recomputation without models

From `02_Reproduction/`, use Python 3.10 or later:

```bash
python -B tools/verify_release.py
python -B tools/recompute_results.py
```

These standard-library tools read archived predictions and check the main scores using the scoring logic. Verify file integrity first. Recomputation writes `validation/recomputed_results.json`; use a working copy to preserve the checksum state. Read all R11 configurations, the three wording variants for new scenarios, and historical regressions together.

## Original-model inference

After obtaining controlled adapters and the compatible base model, check versions and fingerprints against `models/`. Experiment 2's reference object is the original base plus an unmerged adapter. Other export formats require independent equivalence checks.

The archived experiments primarily used Linux/CUDA, Python 3.10, PyTorch 2.11.0+cu130, Transformers 5.8.0, PEFT 0.21.1, Accelerate 1.15.0, and safetensors 0.7.0. The 27B model used bitsandbytes 0.50.2 for NF4 double-quantized loading. Consult each configuration's protocol and environment record for exact versions and loading settings.

V8 visible inputs contain `scope`, `fields`, `rule`, and `evidence`. Prompts are generated with `famu_v8.messages()` and parsed with `decode()`. `target`, `states`, and gold `bindings` provide supervision or scoring information. Full-answer and state-only evaluations use different prompts and their respective task denominators.

## Training entry points

Prepare separate paths for the controlled base, initial adapter, and new outputs. From `02_Reproduction/`:

```bash
cd experiments/v5
python scripts/run_famu_v5.py --base /path/to/base9b --data training_data/famu_v5_20261004 --out /path/to/new_v5_run
cd ../v8
python scripts/run_famu_v8.py --base /path/to/base9b --init /path/to/shared_initial_adapter --data training_data/famu_v8_20261005 --out /path/to/new_v8_run
```

These entry points perform training; `--out` must point to a new working directory. Re-extracting features or training on de-identified text constitutes a new run. Record the actual input, code, and object fingerprints. V8 branch protocols specify training schedules and sample weights.

Each historical version uses its own frozen implementation. The 27B expert router and student relation modules contain dedicated logic and require the corresponding implementations and configurations. Set base, data, and output paths through `--base`, `--data`, and `--out`.

## Interpreting results

Experiment 1 M3 improves on M0 by 46.22 percentage points; also report the three seeds' results of 2, 3, and 4 correct answers among 31 unknown questions. Experiment 2's perfect score on 144 questions belongs to the controlled experimental setting. The main real-material evaluation has 17 questions, 53 items, and 5 source groups. Student-model reporting should include long, short, reordered, and historical regression results; see [R11 results](../experiments/student/R11_results.en.md).
