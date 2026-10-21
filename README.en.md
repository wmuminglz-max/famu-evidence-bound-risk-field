# An Evidence-Bound Risk Field Method for Contract Review

**面向合同审查的证据绑定风险场方法**

[中文](README.md)

This repository provides de-identified data, annotations, experimental code, training and evaluation protocols, and archived predictions for *An Evidence-Bound Risk Field Method for Contract Review*. The materials support inspection of experimental settings, evidence references, and the paper's main reported scores.

## Documentation

| Topic | 中文 | English |
|---|---|---|
| Repository overview | [概览](README.md) | [This document](README.en.md) |
| Dataset | [数据集说明](01_Dataset/README_数据集说明.md) | [Dataset guide](01_Dataset/README.en.md) |
| Reproduction materials | [复现指南](02_Reproduction/README_先读.md) | [Reproduction guide](02_Reproduction/README.en.md) |
| Reproduction objects and fingerprints | [S2.8 对象清单](02_Reproduction/docs/S2.8_对象与复现范围.md) | [Objects and fingerprints](02_Reproduction/docs/S2.8_objects_and_reproduction_scope.en.md) |
| De-identification | [脱敏说明](02_Reproduction/docs/脱敏说明.md) | [De-identification](02_Reproduction/docs/deidentification.en.md) |
| Execution and environment | [运行指南](02_Reproduction/docs/运行与环境.md) | [Execution and environment](02_Reproduction/docs/execution_and_environment.en.md) |
| R11 student models | [实验结果](02_Reproduction/experiments/student/R11_中文结果_连续跨度与训练回放.md) | [Experimental results](02_Reproduction/experiments/student/R11_results.en.md) |
| File integrity | [校验指南](GITHUB_UPLOAD_NOTES.zh-CN.md) | [Integrity verification](GITHUB_UPLOAD_NOTES.md) |

## Directory structure

```text
famu-evidence-bound-risk-field/
├── README.md
├── README.en.md
├── MANIFEST_SHA256.json
├── GITHUB_UPLOAD_NOTES.zh-CN.md
├── GITHUB_UPLOAD_NOTES.md
├── 01_Dataset/          Data, annotations, splits, and provenance indexes
└── 02_Reproduction/     Experimental code, protocols, archived predictions, and scoring tools
```

## Dataset

| Location relative to `01_Dataset/` | Contents and size |
|---|---|
| `01_Professional_Training_Data/` | 77 professional training tasks, 9 internal development tasks, and 1,657 auxiliary training tasks in the labor domain. |
| `02_Experiment_1_Conditional_Judgment/` | 1,213 mixed-task records, 576 pairing relations, and 256 rephrasing stress-test records; Experiment 1, V5. |
| `03_Experiment_2_Evidence_Binding/` | 672 synthetic scenarios: 432 training, 96 development, and 144 test examples; also includes professional-task replay, historical regression data, and frozen-version data; Experiment 2, V8. |
| `04_Real_Material_Evaluation/` | Inputs for 28 questions and gold answers for 87 scored items. The paper's main evaluation uses 17 questions, 53 items, and 5 source groups. |
| `05_Student_Model_Data/` | Relation pairs, paragraph and span annotations, new and historical scenarios, and regression data. `new_scenes.json` contains 1,152 training examples, 72 development examples, and 96 test examples each for long-sentence, short-sentence, and reordered variants. |

See the [data file inventory](01_Dataset/数据文件清单.csv) for per-file counts and [sample splits and identifiers](01_Dataset/样本划分与编号.jsonl) for assignments. Files may contain different versions of the same samples, so their record counts should not be added directly. The 9 internal development tasks also serve as historical regression tasks in later experiments; consult each protocol for their role.

`target`, `states`, gold `bindings`, and gold answers provide supervision or scoring information. Keep these fields separate from test-time model inputs. For real-material evaluation, read `public_inputs.jsonl` and `gold_keys.jsonl` separately. Preserve group identifiers across paired, reordered, and rephrased variants.

## Experiments and reproduction

`02_Reproduction/experiments/` contains V5 conditional judgment, V8 evidence binding, real-material evaluation, 27B reference configurations, and student-model experiments. Code, configurations, and results are stored by experimental version.

Using Python 3.10 or later, run from the repository root:

```bash
cd 02_Reproduction
python -B tools/verify_release.py
python -B tools/recompute_results.py
```

These tools use only the Python standard library. The first checks file hashes, data formats, question references, and evidence positions. The second recomputes the main scores from archived predictions and scoring keys, writing `validation/recomputed_results.json`. Verify integrity before recomputing scores; use a working copy if you need to preserve the directory's checksum state.

`01_Dataset/` provides a standalone data entry point. `02_Reproduction/` supplies the data paths used by the code. The [provenance and checksum mapping](01_Dataset/文件来源与校验.json) identifies the corresponding files.

| Use | Required materials |
|---|---|
| Recompute archived scores | Included predictions, scoring keys, and the Python standard library. |
| Train or run inference on de-identified inputs | Compatible models, initialization objects, and execution environments; report results as new experiments on derived data. |
| Exactly rerun the original experiments | Controlled original weights, caches, unchanged original inputs, and the corresponding environment. |

See the [execution guide](02_Reproduction/docs/execution_and_environment.en.md) for entry points and environments, and the [S2.8 object inventory](02_Reproduction/docs/S2.8_objects_and_reproduction_scope.en.md) for versions and fingerprints.

## Data protection and licensing

Names, organizations, direct contact details, and other identifiers have been de-identified. Dates, amounts, percentages, and logical relationships required by the tasks are retained. See the [de-identification guide](02_Reproduction/docs/deidentification.en.md) for details. Retained task information may still allow sources to be linked; de-identification does not establish irreversible anonymization.

The repository excludes identity mappings, original contract attachments, original expert sign-off documents, model weights, and feature caches. `models/` contains version locks, architecture metadata, and controlled-object fingerprints. Third-party materials remain subject to their existing licenses and usage conditions.

## File verification

The root [MANIFEST_SHA256.json](MANIFEST_SHA256.json) lists repository-relative paths, sizes, and SHA256 hashes, excluding the manifest itself. Each subfolder's manifest uses that subfolder as its root. See [file integrity verification](GITHUB_UPLOAD_NOTES.md) for the procedure.
