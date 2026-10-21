# Dataset guide

[中文](README_数据集说明.md) · [Repository overview](../README.en.md)

This directory provides de-identified experimental data, annotations, sample splits, and provenance indexes for the paper.

## Contents

| Directory | Contents and size |
|---|---|
| `01_Professional_Training_Data/` | 77 professional training tasks, 9 internal development tasks, and 1,657 auxiliary training tasks in the labor domain. |
| `02_Experiment_1_Conditional_Judgment/` | 1,213 mixed-task records, 576 pairing relations, and 256 rephrasing stress-test records. |
| `03_Experiment_2_Evidence_Binding/` | 672 synthetic scenarios: 432 training, 96 development, and 144 test examples; also includes 77 professional replay tasks, 9 historical regression tasks, and professional data from frozen versions. |
| `04_Real_Material_Evaluation/` | `public_inputs.jsonl` contains 28 question inputs; `gold_keys.jsonl` contains answers and sufficient evidence sets for 87 scored items. The main evaluation uses 17 questions, 53 items, and 5 source groups. |
| `05_Student_Model_Data/` | Relation pairs, paragraph and span annotations, new and historical scenarios, and regression data. `new_scenes.json` contains 1,152 training examples, 72 development examples, and 96 test examples each for long-sentence, short-sentence, and reordered variants. |

The [data file inventory](数据文件清单.csv) gives per-file counts, internal collections, and split statistics. [Sample splits and identifiers](样本划分与编号.jsonl) provides identifiers, splits, groups, and condition ordering. An empty split field means that the source file does not specify a split. Real-material input splits are linked to the corresponding answer records.

## Inputs, annotations, and evaluation

Each non-empty JSONL line is one record. JSON files may contain arrays or records organized into internal collections. `target`, `states`, gold `bindings`, and gold answers provide supervision or scoring information; keep them separate from test-time model inputs. Read real-material inputs and answers from separate files.

Samples may appear in several experiments or wording variants. Count independent samples using identifiers such as `pair_group`, and preserve the training, development, and test splits. Experiment 1's 1,213 mixed records span several tasks. The complete real-material collection has 28 questions and 87 items, while the main evaluation has 17 questions and 53 items. The 9 professional development tasks serve as historical regression tasks in later experiments.

## De-identification and provenance

Names, organizations, and sources use stable aliases. Direct contact details, detailed addresses, bank accounts, registration numbers, and similar identifiers have been processed. Dates, amounts, percentages, and logical relationships needed by the tasks are retained. See [de-identification](../02_Reproduction/docs/deidentification.en.md) for details.

De-identified text differs from the original experimental inputs. Training or inference on this text should be reported as a new experiment on derived data. Expert answers correspond to reviews of the original questions. Retained amounts and dates may still support source linkage.

The [provenance and checksum mapping](文件来源与校验.json) identifies data locations and fingerprints within the reproduction materials. [MANIFEST_SHA256.json](MANIFEST_SHA256.json) lists file sizes and SHA256 hashes for this directory, excluding itself. Each experiment's data-level `manifest.json` describes that experiment's data organization.

See the [reproduction guide](../02_Reproduction/README.en.md) for training protocols, scoring scripts, code, and model versions. Third-party data remain subject to their existing licenses and usage conditions.
