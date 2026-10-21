# An Evidence-Bound Risk Field Method for Contract Review

## Dataset and Reproduction Materials (De-identified Release)

**Chinese title: 面向合同审查的证据绑定风险场方法**

[中文说明 / Chinese README](README.md)

This folder brings together the dataset and reproduction materials accompanying *An Evidence-Bound Risk Field Method for Contract Review* (《面向合同审查的证据绑定风险场方法》). It allows readers to inspect the data and annotations, examine the experimental settings, check archived predictions, and recompute the paper's main reported scores.

The two existing de-identified releases from the “法眸智审” (Famu) project have been organized into one companion package. “法眸智审” is the project name used in the source materials; the Chinese and English paper titles above identify the paper associated with this package.

- **Source release date:** October 16, 2026.
- **Combined package date:** October 21, 2026.
- **Packaging method:** The materials were initially copied byte for byte. The combined folder and its subfolders now use English names, with documentation, path indexes, and checksum manifests updated accordingly. Experimental data, annotations, splits, and frozen code are unchanged. Existing source-document filenames are retained. Package-level documentation, a combined checksum manifest, and an assembly record are included.
- **Reproduction scope:** The package supports inspecting de-identified data, reviewing code and protocols, and recomputing scores from archived predictions. Exact reruns of the original experiments require controlled-access weights, caches, and unchanged original inputs.

Directory names use English, and the project name is rendered as **Famu**. Existing source-document filenames retain their original spelling. Use the paths shown below as written. Linked source documents remain in their original language.

## 1. Directory Structure

```text
Famu_Evidence_Bound_Risk_Field_Materials_20261021/
├── README.md                         Chinese package overview
├── README.en.md                      English package overview (this document)
├── MANIFEST_SHA256.json               File sizes and SHA256 hashes for the combined package
├── 整理与校验记录.json                 Source directories, copy counts, and verification results
├── 01_Dataset/                         Data, annotations, splits, and provenance
│   ├── README_数据集说明.md
│   ├── 01_Professional_Training_Data/
│   ├── 02_Experiment_1_Conditional_Judgment/
│   ├── 03_Experiment_2_Evidence_Binding/
│   ├── 04_Real_Material_Evaluation/
│   ├── 05_Student_Model_Data/
│   ├── 数据文件清单.csv
│   ├── 样本划分与编号.jsonl
│   ├── 文件来源与校验.json
│   ├── 整理校验记录.json
│   └── MANIFEST_SHA256.json
└── 02_Reproduction/                       Experimental code, protocols, archived results, and tools
    ├── README_先读.md
    ├── data/
    ├── docs/
    ├── experiments/
    ├── manifests/
    ├── models/
    ├── results/
    ├── tools/
    ├── validation/
    └── MANIFEST_SHA256.json
```

`01_Dataset/` preserves all 45 files from the source dataset release. `02_Reproduction/` preserves all 1,093 files from the source reproduction release. In total, 1,138 original files were copied. These file counts describe package completeness, not the number of independent samples.

## 2. Suggested Reading Order

1. Read this overview for the paper association, package contents, and reproduction scope.
2. Consult the [dataset documentation](01_Dataset/README_数据集说明.md) and [data file inventory](01_Dataset/数据文件清单.csv) to identify the data you need.
3. Read the [reproduction release overview](02_Reproduction/README_先读.md), [objects and reproduction scope](02_Reproduction/docs/S2.8_对象与复现范围.md), and [execution and environment guide](02_Reproduction/docs/运行与环境.md) to check the experimental protocols.
4. Run integrity verification and score recomputation as needed. Before retraining or running model inference, separately prepare the models and environments required by the relevant protocols.

## 3. Dataset Contents

| Location relative to `01_Dataset/` | Contents and size |
|---|---|
| `01_Professional_Training_Data/` | 77 professional training tasks, 9 internal development tasks, and 1,657 auxiliary training tasks in the labor domain. |
| `02_Experiment_1_Conditional_Judgment/` | 1,213 mixed-task records, 576 pairing relations, and 256 rephrasing stress-test records. Corresponds to V5 in the reproduction materials. |
| `03_Experiment_2_Evidence_Binding/` | 672 synthetic scenarios: 432 for training, 96 for development, and 144 for testing. Also preserves professional-task replay data, historical regression data, and data copies from frozen versions. Corresponds to V8. |
| `04_Real_Material_Evaluation/` | Inputs for 28 questions and gold answers covering 87 scored items. The paper's main evaluation uses a subset of 17 questions, 53 scored items, and 5 source groups. |
| `05_Student_Model_Data/` | Relation pairs, paragraph and span annotations, new and historical scenarios, and multiple rounds of regression data. `new_scenes.json` contains 1,152 training examples, 72 development examples, and 96 test examples each for long-sentence, short-sentence, and reordered variants. |

See the [data file inventory](01_Dataset/数据文件清单.csv) and [sample splits and identifiers](01_Dataset/样本划分与编号.jsonl) for record counts, internal collections, and split assignments. Each non-empty JSONL line contains one record. JSON files may contain arrays or collections organized by internal subsets.

**Preserve the original experimental splits when using these data.** `target`, `states`, gold `bindings`, and gold answers provide supervision or scoring information and must not be included in model inputs at test time. For the evaluation on real materials, read `public_inputs.jsonl` and `gold_keys.jsonl` separately. Paired, reordered, and rephrased variants may share an underlying question; retain the original group identifiers.

## 4. Reproduction Materials

| Location relative to `02_Reproduction/` | Purpose |
|---|---|
| `docs/` | Reproduction scope, de-identification details, execution levels, and environment requirements. |
| `experiments/v5/` | Frozen code, data, and results across configurations and random seeds for Experiment 1: conditional judgment. |
| `experiments/v8/` | Code, training schedules, data, and archived outputs for Experiment 2: evidence binding. |
| `experiments/real_eval/` | Answer generation, evidence interfaces, student-model inference, scoring code, and archived predictions for the evaluation on real materials. |
| `experiments/27b/` | The 27B architecture, training implementation, reference-configuration protocols, and archived results. |
| `experiments/student/` | Student-model code, data, configurations, and results for wording variants and regression tests. |
| `data/` | Professional-task data, real-material inputs, and scoring keys retained in the reproduction release. |
| `models/` | Model version locks, architecture metadata, historical fingerprints, and inventories of controlled-access objects. Model weights are not included. |
| `results/`, `validation/` | Archived scores, field-level scoring, source-release checks, and score recomputation records. |
| `manifests/` | Data splits, condition ordering, provenance fingerprints, and de-identification records. |
| `tools/` | Tools for integrity verification, score recomputation from archived predictions, and controlled-access object verification. |

### Integrity Verification and Score Recomputation Without Model Weights

Open a terminal in the combined package folder and use Python 3.10 or later:

```bash
cd "02_Reproduction"
python -B tools/verify_release.py
python -B tools/recompute_results.py
```

These tools use only the Python standard library. The first Python command verifies file hashes, data formats, question references, and evidence positions within the reproduction subfolder. The second recomputes the main scores from archived predictions and scoring keys, writing the output to `validation/recomputed_results.json`. Neither tool loads a model, accesses the network, or starts training. The `-B` option prevents bytecode cache files from affecting manifest verification.

Score recomputation writes to an archived result file. To preserve the delivered package's checksum state, first copy `02_Reproduction/` to a working directory and run the commands there. Run integrity verification before score recomputation. See the [execution and environment guide](02_Reproduction/docs/运行与环境.md) and individual experimental protocols for model inference and retraining requirements.

## 5. Relationship Between the Subfolders and Reproduction Scope

`01_Dataset/` provides a separate data-oriented entry point extracted from the original de-identified reproduction release. `02_Reproduction/` preserves the data paths and archive structure used by the code. Duplicate data files are intentionally retained. The [file provenance and checksum mapping](01_Dataset/文件来源与校验.json) links dataset files to their locations in the reproduction release. Keeping these copies preserves frozen-code paths and historical checksum results.

Do not add record counts across the two subfolders or across experimental copies to report a total dataset size. The 1,213 mixed records in Experiment 1 are not 1,213 independent, manually annotated contract-review questions. Likewise, the complete collection of 28 real-material questions and 87 scored items is larger than the paper's main evaluation subset.

| Use | Support provided by this package |
|---|---|
| Recomputing archived scores | Included predictions and scoring keys support recomputation without a GPU or model weights. |
| Retraining or inference on de-identified inputs | Requires separately prepared models and environments. Because the text has been de-identified, report these runs as new experiments on derived data. |
| Exact reruns of the original experiments | Requires controlled-access original weights, caches, unchanged inputs, and the corresponding environment. This package does not contain all required original objects. |

This packaging work involved organizing and verifying existing materials. It did not add training runs, model inference, expert reviews, or experimental conclusions. Adding this English document does not change the experimental materials.

## 6. De-identification, Provenance, and Licensing

This package retains the de-identification applied in the October 16, 2026 release. Packaging and translation did not rewrite data, code, or annotations. Names, organizations, contact details, and other identifiers were processed under the source release's de-identification procedure. Dates, amounts, percentages, and logical relationships needed for the tasks were retained; see the [de-identification documentation](02_Reproduction/docs/脱敏说明.md). This is a de-identified release and does not establish that its sources cannot be re-identified.

The package excludes identity-mapping tables, original contract attachments, original expert sign-off documents, original model weights, and feature caches. Inventories and fingerprints in `models/` describe and support verification of controlled-access objects; they do not mean that the corresponding weights are publicly available.

The source directory names are:

- `法眸智审_数据集_脱敏版_20261016`
- `法眸智审_论文复现材料_脱敏版_20261016`

Both source directories have been retained. Combining the materials and adding documentation does not grant additional redistribution rights for code or data. Third-party materials remain subject to their existing licenses and conditions. No DOI, public repository, external download link, or application channel for controlled-access materials was created.

## 7. Checksum and Verification Files

- The root [MANIFEST_SHA256.json](MANIFEST_SHA256.json) records the relative path, byte size, and SHA256 hash of every file in the combined package except the root manifest itself. Paths are relative to the combined package folder.
- The manifests with the same filename inside the two subfolders each apply only within their own subfolder. The dataset manifest has been updated for the English directory names; the reproduction manifest is unchanged.
- The [assembly and verification record](整理与校验记录.json) records source file counts, copy consistency, verification against the source manifests, correspondence between duplicate data files, and results from the reproduction release's verification tool. It also records the English directory names, associated index updates, and the addition of the English documentation.
- Existing records in `01_Dataset/整理校验记录.json` and `02_Reproduction/validation/` belong to the source release. They have not been rewritten as records of new experiments.

The package-level documentation organizes and explains the companion materials. Consult the preserved protocols and manifests for specific data splits, training configurations, and scoring definitions.
