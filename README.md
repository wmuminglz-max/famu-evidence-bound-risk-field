# 面向合同审查的证据绑定风险场方法

**An Evidence-Bound Risk Field Method for Contract Review**

[English](README.en.md)

本仓库提供论文《面向合同审查的证据绑定风险场方法》的脱敏数据、标注、实验代码、训练与评测协议及归档预测，支持检查实验设置、核对证据引用和复算主要结果。

## 文档索引

| 内容 | 中文 | English |
|---|---|---|
| 仓库概览 | [本文](README.md) | [Overview](README.en.md) |
| 数据集 | [数据集说明](01_Dataset/README_数据集说明.md) | [Dataset guide](01_Dataset/README.en.md) |
| 复现材料 | [复现指南](02_Reproduction/README_先读.md) | [Reproduction guide](02_Reproduction/README.en.md) |
| 复现对象与指纹 | [S2.8 对象清单](02_Reproduction/docs/S2.8_对象与复现范围.md) | [Objects and fingerprints](02_Reproduction/docs/S2.8_objects_and_reproduction_scope.en.md) |
| 脱敏范围 | [脱敏说明](02_Reproduction/docs/脱敏说明.md) | [De-identification](02_Reproduction/docs/deidentification.en.md) |
| 运行与环境 | [运行指南](02_Reproduction/docs/运行与环境.md) | [Execution and environment](02_Reproduction/docs/execution_and_environment.en.md) |
| R11 学生模型 | [实验结果](02_Reproduction/experiments/student/R11_中文结果_连续跨度与训练回放.md) | [Experimental results](02_Reproduction/experiments/student/R11_results.en.md) |
| 文件完整性 | [校验指南](GITHUB_UPLOAD_NOTES.zh-CN.md) | [Integrity verification](GITHUB_UPLOAD_NOTES.md) |

## 目录结构

```text
famu-evidence-bound-risk-field/
├── README.md
├── README.en.md
├── MANIFEST_SHA256.json
├── GITHUB_UPLOAD_NOTES.zh-CN.md
├── GITHUB_UPLOAD_NOTES.md
├── 01_Dataset/          数据、标注、划分与来源索引
└── 02_Reproduction/     实验代码、协议、归档预测与复算工具
```

## 数据集

| 位置（相对于 `01_Dataset/`） | 内容与规模 |
|---|---|
| `01_Professional_Training_Data/` | 77 条专业训练任务、9 条内部开发任务、1,657 条劳动辅助训练任务。 |
| `02_Experiment_1_Conditional_Judgment/` | 1,213 条混合任务记录、576 条配对关系，以及 256 条新表述压力测试记录；对应实验一 V5。 |
| `03_Experiment_2_Evidence_Binding/` | 672 条合成场景：训练 432 条、开发 96 条、测试 144 条；另含专业回放、历史回归及冻结版本数据；对应实验二 V8。 |
| `04_Real_Material_Evaluation/` | 28 题输入及 87 个评分项的标准答案；论文正式主实验使用其中 17 题、53 项、5 个来源组。 |
| `05_Student_Model_Data/` | 关系对、段落与片段标注、新场景、历史场景及回归数据；`new_scenes.json` 含训练 1,152 条、开发 72 条，以及长句、短句、改序测试各 96 条。 |

逐文件统计见[数据文件清单](01_Dataset/数据文件清单.csv)，样本编号与划分见[样本划分与编号](01_Dataset/样本划分与编号.jsonl)。多个文件可能包含同一批样本的不同版本，记录数不宜直接相加。专业数据中的 9 条内部开发任务在后续实验中也用于历史回归，具体用途以各实验协议为准。

`target`、`states`、标准 `bindings` 和标准答案用于监督或评分。测试时的模型输入应与这些字段分开；真实材料评测分别读取 `public_inputs.jsonl` 和 `gold_keys.jsonl`。配对、改序及不同表述版本应保留原有组编号。

## 实验与复现

`02_Reproduction/experiments/` 包含 V5 条件判断、V8 证据绑定、真实材料评测、27B 参照配置和学生模型实验。各实验的代码、配置与结果按版本分别保存。

使用 Python 3.10 或更高版本，在仓库根目录执行：

```bash
cd 02_Reproduction
python -B tools/verify_release.py
python -B tools/recompute_results.py
```

上述工具仅依赖 Python 标准库。第一条命令检查文件哈希、数据格式、题目引用及证据位置；第二条从归档预测和评分键复算主要成绩，写入 `validation/recomputed_results.json`。先执行完整性核验，再复算成绩；如需保持工作目录的校验状态，可在副本中复算。

`01_Dataset/` 提供独立的数据入口，`02_Reproduction/` 提供代码使用的数据路径。[文件来源与校验](01_Dataset/文件来源与校验.json)列出两处数据的对应关系。

| 使用方式 | 所需材料 |
|---|---|
| 复算归档成绩 | 包内预测、评分键和 Python 标准库。 |
| 在脱敏输入上训练或推理 | 配套模型、初始化对象和相应运行环境；结果应作为派生数据上的新实验报告。 |
| 精确重跑原始实验 | 受控原始权重、缓存、未改动的原始输入及对应环境。 |

运行入口与环境要求见[运行指南](02_Reproduction/docs/运行与环境.md)，对象版本与指纹见[S2.8 对象清单](02_Reproduction/docs/S2.8_对象与复现范围.md)。

## 数据保护与许可

材料中的姓名、机构和直接联系方式等已去标识化；判断所需的日期、金额、比例及逻辑关系予以保留。详细范围见[脱敏说明](02_Reproduction/docs/脱敏说明.md)。保留的任务信息仍可能用于关联来源，去标识化不等于不可逆匿名化。

仓库不包含身份映射、原始合同附件、专家签认原件、模型权重或特征缓存。`models/` 提供版本锁、架构元数据和受控对象指纹。第三方材料适用其原有许可与使用条件。

## 文件校验

根目录的 [MANIFEST_SHA256.json](MANIFEST_SHA256.json) 记录仓库文件的相对路径、大小和 SHA256，清单自身除外。两个子目录中的同名清单分别以所在目录为根。校验步骤见[文件完整性指南](GITHUB_UPLOAD_NOTES.zh-CN.md)。
