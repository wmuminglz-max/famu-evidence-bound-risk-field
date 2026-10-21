# 文件完整性核验

[English](GITHUB_UPLOAD_NOTES.md) · [仓库概览](README.md)

## 校验清单

仓库根目录的 `MANIFEST_SHA256.json` 记录全部文件的相对路径、字节数和 SHA256，清单自身除外。`01_Dataset/` 和 `02_Reproduction/` 中的同名清单分别以所在子目录为根。

在仓库根目录使用 Python 3.10 或更高版本，检查根目录清单中的每个文件：

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

输出为通过核验的文件数。清单中的大小或哈希不一致时，断言会显示对应文件路径。

## 数据与引用核验

从仓库根目录执行：

```bash
cd 02_Reproduction
python -B tools/verify_release.py
```

该工具检查复现目录的文件集合、文件哈希、Python/JSON/JSONL 格式、题目引用、证据字符位置与来源指纹。文档版本和校验清单应配套使用。实验归档中的历史哈希描述对应历史对象，当前仓库文件以所在目录的 `MANIFEST_SHA256.json` 为准。

成绩复算与完整性核验是不同步骤。`tools/recompute_results.py` 使用归档预测和评分键重新评分，结果见 `validation/recomputed_results.json`。模型权重、特征缓存和受控原始输入的范围见[对象清单](02_Reproduction/docs/S2.8_对象与复现范围.md)。
