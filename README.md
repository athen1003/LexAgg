# 中文词归一化服务

把人工输入的中文标签词归一到标准词库。Web API 上传 txt，每行一词，返回归一结果。

## 快速开始

```bash
# 1. 安装
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt

# 2. 下载 fastText 中文模型
python scripts/download_model.py

# 3. 启动服务
uvicorn app.main:app --port 8000

# 4. 调用
curl -F "file=@input.txt" "http://localhost:8000/api/v1/normalize?debug=1" -o output.txt
```

无 fastText 模型也可临时验证路由（使用桩 embedding）：

```bash
python scripts/smoke_test_server.py
```

## 词库格式

`data/vocabulary.csv`：两列 `词,极性`，极性取 `正面` 或 `负面`。

`data/aliases.json`：变体词 → 标准词的显式映射（运营期人工维护）。

## API

| 端点 | 方法 | 说明 |
|---|---|---|
| `/api/v1/health` | GET | 健康检查 |
| `/api/v1/normalize` | POST | 上传 txt 归一。`?model=fasttext\|bge&debug=0\|1` |
| `/api/v1/admin/reload` | POST | 热更新词库 |

## 归一层级

每条结果标注命中层级：
- `L1`：别名表 / 词库精确命中
- `L2`：fastText 余弦相似度 ≥ 阈值（默认 0.6）
- `L3`：编辑距离比率 ≤ 0.3
- `FALLBACK`：未匹配，返回原词

## 开发

```bash
# 运行测试
pytest -v

# 跑 95% 准确率验收
pytest tests/test_accuracy.py -v -s
```

## 调优

- 阈值在 `app/normalizer.py:Normalizer.THRESHOLDS`
- 编辑距离比率 `Normalizer.EDIT_DISTANCE_RATIO = 0.3`
- 运营期失败用例直接补到 `data/aliases.json` 即可
