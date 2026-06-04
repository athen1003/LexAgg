# 中文词归一化服务

把人工输入的中文标签词归一到标准词库。Web API 上传 txt，每行一词，返回归一结果。

## 快速开始

```bash
# 1. 安装
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt

# 2. （可选）首次启动会自动下载 BGE 模型到 models/bge/（~95MB）
#     如需用 fastText 备选模型，运行：python scripts/download_model.py

# 国内用户如 HuggingFace 联网慢，设置镜像：
#     export HF_ENDPOINT=https://hf-mirror.com

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
| `/api/v1/admin/reload` | POST | 热更新词库（受 ADMIN_TOKEN 保护，详见下） |

## 管理端鉴权

`/api/v1/admin/reload` 默认仅允许 loopback（`127.0.0.1`、`::1`）访问；如需远程调用，
设置环境变量 `ADMIN_TOKEN=<secret>`，然后用 Bearer header 调用：

```bash
curl -X POST -H "Authorization: Bearer <secret>" http://host:8000/api/v1/admin/reload
```

未携带或 token 不匹配时返回 `401 {"error": "unauthorized"}`。

## 归一层级

每条结果标注命中层级：
- `L1`：别名表 / 词库精确命中
- `L2`：BGE/fastText 余弦相似度 ≥ 阈值（BGE 0.7, fastText 0.6）
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
