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
| `/api/v1/normalize/excel` | POST | 上传 xlsx 归一（详见 [Excel 上传](#excel-上传)） |
| `/api/v1/admin/reload` | POST | 热更新词库（受 ADMIN_TOKEN 保护，详见下） |

## 管理端鉴权

`/api/v1/admin/reload` 默认仅允许 loopback（`127.0.0.1`、`::1`）访问；如需远程调用，
设置环境变量 `ADMIN_TOKEN=<secret>`，然后用 Bearer header 调用：

```bash
curl -X POST -H "Authorization: Bearer <secret>" http://host:8000/api/v1/admin/reload
```

未携带或 token 不匹配时返回 `401 {"error": "unauthorized"}`。

## Excel 上传

`POST /api/v1/normalize/excel` 接受 xlsx 批量归一。Web UI 直接打开 `http://localhost:8000/` 即可使用。

**输入格式（multipart upload，field=`file`）：**
- 列 0 = 原词（必填）
- 列 1 = 极性提示（可选：`正面` / `负面` / 其他；仅为信息回显，**不影响匹配极性**）
- 无表头（按数据读取）
- 第 0 列为空 / 空白 / NaN 的行会被跳过
- 上限 50,000 行；超过返回 `400 {"error": "too_many_rows", "limit": 50000}`

**查询参数：**
- `model=bge`（默认） / `fasttext`
- `debug=0`（当前为对齐参数，暂无差异）

**输出格式（xlsx，列顺序固定）：**
1. `原词` —— 输入原词
2. `归一词` —— 归一后的标准词
3. `命中层级` —— `L1` / `L2` / `L3` / `FALLBACK`
4. `分数` —— 相似度，保留 4 位小数
5. `输入极性` —— 透传输入列 1 的内容

响应头中携带 `X-Summary`（JSON 字符串），供前端展示层级分布：

```
X-Summary: {"total": 123, "L1": 80, "L2": 30, "L3": 5, "FALLBACK": 8}
```

## 归一层级

每条结果标注命中层级：
- `L1`：别名表 / 词库精确命中
- `L2`：BGE/fastText 余弦相似度 ≥ 接受阈值（BGE 0.7, fastText 0.6）
- `L3`：L2 落入「接受-降级」区间（BGE 0.5–0.7, fastText 0.4–0.6）时，按编辑距离比率 ≤ 0.3 兜底
- `FALLBACK`：未匹配，返回原词

## 开发

```bash
# 运行测试
pytest -v

# 跑 95% 准确率验收
pytest tests/test_accuracy.py -v -s
```

## 调优

- 阈值在 `app/normalizer.py:Normalizer.THRESHOLDS`：
  - `accept` = L2 接受阈值（BGE 0.7, fastText 0.6）
  - `fallback_to_edit` = L3 降级阈值（BGE 0.5, fastText 0.4）
- 编辑距离比率 `Normalizer.EDIT_DISTANCE_RATIO = 0.3`
- 运营期失败用例直接补到 `data/aliases.json` 即可
