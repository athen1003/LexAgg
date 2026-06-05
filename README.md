# 中文词归一化服务

把人工输入的中文标签词归一到标准词库。Web API 上传 txt，每行一词，返回归一结果。

## 快速开始

```bash
# 1. 安装
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt

# 2. 首次启动会自动下载 M3E 模型到 models/m3e_base/（~400MB）
#    如需用 BGE-small 替代，运行：uvicorn 启动时设 model=bge
#    如需用 fastText 备选模型，运行：python scripts/download_model.py

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

`data/vocabulary.csv`：三列 `大类,词,极性`。`极性` 取 `正面` 或 `负面`，`大类` 必填（如 `体感` / `功能` / `质量` 等），用于组织词库并在输出的「归一-大类」列回填归一词所属的类别。

`data/aliases.json`：变体词 → 标准词的显式映射（运营期人工维护）。别名自动继承主词的大类。

## API

| 端点 | 方法 | 说明 |
|---|---|---|
| `/api/v1/health` | GET | 健康检查 |
| `/api/v1/normalize` | POST | 上传 txt 归一。`?model=m3e\|bge\|bge_base\|fasttext&debug=0\|1` |
| `/api/v1/normalize/excel` | POST | 上传 xlsx 归一（详见 [Excel 上传](#excel-上传)） |
| `/api/v1/normalize/excel/template` | GET | 下载 xlsx 导入模板（带表头 + 2 行示例） |
| `/api/v1/admin/reload` | POST | 热更新词库（受 ADMIN_TOKEN 保护，详见下） |
| `/api/v1/admin/fallbacks` | GET | 查看累计的 FALLBACK 词分组（详见 [未匹配词分析](#未匹配词分析)） |
| `/api/v1/admin/fallbacks/reset` | POST | 清空 FALLBACK 累计 |

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
- 列 1 = 极性提示（**强匹配条件**）：
  - `正面` → 只在正面词库桶内搜
  - `负面` → 只在负面词库桶内搜
  - 空 / `未知` / `中性` / 其他任意值 → 该行直接 `FALLBACK`（不乱猜、不跨桶匹配）
- 无表头（按数据读取）。可从 `GET /api/v1/normalize/excel/template` 下载带表头的模板
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
6. `归一-大类` —— 归一词所属的大类（命中时从词库回填，`FALLBACK` 时为空）
7. `建议归一词` —— 仅 `FALLBACK` 行有值：词库中与本词最相似的标准词（无视阈值），供运营期决定是否补为别名
8. `建议分数` —— 建议归一词与本词的余弦相似度
9. `建议-大类` —— 建议归一词所属的大类

响应头中携带 `X-Summary`（JSON 字符串），供前端展示层级分布：

```
X-Summary: {"total": 123, "L1": 80, "L2": 30, "L3": 5, "FALLBACK": 8}
```

## 未匹配词分析

服务在内存中累计所有 `FALLBACK` 词的出现频次。`GET /api/v1/admin/fallbacks` 返回按「建议归一词」分组、按总频次降序的 JSON，供运营期批量发现新词：

```
GET /api/v1/admin/fallbacks?model=bge&min_freq=2&limit=500
```

```json
{
  "total_unique": 87,
  "total_freq": 312,
  "filtered": 23,
  "by_suggestion": [
    {
      "suggested_vocab": "瑕疵",
      "suggestion_category": "质量",
      "suggestion_score": 0.72,
      "total_freq": 145,
      "fallbacks": [
        {"word": "破洞", "freq": 89, "score": 0.68},
        {"word": "漏洞", "freq": 41, "score": 0.62}
      ]
    }
  ]
}
```

运营期典型流程：按 `total_freq` 排序 → 找到频次高的组 → 直接把同组 FALLBACK 词加到 `data/aliases.json` 对应标准词下 → `/admin/reload` 热更新。

参数说明：
- `model`：用哪个 embedding 算建议，默认 `bge`
- `min_freq`：过滤掉累计频次低于此值的词（去噪）
- `limit`：最多返回多少个 FALLBACK 词（按频次截断）

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
  - `accept` = L2 接受阈值（BGE/M3E 0.7, fastText 0.6）
  - `fallback_to_edit` = L3 降级阈值（BGE/M3E 0.5, fastText 0.4）
- 编辑距离比率 `Normalizer.EDIT_DISTANCE_RATIO = 0.3`
- 运营期失败用例直接补到 `data/aliases.json` 即可

## 模型对比

默认使用 **M3E-base**（moka-ai/m3e-base, 768 维, ~400MB）。在我们的近义/口语化测试集（30 条）上：

| 模型 | L2 命中率 (cos ≥ 0.7) | FALLBACK 率 (cos < 0.5) | 平均余弦 |
|---|---:|---:|---:|
| BGE-small-zh (95MB) | 69.6% | 17.4% | 0.727 |
| BGE-base-zh (400MB) | 56.5% | 13.0% | 0.712 |
| **M3E-base (400MB)** | **95.7%** | **0.0%** | **0.861** |

切到 M3E 后，**L2 命中率 +26 个百分点、FALLBACK 率从 17% 降到 0%**。可用 `?model=bge` 切回 BGE 跑同一份输入做 A/B。

注：M3E 余弦分布更"挤"（噪声词也容易 0.6+），所以**强建议保留极性桶过滤**——单桶搜索能压住大部分误判。
