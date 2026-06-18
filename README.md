# 中文词归一化服务

把人工输入的中文标签词归一到标准词库。Web API 上传 txt，每行一词，返回归一结果。

## 快速开始

### 1. 创建 venv

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python -m venv .venv
source .venv/bin/activate
```

### 2. 装 PyTorch（按环境选一条，先装！）

PyTorch 必须**手动装**,直接 `pip install -r requirements.txt` 会拉 CPU 版,NVIDIA 卡装了也跑不了 GPU。
**先装 torch 再装其他依赖**——这样 sentence-transformers 会复用已装的 CUDA torch,不会回退到 CPU 版。

| 环境 | 命令 |
|---|---|
| **Windows / Linux + NVIDIA 卡,驱动 ≥ 531(CUDA 12.x)** | `pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128` |
| **NVIDIA 卡,驱动老(CUDA 11.8)** | `pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu118` |
| **没 NVIDIA 卡 / macOS / 纯 CPU** | `pip install torch==2.11.0`(拉 CPU 版) |

验证 GPU 通了再继续:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```
预期 `CUDA: True <你的显卡名>`,否则检查驱动版本(`nvidia-smi` 看 CUDA Version)。

### 3. 装其他依赖

```bash
pip install -r requirements.txt
```

### 4. 准备模型

M3E 模型约 400MB,首次启动会从 HuggingFace 拉到 `models/m3e_base/`。三种获取方式:

- **联网直拉**:`uvicorn app.main:app --port 8000`,启动时自动下载
- **国内镜像**:`export HF_ENDPOINT=https://hf-mirror.com` 再启动
- **完全离线**:在其他机器下好整个 `m3e-base` 目录,放到 `models/m3e_base/`,启动加 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`

```bash
# Windows
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
uvicorn app.main:app --port 8000

# Linux / macOS
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uvicorn app.main:app --port 8000
```

### 5. 调用

```bash
curl -F "file=@input.txt" "http://localhost:8000/api/v1/normalize?debug=1" -o output.txt
```

Web UI:浏览器打开 `http://localhost:8000/`,直接拖文件即可。

无 fastText 模型也可临时验证路由(使用桩 embedding):

```bash
python scripts/smoke_test_server.py
```

### 换环境时常见问题

| 现象 | 原因 / 解决 |
|---|---|
| 装好 GPU torch 但 `cuda.is_available()=False` | 驱动太老,`nvidia-smi` 看 CUDA Version,需要 ≥ wheel 要求的版本 |
| `pip install` 报 `torch==2.11.0+cu128` 找不到 | 网络问题。手动 `curl -L https://download.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp312-cp312-win_amd64.whl -o torch.whl` 下到本地(约 2.6GB),再 `pip install --no-deps torch.whl` |
| 装完 torch 是 `+cpu` 版 | 没加 `--index-url`,pip 默认走 pypi 拉 CPU |
| 切到 Linux 装不上 | wheel 平台后缀不同(win_amd64 / manylinux),需重新装对应平台的 torch |
| Python 版本不同(cp311 / cp313) | wheel 名字里的 `cp312` 要对应,装错版本会报 "no matching distribution" |

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
| `/api/v1/normalize/json` | POST | JSON 归一：传入词数组 + 标准词库数组（详见 [JSON 归一](#json-归一)） |
| `/api/v1/admin/reload` | POST | 热更新词库（受 ADMIN_TOKEN 保护，详见下） |
| `/api/v1/admin/fallbacks` | GET | 查看累计的 FALLBACK 词分组（详见 [未匹配词分析](#未匹配词分析)） |
| `/api/v1/admin/fallbacks/reset` | POST | 清空 FALLBACK 累计 |

### JSON 归一

`POST /api/v1/normalize/json` —— 不进文件，直接传 JSON。适合前端 / 上游服务直接调，不用落盘。

**请求体 (application/json)：**

```json
{
  "words": [
    {"word": "好用的", "polarity": "正面"},
    "掉色",
    {"word": "不确定", "polarity": ""}
  ],
  "vocab": [
    {"word": "质量好", "polarity": "正面", "category": "质量"},
    {"word": "质量差", "polarity": "负面", "category": "质量"},
    {"word": "掉色", "polarity": "负面", "category": "质量"}
  ],
  "model": "m3e"
}
```

- `words`：每个元素可以是**纯字符串**或 `{word, polarity?}`。上限 50,000
  - `polarity = "正面" / "负面"` → 只在对应单桶搜（强约束）
  - `polarity = ""` / 纯字符串 → 自动推断极性 + 双桶对比
  - `polarity = 其他值`（如 `"?"`, `"未知"`） → 强制 FALLBACK,不乱猜
- `vocab`：标准词库，每个词需 `word` + `polarity`（`正面` / `负面`），`category` 可选
- `model`：可选，默认 `m3e`

**响应：**

```json
{
  "results": [
    {"original": "好用的", "normalized": "质量好", "layer": "L2", "score": 0.8169, "category": "质量", "input_polarity": "正面"},
    {"original": "掉色", "normalized": "掉色", "layer": "L1", "score": 1.0, "category": "质量", "input_polarity": ""},
    {"original": "不确定", "normalized": "不确定", "layer": "FALLBACK", "score": 0.0, "category": "", "input_polarity": "", "suggestion": "质量差", "suggestion_score": 0.45, "suggestion_category": "质量"}
  ],
  "summary": {"total": 3, "L1": 1, "L2": 1, "L3": 0, "FALLBACK": 1},
  "model": "m3e",
  "elapsed_ms": 20.0
}
```

FALLBACK 行附带 `suggestion` / `suggestion_score` / `suggestion_category`，运营期可据此给跳过的词找最近的候选标准词。

## 管理端鉴权

`/api/v1/admin/reload` 默认仅允许 loopback（`127.0.0.1`、`::1`）访问；如需远程调用，
设置环境变量 `ADMIN_TOKEN=<secret>`，然后用 Bearer header 调用：

```bash
curl -X POST -H "Authorization: Bearer <secret>" http://host:8000/api/v1/admin/reload
```

未携带或 token 不匹配时返回 `401 {"error": "unauthorized"}`。

## 部署

### Docker（推荐）

```bash
# 1. 确保 models/m3e_base/ 存在（约 400MB）。如无，在其他能联网的机器下好拷过来:
#    python scripts/download_model.py
#    目录结构: models/m3e_base/models--moka-ai--m3e-base/snapshots/.../model.safetensors

# 2. GPU 服务器（需装 nvidia-container-toolkit）
docker compose up -d

#    国内网络慢时指定镜像:
#    docker build --build-arg TORCH_INDEX=https://mirrors.aliyun.com/pytorch-wheels/cu128 . && docker compose up -d

# 3. CPU 服务器
docker compose --profile cpu up -d

# 4. 验证
curl http://localhost:8000/api/v1/health
```

GPU 前置：
```bash
# Ubuntu/Debian 装 nvidia-container-toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 传统部署（systemd）

```bash
# 服务器上执行:
python3.12 -m venv .venv
source .venv/bin/activate

# 按服务器硬件选一条:
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128  # GPU
pip install torch==2.11.0                                                       # CPU

pip install -r requirements.txt

# 拷贝模型文件到 models/m3e_base/
# 拷贝 data/vocabulary.csv 和 data/aliases.json

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uvicorn app.main:app --host 0.0.0.0 --port 8000
```

systemd 服务文件 `/etc/systemd/system/lexagg.service`：
```ini
[Unit]
Description=LexAgg Word Normalizer
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/lexagg
Environment="HF_HUB_OFFLINE=1"
Environment="TRANSFORMERS_OFFLINE=1"
ExecStart=/opt/lexagg/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now lexagg
```

### 前置 nginx 反代（可选）

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # 限制上传大小
    client_max_body_size 100m;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

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
