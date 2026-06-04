# 中文词归一化服务 - 设计文档

- **日期**：2026-06-04
- **项目**：wordtest
- **作者**：cdong
- **状态**：待审阅

## 1. 背景与目标

用户持有一份中文标准词库（约 190 个词，按"正面/负面"两个极性标注），需要把人工输入的 2-3 万条标签词归一到该词库中对应的标准词。无法匹配时返回原词。

**核心要求**：
- 准确率 ≥ 95%
- 单次 2-3 万词处理时间 ≤ 1 分钟
- 通过 Web API 上传 txt 文件调用
- 词库可热更新

## 2. 用户场景

人工输入的标签词与标准词库之间的关系是模糊的：
- 口语化变体：「凉凉的」「挺凉快」「凉爽」 → 标准词「凉感适宜」
- 错别字/手抖：「毛绒绒」「蓬松」 → 标准词「蓬松」
- 反义防错归：「不舒适」「舒适」必须归到不同极性桶

词库结构（两列 CSV）：

| 词 | 极性 |
|---|---|
| 凉感适宜 | 正面 |
| 凉感太弱或太凉 | 负面 |
| 舒适 | 正面 |
| 不舒适 | 负面 |
| 轻薄（含轻盈、不压身） | 正面 |
| ... | ... |

**括号变体处理**：「轻盈」和「不压身」视作与「轻薄」同极性的标准词独立条目（同样标注为正面），但优先级上由 L1 别名表直接映射到「轻薄」。

## 3. 技术选型

| 组件 | 选型 | 理由 |
|---|---|---|
| Web 框架 | FastAPI | 异步、文件上传友好、自动 OpenAPI |
| 词向量模型 | fastText 中文（cc.zh.300.bin） | 短文本 OOV 友好、CPU 快、依赖轻 |
| 可选模型 | BGE-small-zh | 通过 `model=bge` 参数切换，懒加载 |
| 数据处理 | pandas | 读 CSV |
| 字符串匹配 | Python-Levenshtein | 纯 C 实现，比纯 Python 快 10x |
| 服务 | Uvicorn | FastAPI 标准 ASGI 服务器 |

## 4. 架构与数据流

### 4.1 模块划分

```
D:\workspace\claude\wordtest\
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 入口
│   ├── normalizer.py        # 归一化核心
│   ├── vocabulary.py        # 词库加载与管理
│   ├── similarity.py        # 余弦 + 编辑距离
│   └── embedding/
│       ├── __init__.py
│       ├── base.py          # EmbeddingModel 抽象基类
│       ├── fasttext_impl.py # FastTextEmbedding
│       ├── bge_impl.py      # BgeEmbedding
│       └── factory.py       # 按名字获取模型单例
├── data/
│   ├── vocabulary.csv       # 标准词库
│   └── aliases.json         # L1 别名表
├── models/
│   └── cc.zh.300.bin        # fastText 模型（首次启动时下载）
├── tests/
│   ├── test_normalizer.py
│   ├── test_vocabulary.py
│   ├── test_similarity.py
│   ├── test_api.py
│   └── test_accuracy.py
├── requirements.txt
└── README.md
```

### 4.2 启动期数据流

```
1. VocabularyLoader.load(csv_path)
   - 校验列数、极性取值、重复词
   - 构造 buckets (按极性分桶)
   - 构造 polarity_map (词→极性反查)
   - 构造 alias_map (变体→标准词，从括号展开)

2. FastTextEmbedding.load()
   - 若 models/cc.zh.300.bin 不存在，启动失败（不自动下载，引导用户手动运行 scripts/download_model.py）
   - 预计算所有标准词向量
   - 按极性分桶缓存

3. Normalizer(fastText) 构造完成 → 服务就绪
```

### 4.3 请求期数据流

```
POST /api/v1/normalize?model=fasttext&debug=0
  → 读上传文件 → 按行 splitlines
  → 对每行:
      polarity = _infer_polarity(word)
        - 查 aliases 找到 → 用其极性
        - 查 polarity_map 找到 → 用其极性
        - 都没有 → 极性未知，标记 None
      若 polarity 已知 → 在该桶内匹配
      若 polarity 未知 → 双桶对比取高
      匹配流程 L1→L2→L3 → 兜底返原词
  → 输出 tsv（两列或四列）
  → StreamingResponse 返回下载
```

## 5. 核心组件详细设计

### 5.1 VocabularyLoader

```python
@dataclass
class Vocabulary:
    buckets: dict[str, list[str]]      # {"正面": [...], "负面": [...]}
    polarity_map: dict[str, str]       # {"舒适": "正面", ...}
    alias_map: dict[str, str]          # {"轻盈": "轻薄", ...}

    def reload(self) -> None: ...
    def get_bucket(self, polarity: str) -> list[str]: ...
```

**加载校验**：
- 列数 ≠ 2 → 报错退出
- 极性取值不在 `{"正面", "负面"}` → 报错退出（指出行号）
- 同一词以不同极性重复出现 → 报错退出
- 括号内变体重复出现 → 报错退出
- 词为空字符串 → 报错退出

**括号展开规则**：
- 「轻薄（含轻盈、不压身）」 → 「轻盈」和「不压身」加入 buckets[正面]，并写入 alias_map：「轻盈」→「轻薄」、「不压身」→「轻薄」
- 「瑕疵（含破洞、勾丝、脏）」 → 同理加入负向桶
- 变体词已是独立标准词（如「透气」是独立词，不在括号里）→ 不冲突，单独存在

### 5.2 EmbeddingModel 抽象层

```python
class EmbeddingModel(ABC):
    @abstractmethod
    def load(self) -> None: ...
    @abstractmethod
    def encode(self, words: list[str]) -> np.ndarray: ...
    @property
    @abstractmethod
    def name(self) -> str: ...
    @property
    @abstractmethod
    def dim(self) -> int: ...
```

**FastTextEmbedding**：
- 包装 `fasttext.load_model(path)`
- `encode(words)` → `np.array([model.get_word_vector(w) for w in words])`
- OOV 自动走 subword，返回非零向量

**BgeEmbedding**：
- 用 `sentence-transformers` 的 `SentenceTransformer`
- `encode(words, normalize_embeddings=True)`

**Factory**：
```python
_models: dict[str, EmbeddingModel] = {}
_lock = threading.Lock()

def get_model(name: str) -> EmbeddingModel:
    if name not in _models:
        with _lock:
            if name not in _models:
                impl = {"fasttext": FastTextEmbedding, "bge": BgeEmbedding}[name]
                model = impl()
                model.load()
                _models[name] = model
    return _models[name]
```

### 5.3 Normalizer

```python
@dataclass
class NormalizeResult:
    original: str
    normalized: str
    matched_layer: Literal["L1", "L2", "L3", "FALLBACK"]
    score: float
    elapsed_ms: float

class Normalizer:
    # 注：以下阈值为初始值，实施阶段用样本调优后写入实际值
    THRESHOLDS = {
        "fasttext": {"accept": 0.6, "fallback_to_edit": 0.4},
        "bge":      {"accept": 0.7, "fallback_to_edit": 0.5},
    }
    EDIT_DISTANCE_RATIO = 0.3  # 编辑距离 / max(len(word), len(candidate))

    def normalize(self, word: str) -> NormalizeResult: ...
```

**匹配算法**（极性已知时单桶，极性未知时双桶对比取高）：

```python
def normalize(self, word: str) -> NormalizeResult:
    t0 = time.time()
    
    # 极性推断
    polarity = self._infer_polarity(word)
    
    if polarity:
        result = self._match_in_bucket(word, polarity)
    else:
        # 双桶对比取高
        r_pos = self._match_in_bucket(word, "正面")
        r_neg = self._match_in_bucket(word, "负面")
        result = r_pos if r_pos.score >= r_neg.score else r_neg
        # 极性仍未知 → 强制标 FALLBACK
        if result.matched_layer == "FALLBACK":
            result = NormalizeResult(word, word, "FALLBACK", 0.0, ...)
    
    return result

def _match_in_bucket(self, word: str, polarity: str) -> NormalizeResult:
    candidates = self.vocab.get_bucket(polarity)
    
    # L1 别名
    if word in self.vocab.alias_map:
        std = self.vocab.alias_map[word]
        if self.vocab.polarity_map.get(std) == polarity:
            return NormalizeResult(word, std, "L1", 1.0, ...)
    
    # L2 向量相似度
    word_vec = self.model.encode([word])[0]
    cand_vecs = self.precomputed_vectors[polarity]  # 预计算
    sims = cosine_batch(word_vec, cand_vecs)
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    best_candidate = candidates[best_idx]
    
    if best_sim >= self.THRESHOLDS[self.model.name]["accept"]:
        return NormalizeResult(word, best_candidate, "L2", best_sim, ...)
    
    # L3 编辑距离（兜底）
    if best_sim >= self.THRESHOLDS[self.model.name]["fallback_to_edit"]:
        edit_dists = [Levenshtein.distance(word, c) for c in candidates]
        best_idx = int(np.argmin(edit_dists))
        best_dist = edit_dists[best_idx]
        ratio = best_dist / max(len(word), len(candidates[best_idx]))
        if ratio <= self.EDIT_DISTANCE_RATIO:
            return NormalizeResult(word, candidates[best_idx], "L3", best_sim, ...)
    
    return NormalizeResult(word, word, "FALLBACK", best_sim, ...)
```

### 5.4 API 端点

```python
@app.post("/api/v1/normalize")
async def normalize(
    file: UploadFile,
    model: str = "fasttext",
    debug: int = 0,
):
    """上传 txt（每行一词），返回归一结果 tsv"""
    if file.size > 10 * 1024 * 1024:
        raise HTTPException(413, "file_too_large")
    
    content = await file.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(400, "invalid_encoding")
    
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        raise HTTPException(400, "empty_file")
    
    normalizer = Normalizer(get_model(model), vocab)
    results = [normalizer.normalize(line) for line in lines]
    
    sep = "\t"
    if debug:
        output = "\n".join(f"{r.original}{sep}{r.normalized}{sep}{r.matched_layer}{sep}{r.score:.3f}" for r in results)
    else:
        output = "\n".join(f"{r.original}{sep}{r.normalized}" for r in results)
    
    return StreamingResponse(
        io.StringIO(output),
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=normalized.txt"},
    )

@app.post("/api/v1/admin/reload")
async def reload_vocab():
    """热更新词库（CSV + aliases.json）"""
    try:
        vocab.reload()
        return {"status": "ok", "正面": len(vocab.buckets["正面"]), "负面": len(vocab.buckets["负面"])}
    except Exception as e:
        raise HTTPException(500, f"reload_failed: {e}")

@app.get("/api/v1/health")
async def health():
    return {
        "status": "ok",
        "default_model": "fasttext",
        "vocab_size": len(vocab.buckets["正面"]) + len(vocab.buckets["负面"]),
    }
```

## 6. 错误处理

| 错误 | HTTP | 触发 | 响应 |
|---|---|---|---|
| EmptyFileError | 400 | 上传空文件 | `{"error": "empty_file"}` |
| InvalidEncodingError | 400 | 非 UTF-8 | `{"error": "invalid_encoding"}` |
| UnknownModelError | 400 | `model=xxx` | `{"error": "unknown_model", "supported": ["fasttext", "bge"]}` |
| FileTooLargeError | 413 | > 10MB | `{"error": "file_too_large"}` |
| ReloadError | 500 | reload 失败 | 保留旧词库不变，返回 500 |
| ModelNotLoadedError | 500 | 模型文件缺失 | `{"error": "model_missing", "download_cmd": "python scripts/download_model.py"}` |

**启动期失败**：立即退出（CSV 格式错、模型文件缺），不要带病运行。

**单行异常**：跳过 + WARN 日志，不让一条坏数据卡住整批。

**日志策略**：
- INFO：服务启动、reload 成功、请求摘要（行数/耗时/命中率分布）
- WARN：单行异常、OOV 词
- ERROR：启动失败、reload 失败

## 7. 测试策略

### 7.1 单元测试
- VocabularyLoader：CSV 校验、括号展开、重复检测
- EmbeddingModel：已知词向量、OOV 走 subword、批量 shape
- similarity：cosine、levenshtein 边界值
- Normalizer：L1/L2/L3/FALLBACK 四分支各覆盖、双桶对比、反义防错归

### 7.2 集成测试（FastAPI TestClient）
- 上传 100 行小 txt → 响应格式
- `?debug=1` 输出四列
- `model=bge` 懒加载 + 缓存
- 错误路径：空文件、未知模型、超大文件
- reload 后词库变更生效

### 7.3 端到端验收（95% 硬指标）
- 准备 50-100 条**真实人工输入样本**（实施阶段向用户索取）
- 写 `tests/test_accuracy.py`，断言准确率 ≥ 95%
- 失败用例回流到 L1 别名表，重新跑测试至达标
- 运营期：每次新失败用例 → 补别名表 → 准确率单调上升

### 7.4 性能测试
- 3 万词 < 60 秒
- 命中率分布合理：L1 12% / L2 78% / L3 5% / FALLBACK 5%（预期值）

### 7.5 不测
- fastText 内部正确性（上游责任）
- FastAPI / Pydantic 框架本身

## 8. 依赖与配置

`requirements.txt`：
```
fastapi==0.115.*
uvicorn[standard]==0.32.*
fasttext-wheel==0.9.*
python-Levenshtein==0.26.*
numpy>=1.24
pandas>=2.0
pydantic>=2.0
pytest>=8.0
httpx>=0.27  # TestClient

# BGE 模型按需安装：
# pip install sentence-transformers>=3.0
```

`data/aliases.json` 示例：
```json
{
  "轻盈": "轻薄",
  "不压身": "轻薄",
  "沉重": "厚重",
  "破洞": "瑕疵",
  "勾丝": "瑕疵",
  "脏": "瑕疵"
}
```

**模型下载**：首次启动若 `models/cc.zh.300.bin` 不存在，调用 `python scripts/download_model.py` 从 Facebook 官方下载（~7GB 解压 ~4GB）或 huggingface 镜像下载小型版（~100MB）。

## 9. 实施阶段

待 spec 审阅通过后，将通过 writing-plans 技能创建实现计划。
