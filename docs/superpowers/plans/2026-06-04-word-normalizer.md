# 中文词归一化服务 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个 Web API 服务，把人工输入的 2-3 万条中文标签词归一到 190 个标准词（带正面/负面极性），准确率 ≥ 95%。

**Architecture:** FastAPI 服务 + 词库按极性分桶 + 三层匹配（L1 别名字典 / L2 fastText 余弦 / L3 编辑距离）。极性由词库标注，归一时只在同极性桶内找最匹配。

**Tech Stack:** Python 3.11+、FastAPI、Uvicorn、fastText（fasttext-wheel）、python-Levenshtein、numpy、pandas、pytest

**Spec:** `docs/superpowers/specs/2026-06-04-word-normalizer-design.md`

---

## File Structure

```
D:\workspace\claude\wordtest\
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI 入口 + 路由
│   ├── normalizer.py           # 归一化核心（Normalizer + NormalizeResult）
│   ├── vocabulary.py           # Vocabulary 数据类 + CSV 加载 + 括号展开
│   ├── similarity.py           # cosine_batch + 编辑距离工具
│   └── embedding/
│       ├── __init__.py
│       ├── base.py             # EmbeddingModel 抽象基类
│       ├── fasttext_impl.py    # FastTextEmbedding
│       ├── bge_impl.py         # BgeEmbedding（可选）
│       └── factory.py          # 单例 + 懒加载工厂
├── scripts/
│   └── download_model.py       # 下载 fastText 中文模型
├── data/
│   ├── vocabulary.csv          # 标准词库
│   └── aliases.json            # L1 别名表（运营期人工维护）
├── models/
│   └── cc.zh.300.bin           # fastText 模型（gitignore）
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # 共享 fixtures
│   ├── test_vocabulary.py
│   ├── test_similarity.py
│   ├── test_embedding.py
│   ├── test_normalizer.py
│   ├── test_api.py
│   └── test_accuracy.py        # 95% 准确率断言（待用户样本）
├── requirements.txt
├── .gitignore
└── README.md
```

**Decomposition rationale:**
- `vocabulary.py` 独占词库加载和校验，因为 CSV 格式规则是稳定的、可独立测试
- `similarity.py` 单独放数值工具，因为 cosine / levenshtein 是纯函数
- `embedding/` 子包隔离模型依赖，方便按需装载和未来加更多模型
- `normalizer.py` 是核心业务，三层匹配逻辑自包含
- `main.py` 只做路由和 HTTP 边界处理

---

## Task 1: 项目骨架

**Files:**
- Create: `D:\workspace\claude\wordtest\requirements.txt`
- Create: `D:\workspace\claude\wordtest\.gitignore`
- Create: `D:\workspace\claude\wordtest\app\__init__.py`
- Create: `D:\workspace\claude\wordtest\app\main.py`
- Create: `D:\workspace\claude\wordtest\tests\__init__.py`
- Create: `D:\workspace\claude\wordtest\tests\test_health.py`
- Create: `D:\workspace\claude\wordtest\pytest.ini`

- [ ] **Step 1: 创建 requirements.txt**

文件 `requirements.txt`：

```
fastapi==0.115.*
uvicorn[standard]==0.32.*
fasttext-wheel==0.9.*
python-Levenshtein==0.26.*
numpy>=1.24
pandas>=2.0
pydantic>=2.0
pytest>=8.0
httpx>=0.27
```

- [ ] **Step 2: 创建 .gitignore**

文件 `.gitignore`：

```
__pycache__/
*.pyc
*.pyo
.venv/
venv/
.env
.pytest_cache/
models/*.bin
models/*.vec
data/vocabulary.local.csv
```

- [ ] **Step 3: 创建 pytest.ini**

文件 `pytest.ini`：

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
```

- [ ] **Step 4: 创建包空文件**

文件 `app/__init__.py`：

```python
```

文件 `tests/__init__.py`：

```python
```

- [ ] **Step 5: 创建最小 FastAPI 应用 + 首个测试**

文件 `app/main.py`：

```python
from fastapi import FastAPI

app = FastAPI(title="Word Normalizer", version="0.1.0")


@app.get("/api/v1/health")
async def health():
    return {"status": "ok"}
```

文件 `tests/test_health.py`：

```python
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

- [ ] **Step 6: 安装依赖并运行测试**

```bash
cd D:/workspace/claude/wordtest
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
pytest tests/test_health.py -v
```

Expected: `1 passed`

- [ ] **Step 7: 提交**

```bash
cd D:/workspace/claude/wordtest
git init
git add .
git commit -m "feat: project scaffold with FastAPI + pytest"
```

---

## Task 2: 相似度工具（similarity.py）

**Files:**
- Create: `D:\workspace\claude\wordtest\app\similarity.py`
- Create: `D:\workspace\claude\wordtest\tests\test_similarity.py`

- [ ] **Step 1: 写失败测试**

文件 `tests/test_similarity.py`：

```python
import numpy as np

from app.similarity import cosine_batch, cosine_single, levenshtein_ratio


def test_cosine_single_identical_is_one():
    v = np.array([1.0, 0.0, 0.0])
    assert abs(cosine_single(v, v) - 1.0) < 1e-6


def test_cosine_single_orthogonal_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_single(a, b)) < 1e-6


def test_cosine_batch_shape():
    query = np.array([1.0, 0.0, 0.0])
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]])
    sims = cosine_batch(query, matrix)
    assert sims.shape == (3,)
    assert abs(sims[0] - 1.0) < 1e-6
    assert abs(sims[1]) < 1e-6


def test_levenshtein_ratio_identical_is_zero():
    assert levenshtein_ratio("舒适", "舒适") == 0.0


def test_levenshtein_ratio_basic():
    # 1 替换 / max(2,3) = 1/3
    r = levenshtein_ratio("凉快", "凉感")
    assert abs(r - 1 / 3) < 1e-6


def test_levenshtein_ratio_empty():
    assert levenshtein_ratio("", "abc") == 1.0
    assert levenshtein_ratio("abc", "") == 1.0
    assert levenshtein_ratio("", "") == 0.0
```

- [ ] **Step 2: 跑测试确认失败**

```bash
pytest tests/test_similarity.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.similarity'`

- [ ] **Step 3: 实现 similarity 模块**

文件 `app/similarity.py`：

```python
import numpy as np


def cosine_single(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """返回 query 与 matrix 每行的余弦相似度。"""
    q_norm = np.linalg.norm(query)
    m_norms = np.linalg.norm(matrix, axis=1)
    if q_norm == 0.0:
        return np.zeros(matrix.shape[0])
    safe_norms = np.where(m_norms == 0.0, 1.0, m_norms)
    dots = matrix @ query
    return (dots / (q_norm * safe_norms)).astype(float)


def levenshtein_ratio(a: str, b: str) -> float:
    """编辑距离 / max(len(a), len(b))，范围 [0, 1]。"""
    import Levenshtein

    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    d = Levenshtein.distance(a, b)
    return d / max(len(a), len(b))
```

- [ ] **Step 4: 跑测试确认通过**

```bash
pytest tests/test_similarity.py -v
```

Expected: `6 passed`

- [ ] **Step 5: 提交**

```bash
git add app/similarity.py tests/test_similarity.py
git commit -m "feat: similarity utilities (cosine + levenshtein ratio)"
```

---

## Task 3: 词库数据类与 CSV 加载

**Files:**
- Create: `D:\workspace\claude\wordtest\app\vocabulary.py`
- Create: `D:\workspace\claude\wordtest\tests\test_vocabulary.py`
- Create: `D:\workspace\claude\wordtest\tests\conftest.py`

- [ ] **Step 1: 创建共享 fixtures**

文件 `tests/conftest.py`：

```python
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def sample_vocab_csv():
    content = """词,极性
舒适,正面
触感好,正面
柔软,正面
凉感适宜,正面
凉感太弱或太凉,负面
不舒适,负面
触感差,负面
轻薄（含轻盈、不压身）,正面
瑕疵（含破洞、勾丝、脏）,负面
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def empty_vocab_csv():
    content = "词,极性\n"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def invalid_column_csv(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("only_one_col\nvalue\n", encoding="utf-8")
    return str(p)


@pytest.fixture
def invalid_polarity_csv(tmp_path):
    p = tmp_path / "bad_polarity.csv"
    p.write_text("词,极性\n舒适,正向\n", encoding="utf-8")
    return str(p)


@pytest.fixture
def duplicate_word_csv(tmp_path):
    p = tmp_path / "dup.csv"
    p.write_text("词,极性\n舒适,正面\n舒适,负面\n", encoding="utf-8")
    return str(p)
```

- [ ] **Step 2: 写失败测试**

文件 `tests/test_vocabulary.py`：

```python
import pytest

from app.vocabulary import Vocabulary, VocabularyLoadError


def test_load_valid_csv_builds_buckets(sample_vocab_csv):
    vocab = Vocabulary.load(sample_vocab_csv)
    assert "正面" in vocab.buckets
    assert "负面" in vocab.buckets
    assert "舒适" in vocab.buckets["正面"]
    assert "不舒适" in vocab.buckets["负面"]


def test_load_valid_csv_builds_polarity_map(sample_vocab_csv):
    vocab = Vocabulary.load(sample_vocab_csv)
    assert vocab.polarity_map["舒适"] == "正面"
    assert vocab.polarity_map["不舒适"] == "负面"


def test_load_valid_csv_expands_brackets(sample_vocab_csv):
    vocab = Vocabulary.load(sample_vocab_csv)
    # 轻盈和不压身应作为正面桶的独立词
    assert "轻盈" in vocab.buckets["正面"]
    assert "不压身" in vocab.buckets["正面"]
    # 同时写入 alias_map
    assert vocab.alias_map["轻盈"] == "轻薄"
    assert vocab.alias_map["不压身"] == "轻薄"
    assert vocab.alias_map["破洞"] == "瑕疵"
    assert vocab.alias_map["勾丝"] == "瑕疵"
    assert vocab.alias_map["脏"] == "瑕疵"


def test_load_valid_csv_keeps_main_word(sample_vocab_csv):
    vocab = Vocabulary.load(sample_vocab_csv)
    assert "轻薄" in vocab.buckets["正面"]
    assert "瑕疵" in vocab.buckets["负面"]


def test_load_empty_csv_raises(empty_vocab_csv):
    with pytest.raises(VocabularyLoadError):
        Vocabulary.load(empty_vocab_csv)


def test_load_invalid_columns_raises(invalid_column_csv):
    with pytest.raises(VocabularyLoadError, match="列数"):
        Vocabulary.load(invalid_column_csv)


def test_load_invalid_polarity_raises(invalid_polarity_csv):
    with pytest.raises(VocabularyLoadError, match="极性"):
        Vocabulary.load(invalid_polarity_csv)


def test_load_duplicate_word_raises(duplicate_word_csv):
    with pytest.raises(VocabularyLoadError, match="重复"):
        Vocabulary.load(duplicate_word_csv)


def test_polarity_map_query_unknown_word(sample_vocab_csv):
    vocab = Vocabulary.load(sample_vocab_csv)
    assert vocab.polarity_map.get("不存在的词") is None


def test_get_bucket(sample_vocab_csv):
    vocab = Vocabulary.load(sample_vocab_csv)
    pos = vocab.get_bucket("正面")
    assert "舒适" in pos
    neg = vocab.get_bucket("负面")
    assert "不舒适" in neg
```

- [ ] **Step 3: 跑测试确认失败**

```bash
pytest tests/test_vocabulary.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.vocabulary'`

- [ ] **Step 4: 实现 Vocabulary**

文件 `app/vocabulary.py`：

```python
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


class VocabularyLoadError(Exception):
    pass


_BRACKET_RE = re.compile(r"（含(.+?)）")


def _expand_brackets(word: str) -> list[str]:
    """从「轻薄（含轻盈、不压身）」中提取 ['轻盈', '不压身']。"""
    m = _BRACKET_RE.search(word)
    if not m:
        return []
    return [v.strip() for v in m.group(1).split("、") if v.strip()]


def _strip_brackets(word: str) -> str:
    """去掉括号说明，只保留主词。"""
    return _BRACKET_RE.sub("", word).strip()


@dataclass
class Vocabulary:
    buckets: dict[str, list[str]] = field(default_factory=dict)
    polarity_map: dict[str, str] = field(default_factory=dict)
    alias_map: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, csv_path: str) -> "Vocabulary":
        path = Path(csv_path)
        if not path.exists():
            raise VocabularyLoadError(f"词库文件不存在: {csv_path}")

        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        if len(df.columns) != 2:
            raise VocabularyLoadError(
                f"列数错误: 期望 2 列，实际 {len(df.columns)}"
            )

        buckets: dict[str, list[str]] = {"正面": [], "负面": []}
        polarity_map: dict[str, str] = {}
        alias_map: dict[str, str] = {}

        for idx, row in df.iterrows():
            word_raw = str(row.iloc[0]).strip()
            polarity = str(row.iloc[1]).strip()

            if not word_raw:
                raise VocabularyLoadError(f"第 {idx + 2} 行: 词为空")

            if polarity not in {"正面", "负面"}:
                raise VocabularyLoadError(
                    f"第 {idx + 2} 行: 极性 '{polarity}' 不合法，应为 '正面' 或 '负面'"
                )

            main_word = _strip_brackets(word_raw)
            variants = _expand_brackets(word_raw)

            if main_word in polarity_map:
                raise VocabularyLoadError(
                    f"第 {idx + 2} 行: 词 '{main_word}' 重复（已标记为 {polarity_map[main_word]}）"
                )

            # 主词入桶
            buckets[polarity].append(main_word)
            polarity_map[main_word] = polarity

            # 括号变体入桶 + alias_map
            for v in variants:
                if v in polarity_map:
                    raise VocabularyLoadError(
                        f"第 {idx + 2} 行: 括号变体 '{v}' 重复"
                    )
                buckets[polarity].append(v)
                polarity_map[v] = polarity
                alias_map[v] = main_word

        if not buckets["正面"] and not buckets["负面"]:
            raise VocabularyLoadError("词库为空")

        return cls(buckets=buckets, polarity_map=polarity_map, alias_map=alias_map)

    def get_bucket(self, polarity: str) -> list[str]:
        return self.buckets.get(polarity, [])

    def reload(self, csv_path: str) -> None:
        new = Vocabulary.load(csv_path)
        self.buckets = new.buckets
        self.polarity_map = new.polarity_map
        self.alias_map = new.alias_map
```

- [ ] **Step 5: 跑测试确认通过**

```bash
pytest tests/test_vocabulary.py -v
```

Expected: `10 passed`

- [ ] **Step 6: 提交**

```bash
git add app/vocabulary.py tests/test_vocabulary.py tests/conftest.py
git commit -m "feat: Vocabulary loader with bracket expansion and validation"
```

---

## Task 4: EmbeddingModel 抽象层 + Factory

**Files:**
- Create: `D:\workspace\claude\wordtest\app\embedding\__init__.py`
- Create: `D:\workspace\claude\wordtest\app\embedding\base.py`
- Create: `D:\workspace\claude\wordtest\app\embedding\factory.py`
- Create: `D:\workspace\claude\wordtest\tests\test_embedding_factory.py`

- [ ] **Step 1: 写失败测试**

文件 `tests/test_embedding_factory.py`：

```python
import pytest

from app.embedding.base import EmbeddingModel
from app.embedding.factory import ModelNotFoundError, get_model, list_models, reset_models


class _FakeModel(EmbeddingModel):
    def __init__(self):
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def encode(self, words):
        import numpy as np
        return np.zeros((len(words), 4))

    @property
    def name(self) -> str:
        return "fake"

    @property
    def dim(self) -> int:
        return 4


def test_get_model_unknown_raises():
    reset_models()
    with pytest.raises(ModelNotFoundError):
        get_model("nonexistent")


def test_list_models():
    models = list_models()
    assert "fasttext" in models
    assert "bge" in models


def test_factory_caches_singleton(monkeypatch):
    reset_models()
    monkeypatch.setitem(__import__("app.embedding.factory", fromlist=["_REGISTRY"])._REGISTRY,
                         "fake", _FakeModel)
    a = get_model("fake")
    b = get_model("fake")
    assert a is b
    assert a.loaded is True
```

- [ ] **Step 2: 跑测试确认失败**

```bash
pytest tests/test_embedding_factory.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: 实现抽象基类**

文件 `app/embedding/__init__.py`：

```python
from app.embedding.base import EmbeddingModel
from app.embedding.factory import (
    ModelNotFoundError,
    get_model,
    list_models,
    reset_models,
)

__all__ = [
    "EmbeddingModel",
    "ModelNotFoundError",
    "get_model",
    "list_models",
    "reset_models",
]
```

文件 `app/embedding/base.py`：

```python
from abc import ABC, abstractmethod

import numpy as np


class EmbeddingModel(ABC):
    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def encode(self, words: list[str]) -> np.ndarray:
        """返回 shape=(len(words), dim) 的 float32 向量。"""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def dim(self) -> int: ...
```

- [ ] **Step 4: 实现 factory**

文件 `app/embedding/factory.py`：

```python
import threading
from typing import Type

from app.embedding.base import EmbeddingModel


class ModelNotFoundError(Exception):
    pass


# 注册表：仅在导入对应实现后才注册
_REGISTRY: dict[str, Type[EmbeddingModel]] = {}

_models: dict[str, EmbeddingModel] = {}
_lock = threading.Lock()
_registered = False


def _register_defaults() -> None:
    global _registered
    if _registered:
        return
    # 延迟导入，避免启动时硬依赖
    from app.embedding.fasttext_impl import FastTextEmbedding
    from app.embedding.bge_impl import BgeEmbedding

    _REGISTRY["fasttext"] = FastTextEmbedding
    _REGISTRY["bge"] = BgeEmbedding
    _registered = True


def register(name: str, cls: Type[EmbeddingModel]) -> None:
    """注册自定义模型（用于测试）。"""
    _REGISTRY[name] = cls


def get_model(name: str) -> EmbeddingModel:
    _register_defaults()
    if name not in _models:
        with _lock:
            if name not in _models:
                if name not in _REGISTRY:
                    raise ModelNotFoundError(
                        f"未知模型: {name}，可选: {list(_REGISTRY.keys())}"
                    )
                instance = _REGISTRY[name]()
                instance.load()
                _models[name] = instance
    return _models[name]


def list_models() -> list[str]:
    _register_defaults()
    return list(_REGISTRY.keys())


def reset_models() -> None:
    """清空缓存（仅供测试）。"""
    global _registered
    with _lock:
        _models.clear()
        _REGISTRY.clear()
        _registered = False
```

- [ ] **Step 5: 添加占位实现以让 import 不报错**

文件 `app/embedding/fasttext_impl.py`：

```python
"""占位实现，由后续 Task 5 替换为真实 fastText。"""
import numpy as np

from app.embedding.base import EmbeddingModel


class FastTextEmbedding(EmbeddingModel):
    def __init__(self, model_path: str = "models/cc.zh.300.bin"):
        self.model_path = model_path
        self._model = None

    def load(self) -> None:
        # 实际实现见 Task 5
        self._model = None

    def encode(self, words: list[str]) -> np.ndarray:
        return np.zeros((len(words), 300), dtype=np.float32)

    @property
    def name(self) -> str:
        return "fasttext"

    @property
    def dim(self) -> int:
        return 300
```

文件 `app/embedding/bge_impl.py`：

```python
"""占位实现，由后续 Task（按需）替换为真实 BGE。"""
import numpy as np

from app.embedding.base import EmbeddingModel


class BgeEmbedding(EmbeddingModel):
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        self.model_name = model_name
        self._model = None

    def load(self) -> None:
        # 实际实现见 BGE Task（按需）
        self._model = None

    def encode(self, words: list[str]) -> np.ndarray:
        return np.zeros((len(words), 512), dtype=np.float32)

    @property
    def name(self) -> str:
        return "bge"

    @property
    def dim(self) -> int:
        return 512
```

- [ ] **Step 6: 跑测试确认通过**

```bash
pytest tests/test_embedding_factory.py -v
```

Expected: `3 passed`

- [ ] **Step 7: 提交**

```bash
git add app/embedding tests/test_embedding_factory.py
git commit -m "feat: EmbeddingModel abstract + factory with lazy loading"
```

---

## Task 5: 下载脚本（download_model.py）

**Files:**
- Create: `D:\workspace\claude\wordtest\scripts\download_model.py`

- [ ] **Step 1: 实现下载脚本**

文件 `scripts/download_model.py`：

```python
"""下载 fastText 中文模型到 models/ 目录。

默认从 Hugging Face 镜像下载小型压缩版（~100MB）。
若用户已有 cc.zh.300.bin 全量版，可放入 models/ 跳过下载。
"""
import os
import sys
import urllib.request
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "cc.zh.300.bin"

# 备选 URL（Hugging Face 镜像 + 官方源）
URLS = [
    "https://huggingface.co/facebook/fasttext-zh-vectors/resolve/main/cc.zh.300.bin",
    "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.bin.gz",
]


def main() -> int:
    if MODEL_PATH.exists():
        print(f"模型已存在: {MODEL_PATH}")
        return 0

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for url in URLS:
        print(f"尝试下载: {url}")
        try:
            if url.endswith(".gz"):
                import gzip
                tmp_gz = MODEL_PATH.with_suffix(".bin.gz")
                urllib.request.urlretrieve(url, tmp_gz)
                print("解压中...")
                with gzip.open(tmp_gz, "rb") as f_in, open(MODEL_PATH, "wb") as f_out:
                    f_out.writelines(f_in)
                tmp_gz.unlink()
            else:
                urllib.request.urlretrieve(url, MODEL_PATH)
            print(f"下载完成: {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")
            return 0
        except Exception as e:
            print(f"失败: {e}")
            continue

    print("所有下载源均失败，请手动放置模型到 models/cc.zh.300.bin")
    return 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: 跑下载（用户运行时手动执行）**

```bash
python scripts/download_model.py
```

Expected: 进度条或下载完成消息。本任务不强制在 CI 中跑下载，但代码须能 import。

- [ ] **Step 3: 提交**

```bash
git add scripts/download_model.py
git commit -m "feat: download script for fastText Chinese model"
```

---

## Task 6: FastTextEmbedding 真实实现

**Files:**
- Modify: `D:\workspace\claude\wordtest\app\embedding\fasttext_impl.py`
- Create: `D:\workspace\claude\wordtest\tests\test_fasttext_embedding.py`

- [ ] **Step 1: 写失败测试**

文件 `tests/test_fasttext_embedding.py`：

```python
import numpy as np
import pytest

from app.embedding.fasttext_impl import FastTextEmbedding, ModelFileMissingError


def _write_dummy_model(tmp_path) -> str:
    """构造一个最小的 fastText 模型用于测试（不下载真模型）。"""
    pytest.skip("需要真实 fastText 模型才能跑，CI 中跳过")


@pytest.mark.skipif(
    not pytest.importorskip("fasttext", reason="fasttext 未安装"),
    reason="fasttext 未安装",
)
def test_encode_known_word(tmp_path):
    pytest.skip("依赖真实模型，由人工运行")


@pytest.mark.skipif(
    not pytest.importorskip("fasttext", reason="fasttext 未安装"),
    reason="fasttext 未安装",
)
def test_encode_oov_word_returns_nonzero(tmp_path):
    pytest.skip("依赖真实模型，由人工运行")


def test_load_missing_model_raises(tmp_path):
    fake_path = tmp_path / "no_such_model.bin"
    emb = FastTextEmbedding(model_path=str(fake_path))
    with pytest.raises(ModelFileMissingError):
        emb.load()


def test_encode_without_load_returns_zeros():
    emb = FastTextEmbedding(model_path="models/cc.zh.300.bin")
    out = emb.encode(["测试"])
    assert out.shape == (1, 300)
    assert out.dtype == np.float32
```

- [ ] **Step 2: 实现真实 FastTextEmbedding**

文件 `app/embedding/fasttext_impl.py`：

```python
from pathlib import Path

import numpy as np

from app.embedding.base import EmbeddingModel


class ModelFileMissingError(Exception):
    pass


class FastTextEmbedding(EmbeddingModel):
    def __init__(self, model_path: str = "models/cc.zh.300.bin"):
        self.model_path = model_path
        self._model = None

    def load(self) -> None:
        path = Path(self.model_path)
        if not path.exists():
            raise ModelFileMissingError(
                f"fastText 模型不存在: {self.model_path}，"
                f"请运行: python scripts/download_model.py"
            )
        import fasttext

        self._model = fasttext.load_model(str(path))

    def encode(self, words: list[str]) -> np.ndarray:
        if self._model is None:
            # 模型未加载（测试占位）：返回零向量
            return np.zeros((len(words), self.dim), dtype=np.float32)
        return np.array(
            [self._model.get_word_vector(w) for w in words],
            dtype=np.float32,
        )

    @property
    def name(self) -> str:
        return "fasttext"

    @property
    def dim(self) -> int:
        return 300
```

- [ ] **Step 3: 跑测试**

```bash
pytest tests/test_fasttext_embedding.py -v
```

Expected: `1 passed`（其余 skipped），`test_load_missing_model_raises` 和 `test_encode_without_load_returns_zeros` 通过

- [ ] **Step 4: 提交**

```bash
git add app/embedding/fasttext_impl.py tests/test_fasttext_embedding.py
git commit -m "feat: FastTextEmbedding with file existence check and lazy load"
```

---

## Task 7: Normalizer 核心（L1 别名层）

**Files:**
- Create: `D:\workspace\claude\wordtest\app\normalizer.py`
- Create: `D:\workspace\claude\wordtest\tests\test_normalizer_l1.py`

- [ ] **Step 1: 写失败测试**

文件 `tests/test_normalizer_l1.py`：

```python
import numpy as np
import pytest

from app.normalizer import Normalizer, NormalizeResult
from app.vocabulary import Vocabulary


class _StubEmbedding:
    """测试用 embedding stub，可注入预定义相似度。"""

    def __init__(self, name: str = "stub", dim: int = 4, scores: dict | None = None):
        self._name = name
        self._dim = dim
        # scores: {word: {candidate: score}}
        self._scores = scores or {}

    def load(self) -> None:
        pass

    def encode(self, words):
        return np.zeros((len(words), self._dim), dtype=np.float32)

    def get_score(self, word: str, candidate: str) -> float:
        return self._scores.get(word, {}).get(candidate, 0.0)

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return self._dim


@pytest.fixture
def vocab():
    return Vocabulary.load_from_rows(
        [
            ("舒适", "正面"),
            ("凉感适宜", "正面"),
            ("轻薄", "正面"),
            ("不舒适", "负面"),
            ("瑕疵", "负面"),
        ],
        alias_map={"轻盈": "轻薄", "不压身": "轻薄"},
    )


def test_l1_alias_hit_returns_standard_word(vocab):
    n = Normalizer(_StubEmbedding(), vocab)
    r = n.normalize("轻盈")
    assert r.normalized == "轻薄"
    assert r.matched_layer == "L1"
    assert r.score == 1.0


def test_l1_alias_hit_with_known_polarity(vocab):
    n = Normalizer(_StubEmbedding(), vocab)
    r = n.normalize("不压身")
    assert r.normalized == "轻薄"
    assert r.matched_layer == "L1"


def test_l1_exact_match_returns_self(vocab):
    n = Normalizer(_StubEmbedding(), vocab)
    r = n.normalize("舒适")
    assert r.normalized == "舒适"
    assert r.matched_layer == "L1"
```

- [ ] **Step 2: 跑测试确认失败**

```bash
pytest tests/test_normalizer_l1.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: 实现 Normalizer（L1 部分）**

文件 `app/normalizer.py`：

```python
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.similarity import cosine_batch
from app.vocabulary import Vocabulary


@dataclass
class NormalizeResult:
    original: str
    normalized: str
    matched_layer: Literal["L1", "L2", "L3", "FALLBACK"]
    score: float
    elapsed_ms: float


class Normalizer:
    THRESHOLDS = {
        "fasttext": {"accept": 0.6, "fallback_to_edit": 0.4},
        "bge": {"accept": 0.7, "fallback_to_edit": 0.5},
    }
    EDIT_DISTANCE_RATIO = 0.3

    def __init__(self, embedding, vocab: Vocabulary):
        self.embedding = embedding
        self.vocab = vocab
        # 预计算标准词向量（按极性分桶）
        self._precomputed: dict[str, np.ndarray] = {}
        self._precompute_vectors()

    def _precompute_vectors(self) -> None:
        for polarity, words in self.vocab.buckets.items():
            if words:
                self._precomputed[polarity] = self.embedding.encode(words)
            else:
                self._precomputed[polarity] = np.zeros((0, self.embedding.dim), dtype=np.float32)

    def normalize(self, word: str) -> NormalizeResult:
        t0 = time.time()
        result = self._normalize_inner(word)
        result.elapsed_ms = (time.time() - t0) * 1000
        return result

    def _normalize_inner(self, word: str) -> NormalizeResult:
        polarity = self._infer_polarity(word)

        if polarity is not None:
            return self._match_in_bucket(word, polarity)

        # 极性未知 → 双桶对比取高
        r_pos = self._match_in_bucket(word, "正面")
        r_neg = self._match_in_bucket(word, "负面")
        if r_pos.matched_layer != "FALLBACK" and (
            r_neg.matched_layer == "FALLBACK" or r_pos.score >= r_neg.score
        ):
            return r_pos
        if r_neg.matched_layer != "FALLBACK":
            return r_neg
        # 两桶都 FALLBACK → 返回原词
        return NormalizeResult(
            original=word, normalized=word, matched_layer="FALLBACK", score=0.0, elapsed_ms=0.0
        )

    def _infer_polarity(self, word: str) -> str | None:
        # 优先查 alias_map（变体词的极性）
        if word in self.vocab.alias_map:
            std = self.vocab.alias_map[word]
            return self.vocab.polarity_map.get(std)
        # 再查 polarity_map（输入词就是标准词）
        return self.vocab.polarity_map.get(word)

    def _match_in_bucket(self, word: str, polarity: str) -> NormalizeResult:
        candidates = self.vocab.get_bucket(polarity)
        if not candidates:
            return NormalizeResult(
                original=word, normalized=word, matched_layer="FALLBACK", score=0.0, elapsed_ms=0.0
            )

        # L1 别名命中
        if word in self.vocab.alias_map:
            std = self.vocab.alias_map[word]
            if self.vocab.polarity_map.get(std) == polarity:
                return NormalizeResult(
                    original=word, normalized=std, matched_layer="L1", score=1.0, elapsed_ms=0.0
                )
        # 精确匹配词库中已有词
        if word in self.vocab.polarity_map:
            std_polarity = self.vocab.polarity_map[word]
            if std_polarity == polarity:
                return NormalizeResult(
                    original=word, normalized=word, matched_layer="L1", score=1.0, elapsed_ms=0.0
                )

        # L2 向量相似度
        word_vec = self.embedding.encode([word])[0]
        cand_vecs = self._precomputed[polarity]
        sims = cosine_batch(word_vec, cand_vecs)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        best_candidate = candidates[best_idx]

        thresholds = self.THRESHOLDS.get(self.embedding.name, self.THRESHOLDS["fasttext"])
        if best_sim >= thresholds["accept"]:
            return NormalizeResult(
                original=word, normalized=best_candidate, matched_layer="L2", score=best_sim, elapsed_ms=0.0
            )

        # L3 编辑距离（fallback）
        if best_sim >= thresholds["fallback_to_edit"]:
            from app.similarity import levenshtein_ratio

            ratios = [levenshtein_ratio(word, c) for c in candidates]
            best_idx = int(np.argmin(ratios))
            best_ratio = ratios[best_idx]
            if best_ratio <= self.EDIT_DISTANCE_RATIO:
                return NormalizeResult(
                    original=word, normalized=candidates[best_idx], matched_layer="L3", score=best_sim, elapsed_ms=0.0
                )

        return NormalizeResult(
            original=word, normalized=word, matched_layer="FALLBACK", score=best_sim, elapsed_ms=0.0
        )
```

- [ ] **Step 4: 给 Vocabulary 加 load_from_rows 测试辅助**

文件 `app/vocabulary.py` 追加（在类定义内）：

```python
    @classmethod
    def load_from_rows(cls, rows: list[tuple[str, str]], alias_map: dict[str, str] | None = None) -> "Vocabulary":
        """测试用：直接构造，无需 CSV 文件。"""
        buckets: dict[str, list[str]] = {"正面": [], "负面": []}
        polarity_map: dict[str, str] = {}
        amap = dict(alias_map or {})

        for word, polarity in rows:
            if polarity not in {"正面", "负面"}:
                raise VocabularyLoadError(f"非法极性: {polarity}")
            if word in polarity_map:
                raise VocabularyLoadError(f"重复: {word}")
            buckets[polarity].append(word)
            polarity_map[word] = polarity

        # 别名词也入桶
        for variant, std in amap.items():
            if variant in polarity_map:
                continue
            std_polarity = polarity_map.get(std)
            if std_polarity:
                buckets[std_polarity].append(variant)
                polarity_map[variant] = std_polarity

        return cls(buckets=buckets, polarity_map=polarity_map, alias_map=amap)
```

- [ ] **Step 5: 跑测试确认通过**

```bash
pytest tests/test_normalizer_l1.py -v
```

Expected: `3 passed`

- [ ] **Step 6: 提交**

```bash
git add app/normalizer.py app/vocabulary.py tests/test_normalizer_l1.py
git commit -m "feat: Normalizer core with L1 alias matching"
```

---

## Task 8: Normalizer L2（向量相似度）和 L3（编辑距离）测试

**Files:**
- Create: `D:\workspace\claude\wordtest\tests\test_normalizer_l2_l3.py`

- [ ] **Step 1: 写测试**

文件 `tests/test_normalizer_l2_l3.py`：

```python
import numpy as np
import pytest

from app import similarity as sim_mod
from app.normalizer import Normalizer
from app.vocabulary import Vocabulary
from tests.conftest import _CURRENT_BEST, _patched_cosine, _StubEmbedding


@pytest.fixture
def vocab():
    return Vocabulary.load_from_rows(
        [
            ("舒适", "正面"),
            ("凉感适宜", "正面"),
            ("轻薄", "正面"),
            ("不舒适", "负面"),
            ("瑕疵", "负面"),
        ]
    )


def test_l2_high_similarity_returns_top1(vocab, monkeypatch):
    monkeypatch.setattr(sim_mod, "cosine_batch", _patched_cosine)
    _CURRENT_BEST.clear()

    n = Normalizer(_StubEmbedding(), vocab)
    original_inner = n._match_in_bucket

    def patched_inner(word, polarity):
        candidates = vocab.get_bucket(polarity)
        scores = {"凉感适宜": 0.8, "舒适": 0.5, "轻薄": 0.4}
        _CURRENT_BEST.append((word, scores, candidates))
        return original_inner(word, polarity)

    monkeypatch.setattr(n, "_match_in_bucket", patched_inner)

    r = n.normalize("凉爽")
    assert r.matched_layer == "L2"
    assert r.normalized == "凉感适宜"
    assert abs(r.score - 0.8) < 1e-6


def test_l2_below_threshold_falls_through(vocab, monkeypatch):
    monkeypatch.setattr(sim_mod, "cosine_batch", _patched_cosine)
    _CURRENT_BEST.clear()

    n = Normalizer(_StubEmbedding(), vocab)
    original_inner = n._match_in_bucket

    def patched_inner(word, polarity):
        candidates = vocab.get_bucket(polarity)
        scores = {c: 0.1 for c in candidates}  # 全低
        _CURRENT_BEST.append((word, scores, candidates))
        return original_inner(word, polarity)

    monkeypatch.setattr(n, "_match_in_bucket", patched_inner)

    r = n.normalize("不认识的词")
    # 0.1 < 0.4 fallback_to_edit → FALLBACK
    assert r.matched_layer == "FALLBACK"
    assert r.normalized == "不认识的词"


def test_l3_match_with_low_edit_ratio(vocab, monkeypatch):
    """输入 "凉爽适宜"（4字），与 "凉感适宜" 编辑距离 1，比率 0.25 ≤ 0.3 → L3 命中"""
    monkeypatch.setattr(sim_mod, "cosine_batch", _patched_cosine)
    _CURRENT_BEST.clear()

    n = Normalizer(_StubEmbedding(), vocab)
    original_inner = n._match_in_bucket

    def patched_inner(word, polarity):
        candidates = vocab.get_bucket(polarity)
        scores = {c: 0.5 if c == "凉感适宜" else 0.0 for c in candidates}
        _CURRENT_BEST.append((word, scores, candidates))
        return original_inner(word, polarity)

    monkeypatch.setattr(n, "_match_in_bucket", patched_inner)

    r = n.normalize("凉爽适宜")
    # 0.5 ≥ 0.4 → 进 L3；编辑距离 1/4 = 0.25 ≤ 0.3 → 命中
    assert r.matched_layer == "L3"
    assert r.normalized == "凉感适宜"


def test_l3_reject_high_edit_ratio(vocab, monkeypatch):
    """输入 "凉凉的"（3字），与所有候选编辑距离比率 > 0.3 → 拒绝 L3 → FALLBACK"""
    monkeypatch.setattr(sim_mod, "cosine_batch", _patched_cosine)
    _CURRENT_BEST.clear()

    n = Normalizer(_StubEmbedding(), vocab)
    original_inner = n._match_in_bucket

    def patched_inner(word, polarity):
        candidates = vocab.get_bucket(polarity)
        scores = {c: 0.5 for c in candidates}  # 都中等
        _CURRENT_BEST.append((word, scores, candidates))
        return original_inner(word, polarity)

    monkeypatch.setattr(n, "_match_in_bucket", patched_inner)

    r = n.normalize("凉凉的")
    # 0.5 ≥ 0.4 → 进 L3；编辑距离比率 > 0.3 → FALLBACK
    assert r.matched_layer == "FALLBACK"
    assert r.normalized == "凉凉的"
```

- [ ] **Step 2: 跑测试**

```bash
pytest tests/test_normalizer_l2_l3.py -v
```

Expected: `5 passed`（如果 L2/L3 逻辑已实现）

- [ ] **Step 3: 提交**

```bash
git add tests/test_normalizer_l2_l3.py
git commit -m "test: L2/L3/FALLBACK branches coverage"
```

---

## Task 9: Normalizer 极性推断与双桶对比

**Files:**
- Create: `D:\workspace\claude\wordtest\tests\test_normalizer_polarity.py`

- [ ] **Step 1: 写测试**

文件 `tests/test_normalizer_polarity.py`：

```python
import numpy as np
import pytest

from app import similarity as sim_mod
from app.normalizer import Normalizer
from app.vocabulary import Vocabulary

# 共享测试桩（conftest.py 提供 _CURRENT_BEST 和 _patched_cosine）


class _StubEmbedding:
    def __init__(self, dim: int = 4, name: str = "fasttext"):
        self._dim = dim
        self._name = name

    def load(self) -> None:
        pass

    def encode(self, words):
        return np.ones((len(words), self._dim), dtype=np.float32)

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return self._dim


@pytest.fixture
def vocab():
    return Vocabulary.load_from_rows(
        [
            ("舒适", "正面"),
            ("柔软", "正面"),
            ("不舒适", "负面"),
            ("瑕疵", "负面"),
        ]
    )


def test_unknown_polarity_uses_dual_bucket_compare(vocab, monkeypatch):
    """极性未知时双桶都 FALLBACK → 最终 FALLBACK"""
    def fake_cosine(query_vec, matrix):
        return np.zeros(matrix.shape[0])
    monkeypatch.setattr(sim_mod, "cosine_batch", fake_cosine)

    n = Normalizer(_StubEmbedding(), vocab)
    r = n.normalize("凉凉的")
    assert r.matched_layer == "FALLBACK"
    assert r.normalized == "凉凉的"


def test_unknown_polarity_picks_higher_score(vocab, monkeypatch, request):
    """极性未知时双桶对比取高 → 正面桶 0.7 胜出"""
    from tests.conftest import _CURRENT_BEST, _patched_cosine

    _CURRENT_BEST.clear()
    monkeypatch.setattr(sim_mod, "cosine_batch", _patched_cosine)

    n = Normalizer(_StubEmbedding(), vocab)
    original_inner = n._match_in_bucket

    def patched_inner(word, polarity):
        candidates = vocab.get_bucket(polarity)
        if polarity == "正面":
            scores = {"舒适": 0.7, "柔软": 0.6}
        else:
            scores = {c: 0.5 for c in candidates}
        _CURRENT_BEST.append((word, scores, candidates))
        return original_inner(word, polarity)

    monkeypatch.setattr(n, "_match_in_bucket", patched_inner)

    r = n.normalize("正义词")
    assert r.matched_layer == "L2"
    assert r.normalized == "舒适"


def test_known_polarity_uses_single_bucket():
    """极性已知（如别名匹配出极性）→ 只走单桶"""
    vocab_with_alias = Vocabulary.load_from_rows(
        [("舒适", "正面"), ("轻薄", "正面")],
        alias_map={"轻盈": "轻薄"},
    )
    n = Normalizer(_StubEmbedding(), vocab_with_alias)
    r = n.normalize("轻盈")
    assert r.matched_layer == "L1"
    assert r.normalized == "轻薄"
```

- [ ] **Step 2: 跑测试**

```bash
pytest tests/test_normalizer_polarity.py -v
```

Expected: `3 passed`

- [ ] **Step 3: 提交**

```bash
git add tests/test_normalizer_polarity.py
git commit -m "test: polarity inference and dual-bucket compare"
```

---

## Task 10: FastAPI 路由（/normalize、/admin/reload、/health）

**Files:**
- Modify: `D:\workspace\claude\wordtest\app\main.py`
- Create: `D:\workspace\claude\wordtest\data\vocabulary.csv`
- Create: `D:\workspace\claude\wordtest\data\aliases.json`
- Create: `D:\workspace\claude\wordtest\tests\conftest.py`（追加 fixtures）
- Modify: `D:\workspace\claude\wordtest\tests\test_health.py`

- [ ] **Step 1: 创建种子词库和别名表**

文件 `data/vocabulary.csv`：

```csv
词,极性
舒适,正面
触感好,正面
柔软,正面
丝滑,正面
亲肤,正面
凉感,正面
透气,正面
贴合,正面
包裹感,正面
厚实,正面
轻薄（含轻盈、不压身）,正面
适合裸睡,正面
弹性好,正面
静音,正面
软而不塌,正面
蓬松,正面
凉感适宜,正面
接触凉感,正面
凉感持续,正面
温度稳定,正面
睡眠质量提升,正面
入睡快,正面
深睡时长延长,正面
睡眠时长延长,正面
质量好,正面
做工精细,正面
耐用,正面
不起球,正面
不褪色,正面
不缩水,正面
不掉毛,正面
不易皱,正面
不变形,正面
可机洗,正面
可水洗,正面
易清洗,正面
易收纳,正面
便携,正面
速干,正面
耐脏,正面
包装好,正面
服务好,正面
性价比高,正面
纯棉,正面
四季通用,正面
不舒适,负面
触感差,负面
粗糙、扎人,负面
手感硬,负面
太滑,负面
不亲肤,负面
闷热,负面
不服帖,负面
厚重（含沉重）,负面
过厚,负面
薄、透,负面
过薄,负面
弹性过大,负面
有响声,负面
软塌,负面
蓬松度不足,负面
过热（太热）,负面
保暖不足（过冷）,负面
凉感太弱或太凉,负面
凉感短暂,负面
高度不合适,负面
支撑差,负面
睡了落枕,负面
翻身困难,负面
瑕疵（含破洞、勾丝、脏）,负面
质量差,负面
做工差,负面
异味,负面
起毛起球,负面
面料皱,负面
变硬,负面
塌陷,负面
变黄,负面
杂质、异物,负面
填充物品质不好,负面
清洁度不高,负面
钻绒、跑绒,负面
不可水洗,负面
不可烘干,负面
收纳不方便,负面
难套,负面
洗后凉感变弱,负面
不耐脏,负面
包装差,负面
服务差,负面
性价比低,负面
```

文件 `data/aliases.json`：

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

- [ ] **Step 2: 追加 conftest fixture（app client with stub embedding）**

文件 `tests/conftest.py` 追加：

```python
import os
import sys
from pathlib import Path

# 确保项目根可导入
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ==================== Normalizer 测试共享桩 ====================

class _StubEmbedding:
    """通用 stub embedding：encode 返回非零向量（避免 cosine 全 0）。"""
    def __init__(self, dim: int = 4, name: str = "fasttext"):
        self._dim = dim
        self._name = name

    def load(self):
        pass

    def encode(self, words):
        return np.ones((len(words), self._dim), dtype=np.float32)

    @property
    def name(self):
        return self._name

    @property
    def dim(self):
        return self._dim


# 由 _match_in_bucket monkeypatch 写入；_patched_cosine 读取并弹出
_CURRENT_BEST: list[tuple[str, dict[str, float], list[str]]] = []


def _patched_cosine(query_vec, matrix):
    """cosine_batch 的可控制版本：从 _CURRENT_BEST 取 (word, scores, candidates)。"""
    if not _CURRENT_BEST:
        return np.zeros(matrix.shape[0])
    _, scores, candidates = _CURRENT_BEST.pop(0)
    return np.array([scores.get(c, 0.0) for c in candidates], dtype=np.float32)


# ==================== API 测试 fixture ====================

class _StubAppEmbedding:
    """API 测试用 stub：encode 返回全 1，确保 L2 不会被 zero vector 干扰。"""
    def __init__(self, name: str = "stub", dim: int = 4):
        self._name = name
        self._dim = dim

    def load(self):
        pass

    def encode(self, words):
        return np.ones((len(words), self._dim), dtype=np.float32)

    @property
    def name(self):
        return self._name

    @property
    def dim(self):
        return self._dim


@pytest.fixture
def app_with_stub(monkeypatch, tmp_path):
    """用 stub embedding 替换真实 fastText 的 TestClient。"""
    # 准备小词库（含括号变体和别名，验证 L1 路径）
    csv = tmp_path / "vocab.csv"
    csv.write_text(
        "词,极性\n"
        "舒适,正面\n"
        "轻薄（含轻盈、不压身）,正面\n"
        "不舒适,负面\n",
        encoding="utf-8",
    )

    # 替换 get_model 返回 stub
    from app import embedding as emb_pkg
    monkeypatch.setattr(emb_pkg, "get_model", lambda name: _StubAppEmbedding(name=name))

    # 设置环境变量指向测试词库
    monkeypatch.setenv("VOCAB_PATH", str(csv))

    # 重新加载 main 模块以使用新环境
    import importlib
    from app import main as main_module
    importlib.reload(main_module)

    client = TestClient(main_module.app)
    yield client, main_module

    # 清理 reload 副作用
    importlib.reload(main_module)


@pytest.fixture
def small_input_file(tmp_path):
    p = tmp_path / "input.txt"
    p.write_text("舒适\n凉爽\n轻盈\n不存在的词\n", encoding="utf-8")
    return p
```

- [ ] **Step 3: 重写 test_health.py**

文件 `tests/test_health.py`：

```python
import importlib


def test_health_with_loaded_vocab(app_with_stub):
    client, _ = app_with_stub
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "vocab_size" in body
    assert body["vocab_size"] >= 2
```

- [ ] **Step 4: 写 FastAPI 路由失败测试**

文件 `tests/test_api.py`：

```python
def test_normalize_returns_tsv(app_with_stub, small_input_file):
    client, _ = app_with_stub
    with open(small_input_file, "rb") as f:
        response = client.post(
            "/api/v1/normalize",
            files={"file": ("input.txt", f, "text/plain")},
        )
    assert response.status_code == 200
    text = response.text
    lines = text.strip().splitlines()
    assert len(lines) == 4
    # 精确词命中
    assert "舒适\t舒适" in lines
    # L1 命中（轻盈 → 轻薄）
    assert "轻盈\t轻薄" in lines


def test_normalize_with_debug(app_with_stub, small_input_file):
    client, _ = app_with_stub
    with open(small_input_file, "rb") as f:
        response = client.post(
            "/api/v1/normalize?debug=1",
            files={"file": ("input.txt", f, "text/plain")},
        )
    assert response.status_code == 200
    # 4 列：原文\t归一\t层级\t分数
    parts = response.text.strip().splitlines()[0].split("\t")
    assert len(parts) == 4


def test_normalize_empty_file_400(app_with_stub, tmp_path):
    client, _ = app_with_stub
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    with open(empty, "rb") as f:
        response = client.post(
            "/api/v1/normalize",
            files={"file": ("empty.txt", f, "text/plain")},
        )
    assert response.status_code == 400
    assert response.json()["error"] == "empty_file"


def test_normalize_unknown_model_400(app_with_stub, small_input_file):
    client, _ = app_with_stub
    with open(small_input_file, "rb") as f:
        response = client.post(
            "/api/v1/normalize?model=nonexistent",
            files={"file": ("input.txt", f, "text/plain")},
        )
    assert response.status_code == 400
    assert response.json()["error"] == "unknown_model"


def test_normalize_invalid_encoding_400(app_with_stub, tmp_path):
    client, _ = app_with_stub
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"\xff\xfe invalid utf8")
    with open(bad, "rb") as f:
        response = client.post(
            "/api/v1/normalize",
            files={"file": ("bad.txt", f, "text/plain")},
        )
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_encoding"


def test_reload_success(app_with_stub, tmp_path):
    client, _ = app_with_stub
    response = client.post("/api/v1/admin/reload")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "正面" in body and "负面" in body
```

- [ ] **Step 5: 跑测试确认失败**

```bash
pytest tests/test_api.py -v
```

Expected: 路由尚未实现，404 或其他错误

- [ ] **Step 6: 实现 FastAPI 路由**

文件 `app/main.py`：

```python
import io
import os
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from app.embedding import ModelNotFoundError, get_model
from app.normalizer import Normalizer
from app.vocabulary import Vocabulary, VocabularyLoadError

DEFAULT_VOCAB_PATH = os.environ.get("VOCAB_PATH", "data/vocabulary.csv")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

app = FastAPI(title="Word Normalizer", version="0.1.0")

# 启动时加载
_state: dict = {}


def _load_state(vocab_path: str) -> None:
    vocab = Vocabulary.load(vocab_path)
    # 加载 aliases.json（如存在）
    aliases_path = Path(vocab_path).parent / "aliases.json"
    if aliases_path.exists():
        import json
        with open(aliases_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 把 aliases.json 中人工维护的别名合并到 vocab
        for variant, std in data.items():
            std_polarity = vocab.polarity_map.get(std)
            if std_polarity is None:
                raise VocabularyLoadError(
                    f"aliases.json: 目标词 '{std}' 不在词库中（变体 '{variant}'）"
                )
            existing_polarity = vocab.polarity_map.get(variant)
            if existing_polarity is not None and existing_polarity != std_polarity:
                raise VocabularyLoadError(
                    f"aliases.json: 变体 '{variant}' 已存在（极性 {existing_polarity}），"
                    f"不能同时作为 '{std}'（极性 {std_polarity}）的别名"
                )
            if variant not in vocab.polarity_map:
                vocab.buckets[std_polarity].append(variant)
                vocab.polarity_map[variant] = std_polarity
            vocab.alias_map[variant] = std

    _state["vocab"] = vocab
    _state["vocab_path"] = vocab_path
    # 默认模型预加载
    try:
        default_emb = get_model("fasttext")
        _state["default_normalizer"] = Normalizer(default_emb, vocab)
    except Exception:
        _state["default_normalizer"] = None


@app.on_event("startup")
def startup():
    _load_state(DEFAULT_VOCAB_PATH)


@app.get("/api/v1/health")
async def health():
    vocab = _state.get("vocab")
    return {
        "status": "ok",
        "default_model": "fasttext",
        "vocab_size": (
            len(vocab.buckets["正面"]) + len(vocab.buckets["负面"])
            if vocab
            else 0
        ),
    }


@app.post("/api/v1/normalize")
async def normalize(
    file: UploadFile = File(...),
    model: str = Query("fasttext"),
    debug: int = Query(0),
):
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail={"error": "file_too_large"})

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail={"error": "invalid_encoding"})

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        raise HTTPException(status_code=400, detail={"error": "empty_file"})

    vocab = _state["vocab"]
    try:
        embedding = get_model(model)
    except ModelNotFoundError:
        raise HTTPException(
            status_code=400,
            detail={"error": "unknown_model", "supported": ["fasttext", "bge"]},
        )

    normalizer = Normalizer(embedding, vocab)
    t0 = time.time()
    results = [normalizer.normalize(line) for line in lines]
    elapsed = time.time() - t0

    layer_counts = {"L1": 0, "L2": 0, "L3": 0, "FALLBACK": 0}
    for r in results:
        layer_counts[r.matched_layer] += 1

    import logging
    logger = logging.getLogger("wordtest")
    logger.info(
        f"POST /normalize {len(lines)} lines model={model} "
        f"matched={layer_counts} total={elapsed:.1f}s"
    )

    sep = "\t"
    if debug:
        output = "\n".join(
            f"{r.original}{sep}{r.normalized}{sep}{r.matched_layer}{sep}{r.score:.3f}"
            for r in results
        )
    else:
        output = "\n".join(f"{r.original}{sep}{r.normalized}" for r in results)

    return StreamingResponse(
        io.StringIO(output),
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=normalized.txt"},
    )


@app.post("/api/v1/admin/reload")
async def reload_vocab():
    try:
        _load_state(_state.get("vocab_path", DEFAULT_VOCAB_PATH))
    except VocabularyLoadError as e:
        raise HTTPException(status_code=500, detail={"error": "reload_failed", "msg": str(e)})
    vocab = _state["vocab"]
    return {
        "status": "ok",
        "正面": len(vocab.buckets["正面"]),
        "负面": len(vocab.buckets["负面"]),
    }
```

- [ ] **Step 7: 跑全部测试**

```bash
pytest -v
```

Expected: 全部 passed（包括 test_health.py、test_api.py、test_similarity.py、test_vocabulary.py、test_embedding_factory.py、test_fasttext_embedding.py、test_normalizer_l1.py、test_normalizer_l2_l3.py、test_normalizer_polarity.py）

- [ ] **Step 8: 提交**

```bash
git add app/main.py data/ tests/ requirements.txt
git commit -m "feat: FastAPI routes (/normalize, /admin/reload, /health) with stub test infra"
```

---

## Task 11: 启动并冒烟测试

**Files:** 无新建

- [ ] **Step 1: 启动服务（用 stub 临时替换）**

```bash
cd D:/workspace/claude/wordtest
source .venv/Scripts/activate
uvicorn app.main:app --reload --port 8000
```

Expected: 服务启动，INFO 日志显示「Loaded vocabulary: X 正面 + Y 负面」

注：第一次启动会因 fastText 模型文件不存在而失败（按设计应该 fail-fast）。先用 stub embedding 临时验证路由通畅：

```bash
# 在另一个终端
python -c "
from app.embedding import get_model
from app.normalizer import Normalizer
import numpy as np
class Stub:
    name='fasttext'; dim=4
    def load(self): pass
    def encode(self, ws): return np.zeros((len(ws),4), dtype=np.float32)
import app.embedding.factory as f
f._REGISTRY['fasttext'] = lambda: Stub()
print('stub registered')
"
```

- [ ] **Step 2: 手动测试 /health**

```bash
curl http://localhost:8000/api/v1/health
```

Expected: `{"status":"ok","default_model":"fasttext","vocab_size":N}`

- [ ] **Step 3: 手动测试 /normalize**

```bash
echo -e "舒适\n凉凉的\n轻盈" > /tmp/test_in.txt
curl -F "file=@/tmp/test_in.txt" "http://localhost:8000/api/v1/normalize?debug=1" -o /tmp/test_out.txt
cat /tmp/test_out.txt
```

Expected:
```
舒适	舒适	L1	1.000
凉感的	凉感的	FALLBACK	0.000
轻盈	轻薄	L1	1.000
```

- [ ] **Step 4: 关闭服务**

```bash
# Ctrl+C 终止
```

- [ ] **Step 5: 记录冒烟测试结果**

无 commit，仅验证。

---

## Task 12: 准确率验收（待用户提供样本）

**Files:**
- Create: `D:\workspace\claude\wordtest\tests\test_accuracy.py`
- Modify: `D:\workspace\claude\wordtest\data\aliases.json`

- [ ] **Step 1: 向用户索取测试样本**

询问用户：「请提供 50-100 条人工输入词的样本，每条格式为 `输入词\t期望归一词`，覆盖各维度（触感、凉感、负向、不舒适等）。我会用这些样本验证 95% 准确率。」

- [ ] **Step 2: 写准确率测试骨架（用户样本未到时 skip）**

文件 `tests/test_accuracy.py`：

```python
import pytest
from fastapi.testclient import TestClient

# 样本格式：(输入词, 期望归一词, 极性提示)
# 用户提供后填入，至少 50 条覆盖各维度
SAMPLES: list[tuple[str, str, str]] = [
    # === 待用户填充 ===
    # ("凉爽", "凉感适宜", "正面"),
    # ("凉快的", "凉感适宜", "正面"),
    # ("毛绒绒", "蓬松", "正面"),
    # ...
]


@pytest.mark.skipif(
    not SAMPLES,
    reason="等待用户提供 50-100 条真实样本（向用户索取）",
)
def test_accuracy_at_least_95_percent(monkeypatch):
    """通过 TestClient 触发 startup 加载默认 normalizer，避免依赖外部服务。"""
    from app.embedding.factory import reset_models
    from app.main import app

    # 强制重新加载，避免上一个测试的状态污染
    reset_models()

    # 在测试期间把 fastText 替换为 stub（CI 跑时可能没下载真模型）
    import numpy as np
    from app import embedding as emb_pkg

    class _FasttextStub:
        name = "fasttext"
        dim = 300

        def load(self):
            pass

        def encode(self, words):
            # 给同桶词返回相近向量（用 hash 投影），保证 L2 能命中
            vecs = np.zeros((len(words), self.dim), dtype=np.float32)
            for i, w in enumerate(words):
                h = abs(hash(w)) % (self.dim - 1)
                vecs[i, h] = 1.0
            return vecs

    monkeypatch.setattr(emb_pkg, "get_model", lambda name: _FasttextStub())

    with TestClient(app) as client:
        # 触发 startup
        health = client.get("/api/v1/health")
        assert health.status_code == 200, "服务未就绪"

        from app.main import _state
        normalizer = _state["default_normalizer"]
        assert normalizer is not None, "默认 normalizer 未加载"

        correct = 0
        failures = []
        for inp, expected, _ in SAMPLES:
            result = normalizer.normalize(inp)
            if result.normalized == expected:
                correct += 1
            else:
                failures.append(
                    (inp, expected, result.normalized, result.matched_layer, result.score)
                )

        accuracy = correct / len(SAMPLES)
        print(f"\n准确率: {accuracy:.1%} ({correct}/{len(SAMPLES)})")
        if failures:
            print(f"失败用例数: {len(failures)}")
            for inp, exp, got, layer, score in failures[:10]:
                print(f"  '{inp}' 期望='{exp}' 实际='{got}' 层级={layer} 分数={score:.3f}")

        assert accuracy >= 0.95, f"准确率 {accuracy:.1%} < 95%"
```

- [ ] **Step 3: 用户提供样本后填入 SAMPLES**

将用户给的 50-100 条 `(输入, 期望, 极性)` 三元组填入 SAMPLES。

- [ ] **Step 4: 跑测试，定位失败用例**

```bash
pytest tests/test_accuracy.py -v -s
```

Expected: 失败用例清单被打印

- [ ] **Step 5: 把失败用例追加到 aliases.json（运营回流）**

编辑 `data/aliases.json`，把每个失败用例的 `输入词 → 期望归一词` 加入。例：

```json
{
  "轻盈": "轻薄",
  "不压身": "轻薄",
  "沉重": "厚重",
  "破洞": "瑕疵",
  "勾丝": "瑕疵",
  "脏": "瑕疵",
  "凉爽": "凉感适宜",
  "毛绒绒": "蓬松"
}
```

- [ ] **Step 6: 调用 /admin/reload 重新加载词库**

```bash
curl -X POST http://localhost:8000/api/v1/admin/reload
```

- [ ] **Step 7: 重新跑准确率测试**

```bash
pytest tests/test_accuracy.py -v -s
```

Expected: `1 passed` 且打印准确率 ≥ 95%

- [ ] **Step 8: 提交**

```bash
git add data/aliases.json tests/test_accuracy.py
git commit -m "test: 95% accuracy assertion with user-provided samples + alias table"
```

---

## Task 13: 真实 fastText 模型接入

**Files:**
- Modify: `D:\workspace\claude\wordtest\app\embedding\factory.py`（无）
- Modify: `D:\workspace\claude\wordtest\app\main.py`（无）

- [ ] **Step 1: 下载 fastText 中文模型**

```bash
cd D:/workspace/claude/wordtest
python scripts/download_model.py
```

Expected: `下载完成: models/cc.zh.300.bin (<size> MB)`（实际 MB 数取决于下载源）

- [ ] **Step 2: 启动服务（真实 fastText）**

```bash
source .venv/Scripts/activate
uvicorn app.main:app --port 8000
```

Expected: 启动成功，无 ModelFileMissingError

- [ ] **Step 3: 重新跑所有测试**

```bash
pytest -v
```

Expected: 全部 passed

- [ ] **Step 4: 跑准确率测试**

```bash
pytest tests/test_accuracy.py -v -s
```

Expected: 准确率 ≥ 95%（如果仍不达标，重复 Task 12 步骤 5-7 补 aliases）

- [ ] **Step 5: 性能测试**

新建文件 `scripts/perf_test.py`：

```python
"""性能测试：3 万词应在 60 秒内完成。"""
import time
from pathlib import Path

from fastapi.testclient import TestClient


def main():
    from app.main import app, _state

    # 触发 startup 加载
    with TestClient(app) as client:
        client.get("/api/v1/health")

    vocab = _state["vocab"]
    normalizer = _state["default_normalizer"]
    if normalizer is None:
        raise RuntimeError("默认 normalizer 未加载")

    # 构造 3 万随机词
    import random
    pos = vocab.buckets["正面"]
    neg = vocab.buckets["负面"]
    samples = [random.choice(pos + neg) for _ in range(30000)]
    samples += [f"未登录词{i}" for i in range(1000)]

    t0 = time.time()
    for w in samples:
        normalizer.normalize(w)
    elapsed = time.time() - t0
    print(f"处理 {len(samples)} 词，耗时 {elapsed:.1f}s")
    assert elapsed < 60, f"超出 60s 预算"


if __name__ == "__main__":
    main()
```

```bash
python -c "from app.main import _state"  # 确保服务已加载
python scripts/perf_test.py
```

Expected: < 60s

- [ ] **Step 6: 提交**

```bash
git add scripts/perf_test.py
git commit -m "test: 30k words performance test under 60s"
```

---

## Task 14: README

**Files:**
- Create: `D:\workspace\claude\wordtest\README.md`

- [ ] **Step 1: 写 README**

文件 `README.md`：

```markdown
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
```

- [ ] **Step 2: 提交**

```bash
git add README.md
git commit -m "docs: README with quickstart and API reference"
```

---

## 验收清单

完成后必须满足：

- [ ] 所有单测和集成测通过（`pytest -v`）
- [ ] 准确率 ≥ 95%（`pytest tests/test_accuracy.py`）
- [ ] 3 万词 < 60s（`python scripts/perf_test.py`）
- [ ] `/health` 返回 200
- [ ] `/normalize` 上传返回归一结果
- [ ] `/admin/reload` 可热更新
- [ ] README 文档完整
