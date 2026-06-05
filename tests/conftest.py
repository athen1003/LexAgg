import tempfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def sample_vocab_csv():
    content = """大类,词,极性
体感,舒适,正面
体感,触感好,正面
体感,柔软,正面
体感,凉感适宜,正面
体感,凉感太弱或太凉,负面
体感,不舒适,负面
体感,触感差,负面
清洁打理,轻薄（含轻盈、不压身）,正面
质量,瑕疵（含破洞、勾丝、脏）,负面
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
    content = "大类,词,极性\n"
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
    p.write_text("大类,词,极性\n体感,舒适,正向\n", encoding="utf-8")
    return str(p)


@pytest.fixture
def duplicate_word_csv(tmp_path):
    p = tmp_path / "dup.csv"
    p.write_text("大类,词,极性\n体感,舒适,正面\n体感,舒适,负面\n", encoding="utf-8")
    return str(p)


# ==================== Normalizer 共享测试桩 ====================

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
        "大类,词,极性\n"
        "体感,舒适,正面\n"
        "清洁打理,轻薄（含轻盈、不压身）,正面\n"
        "体感,不舒适,负面\n",
        encoding="utf-8",
    )

    # 替换 get_model 返回 stub（未知模型仍抛 ModelNotFoundError）
    from app import embedding as emb_pkg
    from app.embedding.factory import ModelNotFoundError

    _stub_models: dict[str, "_StubAppEmbedding"] = {}

    def _stub_get_model(name: str) -> _StubAppEmbedding:
        if name not in {"fasttext", "bge", "bge_base", "m3e"}:
            raise ModelNotFoundError(f"未知模型: {name}")
        if name not in _stub_models:
            _stub_models[name] = _StubAppEmbedding(name=name)
        return _stub_models[name]

    monkeypatch.setattr(emb_pkg, "get_model", _stub_get_model)

    # 设置环境变量指向测试词库
    monkeypatch.setenv("VOCAB_PATH", str(csv))

    # 重新加载 main 模块以使用新环境
    import importlib
    from app import main as main_module
    importlib.reload(main_module)

    # 用 context manager 触发 startup 事件加载词库
    with TestClient(main_module.app) as client:
        yield client, main_module

    # 清理 reload 副作用
    importlib.reload(main_module)


@pytest.fixture
def small_input_file(tmp_path):
    p = tmp_path / "input.txt"
    p.write_text("舒适\n凉爽\n轻盈\n不存在的词\n", encoding="utf-8")
    return p
