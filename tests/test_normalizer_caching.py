"""Test that Normalizer instances are cached by model name in app.main.

Bug: every POST /api/v1/normalize call rebuilt a Normalizer, which encodes
all ~190 vocab words via _precompute_vectors() in __init__.

Fix: cache by model name in _state["normalizers"]; build default at startup,
build others lazily on first request.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient


class _CountingEmbedding:
    """Stub embedding that counts encode() calls."""

    def __init__(self, name: str = "bge", dim: int = 4):
        self._name = name
        self._dim = dim
        self.encode_calls: list[list[str]] = []

    def load(self) -> None:
        pass

    def encode(self, words):
        self.encode_calls.append(list(words))
        return np.ones((len(words), self._dim), dtype=np.float32)

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return self._dim


@pytest.fixture
def app_with_counting_stub(monkeypatch, tmp_path):
    """TestClient wired up with a counting stub that supports both 'bge' and 'fasttext'."""
    csv = tmp_path / "vocab.csv"
    csv.write_text(
        "大类,词,极性\n"
        "体感,舒适,正面\n"
        "清洁打理,轻薄,正面\n"
        "体感,不舒适,负面\n"
        "质量,瑕疵,负面\n",
        encoding="utf-8",
    )

    from app import embedding as emb_pkg
    from app.embedding.factory import ModelNotFoundError

    _stub_models: dict[str, _CountingEmbedding] = {}

    def _stub_get_model(name: str) -> _CountingEmbedding:
        if name not in {"fasttext", "bge", "bge_base", "m3e"}:
            raise ModelNotFoundError(f"未知模型: {name}")
        if name not in _stub_models:
            _stub_models[name] = _CountingEmbedding(name=name)
        return _stub_models[name]

    monkeypatch.setattr(emb_pkg, "get_model", _stub_get_model)
    monkeypatch.setenv("VOCAB_PATH", str(csv))

    import importlib
    from app import main as main_module
    importlib.reload(main_module)

    with TestClient(main_module.app) as client:
        yield client, main_module, _stub_models

    importlib.reload(main_module)


def _input_file(tmp_path, lines: list[str]):
    p = tmp_path / "in.txt"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def test_normalizers_cached_identical_per_model(app_with_counting_stub, tmp_path):
    """Two POSTs to /api/v1/normalize with model=bge should yield the same
    Normalizer instance from the cache."""
    client, main_module, _ = app_with_counting_stub

    f = _input_file(tmp_path, ["舒适"])
    with open(f, "rb") as fp:
        client.post("/api/v1/normalize?model=bge", files={"file": ("in.txt", fp, "text/plain")})

    a = main_module._state["normalizers"]["bge"]
    b = main_module._state["normalizers"]["bge"]
    assert a is b, "Normalizer for bge should be the same instance across requests"


def test_normalizer_cache_avoids_repeated_precompute(app_with_counting_stub, tmp_path):
    """Two POSTs with the same model should NOT trigger a second precompute.
    Precompute encodes all vocab words (2 polarities × n words). The 2nd
    request should add exactly n input-word encodes (1 per line), no more."""
    client, _, stub_models = app_with_counting_stub
    f = _input_file(tmp_path, ["舒适", "不舒适", "瑕疵"])
    lines = ["舒适", "不舒适", "瑕疵"]

    # 1st request
    with open(f, "rb") as fp:
        client.post("/api/v1/normalize?model=bge", files={"file": ("in.txt", fp, "text/plain")})
    n_after_first = sum(len(c) for c in stub_models["bge"].encode_calls)

    # 2nd request with same model
    with open(f, "rb") as fp:
        client.post("/api/v1/normalize?model=bge", files={"file": ("in.txt", fp, "text/plain")})
    n_after_second = sum(len(c) for c in stub_models["bge"].encode_calls)

    # Second request should add exactly len(lines) encodes (one per line)
    delta = n_after_second - n_after_first
    assert delta == len(lines), (
        f"2nd request should add {len(lines)} input encodes "
        f"(no re-precompute), but added {delta}. "
        f"Total encode calls: {stub_models['bge'].encode_calls}"
    )


def test_normalizer_lazy_build_for_non_default_model(app_with_counting_stub, tmp_path):
    """First request with model=fasttext should build the normalizer; second
    should reuse the cached one."""
    client, main_module, _ = app_with_counting_stub

    assert "fasttext" not in main_module._state["normalizers"]

    f = _input_file(tmp_path, ["舒适"])
    with open(f, "rb") as fp:
        client.post(
            "/api/v1/normalize?model=fasttext",
            files={"file": ("in.txt", fp, "text/plain")},
        )

    assert "fasttext" in main_module._state["normalizers"]
    a = main_module._state["normalizers"]["fasttext"]
    b = main_module._state["normalizers"]["fasttext"]
    assert a is b


def test_default_normalizer_built_at_startup(app_with_counting_stub):
    """The m3e (default) normalizer should be present after startup, no
    request needed."""
    _, main_module, _ = app_with_counting_stub
    assert "m3e" in main_module._state["normalizers"]
    assert "default_normalizer" not in main_module._state
