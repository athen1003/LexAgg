"""Tests for /api/v1/admin/fallbacks — 累计 FALLBACK 词 + 建议归类 + 分组。"""
import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


class _StubEmbedding:
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
def admin_app(monkeypatch, tmp_path):
    """TestClient with stub embeddings + 简易词库。"""
    csv = tmp_path / "vocab.csv"
    csv.write_text(
        "大类,词,极性\n"
        "体感,舒适,正面\n"
        "清洁打理,轻盈,正面\n"
        "体感,不舒适,负面\n"
        "质量,瑕疵,负面\n",
        encoding="utf-8",
    )

    from app import embedding as emb_pkg
    from app.embedding.factory import ModelNotFoundError

    _stub_models: dict[str, _StubEmbedding] = {}

    def _stub_get_model(name: str) -> _StubEmbedding:
        if name not in {"fasttext", "bge", "bge_base", "m3e"}:
            raise ModelNotFoundError(f"未知模型: {name}")
        if name not in _stub_models:
            _stub_models[name] = _StubEmbedding(name=name)
        return _stub_models[name]

    monkeypatch.setattr(emb_pkg, "get_model", _stub_get_model)
    monkeypatch.setenv("VOCAB_PATH", str(csv))
    monkeypatch.setenv("ADMIN_TOKEN", "test-token")

    import importlib
    from app import main as main_module
    importlib.reload(main_module)

    with TestClient(main_module.app) as client:
        yield client, main_module

    importlib.reload(main_module)


AUTH = {"Authorization": "Bearer test-token"}


def _make_xlsx(rows: list[list]) -> bytes:
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_excel(buf, index=False, header=False, engine="openpyxl")
    return buf.getvalue()


def _post_xlsx(client, content: bytes):
    return client.post(
        "/api/v1/normalize/excel",
        files={"file": (
            "in.xlsx",
            io.BytesIO(content),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )},
    )


def test_admin_fallbacks_empty(admin_app):
    """未上传数据 → 端点返回空结构。"""
    client, _ = admin_app
    client.post("/api/v1/admin/fallbacks/reset", headers=AUTH)
    r = client.get("/api/v1/admin/fallbacks", headers=AUTH)
    assert r.status_code == 200
    data = r.json()
    assert data["total_unique"] == 0
    assert data["total_freq"] == 0
    assert data["filtered"] == 0
    assert data["by_suggestion"] == []


def test_admin_fallbacks_groups_by_suggestion(admin_app):
    """上传含 FALLBACK 词的 xlsx,端点按「建议归一词」分组、按总频次降序。"""
    client, _ = admin_app
    client.post("/api/v1/admin/fallbacks/reset", headers=AUTH)

    rows = [
        ["随机新词A", "未知"],  # FALLBACK
        ["随机新词B", "未知"],  # FALLBACK
        ["舒适", "正面"],        # L1,不计入 fallback
    ]
    r = _post_xlsx(client, _make_xlsx(rows))
    assert r.status_code == 200

    r = client.get("/api/v1/admin/fallbacks", headers=AUTH)
    data = r.json()
    assert data["total_unique"] == 2
    assert data["total_freq"] == 2
    assert data["filtered"] == 2
    all_fallbacks = [f for g in data["by_suggestion"] for f in g["fallbacks"]]
    assert len(all_fallbacks) == 2
    seen = {f["word"] for f in all_fallbacks}
    assert seen == {"随机新词A", "随机新词B"}
    for f in all_fallbacks:
        assert f["freq"] == 1
        assert f["score"] > 0
    freqs = [g["total_freq"] for g in data["by_suggestion"]]
    assert freqs == sorted(freqs, reverse=True)


def test_admin_fallbacks_min_freq_filter(admin_app):
    """min_freq 过滤掉低频词。"""
    client, _ = admin_app
    client.post("/api/v1/admin/fallbacks/reset", headers=AUTH)

    rows = [
        ["随机新词A", "未知"],
        ["随机新词A", "未知"],
        ["随机新词B", "未知"],
    ]
    r = _post_xlsx(client, _make_xlsx(rows))
    assert r.status_code == 200

    r = client.get("/api/v1/admin/fallbacks?min_freq=1", headers=AUTH)
    assert r.json()["filtered"] == 2

    r = client.get("/api/v1/admin/fallbacks?min_freq=2", headers=AUTH)
    data = r.json()
    assert data["filtered"] == 1
    all_fallbacks = [f for g in data["by_suggestion"] for f in g["fallbacks"]]
    assert all_fallbacks[0]["word"] == "随机新词A"
    assert all_fallbacks[0]["freq"] == 2


def test_admin_fallbacks_reset_clears_accumulator(admin_app):
    """/fallbacks/reset 清空累计,后续 GET 返回空。"""
    client, _ = admin_app

    r = _post_xlsx(client, _make_xlsx([["随机新词", "未知"]]))
    assert r.status_code == 200
    r = client.get("/api/v1/admin/fallbacks", headers=AUTH)
    assert r.json()["total_unique"] == 1

    r = client.post("/api/v1/admin/fallbacks/reset", headers=AUTH)
    assert r.status_code == 200

    r = client.get("/api/v1/admin/fallbacks", headers=AUTH)
    data = r.json()
    assert data["total_unique"] == 0
    assert data["total_freq"] == 0
    assert data["by_suggestion"] == []


def test_admin_fallbacks_unknown_model(admin_app):
    """?model=foo → 400 unknown_model(需先有 FALLBACK 数据,才会触发 model 校验)。"""
    client, _ = admin_app
    client.post("/api/v1/admin/fallbacks/reset", headers=AUTH)
    _post_xlsx(client, _make_xlsx([["随机新词", "未知"]]))

    r = client.get("/api/v1/admin/fallbacks?model=foo", headers=AUTH)
    assert r.status_code == 400
    assert r.json()["error"] == "unknown_model"
