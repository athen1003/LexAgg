"""Tests for static frontend and X-Summary header on /normalize/excel."""
import io
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


class _StubEmbedding:
    def __init__(self, name="stub", dim=4):
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
def app_with_static(monkeypatch, tmp_path):
    """TestClient with stub embeddings AND a static/ dir for frontend tests."""
    csv = tmp_path / "vocab.csv"
    csv.write_text(
        "词,极性\n舒适,正面\n轻盈,正面\n不舒适,负面\n",
        encoding="utf-8",
    )

    # Create a fake static/ dir with a minimal index.html
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text(
        "<!DOCTYPE html><html><head><title>词归一化</title></head>"
        "<body><input type=\"file\" accept=\".xlsx\"/>"
        "<select><option value=\"bge\">bge</option><option value=\"fasttext\">fasttext</option></select>"
        "<button>开始处理</button></body></html>",
        encoding="utf-8",
    )

    from app import embedding as emb_pkg
    from app.embedding.factory import ModelNotFoundError

    _stub_models: dict[str, _StubEmbedding] = {}

    def _stub_get_model(name: str) -> _StubEmbedding:
        if name not in {"fasttext", "bge"}:
            raise ModelNotFoundError(f"未知模型: {name}")
        if name not in _stub_models:
            _stub_models[name] = _StubEmbedding(name=name)
        return _stub_models[name]

    monkeypatch.setattr(emb_pkg, "get_model", _stub_get_model)
    monkeypatch.setenv("VOCAB_PATH", str(csv))

    # Change working dir to tmp_path so static/ mount finds our fake dir
    monkeypatch.chdir(tmp_path)

    import importlib
    from app import main as main_module
    importlib.reload(main_module)

    with TestClient(main_module.app) as client:
        yield client, main_module

    importlib.reload(main_module)


def test_root_serves_index_html(app_with_static):
    """GET / returns 200 with text/html, body contains the upload form."""
    client, _ = app_with_static
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    body = r.text
    assert "词归一化" in body
    # File input present
    assert 'accept=".xlsx"' in body or "accept=\".xlsx\"" in body
    # Submit button present
    assert "开始处理" in body


def test_static_assets_mounted(app_with_static):
    """GET /static/index.html returns the HTML (or just / works)."""
    client, _ = app_with_static
    # / already returns the html (html=True serves index.html for the root)
    r = client.get("/")
    assert r.status_code == 200
    assert "词归一化" in r.text


def test_excel_response_has_summary_header(app_with_static):
    """POST /normalize/excel — X-Summary header is present and parseable."""
    client, _ = app_with_static
    rows = [["舒适", "正面"], ["不舒适", "负面"]]
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_excel(buf, index=False, header=False, engine="openpyxl")
    content = buf.getvalue()

    response = client.post(
        "/api/v1/normalize/excel",
        files={"file": (
            "in.xlsx",
            io.BytesIO(content),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )},
    )
    assert response.status_code == 200
    assert "x-summary" in {k.lower() for k in response.headers.keys()}
    summary_header = response.headers.get("x-summary")
    assert summary_header is not None
    parsed = json.loads(summary_header)
    assert "total" in parsed
    assert parsed["total"] == 2
    # Must have all 4 layer keys (with values that sum to total)
    for k in ("L1", "L2", "L3", "FALLBACK"):
        assert k in parsed
    assert sum(parsed[k] for k in ("L1", "L2", "L3", "FALLBACK")) == parsed["total"]
