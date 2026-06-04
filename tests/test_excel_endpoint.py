"""Tests for POST /api/v1/normalize/excel — Excel upload endpoint."""
import io
import json
import sys

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


# ==================== Fixtures ====================


class _StubEmbedding:
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
def excel_app(monkeypatch, tmp_path):
    """TestClient wired with stub embeddings (bge + fasttext)."""
    csv = tmp_path / "vocab.csv"
    csv.write_text(
        "词,极性\n"
        "舒适,正面\n"
        "轻盈,正面\n"
        "不舒适,负面\n"
        "瑕疵,负面\n",
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

    import importlib
    from app import main as main_module
    importlib.reload(main_module)

    with TestClient(main_module.app) as client:
        yield client, main_module

    importlib.reload(main_module)


def _make_xlsx(rows: list[list]) -> bytes:
    """Build a minimal xlsx (no header row) from a 2D list of strings."""
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_excel(buf, index=False, header=False, engine="openpyxl")
    return buf.getvalue()


def _post_xlsx(client, content: bytes, filename: str = "input.xlsx",
               content_type: str = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
               query: str = ""):
    return client.post(
        f"/api/v1/normalize/excel{query}",
        files={"file": (filename, io.BytesIO(content), content_type)},
    )


def _read_xlsx_response(content: bytes) -> pd.DataFrame:
    """Read response xlsx, preserving empty strings (not converting to NaN)."""
    return pd.read_excel(
        io.BytesIO(content), dtype=str, header=None, keep_default_na=False
    )


# ==================== Tests ====================


def test_excel_round_trip(excel_app):
    """Upload a 5-row xlsx (3 valid + 2 empty col-0), get xlsx back, parse,
    assert output has 3 rows + headers in the right order, and that 归一词
    matches Normalizer.normalize(word).normalized for each input."""
    client, main_module = excel_app
    rows = [
        ["舒适", "正面"],   # L1 exact match
        ["轻盈", "正面"],   # FALLBACK (no alias defined for 轻盈 in this vocab)
        ["完全不存在的词", "负面"],  # FALLBACK
        ["", "正面"],       # empty col 0 → skip
        ["  ", "负面"],     # whitespace col 0 → skip
    ]
    content = _make_xlsx(rows)
    response = _post_xlsx(client, content)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith(
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # Parse the response xlsx
    out_df = _read_xlsx_response(response.content)
    # row 0 = headers, rows 1..n = data
    assert len(out_df) == 1 + 3, f"expected 4 rows (1 header + 3 data), got {len(out_df)}"
    # Header order
    headers = list(out_df.iloc[0])
    assert headers == ["原词", "归一词", "命中层级", "分数", "输入极性"]

    data = out_df.iloc[1:].reset_index(drop=True)
    # 归一词 for each input matches what the normalizer would return
    normalizer = main_module._state["normalizers"]["bge"]
    for i, word in enumerate(["舒适", "轻盈", "完全不存在的词"]):
        assert data.iloc[i, 0] == word, f"row {i} 原词 mismatch"
        expected_normalized = normalizer.normalize(word).normalized
        assert data.iloc[i, 1] == expected_normalized, (
            f"row {i} 归一词 mismatch: got {data.iloc[i, 1]!r}, expected {expected_normalized!r}"
        )


def test_excel_polarity_hint_preserved(excel_app):
    """Input with 正面/负面 in col 1, output 输入极性 column matches."""
    client, _ = excel_app
    rows = [
        ["舒适", "正面"],
        ["不舒适", "负面"],
        ["瑕疵", ""],          # blank polarity hint
        ["轻盈", "未知"],       # unknown polarity hint — passed through as-is
    ]
    content = _make_xlsx(rows)
    response = _post_xlsx(client, content)
    assert response.status_code == 200, response.text

    out_df = _read_xlsx_response(response.content)
    data = out_df.iloc[1:].reset_index(drop=True)
    assert list(data.iloc[:, 4]) == ["正面", "负面", "", "未知"]


def test_excel_skips_empty_rows(excel_app):
    """Input with blank first-column rows, output excludes them."""
    client, _ = excel_app
    rows = [
        ["舒适", "正面"],
        [None, "负面"],      # NaN
        ["", "正面"],         # empty string
        ["  ", "负面"],       # whitespace
        ["轻盈", "正面"],
    ]
    content = _make_xlsx(rows)
    response = _post_xlsx(client, content)
    assert response.status_code == 200, response.text

    out_df = _read_xlsx_response(response.content)
    # 1 header + 2 data rows
    assert len(out_df) == 3, f"expected 3 rows (1 header + 2 data), got {len(out_df)}"
    data = out_df.iloc[1:].reset_index(drop=True)
    assert list(data.iloc[:, 0]) == ["舒适", "轻盈"]


def test_excel_too_many_rows(excel_app):
    """50,001 rows → 400 too_many_rows."""
    client, _ = excel_app
    rows = [[f"词{i}", ""] for i in range(50_001)]
    content = _make_xlsx(rows)
    response = _post_xlsx(client, content)
    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "too_many_rows"
    assert body["limit"] == 50_000


def test_excel_non_xlsx_rejected(excel_app):
    """Upload a .txt file, expect 400 invalid_file_type."""
    client, _ = excel_app
    # Send txt content with txt content-type
    response = _post_xlsx(
        client,
        "舒适\n不舒适\n".encode("utf-8"),
        filename="input.txt",
        content_type="text/plain",
    )
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_file_type"


def test_excel_unknown_model(excel_app):
    """?model=foo → 400 unknown_model."""
    client, _ = excel_app
    rows = [["舒适", "正面"]]
    content = _make_xlsx(rows)
    response = _post_xlsx(client, content, query="?model=foo")
    assert response.status_code == 400
    assert response.json()["error"] == "unknown_model"


def test_excel_summary_fields(excel_app):
    """Response xlsx has exactly the 5 expected columns in the right order
    (header row)."""
    client, _ = excel_app
    rows = [["舒适", "正面"], ["不舒适", "负面"]]
    content = _make_xlsx(rows)
    response = _post_xlsx(client, content)
    assert response.status_code == 200

    out_df = _read_xlsx_response(response.content)
    headers = list(out_df.iloc[0])
    assert headers == ["原词", "归一词", "命中层级", "分数", "输入极性"]
    # Exactly 5 columns
    assert len(out_df.columns) == 5
