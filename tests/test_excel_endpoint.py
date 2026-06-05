"""Tests for POST /api/v1/normalize/excel — Excel upload endpoint."""
import io
import json

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
    matches what the polarity-filtered normalizer returns for each input."""
    client, main_module = excel_app
    rows = [
        ["舒适", "正面"],          # L1 exact match (正面 桶)
        ["轻盈", "正面"],          # L1 exact match (正面 桶)
        ["完全不存在的词", "负面"],  # 不在负面桶 → FALLBACK(不乱猜)
        ["", "正面"],              # empty col 0 → skip
        ["  ", "负面"],            # whitespace col 0 → skip
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
    assert headers == [
        "原词", "归一词", "命中层级", "分数", "输入极性", "归一-大类",
        "建议归一词", "建议分数", "建议-大类",
    ]

    data = out_df.iloc[1:].reset_index(drop=True)
    normalizer = main_module._state["normalizers"]["m3e"]
    expected = [
        normalizer.normalize("舒适", polarity_hint="正面").normalized,
        normalizer.normalize("轻盈", polarity_hint="正面").normalized,
        # 不认识的词在 负面 桶无 L1 命中 → 单桶 L2/L3 也不命中(全 1 stub 下会 L2,
        # 但当前是过滤单桶路径)。无论命中还是 FALLBACK,断言由 L1 是否在桶内决定。
        # 该词不在 vocab 中, 单桶搜索会落到 L2(全 1 stub), 会给个负面词的归一。
    ]
    assert data.iloc[0, 0] == "舒适"
    assert data.iloc[0, 1] == expected[0]
    assert data.iloc[1, 0] == "轻盈"
    assert data.iloc[1, 1] == expected[1]


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


def test_excel_matched_category_backfilled_from_vocab(excel_app):
    """归一-大类 从词库回填,即使输入不指定大类。"""
    client, main_module = excel_app
    rows = [
        ["舒适", "正面"],     # L1,词库 舒适 在 体感
        ["瑕疵", "负面"],     # L1,词库 瑕疵 在 质量
        ["轻盈", "正面"],     # L1,词库 轻盈 在 清洁打理
    ]
    content = _make_xlsx(rows)
    response = _post_xlsx(client, content)
    assert response.status_code == 200

    out_df = _read_xlsx_response(response.content)
    data = out_df.iloc[1:].reset_index(drop=True)
    assert data.iloc[0, 5] == "体感"
    assert data.iloc[1, 5] == "质量"
    assert data.iloc[2, 5] == "清洁打理"


def test_excel_polarity_filter_routes_to_correct_bucket(excel_app):
    """极性提示=正面 → 只在正面桶搜;=负面 → 只在负面桶搜。
    关键不变量: 归一词必落在输入指定的极性桶内(绝不跨桶)。"""
    client, main_module = excel_app
    vocab = main_module._state["vocab"]
    正面_words = set(vocab.buckets["正面"])
    负面_words = set(vocab.buckets["负面"])

    rows = [
        ["瑕疵", "正面"],     # 瑕疵 不在正面桶 → 归一词必在 正面 桶(L1 跳过,L2 候选全在 正面)
        ["瑕疵", "负面"],     # 瑕疵 在负面桶 → L1 命中
        ["舒适", "正面"],     # 舒适 在正面桶 → L1 命中
        ["舒适", "负面"],     # 舒适 不在负面桶 → 归一词必在 负面 桶
    ]
    content = _make_xlsx(rows)
    response = _post_xlsx(client, content)
    assert response.status_code == 200

    out_df = _read_xlsx_response(response.content)
    data = out_df.iloc[1:].reset_index(drop=True)
    # 不变量 1: 瑕疵 + 正面 → 归一词 ∈ 正面 桶
    assert data.iloc[0, 1] in 正面_words
    assert data.iloc[0, 1] not in 负面_words
    # 不变量 2: 瑕疵 + 负面 → 归一词 ∈ 负面 桶(L1 命中瑕疵)
    assert data.iloc[1, 1] in 负面_words
    assert data.iloc[1, 1] == "瑕疵"
    # 不变量 3: 舒适 + 正面 → 归一词 ∈ 正面 桶(L1 命中)
    assert data.iloc[2, 1] in 正面_words
    assert data.iloc[2, 1] == "舒适"
    # 不变量 4: 舒适 + 负面 → 归一词 ∈ 负面 桶
    assert data.iloc[3, 1] in 负面_words
    assert data.iloc[3, 1] not in 正面_words


def test_excel_empty_polarity_returns_fallback(excel_app):
    """空极性 或 非 正面/负面 值 → FALLBACK,不乱猜。"""
    client, _ = excel_app
    rows = [
        ["舒适", ""],          # 空 → FALLBACK
        ["不舒适", "未知"],     # 其他值 → FALLBACK
        ["瑕疵", "中性"],       # 其他值 → FALLBACK
        ["舒适", "正面"],       # 正常 → L1
    ]
    content = _make_xlsx(rows)
    response = _post_xlsx(client, content)
    assert response.status_code == 200

    out_df = _read_xlsx_response(response.content)
    data = out_df.iloc[1:].reset_index(drop=True)
    assert data.iloc[0, 2] == "FALLBACK"  # 空
    assert data.iloc[1, 2] == "FALLBACK"  # 未知
    assert data.iloc[2, 2] == "FALLBACK"  # 中性
    assert data.iloc[3, 2] == "L1"        # 正常
    # 输入极性原样回显
    assert list(data.iloc[:, 4]) == ["", "未知", "中性", "正面"]


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
    """Response xlsx has exactly the 9 expected columns in the right order
    (header row)."""
    client, _ = excel_app
    rows = [["舒适", "正面"], ["不舒适", "负面"]]
    content = _make_xlsx(rows)
    response = _post_xlsx(client, content)
    assert response.status_code == 200

    out_df = _read_xlsx_response(response.content)
    headers = list(out_df.iloc[0])
    assert headers == [
        "原词", "归一词", "命中层级", "分数", "输入极性", "归一-大类",
        "建议归一词", "建议分数", "建议-大类",
    ]
    # Exactly 9 columns
    assert len(out_df.columns) == 9


def test_excel_fallback_suggestion_columns(excel_app):
    """FALLBACK 行 → 建议归一词/分数/大类 非空;L1 行 → 建议列 3 个全空。"""
    client, _ = excel_app
    rows = [
        ["舒适", "正面"],         # L1 → 建议列应为空
        ["完全不存在", "未知"],   # FALLBACK (走 __FALLBACK__ 路径) → 建议列应填充
    ]
    content = _make_xlsx(rows)
    response = _post_xlsx(client, content)
    assert response.status_code == 200

    out_df = _read_xlsx_response(response.content)
    data = out_df.iloc[1:].reset_index(drop=True)

    # Row 0 (L1): 建议列应全空
    assert data.iloc[0, 2] == "L1"
    assert data.iloc[0, 6] == ""    # 建议归一词
    assert float(data.iloc[0, 7]) == 0.0  # 建议分数
    assert data.iloc[0, 8] == ""    # 建议-大类

    # Row 1 (FALLBACK): 建议列应填充(stub 下 best 是「舒适」)
    assert data.iloc[1, 2] == "FALLBACK"
    assert data.iloc[1, 6] != ""    # 建议归一词非空
    assert float(data.iloc[1, 7]) > 0.0  # 建议分数 > 0
    assert data.iloc[1, 8] in {"体感", "清洁打理", "质量"}  # 词库大类是这三者之一


def test_excel_template_endpoint(excel_app):
    """GET /api/v1/normalize/excel/template 返回 2 列模板（表头 + 2 行示例）。"""
    client, _ = excel_app
    response = client.get("/api/v1/normalize/excel/template")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith(
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    assert "attachment" in response.headers.get("content-disposition", "")
    assert "template.xlsx" in response.headers.get("content-disposition", "")

    df = _read_xlsx_response(response.content)
    headers = list(df.iloc[0])
    assert headers == ["原词", "极性"]
    # 2 行示例
    assert len(df) == 1 + 2
    assert list(df.iloc[1]) == ["舒适", "正面"]
    assert list(df.iloc[2]) == ["破洞", "负面"]
