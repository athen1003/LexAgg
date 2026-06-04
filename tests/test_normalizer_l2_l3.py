import numpy as np
import pytest

from app import normalizer as norm_mod
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
    """L2 命中：相似度 0.8 ≥ 0.6 阈值"""
    monkeypatch.setattr(norm_mod, "cosine_batch", _patched_cosine)
    _CURRENT_BEST.clear()

    n = Normalizer(_StubEmbedding(), vocab)
    original_inner = n._match_in_bucket

    def patched_inner(word, polarity, word_vec):
        candidates = vocab.get_bucket(polarity)
        scores = {"凉感适宜": 0.8, "舒适": 0.5, "轻薄": 0.4}
        _CURRENT_BEST.append((word, scores, candidates))
        return original_inner(word, polarity, word_vec)

    monkeypatch.setattr(n, "_match_in_bucket", patched_inner)

    r = n.normalize("凉爽")
    assert r.matched_layer == "L2"
    assert r.normalized == "凉感适宜"
    assert abs(r.score - 0.8) < 1e-6


def test_l2_below_threshold_falls_through(vocab, monkeypatch):
    """L2 全部相似度 0.1 < 0.4 fallback_to_edit 阈值 → FALLBACK"""
    monkeypatch.setattr(norm_mod, "cosine_batch", _patched_cosine)
    _CURRENT_BEST.clear()

    n = Normalizer(_StubEmbedding(), vocab)
    original_inner = n._match_in_bucket

    def patched_inner(word, polarity, word_vec):
        candidates = vocab.get_bucket(polarity)
        scores = {c: 0.1 for c in candidates}  # 全低
        _CURRENT_BEST.append((word, scores, candidates))
        return original_inner(word, polarity, word_vec)

    monkeypatch.setattr(n, "_match_in_bucket", patched_inner)

    r = n.normalize("不认识的词")
    # 0.1 < 0.4 fallback_to_edit → FALLBACK
    assert r.matched_layer == "FALLBACK"
    assert r.normalized == "不认识的词"


def test_l3_match_with_low_edit_ratio(vocab, monkeypatch):
    """输入 "凉爽适宜"（4字），与 "凉感适宜" 编辑距离 1，比率 1/4 = 0.25 ≤ 0.3 → L3 命中"""
    monkeypatch.setattr(norm_mod, "cosine_batch", _patched_cosine)
    _CURRENT_BEST.clear()

    n = Normalizer(_StubEmbedding(), vocab)
    original_inner = n._match_in_bucket

    def patched_inner(word, polarity, word_vec):
        candidates = vocab.get_bucket(polarity)
        scores = {c: 0.5 if c == "凉感适宜" else 0.0 for c in candidates}
        _CURRENT_BEST.append((word, scores, candidates))
        return original_inner(word, polarity, word_vec)

    monkeypatch.setattr(n, "_match_in_bucket", patched_inner)

    r = n.normalize("凉爽适宜")
    # 0.5 ≥ 0.4 → 进 L3；编辑距离 1/4 = 0.25 ≤ 0.3 → 命中
    assert r.matched_layer == "L3"
    assert r.normalized == "凉感适宜"


def test_l3_reject_high_edit_ratio(vocab, monkeypatch):
    """输入 "凉凉的"（3字），与所有候选编辑距离比率 > 0.3 → 拒绝 L3 → FALLBACK"""
    monkeypatch.setattr(norm_mod, "cosine_batch", _patched_cosine)
    _CURRENT_BEST.clear()

    n = Normalizer(_StubEmbedding(), vocab)
    original_inner = n._match_in_bucket

    def patched_inner(word, polarity, word_vec):
        candidates = vocab.get_bucket(polarity)
        scores = {c: 0.5 for c in candidates}  # 都中等
        _CURRENT_BEST.append((word, scores, candidates))
        return original_inner(word, polarity, word_vec)

    monkeypatch.setattr(n, "_match_in_bucket", patched_inner)

    r = n.normalize("凉凉的")
    # 0.5 ≥ 0.4 → 进 L3；编辑距离比率 > 0.3 → FALLBACK
    assert r.matched_layer == "FALLBACK"
    assert r.normalized == "凉凉的"
