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
            ("柔软", "正面"),
            ("不舒适", "负面"),
            ("瑕疵", "负面"),
        ]
    )


def test_unknown_polarity_uses_dual_bucket_compare(vocab, monkeypatch):
    """极性未知时双桶都 FALLBACK → 最终 FALLBACK"""
    def fake_cosine(query_vec, matrix):
        return np.zeros(matrix.shape[0])
    monkeypatch.setattr(norm_mod, "cosine_batch", fake_cosine)

    n = Normalizer(_StubEmbedding(), vocab)
    r = n.normalize("凉凉的")
    assert r.matched_layer == "FALLBACK"
    assert r.normalized == "凉凉的"


def test_unknown_polarity_picks_higher_score(vocab, monkeypatch):
    """极性未知时双桶对比取高 → 正面桶 0.7 胜出"""
    _CURRENT_BEST.clear()
    monkeypatch.setattr(norm_mod, "cosine_batch", _patched_cosine)

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
