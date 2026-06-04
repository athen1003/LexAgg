import numpy as np
import pytest

from app.normalizer import Normalizer, NormalizeResult
from app.vocabulary import Vocabulary


class _StubEmbedding:
    """测试用 embedding stub，可注入预定义相似度。"""

    def __init__(self, name: str = "stub", dim: int = 4):
        self._name = name
        self._dim = dim

    def load(self) -> None:
        pass

    def encode(self, words):
        return np.zeros((len(words), self._dim), dtype=np.float32)

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
