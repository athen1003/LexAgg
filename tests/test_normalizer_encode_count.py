"""Test that _normalize_inner encodes the input word exactly once per call.

For unknown-polarity inputs (dual-bucket path), we should encode ONCE, then
reuse the vector across both buckets. For known-polarity inputs, we encode
once per call (the encode is cheap relative to dual-bucket encoding twice,
and the L1 branches still run as fast-exit checks).

The previous bug: unknown-polarity inputs encoded twice (once per bucket).
"""
import numpy as np

from app.normalizer import Normalizer
from app.vocabulary import Vocabulary


class _CountingEmbedding:
    """Stub embedding that counts encode() calls and the words passed in."""

    def __init__(self, dim: int = 4, name: str = "bge"):
        self._dim = dim
        self._name = name
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


def _vocab():
    return Vocabulary.load_from_rows(
        [
            ("舒适", "正面"),
            ("柔软", "正面"),
            ("不舒适", "负面"),
            ("瑕疵", "负面"),
        ]
    )


def test_unknown_polarity_encodes_once_per_input():
    """5 unknown-polarity inputs should produce exactly 5 input encodes total,
    NOT 10 (which is what the bug produces: 2 buckets * 1 encode each)."""
    emb = _CountingEmbedding()
    n = Normalizer(emb, _vocab())
    # Precompute already ran during Normalizer.__init__; reset the log.
    emb.encode_calls.clear()

    inputs = ["不认识的词A", "不认识的词B", "不认识的词C", "不认识的词D", "不认识的词E"]
    for w in inputs:
        n.normalize(w)

    total_inputs = sum(len(c) for c in emb.encode_calls)
    assert total_inputs == len(inputs), (
        f"Expected {len(inputs)} input encodes, got {total_inputs}. "
        f"Calls: {emb.encode_calls}"
    )
    for c in emb.encode_calls:
        assert len(c) == 1, f"Each encode call should be 1 word, got {c}"


def test_known_polarity_encodes_once_per_input():
    """5 known-polarity inputs → 5 encode calls (one per call)."""
    emb = _CountingEmbedding()
    n = Normalizer(emb, _vocab())
    emb.encode_calls.clear()

    # All five have known polarity (alias_map path or L1 exact match).
    # With the simpler design we always encode once per call.
    inputs = ["舒适", "柔软", "不舒适", "瑕疵", "舒适"]
    for w in inputs:
        n.normalize(w)

    total_inputs = sum(len(c) for c in emb.encode_calls)
    assert total_inputs == 5, (
        f"Expected 5 input encodes, got {total_inputs}. "
        f"Calls: {emb.encode_calls}"
    )


def test_zero_inputs_zero_encodes():
    """No normalize calls → no encodes (sanity check)."""
    emb = _CountingEmbedding()
    Normalizer(emb, _vocab())
    emb.encode_calls.clear()
    assert sum(len(c) for c in emb.encode_calls) == 0
