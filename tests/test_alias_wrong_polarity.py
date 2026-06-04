"""Test that the L1 alias branch in _match_in_bucket is correctly
skipped when the alias target's polarity does not match the bucket
polarity.

The branch in app/normalizer.py:
    if word in self.vocab.alias_map:
        std = self.vocab.alias_map[word]
        if self.vocab.polarity_map.get(std) == polarity:
            return NormalizeResult(... L1 ...)

The polarity guard is defensive: in consistent data, _infer_polarity
follows alias_map to determine the bucket, so std polarity always
matches. But if data is inconsistent (alias target in different bucket),
the guard must skip the L1 hit so we don't silently return the
wrong-polarity standard.
"""
import numpy as np
import pytest

from app.normalizer import Normalizer
from app.vocabulary import Vocabulary
from tests.conftest import _StubEmbedding


def test_l1_alias_skipped_when_std_polarity_mismatches_bucket():
    """alias_map[好] = 差 (负面). 好 is in 正面 bucket.

    Calling _match_in_bucket("好", "正面", word_vec) directly:
    - L1 alias: alias_map[好]=差, polarity_map[差]=负面, != bucket 正面
      → branch SKIPPED (this is the guard under test)
    - L1 exact: polarity_map[好]=正面, == 正面 → L1 hit returns "好"
    - The result is "好" (correct, not the wrong-polarity std "差")
    """
    vocab = Vocabulary.load_from_rows([("好", "正面"), ("差", "负面")])
    vocab.alias_map["好"] = "差"  # 差 is 负面, but we test 正面 bucket

    n = Normalizer(_StubEmbedding(), vocab)
    word_vec = np.zeros(n.embedding.dim, dtype=np.float32)

    r = n._match_in_bucket("好", "正面", word_vec)
    # L1 exact match (好 is in 正面 bucket) wins.
    assert r.matched_layer == "L1"
    assert r.normalized == "好"
    # Critically: the wrong-polarity alias std is NOT returned.
    assert r.normalized != "差"


def test_l1_alias_skipped_word_not_in_polarity_map_falls_through():
    """Stronger test: variant is NOT in polarity_map (so L1 exact also
    misses). alias_map[奇怪] = 差 (负面). We call with 正面 bucket.
    The L1 alias branch's polarity guard should skip → L2 runs →
    L2 either hits or returns FALLBACK.

    With a stub that returns a moderate similarity for the bucket
    candidates, we expect the L1 alias branch to be skipped (no L1
    hit) and the result to NOT be "差".
    """
    vocab = Vocabulary.load_from_rows([("好", "正面"), ("差", "负面")])
    # 奇怪 is not in polarity_map. Manually set alias.
    vocab.alias_map["奇怪"] = "差"  # alias points to 负面 std
    assert "奇怪" not in vocab.polarity_map  # L1 exact will miss

    n = Normalizer(_StubEmbedding(name="fasttext"), vocab)
    word_vec = np.zeros(n.embedding.dim, dtype=np.float32)

    # Patch cosine_batch to return low similarity → goes to FALLBACK
    import app.normalizer as norm_mod
    original = norm_mod.cosine_batch
    def low_sim(query_vec, matrix):
        return np.full(matrix.shape[0], 0.1)  # below fasttext accept=0.6
    norm_mod.cosine_batch = low_sim

    try:
        r = n._match_in_bucket("奇怪", "正面", word_vec)
    finally:
        norm_mod.cosine_batch = original

    # The L1 alias branch should have been skipped (std polarity 负面 != bucket 正面).
    # L1 exact also misses (word not in polarity_map).
    # L2 has low similarity → FALLBACK.
    assert r.matched_layer == "FALLBACK"
    assert r.normalized == "奇怪"  # original, not the wrong-polarity "差"
    assert r.normalized != "差"
