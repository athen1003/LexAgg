"""Test that BgeEmbedding.encode raises if load() was never called.

Bug: encode silently returned zeros if self._model is None. That means
every input silently became FALLBACK with no error.
"""
import pytest

from app.embedding.bge_impl import BgeEmbedding


def test_encode_without_load_raises():
    """Instantiating BgeEmbedding without calling load() should make
    encode() raise RuntimeError, not silently return zeros."""
    emb = BgeEmbedding()
    assert emb._model is None
    with pytest.raises(RuntimeError, match="load"):
        emb.encode(["测试"])
