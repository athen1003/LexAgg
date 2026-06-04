import pytest

from app.embedding.base import EmbeddingModel
from app.embedding.factory import ModelNotFoundError, get_model, list_models, reset_models


class _FakeModel(EmbeddingModel):
    def __init__(self):
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def encode(self, words):
        import numpy as np
        return np.zeros((len(words), 4))

    @property
    def name(self) -> str:
        return "fake"

    @property
    def dim(self) -> int:
        return 4


def test_get_model_unknown_raises():
    reset_models()
    with pytest.raises(ModelNotFoundError):
        get_model("nonexistent")


def test_list_models():
    models = list_models()
    assert "fasttext" in models
    assert "bge" in models


def test_factory_caches_singleton(monkeypatch):
    reset_models()
    monkeypatch.setitem(__import__("app.embedding.factory", fromlist=["_REGISTRY"])._REGISTRY,
                         "fake", _FakeModel)
    a = get_model("fake")
    b = get_model("fake")
    assert a is b
    assert a.loaded is True
