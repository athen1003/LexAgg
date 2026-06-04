"""BGE-small-zh embedding via sentence-transformers.

Auto-adapts device: uses CUDA if available, falls back to CPU.
Model: BAAI/bge-small-zh-v1.5 (~95MB, dim=512)
Auto-downloads on first use to local models/bge/ cache.
"""
from pathlib import Path

import numpy as np

from app.embedding.base import EmbeddingModel


class BgeEmbedding(EmbeddingModel):
    _MODEL_NAME = "BAAI/bge-small-zh-v1.5"
    _CACHE_DIR = str(
        Path(__file__).resolve().parent.parent.parent / "models" / "bge"
    )

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or self._MODEL_NAME
        self._model = None
        self._device = "unknown"

    def load(self) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        Path(self._CACHE_DIR).mkdir(parents=True, exist_ok=True)
        self._model = SentenceTransformer(
            self.model_name,
            device=device,
            cache_folder=self._CACHE_DIR,
        )
        self._device = device

    def encode(self, words: list[str]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("BgeEmbedding.encode called before load()")
        # normalize_embeddings=True 让 cosine 退化为点积
        vecs = self._model.encode(
            words,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    @property
    def name(self) -> str:
        return "bge"

    @property
    def dim(self) -> int:
        return 512
