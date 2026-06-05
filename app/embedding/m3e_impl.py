"""M3E-base embedding via sentence-transformers.

Model: moka-ai/m3e-base (~400MB, dim=768)
中文社区常胜军,在中文 STS 任务上常优于 BGE。
"""
from pathlib import Path

import numpy as np

from app.embedding.base import EmbeddingModel


class M3eEmbedding(EmbeddingModel):
    _MODEL_NAME = "moka-ai/m3e-base"
    _CACHE_DIR = str(
        Path(__file__).resolve().parent.parent.parent / "models" / "m3e_base"
    )
    _DIM = 768

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
            raise RuntimeError("M3eEmbedding.encode called before load()")
        vecs = self._model.encode(
            words,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    @property
    def name(self) -> str:
        return "m3e"

    @property
    def dim(self) -> int:
        return self._DIM
