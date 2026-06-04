"""占位实现，由后续 Task（按需）替换为真实 BGE。"""
import numpy as np

from app.embedding.base import EmbeddingModel


class BgeEmbedding(EmbeddingModel):
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        self.model_name = model_name
        self._model = None

    def load(self) -> None:
        # 实际实现见 BGE Task（按需）
        self._model = None

    def encode(self, words: list[str]) -> np.ndarray:
        return np.zeros((len(words), 512), dtype=np.float32)

    @property
    def name(self) -> str:
        return "bge"

    @property
    def dim(self) -> int:
        return 512
