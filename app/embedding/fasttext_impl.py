"""占位实现，由后续 Task 6 替换为真实 fastText。"""
import numpy as np

from app.embedding.base import EmbeddingModel


class FastTextEmbedding(EmbeddingModel):
    def __init__(self, model_path: str = "models/cc.zh.300.bin"):
        self.model_path = model_path
        self._model = None

    def load(self) -> None:
        # 实际实现见 Task 6
        self._model = None

    def encode(self, words: list[str]) -> np.ndarray:
        return np.zeros((len(words), 300), dtype=np.float32)

    @property
    def name(self) -> str:
        return "fasttext"

    @property
    def dim(self) -> int:
        return 300
