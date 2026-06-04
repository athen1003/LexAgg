from abc import ABC, abstractmethod

import numpy as np


class EmbeddingModel(ABC):
    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def encode(self, words: list[str]) -> np.ndarray:
        """返回 shape=(len(words), dim) 的 float32 向量。"""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def dim(self) -> int: ...
