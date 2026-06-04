"""FastTextEmbedding via gensim (Python 3.13 兼容).

注：原计划用 fasttext-wheel，Task 1 fix 改用 gensim 加载 Facebook fastText .bin 文件。
gensim 不支持 subword OOV，对未登录词采用拆字符平均回退。
"""
from pathlib import Path

import numpy as np

from app.embedding.base import EmbeddingModel


class ModelFileMissingError(Exception):
    pass


class FastTextEmbedding(EmbeddingModel):
    def __init__(self, model_path: str = "models/cc.zh.300.bin"):
        self.model_path = model_path
        self._model = None  # gensim KeyedVectors

    def load(self) -> None:
        path = Path(self.model_path)
        if not path.exists():
            raise ModelFileMissingError(
                f"fastText 模型不存在: {self.model_path}，"
                f"请运行: python scripts/download_model.py"
            )
        from gensim.models import KeyedVectors

        self._model = KeyedVectors.load_word2vec_format(str(path), binary=True)

    def _encode_one(self, word: str) -> np.ndarray:
        """单词编码：先查表，OOV 拆字符平均。"""
        try:
            return self._model.get_vector(word, norm=True).astype(np.float32)
        except KeyError:
            pass

        # OOV 拆字符回退
        chars = [c for c in word if c.strip()]
        char_vecs = []
        for c in chars:
            try:
                char_vecs.append(self._model.get_vector(c, norm=True))
            except KeyError:
                continue
        if char_vecs:
            mean = np.mean(char_vecs, axis=0)
            # 重新归一化
            norm = np.linalg.norm(mean)
            if norm > 0:
                mean = mean / norm
            return mean.astype(np.float32)
        # 完全没找到：返回零向量
        return np.zeros(self.dim, dtype=np.float32)

    def encode(self, words: list[str]) -> np.ndarray:
        if self._model is None:
            # 模型未加载（测试占位）：返回零向量
            return np.zeros((len(words), self.dim), dtype=np.float32)
        return np.array(
            [self._encode_one(w) for w in words],
            dtype=np.float32,
        )

    @property
    def name(self) -> str:
        return "fasttext"

    @property
    def dim(self) -> int:
        return 300
