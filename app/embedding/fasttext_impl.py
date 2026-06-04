"""FastTextEmbedding via gensim (Python 3.13 兼容).

Facebook fastText .bin 是私有二进制格式（不是 word2vec binary），
必须用 gensim.models.fasttext.load_facebook_vectors 加载。
返回的 FastTextKeyedVectors 原生支持 subword OOV：
get_vector(oov_word) 通过 char n-gram 合成向量。
"""
from pathlib import Path

import numpy as np

from app.embedding.base import EmbeddingModel


class ModelFileMissingError(Exception):
    pass


class FastTextEmbedding(EmbeddingModel):
    def __init__(self, model_path: str = "models/cc.zh.300.bin"):
        self.model_path = model_path
        self._model = None  # gensim FastTextKeyedVectors

    def load(self) -> None:
        path = Path(self.model_path)
        if not path.exists():
            raise ModelFileMissingError(
                f"fastText 模型不存在: {self.model_path}，"
                f"请运行: python scripts/download_model.py"
            )
        from gensim.models.fasttext import load_facebook_vectors

        self._model = load_facebook_vectors(str(path))

    def _encode_one(self, word: str) -> np.ndarray:
        """单词编码：FastTextKeyedVectors.get_vector 对 OOV 走 subword 合成。"""
        try:
            return self._model.get_vector(word, norm=True).astype(np.float32)
        except KeyError:
            # subword 也无法合成（空词或全部字符未见过）
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
