"""Task 11 冒烟测试用：注册 stub embedding 后启动 uvicorn。

实际生产环境需先运行 `python scripts/download_model.py` 获取 fastText 模型。
本脚本仅用于无模型环境下验证路由通畅，不应在生产使用。

用法：python scripts/smoke_test_server.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.embedding import factory
from app.embedding.base import EmbeddingModel


class _SmokeStub(EmbeddingModel):
    _name = "fasttext"
    _dim = 4

    def load(self) -> None:
        pass

    def encode(self, words):
        return np.ones((len(words), self._dim), dtype=np.float32)

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return self._dim


# 抢在 _register_defaults 之前注册，让 get_model("fasttext") 返回 stub
factory._REGISTRY["fasttext"] = _SmokeStub
factory._registered = True  # 阻止默认注册覆盖

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, log_level="info")
