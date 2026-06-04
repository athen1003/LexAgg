import threading
from typing import Type

from app.embedding.base import EmbeddingModel


class ModelNotFoundError(Exception):
    pass


# 注册表：仅在导入对应实现后才注册
_REGISTRY: dict[str, Type[EmbeddingModel]] = {}

_models: dict[str, EmbeddingModel] = {}
_lock = threading.Lock()
_registered = False


def _register_defaults() -> None:
    global _registered
    if _registered:
        return
    # 延迟导入，避免启动时硬依赖
    from app.embedding.fasttext_impl import FastTextEmbedding
    from app.embedding.bge_impl import BgeEmbedding

    _REGISTRY["fasttext"] = FastTextEmbedding
    _REGISTRY["bge"] = BgeEmbedding
    _registered = True


def register(name: str, cls: Type[EmbeddingModel]) -> None:
    """注册自定义模型（用于测试）。"""
    _REGISTRY[name] = cls


def get_model(name: str) -> EmbeddingModel:
    _register_defaults()
    if name not in _models:
        with _lock:
            if name not in _models:
                if name not in _REGISTRY:
                    raise ModelNotFoundError(
                        f"未知模型: {name}，可选: {list(_REGISTRY.keys())}"
                    )
                instance = _REGISTRY[name]()
                instance.load()
                _models[name] = instance
    return _models[name]


def list_models() -> list[str]:
    _register_defaults()
    return list(_REGISTRY.keys())


def reset_models() -> None:
    """清空缓存（仅供测试）。"""
    global _registered
    with _lock:
        _models.clear()
        _REGISTRY.clear()
        _registered = False
