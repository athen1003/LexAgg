"""性能测试：3 万词应在 60 秒内完成。"""
import time
from pathlib import Path

from fastapi.testclient import TestClient


def main():
    from app.main import app, _state

    # 触发 startup 加载
    with TestClient(app) as client:
        client.get("/api/v1/health")

    vocab = _state["vocab"]
    normalizer = _state["default_normalizer"]
    if normalizer is None:
        raise RuntimeError("默认 normalizer 未加载")

    # 构造 3 万随机词
    import random
    pos = vocab.buckets["正面"]
    neg = vocab.buckets["负面"]
    samples = [random.choice(pos + neg) for _ in range(30000)]
    samples += [f"未登录词{i}" for i in range(1000)]

    t0 = time.time()
    for w in samples:
        normalizer.normalize(w)
    elapsed = time.time() - t0
    print(f"处理 {len(samples)} 词，耗时 {elapsed:.1f}s")
    assert elapsed < 60, f"超出 60s 预算"


if __name__ == "__main__":
    main()
