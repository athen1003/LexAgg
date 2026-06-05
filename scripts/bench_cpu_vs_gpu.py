"""M3E / BGE-small / BGE-base 三模型 × CPU vs GPU 实测。

目标:回答"M3E 不上 GPU 行不行"和"BGE-small 要不要上 GPU"。
"""
import time
import torch

from app.embedding.m3e_impl import M3eEmbedding
from app.embedding.bge_impl import BgeEmbedding, BgeBaseEmbedding


def make_model(cls, name):
    emb = cls()
    emb._name_label = name
    return emb


def bench_one(emb, vocab_words, query_words, device: str):
    """跑指定 device,返回 (load_s, precompute_ms, batch_p50_dict)。"""
    # 先按 device load
    emb._device = "unknown"
    emb._model = None
    # load() 内部会自动选 device,这里我们 hack 一下强制 device
    import os
    from pathlib import Path

    import torch
    from sentence_transformers import SentenceTransformer

    cache_dir = emb._CACHE_DIR
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    emb._model = SentenceTransformer(
        emb.model_name, device=device, cache_folder=cache_dir
    )
    emb._device = device
    t_load = time.time() - t0

    t0 = time.time()
    emb.encode(vocab_words)
    t_pre = time.time() - t0

    batches = {}
    for n in [1, 10, 50, 100]:
        batch = (query_words * (n // len(query_words) + 1))[:n]
        times = []
        for _ in range(5):
            t0 = time.time()
            emb.encode(batch)
            times.append(time.time() - t0)
        times.sort()
        batches[n] = times[len(times) // 2] * 1000  # ms

    return {
        "load_s": t_load,
        "pre_ms": t_pre * 1000,
        "batches": batches,
    }


def main():
    vocab_words = [f"词{i}" for i in range(192)]
    query_words = [
        "舒适", "贼舒服", "挺柔软", "触感好", "瑕疵", "破洞",
        "缩水", "起球", "紧绷", "凉感适宜", "保暖", "透气",
    ]

    print(f"torch threads (CPU) = {torch.get_num_threads()}")
    print(f"GPU = {torch.cuda.get_device_name(0)}\n")

    rows = []
    for cls, label in [
        (BgeEmbedding, "BGE-small (24M)"),
        (BgeBaseEmbedding, "BGE-base (100M)"),
        (M3eEmbedding, "M3E-base (100M)"),
    ]:
        emb = make_model(cls, label)
        for device in ["cpu", "cuda"]:
            try:
                r = bench_one(emb, vocab_words, query_words, device)
                rows.append((label, device, r))
            except Exception as e:
                print(f"  [{label}/{device}] FAILED: {e}")
                continue
            print(f"[{label}/{device}]")
            print(f"  load={r['load_s']:.1f}s  precompute_192={r['pre_ms']:.0f}ms")
            print(f"  batch=1: {r['batches'][1]:.0f}ms   "
                  f"batch=10: {r['batches'][10]:.0f}ms   "
                  f"batch=50: {r['batches'][50]:.0f}ms   "
                  f"batch=100: {r['batches'][100]:.0f}ms")

    # 汇总
    print("\n" + "=" * 80)
    print(f"{'model':<20} {'device':<8} {'pre_192ms':>10} {'b50 CPU':>10} {'b50 GPU':>10} {'speedup':>10}")
    print("-" * 80)
    by_label = {}
    for label, device, r in rows:
        by_label.setdefault(label, {})[device] = r
    for label, devs in by_label.items():
        if "cpu" in devs and "gpu" in devs:
            speedup = devs["cpu"]["batches"][50] / max(devs["gpu"]["batches"][50], 0.1)
            print(f"{label:<20} {'cpu':<8} {devs['cpu']['pre_ms']:>10.0f} "
                  f"{devs['cpu']['batches'][50]:>10.0f} "
                  f"{'-':>10} {'-':>10}")
            print(f"{'':<20} {'cuda':<8} {devs['gpu']['pre_ms']:>10.0f} "
                  f"{'-':>10} {devs['gpu']['batches'][50]:>10.0f} {speedup:>9.1f}x")


if __name__ == "__main__":
    main()
