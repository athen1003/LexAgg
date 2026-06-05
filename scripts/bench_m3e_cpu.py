"""M3E CPU 推理速度实测。

模拟一个 /normalize 请求的工作量:把 ~200 词词库一次性 precompute,
再编码 50 个查询词。报告总耗时,推算单请求 P50。
"""
import time
import torch

# 强制 CPU(避免无意中用到 GPU,让对比真实)
torch.set_num_threads(max(1, torch.get_num_threads()))

from app.embedding.m3e_impl import M3eEmbedding
from app.embedding.bge_impl import BgeEmbedding


def bench(name: str, emb, vocab_words: list[str], query_words: list[str]):
    # 强制 CPU
    emb._device = "cpu"
    if emb._model is not None and hasattr(emb._model, "_modules"):
        emb._model = emb._model.to("cpu")

    print(f"\n[{name}] dim={emb.dim} device={emb._device}")
    t0 = time.time()
    emb.load()
    t_load = time.time() - t0
    print(f"  load:        {t_load:.2f}s")

    t0 = time.time()
    precomputed = emb.encode(vocab_words)
    t_pre = time.time() - t0
    print(f"  precompute {len(vocab_words):>4} words: {t_pre*1000:>6.0f}ms"
          f"  ({t_pre*1000/len(vocab_words):.1f}ms/word)")

    # 模拟不同批量
    for n in [1, 10, 50, 100]:
        batch = query_words[:n] if n <= len(query_words) else query_words
        # 多跑几次取 P50
        times = []
        for _ in range(5):
            t0 = time.time()
            emb.encode(batch)
            times.append(time.time() - t0)
        times.sort()
        p50 = times[len(times) // 2]
        print(f"  encode batch={n:>3}:  p50={p50*1000:>6.0f}ms"
              f"  ({p50*1000/n:.1f}ms/word)")


def main():
    # 真实词库规模
    vocab_words = [f"词{i}" for i in range(192)]
    # 真实查询:1-3 字中文词
    query_words = [
        "舒适", "贼舒服", "挺柔软", "触感好", "瑕疵", "破洞",
        "缩水", "起球", "紧绷", "凉感适宜", "保暖", "透气",
    ] * 4  # 48 个

    print(f"torch threads = {torch.get_num_threads()}")

    m3e = M3eEmbedding()
    bench("M3E-base", m3e, vocab_words, query_words)

    bge = BgeEmbedding()
    bench("BGE-small", bge, vocab_words, query_words)


if __name__ == "__main__":
    main()
