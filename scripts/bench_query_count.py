"""端到端 /normalize 请求耗时:扫描不同 N (查询数) × V (词库规模) × 设备。

回答:每请求的 N(查询数)对耗时的影响有多大?和 V 词库规模比呢?
"""
import time
import numpy as np
import torch

from app.embedding.m3e_impl import M3eEmbedding
from app.embedding.bge_impl import BgeEmbedding
from app.normalizer import Normalizer
from app.vocabulary import Vocabulary


def make_vocab(vocab_size: int) -> Vocabulary:
    """构造一个 vocab_size 词的假词库 (正面/负面各一半)。"""
    rows = ["大类,词,极性"]
    half = vocab_size // 2
    for i in range(half):
        rows.append(f"正面类,正面词{i},正面")
    for i in range(half):
        rows.append(f"负面类,负面词{i},负面")
    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.write(fd, "\n".join(rows).encode("utf-8"))
    os.close(fd)
    return Vocabulary.load(path)


def run(emb, vocab_obj: Vocabulary, n_queries: int, n_pos: int, n_neg: int, device_label: str):
    norm = Normalizer(emb, vocab_obj)
    # 用"贼XX"/"挺XX"/口语化前缀, 强制走 L2 embedding 路径(避开 L1 字典查表)
    pos_base = vocab_obj.buckets["正面"]
    neg_base = vocab_obj.buckets["负面"]
    pos_queries = [f"贼{pos_base[i % len(pos_base)][-2:]}" for i in range(n_pos)]
    neg_queries = [f"挺{neg_base[i % len(neg_base)][-2:]}" for i in range(n_neg)]
    noise = [f"未登录噪声abc{i}" for i in range(n_queries - n_pos - n_neg)]
    queries = (pos_queries + neg_queries + noise)[:n_queries]
    hints = (["正面"] * n_pos) + (["负面"] * n_neg) + (["正面"] * (n_queries - n_pos - n_neg))
    hints = hints[:n_queries]

    t0 = time.time()
    results = norm.normalize_batch(queries, polarity_hints=hints)
    t = time.time() - t0
    layers = [r.matched_layer for r in results]
    return t, dict(L1=layers.count("L1"), L2=layers.count("L2"),
                   L3=layers.count("L3"), FALLBACK=layers.count("FALLBACK"))


def main():
    print(f"CPU threads = {torch.get_num_threads()}, GPU = {torch.cuda.get_device_name(0)}\n")

    # 先 warmup
    m3e = M3eEmbedding()
    m3e._model = None  # reset
    from sentence_transformers import SentenceTransformer
    from pathlib import Path
    Path(m3e._CACHE_DIR).mkdir(parents=True, exist_ok=True)

    # 准备 models
    bge_cpu = SentenceTransformer(BgeEmbedding._MODEL_NAME, device="cpu",
                                   cache_folder=BgeEmbedding._CACHE_DIR)
    bge_gpu = SentenceTransformer(BgeEmbedding._MODEL_NAME, device="cuda",
                                   cache_folder=BgeEmbedding._CACHE_DIR)
    m3e_cpu = SentenceTransformer(M3eEmbedding._MODEL_NAME, device="cpu",
                                   cache_folder=M3eEmbedding._CACHE_DIR)
    m3e_gpu = SentenceTransformer(M3eEmbedding._MODEL_NAME, device="cuda",
                                   cache_folder=M3eEmbedding._CACHE_DIR)

    def wrap(emb_inner, dim, name_label, dev_label):
        from app.embedding.base import EmbeddingModel
        class W(EmbeddingModel):
            def load(self): pass
            def encode(self, words):
                v = emb_inner.encode(words, normalize_embeddings=True,
                                     batch_size=64, show_progress_bar=False)
                return np.asarray(v, dtype=np.float32)
            @property
            def name(self): return name_label
            @property
            def dim(self): return dim
        return W()

    bge_cpu_w = wrap(bge_cpu, 512, "bge", "cpu")
    bge_gpu_w = wrap(bge_gpu, 512, "bge", "cuda")
    m3e_cpu_w = wrap(m3e_cpu, 768, "m3e", "cpu")
    m3e_gpu_w = wrap(m3e_gpu, 768, "m3e", "cuda")

    print(f"{'V(词库)':<10}{'N(查询)':<10}{'device':<8}{'耗时(秒)':<12}{'L1':<6}{'L2':<6}{'L3':<6}{'FB':<6}{'查询/秒':<10}")
    print("-" * 76)
    for V in [192, 1000, 5000]:
        vocab = make_vocab(V)
        for N in [50, 500, 5000, 20000]:
            for emb, dev in [(bge_cpu_w, "cpu"), (bge_gpu_w, "cuda"),
                             (m3e_cpu_w, "cpu"), (m3e_gpu_w, "cuda")]:
                # 跑 3 次取 P50
                times = []
                for _ in range(3):
                    t, layers = run(emb, vocab, N, N // 2, N // 2, dev)
                    times.append(t)
                times.sort()
                p50 = times[1]
                qps = N / p50
                print(f"{V:<10}{N:<10}{dev:<8}{p50:<12.3f}"
                      f"{layers['L1']:<6}{layers['L2']:<6}{layers['L3']:<6}{layers['FALLBACK']:<6}"
                      f"{qps:<10.0f}")


if __name__ == "__main__":
    main()
