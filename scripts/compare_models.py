"""模型对比脚本:同一批中文"近义/相似/疑似错词"测试对,3 个模型各跑一遍,比余弦分布。

测试对设计原则:
- 体感正面:同义/口语化表达
- 体感负面:同义/口语化表达
- 跨类干扰:已知 BGE 容易误判的 pair(同字符不同极性)
- 错字:一两个字不同的 typo

每条 (input, expected) 期望模型给出 input 与 expected 的余弦高(理想 ≥ 0.7)。
"""
import time
import numpy as np
from app.embedding.factory import get_model
from app.vocabulary import Vocabulary

# (input, expected_vocab_word, polarity, note)
TEST_PAIRS = [
    # 体感-正面 同义/口语化
    ("贼舒服", "舒适", "正面", "口语化"),
    ("凉凉", "凉感适宜", "正面", "叠词"),
    ("凉快的", "凉感适宜", "正面", "加后缀"),
    ("挺柔软", "柔软", "正面", "加前缀"),
    ("丝滑得很", "丝滑", "正面", "加后缀"),
    ("贼透气", "透气", "正面", "口语化"),
    ("暖暖的", "保暖", "正面", "近义"),
    ("毛绒", "蓬松", "正面", "近义"),

    # 体感-负面 同义/口语化
    ("硌得慌", "硌", "负面", "口语化"),
    ("紧巴巴", "紧绷", "负面", "叠词"),
    ("磨皮肤", "磨", "负面", "近义"),
    ("起球严重", "起球", "负面", "加后缀"),
    ("缩水了", "缩水", "负面", "加后缀"),

    # 跨类干扰 (BGE 容易翻车的)
    ("手感不错", "触感好", "正面", "正面+触感"),
    ("手感贼好", "触感好", "正面", "正面+触感+口语"),
    ("手感硬", "手感硬", "负面", "L1 直接命中"),
    ("贼硬", "手感硬", "负面", "口语化"),

    # 错字 / 模糊匹配
    ("舒式", "舒适", "正面", "形近错字"),
    ("丝滑滑", "丝滑", "正面", "叠字"),
    ("轻来轻去", "轻盈", "正面", "语音变体"),
    ("破了个洞", "瑕疵", "负面", "口语化"),

    # 一些完全随机的噪音 (期望低余弦,验证模型能区分)
    ("随便一个词xyz", "舒适", "正面", "噪声"),
    ("不沾边", "舒适", "正面", "噪声"),
]

THRESHOLD_L2 = 0.7  # BGE 接受阈值
THRESHOLD_L3 = 0.5  # BGE 降级阈值


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def run_model(model_name: str, vocab: Vocabulary) -> list[tuple[str, float]]:
    """对每个测试对,返回 (input, cosine_vs_expected)。"""
    emb = get_model(model_name)
    print(f"\n[{model_name}] dim={emb.dim} device={getattr(emb, '_device', 'unknown')}")

    # 编码所有 expected + 所有 input,批量做
    inputs = [p[0] for p in TEST_PAIRS]
    expecteds = [p[1] for p in TEST_PAIRS]
    all_words = list(set(inputs + expecteds))
    t0 = time.time()
    vecs = emb.encode(all_words)
    encode_t = time.time() - t0
    vec_map = {w: vecs[i] for i, w in enumerate(all_words)}
    print(f"  encoded {len(all_words)} words in {encode_t:.2f}s")

    results = []
    for inp, exp in zip(inputs, expecteds):
        c = cosine(vec_map[inp], vec_map[exp])
        results.append((inp, c))
    return results


def main():
    vocab = Vocabulary.load("data/vocabulary.csv")
    print(f"vocab size: {len(vocab.buckets['正面']) + len(vocab.buckets['负面'])}")

    all_results = {}
    for name in ["bge", "bge_base", "m3e"]:
        try:
            all_results[name] = run_model(name, vocab)
        except Exception as e:
            print(f"[{name}] FAILED: {e}")
            all_results[name] = [(p[0], 0.0) for p in TEST_PAIRS]

    # 打印对照表
    print("\n" + "=" * 80)
    print(f"{'input':<12} {'expected':<10} {'note':<10} {'BGE-small':>10} {'BGE-base':>10} {'M3E-base':>10}")
    print("-" * 80)
    for i, (inp, exp, pol, note) in enumerate(TEST_PAIRS):
        row = f"{inp:<12} {exp:<10} {note:<10}"
        for name in ["bge", "bge_base", "m3e"]:
            r = all_results[name][i]
            row += f" {r[1]:>10.4f}"
        print(row)

    # 汇总
    print("\n" + "=" * 60)
    print(f"{'metric':<28} {'BGE-small':>10} {'BGE-base':>10} {'M3E-base':>10}")
    print("-" * 60)
    for metric_name, threshold_op in [
        (f"% >= {THRESHOLD_L2} (L2 命中)", lambda x: x >= THRESHOLD_L2),
        (f"% >= {THRESHOLD_L3} (L3 区间)", lambda x: x >= THRESHOLD_L3),
        (f"% <  {THRESHOLD_L3} (FALLBACK)", lambda x: x < THRESHOLD_L3),
        ("平均余弦", None),
    ]:
        row = f"{metric_name:<28}"
        for name in ["bge", "bge_base", "m3e"]:
            vals = [r[1] for r in all_results[name]]
            if threshold_op:
                pct = sum(1 for v in vals if threshold_op(v)) / len(vals)
                row += f" {pct * 100:>9.1f}%"
            else:
                row += f" {np.mean(vals):>10.4f}"
        print(row)


if __name__ == "__main__":
    main()
