import time
from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.similarity import cosine_batch
from app.vocabulary import Vocabulary


@dataclass
class NormalizeResult:
    original: str
    normalized: str
    matched_layer: Literal["L1", "L2", "L3", "FALLBACK"]
    score: float
    elapsed_ms: float


class Normalizer:
    # 注：以下阈值为初始值，实施阶段用样本调优后写入实际值
    THRESHOLDS = {
        "fasttext": {"accept": 0.6, "fallback_to_edit": 0.4},
        "bge": {"accept": 0.7, "fallback_to_edit": 0.5},
    }
    EDIT_DISTANCE_RATIO = 0.3  # 编辑距离 / max(len(word), len(candidate))

    def __init__(self, embedding, vocab: Vocabulary):
        self.embedding = embedding
        self.vocab = vocab
        # 预计算标准词向量（按极性分桶）
        self._precomputed: dict[str, np.ndarray] = {}
        self._precompute_vectors()

    def _precompute_vectors(self) -> None:
        for polarity, words in self.vocab.buckets.items():
            if words:
                self._precomputed[polarity] = self.embedding.encode(words)
            else:
                self._precomputed[polarity] = np.zeros(
                    (0, self.embedding.dim), dtype=np.float32
                )

    def normalize(self, word: str) -> NormalizeResult:
        t0 = time.time()
        result = self._normalize_inner(word)
        result.elapsed_ms = (time.time() - t0) * 1000
        return result

    def _normalize_inner(self, word: str) -> NormalizeResult:
        polarity = self._infer_polarity(word)

        if polarity is not None:
            return self._match_in_bucket(word, polarity)

        # 极性未知 → 双桶对比取高
        r_pos = self._match_in_bucket(word, "正面")
        r_neg = self._match_in_bucket(word, "负面")
        if r_pos.matched_layer != "FALLBACK" and (
            r_neg.matched_layer == "FALLBACK" or r_pos.score >= r_neg.score
        ):
            return r_pos
        if r_neg.matched_layer != "FALLBACK":
            return r_neg
        # 两桶都 FALLBACK → 返回原词
        return NormalizeResult(
            original=word, normalized=word, matched_layer="FALLBACK", score=0.0, elapsed_ms=0.0
        )

    def _infer_polarity(self, word: str) -> str | None:
        # 优先查 alias_map（变体词的极性）
        if word in self.vocab.alias_map:
            std = self.vocab.alias_map[word]
            return self.vocab.polarity_map.get(std)
        # 再查 polarity_map（输入词就是标准词）
        return self.vocab.polarity_map.get(word)

    def _match_in_bucket(self, word: str, polarity: str) -> NormalizeResult:
        candidates = self.vocab.get_bucket(polarity)
        if not candidates:
            return NormalizeResult(
                original=word, normalized=word, matched_layer="FALLBACK", score=0.0, elapsed_ms=0.0
            )

        # L1 别名命中（仅当别名映射的标准词同极性时）
        if word in self.vocab.alias_map:
            std = self.vocab.alias_map[word]
            if self.vocab.polarity_map.get(std) == polarity:
                return NormalizeResult(
                    original=word, normalized=std, matched_layer="L1", score=1.0, elapsed_ms=0.0
                )
        # L1 精确匹配词库中已有词
        if word in self.vocab.polarity_map:
            std_polarity = self.vocab.polarity_map[word]
            if std_polarity == polarity:
                return NormalizeResult(
                    original=word, normalized=word, matched_layer="L1", score=1.0, elapsed_ms=0.0
                )

        # L2 向量相似度
        word_vec = self.embedding.encode([word])[0]
        cand_vecs = self._precomputed[polarity]
        sims = cosine_batch(word_vec, cand_vecs)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        best_candidate = candidates[best_idx]

        thresholds = self.THRESHOLDS.get(self.embedding.name, self.THRESHOLDS["fasttext"])
        if best_sim >= thresholds["accept"]:
            return NormalizeResult(
                original=word, normalized=best_candidate, matched_layer="L2", score=best_sim, elapsed_ms=0.0
            )

        # L3 编辑距离（fallback）
        if best_sim >= thresholds["fallback_to_edit"]:
            from app.similarity import levenshtein_ratio

            ratios = [levenshtein_ratio(word, c) for c in candidates]
            best_idx = int(np.argmin(ratios))
            best_ratio = ratios[best_idx]
            if best_ratio <= self.EDIT_DISTANCE_RATIO:
                return NormalizeResult(
                    original=word,
                    normalized=candidates[best_idx],
                    matched_layer="L3",
                    score=best_sim,
                    elapsed_ms=0.0,
                )

        return NormalizeResult(
            original=word, normalized=word, matched_layer="FALLBACK", score=best_sim, elapsed_ms=0.0
        )
