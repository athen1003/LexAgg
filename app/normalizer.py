import time
from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.similarity import cosine_batch, levenshtein_ratio
from app.vocabulary import Vocabulary


@dataclass
class NormalizeResult:
    original: str
    normalized: str
    matched_layer: Literal["L1", "L2", "L3", "FALLBACK"]
    score: float
    elapsed_ms: float
    matched_category: str = ""
    # 仅 FALLBACK 时填充：无视阈值下,词库里与本词最接近的标准词。供运营期决策。
    best_candidate: str = ""
    best_candidate_score: float = 0.0
    best_candidate_category: str = ""


class Normalizer:
    # 注：以下阈值为初始值，实施阶段用样本调优后写入实际值
    THRESHOLDS = {
        "fasttext": {"accept": 0.6, "fallback_to_edit": 0.4},
        "bge": {"accept": 0.7, "fallback_to_edit": 0.5},
        "bge_base": {"accept": 0.7, "fallback_to_edit": 0.5},
        "m3e": {"accept": 0.7, "fallback_to_edit": 0.5},
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

    def normalize(self, word: str, polarity_hint: str = "") -> NormalizeResult:
        t0 = time.time()
        if polarity_hint in {"正面", "负面"}:
            word_vec = self.embedding.encode([word])[0]
            result = self._match_in_bucket(word, polarity_hint, word_vec)
        elif polarity_hint == "":
            result = self._normalize_inner(word)
        else:
            # 极性提示非 正面/负面 且非空 → 不匹配,避免乱猜
            result = NormalizeResult(
                original=word,
                normalized=word,
                matched_layer="FALLBACK",
                score=0.0,
                elapsed_ms=0.0,
            )
        result.elapsed_ms = (time.time() - t0) * 1000
        return result

    def normalize_batch(
        self, words: list[str], polarity_hints: list[str]
    ) -> list[NormalizeResult]:
        """批量归一。polarity_hints[i] 对应 words[i]。

        极性提示 ∈ {正面, 负面} → 单桶;空 → dual-bucket;其他 → FALLBACK(但仍算建议)。
        内部把「需要 encode 的行」合并为一次 encode() 调用,避免逐行
        Python/CUDA launch overhead(单条调 GPU 慢 10-100×)。
        """
        n = len(words)
        assert n == len(polarity_hints), "words 与 polarity_hints 长度不一致"
        t0 = time.time()

        results: list[NormalizeResult | None] = [None] * n
        needs_vec_idx: list[int] = []  # 需要 encode 的行索引
        vec_polarity: list[str] = []   # 对应极性,或 "__DUAL__"/"__SUGGEST__"

        pos_bucket = self.vocab.get_bucket("正面")
        neg_bucket = self.vocab.get_bucket("负面")

        for i in range(n):
            word = words[i]
            pol = polarity_hints[i]

            # 极性提示非法 → 归一强制 FALLBACK,但仍计算建议(跨桶 best)
            if pol not in {"正面", "负面", ""}:
                if pos_bucket or neg_bucket:
                    needs_vec_idx.append(i)
                    vec_polarity.append("__SUGGEST__")
                else:
                    results[i] = self._fallback_result(word)
                continue

            # 极性为空 → 推断
            if pol == "":
                inferred = self._infer_polarity(word)
                pol = inferred if inferred is not None else "__DUAL__"

            # 单桶路径先做 L1 查表(零成本,免一次 encode)
            if pol in {"正面", "负面"}:
                l1 = self._l1_lookup(word, pol)
                if l1 is not None:
                    results[i] = l1
                    continue

            # 需要 encode:确认目标桶非空
            if pol == "__DUAL__":
                if not pos_bucket and not neg_bucket:
                    results[i] = self._fallback_result(word)
                    continue
            else:
                if not self.vocab.get_bucket(pol):
                    results[i] = self._fallback_result(word)
                    continue
            needs_vec_idx.append(i)
            vec_polarity.append(pol)

        # 一次性 batch encode 所有需要向量的输入
        if needs_vec_idx:
            words_to_encode = [words[i] for i in needs_vec_idx]
            vecs = self.embedding.encode(words_to_encode)
            for j, idx in enumerate(needs_vec_idx):
                word = words[idx]
                pol = vec_polarity[j]
                vec = vecs[j]
                if pol == "__SUGGEST__":
                    cand, score, cat = self._suggest_candidate(vec)
                    results[idx] = NormalizeResult(
                        original=word, normalized=word, matched_layer="FALLBACK",
                        score=0.0, elapsed_ms=0.0,
                        best_candidate=cand, best_candidate_score=score,
                        best_candidate_category=cat,
                    )
                elif pol == "__DUAL__":
                    r_pos = self._score_in_bucket(word, "正面", vec) if pos_bucket else None
                    r_neg = self._score_in_bucket(word, "负面", vec) if neg_bucket else None
                    results[idx] = self._pick_dual(word, r_pos, r_neg)
                else:
                    results[idx] = self._score_in_bucket(word, pol, vec)

        elapsed_ms = (time.time() - t0) * 1000
        for r in results:
            # elapsed_ms 是「整批」耗时,各条共享(单条字段意义不大)
            r.elapsed_ms = elapsed_ms
        return results  # type: ignore[return-value]

    @staticmethod
    def _fallback_result(word: str) -> NormalizeResult:
        return NormalizeResult(
            original=word, normalized=word, matched_layer="FALLBACK",
            score=0.0, elapsed_ms=0.0,
        )

    def _l1_lookup(self, word: str, polarity: str) -> NormalizeResult | None:
        """L1 查 alias_map / polarity_map,命中且极性匹配返回结果,否则 None。"""
        if word in self.vocab.alias_map:
            std = self.vocab.alias_map[word]
            if self.vocab.polarity_map.get(std) == polarity:
                return NormalizeResult(
                    original=word, normalized=std, matched_layer="L1",
                    score=1.0, elapsed_ms=0.0,
                    matched_category=self.vocab.category_map.get(std, ""),
                )
        if word in self.vocab.polarity_map:
            std_polarity = self.vocab.polarity_map[word]
            if std_polarity == polarity:
                return NormalizeResult(
                    original=word, normalized=word, matched_layer="L1",
                    score=1.0, elapsed_ms=0.0,
                    matched_category=self.vocab.category_map.get(word, ""),
                )
        return None

    def _score_in_bucket(
        self, word: str, polarity: str, word_vec: np.ndarray
    ) -> NormalizeResult:
        """给定预编码向量,跑 L2 → L3 → FALLBACK(不再做 L1 查表)。"""
        candidates = self.vocab.get_bucket(polarity)
        cand_vecs = self._precomputed[polarity]
        sims = cosine_batch(word_vec, cand_vecs)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        best_candidate = candidates[best_idx]

        thresholds = self.THRESHOLDS.get(self.embedding.name, self.THRESHOLDS["fasttext"])
        if best_sim >= thresholds["accept"]:
            return NormalizeResult(
                original=word, normalized=best_candidate, matched_layer="L2",
                score=best_sim, elapsed_ms=0.0,
                matched_category=self.vocab.category_map.get(best_candidate, ""),
            )

        if best_sim >= thresholds["fallback_to_edit"]:
            ratios = [levenshtein_ratio(word, c) for c in candidates]
            best_idx = int(np.argmin(ratios))
            best_ratio = ratios[best_idx]
            if best_ratio <= self.EDIT_DISTANCE_RATIO:
                return NormalizeResult(
                    original=word, normalized=candidates[best_idx],
                    matched_layer="L3", score=best_sim, elapsed_ms=0.0,
                    matched_category=self.vocab.category_map.get(candidates[best_idx], ""),
                )

        return NormalizeResult(
            original=word, normalized=word, matched_layer="FALLBACK",
            score=best_sim, elapsed_ms=0.0,
            best_candidate=best_candidate,
            best_candidate_score=best_sim,
            best_candidate_category=self.vocab.category_map.get(best_candidate, ""),
        )

    @staticmethod
    def _pick_dual(
        word: str,
        r_pos: NormalizeResult | None,
        r_neg: NormalizeResult | None,
    ) -> NormalizeResult:
        """双桶对比:优先非 FALLBACK,同分取正;两桶都 FALLBACK → 取分数较高者
        (仍保留 best_candidate 信息,供建议列使用)。"""
        def non_fallback(r):
            return r is not None and r.matched_layer != "FALLBACK"

        if non_fallback(r_pos) and (r_neg is None or r_pos.score >= r_neg.score):  # type: ignore[union-attr]
            return r_pos  # type: ignore[return-value]
        if non_fallback(r_neg) and (r_pos is None or r_neg.score > r_pos.score):  # type: ignore[union-attr]
            return r_neg  # type: ignore[return-value]
        # 两桶都 FALLBACK(或一桶 None):保留较高分的 best_candidate
        cands = [r for r in (r_pos, r_neg) if r is not None]
        if not cands:
            return Normalizer._fallback_result(word)
        return max(cands, key=lambda r: r.score)

    def _suggest_candidate(self, vec: np.ndarray) -> tuple[str, float, str]:
        """无视阈值、跨双桶 argmax,返回 (候选词, 余弦, 大类)。给建议列用。"""
        best: tuple[str, float, str] = ("", 0.0, "")
        pos_bucket = self.vocab.get_bucket("正面")
        if pos_bucket:
            sims = cosine_batch(vec, self._precomputed["正面"])
            idx = int(np.argmax(sims))
            s = float(sims[idx])
            if s > best[1]:
                cand = pos_bucket[idx]
                best = (cand, s, self.vocab.category_map.get(cand, ""))
        neg_bucket = self.vocab.get_bucket("负面")
        if neg_bucket:
            sims = cosine_batch(vec, self._precomputed["负面"])
            idx = int(np.argmax(sims))
            s = float(sims[idx])
            if s > best[1]:
                cand = neg_bucket[idx]
                best = (cand, s, self.vocab.category_map.get(cand, ""))
        return best

    def _normalize_inner(self, word: str) -> NormalizeResult:
        polarity = self._infer_polarity(word)

        # Always encode once up front. The L1 branches in _match_in_bucket
        # may return early (cheap), but for unknown-polarity / L1-miss cases
        # we want the same vector threaded through both bucket evaluations
        # to avoid encoding the same input word twice.
        word_vec = self.embedding.encode([word])[0]

        if polarity is not None:
            return self._match_in_bucket(word, polarity, word_vec)

        # 极性未知 → 双桶对比取高（共用同一 word_vec）
        r_pos = self._match_in_bucket(word, "正面", word_vec)
        r_neg = self._match_in_bucket(word, "负面", word_vec)
        if r_pos.matched_layer != "FALLBACK" and (
            r_neg.matched_layer == "FALLBACK" or r_pos.score >= r_neg.score
        ):
            return r_pos
        if r_neg.matched_layer != "FALLBACK":
            return r_neg
        # 两桶都 FALLBACK → 取分数较高者,保留 best_candidate
        best = r_pos if r_pos.score >= r_neg.score else r_neg
        return NormalizeResult(
            original=word, normalized=word, matched_layer="FALLBACK",
            score=0.0, elapsed_ms=0.0,
            best_candidate=best.best_candidate,
            best_candidate_score=best.best_candidate_score,
            best_candidate_category=best.best_candidate_category,
        )

    def _infer_polarity(self, word: str) -> str | None:
        # 优先查 alias_map（变体词的极性）
        if word in self.vocab.alias_map:
            std = self.vocab.alias_map[word]
            return self.vocab.polarity_map.get(std)
        # 再查 polarity_map（输入词就是标准词）
        return self.vocab.polarity_map.get(word)

    def _match_in_bucket(
        self, word: str, polarity: str, word_vec: np.ndarray
    ) -> NormalizeResult:
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
                    original=word, normalized=std, matched_layer="L1", score=1.0, elapsed_ms=0.0,
                    matched_category=self.vocab.category_map.get(std, ""),
                )
        # L1 精确匹配词库中已有词
        if word in self.vocab.polarity_map:
            std_polarity = self.vocab.polarity_map[word]
            if std_polarity == polarity:
                return NormalizeResult(
                    original=word, normalized=word, matched_layer="L1", score=1.0, elapsed_ms=0.0,
                    matched_category=self.vocab.category_map.get(word, ""),
                )

        # L2 向量相似度（word_vec 由调用方预编码，避免重复 encode）
        cand_vecs = self._precomputed[polarity]
        sims = cosine_batch(word_vec, cand_vecs)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        best_candidate = candidates[best_idx]

        thresholds = self.THRESHOLDS.get(self.embedding.name, self.THRESHOLDS["fasttext"])
        if best_sim >= thresholds["accept"]:
            return NormalizeResult(
                original=word, normalized=best_candidate, matched_layer="L2", score=best_sim, elapsed_ms=0.0,
                matched_category=self.vocab.category_map.get(best_candidate, ""),
            )

        # L3 编辑距离（fallback）
        if best_sim >= thresholds["fallback_to_edit"]:
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
                    matched_category=self.vocab.category_map.get(candidates[best_idx], ""),
                )

        return NormalizeResult(
            original=word, normalized=word, matched_layer="FALLBACK", score=best_sim, elapsed_ms=0.0,
            best_candidate=best_candidate, best_candidate_score=best_sim,
            best_candidate_category=self.vocab.category_map.get(best_candidate, ""),
        )