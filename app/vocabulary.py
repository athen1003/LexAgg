import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


class VocabularyLoadError(Exception):
    pass


_BRACKET_RE = re.compile(r"（含(.+?)）")


def _expand_brackets(word: str) -> list[str]:
    """从「轻薄（含轻盈、不压身）」中提取 ['轻盈', '不压身']。"""
    m = _BRACKET_RE.search(word)
    if not m:
        return []
    return [v.strip() for v in m.group(1).split("、") if v.strip()]


def _strip_brackets(word: str) -> str:
    """去掉括号说明，只保留主词。"""
    return _BRACKET_RE.sub("", word).strip()


@dataclass
class Vocabulary:
    buckets: dict[str, list[str]] = field(default_factory=dict)
    polarity_map: dict[str, str] = field(default_factory=dict)
    alias_map: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, csv_path: str) -> "Vocabulary":
        path = Path(csv_path)
        if not path.exists():
            raise VocabularyLoadError(f"词库文件不存在: {csv_path}")

        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        if len(df.columns) != 2:
            raise VocabularyLoadError(
                f"列数错误: 期望 2 列，实际 {len(df.columns)}"
            )

        buckets: dict[str, list[str]] = {"正面": [], "负面": []}
        polarity_map: dict[str, str] = {}
        alias_map: dict[str, str] = {}

        for idx, row in df.iterrows():
            word_raw = str(row.iloc[0]).strip()
            polarity = str(row.iloc[1]).strip()

            if not word_raw:
                raise VocabularyLoadError(f"第 {idx + 2} 行: 词为空")

            if polarity not in {"正面", "负面"}:
                raise VocabularyLoadError(
                    f"第 {idx + 2} 行: 极性 '{polarity}' 不合法，应为 '正面' 或 '负面'"
                )

            main_word = _strip_brackets(word_raw)
            variants = _expand_brackets(word_raw)

            if main_word in polarity_map:
                raise VocabularyLoadError(
                    f"第 {idx + 2} 行: 词 '{main_word}' 重复（已标记为 {polarity_map[main_word]}）"
                )

            # 主词入桶
            buckets[polarity].append(main_word)
            polarity_map[main_word] = polarity

            # 括号变体入桶 + alias_map
            for v in variants:
                if v in polarity_map:
                    raise VocabularyLoadError(
                        f"第 {idx + 2} 行: 括号变体 '{v}' 重复"
                    )
                buckets[polarity].append(v)
                polarity_map[v] = polarity
                alias_map[v] = main_word

        if not buckets["正面"] and not buckets["负面"]:
            raise VocabularyLoadError("词库为空")

        return cls(buckets=buckets, polarity_map=polarity_map, alias_map=alias_map)

    def get_bucket(self, polarity: str) -> list[str]:
        return self.buckets.get(polarity, [])

    def reload(self, csv_path: str) -> None:
        new = Vocabulary.load(csv_path)
        self.buckets = new.buckets
        self.polarity_map = new.polarity_map
        self.alias_map = new.alias_map
