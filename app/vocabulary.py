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
    category_map: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, csv_path: str) -> "Vocabulary":
        path = Path(csv_path)
        if not path.exists():
            raise VocabularyLoadError(f"词库文件不存在: {csv_path}")

        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        if len(df.columns) != 3:
            raise VocabularyLoadError(
                f"列数错误: 期望 3 列（大类、词、极性），实际 {len(df.columns)}"
            )

        buckets: dict[str, list[str]] = {"正面": [], "负面": []}
        polarity_map: dict[str, str] = {}
        alias_map: dict[str, str] = {}
        category_map: dict[str, str] = {}

        for idx, row in df.iterrows():
            category = str(row.iloc[0]).strip()
            word_raw = str(row.iloc[1]).strip()
            polarity = str(row.iloc[2]).strip()

            if not category:
                raise VocabularyLoadError(f"第 {idx + 2} 行: 大类为空")
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
            category_map[main_word] = category

            # 括号变体入桶 + alias_map
            for v in variants:
                if v in polarity_map:
                    raise VocabularyLoadError(
                        f"第 {idx + 2} 行: 括号变体 '{v}' 重复"
                    )
                buckets[polarity].append(v)
                polarity_map[v] = polarity
                category_map[v] = category
                alias_map[v] = main_word

        if not buckets["正面"] and not buckets["负面"]:
            raise VocabularyLoadError("词库为空")

        return cls(
            buckets=buckets,
            polarity_map=polarity_map,
            alias_map=alias_map,
            category_map=category_map,
        )

    def get_bucket(self, polarity: str) -> list[str]:
        return self.buckets.get(polarity, [])

    def get_words_by_category(self, category: str) -> list[str]:
        """返回指定大类下的所有词（不分极性）。"""
        return [w for w, c in self.category_map.items() if c == category]

    @classmethod
    def from_json(cls, items: list[dict]) -> "Vocabulary":
        """从 JSON 反序列化。items 为 [{word, polarity, category?}]。"""
        buckets: dict[str, list[str]] = {"正面": [], "负面": []}
        polarity_map: dict[str, str] = {}
        category_map: dict[str, str] = {}

        for i, item in enumerate(items):
            word = str(item.get("word", "")).strip()
            polarity = str(item.get("polarity", "")).strip()
            category = str(item.get("category", "")).strip()

            if not word:
                raise VocabularyLoadError(f"vocab[{i}]: word 为空")
            if polarity not in {"正面", "负面"}:
                raise VocabularyLoadError(
                    f"vocab[{i}]: polarity '{polarity}' 不合法,应填 '正面' 或 '负面'"
                )
            if word in polarity_map:
                raise VocabularyLoadError(
                    f"vocab[{i}]: 词 '{word}' 重复(已标记为 {polarity_map[word]})"
                )

            buckets[polarity].append(word)
            polarity_map[word] = polarity
            category_map[word] = category

        if not buckets["正面"] and not buckets["负面"]:
            raise VocabularyLoadError("vocab 为空: 至少需要 1 个正面或负面词")

        return cls(
            buckets=buckets,
            polarity_map=polarity_map,
            category_map=category_map,
        )

    @classmethod
    def load_from_rows(
        cls,
        rows: list[tuple[str, str]],
        alias_map: dict[str, str] | None = None,
        categories: dict[str, str] | None = None,
    ) -> "Vocabulary":
        """测试用：直接构造，无需 CSV 文件。categories 可选，未提供的词归入空大类。"""
        buckets: dict[str, list[str]] = {"正面": [], "负面": []}
        polarity_map: dict[str, str] = {}
        amap = dict(alias_map or {})
        cmap: dict[str, str] = dict(categories or {})

        for word, polarity in rows:
            if polarity not in {"正面", "负面"}:
                raise VocabularyLoadError(f"非法极性: {polarity}")
            if word in polarity_map:
                raise VocabularyLoadError(f"重复: {word}")
            buckets[polarity].append(word)
            polarity_map[word] = polarity
            # 类别未提供时记为空串
            cmap.setdefault(word, "")

        # 别名词也入桶
        for variant, std in amap.items():
            if variant in polarity_map:
                continue
            std_polarity = polarity_map.get(std)
            if std_polarity:
                buckets[std_polarity].append(variant)
                polarity_map[variant] = std_polarity
            # 别名词继承标准词的类别（未指定时为空）
            if variant not in cmap:
                cmap[variant] = cmap.get(std, "")

        return cls(
            buckets=buckets,
            polarity_map=polarity_map,
            alias_map=amap,
            category_map=cmap,
        )

    def reload(self, csv_path: str) -> None:
        new = Vocabulary.load(csv_path)
        self.buckets = new.buckets
        self.polarity_map = new.polarity_map
        self.alias_map = new.alias_map
        self.category_map = new.category_map
