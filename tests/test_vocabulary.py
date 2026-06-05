import pytest

from app.vocabulary import Vocabulary, VocabularyLoadError


def test_load_valid_csv_builds_buckets(sample_vocab_csv):
    vocab = Vocabulary.load(sample_vocab_csv)
    assert "正面" in vocab.buckets
    assert "负面" in vocab.buckets
    assert "舒适" in vocab.buckets["正面"]
    assert "不舒适" in vocab.buckets["负面"]


def test_load_valid_csv_builds_polarity_map(sample_vocab_csv):
    vocab = Vocabulary.load(sample_vocab_csv)
    assert vocab.polarity_map["舒适"] == "正面"
    assert vocab.polarity_map["不舒适"] == "负面"


def test_load_valid_csv_expands_brackets(sample_vocab_csv):
    vocab = Vocabulary.load(sample_vocab_csv)
    # 轻盈和不压身应作为正面桶的独立词
    assert "轻盈" in vocab.buckets["正面"]
    assert "不压身" in vocab.buckets["正面"]
    # 同时写入 alias_map
    assert vocab.alias_map["轻盈"] == "轻薄"
    assert vocab.alias_map["不压身"] == "轻薄"
    assert vocab.alias_map["破洞"] == "瑕疵"
    assert vocab.alias_map["勾丝"] == "瑕疵"
    assert vocab.alias_map["脏"] == "瑕疵"


def test_load_valid_csv_keeps_main_word(sample_vocab_csv):
    vocab = Vocabulary.load(sample_vocab_csv)
    assert "轻薄" in vocab.buckets["正面"]
    assert "瑕疵" in vocab.buckets["负面"]


def test_load_empty_csv_raises(empty_vocab_csv):
    with pytest.raises(VocabularyLoadError):
        Vocabulary.load(empty_vocab_csv)


def test_load_invalid_columns_raises(invalid_column_csv):
    with pytest.raises(VocabularyLoadError, match="列数"):
        Vocabulary.load(invalid_column_csv)


def test_load_invalid_polarity_raises(invalid_polarity_csv):
    with pytest.raises(VocabularyLoadError, match="极性"):
        Vocabulary.load(invalid_polarity_csv)


def test_load_duplicate_word_raises(duplicate_word_csv):
    with pytest.raises(VocabularyLoadError, match="重复"):
        Vocabulary.load(duplicate_word_csv)


def test_polarity_map_query_unknown_word(sample_vocab_csv):
    vocab = Vocabulary.load(sample_vocab_csv)
    assert vocab.polarity_map.get("不存在的词") is None


def test_get_bucket(sample_vocab_csv):
    vocab = Vocabulary.load(sample_vocab_csv)
    pos = vocab.get_bucket("正面")
    assert "舒适" in pos
    neg = vocab.get_bucket("负面")
    assert "不舒适" in neg


# ==================== 大类 / category_map 测试 ====================


def test_load_populates_category_map(sample_vocab_csv):
    vocab = Vocabulary.load(sample_vocab_csv)
    assert vocab.category_map["舒适"] == "体感"
    assert vocab.category_map["不舒适"] == "体感"
    assert vocab.category_map["轻盈"] == "清洁打理"  # 括号变体继承主词的大类
    assert vocab.category_map["破洞"] == "质量"


def test_get_words_by_category(sample_vocab_csv):
    vocab = Vocabulary.load(sample_vocab_csv)
    体感 = vocab.get_words_by_category("体感")
    assert "舒适" in 体感
    assert "不舒适" in 体感
    assert "轻盈" not in 体感  # 在 清洁打理 不在 体感
    质量 = vocab.get_words_by_category("质量")
    assert "瑕疵" in 质量
    assert "破洞" in 质量  # 括号变体
    # 不存在的大类返回空列表
    assert vocab.get_words_by_category("不存在的大类") == []


def test_load_from_rows_with_categories():
    vocab = Vocabulary.load_from_rows(
        [("舒适", "正面"), ("不舒适", "负面")],
        categories={"舒适": "体感", "不舒适": "体感"},
    )
    assert vocab.category_map["舒适"] == "体感"
    assert vocab.get_words_by_category("体感") == ["舒适", "不舒适"]


def test_load_from_rows_without_categories_gives_empty_category():
    vocab = Vocabulary.load_from_rows([("舒适", "正面")])
    assert vocab.category_map["舒适"] == ""
    assert vocab.get_words_by_category("") == ["舒适"]
    assert vocab.get_words_by_category("任何大类") == []


def test_load_rejects_empty_category(tmp_path):
    p = tmp_path / "no_cat.csv"
    p.write_text("大类,词,极性\n,舒适,正面\n", encoding="utf-8")
    with pytest.raises(VocabularyLoadError, match="大类为空"):
        Vocabulary.load(str(p))


def test_load_rejects_wrong_column_count(tmp_path):
    p = tmp_path / "wrong_cols.csv"
    p.write_text("词,极性\n舒适,正面\n", encoding="utf-8")
    with pytest.raises(VocabularyLoadError, match="列数"):
        Vocabulary.load(str(p))
