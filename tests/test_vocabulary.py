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
