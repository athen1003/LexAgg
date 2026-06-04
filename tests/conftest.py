import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def sample_vocab_csv():
    content = """词,极性
舒适,正面
触感好,正面
柔软,正面
凉感适宜,正面
凉感太弱或太凉,负面
不舒适,负面
触感差,负面
轻薄（含轻盈、不压身）,正面
瑕疵（含破洞、勾丝、脏）,负面
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def empty_vocab_csv():
    content = "词,极性\n"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def invalid_column_csv(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("only_one_col\nvalue\n", encoding="utf-8")
    return str(p)


@pytest.fixture
def invalid_polarity_csv(tmp_path):
    p = tmp_path / "bad_polarity.csv"
    p.write_text("词,极性\n舒适,正向\n", encoding="utf-8")
    return str(p)


@pytest.fixture
def duplicate_word_csv(tmp_path):
    p = tmp_path / "dup.csv"
    p.write_text("词,极性\n舒适,正面\n舒适,负面\n", encoding="utf-8")
    return str(p)
