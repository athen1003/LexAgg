import numpy as np

from app.similarity import cosine_batch, cosine_single, levenshtein_ratio


def test_cosine_single_identical_is_one():
    v = np.array([1.0, 0.0, 0.0])
    assert abs(cosine_single(v, v) - 1.0) < 1e-6


def test_cosine_single_orthogonal_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_single(a, b)) < 1e-6


def test_cosine_batch_shape():
    query = np.array([1.0, 0.0, 0.0])
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]])
    sims = cosine_batch(query, matrix)
    assert sims.shape == (3,)
    assert abs(sims[0] - 1.0) < 1e-6
    assert abs(sims[1]) < 1e-6


def test_levenshtein_ratio_identical_is_zero():
    assert levenshtein_ratio("舒适", "舒适") == 0.0


def test_levenshtein_ratio_basic():
    # 1 替换 / max(2,2) = 1/2
    r = levenshtein_ratio("凉快", "凉感")
    assert abs(r - 1 / 2) < 1e-6


def test_levenshtein_ratio_empty():
    assert levenshtein_ratio("", "abc") == 1.0
    assert levenshtein_ratio("abc", "") == 1.0
    assert levenshtein_ratio("", "") == 0.0
