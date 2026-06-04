import numpy as np


def cosine_single(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """返回 query 与 matrix 每行的余弦相似度。"""
    q_norm = np.linalg.norm(query)
    m_norms = np.linalg.norm(matrix, axis=1)
    if q_norm == 0.0:
        return np.zeros(matrix.shape[0])
    safe_norms = np.where(m_norms == 0.0, 1.0, m_norms)
    dots = matrix @ query
    return (dots / (q_norm * safe_norms)).astype(float)


def levenshtein_ratio(a: str, b: str) -> float:
    """编辑距离 / max(len(a), len(b))，范围 [0, 1]。"""
    import Levenshtein

    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    d = Levenshtein.distance(a, b)
    return d / max(len(a), len(b))
