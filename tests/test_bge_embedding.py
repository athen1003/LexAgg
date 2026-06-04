"""BGE-small-zh 集成测试：首次运行会自动下载 ~95MB 模型。"""
import os

import numpy as np
import pytest

from app.embedding.bge_impl import BgeEmbedding


@pytest.mark.skipif(
    os.environ.get("WORDTEST_SKIP_BGE") == "1",
    reason="BGE 测试需要联网下载模型，可设置 WORDTEST_SKIP_BGE=1 跳过",
)
def test_bge_loads_and_encodes():
    emb = BgeEmbedding()
    emb.load()
    vecs = emb.encode(["舒适", "凉爽", "毛绒绒"])
    assert vecs.shape == (3, 512)
    assert vecs.dtype == np.float32
    # 归一化后每行向量模长应该接近 1
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), f"向量未归一化: {norms}"


@pytest.mark.skipif(
    os.environ.get("WORDTEST_SKIP_BGE") == "1",
    reason="BGE 测试需要联网下载模型",
)
def test_bge_similar_words_have_higher_cosine():
    emb = BgeEmbedding()
    emb.load()
    vecs = emb.encode(["凉爽", "凉感适宜", "厚重"])
    # 凉爽 vs 凉感适宜 应比 凉爽 vs 厚重 更相似
    sim_close = float(np.dot(vecs[0], vecs[1]))
    sim_far = float(np.dot(vecs[0], vecs[2]))
    assert sim_close > sim_far, f"语义距离反了: 近={sim_close:.3f} 远={sim_far:.3f}"
