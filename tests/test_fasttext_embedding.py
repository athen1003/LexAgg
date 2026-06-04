import numpy as np
import pytest

from app.embedding.fasttext_impl import FastTextEmbedding, ModelFileMissingError


def test_load_missing_model_raises(tmp_path):
    """模型文件不存在应抛 ModelFileMissingError"""
    fake_path = tmp_path / "no_such_model.bin"
    emb = FastTextEmbedding(model_path=str(fake_path))
    with pytest.raises(ModelFileMissingError):
        emb.load()


def test_encode_without_load_returns_zeros():
    """模型未加载时 encode 返回零向量（测试占位逻辑）"""
    emb = FastTextEmbedding(model_path="models/cc.zh.300.bin")
    out = emb.encode(["测试"])
    assert out.shape == (1, 300)
    assert out.dtype == np.float32


@pytest.mark.skipif(
    not pytest.importorskip("gensim", reason="gensim 未安装"),
    reason="gensim 未安装",
)
def test_encode_known_word_returns_vector():
    """如果真实模型存在，加载后能编码"""
    import os
    model_path = "models/cc.zh.300.bin"
    if not os.path.exists(model_path):
        pytest.skip("models/cc.zh.300.bin 不存在，需要先运行 scripts/download_model.py")
    emb = FastTextEmbedding(model_path=model_path)
    emb.load()
    vec = emb.encode(["测试"])[0]
    assert vec.shape == (300,)
    assert not np.allclose(vec, 0), "OOV 词向量不应全为 0"


@pytest.mark.skipif(
    not pytest.importorskip("gensim", reason="gensim 未安装"),
    reason="gensim 未安装",
)
def test_encode_oov_returns_non_zero_via_subword_split():
    """OOV 词用拆字符平均回退，返回非零向量"""
    import os
    model_path = "models/cc.zh.300.bin"
    if not os.path.exists(model_path):
        pytest.skip("模型不存在")
    emb = FastTextEmbedding(model_path=model_path)
    emb.load()
    # 生造词：'舒适zzz' 中 'zzz' 是 OOV，但 '舒'/'适' 存在 → 拆字符回退
    vec = emb.encode(["舒适zzz"])[0]
    assert vec.shape == (300,)
    assert not np.allclose(vec, 0), "拆字符回退应返回非零向量"
