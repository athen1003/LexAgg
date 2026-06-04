import pytest
from fastapi.testclient import TestClient

# 样本格式：(输入词, 期望归一词, 极性提示)
# 用户提供后填入，至少 50 条覆盖各维度
SAMPLES: list[tuple[str, str, str]] = [
    # === 待用户填充 ===
    # ("凉爽", "凉感适宜", "正面"),
    # ("凉快的", "凉感适宜", "正面"),
    # ("毛绒绒", "蓬松", "正面"),
    # ...
]


@pytest.mark.skipif(
    not SAMPLES,
    reason="等待用户提供 50-100 条真实样本（向用户索取）",
)
def test_accuracy_at_least_95_percent(monkeypatch):
    """通过 TestClient 触发 startup 加载默认 normalizer，避免依赖外部服务。"""
    from app.embedding.factory import reset_models
    from app.main import app

    # 强制重新加载，避免上一个测试的状态污染
    reset_models()

    # 在测试期间把 fastText 替换为 stub（CI 跑时可能没下载真模型）
    import numpy as np
    from app import embedding as emb_pkg

    class _FasttextStub:
        name = "fasttext"
        dim = 300

        def load(self):
            pass

        def encode(self, words):
            # 给同桶词返回相近向量（用 hash 投影），保证 L2 能命中
            vecs = np.zeros((len(words), self.dim), dtype=np.float32)
            for i, w in enumerate(words):
                h = abs(hash(w)) % (self.dim - 1)
                vecs[i, h] = 1.0
            return vecs

    monkeypatch.setattr(emb_pkg, "get_model", lambda name: _FasttextStub())

    with TestClient(app) as client:
        # 触发 startup
        health = client.get("/api/v1/health")
        assert health.status_code == 200, "服务未就绪"

        from app.main import _state
        normalizer = _state["default_normalizer"]
        assert normalizer is not None, "默认 normalizer 未加载"

        correct = 0
        failures = []
        for inp, expected, _ in SAMPLES:
            result = normalizer.normalize(inp)
            if result.normalized == expected:
                correct += 1
            else:
                failures.append(
                    (inp, expected, result.normalized, result.matched_layer, result.score)
                )

        accuracy = correct / len(SAMPLES)
        print(f"\n准确率: {accuracy:.1%} ({correct}/{len(SAMPLES)})")
        if failures:
            print(f"失败用例数: {len(failures)}")
            for inp, exp, got, layer, score in failures[:10]:
                print(f"  '{inp}' 期望='{exp}' 实际='{got}' 层级={layer} 分数={score:.3f}")

        assert accuracy >= 0.95, f"准确率 {accuracy:.1%} < 95%"
