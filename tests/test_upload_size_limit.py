"""Test that the /normalize upload rejects oversized files.

The fix: stream-read in 64KB chunks, abort with 413 when accumulated size
exceeds MAX_FILE_SIZE.

Test approach: send a file > MAX_FILE_SIZE and verify the response is 413.
(The spec acknowledges this is the easy path — verifying the streaming
behavior itself would require deeper ASGI introspection that TestClient
doesn't expose cleanly. The chunked-read pattern is verified by code review.)
"""
import importlib
import sys

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client_with_small_limit(monkeypatch, tmp_path):
    """TestClient with MAX_FILE_SIZE overridden to a small value (1KB)."""
    csv = tmp_path / "vocab.csv"
    csv.write_text("大类,词,极性\n体感,舒适,正面\n体感,不舒适,负面\n", encoding="utf-8")

    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("app"):
            del sys.modules[mod_name]

    monkeypatch.setenv("VOCAB_PATH", str(csv))

    from app.embedding import factory
    from app.embedding.base import EmbeddingModel

    class _Stub(EmbeddingModel):
        def load(self): pass
        def encode(self, words):
            import numpy as np
            return np.ones((len(words), 4), dtype=np.float32)
        @property
        def name(self): return "bge"
        @property
        def dim(self): return 4

    factory.reset_models()
    factory._REGISTRY["bge"] = _Stub
    factory._REGISTRY["bge_base"] = _Stub
    factory._REGISTRY["m3e"] = _Stub
    factory._REGISTRY["fasttext"] = _Stub
    factory._registered = True

    from app import main as main_module
    importlib.reload(main_module)
    # Override the limit
    monkeypatch.setattr(main_module, "MAX_FILE_SIZE", 1024)  # 1KB

    with TestClient(main_module.app) as client:
        yield client, main_module


def test_upload_under_limit_succeeds(client_with_small_limit, tmp_path):
    """Small file under 1KB → 200."""
    client, _ = client_with_small_limit
    f = tmp_path / "small.txt"
    f.write_text("舒适\n", encoding="utf-8")
    with open(f, "rb") as fp:
        r = client.post("/api/v1/normalize", files={"file": ("small.txt", fp, "text/plain")})
    assert r.status_code == 200


def test_upload_over_limit_rejected_413(client_with_small_limit, tmp_path):
    """12KB file > 1KB limit → 413."""
    client, _ = client_with_small_limit
    f = tmp_path / "big.txt"
    f.write_bytes(b"x" * 12 * 1024)  # 12KB
    with open(f, "rb") as fp:
        r = client.post("/api/v1/normalize", files={"file": ("big.txt", fp, "text/plain")})
    assert r.status_code == 413
    assert r.json() == {"error": "file_too_large"}


def test_upload_source_uses_chunked_read(client_with_small_limit):
    """Verify the implementation uses chunked file.read(N) (not file.read()).

    The spec acknowledges this is the 'easier' verification path — checking
    that the implementation uses streaming reads. We use AST inspection of
    the normalize function source.
    """
    import ast
    import inspect
    from app import main as main_module
    src = inspect.getsource(main_module.normalize)
    tree = ast.parse(src)
    # Find all calls to file.read(...) and check at least one has a non-(-1) arg
    found_chunked = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if (isinstance(func, ast.Attribute) and func.attr == "read" and
                    node.args and not (isinstance(node.args[0], ast.Constant) and
                                       node.args[0].value == -1)):
                found_chunked = True
                break
    assert found_chunked, (
        "normalize() should use chunked file.read(N) with N != -1. "
        "Bug: full read was buffering the body before the size check."
    )
