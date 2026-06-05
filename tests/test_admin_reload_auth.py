"""Test /api/v1/admin/reload authentication.

Behavior:
- ADMIN_TOKEN unset: loopback (127.0.0.1, ::1) is allowed; non-loopback → 401.
- ADMIN_TOKEN=secret: requires Authorization: Bearer secret.
  - No header → 401
  - Wrong token → 401
  - Correct token → 200

Implementation note: helper functions (_is_loopback, _check_admin_auth) are
exercised directly so we don't have to manipulate the ASGI client.host in
TestClient (which sends from 'testclient' rather than 127.0.0.1).
"""
import importlib
import sys

import pytest


@pytest.fixture
def fresh_main(monkeypatch, tmp_path):
    """Reload app.main with stubs registered BEFORE the reimport.

    Avoids sys.modules pollution: we don't delete app.* modules, we just
    reload app.main with ADMIN_TOKEN (or without) and ensure factory
    has the right stubs registered before _load_state runs.
    """
    csv = tmp_path / "vocab.csv"
    csv.write_text("大类,词,极性\n体感,舒适,正面\n体感,不舒适,负面\n", encoding="utf-8")

    # Stub the factory BEFORE reloading main, so _load_state's get_model("bge")
    # finds the stub.
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
    factory._REGISTRY["fasttext"] = _Stub
    factory._registered = True  # skip the re-registration in _register_defaults

    monkeypatch.setenv("VOCAB_PATH", str(csv))

    from app import main as main_module
    importlib.reload(main_module)
    yield main_module

    # Teardown: clear our stub override to avoid bleeding into other tests.
    factory.reset_models()


# ==================== Helper-level unit tests ====================

def test_is_loopback_recognizes_loopback(monkeypatch):
    from app import main as main_module
    assert main_module._is_loopback("127.0.0.1") is True
    assert main_module._is_loopback("127.0.0.5") is True
    assert main_module._is_loopback("::1") is True
    assert main_module._is_loopback("192.168.1.1") is False
    assert main_module._is_loopback("8.8.8.8") is False
    assert main_module._is_loopback(None) is False
    assert main_module._is_loopback("not-an-ip") is False


def test_check_admin_auth_no_token_loopback_allows(monkeypatch):
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    from app import main as main_module
    importlib.reload(main_module)
    assert main_module._check_admin_auth("127.0.0.1", None) is None
    assert main_module._check_admin_auth("::1", None) is None


def test_check_admin_auth_no_token_non_loopback_denies(monkeypatch):
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    from app import main as main_module
    importlib.reload(main_module)
    resp = main_module._check_admin_auth("192.168.1.1", None)
    assert resp is not None
    assert resp.status_code == 401


def test_check_admin_auth_token_set_no_header_denies(monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    from app import main as main_module
    importlib.reload(main_module)
    resp = main_module._check_admin_auth("127.0.0.1", None)
    assert resp is not None
    assert resp.status_code == 401
    resp = main_module._check_admin_auth("127.0.0.1", "")
    assert resp is not None
    assert resp.status_code == 401
    resp = main_module._check_admin_auth("127.0.0.1", "Basic secret")
    assert resp is not None
    assert resp.status_code == 401


def test_check_admin_auth_token_set_wrong_token_denies(monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    from app import main as main_module
    importlib.reload(main_module)
    resp = main_module._check_admin_auth("127.0.0.1", "Bearer wrong-token")
    assert resp is not None
    assert resp.status_code == 401


def test_check_admin_auth_token_set_correct_token_allows(monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    from app import main as main_module
    importlib.reload(main_module)
    assert main_module._check_admin_auth("127.0.0.1", "Bearer secret") is None


# ==================== Integration: full handler with ADMIN_TOKEN path ====================

def test_admin_reload_integration_token_required(fresh_main, monkeypatch):
    """With ADMIN_TOKEN=secret, the integration test exercises the Bearer path."""
    from fastapi.testclient import TestClient
    monkeypatch.setenv("ADMIN_TOKEN", "integration-secret")
    importlib.reload(fresh_main)
    with TestClient(fresh_main.app) as client:
        # No header → 401
        r1 = client.post("/api/v1/admin/reload")
        assert r1.status_code == 401
        # Wrong token → 401
        r2 = client.post(
            "/api/v1/admin/reload",
            headers={"Authorization": "Bearer wrong"},
        )
        assert r2.status_code == 401
        # Correct token → 200
        r3 = client.post(
            "/api/v1/admin/reload",
            headers={"Authorization": "Bearer integration-secret"},
        )
        assert r3.status_code == 200
        assert r3.json()["status"] == "ok"
