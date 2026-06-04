def test_normalize_returns_tsv(app_with_stub, small_input_file):
    client, _ = app_with_stub
    with open(small_input_file, "rb") as f:
        response = client.post(
            "/api/v1/normalize",
            files={"file": ("input.txt", f, "text/plain")},
        )
    assert response.status_code == 200
    text = response.text
    lines = text.strip().splitlines()
    assert len(lines) == 4
    # 精确词命中
    assert "舒适\t舒适" in lines
    # L1 命中（轻盈 → 轻薄）
    assert "轻盈\t轻薄" in lines


def test_normalize_with_debug(app_with_stub, small_input_file):
    client, _ = app_with_stub
    with open(small_input_file, "rb") as f:
        response = client.post(
            "/api/v1/normalize?debug=1",
            files={"file": ("input.txt", f, "text/plain")},
        )
    assert response.status_code == 200
    # 4 列：原文\t归一\t层级\t分数
    parts = response.text.strip().splitlines()[0].split("\t")
    assert len(parts) == 4


def test_normalize_empty_file_400(app_with_stub, tmp_path):
    client, _ = app_with_stub
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    with open(empty, "rb") as f:
        response = client.post(
            "/api/v1/normalize",
            files={"file": ("empty.txt", f, "text/plain")},
        )
    assert response.status_code == 400
    assert response.json()["error"] == "empty_file"


def test_normalize_unknown_model_400(app_with_stub, small_input_file):
    client, _ = app_with_stub
    with open(small_input_file, "rb") as f:
        response = client.post(
            "/api/v1/normalize?model=nonexistent",
            files={"file": ("input.txt", f, "text/plain")},
        )
    assert response.status_code == 400
    assert response.json()["error"] == "unknown_model"


def test_normalize_invalid_encoding_400(app_with_stub, tmp_path):
    client, _ = app_with_stub
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"\xff\xfe invalid utf8")
    with open(bad, "rb") as f:
        response = client.post(
            "/api/v1/normalize",
            files={"file": ("bad.txt", f, "text/plain")},
        )
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_encoding"


def test_reload_success(app_with_stub, tmp_path):
    client, _ = app_with_stub
    response = client.post("/api/v1/admin/reload")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "正面" in body and "负面" in body
