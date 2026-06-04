def test_health_with_loaded_vocab(app_with_stub):
    client, _ = app_with_stub
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "vocab_size" in body
    assert body["vocab_size"] >= 2
