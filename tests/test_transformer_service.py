# tests/test_transformer_service.py
def test_health_ok(transformer_client):
    r = transformer_client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["model"] == "transformer_ticket_classifier"
    assert data["num_labels"] > 0


def test_predict_basic(transformer_client):
    payload = {
        "text": "I cannot access my VPN account since yesterday.",
        "top_k": 3,
    }
    r = transformer_client.post("/predict", json=payload)
    assert r.status_code == 200

    data = r.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
    assert 1 <= len(data["predictions"]) <= payload["top_k"]

    first = data["predictions"][0]
    assert "label" in first
    assert "score" in first
    assert 0.0 <= first["score"] <= 1.0
