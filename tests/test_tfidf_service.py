# tests/test_tfidf_service.py
def test_health_ok(tfidf_client):
    r = tfidf_client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_single_text(tfidf_client):
    payload = {
        "text": "My laptop is not turning on, maybe a hardware issue.",
        "top_k": 3,
    }
    r = tfidf_client.post("/predict", json=payload)
    assert r.status_code == 200

    data = r.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
    assert len(data["predictions"]) == 1

    pred = data["predictions"][0]
    assert "label" in pred
    # probabilities peuvent être None si pas de predict_proba, mais chez toi elles existent
    if pred["probabilities"] is not None:
        assert isinstance(pred["probabilities"], dict)
        # top_k classes retournées
        assert 1 <= len(pred["probabilities"]) <= payload["top_k"]


def test_predict_missing_text_400(tfidf_client):
    # ni text ni texts => doit renvoyer 400
    payload = {"top_k": 3}
    r = tfidf_client.post("/predict", json=payload)
    assert r.status_code == 400
