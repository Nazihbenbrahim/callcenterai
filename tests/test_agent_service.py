# tests/test_agent_service.py
import requests

from src.services.agent_service.main import TFIDF_URL, TRANSFORMER_URL


def test_health_ok(agent_client):
    r = agent_client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["service"] == "agent"


def test_agent_routes_to_tfidf_when_confident(agent_client, monkeypatch):
    """
    Cas : texte court, langue EN, TF-IDF très confiant (>= 0.8)
    => chosen_model doit être 'tfidf'
    """

    def fake_post(url, json, timeout):
        # On ne simule que TF-IDF, Transformer ne sera pas appelé dans ce cas
        assert url == TFIDF_URL

        class Resp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "predictions": [
                        {
                            "label": "Hardware",
                            "probabilities": {
                                "Hardware": 0.92,
                                "Access": 0.05,
                                "HR Support": 0.03,
                            },
                        }
                    ],
                    "model_version": "tfidf_svm_v1_local",
                    "runtime_ms": 5.0,
                }

        return Resp()

    monkeypatch.setattr(requests, "post", fake_post)

    payload = {
        "text": "Laptop not turning on.",  # court, langue EN
        "top_k": 3,
    }
    r = agent_client.post("/classify", json=payload)
    assert r.status_code == 200

    data = r.json()
    assert data["chosen_model"] == "tfidf"
    assert "Routage vers TF-IDF" in data["explanation"]
    assert len(data["predictions"]) <= payload["top_k"]


def test_agent_routes_to_transformer_when_low_conf(agent_client, monkeypatch):
    """
    Cas : TF-IDF répond mais avec confiance faible < 0.8
    => l'agent doit fallback vers Transformer
    """

    def fake_post(url, json, timeout):
        if url == TFIDF_URL:
            # Réponse TF-IDF avec probabilité max < 0.8
            class TfidfResp:
                status_code = 200

                def raise_for_status(self):
                    pass

                def json(self):
                    return {
                        "predictions": [
                            {
                                "label": "Hardware",
                                "probabilities": {
                                    "Hardware": 0.50,
                                    "Access": 0.30,
                                    "HR Support": 0.20,
                                },
                            }
                        ],
                        "model_version": "tfidf_svm_v1_local",
                        "runtime_ms": 10.0,
                    }

            return TfidfResp()

        elif url == TRANSFORMER_URL:
            # Réponse simulée du Transformer
            class TransResp:
                status_code = 200

                def raise_for_status(self):
                    pass

                def json(self):
                    return {
                        "predictions": [
                            {"label": "HR Support", "score": 0.85},
                            {"label": "Hardware", "score": 0.10},
                            {"label": "Access", "score": 0.05},
                        ],
                        "model_version": "transformer_v1_local",
                        "runtime_ms": 50.0,
                    }

            return TransResp()

        else:
            raise AssertionError(f"Unexpected URL called: {url}")

    monkeypatch.setattr(requests, "post", fake_post)

    payload = {
        "text": "I would like to request HR support about my contract.",
        "top_k": 3,
    }
    r = agent_client.post("/classify", json=payload)
    assert r.status_code == 200

    data = r.json()
    assert data["chosen_model"] == "transformer"
    assert "Routage vers Transformer" in data["explanation"]
    assert data["predictions"][0]["label"] == "HR Support"
