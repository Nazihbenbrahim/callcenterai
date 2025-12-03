from pathlib import Path
import os
import re
import time
from typing import List, Literal

import requests
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

# ==========================
#  Config + modèles Pydantic
# ==========================

class AgentRequest(BaseModel):
    text: str
    top_k: int = 3


class ModelPrediction(BaseModel):
    label: str
    score: float


class AgentResponse(BaseModel):
    chosen_model: Literal["tfidf", "transformer"]
    explanation: str
    scrubbed_text: str
    predictions: List[ModelPrediction]
    runtime_ms: float


app = FastAPI(title="CallCenterAI Agent Service")

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# URLs des services en local
TFIDF_URL = os.getenv("TFIDF_URL", "http://127.0.0.1:8000/predict")
TRANSFORMER_URL = os.getenv("TRANSFORMER_URL", "http://127.0.0.1:8001/predict")

print(f"🔗 TF-IDF service URL       : {TFIDF_URL}")
print(f"🔗 Transformer service URL  : {TRANSFORMER_URL}")

# ==============
#  PII scrubbing
# ==============

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d\s\-]{7,}\d")
ID_RE = re.compile(r"\b\d{8,}\b")  # longues suites de chiffres (numéro de compte, carte, etc.)


def scrub_pii(text: str) -> str:
    """Masque les infos sensibles de base."""
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    text = ID_RE.sub("[ID]", text)
    return text


def simple_lang_hint(text: str) -> str:
    """
    Très simple 'hint' de langue :
    - si beaucoup de caractères non ASCII -> on considère 'other'
    - sinon -> 'en' (ça suffit pour le routage de démo)
    """
    non_ascii = sum(1 for c in text if ord(c) > 127)
    ratio = non_ascii / max(len(text), 1)
    return "other" if ratio > 0.2 else "en"


# ==========
#  Metrics
# ==========

TOTAL_REQUESTS = 0
TFIDF_CALLS = 0
TRANSFORMER_CALLS = 0
ERROR_COUNT = 0


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "agent",
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """
    Format texte simple compatible Prometheus.
    (Prometheus viendra plus tard, pour l’instant on prépare juste la sortie.)
    """
    lines = [
        "# HELP callcenterai_agent_requests_total Total number of requests received by the agent",
        "# TYPE callcenterai_agent_requests_total counter",
        f"callcenterai_agent_requests_total {TOTAL_REQUESTS}",
        "# HELP callcenterai_agent_tfidf_calls_total Total number of routed requests to TF-IDF",
        "# TYPE callcenterai_agent_tfidf_calls_total counter",
        f"callcenterai_agent_tfidf_calls_total {TFIDF_CALLS}",
        "# HELP callcenterai_agent_transformer_calls_total Total number of routed requests to Transformer",
        "# TYPE callcenterai_agent_transformer_calls_total counter",
        f"callcenterai_agent_transformer_calls_total {TRANSFORMER_CALLS}",
        "# HELP callcenterai_agent_errors_total Total number of errors in the agent",
        "# TYPE callcenterai_agent_errors_total counter",
        f"callcenterai_agent_errors_total {ERROR_COUNT}",
    ]
    return "\n".join(lines)


# ===========
#  Core agent
# ===========

@app.post("/classify", response_model=AgentResponse)
def classify(req: AgentRequest):
    global TOTAL_REQUESTS, TFIDF_CALLS, TRANSFORMER_CALLS, ERROR_COUNT

    TOTAL_REQUESTS += 1
    start = time.time()

    original_text = req.text
    scrubbed = scrub_pii(original_text)
    lang = simple_lang_hint(original_text)
    text_len = len(original_text)

    # 1️⃣ Appel TF-IDF d’abord (rapide)
    try:
        tfidf_payload = {"text": scrubbed, "top_k": req.top_k}
        tfidf_resp = requests.post(TFIDF_URL, json=tfidf_payload, timeout=5)
        tfidf_resp.raise_for_status()
        tfidf_json = tfidf_resp.json()
    except Exception as e:
        ERROR_COUNT += 1
        # En cas d’erreur TF-IDF, fallback direct sur Transformer
        transformer_predictions, rt_ms = _call_transformer(scrubbed, req.top_k, start)
        explanation = f"TF-IDF unavailable (error: {e}), fallback to Transformer."
        TRANSFORMER_CALLS += 1
        return AgentResponse(
            chosen_model="transformer",
            explanation=explanation,
            scrubbed_text=scrubbed,
            predictions=transformer_predictions,
            runtime_ms=rt_ms,
        )

    # Analyse confiance TF-IDF
    top_label = tfidf_json["predictions"][0]["label"]
    probas: dict = tfidf_json["predictions"][0]["probabilities"]
    top_prob = max(probas.values())

    # 2️⃣ Heuristique de routage
    # - texte court (< 200 caractères)
    # - langue "en" (approximation)
    # - confiance TF-IDF >= 0.80
    if text_len < 200 and lang == "en" and top_prob >= 0.80:
        TFIDF_CALLS += 1
        runtime_ms = (time.time() - start) * 1000.0
        explanation = (
            f"Routage vers TF-IDF : texte court ({text_len} chars), langue={lang}, "
            f"top_prob={top_prob:.2f} >= 0.80."
        )
        preds = [
            ModelPrediction(label=lbl, score=float(score))
            for lbl, score in sorted(probas.items(), key=lambda x: x[1], reverse=True)[: req.top_k]
        ]
        return AgentResponse(
            chosen_model="tfidf",
            explanation=explanation,
            scrubbed_text=scrubbed,
            predictions=preds,
            runtime_ms=runtime_ms,
        )

    # Sinon, on route vers Transformer
    transformer_predictions, rt_ms = _call_transformer(scrubbed, req.top_k, start)
    TRANSFORMER_CALLS += 1
    explanation = (
        f"Routage vers Transformer : "
        f"conditions TF-IDF non satisfaites (len={text_len}, lang={lang}, top_prob={top_prob:.2f})."
    )

    return AgentResponse(
        chosen_model="transformer",
        explanation=explanation,
        scrubbed_text=scrubbed,
        predictions=transformer_predictions,
        runtime_ms=rt_ms,
    )


def _call_transformer(scrubbed_text: str, top_k: int, start_time: float):
    payload = {"text": scrubbed_text, "top_k": top_k}
    resp = requests.post(TRANSFORMER_URL, json=payload, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    preds = [
        ModelPrediction(label=p["label"], score=p["score"])
        for p in data["predictions"][:top_k]
    ]
    runtime_ms = (time.time() - start_time) * 1000.0
    return preds, runtime_ms
