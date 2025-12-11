import time
from pathlib import Path
from typing import Dict, List, Optional

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Histogram,
                               generate_latest)
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

# =========================
# Config & chemins
# =========================


def get_project_root() -> Path:
    # main.py est dans: project/src/services/tfidf_service/main.py
    # -> parents[3] = project/
    return Path(__file__).resolve().parents[3]


PROJECT_ROOT = get_project_root()
MODEL_PATH = PROJECT_ROOT / "models" / "tfidf_svm" / "model.joblib"


# =========================
# Prometheus metrics
# =========================

REQUEST_COUNT = Counter(
    "tfidf_requests_total",
    "Total number of TF-IDF prediction requests",
    ["endpoint"],
)

REQUEST_LATENCY = Histogram(
    "tfidf_request_latency_seconds",
    "Latency of TF-IDF prediction requests",
    ["endpoint"],
)

REQUEST_ERRORS = Counter(
    "tfidf_request_errors_total",
    "Total number of errors in TF-IDF prediction requests",
    ["endpoint"],
)


# =========================
# Pydantic schemas
# =========================


class PredictRequest(BaseModel):
    text: Optional[str] = None
    texts: Optional[List[str]] = None
    top_k: int = 3


class Prediction(BaseModel):
    label: str
    probabilities: Optional[Dict[str, float]] = None


class PredictResponse(BaseModel):
    predictions: List[Prediction]
    model_version: Optional[str] = None
    runtime_ms: float


# =========================
# FastAPI app
# =========================

app = FastAPI(
    title="CallCenterAI - TFIDF Service",
    version="1.0.0",
    description="TF-IDF + LinearSVC classifier for IT service tickets.",
)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

MODEL = None  # chargé au startup
CLASSES = None


@app.on_event("startup")
def load_model():
    global MODEL, CLASSES
    print(f"📦 Chargement du modèle TF-IDF depuis : {MODEL_PATH}")
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
    MODEL = joblib.load(MODEL_PATH)
    if hasattr(MODEL, "classes_"):
        CLASSES = list(MODEL.classes_)
    print("✅ Modèle TF-IDF chargé.")


@app.get("/health")
def health():
    """Simple healthcheck."""
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.get("/metrics")
def metrics():
    """Endpoint Prometheus pour scraper les métriques."""
    data = generate_latest()
    return PlainTextResponse(data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    REQUEST_COUNT.labels(endpoint="/predict").inc()
    start_time = time.time()

    try:
        # Normaliser les entrées : supporter text OU texts
        if req.text and req.text.strip():
            texts = [req.text]
        elif req.texts and len(req.texts) > 0:
            texts = req.texts
        else:
            raise HTTPException(
                status_code=400,
                detail="You must provide either 'text' or 'texts' with at least one item.",
            )

        top_k = max(1, req.top_k)

        # Prédiction
        preds = MODEL.predict(texts)

        proba = None
        if hasattr(MODEL, "predict_proba"):
            proba = MODEL.predict_proba(texts)

        results: List[Prediction] = []

        for i, txt in enumerate(texts):
            label = str(preds[i])
            probs_dict: Optional[Dict[str, float]] = None

            if proba is not None:
                prob_vec = proba[i]
                # top_k classes
                top_indices = prob_vec.argsort()[::-1][:top_k]
                probs_dict = {str(CLASSES[j]): float(prob_vec[j]) for j in top_indices}

            results.append(
                Prediction(
                    label=label,
                    probabilities=probs_dict,
                )
            )

        elapsed_ms = (time.time() - start_time) * 1000.0
        REQUEST_LATENCY.labels(endpoint="/predict").observe(elapsed_ms / 1000.0)

        return PredictResponse(
            predictions=results,
            model_version="tfidf_svm_v1_local",
            runtime_ms=elapsed_ms,
        )

    except HTTPException:
        REQUEST_ERRORS.labels(endpoint="/predict").inc()
        raise
    except Exception as e:
        REQUEST_ERRORS.labels(endpoint="/predict").inc()
        raise HTTPException(status_code=500, detail=str(e))
