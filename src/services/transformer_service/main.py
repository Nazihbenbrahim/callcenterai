from pathlib import Path
import json
import time
from typing import List, Dict

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from prometheus_fastapi_instrumentator import Instrumentator


class PredictRequest(BaseModel):
    text: str
    top_k: int = 3


class ClassScore(BaseModel):
    label: str
    score: float


class PredictResponse(BaseModel):
    predictions: List[ClassScore]
    model_version: str
    runtime_ms: float


app = FastAPI(title="Transformer Ticket Classifier")
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "models" / "transformer" / "model"
LABEL_MAP_PATH = PROJECT_ROOT / "models" / "transformer" / "label_mapping.json"


print(f"📦 Chargement du modèle Transformer depuis : {MODEL_DIR}")
with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    LABELS = json.load(f)["classes"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("✅ Modèle Transformer chargé sur :", device)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "transformer_ticket_classifier",
        "num_labels": len(LABELS),
        "device": str(device),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = time.time()

    encoded = tokenizer(
        req.text,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt",
    )

    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)[0]

    top_k = min(req.top_k, len(LABELS))
    topk = torch.topk(probs, k=top_k)

    predictions = [
        ClassScore(label=LABELS[int(idx)], score=float(score))
        for idx, score in zip(topk.indices, topk.values)
    ]

    runtime_ms = (time.time() - start) * 1000.0

    return PredictResponse(
        predictions=predictions,
        model_version="transformer_v1_local",
        runtime_ms=runtime_ms,
    )
