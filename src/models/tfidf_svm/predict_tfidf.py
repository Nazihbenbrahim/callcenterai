from pathlib import Path

import joblib
import pandas as pd


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_model():
    project_root = get_project_root()
    model_path = project_root / "models" / "tfidf_svm" / "model.joblib"
    print(f"📦 Chargement du modèle depuis : {model_path}")
    pipeline = joblib.load(model_path)
    return pipeline


def predict_texts(texts):
    model = load_model()
    preds = model.predict(texts)

    # si le modèle est calibré, on a predict_proba
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(texts)

    for i, txt in enumerate(texts):
        print("\n============================")
        print(f"📝 Texte {i+1}: {txt[:200]}...")
        print(f"➡️  Prédiction: {preds[i]}")
        if proba is not None:
            # top 3 classes
            probs = proba[i]
            classes = model.classes_
            top_idx = probs.argsort()[::-1][:3]
            print("📊 Top 3 classes:")
            for j in top_idx:
                print(f"  - {classes[j]}: {probs[j]:.3f}")


def main():
    exemples = [
        "My laptop is not turning on, I think there is a hardware problem.",
        "I cannot access my VPN account since yesterday.",
        "I would like to request HR support about my contract.",
        "Please purchase a new storage device for our team.",
    ]
    predict_texts(exemples)


if __name__ == "__main__":
    main()
