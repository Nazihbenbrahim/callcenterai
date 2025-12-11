from pathlib import Path
import json

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def main():
    project_root = Path(__file__).resolve().parents[3]
    model_dir = project_root / "models" / "transformer" / "model"
    label_map_path = project_root / "models" / "transformer" / "label_mapping.json"

    with open(label_map_path, "r", encoding="utf-8") as f:
        labels = json.load(f)["classes"]

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    text = "My laptop is not turning on, I think there is a hardware problem."
    encoded = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)[0]

    topk = torch.topk(probs, k=3)
    print("Texte:", text)
    for idx, score in zip(topk.indices, topk.values):
        print(f"- {labels[int(idx)]}: {float(score):.3f}")


if __name__ == "__main__":
    main()
