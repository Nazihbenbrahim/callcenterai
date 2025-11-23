from pathlib import Path
import json

import mlflow
import mlflow.transformers
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from mlflow.models.signature import infer_signature
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate


MODEL_NAME = "distilbert-base-multilingual-cased"
EXPERIMENT_NAME = "callcenterai-transformer"


def get_project_root() -> Path:
    # train_transformer.py est dans: project/src/models/transformer/train_transformer.py
    # -> parents[3] = project/
    return Path(__file__).resolve().parents[3]


def load_data() -> tuple[DatasetDict, LabelEncoder]:
    project_root = get_project_root()
    data_dir = project_root / "data" / "processed"

    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    # Encode des labels
    le = LabelEncoder()
    le.fit(train_df["Topic_group"])

    train_df["label"] = le.transform(train_df["Topic_group"])
    val_df["label"] = le.transform(val_df["Topic_group"])
    test_df["label"] = le.transform(test_df["Topic_group"])

    # HuggingFace Datasets
    train_ds = Dataset.from_pandas(train_df[["Document", "label"]])
    val_ds = Dataset.from_pandas(val_df[["Document", "label"]])
    test_ds = Dataset.from_pandas(test_df[["Document", "label"]])

    ds = DatasetDict(
        {
            "train": train_ds,
            "validation": val_ds,
            "test": test_ds,
        }
    )

    return ds, le


def tokenize_datasets(ds: DatasetDict, tokenizer, max_length: int = 256) -> DatasetDict:
    def preprocess(examples):
        return tokenizer(
            examples["Document"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = ds.map(preprocess, batched=True)
    tokenized = tokenized.remove_columns(["Document"])
    tokenized.set_format("torch")
    return tokenized


def main():
    project_root = get_project_root()
    models_dir = project_root / "models" / "transformer"
    model_out_dir = models_dir / "model"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_out_dir.mkdir(parents=True, exist_ok=True)

    # 🔧 MLflow avec la même DB SQLite
    tracking_db = project_root / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 📥 Données + encodage des labels
    ds, label_encoder = load_data()
    num_labels = len(label_encoder.classes_)

    # 💾 Sauvegarder le mapping labels
    label_map_path = models_dir / "label_mapping.json"
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "classes": label_encoder.classes_.tolist(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"💾 Label mapping sauvegardé dans : {label_map_path}")

    # 🔤 Tokenizer / modèle
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
    )

    tokenized_ds = tokenize_datasets(ds, tokenizer, max_length=256)

    # 📏 métriques
    accuracy = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy.compute(predictions=preds, references=labels)["accuracy"]
        f1_macro = f1_metric.compute(
            predictions=preds, references=labels, average="macro"
        )["f1"]
        return {"accuracy": acc, "f1_macro": f1_macro}

        # ⚙️ Hyperparamètres (robuste à différentes versions de transformers)
    try:
        # ✅ Version "moderne" (transformers 4.x)
        training_args = TrainingArguments(
            output_dir=str(project_root / "outputs" / "transformer"),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=200,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            learning_rate=5e-5,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            report_to=[],  # pas de wandb & co
        )
    except TypeError as e:
        # 🔁 Fallback pour les anciennes versions de transformers
        print("⚠️ TrainingArguments ne supporte pas certains paramètres dans cet environnement.")
        print("   Détail erreur:", e)
        print("   ➜ Utilisation d'une configuration minimale compatible.")

        training_args = TrainingArguments(
            output_dir=str(project_root / "outputs" / "transformer"),
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            learning_rate=5e-5,
            weight_decay=0.01,
        )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    with mlflow.start_run():
        # Log params
        mlflow.log_params(
            {
                "model_name": MODEL_NAME,
                "num_labels": num_labels,
                "num_train_epochs": training_args.num_train_epochs,
                "learning_rate": training_args.learning_rate,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "max_length": 256,
            }
        )

        print("🚀 Fine-tuning Transformer...")
        trainer.train()

        print("🔎 Évaluation sur le test set...")
        metrics_test = trainer.evaluate(tokenized_ds["test"])
        print(metrics_test)

        # Log des métriques
        mlflow.log_metrics(
            {
                "test_accuracy": float(metrics_test.get("eval_accuracy", 0.0)),
                "test_f1_macro": float(metrics_test.get("eval_f1_macro", 0.0)),
            }
        )

        # 💾 Sauvegarde du modèle HF (+ tokenizer) dans models/transformer/model
        trainer.save_model(str(model_out_dir))
        tokenizer.save_pretrained(str(model_out_dir))
        print(f"💾 Modèle HF sauvegardé dans : {model_out_dir}")

        # 🔍 Signature + input_example pour MLflow
        example_texts = [
            "My laptop is not turning on, I think there is a hardware problem.",
            "I cannot access my VPN account since yesterday.",
        ]
        encoded_example = tokenizer(
            example_texts,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="np",
        )

        logits = model(
            input_ids=encoded_example["input_ids"],
            attention_mask=encoded_example["attention_mask"],
        ).logits.detach().cpu().numpy()

        signature = infer_signature(encoded_example, logits)

        # Log du modèle dans MLflow (registry)
        mlflow.transformers.log_model(
            transformers_model=model,
            artifact_path="transformer_model",
            task="text-classification",
            tokenizer=tokenizer,
            signature=signature,
            input_example=example_texts,
            registered_model_name="transformer_ticket_classifier",
        )


if __name__ == "__main__":
    main()
