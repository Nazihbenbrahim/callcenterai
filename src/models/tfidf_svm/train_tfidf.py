from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

PROJECT_NAME = "callcenterai-tfidf-svm"


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def setup_mlflow(project_root: Path) -> None:
    tracking_db_path = project_root / "mlflow.db"
    tracking_uri = f"sqlite:///{tracking_db_path}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment(PROJECT_NAME)


def load_data():
    project_root = get_project_root()
    train_path = project_root / "data" / "processed" / "train.csv"
    val_path = project_root / "data" / "processed" / "val.csv"

    print(f"📄 Lecture train : {train_path}")
    print(f"📄 Lecture val   : {val_path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    X_train = train_df["Document"].astype(str).values
    y_train = train_df["Topic_group"].astype(str).values

    X_val = val_df["Document"].astype(str).values
    y_val = val_df["Topic_group"].astype(str).values

    return X_train, y_train, X_val, y_val


def build_pipeline():
    max_features = 50000
    ngram_range = (1, 2)
    C = 1.0
    use_calibration = True

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
    )

    base_clf = LinearSVC(C=C, class_weight="balanced")

    if use_calibration:
        clf = CalibratedClassifierCV(base_clf, cv=3)
    else:
        clf = base_clf

    pipeline = Pipeline(
        [
            ("tfidf", vectorizer),
            ("clf", clf),
        ]
    )

    params = {
        "max_features": max_features,
        "ngram_range": ngram_range,
        "C": C,
        "use_calibration": use_calibration,
    }

    return pipeline, params


def main():
    project_root = get_project_root()
    models_dir = project_root / "models" / "tfidf_svm"
    models_dir.mkdir(parents=True, exist_ok=True)

    setup_mlflow(project_root)

    X_train, y_train, X_val, y_val = load_data()
    pipeline, params = build_pipeline()

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.set_tag("author", "Nazih")
        mlflow.set_tag("model_type", "TFIDF_SVM")

        print("🚀 Entraînement...")
        pipeline.fit(X_train, y_train)

        print("🔎 Évaluation...")
        y_pred = pipeline.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1_macro = f1_score(y_val, y_pred, average="macro")

        print("Accuracy:", acc)
        print("F1 macro:", f1_macro)

        mlflow.log_metric("val_accuracy", acc)
        mlflow.log_metric("val_f1_macro", f1_macro)

        model_path = models_dir / "model.joblib"
        joblib.dump(pipeline, model_path)
        print(f"💾 Modele sauvegardé : {model_path}")

        input_example = X_val[:5].tolist()
        signature = infer_signature(X_val[:5], y_pred[:5])

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="tfidf_svm_model",
            registered_model_name="tfidf_svm_model",
            input_example=input_example,
            signature=signature,
        )


if __name__ == "__main__":
    main()
