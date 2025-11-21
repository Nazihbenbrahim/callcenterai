from pathlib import Path

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi


def main():
    # Racine du projet : .../callcenterai
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset = "adisongoh/it-service-ticket-classification-dataset"

    api = KaggleApi()
    api.authenticate()

    print("⬇️  Downloading ALL files from Kaggle dataset...")
    api.dataset_download_files(
        dataset,
        path=str(data_dir),
        unzip=True,  # 🔥 important : on décompresse directement
    )

    print(f"📂 Contenu du dossier raw : {data_dir}")
    csv_files = list(data_dir.rglob("*.csv"))
    print("🔍 CSV trouvés :")
    for f in csv_files:
        print("  -", f)

    if not csv_files:
        raise RuntimeError("Aucun fichier CSV trouvé après téléchargement Kaggle.")

    # On essaie de choisir le bon fichier (celui avec 'tickets' ou 'all_tickets')
    chosen = None
    for f in csv_files:
        name = f.name.lower()
        if "ticket" in name or "all_tickets" in name:
            chosen = f
            break

    if chosen is None:
        # Sinon on prend le premier par défaut
        chosen = csv_files[0]

    print(f"✅ Fichier choisi : {chosen}")

    # Lecture du CSV (on tente utf-8 puis latin1 au cas où)
    try:
        df = pd.read_csv(chosen)
    except UnicodeDecodeError:
        print("⚠️ Problème d'encodage UTF-8, on tente en latin1...")
        df = pd.read_csv(chosen, encoding="latin1")

    print("Shape:", df.shape)
    print("Colonnes :", df.columns.tolist())
    print(df.head())

    # On sauvegarde un fichier propre standardisé
    final_csv = data_dir / "it_tickets.csv"
    df.to_csv(final_csv, index=False)
    print(f"💾 Dataset standardisé sauvegardé dans : {final_csv}")


if __name__ == "__main__":
    main()
