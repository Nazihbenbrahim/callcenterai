from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    project_root = Path(__file__).resolve().parents[2]
    raw_path = project_root / "data" / "raw" / "it_tickets.csv"
    out_dir = project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"📄 Lecture du dataset brut : {raw_path}")
    df = pd.read_csv(raw_path)

    # On garde uniquement les colonnes utiles
    df = df[["Document", "Topic_group"]].dropna()
    print("Dataset complet :", df.shape)

    # 70% train, 15% val, 15% test (2 étapes)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,
        random_state=42,
        stratify=df["Topic_group"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df["Topic_group"],
    )

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    print("Train :", train_df.shape)
    print("Val   :", val_df.shape)
    print("Test  :", test_df.shape)
    print(f"💾 Fichiers enregistrés dans : {out_dir}")


if __name__ == "__main__":
    main()
