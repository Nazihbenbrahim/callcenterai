from pathlib import Path

import pandas as pd


def main():
    train_path = Path("data/processed/train.csv")
    df = pd.read_csv(train_path)

    print("Exemples de documents :")
    print(df["Document"].head(), "\n")

    print("Répartition des classes :")
    print(df["Topic_group"].value_counts())


if __name__ == "__main__":
    main()
