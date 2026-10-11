"""
IID federated data partitioning for 3 simulated client organizations.

Loads dataset/train.parquet only and performs a stratified 3-way split on
the 'label' column into roughly equal-sized, similarly-distributed partitions
(Client A, B, C), saved as dataset/client_A.parquet, client_B.parquet, and
client_C.parquet. No scaling is applied here — that happens per-client inside
the Flower client code in a later step.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
LABEL_COL = "label"

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "dataset" / "train.parquet"
CLIENT_PATHS = {
    "A": BASE_DIR / "dataset" / "client_A.parquet",
    "B": BASE_DIR / "dataset" / "client_B.parquet",
    "C": BASE_DIR / "dataset" / "client_C.parquet",
}


def partition_iid(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    # Split off Client A (~1/3), stratified on label.
    client_a, remainder = train_test_split(
        df,
        test_size=2 / 3,
        stratify=df[LABEL_COL],
        random_state=RANDOM_STATE,
    )
    # Split the remaining 2/3 evenly into Client B and Client C.
    client_b, client_c = train_test_split(
        remainder,
        test_size=0.5,
        stratify=remainder[LABEL_COL],
        random_state=RANDOM_STATE,
    )
    return {"A": client_a, "B": client_b, "C": client_c}


def print_summary(partitions: dict[str, pd.DataFrame]) -> None:
    header = f"{'Client':<8}{'Rows':>10}{'Benign':>12}{'Attack':>12}{'Attack %':>12}"
    print(header)
    print("-" * len(header))
    for name, df in partitions.items():
        rows = len(df)
        benign = int((df[LABEL_COL] == 0).sum())
        attack = int((df[LABEL_COL] == 1).sum())
        attack_pct = attack / rows * 100
        print(f"{name:<8}{rows:>10,}{benign:>12,}{attack:>12,}{attack_pct:>11.2f}%")


def main() -> None:
    df = pd.read_parquet(TRAIN_PATH)
    print(f"Loaded {TRAIN_PATH}: {df.shape[0]:,} rows, {df.shape[1]} columns")

    partitions = partition_iid(df)

    for name, client_df in partitions.items():
        path = CLIENT_PATHS[name]
        client_df.to_parquet(path, index=False)
        print(f"Saved Client {name} partition to {path}: {client_df.shape[0]:,} rows")

    print()
    print_summary(partitions)


if __name__ == "__main__":
    main()
