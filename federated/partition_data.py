"""
IID federated data partitioning for the 3 simulated client organizations.

Loads <data.dataset_dir>/train.parquet only (never the test set) and performs
a stratified 3-way split on the label into roughly equal-sized, similarly
distributed partitions, saved as client_A.parquet, client_B.parquet and
client_C.parquet. No scaling is applied here: each client fits its own scaler
on its own training rows (core.dataset.load_client_data).

Run from the project root (once, after centralized.train_rf has created the split):
    python -m federated.partition_data              # refuses to overwrite existing client files
    python -m federated.partition_data --overwrite
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from core.config import DEFAULT_CONFIG_PATH, load_config
from core.dataset import CLIENT_IDS, LABEL_COL, client_path

TRAIN_FILENAME = "train.parquet"


def partition_iid(df: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    """Stratified split into len(CLIENT_IDS) == 3 equal-sized partitions."""
    # Split off Client A (~1/3), stratified on label.
    client_a, remainder = train_test_split(df, test_size=2 / 3, stratify=df[LABEL_COL], random_state=seed)
    # Split the remaining 2/3 evenly into Client B and Client C.
    client_b, client_c = train_test_split(remainder, test_size=0.5, stratify=remainder[LABEL_COL], random_state=seed)
    return dict(zip(CLIENT_IDS, (client_a, client_b, client_c)))


def print_summary(partitions: dict[str, pd.DataFrame]) -> None:
    header = f"{'Client':<8}{'Rows':>10}{'Benign':>12}{'Attack':>12}{'Attack %':>12}"
    print(header)
    print("-" * len(header))
    for name, df in partitions.items():
        rows = len(df)
        attack = int((df[LABEL_COL] == 1).sum())
        print(f"{name:<8}{rows:>10,}{rows - attack:>12,}{attack:>12,}{attack / rows * 100:>11.2f}%")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Partition train.parquet into client A/B/C files")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--overwrite", action="store_true", help="replace existing client files")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    dataset_dir = cfg.data.dataset_dir
    existing = [client_path(cid, dataset_dir) for cid in CLIENT_IDS if client_path(cid, dataset_dir).exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"Client files already exist ({', '.join(p.name for p in existing)}). "
                         f"Pass --overwrite to regenerate them.")

    train_path = dataset_dir / TRAIN_FILENAME
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} not found — run python -m centralized.train_rf first (it creates the split)")
    df = pd.read_parquet(train_path)
    print(f"Loaded {train_path}: {df.shape[0]:,} rows, {df.shape[1]} columns")

    partitions = partition_iid(df, cfg.seed)
    for cid, client_df in partitions.items():
        path = client_path(cid, dataset_dir)
        client_df.to_parquet(path, index=False)
        print(f"Saved Client {cid} partition to {path}: {client_df.shape[0]:,} rows")
    print()
    print_summary(partitions)


if __name__ == "__main__":
    main()
