"""
Exploratory data analysis for the cleaned CICIDS2017 dataset.

Reads dataset/cleaned_data.parquet, prints a before/after-dedup and final
shape summary, and saves a class distribution bar chart to graphs/.

Re-derives the raw (pre-dedup) row count directly from dataset/raw/ so this
script can be run independently of preprocess.py's console output.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "dataset" / "raw"
CLEANED_PATH = BASE_DIR / "dataset" / "cleaned_data.parquet"
GRAPHS_DIR = BASE_DIR / "graphs"

RAW_FILENAMES = [
    "Benign-Monday-no-metadata.parquet",
    "Bruteforce-Tuesday-no-metadata.parquet",
    "DoS-Wednesday-no-metadata.parquet",
    "WebAttacks-Thursday-no-metadata.parquet",
    "Infiltration-Thursday-no-metadata.parquet",
    "Botnet-Friday-no-metadata.parquet",
    "Portscan-Friday-no-metadata.parquet",
    "DDoS-Friday-no-metadata.parquet",
]


def count_raw_rows() -> int:
    total = 0
    for filename in RAW_FILENAMES:
        path = RAW_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Expected raw file not found: {path}. "
                f"Place all 8 CICIDS2017 parquet files in {RAW_DIR}."
            )
        total += pq.ParquetFile(path).metadata.num_rows
    return total


def plot_class_distribution(df: pd.DataFrame) -> Path:
    counts = df["label"].value_counts().sort_index()
    labels = ["Benign (0)", "Attack (1)"]
    values = [counts.get(0, 0), counts.get(1, 0)]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, values, color=["#4C72B0", "#C44E52"])
    ax.set_title("Class Distribution: Benign vs Attack")
    ax.set_ylabel("Number of flows")
    ax.set_ylim(0, max(values) * 1.15)

    total = sum(values)
    for bar, value in zip(bars, values):
        pct = value / total * 100
        ax.annotate(
            f"{value:,}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = GRAPHS_DIR / "label_distribution.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    if not CLEANED_PATH.exists():
        raise FileNotFoundError(
            f"{CLEANED_PATH} not found. Run preprocessing/preprocess.py first."
        )

    df = pd.read_parquet(CLEANED_PATH)

    rows_before = count_raw_rows()
    rows_after = len(df)

    print("EDA Summary")
    print("-----------")
    print(f"Rows before dedup (raw, merged): {rows_before:,}")
    print(f"Rows after dedup (cleaned):      {rows_after:,}")
    print(f"Duplicates removed:              {rows_before - rows_after:,}")
    print(f"Final shape:                     {df.shape[0]:,} rows x {df.shape[1]} columns")

    output_path = plot_class_distribution(df)
    print(f"Saved class distribution chart to {output_path}")


if __name__ == "__main__":
    main()
