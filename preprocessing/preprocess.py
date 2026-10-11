"""
Preprocessing pipeline for the CICIDS2017 (dhoogla-cleaned) dataset.

Loads the 8 pre-cleaned parquet files from dataset/raw/, validates that they
share an identical schema, merges them, removes exact duplicate rows, encodes
the Label column into a binary 'label' column (Benign=0, Attack=1), and saves
the result to dataset/cleaned_data.parquet.

The source files are already free of nulls/infinite values and identifier
columns (Flow ID / IPs / Timestamp), so no such handling is done here.
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "dataset" / "raw"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "dataset" / "cleaned_data.parquet"

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


def load_raw_files() -> list[pd.DataFrame]:
    dataframes = []
    for filename in RAW_FILENAMES:
        path = RAW_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Expected raw file not found: {path}. "
                f"Place all 8 CICIDS2017 parquet files in {RAW_DIR}."
            )
        df = pd.read_parquet(path)
        print(f"Loaded {filename}: {df.shape[0]:,} rows, {df.shape[1]} columns")
        dataframes.append(df)
    return dataframes


def validate_identical_columns(dataframes: list[pd.DataFrame]) -> None:
    reference_columns = list(dataframes[0].columns)
    for filename, df in zip(RAW_FILENAMES, dataframes):
        if list(df.columns) != reference_columns:
            raise ValueError(
                f"Column mismatch detected in {filename}.\n"
                f"Expected columns (from {RAW_FILENAMES[0]}):\n{reference_columns}\n"
                f"Got columns:\n{list(df.columns)}"
            )
    print(f"All {len(dataframes)} files have identical columns ({len(reference_columns)} columns).")


def merge_dataframes(dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    merged = pd.concat(dataframes, ignore_index=True)
    print(f"Merged shape: {merged.shape[0]:,} rows, {merged.shape[1]} columns")
    return merged


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    rows_before = len(df)
    df = df.drop_duplicates()
    rows_after = len(df)
    removed = rows_before - rows_after
    print(f"Removed {removed:,} duplicate rows ({rows_before:,} -> {rows_after:,})")
    return df.copy()


def encode_label(df: pd.DataFrame) -> pd.DataFrame:
    df["label"] = (df["Label"].astype(str) != "Benign").astype(int)
    df = df.drop(columns=["Label"])

    counts = df["label"].value_counts().sort_index()
    total = len(df)
    print("Class distribution after encoding:")
    for value, count in counts.items():
        name = "Benign (0)" if value == 0 else "Attack (1)"
        print(f"  {name}: {count:,} ({count / total:.2%})")

    return df


def main() -> None:
    dataframes = load_raw_files()
    validate_identical_columns(dataframes)
    merged = merge_dataframes(dataframes)
    deduped = remove_duplicates(merged)
    cleaned = encode_label(deduped)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved cleaned data to {OUTPUT_PATH} — final shape: {cleaned.shape}")


if __name__ == "__main__":
    main()
