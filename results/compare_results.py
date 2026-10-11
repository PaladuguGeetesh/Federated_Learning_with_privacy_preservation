"""
Builds the final centralized-vs-federated comparison table.

Loads results/centralized_rf_metrics.json, results/federated_lr_metrics.json,
and results/federated_mlp_metrics.json (using only the final round of the
federated files), and saves a comparison table to final_comparison.csv and
final_comparison.json.
"""

import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RF_METRICS_PATH = BASE_DIR / "centralized_rf_metrics.json"
LR_METRICS_PATH = BASE_DIR / "federated_lr_metrics.json"
MLP_METRICS_PATH = BASE_DIR / "federated_mlp_metrics.json"

OUTPUT_CSV_PATH = BASE_DIR / "final_comparison.csv"
OUTPUT_JSON_PATH = BASE_DIR / "final_comparison.json"

METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def load_rf_metrics() -> dict:
    with open(RF_METRICS_PATH) as f:
        data = json.load(f)
    metrics = data["metrics"]
    return {
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1_score"],
        "roc_auc": metrics["roc_auc"],
    }


def load_final_round_metrics(path: Path) -> dict:
    with open(path) as f:
        records = json.load(f)
    final_record = max(records, key=lambda r: r["round"])
    return {
        "accuracy": final_record["accuracy"],
        "precision": final_record["precision"],
        "recall": final_record["recall"],
        "f1": final_record["f1_score"],
        "roc_auc": final_record["roc_auc"],
    }


def build_comparison_table() -> pd.DataFrame:
    rf_metrics = load_rf_metrics()
    lr_metrics = load_final_round_metrics(LR_METRICS_PATH)
    mlp_metrics = load_final_round_metrics(MLP_METRICS_PATH)

    rows = {
        "Centralized RF": rf_metrics,
        "Federated LR": lr_metrics,
        "Federated MLP": mlp_metrics,
    }

    df = pd.DataFrame.from_dict(rows, orient="index", columns=METRIC_COLUMNS)
    df.index.name = "model"
    return df


def print_accuracy_gap_summary(df: pd.DataFrame) -> None:
    rf_accuracy = df.loc["Centralized RF", "accuracy"]
    lr_accuracy = df.loc["Federated LR", "accuracy"]
    mlp_accuracy = df.loc["Federated MLP", "accuracy"]

    mlp_gap_pp = (rf_accuracy - mlp_accuracy) * 100
    lr_gap_pp = (rf_accuracy - lr_accuracy) * 100

    print(
        f"Federated MLP is within {mlp_gap_pp:.2f} percentage points of the "
        f"centralized Random Forest baseline; "
        f"Federated LR is within {lr_gap_pp:.2f} points."
    )


def main() -> None:
    df = build_comparison_table()

    print("Comparison table:")
    print(df.to_string(float_format=lambda v: f"{v:.4f}"))

    df.to_csv(OUTPUT_CSV_PATH)
    print(f"Saved comparison table to {OUTPUT_CSV_PATH}")

    df.reset_index().to_json(OUTPUT_JSON_PATH, orient="records", indent=2)
    print(f"Saved comparison table to {OUTPUT_JSON_PATH}")

    print_accuracy_gap_summary(df)


if __name__ == "__main__":
    main()
