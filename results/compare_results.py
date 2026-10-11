"""
Builds the final centralized-vs-federated comparison table (held-out test set).

Reads the four one-time test results via results/sources.py (nothing is
re-evaluated) and writes <paths.results_dir>/final_comparison.csv and
final_comparison.json, printing the centralized-vs-federated gaps in
percentage points.

Run from the project root:
    python -m results.compare_results
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from core.config import DEFAULT_CONFIG_PATH, load_config
from results.sources import load_test_metrics

# The five metrics of the original comparison, then the additional columns.
METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
EXTRA_COLUMNS = ["macro_f1", "pr_auc", "fpr", "false_negatives", "false_positives", "threshold"]
GAP_COLUMNS = METRIC_COLUMNS + ["macro_f1", "pr_auc"]


def build_comparison_table(results_dir: Path) -> pd.DataFrame:
    rows = {}
    for name, m in load_test_metrics(results_dir).items():
        rows[name] = {**{c: getattr(m, c) for c in METRIC_COLUMNS + ["macro_f1", "pr_auc", "fpr", "threshold"]},
                      "false_negatives": m.fn, "false_positives": m.fp}
    df = pd.DataFrame.from_dict(rows, orient="index")[METRIC_COLUMNS + EXTRA_COLUMNS]
    df.index.name = "model"
    return df


def gap_pp(df: pd.DataFrame, reference: str, other: str) -> pd.Series:
    """reference minus other, in percentage points."""
    return (df.loc[reference, GAP_COLUMNS] - df.loc[other, GAP_COLUMNS]).astype(float) * 100


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build final_comparison.csv/json from the test results")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)
    results_dir = load_config(args.config).paths.results_dir

    df = build_comparison_table(results_dir)
    print("Held-out test set comparison:")
    print(df.to_string(float_format=lambda v: f"{v:.4f}"))

    ann_gap = gap_pp(df, "Centralized ANN (tuned)", "Federated ANN")
    rf_gap = gap_pp(df, "Centralized RF", "Federated ANN")
    print("\nCentralized ANN minus Federated ANN (percentage points):")
    print(ann_gap.to_string(float_format=lambda v: f"{v:+.2f}"))
    print(f"\nFederated ANN is within {abs(ann_gap['accuracy']):.2f} pp accuracy / {abs(ann_gap['f1']):.2f} pp F1 "
          f"of the centralized tuned ANN, and {rf_gap['accuracy']:.2f} pp accuracy / {rf_gap['f1']:.2f} pp F1 "
          f"behind the centralized RF.")

    csv_path, json_path = results_dir / "final_comparison.csv", results_dir / "final_comparison.json"
    df.to_csv(csv_path)
    payload = {
        "table": json.loads(df.reset_index().to_json(orient="records")),
        "gap_pp_centralized_ann_minus_federated_ann": ann_gap.round(4).to_dict(),
        "gap_pp_centralized_rf_minus_federated_ann": rf_gap.round(4).to_dict(),
        "notes": [
            "All rows: same held-out test set (dataset/test.parquet), each evaluated once.",
            "ANN/LR thresholds were selected on validation (max attack-F1); RF uses its default 0.5.",
            "RF was trained on all of train.parquet; the ANN/LR models on the same rows minus the 10% validation slices.",
            "RF PR-AUC was not recorded by the RF's original test evaluation; its FPR/FN/macro-F1 are recovered "
            "from its stored metrics.",
        ],
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {csv_path}\nSaved {json_path}")


if __name__ == "__main__":
    main()
