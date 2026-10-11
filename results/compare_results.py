"""
Builds the final centralized-vs-federated comparison table (held-out test set).

Sources (each written by a one-time test evaluation; nothing is re-evaluated here):
    centralized_rf_metrics.json      Centralized RF   (centralized/train_rf.py, threshold 0.5)
    ann_centralized_metrics.json     Centralized ANN  (core/optuna_search.py --final-test)
    lr_federated_test_metrics.json   Federated LR     (core/evaluate.py --final-test)
    ann_federated_test_metrics.json  Federated ANN    (core/evaluate.py --final-test)

Saves final_comparison.csv and final_comparison.json, and prints the
centralized-vs-federated gaps in percentage points.

Replaces the earlier version, which read federated_lr_metrics.json /
federated_mlp_metrics.json — per-round metrics from the old federated code
that evaluated on the test set every round.
"""

import json
import math
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RF_METRICS_PATH = BASE_DIR / "centralized_rf_metrics.json"
CENTRAL_ANN_PATH = BASE_DIR / "ann_centralized_metrics.json"
FED_LR_PATH = BASE_DIR / "lr_federated_test_metrics.json"
FED_ANN_PATH = BASE_DIR / "ann_federated_test_metrics.json"

OUTPUT_CSV_PATH = BASE_DIR / "final_comparison.csv"
OUTPUT_JSON_PATH = BASE_DIR / "final_comparison.json"

# The five metrics of the original comparison, then the additional columns.
METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
EXTRA_COLUMNS = ["macro_f1", "pr_auc", "fpr", "false_negatives", "false_positives", "threshold"]


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run that model's one-time test evaluation first")
    with open(path) as f:
        return json.load(f)


def _from_test_metrics(test: dict) -> dict:
    """Row from a core.evaluate BinaryMetrics dict."""
    return {**{k: test[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "macro_f1", "pr_auc", "fpr"]},
            "false_negatives": test["fn"], "false_positives": test["fp"], "threshold": test["threshold"],
            "n_attack": test["n_attack"], "n_benign": test["n_benign"]}


def load_rf_row(n_attack: int, n_benign: int) -> dict:
    """RF row. Its metrics file stores only the five metrics (threshold 0.5); the
    confusion counts are recovered exactly from recall/precision and the test set's
    class counts, and checked against the stored accuracy. PR-AUC was not recorded."""
    m = _load(RF_METRICS_PATH)["metrics"]
    tp = round(m["recall"] * n_attack)
    fp = round(tp / m["precision"] - tp)
    fn, tn = n_attack - tp, n_benign - fp
    if not math.isclose((tp + tn) / (n_attack + n_benign), m["accuracy"], abs_tol=1e-9):
        raise ValueError("Recovered RF confusion counts do not reproduce its stored accuracy")
    f1_benign = 2 * tn / (2 * tn + fn + fp)
    return {"accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"], "f1": m["f1_score"],
            "roc_auc": m["roc_auc"], "macro_f1": (m["f1_score"] + f1_benign) / 2, "pr_auc": float("nan"),
            "fpr": fp / n_benign, "false_negatives": fn, "false_positives": fp, "threshold": 0.5,
            "n_attack": n_attack, "n_benign": n_benign}


def build_comparison_table() -> pd.DataFrame:
    central_ann = _from_test_metrics(_load(CENTRAL_ANN_PATH)["test_metrics"])
    fed_lr = _from_test_metrics(_load(FED_LR_PATH)["test_metrics"])
    fed_ann = _from_test_metrics(_load(FED_ANN_PATH)["test_metrics"])
    sizes = {(r["n_attack"], r["n_benign"]) for r in (central_ann, fed_lr, fed_ann)}
    if len(sizes) != 1:
        raise ValueError(f"Results were computed on different test sets: {sizes}")
    rf = load_rf_row(*sizes.pop())

    rows = {"Centralized RF": rf, "Centralized ANN (tuned)": central_ann,
            "Federated LR": fed_lr, "Federated ANN": fed_ann}
    df = pd.DataFrame.from_dict(rows, orient="index")[METRIC_COLUMNS + EXTRA_COLUMNS]
    df.index.name = "model"
    return df


def gap_pp(df: pd.DataFrame, reference: str, other: str) -> pd.Series:
    """reference minus other, in percentage points, for the five main metrics plus macro-F1 and PR-AUC."""
    cols = METRIC_COLUMNS + ["macro_f1", "pr_auc"]
    return (df.loc[reference, cols] - df.loc[other, cols]).astype(float) * 100


def main() -> None:
    df = build_comparison_table()

    print("Held-out test set comparison:")
    print(df.to_string(float_format=lambda v: f"{v:.4f}"))

    ann_gap = gap_pp(df, "Centralized ANN (tuned)", "Federated ANN")
    rf_gap = gap_pp(df, "Centralized RF", "Federated ANN")
    print("\nCentralized ANN minus Federated ANN (percentage points):")
    print(ann_gap.to_string(float_format=lambda v: f"{v:+.2f}"))
    print(f"\nFederated ANN is within {abs(ann_gap['accuracy']):.2f} pp accuracy / {abs(ann_gap['f1']):.2f} pp F1 "
          f"of the centralized tuned ANN, and {rf_gap['accuracy']:.2f} pp accuracy / {rf_gap['f1']:.2f} pp F1 "
          f"behind the centralized RF.")

    df.to_csv(OUTPUT_CSV_PATH)
    payload = {
        "table": json.loads(df.reset_index().to_json(orient="records")),
        "gap_pp_centralized_ann_minus_federated_ann": ann_gap.round(4).to_dict(),
        "gap_pp_centralized_rf_minus_federated_ann": rf_gap.round(4).to_dict(),
        "notes": [
            "All rows: same held-out test set (dataset/test.parquet), each evaluated once.",
            "ANN/LR thresholds were selected on validation (max attack-F1); RF uses its default 0.5.",
            "RF was trained on all of train.parquet; the ANN/LR models on the same rows minus the 10% validation slices.",
            "RF PR-AUC was not recorded by train_rf.py; its FPR/FN/macro-F1 are recovered from its stored metrics.",
        ],
    }
    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {OUTPUT_CSV_PATH}\nSaved {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()
