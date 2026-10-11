"""
Round-by-round convergence summary for the federated models.

Loads results/centralized_rf_metrics.json (reference ceiling accuracy) and,
for each of results/federated_lr_metrics.json and
results/federated_mlp_metrics.json, prints the full round-by-round metrics
table plus milestone lines (first round to cross 90/95/98% accuracy, total
accuracy gain from round 1 to the final round, and the final gap to the RF
ceiling).

Read-only: does not modify any existing results files.
"""

import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RF_METRICS_PATH = BASE_DIR / "centralized_rf_metrics.json"

MODELS = [
    ("Federated LR", BASE_DIR / "federated_lr_metrics.json"),
    ("Federated MLP", BASE_DIR / "federated_mlp_metrics.json"),
]

MILESTONE_THRESHOLDS = [0.90, 0.95, 0.98]
TABLE_COLUMNS = ["round", "accuracy", "precision", "recall", "f1", "roc_auc", "gap_to_rf_pp"]


def load_rf_accuracy() -> float:
    with open(RF_METRICS_PATH) as f:
        data = json.load(f)
    return data["metrics"]["accuracy"]


def load_rounds_df(path: Path, rf_accuracy: float) -> pd.DataFrame:
    with open(path) as f:
        records = json.load(f)

    df = pd.DataFrame(records)
    df = df.rename(columns={"f1_score": "f1"})
    df["gap_to_rf_pp"] = (rf_accuracy - df["accuracy"]) * 100
    df = df.sort_values("round").reset_index(drop=True)
    return df


def print_table(model_name: str, df: pd.DataFrame) -> None:
    header = f"{model_name} — Round-by-Round Convergence"
    print(header)
    print("-" * len(header))
    print(df[TABLE_COLUMNS].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()


def print_milestones(df: pd.DataFrame) -> None:
    for threshold in MILESTONE_THRESHOLDS:
        reached = df[df["accuracy"] >= threshold]
        if reached.empty:
            print(f"  Never reached {threshold * 100:.0f}% accuracy")
        else:
            first_round = int(reached.iloc[0]["round"])
            print(f"  First reached {threshold * 100:.0f}% accuracy at round {first_round}")

    round1 = df[df["round"] == 1]
    final_row = df.loc[df["round"].idxmax()]
    if not round1.empty:
        gain_pp = (final_row["accuracy"] - round1.iloc[0]["accuracy"]) * 100
        print(f"  Accuracy gain (round 1 -> round {int(final_row['round'])}): {gain_pp:+.4f} pp")
    else:
        print("  Accuracy gain (round 1 -> final): round 1 not found")

    print(f"  Final gap to RF ceiling: {final_row['gap_to_rf_pp']:.4f} pp")
    print()


def main() -> None:
    rf_accuracy = load_rf_accuracy()
    print(f"Centralized RF reference accuracy (ceiling): {rf_accuracy:.4f}")
    print()

    for model_name, path in MODELS:
        df = load_rounds_df(path, rf_accuracy)
        print_table(model_name, df)
        print_milestones(df)


if __name__ == "__main__":
    main()
