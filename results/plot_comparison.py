"""
Plots the final centralized-vs-federated comparison charts:
- A grouped bar chart of all 5 metrics across the 3 models.
- A convergence overlay of accuracy vs round for both federated models,
  with a horizontal reference line at the Centralized RF's accuracy.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
GRAPHS_DIR = BASE_DIR.parent / "graphs"

RF_METRICS_PATH = BASE_DIR / "centralized_rf_metrics.json"
LR_METRICS_PATH = BASE_DIR / "federated_lr_metrics.json"
MLP_METRICS_PATH = BASE_DIR / "federated_mlp_metrics.json"

BAR_CHART_PATH = GRAPHS_DIR / "final_comparison_bar.png"
CONVERGENCE_OVERLAY_PATH = GRAPHS_DIR / "convergence_overlay.png"

METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
MODEL_COLORS = {
    "Centralized RF": "#4C72B0",
    "Federated LR": "#DD8452",
    "Federated MLP": "#55A868",
}


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


def load_all_rounds(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def load_final_round_metrics(records: list[dict]) -> dict:
    final_record = max(records, key=lambda r: r["round"])
    return {
        "accuracy": final_record["accuracy"],
        "precision": final_record["precision"],
        "recall": final_record["recall"],
        "f1": final_record["f1_score"],
        "roc_auc": final_record["roc_auc"],
    }


def plot_bar_chart(rf_metrics, lr_metrics, mlp_metrics) -> None:
    models = ["Centralized RF", "Federated LR", "Federated MLP"]
    metrics_by_model = {
        "Centralized RF": rf_metrics,
        "Federated LR": lr_metrics,
        "Federated MLP": mlp_metrics,
    }

    x = np.arange(len(METRIC_COLUMNS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model in enumerate(models):
        values = [metrics_by_model[model][m] for m in METRIC_COLUMNS]
        offset = (i - 1) * width
        ax.bar(x + offset, values, width, label=model, color=MODEL_COLORS[model])

    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_title("Centralized vs Federated — Metric Comparison")
    ax.legend()

    fig.tight_layout()
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(BAR_CHART_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved bar chart to {BAR_CHART_PATH}")


def plot_convergence_overlay(rf_accuracy, lr_records, mlp_records) -> None:
    lr_rounds = [r["round"] for r in sorted(lr_records, key=lambda r: r["round"])]
    lr_accuracies = [r["accuracy"] for r in sorted(lr_records, key=lambda r: r["round"])]

    mlp_rounds = [r["round"] for r in sorted(mlp_records, key=lambda r: r["round"])]
    mlp_accuracies = [r["accuracy"] for r in sorted(mlp_records, key=lambda r: r["round"])]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(lr_rounds, lr_accuracies, marker="o", label="Federated LR", color=MODEL_COLORS["Federated LR"])
    ax.plot(mlp_rounds, mlp_accuracies, marker="o", label="Federated MLP", color=MODEL_COLORS["Federated MLP"])
    ax.axhline(
        rf_accuracy,
        linestyle="--",
        color=MODEL_COLORS["Centralized RF"],
        label="Centralized RF (ceiling)",
    )

    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy (centralized test set)")
    ax.set_title("Federated Convergence vs Centralized Baseline")
    ax.legend(loc="lower right")

    fig.tight_layout()
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CONVERGENCE_OVERLAY_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved convergence overlay to {CONVERGENCE_OVERLAY_PATH}")


def main() -> None:
    rf_metrics = load_rf_metrics()
    lr_records = load_all_rounds(LR_METRICS_PATH)
    mlp_records = load_all_rounds(MLP_METRICS_PATH)

    lr_final = load_final_round_metrics(lr_records)
    mlp_final = load_final_round_metrics(mlp_records)

    plot_bar_chart(rf_metrics, lr_final, mlp_final)
    plot_convergence_overlay(rf_metrics["accuracy"], lr_records, mlp_records)


if __name__ == "__main__":
    main()
