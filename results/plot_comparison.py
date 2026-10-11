"""
Plots built from the saved result files (nothing is re-evaluated):

  final_comparison_bar.png       held-out test errors per model (missed attacks, false alarms)
  final_comparison_metrics.png   held-out test scores per model (dot plot)
  convergence_overlay.png        federated validation PR-AUC by round vs the centralized ANN
  federated_lr_convergence.png   federated LR per-round validation metrics
  federated_ann_convergence.png  federated ANN per-round validation metrics

Colour follows the model family (RF blue, ANN orange, LR aqua); centralized vs
federated is shown by fill (solid vs striped bars, filled vs hollow dots) and
line style, so identity never relies on colour alone.

Run from the project root:
    python -m results.plot_comparison
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.config import DEFAULT_CONFIG_PATH, load_config
from results.sources import centralized_ann_best_val_pr_auc, load_round_histories, load_test_metrics

# Reference palette (light mode): first three categorical slots, validated all-pairs.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
AXIS = "#b5b3ad"
FAMILY_COLOR = {"RF": "#2a78d6", "ANN": "#eb6834", "LR": "#1baf7a"}

MODEL_ORDER = ["Centralized RF", "Centralized ANN (tuned)", "Federated ANN", "Federated LR"]
MODEL_STYLE = {  # family colour + centralized/federated encoding
    "Centralized RF": {"family": "RF", "federated": False},
    "Centralized ANN (tuned)": {"family": "ANN", "federated": False},
    "Federated ANN": {"family": "ANN", "federated": True},
    "Federated LR": {"family": "LR", "federated": True},
}
SCORE_METRICS = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall (attack)"),
                 ("f1", "F1 (attack)"), ("macro_f1", "Macro-F1"), ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")]


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "text.color": TEXT_PRIMARY, "axes.labelcolor": TEXT_SECONDARY,
        "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
        "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "legend.frameon": False,
    })


def _color(name: str) -> str:
    return FAMILY_COLOR[MODEL_STYLE[name]["family"]]


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_test_errors(metrics: dict, path: Path) -> None:
    """Horizontal bars: missed attacks (FN) and false alarms (FP) on the test set, one panel each."""
    names = [n for n in MODEL_ORDER if n in metrics]
    n_test = next(iter(metrics.values())).n_samples
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    y = np.arange(len(names))[::-1]
    for ax, (attr, title) in zip(axes, [("fn", "Missed attacks (false negatives)"),
                                        ("fp", "False alarms (false positives)")]):
        values = [getattr(metrics[n], attr) for n in names]
        for yi, n, v in zip(y, names, values):
            fed = MODEL_STYLE[n]["federated"]
            # Striped = federated; the surface-coloured edge doubles as the gap between bars.
            ax.barh(yi, v, height=0.62, color=_color(n), edgecolor=SURFACE, linewidth=2,
                    hatch="////" if fed else None)
            ax.text(v + max(values) * 0.015, yi, f"{v:,}", va="center", ha="left", color=TEXT_PRIMARY, fontsize=9)
        ax.set_title(title, loc="left")
        ax.set_xlim(0, max(values) * 1.18)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("flows")
    axes[0].set_yticks(y, names)
    axes[0].tick_params(axis="y", length=0)
    fig.suptitle(f"Held-out test errors by model ({n_test:,} flows) — fewer is better",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, -0.04, "Solid bars = centralized, striped = federated. ANN/LR thresholds selected on validation; "
             "RF uses 0.5.", color=TEXT_SECONDARY, fontsize=9)
    fig.tight_layout()
    _save(fig, path)


def plot_test_scores(metrics: dict, path: Path) -> None:
    """Dot plot of the score metrics (x axis zoomed to 0.90-1.00; dots carry no zero-baseline meaning)."""
    names = [n for n in MODEL_ORDER if n in metrics]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    rows = np.arange(len(SCORE_METRICS))[::-1]
    offsets = np.linspace(0.24, -0.24, len(names))  # dodge so near-equal values don't hide each other
    for n, off in zip(names, offsets):
        fed = MODEL_STYLE[n]["federated"]
        xs = [getattr(metrics[n], key) for key, _ in SCORE_METRICS]
        ax.scatter(xs, rows + off, s=46, color=SURFACE if fed else _color(n), edgecolor=_color(n),
                   linewidth=2, zorder=3, label=n, clip_on=False)
    if np.isnan(metrics["Centralized RF"].pr_auc):
        ax.text(0.9005, rows[-1] + offsets[0], "RF PR-AUC not recorded", va="center", fontsize=8, color=TEXT_SECONDARY)
    ax.set_yticks(rows, [label for _, label in SCORE_METRICS])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.90, 1.003)  # headroom so dots at ~1.0 are not clipped
    ax.set_xticks(np.arange(0.90, 1.001, 0.02))
    ax.set_xlabel("score on the held-out test set (axis starts at 0.90)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Held-out test scores by model — higher is better", loc="left")
    ax.legend(loc="lower left", bbox_to_anchor=(0, -0.42), ncol=2, fontsize=9)
    fig.tight_layout()
    _save(fig, path)


def plot_convergence_overlay(histories: dict, reference: float, path: Path) -> None:
    """Validation PR-AUC by round for both federated models, against the centralized ANN."""
    fig, ax = plt.subplots(figsize=(8, 4.6))
    rounds = sorted(set().union(*[set(df["round"]) for df in histories.values()]) - {0})
    for name, df in histories.items():
        d = df[df["round"] > 0]
        ax.plot(d["round"], d["val_pr_auc"], color=_color(name), linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1, label=name, zorder=3)
        last = d.iloc[-1]
        # The federated ANN ends just under the centralized reference line: label it below the line.
        dy = -9 if last["val_pr_auc"] < reference and reference - last["val_pr_auc"] < 0.005 else 0
        ax.annotate(f"{name}  {last['val_pr_auc']:.4f}", (last["round"], last["val_pr_auc"]), xytext=(8, dy),
                    textcoords="offset points", va="center", fontsize=9, color=TEXT_PRIMARY)
    ax.axhline(reference, color=FAMILY_COLOR["ANN"], linewidth=1.5, linestyle="--", zorder=2,
               label="Centralized ANN (tuned), best epoch")
    ax.annotate(f"Centralized ANN  {reference:.4f}", (max(rounds), reference), xytext=(8, 7),
                textcoords="offset points", va="center", fontsize=9, color=TEXT_PRIMARY)
    ax.set_xticks(rounds)
    ax.set_xlim(0.5, max(rounds) + 4)
    ax.set_xlabel("communication round")
    ax.set_ylabel("validation PR-AUC")
    ax.set_title("Federated convergence on the validation set (round 0, untrained, omitted)", loc="left")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    _save(fig, path)


def plot_model_convergence(name: str, df: pd.DataFrame, path: Path) -> None:
    """Small multiples: validation loss, PR-AUC, attack recall and FPR by round for one federated model."""
    d = df[df["round"] > 0]
    best = df.attrs["best_round"]
    panels = [("val_loss", "Validation loss (BCE)"), ("val_pr_auc", "Validation PR-AUC"),
              ("val_attack_recall", "Attack recall at threshold 0.5"), ("val_fpr", "False-positive rate at threshold 0.5")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
    for ax, (col, title) in zip(axes.flat, panels):
        ax.axvline(best, color=AXIS, linewidth=1, linestyle=":", zorder=1)
        ax.plot(d["round"], d[col], color=_color(name), linewidth=2, marker="o", markersize=4.5,
                markeredgecolor=SURFACE, markeredgewidth=1, zorder=3)
        b = d[d["round"] == best].iloc[0]
        ax.scatter([best], [b[col]], s=90, facecolor="none", edgecolor=TEXT_PRIMARY, linewidth=1.5, zorder=4)
        ax.set_title(title, loc="left", fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel("communication round")
        ax.set_xticks(list(d["round"]))
    axes[0, 1].annotate(f"best round {best}\n(by val PR-AUC)", (best, d[d['round'] == best]['val_pr_auc'].iloc[0]),
                        xytext=(-60, -34), textcoords="offset points", fontsize=8, color=TEXT_SECONDARY,
                        arrowprops={"arrowstyle": "-", "color": AXIS})
    fig.suptitle(f"{name} — {df.attrs['rounds']} rounds x {df.attrs['local_epochs']} local epochs, validation set",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Regenerate comparison and convergence plots")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    results_dir, graphs_dir = cfg.paths.results_dir, cfg.paths.graphs_dir

    _style()
    metrics = load_test_metrics(results_dir)
    plot_test_errors(metrics, graphs_dir / "final_comparison_bar.png")
    plot_test_scores(metrics, graphs_dir / "final_comparison_metrics.png")

    histories = load_round_histories(results_dir)
    plot_convergence_overlay(histories, centralized_ann_best_val_pr_auc(results_dir), graphs_dir / "convergence_overlay.png")
    files = {"Federated LR": "federated_lr_convergence.png", "Federated ANN": "federated_ann_convergence.png"}
    for name, df in histories.items():
        plot_model_convergence(name, df, graphs_dir / files[name])


if __name__ == "__main__":
    main()
