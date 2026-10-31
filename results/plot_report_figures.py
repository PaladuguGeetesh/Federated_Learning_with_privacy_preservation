"""
Figures for the project report. Nothing is trained or evaluated here: every number comes from the saved
result files (results/*.json, results/optuna/*.csv) and every snapshot from a saved log.

Data-driven figures
  centralized_ann_training_curves.png   train/validation loss and validation PR-AUC of the tuned centralized ANN
  optuna_history.png                    Optuna trial scores and durations
  local_epochs_comparison.png           federated ANN validation PR-AUC with 1 vs 3 local epochs per round
  client_train_loss.png                 per-client local training loss per federated round
  confusion_matrices.png                held-out test confusion matrices of the four models
Drawn diagrams
  concept_centralized_vs_federated.png, class_diagram.png, activity_diagram.png, use_case_diagram.png, gantt_chart.png
Terminal-style snapshots (graphs/snapshots/*.png), rendered from saved logs and result files

Run from the project root:
    python -m results.plot_report_figures
"""

import argparse
import json
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

from core.config import DEFAULT_CONFIG_PATH, load_config
from results.plot_comparison import AXIS, FAMILY_COLOR, GRID, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, _style
from results.sources import load_test_metrics

RED, NAVY, GREY, GREEN, LIGHT = "#C00000", "#1F3864", "#595959", "#2E7D5B", "#F2F2F2"
SERIF = "serif"


def _save(fig, path: Path, dpi: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------------------------- data-driven
def training_curves(results: Path, out: Path) -> None:
    payload = json.load(open(results / "ann_centralized_metrics.json"))
    h = pd.DataFrame(payload["retrain"]["history"])
    best = payload["retrain"]["best_epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    axes[0].plot(h["epoch"], h["train_loss"], color=FAMILY_COLOR["ANN"], linewidth=2, marker="o", markersize=4, label="train loss")
    axes[0].plot(h["epoch"], h["val_loss"], color=NAVY, linewidth=2, marker="s", markersize=4, label="validation loss")
    axes[0].set_title("Loss per epoch", loc="left")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("BCE loss"); axes[0].legend()
    axes[1].plot(h["epoch"], h["val_pr_auc"], color=GREEN, linewidth=2, marker="o", markersize=4)
    axes[1].axvline(best, color=AXIS, linestyle=":", linewidth=1.2)
    axes[1].scatter([best], [h.loc[h["epoch"] == best, "val_pr_auc"].iloc[0]], s=90, facecolor="none", edgecolor=TEXT_PRIMARY, linewidth=1.5, zorder=4)
    axes[1].annotate(f"best epoch {best}", (best, h.loc[h["epoch"] == best, "val_pr_auc"].iloc[0]), xytext=(-70, -28),
                     textcoords="offset points", fontsize=9, color=TEXT_SECONDARY, arrowprops={"arrowstyle": "-", "color": AXIS})
    axes[1].set_title("Validation PR-AUC (early-stopping metric)", loc="left")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("PR-AUC")
    for ax in axes: ax.set_xticks(h["epoch"][::2])
    fig.suptitle("Centralized ANN (Optuna trial 14): retraining history", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, out)


def optuna_history(results: Path, out: Path) -> None:
    t = pd.read_csv(results / "optuna" / "ann_centralized_trials.csv")
    t["seconds"] = pd.to_timedelta(t["duration"]).dt.total_seconds()
    done, pruned = t[t["state"] == "COMPLETE"], t[t["state"] == "PRUNED"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    ax = axes[0]
    ax.scatter(done["number"], done["value"], s=42, color=FAMILY_COLOR["ANN"], label="complete", zorder=3)
    best_so_far = done["value"].cummax()
    ax.step(done["number"], best_so_far, where="post", color=NAVY, linewidth=1.8, label="best so far")
    ax.set_ylim(0.985, 1.0)
    ax.scatter(pruned["number"], np.full(len(pruned), 0.9857), marker="x", s=36, color=AXIS, label="pruned (not scored)", zorder=3)
    low = done[done["value"] < 0.985]
    ax.set_title("Objective: validation PR-AUC per trial", loc="left")
    ax.set_xlabel("trial number"); ax.set_ylabel("validation PR-AUC")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8, frameon=False)
    if len(low): ax.text(0.01, 0.97, f"{len(low)} complete trial(s) below 0.985 not shown", transform=ax.transAxes, fontsize=8, color=TEXT_SECONDARY, va="top")
    ax = axes[1]
    colors = t["state"].map({"COMPLETE": FAMILY_COLOR["ANN"], "PRUNED": AXIS})
    ax.bar(t["number"], t["seconds"], color=colors, width=0.8)
    ax.set_title("Duration per trial (seconds)", loc="left")
    ax.set_xlabel("trial number"); ax.set_ylabel("seconds")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=FAMILY_COLOR["ANN"], label="complete"), Patch(color=AXIS, label="pruned")], fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    fig.suptitle(f"Optuna study: {len(t)} trials ({len(done)} complete, {len(pruned)} pruned), best trial #{int(done.loc[done['value'].idxmax(), 'number'])}",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, out)


def local_epochs(results: Path, out: Path) -> None:
    three = pd.DataFrame(json.load(open(results / "federated_ann_validation_rounds.json"))["history"])
    one = pd.DataFrame(json.load(open(results / "federated_ann_local1" / "federated_ann_validation_rounds.json"))["history"])
    ref = json.load(open(results / "ann_centralized_metrics.json"))["retrain"]["best_val_pr_auc"]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for df, label, style, color in [(three, "3 local epochs per round", "-", FAMILY_COLOR["ANN"]), (one, "1 local epoch per round", "--", NAVY)]:
        d = df[df["round"] > 0]
        ax.plot(d["round"], d["val_pr_auc"], style, color=color, linewidth=2, marker="o", markersize=4.5, label=label)
    ax.axhline(ref, color=GREEN, linewidth=1.4, linestyle=":", label="centralized ANN (best epoch)")
    ax.set_xticks(range(1, 16)); ax.set_xlabel("communication round"); ax.set_ylabel("validation PR-AUC")
    ax.set_title("Federated ANN: effect of local epochs per round (validation set)", loc="left")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, out)


def client_train_loss(results: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=False)
    for ax, (fname, title) in zip(axes, [("federated_ann_validation_rounds.json", "Federated ANN"),
                                         ("federated_logistic_regression_validation_rounds.json", "Federated LR")]):
        hist = json.load(open(results / fname))["history"]
        rows = [(h["round"], c["client"], c["train_loss"]) for h in hist if "fit" in h for c in h["fit"]["clients"]]
        df = pd.DataFrame(rows, columns=["round", "client", "loss"])
        for client, marker in zip("ABC", "osD"):
            d = df[df["client"] == client]
            ax.plot(d["round"], d["loss"], marker=marker, markersize=4, linewidth=1.6, label=f"client {client}")
        ax.set_title(f"{title}: local training loss", loc="left")
        ax.set_xlabel("communication round"); ax.set_ylabel("loss after last local epoch"); ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out)


def confusion_matrices(results: Path, out: Path) -> None:
    metrics = load_test_metrics(results)
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.4))
    for ax, (name, m) in zip(axes.flat, metrics.items()):
        cm = np.array(m.confusion_matrix)
        share = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n({share[i, j] * 100:.2f}%)", ha="center", va="center", fontsize=10,
                        color="white" if share[i, j] > 0.5 else TEXT_PRIMARY)
        ax.set_xticks([0, 1], ["Benign", "Attack"]); ax.set_yticks([0, 1], ["Benign", "Attack"])
        ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.grid(False)
        ax.set_title(f"{name}  (threshold {m.threshold:.3f})", fontsize=10)
    fig.suptitle(f"Held-out test confusion matrices ({metrics['Centralized RF'].n_samples:,} flows; cell shares are per actual class)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, out)


# ---------------------------------------------------------------------------------------------- drawn diagrams
def _canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off"); ax.set_aspect("equal")
    return fig, ax


def _rbox(ax, x, y, w, h, edge=NAVY, fill="white", lw=1.8, r=0.08):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.0,rounding_size={r}", linewidth=lw, edgecolor=edge, facecolor=fill))


def _arrow(ax, p, q, color=GREY, lw=1.6, style="-|>", rad=0.0, ls="-", ms=14):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, linewidth=lw, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def concept_diagram(out: Path) -> None:
    fig, ax = _canvas(12, 5.0)
    ax.text(3.0, 4.75, "Centralized learning", ha="center", fontsize=14, fontweight="bold", color=RED, family=SERIF)
    ax.text(9.0, 4.75, "Federated learning", ha="center", fontsize=14, fontweight="bold", color=GREEN, family=SERIF)
    ax.plot([6, 6], [0.2, 4.6], color=GRID, linewidth=2)
    for i, y in enumerate([3.3, 2.1, 0.9]):
        label = "ABC"[i]
        _rbox(ax, 0.3, y, 1.9, 0.9, NAVY)
        ax.text(1.25, y + 0.62, f"Organization {label}", ha="center", fontsize=10.5, fontweight="bold", color=NAVY, family=SERIF)
        ax.text(1.25, y + 0.25, "raw network flows", ha="center", fontsize=9.5, family=SERIF)
        _arrow(ax, (2.25, y + 0.45), (3.55, 2.55 + (1 - i) * 0.3), RED)
    _rbox(ax, 3.6, 1.6, 2.1, 1.9, RED, "#FDF2F2")
    ax.text(4.65, 3.1, "Central server", ha="center", fontsize=11, fontweight="bold", color=RED, family=SERIF)
    ax.text(4.65, 2.6, "collects ALL raw data", ha="center", fontsize=9.5, family=SERIF)
    ax.text(4.65, 2.2, "trains one model", ha="center", fontsize=9.5, family=SERIF)
    ax.text(4.65, 1.8, "privacy risk", ha="center", fontsize=9.5, family=SERIF, style="italic", color=RED)
    for i, y in enumerate([3.3, 2.1, 0.9]):
        label = "ABC"[i]
        _rbox(ax, 6.3, y, 2.1, 0.9, NAVY)
        ax.text(7.35, y + 0.62, f"Organization {label}", ha="center", fontsize=10.5, fontweight="bold", color=NAVY, family=SERIF)
        ax.text(7.35, y + 0.25, "data stays local", ha="center", fontsize=9.5, family=SERIF)
        _arrow(ax, (8.45, y + 0.55), (9.55, 2.75 + (1 - i) * 0.25), GREEN)
        _arrow(ax, (9.55, 2.45 + (1 - i) * 0.25), (8.45, y + 0.3), NAVY, ls="--")
    _rbox(ax, 9.6, 1.6, 2.1, 1.9, GREEN, "#EEF7F2")
    ax.text(10.65, 3.1, "Server", ha="center", fontsize=11, fontweight="bold", color=GREEN, family=SERIF)
    ax.text(10.65, 2.6, "aggregates model", ha="center", fontsize=9.5, family=SERIF)
    ax.text(10.65, 2.2, "updates (FedAvg)", ha="center", fontsize=9.5, family=SERIF)
    ax.text(10.65, 1.8, "never sees raw data", ha="center", fontsize=9.5, family=SERIF, style="italic", color=GREEN)
    ax.text(3.0, 0.1, "red arrows: raw data leaves the organization", ha="center", fontsize=9, color=RED, family=SERIF)
    ax.text(9.0, 0.1, "solid: local model updates up; dashed: global model down", ha="center", fontsize=9, color=GREEN, family=SERIF)
    _save(fig, out)


CLASSES = {
    "FederatedSettings": ("", ["model: str", "ann: ANNConfig | None", "training: TrainingConfig", "source: str"], ["describe()"]),
    "ANN": ("nn.Module", ["input_dim: int", "config: ANNConfig", "blocks: ModuleList", "output: Linear"], ["forward(x)"]),
    "LogisticRegression": ("nn.Module", ["input_dim: int", "linear: Linear"], ["forward(x)"]),
    "ClientData": ("", ["client_id: str", "train, val: SplitArrays", "scaler: StandardScaler", "feature_names: list"], []),
    "IntrusionClient": ("fl.client.NumPyClient", ["client_id: str", "settings: FederatedSettings", "data: ClientData", "model: nn.Module", "layout: ParameterLayout"],
                        ["get_parameters()", "fit(parameters, config)"]),
    "ParameterLayout": ("", ["trainable: tuple", "buffers: tuple"], ["from_model(model)", "split(arrays)"]),
    "IntrusionDataset": ("torch Dataset", ["features: Tensor [N, F]", "labels: Tensor [N, 1]"], ["__getitems__(idx)", "num_features"]),
    "FedAvgWithBuffers": ("flwr FedAvg", ["layout: ParameterLayout", "round_log: list"], ["aggregate_fit(...)"]),
    "RoundEvaluator": ("", ["model, layout, loader", "manager: CheckpointManager", "history: list", "best_round, best_pr_auc"], ["__call__(round, params, cfg)"]),
    "Checkpoint": ("", ["epoch, model_state", "architecture, history", "feature_names", "scaler_state"], ["build_model()", "build_scaler()"]),
    "CheckpointManager": ("", ["directory: Path", "best_path, latest_path"], ["save_best(), save_latest()", "load_best(), load_latest()"]),
    "BinaryMetrics": ("", ["threshold, n_samples", "tp, fp, tn, fn", "accuracy, precision, recall", "f1, roc_auc, pr_auc"], ["from_counts(...)", "format_report()"]),
}
CLASS_GRID = [["ANN", "FederatedSettings", "LogisticRegression"],
              ["ClientData", "IntrusionClient", "ParameterLayout"],
              ["IntrusionDataset", "RoundEvaluator", "FedAvgWithBuffers"],
              ["Checkpoint", "CheckpointManager", "BinaryMetrics"]]
CLASS_LINKS = [("FederatedSettings", "ANN", "builds", "-"), ("FederatedSettings", "LogisticRegression", "builds", "-"),
               ("IntrusionClient", "FederatedSettings", "uses", "-"), ("IntrusionClient", "ClientData", "trains on", "-"),
               ("IntrusionClient", "ParameterLayout", "uses", "-"), ("ClientData", "IntrusionDataset", "to_dataset()", "-"),
               ("FedAvgWithBuffers", "ParameterLayout", "uses", "-"),
               ("RoundEvaluator", "CheckpointManager", "saves via", "-"), ("RoundEvaluator", "BinaryMetrics", "computes", "-"),
               ("CheckpointManager", "Checkpoint", "stores", "-")]


def class_diagram(out: Path) -> None:
    bw, gapx, gapy, lh = 2.2, 0.6, 0.55, 0.19
    heights = {n: 0.36 + lh * (len(a) + len(m)) + 0.2 + (0.18 if st else 0) for n, (st, a, m) in CLASSES.items()}
    row_h = [max(heights[n] for n in row) for row in CLASS_GRID]
    W = 3 * bw + 2 * gapx + 0.2
    H = sum(row_h) + gapy * (len(row_h) - 1) + 0.2
    fig, ax = _canvas(W, H)
    pos, y_top = {}, H - 0.1
    for r, row in enumerate(CLASS_GRID):
        for c, name in enumerate(row):
            x = 0.1 + c * (bw + gapx)
            st, attrs, methods = CLASSES[name]
            h = heights[name]
            y = y_top - h
            pos[name] = (x, y, bw, h)
            ax.add_patch(Rectangle((x, y), bw, h, linewidth=1.5, edgecolor=NAVY, facecolor="white"))
            head = 0.36 + (0.18 if st else 0)
            ax.add_patch(Rectangle((x, y + h - head), bw, head, linewidth=1.5, edgecolor=NAVY, facecolor="#E8EEF7"))
            ax.text(x + bw / 2, y + h - 0.2, name, ha="center", va="center", fontsize=9, fontweight="bold", color=NAVY, family=SERIF)
            if st: ax.text(x + bw / 2, y + h - 0.4, f"<<{st}>>", ha="center", va="center", fontsize=7, style="italic", color=GREY, family=SERIF)
            ty = y + h - head - 0.14
            for a in attrs:
                ax.text(x + 0.07, ty, a, ha="left", va="center", fontsize=7.2, family="monospace"); ty -= lh
            sep = ty + lh / 2 - 0.02
            ax.plot([x, x + bw], [sep, sep], color=NAVY, linewidth=1)
            ty -= 0.04
            for m in methods:
                ax.text(x + 0.07, ty, m, ha="left", va="center", fontsize=7.2, family="monospace"); ty -= lh
        y_top -= row_h[r] + gapy

    def edge_point(name, other):
        x, y, w, h = pos[name]
        ox, oy, ow, oh = pos[other]
        cx, cy, ocx, ocy = x + w / 2, y + h / 2, ox + ow / 2, oy + oh / 2
        if abs(ocy - cy) > abs(ocx - cx) * 0.6 and abs(ocy - cy) > (h + oh) / 2 * 0.9:   # vertical neighbour
            return (cx + (0.25 if ocx > cx else -0.25 if ocx < cx else 0), y if ocy < cy else y + h)
        return (x + w if ocx > cx else x, cy)

    for a, b, label, ls in CLASS_LINKS:
        p, q = edge_point(a, b), edge_point(b, a)
        _arrow(ax, p, q, NAVY, lw=1.2, ls=ls, ms=11)
        if abs(q[1] - p[1]) > 0.3:   # label only the vertical / diagonal links (horizontal gaps are too narrow)
            ax.text((p[0] + q[0]) / 2 + 0.06, (p[1] + q[1]) / 2, label, ha="left", va="center", fontsize=6.8, style="italic", color=GREY, family=SERIF,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})
    _save(fig, out)


def activity_diagram(out: Path) -> None:
    fig, ax = _canvas(7.6, 10.6)
    cx = 3.8

    def box(x, y, text, w=4.3, h=0.55, edge=NAVY, fill="white", size=8.8):
        _rbox(ax, x - w / 2, y - h / 2, w, h, edge, fill, r=0.14)
        ax.text(x, y, text, ha="center", va="center", fontsize=size, family=SERIF)

    def arr(x1, y1, x2, y2, label=None, rad=0.0, ls="-"):
        _arrow(ax, (x1, y1), (x2, y2), GREY, lw=1.3, ms=11, rad=rad, ls=ls)
        if label:
            ax.text((x1 + x2) / 2 + 0.1, (y1 + y2) / 2, label, fontsize=8, style="italic", color=GREY, family=SERIF)

    def bar(x1, x2, y):
        ax.add_patch(Rectangle((x1, y - 0.04), x2 - x1, 0.08, color="black"))

    ax.add_patch(Circle((cx, 10.35), 0.13, color="black"))
    arr(cx, 10.22, cx, 9.97)
    box(cx, 9.65, "Load config; preprocess: merge 8 files,\ndrop duplicates, binary label", w=4.6, h=0.65)
    arr(cx, 9.325, cx, 9.13)
    box(cx, 8.85, "Stratified 80/20 split (only if missing): train / test", w=4.8)
    arr(cx, 8.575, cx, 8.34)
    bar(1.2, 6.4, 8.3)
    lx, rx = 1.9, 5.3
    arr(lx, 8.26, lx, 8.0); arr(rx, 8.26, rx, 8.0)
    box(lx, 7.6, "Train Random Forest on all\ntrain rows; evaluate once\non the test set", w=3.0, h=0.8, size=8.4)
    box(rx, 7.6, "Partition train into clients\nA, B, C; per client 90% train /\n10% validation, own scaler", w=3.0, h=0.8, size=8.4)
    arr(rx, 7.2, rx, 6.87)
    box(rx, 6.45, "Optuna search (40 trials); retrain\nthe best trial; evaluate once\non the test set", w=3.0, h=0.8, size=8.4)
    arr(lx, 7.2, lx, 5.6)
    arr(rx, 6.05, rx, 5.6)
    bar(1.2, 6.4, 5.56)
    arr(cx, 5.52, cx, 5.3)

    ax.add_patch(FancyBboxPatch((0.35, 2.1), 6.9, 3.15, boxstyle="round,pad=0,rounding_size=0.12", linewidth=1.5, edgecolor=GREEN, facecolor="none", linestyle="--"))
    ax.text(0.5, 5.07, "for each federated model (Logistic Regression, ANN): rounds 1 to 15", fontsize=8.2, style="italic", color=GREEN, family=SERIF, va="center")
    box(cx, 4.65, "Server broadcasts the global weights", w=3.8, h=0.45, size=8.4)
    arr(cx, 4.425, cx, 4.19)
    bar(1.0, 6.6, 4.15)
    xs = (1.9, 3.8, 5.7)
    for x, n in zip(xs, "ABC"):
        arr(x, 4.11, x, 3.9)
        box(x, 3.55, f"Client {n}: 3 local\nepochs on own data", w=1.7, h=0.65, size=7.8)
        arr(x, 3.225, x, 3.04)
    bar(1.0, 6.6, 3.0)
    arr(cx, 2.96, cx, 2.85)
    box(cx, 2.55, "FedAvg aggregation; validate on the combined\nvalidation set; save best.pt / latest.pt", w=4.9, h=0.6, size=8.2)
    arr(6.25, 2.55, 6.95, 2.55); arr(6.95, 2.55, 6.95, 4.65); arr(6.95, 4.65, 5.7, 4.65)
    ax.text(7.12, 3.6, "next round", fontsize=7.6, style="italic", color=GREY, family=SERIF, ha="center", va="center", rotation=90)
    arr(cx, 2.1, cx, 1.78)
    box(cx, 1.45, "Select threshold on validation (max attack-F1);\nevaluate the best checkpoint once on the test set", w=5.4, h=0.65, size=8.4)
    arr(cx, 1.125, cx, 0.95)
    box(cx, 0.68, "Compare all models: tables and plots", w=4.0, h=0.5, size=8.6)
    arr(cx, 0.43, cx, 0.3)
    ax.add_patch(Circle((cx, 0.17), 0.13, facecolor="white", edgecolor="black", linewidth=1.6))
    ax.add_patch(Circle((cx, 0.17), 0.065, color="black"))
    _save(fig, out)


def use_case_diagram(out: Path) -> None:
    fig, ax = _canvas(9.0, 7.0)
    ax.add_patch(Rectangle((2.2, 0.2), 4.9, 6.6, linewidth=1.8, edgecolor=NAVY, facecolor="none"))
    ax.text(4.65, 6.55, "Federated malware-detection framework", ha="center", fontsize=11, fontweight="bold", color=NAVY, family=SERIF)
    cases = {"run": (4.65, 5.85, "Run the whole pipeline"), "prep": (4.65, 5.1, "Preprocess and split data"),
             "cent": (4.65, 4.35, "Train centralized baselines"), "tune": (4.65, 3.6, "Tune hyperparameters (Optuna)"),
             "local": (4.65, 2.85, "Train locally on client data"), "agg": (4.65, 2.1, "Aggregate updates (FedAvg)"),
             "val": (4.65, 1.35, "Validate global model each round"), "test": (4.65, 0.6, "Evaluate on test set once; report")}
    for k, (x, y, t) in cases.items():
        ax.add_patch(Ellipse((x, y), 4.3, 0.58, linewidth=1.5, edgecolor=NAVY, facecolor="#F4F7FB"))
        ax.text(x, y, t, ha="center", va="center", fontsize=8.8, family=SERIF)

    def actor(x, y, label):
        ax.add_patch(Circle((x, y + 0.5), 0.17, fill=False, linewidth=1.6, edgecolor=NAVY))
        ax.plot([x, x], [y + 0.33, y - 0.15], color=NAVY, linewidth=1.6)
        ax.plot([x - 0.3, x + 0.3], [y + 0.2, y + 0.2], color=NAVY, linewidth=1.6)
        ax.plot([x, x - 0.25], [y - 0.15, y - 0.55], color=NAVY, linewidth=1.6); ax.plot([x, x + 0.25], [y - 0.15, y - 0.55], color=NAVY, linewidth=1.6)
        ax.text(x, y - 0.85, label, ha="center", va="top", fontsize=9, fontweight="bold", family=SERIF)

    actor(1.0, 3.9, "Researcher /\nsecurity analyst")
    actor(8.0, 3.7, "Client organization\n(simulated A, B, C)")
    actor(8.0, 1.3, "Flower server")
    for k in ["run", "prep", "cent", "tune", "test"]:
        x, y, _ = cases[k]; _arrow(ax, (1.35, 3.9 + 0.2), (x - 2.15, y), GREY, lw=1.1, style="-")
    for k in ["local"]:
        x, y, _ = cases[k]; _arrow(ax, (7.7, 3.7 + 0.2), (x + 2.15, y), GREY, lw=1.1, style="-")
    for k, ay in [("agg", 1.3 + 0.2), ("val", 1.3 + 0.2)]:
        x, y, _ = cases[k]; _arrow(ax, (7.7, ay), (x + 2.15, y), GREY, lw=1.1, style="-")
    _arrow(ax, (4.65, 5.56), (4.65, 5.39), GREY, lw=1.0, ls="--", ms=9)
    ax.text(4.75, 5.47, "<<include>>", fontsize=7, style="italic", color=GREY, family=SERIF)
    _save(fig, out)


def gantt(out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.2), gridspec_kw={"height_ratios": [1.0, 1.55]})
    D = lambda s: datetime.strptime(s, "%Y-%m-%d %H:%M")
    ax = axes[0]
    tasks = [("Literature survey, requirements,\ndesign (Project Work 1)", "2026-06-01 00:00", "2026-06-30 23:59", NAVY),
             ("Preprocessing, EDA, RF baseline,\nclient partition", "2026-08-26 00:00", "2026-08-26 23:59", FAMILY_COLOR["RF"]),
             ("Framework phases 1-8: training,\ntuning, federated, tests", "2026-10-11 00:00", "2026-10-11 23:59", FAMILY_COLOR["ANN"]),
             ("Review-2 presentation,\nrunner, report", "2026-10-30 00:00", "2026-11-15 00:00", GREEN)]
    for i, (name, s, e, c) in enumerate(tasks):
        ax.barh(i, (D(e) - D(s)).total_seconds() / 86400 + 0.0, left=mdates.date2num(D(s)), height=0.5, color=c, edgecolor="white")
    ax.set_yticks(range(len(tasks)), [t[0] for t in tasks], fontsize=9)
    ax.set_ylim(-0.5, len(tasks) - 0.5); ax.invert_yaxis()
    ax.xaxis_date(); ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_xlim(mdates.date2num(D("2026-06-01 00:00")), mdates.date2num(D("2026-11-30 00:00")))
    ax.set_title("(a) Project timeline, June to November 2026", loc="left", fontsize=10)
    ax = axes[1]
    phases = [("Phases 1-2: model files, dataset", "2026-10-11 14:55", "2026-10-11 15:04"),
              ("Phases 3-4: ANN, config, training", "2026-10-11 15:04", "2026-10-11 15:11"),
              ("Phase 5: evaluation, thresholds", "2026-10-11 15:11", "2026-10-11 15:18"),
              ("Phase 6: Optuna search code", "2026-10-11 15:18", "2026-10-11 15:28"),
              ("Optuna study (40 trials)", "2026-10-11 15:29", "2026-10-11 16:16"),
              ("Final retrain, single test run", "2026-10-11 16:17", "2026-10-11 16:23"),
              ("Phase 7: federated code", "2026-10-11 16:23", "2026-10-11 18:50"),
              ("Federated runs (LR, ANN)", "2026-10-11 18:50", "2026-10-11 18:58"),
              ("Local-epoch run, test evaluations", "2026-10-11 18:58", "2026-10-11 19:03"),
              ("Phase 8: config, reports, tests", "2026-10-11 19:03", "2026-10-11 19:19")]
    palette = [FAMILY_COLOR["ANN"], FAMILY_COLOR["ANN"], FAMILY_COLOR["ANN"], FAMILY_COLOR["ANN"], NAVY, NAVY, FAMILY_COLOR["LR"], FAMILY_COLOR["LR"], FAMILY_COLOR["LR"], GREY]
    for i, ((name, s, e), c) in enumerate(zip(phases, palette)):
        ax.barh(i, (D(e) - D(s)).total_seconds() / 86400, left=mdates.date2num(D(s)), height=0.55, color=c)
    ax.set_yticks(range(len(phases)), [p[0] for p in phases], fontsize=9); ax.invert_yaxis()
    ax.xaxis_date(); ax.xaxis.set_major_locator(mdates.HourLocator(interval=1)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlim(mdates.date2num(D("2026-10-11 14:30")), mdates.date2num(D("2026-10-11 19:30")))
    ax.set_title("(b) Implementation day, 11 October 2026", loc="left", fontsize=10)
    ax.set_xlabel("time of day")
    fig.tight_layout()
    _save(fig, out)


# ---------------------------------------------------------------------------------------------- terminal snapshots
def snapshot(lines: list[str], title: str, out: Path, width_chars: int = 100, font: float = 7.8) -> None:
    wrapped = []
    for ln in lines:
        ln = ln.rstrip().replace("\t", "    ")
        wrapped += textwrap.wrap(ln, width_chars, subsequent_indent="    ", drop_whitespace=False, break_long_words=True) or [""]
    cw = font * 0.6 / 72                    # inches per character (monospace)
    lh = font * 1.45 / 72
    w, h = width_chars * cw + 0.5, len(wrapped) * lh + 0.85
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.02, 0.02), w - 0.04, h - 0.04, boxstyle="round,pad=0,rounding_size=0.08", facecolor="#F7F7F7", edgecolor="#BDBDBD", linewidth=1.2))
    ax.add_patch(Rectangle((0.02, h - 0.42), w - 0.04, 0.4, facecolor="#E4E4E4", edgecolor="#BDBDBD", linewidth=1.0))
    for i, c in enumerate(["#E5534B", "#E5A93C", "#4CAF50"]):
        ax.add_patch(Circle((0.25 + i * 0.2, h - 0.22), 0.065, color=c))
    ax.text(w / 2, h - 0.22, title, ha="center", va="center", fontsize=8.5, color="#333333", family="DejaVu Sans")
    for i, ln in enumerate(wrapped):
        ax.text(0.22, h - 0.62 - i * lh, ln, ha="left", va="center", fontsize=font, family="DejaVu Sans Mono", color="#1A1A1A")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def _ansi(s: str) -> str:
    import re
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _read(path: Path) -> list[str]:
    return [_ansi(l.rstrip("\n")) for l in open(path, errors="replace")]


def snapshots(results: Path, out_dir: Path, project_root: Path) -> None:
    def pick(lines, start_pat, n):
        i = next(k for k, l in enumerate(lines) if start_pat in l)
        return lines[i:i + n]

    strip_root = lambda l: l.replace(str(project_root) + "/", "").replace(str(results) + "/", "results/")
    cent = _read(results / "optuna" / "ann_centralized_search.log")
    trials = [l for l in cent if l.startswith("[trial")][:5] + ["..."] + [l for l in cent if l.startswith("[trial")][-3:] + [l for l in cent if l.startswith(("Study finished", "Best trial", "Saved"))]
    snapshot([strip_root(l)[:200] for l in trials], "Optuna search (results/optuna/ann_centralized_search.log)", out_dir / "snap_optuna_search.png", 104)
    final = _read(results / "optuna" / "ann_centralized_final.log")
    snapshot([strip_root(l) for l in final if l.strip()][:46], "Centralized ANN: final retrain and single test evaluation", out_dir / "snap_centralized_final.png")
    fed = _read(results / "federated_ann_train.log")
    keep = [l for l in fed if l.startswith(("Model:", "Rounds:", "Identical", "[round", "Done in"))]
    snapshot([strip_root(l) for l in keep], "Federated ANN training (results/federated_ann_train.log)", out_dir / "snap_federated_ann.png", 119, 7.2)
    lr = _read(results / "federated_logistic_regression_train.log")
    keep = [l for l in lr if l.startswith(("Model:", "Rounds:", "Identical", "[round", "Done in"))]
    snapshot([strip_root(l) for l in keep], "Federated Logistic Regression training", out_dir / "snap_federated_lr.png", 119, 7.2)
    for name, f in [("ann", "ann_federated_test_eval.log"), ("lr", "lr_federated_test_eval.log")]:
        snapshot([strip_root(l) for l in _read(results / f) if l.strip()], f"Final test evaluation: federated {name.upper()} (results/{f})", out_dir / f"snap_eval_{name}.png")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Regenerate the report figures")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    results, graphs = cfg.paths.results_dir, cfg.paths.graphs_dir
    _style()
    training_curves(results, graphs / "centralized_ann_training_curves.png")
    optuna_history(results, graphs / "optuna_history.png")
    local_epochs(results, graphs / "local_epochs_comparison.png")
    client_train_loss(results, graphs / "client_train_loss.png")
    confusion_matrices(results, graphs / "confusion_matrices.png")
    concept_diagram(graphs / "concept_centralized_vs_federated.png")
    class_diagram(graphs / "class_diagram.png")
    activity_diagram(graphs / "activity_diagram.png")
    use_case_diagram(graphs / "use_case_diagram.png")
    gantt(graphs / "gantt_chart.png")
    snapshots(results, graphs / "snapshots", cfg.source_path.parent)


if __name__ == "__main__":
    main()
