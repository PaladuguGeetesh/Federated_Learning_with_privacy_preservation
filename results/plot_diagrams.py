"""
Design diagrams for the project (no model is trained or evaluated here):

  system_architecture.png   clients A/B/C, FedAvg server, validation and the one-time test evaluation
  data_pipeline.png         raw files -> cleaned data -> train/test split -> client partitions -> train/validation
  federated_round_flow.png  the five steps of one federated communication round

The row counts and settings drawn here are read from the dataset files' metadata and from config.yaml,
not typed in, so the figures stay in sync with the project.

Run from the project root:
    python -m results.plot_diagrams
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from core.config import DEFAULT_CONFIG_PATH, load_config
from core.dataset import CLIENT_IDS, TEST_FILENAME

RED, NAVY, GREY, LIGHT, GREEN = "#C00000", "#1F3864", "#595959", "#F2F2F2", "#2E7D5B"
RAW_FILENAMES = [
    "Benign-Monday-no-metadata.parquet", "Bruteforce-Tuesday-no-metadata.parquet",
    "DoS-Wednesday-no-metadata.parquet", "WebAttacks-Thursday-no-metadata.parquet",
    "Infiltration-Thursday-no-metadata.parquet", "Botnet-Friday-no-metadata.parquet",
    "Portscan-Friday-no-metadata.parquet", "DDoS-Friday-no-metadata.parquet",
]


def _rows(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def _box(ax, x, y, w, h, title, lines=(), edge=NAVY, fill="white", title_color=None, size=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", linewidth=2,
                                edgecolor=edge, facecolor=fill))
    top = y + h - 0.28
    ax.text(x + w / 2, top, title, ha="center", va="center", fontsize=size + 1, fontweight="bold",
            color=title_color or edge, family="serif")
    for i, line in enumerate(lines):
        ax.text(x + w / 2, top - 0.42 - 0.34 * i, line, ha="center", va="center", fontsize=size - 1,
                color="#222222", family="serif")


def _arrow(ax, p, q, color=GREY, label=None, label_dy=0.2, rad=0.0, lw=2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=18, linewidth=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + label_dy, label, ha="center", va="center",
                fontsize=9.5, color=color, family="serif", style="italic")


def _canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def system_architecture(cfg, sizes: dict[str, int], n_val: int, n_test: int, path: Path) -> None:
    fig, ax = _canvas(13, 7.0)
    for i, cid in enumerate(CLIENT_IDS):
        y = 4.45 - i * 2.05
        _box(ax, 0.3, y, 3.9, 1.7, f"Organization {cid}",
             [f"local data: {sizes[cid]:,} flows", "own scaler + local training",
              f"{cfg.federated.local_epochs} local epochs / round"], edge=NAVY)
        up, down = 4.15 - i * 0.7, 3.85 - i * 0.7  # distinct landing points on the server's left edge
        _arrow(ax, (4.25, y + 1.05), (6.05, up), NAVY)
        _arrow(ax, (6.05, down), (4.25, y + 0.6), RED)
    ax.text(7.6, 1.6, "blue arrows: weights + sample count n_k (to server)", ha="center", fontsize=10, color=NAVY, family="serif")
    ax.text(7.6, 1.2, "red arrows: global model (to organizations)", ha="center", fontsize=10, color=RED, family="serif")
    _box(ax, 6.1, 2.3, 3.0, 2.5, "Server", ["FedAvg", "(sample-weighted)", "receives weights only"], edge=RED, fill="#FDF2F2")
    _arrow(ax, (9.15, 3.9), (10.1, 4.9))
    _arrow(ax, (9.15, 3.2), (10.1, 2.0))
    _box(ax, 10.1, 4.2, 2.7, 1.75, "Validation", [f"{n_val:,} flows", "every round", "PR-AUC -> best.pt"], edge=GREEN)
    _box(ax, 10.1, 0.9, 2.7, 1.75, "Final test (once)", [f"{n_test:,} flows", "threshold from validation"], edge=GREY)
    ax.text(6.5, 6.85, "Federated learning system: training data stays at each organization", ha="center",
            fontsize=13, fontweight="bold", color=RED, family="serif")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def data_pipeline(cfg, counts: dict[str, int], path: Path) -> None:
    fig, ax = _canvas(13, 5.4)
    steps = [
        ("8 raw files", [f"{counts['raw']:,} flows", "CICIDS2017", "(dhoogla cleaned)"], NAVY),
        ("Cleaned data", [f"{counts['cleaned']:,} flows", f"{counts['raw'] - counts['cleaned']:,} duplicates removed",
                          "binary label"], NAVY),
        ("Stratified split", [f"seed {cfg.seed}", f"{int((1 - cfg.data.test_size) * 100)}% train / "
                                                   f"{int(cfg.data.test_size * 100)}% test"], NAVY),
    ]
    x = 0.2
    for i, (title, lines, color) in enumerate(steps):
        _box(ax, x, 3.0, 2.8, 1.9, title, lines, edge=color)
        if i < len(steps) - 1:
            _arrow(ax, (x + 2.85, 3.95), (x + 3.35, 3.95))
        x += 3.35
    _box(ax, 10.25, 3.7, 2.55, 1.3, "Test set", [f"{counts['test']:,}", "held-out"], edge=GREY)
    _box(ax, 10.25, 2.0, 2.55, 1.3, "Train set", [f"{counts['train']:,}", "IID 3-way split"], edge=RED)
    _arrow(ax, (9.35, 4.15), (10.2, 4.35))
    _arrow(ax, (9.35, 3.75), (10.2, 2.75))
    ax.plot([11.5, 11.5, 0.5 + 1.45], [2.0, 1.85, 1.85], color=GREY, linewidth=1.6)
    ax.plot([0.5 + 1.45, 0.5 + 2 * 3.2 + 1.45], [1.85, 1.85], color=GREY, linewidth=1.6)
    for i, cid in enumerate(CLIENT_IDS):
        xi = 0.5 + i * 3.2
        _box(ax, xi, 0.1, 2.9, 1.5, f"Client {cid}", [f"{counts[cid]:,} flows", "~90% train / ~10% validation"],
             edge=NAVY, size=10)
        _arrow(ax, (xi + 1.45, 1.85), (xi + 1.45, 1.65), GREY, lw=1.6)
    ax.text(0.2, 5.3, "Data pipeline and splits", fontsize=13, fontweight="bold", color=RED, family="serif")
    ax.text(10.2, 0.8, f"validation slices\ncombined: {counts['val']:,}", fontsize=10, color=GREY, family="serif")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def round_flow(cfg, path: Path) -> None:
    fig, ax = _canvas(13, 3.5)
    steps = [
        ("1  Broadcast", ["server sends the", "global weights"]),
        ("2  Local training", [f"{cfg.federated.local_epochs} epochs on own data", "AdamW, grad clipping"]),
        ("3  Upload", ["updated weights", "+ sample count n_k"]),
        ("4  Aggregate", ["average of weights,", "weighted by n_k"]),
        ("5  Validate", ["score on validation set", "save best / latest"]),
    ]
    w, gap = 2.2, 0.35
    for i, (title, lines) in enumerate(steps):
        x = 0.2 + i * (w + gap)
        _box(ax, x, 1.4, w, 1.9, title, lines, edge=RED if i == 3 else NAVY, size=10.5)
        if i < len(steps) - 1:
            _arrow(ax, (x + w + 0.02, 2.35), (x + w + gap - 0.02, 2.35))
    _arrow(ax, (11.5, 1.35), (1.3, 1.35), GREY, f"repeat for {cfg.federated.rounds} communication rounds", -0.45, rad=-0.18, lw=1.4)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Regenerate the design diagrams")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    d, graphs = cfg.data.dataset_dir, cfg.paths.graphs_dir
    graphs.mkdir(parents=True, exist_ok=True)

    sizes = {cid: _rows(d / f"client_{cid}.parquet") for cid in CLIENT_IDS}
    counts = {"raw": sum(_rows(d / "raw" / f) for f in RAW_FILENAMES), "cleaned": _rows(d / "cleaned_data.parquet"),
              "train": _rows(d / "train.parquet"), "test": _rows(d / TEST_FILENAME), **sizes}
    # Validation rows = what core.dataset's stratified split takes from each client file.
    from core.dataset import load_raw_validation
    n_val = len(load_raw_validation(d, cfg.data.val_fraction, cfg.seed)[1])
    counts["val"] = n_val

    system_architecture(cfg, sizes, n_val, counts["test"], graphs / "system_architecture.png")
    data_pipeline(cfg, counts, graphs / "data_pipeline.png")
    round_flow(cfg, graphs / "federated_round_flow.png")
    print(f"Saved diagrams to {graphs}")


if __name__ == "__main__":
    main()
