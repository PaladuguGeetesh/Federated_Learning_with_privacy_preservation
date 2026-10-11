"""
Round-by-round convergence summary of the federated runs (VALIDATION data only).

For each federated model: the per-round validation table written by
federated/server.py, plus milestones (first round reaching given validation
PR-AUC levels) and the gap to the centralized tuned ANN's best validation
PR-AUC — measured on the same validation rows with the same scaling, so the
gap is like-for-like. Threshold-dependent columns use the default 0.5
threshold that the server scores each round with. Read-only.

Run from the project root:
    python -m results.print_convergence
"""

import argparse
from pathlib import Path

from core.config import DEFAULT_CONFIG_PATH, load_config
from results.sources import centralized_ann_best_val_pr_auc, load_round_histories

TABLE_COLUMNS = ["round", "val_loss", "val_pr_auc", "val_macro_f1", "val_attack_recall", "val_false_negatives", "val_fpr"]
PR_AUC_MILESTONES = [0.95, 0.99, 0.995]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print federated validation convergence")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)
    results_dir = load_config(args.config).paths.results_dir

    reference = centralized_ann_best_val_pr_auc(results_dir)
    print(f"Centralized tuned ANN, best validation PR-AUC: {reference:.5f}\n")

    for name, df in load_round_histories(results_dir).items():
        title = (f"{name} — {df.attrs['rounds']} rounds x {df.attrs['local_epochs']} local epochs "
                 f"({df.attrs['wall_clock_seconds']:.0f}s), validation set")
        print(title)
        print("-" * len(title))
        print(df[TABLE_COLUMNS].to_string(index=False, float_format=lambda v: f"{v:.5f}"))
        trained = df[df["round"] > 0]
        for level in PR_AUC_MILESTONES:
            reached = trained[trained["val_pr_auc"] >= level]
            where = f"round {int(reached.iloc[0]['round'])}" if len(reached) else "never"
            print(f"  First round with val PR-AUC >= {level}: {where}")
        best = trained.loc[trained["val_pr_auc"].idxmax()]
        print(f"  Best round (val PR-AUC): {int(best['round'])} — {best['val_pr_auc']:.5f}; "
              f"gap to centralized ANN: {(reference - best['val_pr_auc']) * 100:+.3f} pp\n")


if __name__ == "__main__":
    main()
