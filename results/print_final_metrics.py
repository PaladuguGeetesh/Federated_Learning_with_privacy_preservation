"""
Prints each model's held-out test result using core.evaluate's report format,
followed by a compact side-by-side table. Read-only.

Run from the project root:
    python -m results.print_final_metrics
"""

import argparse
from pathlib import Path

from core.config import DEFAULT_CONFIG_PATH, load_config
from results.sources import load_test_metrics, load_test_payload

TABLE_COLUMNS = ["accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc", "pr_auc", "fpr"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print the held-out test results")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)
    results_dir = load_config(args.config).paths.results_dir

    metrics = load_test_metrics(results_dir)
    for name, m in metrics.items():
        payload = load_test_payload(results_dir, name)
        selection = payload.get("threshold_selection")
        how = (f"threshold selected on validation ({selection['strategy']})" if selection
               else "fixed default threshold")
        print(m.format_report(f"{name} — held-out test, {how}"))
        print()

    width = 26
    header = f"{'Model':<{width}}" + "".join(f"{c:>11}" for c in TABLE_COLUMNS) + f"{'missed':>9}{'false+':>9}"
    print(header)
    print("-" * len(header))
    for name, m in metrics.items():
        values = "".join(f"{getattr(m, c):>11.4f}" for c in TABLE_COLUMNS)
        print(f"{name:<{width}}{values}{m.fn:>9,}{m.fp:>9,}")
    print("\n(RF pr_auc is nan: its original test evaluation did not record PR-AUC.)")


if __name__ == "__main__":
    main()
