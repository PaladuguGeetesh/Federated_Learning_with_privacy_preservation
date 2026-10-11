"""
Loaders for the result files the reporting scripts read. Nothing is evaluated
here: every number comes from a file written by a one-time test evaluation or
a training run, and metrics are rebuilt through core.evaluate.BinaryMetrics so
no script carries its own copy of the metric arithmetic.
"""

import json
import math
from pathlib import Path

import pandas as pd

from core.evaluate import BinaryMetrics

# Display name -> held-out test result file (one-time evaluations).
TEST_RESULT_FILES = {
    "Centralized RF": "centralized_rf_metrics.json",
    "Centralized ANN (tuned)": "ann_centralized_metrics.json",
    "Federated LR": "lr_federated_test_metrics.json",
    "Federated ANN": "ann_federated_test_metrics.json",
}
# Display name -> per-round federated validation file (federated/server.py).
ROUND_FILES = {
    "Federated LR": "federated_logistic_regression_validation_rounds.json",
    "Federated ANN": "federated_ann_validation_rounds.json",
}
CENTRALIZED_ANN = "Centralized ANN (tuned)"


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run the step that produces it first (see README)")
    with open(path) as f:
        return json.load(f)


def _rf_metrics(payload: dict, n_attack: int, n_benign: int) -> BinaryMetrics:
    """RF test metrics. Newer train_rf.py output has a full test_metrics block; the
    earlier format stored only five metrics at threshold 0.5, so its confusion counts
    are recovered exactly from recall/precision and the test set's class counts (and
    checked against its stored accuracy). PR-AUC was not recorded in that format."""
    if "test_metrics" in payload:
        return BinaryMetrics.from_dict(payload["test_metrics"])
    m = payload["metrics"]
    tp = round(m["recall"] * n_attack)
    fp = round(tp / m["precision"] - tp)
    fn, tn = n_attack - tp, n_benign - fp
    if not math.isclose((tp + tn) / (n_attack + n_benign), m["accuracy"], abs_tol=1e-9):
        raise ValueError("Recovered RF confusion counts do not reproduce its stored accuracy")
    return BinaryMetrics.from_counts(tp=tp, fp=fp, tn=tn, fn=fn, threshold=0.5,
                                     roc_auc=m["roc_auc"], pr_auc=float("nan"))


def load_test_metrics(results_dir: Path) -> dict[str, BinaryMetrics]:
    """Held-out test metrics for every model, in display order. All must share one test set."""
    others = {name: BinaryMetrics.from_dict(_load(results_dir / f)["test_metrics"])
              for name, f in TEST_RESULT_FILES.items() if name != "Centralized RF"}
    sizes = {(m.n_attack, m.n_benign) for m in others.values()}
    if len(sizes) != 1:
        raise ValueError(f"Results were computed on different test sets: {sizes}")
    n_attack, n_benign = sizes.pop()
    rf = _rf_metrics(_load(results_dir / TEST_RESULT_FILES["Centralized RF"]), n_attack, n_benign)
    if (rf.n_attack, rf.n_benign) != (n_attack, n_benign):
        raise ValueError("RF was evaluated on a different test set than the other models")
    return {"Centralized RF": rf, **others}


def load_test_payload(results_dir: Path, name: str) -> dict:
    """Raw JSON of one model's test result (threshold selection details etc.)."""
    return _load(results_dir / TEST_RESULT_FILES[name])


def load_round_histories(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Per-round federated validation history (round 0 = untrained initial model)."""
    out = {}
    for name, f in ROUND_FILES.items():
        payload = _load(results_dir / f)
        df = pd.DataFrame(payload["history"]).drop(columns=["fit"], errors="ignore")
        df.attrs.update(best_round=payload["best_round_by_val_pr_auc"], rounds=payload["rounds"],
                        local_epochs=payload["local_epochs"], wall_clock_seconds=payload["wall_clock_seconds"])
        out[name] = df.sort_values("round").reset_index(drop=True)
    return out


def centralized_ann_best_val_pr_auc(results_dir: Path) -> float:
    """Validation PR-AUC of the centralized tuned ANN at its best epoch — same validation rows,
    same (pooled = combined) scaling as the federated per-round validation."""
    return float(_load(results_dir / TEST_RESULT_FILES[CENTRALIZED_ANN])["retrain"]["best_val_pr_auc"])
