"""
Model-agnostic evaluation for binary intrusion detection (0 = Benign, 1 = Attack).

Three layers, kept deliberately separate:

1. Scoring — turn any model into P(attack) per row:
       predict_proba()          torch nn.Module (ANN / LogisticRegression) or any
                                sklearn-style model with predict_proba (Random Forest)
       predict_proba_loader()   torch model over a DataLoader (returns labels too)
   Torch models run in eval mode under torch.no_grad(); the sigmoid is applied
   only here, when turning logits into probabilities.

2. Metrics — compute_metrics(y_true, y_prob, threshold) works from labels and
   probabilities only, so it is identical for RF, LR and the ANN.

3. Thresholds and the test set:
       select_threshold()   validation arrays in -> threshold out. Takes plain
                            arrays only, never paths/loaders, so it cannot read
                            the test set.
       evaluate_on_test()   requires an already-chosen threshold and an
                            already-fitted scaler (or an explicit None for the
                            unscaled RF). It never selects a threshold and never
                            fits anything. The Optuna objective never calls it;
                            only the single final step after the study does.

Final test evaluation of a checkpoint (run once, after all tuning is complete):
    python -m core.evaluate --checkpoint models/ann_centralized/best.pt --name ann_centralized --final-test
"""

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.exceptions import NotFittedError
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.utils.data import DataLoader

from core.checkpoint import load_checkpoint
from core.config import DEFAULT_CONFIG_PATH, EvaluationConfig, load_config
from core.dataset import apply_scaler, get_device, load_raw_validation, load_test_set, to_device

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ---------------------------------------------------------------------------
# 1. Scoring
# ---------------------------------------------------------------------------


def _sigmoid_float64(logits: torch.Tensor) -> np.ndarray:
    """Logits -> probabilities in float64 (float32 sigmoid rounds to exactly 1.0 for
    logits above ~17, which would create artificial ties among confident attacks).
    Moved to CPU first: MPS has no float64 support."""
    return torch.sigmoid(logits.detach().cpu().double()).numpy().reshape(-1)


def predict_proba(
    model: Any, X: np.ndarray, *, batch_size: int, device: torch.device | None = None
) -> np.ndarray:
    """P(attack) for each row of X, for a torch model or an sklearn-style classifier.

    Args:
        model: nn.Module returning one logit per row, or an object with predict_proba().
        X: Features, already scaled the way the model was trained.
        batch_size: Rows per forward pass (torch models only).
        device: Device a torch model lives on (defaults to the model's own device).
    """
    if isinstance(model, nn.Module):
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if device is None:
            device = next(model.parameters()).device
        model.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                xb = torch.as_tensor(X[start : start + batch_size], dtype=torch.float32, device=device)
                chunks.append(model(xb))
        if not chunks:
            raise ValueError("X has no rows")
        return _sigmoid_float64(torch.cat(chunks))

    if hasattr(model, "predict_proba"):
        classes = list(getattr(model, "classes_", [0, 1]))
        if 1 not in classes:
            raise ValueError(f"Model classes {classes} do not include the attack class 1")
        return np.asarray(model.predict_proba(X))[:, classes.index(1)].astype(np.float64)

    raise TypeError(f"Cannot score model of type {type(model).__name__}: not an nn.Module and no predict_proba()")


def predict_proba_loader(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """(y_true as int64, P(attack) as float64) over a DataLoader, in eval mode under no_grad."""
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for batch in loader:
            features, y = to_device(batch, device)
            logits.append(model(features))
            labels.append(y)
    if not logits:
        raise ValueError("Loader produced no batches")
    y_true = torch.cat(labels).cpu().numpy().reshape(-1).astype(np.int64)
    return y_true, _sigmoid_float64(torch.cat(logits))


# ---------------------------------------------------------------------------
# 2. Metrics
# ---------------------------------------------------------------------------


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


@dataclass(frozen=True)
class BinaryMetrics:
    """Metrics at one decision threshold. Positive class = Attack (1)."""

    threshold: float
    n_samples: int
    n_attack: int  # actual attacks = tp + fn
    n_benign: int  # actual benign  = tn + fp
    tp: int
    fp: int
    tn: int
    fn: int  # false negatives = missed attacks
    accuracy: float
    precision: float  # attack-class precision
    recall: float  # attack-class recall (detection rate)
    f1: float  # attack-class F1
    macro_f1: float  # mean of benign F1 and attack F1
    specificity: float  # benign recall
    fpr: float  # false-positive rate = fp / (fp + tn)
    fnr: float  # false-negative rate = fn / (fn + tp)
    roc_auc: float  # threshold-free; NaN if only one class present
    pr_auc: float  # threshold-free average precision for the attack class; NaN if no attacks

    @property
    def attack_recall(self) -> float:
        return self.recall

    @property
    def false_negatives(self) -> int:
        return self.fn

    @property
    def confusion_matrix(self) -> list[list[int]]:
        """[[tn, fp], [fn, tp]] — rows = actual (benign, attack), cols = predicted (sklearn layout)."""
        return [[self.tn, self.fp], [self.fn, self.tp]]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["attack_recall"] = self.attack_recall
        d["false_negatives"] = self.false_negatives
        d["confusion_matrix"] = self.confusion_matrix
        return d

    def format_report(self, title: str = "Evaluation") -> str:
        lines = [
            f"{title} (threshold = {self.threshold:.6g}, n = {self.n_samples:,})",
            f"  Attack recall (detection rate): {self.recall:.4f}  ({self.tp:,} of {self.n_attack:,} attacks detected)",
            f"  False negatives (missed attacks): {self.fn:,}  (FNR {self.fnr:.4f})",
            f"  False positives (false alarms):   {self.fp:,}  (FPR {self.fpr:.4f})",
            f"  Accuracy {self.accuracy:.4f}  Precision {self.precision:.4f}  F1 {self.f1:.4f}  Macro-F1 {self.macro_f1:.4f}",
            f"  ROC-AUC {self.roc_auc:.4f}  PR-AUC {self.pr_auc:.4f}",
            "  Confusion matrix (rows = actual, cols = predicted):",
            f"                 pred Benign   pred Attack",
            f"    act Benign  {self.tn:>12,}  {self.fp:>12,}",
            f"    act Attack  {self.fn:>12,}  {self.tp:>12,}",
        ]
        return "\n".join(lines)


def _validate_inputs(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    if len(y_true) != len(y_prob):
        raise ValueError(f"y_true and y_prob length mismatch: {len(y_true)} vs {len(y_prob)}")
    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics on zero samples")
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError("y_true must contain only 0 (Benign) and 1 (Attack)")
    if not np.isfinite(y_prob).all() or (y_prob < 0).any() or (y_prob > 1).any():
        raise ValueError("y_prob must be finite probabilities in [0, 1] (apply sigmoid to logits first)")
    return y_true.astype(np.int64), y_prob


def _validate_threshold(threshold: float) -> float:
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    return threshold


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> BinaryMetrics:
    """All metrics from true labels and P(attack). A row is predicted Attack if y_prob >= threshold."""
    y_true, y_prob = _validate_inputs(y_true, y_prob)
    threshold = _validate_threshold(threshold)
    y_pred = (y_prob >= threshold).astype(np.int64)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1_attack = _safe_div(2 * tp, 2 * tp + fp + fn)
    f1_benign = _safe_div(2 * tn, 2 * tn + fn + fp)
    both_classes = 0 < y_true.sum() < len(y_true)

    return BinaryMetrics(
        threshold=threshold,
        n_samples=len(y_true),
        n_attack=tp + fn,
        n_benign=tn + fp,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        accuracy=_safe_div(tp + tn, len(y_true)),
        precision=precision,
        recall=recall,
        f1=f1_attack,
        macro_f1=(f1_attack + f1_benign) / 2,
        specificity=_safe_div(tn, tn + fp),
        fpr=_safe_div(fp, fp + tn),
        fnr=_safe_div(fn, fn + tp),
        roc_auc=float(roc_auc_score(y_true, y_prob)) if both_classes else float("nan"),
        # Average precision (step-wise PR-AUC), not trapezoidal auc(recall, precision),
        # which over-estimates the area under a PR curve.
        pr_auc=float(average_precision_score(y_true, y_prob)) if y_true.sum() > 0 else float("nan"),
    )


def make_val_metrics_fn(threshold: float) -> Callable[[nn.Module, DataLoader, torch.device], dict[str, float]]:
    """Per-epoch validation metrics for core.train.fit(extra_val_metrics_fn=...).

    Provides val_pr_auc (Phase 6's objective / early-stopping metric) plus
    threshold-dependent metrics at a fixed threshold for monitoring.
    """
    threshold = _validate_threshold(threshold)

    def fn(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
        y_true, y_prob = predict_proba_loader(model, loader, device)
        m = compute_metrics(y_true, y_prob, threshold)
        return {
            "val_pr_auc": m.pr_auc,
            "val_roc_auc": m.roc_auc,
            "val_f1": m.f1,
            "val_macro_f1": m.macro_f1,
            "val_attack_recall": m.recall,
            "val_fpr": m.fpr,
        }

    return fn


# ---------------------------------------------------------------------------
# 3a. Threshold selection (validation data only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdSelection:
    """A decision threshold chosen on validation data, with the validation metrics at that threshold."""

    threshold: float
    strategy: str
    target_recall: float | None
    validation_metrics: BinaryMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "strategy": self.strategy,
            "target_recall": self.target_recall,
            "selected_on": "validation",
            "validation_metrics": self.validation_metrics.to_dict(),
        }


def _counts_per_candidate_threshold(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For every distinct score t (descending): tp and fp when predicting Attack for y_prob >= t."""
    order = np.argsort(-y_prob, kind="mergesort")
    p_sorted, y_sorted = y_prob[order], y_true[order]
    # last position of each run of equal scores, so tied scores share one threshold
    last_of_run = np.r_[np.flatnonzero(np.diff(p_sorted)), len(p_sorted) - 1]
    tp = np.cumsum(y_sorted)[last_of_run]
    fp = (last_of_run + 1) - tp
    return p_sorted[last_of_run], tp, fp


def select_threshold(y_val: np.ndarray, p_val: np.ndarray, config: EvaluationConfig) -> ThresholdSelection:
    """Choose the decision threshold from VALIDATION labels and probabilities only.

    Takes plain arrays (no paths, loaders or scalers), so it has no way to read
    the test set; callers pass validation predictions.

    Strategies (config.threshold_strategy):
      max_f1 (default) — threshold maximizing attack-class F1 on validation.
          Chosen as the default because it directly trades missed attacks (FN)
          against false alarms (FP) on the minority class (~15% of rows), and
          needs no extra operating-point number. Ties go to the highest
          threshold (fewest false alarms).
      For the searched strategies the returned threshold is the midpoint
      between the winning validation score and the next lower distinct score
      (same validation predictions, more margin on unseen data).
      max_macro_f1 — threshold maximizing the mean of benign and attack F1.
      target_recall — highest threshold whose validation attack recall is
          >= config.target_recall, i.e. best precision subject to a minimum
          detection rate. Use when a recall requirement is given.
      fixed — config.default_threshold (0.5), no search.
    """
    y_val, p_val = _validate_inputs(y_val, p_val)
    strategy = config.threshold_strategy

    if strategy == "fixed":
        threshold = config.default_threshold
    else:
        n_pos = int(y_val.sum())
        n_neg = len(y_val) - n_pos
        if n_pos == 0 or n_neg == 0:
            raise ValueError(f"Threshold strategy {strategy!r} needs both classes in the validation data")
        thresholds, tp, fp = _counts_per_candidate_threshold(y_val, p_val)
        fn = n_pos - tp
        tn = n_neg - fp
        f1_attack = 2 * tp / (2 * tp + fp + fn)
        if strategy == "max_f1":
            best = int(np.argmax(f1_attack))  # first max = highest threshold among ties
        elif strategy == "max_macro_f1":
            f1_benign = 2 * tn / np.maximum(2 * tn + fn + fp, 1)
            best = int(np.argmax((f1_attack + f1_benign) / 2))
        elif strategy == "target_recall":
            reaches = np.flatnonzero(tp / n_pos >= config.target_recall)
            best = int(reaches[0])  # thresholds descend, so first = highest qualifying threshold
        else:
            raise ValueError(f"Unknown threshold strategy {strategy!r}")
        # Place the threshold midway between the chosen score and the next lower
        # distinct score. Validation predictions are identical anywhere in that
        # gap, but sitting exactly on a validation score would put the boundary
        # right at the edge of the training-time decision, so a test row scoring
        # a hair below it flips class.
        if best + 1 < len(thresholds):
            threshold = float((thresholds[best] + thresholds[best + 1]) / 2)
        else:
            threshold = float(thresholds[best])

    return ThresholdSelection(
        threshold=threshold,
        strategy=strategy,
        target_recall=config.target_recall if strategy == "target_recall" else None,
        validation_metrics=compute_metrics(y_val, p_val, threshold),
    )


# ---------------------------------------------------------------------------
# 3b. Final test-set evaluation (fixed threshold + fitted scaler only)
# ---------------------------------------------------------------------------


def evaluate_on_test(
    model: Any,
    *,
    scaler: StandardScaler | None,
    threshold: float,
    feature_names: list[str],
    dataset_dir: Path,
    batch_size: int,
    device: torch.device | None = None,
) -> BinaryMetrics:
    """Evaluate a finished model on the held-out test set. Call once, after tuning is complete.

    Args:
        model: Trained torch model or sklearn-style classifier.
        scaler: The scaler fitted on this model's training data (already fitted;
            nothing is fitted here). Pass None explicitly only for a model trained
            on unscaled features (Random Forest).
        threshold: Decision threshold already chosen on validation data
            (ThresholdSelection.threshold). This function never selects one.
        feature_names: Training feature schema; the test columns must match it.
    """
    threshold = _validate_threshold(threshold)
    if scaler is not None:
        try:
            check_is_fitted(scaler)
        except NotFittedError as e:
            raise ValueError("evaluate_on_test requires an already-fitted scaler") from e

    test = load_test_set(scaler, feature_names, dataset_dir)
    y_prob = predict_proba(model, test.X, batch_size=batch_size, device=device)
    return compute_metrics(test.y, y_prob, threshold)


def main() -> None:
    """Final evaluation of a saved checkpoint: select a threshold on validation, then score the test set once."""
    parser = argparse.ArgumentParser(description="Final held-out test evaluation of a checkpoint")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, required=True, help="e.g. models/ann_centralized/best.pt")
    parser.add_argument("--name", required=True, help="results file name stem, e.g. ann_centralized")
    parser.add_argument("--final-test", action="store_true",
                        help="required: confirms tuning is finished and the test set may be used")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing results file")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    if not args.final_test:
        raise SystemExit("Refusing to touch the test set without --final-test (only run this after all tuning is done).")
    out_path = args.results_dir / f"{args.name}_test_metrics.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"{out_path} already exists (test set already evaluated). Pass --overwrite to replace it.")

    cfg = load_config(args.config)
    device = get_device(cfg.device)
    checkpoint = load_checkpoint(args.checkpoint)
    model = checkpoint.build_model(device)
    scaler = checkpoint.build_scaler()

    # Threshold: chosen on the validation rows, scaled with the checkpoint's own scaler.
    X_val_raw, y_val, val_features = load_raw_validation(cfg.data.dataset_dir, cfg.data.val_fraction, cfg.seed)
    if val_features != checkpoint.feature_names:
        raise ValueError("Validation feature schema does not match the checkpoint's feature schema")
    p_val = predict_proba(model, apply_scaler(scaler, X_val_raw), batch_size=cfg.data.eval_batch_size, device=device)
    selection = select_threshold(y_val, p_val, cfg.evaluation)
    print(selection.validation_metrics.format_report(f"Validation ({selection.strategy})"))

    test_metrics = evaluate_on_test(
        model,
        scaler=scaler,
        threshold=selection.threshold,
        feature_names=checkpoint.feature_names,
        dataset_dir=cfg.data.dataset_dir,
        batch_size=cfg.data.eval_batch_size,
        device=device,
    )
    print()
    print(test_metrics.format_report("HELD-OUT TEST"))

    args.results_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "checkpoint_epoch": checkpoint.epoch,
                "architecture": checkpoint.architecture,
                "threshold_selection": selection.to_dict(),
                "test_metrics": test_metrics.to_dict(),
            },
            f,
            indent=2,
        )
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
