"""
Early stopping on a monitored validation metric.

Supports both directions explicitly:
    mode="min"  lower is better   (e.g. val_loss)
    mode="max"  higher is better  (e.g. val_pr_auc, val_f1 — Phase 6's objective)

An epoch counts as an improvement only if it beats the best value so far by
more than min_delta. Training should stop once `patience` consecutive epochs
pass without an improvement.
"""

import math
from typing import Any

from core.config import EarlyStoppingConfig


class EarlyStopping:
    """Patience-based early stopping for a lower- or higher-is-better metric."""

    def __init__(self, metric: str, mode: str, patience: int, min_delta: float = 0.0) -> None:
        """
        Args:
            metric: Name of the monitored history key (e.g. "val_loss", "val_pr_auc").
            mode: "min" if lower is better, "max" if higher is better.
            patience: Consecutive non-improving epochs allowed before stopping (>= 1).
            min_delta: Minimum change over the best value that counts as an improvement (>= 0).
        """
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if min_delta < 0:
            raise ValueError(f"min_delta must be >= 0, got {min_delta}")

        self.metric = metric
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta

        self.best_value: float | None = None
        self.best_epoch: int | None = None
        self.epochs_without_improvement = 0
        self.should_stop = False

    @classmethod
    def from_config(cls, config: EarlyStoppingConfig) -> "EarlyStopping":
        return cls(config.metric, config.mode, config.patience, config.min_delta)

    def is_improvement(self, value: float) -> bool:
        """Whether `value` beats the current best by more than min_delta (NaN never improves)."""
        if math.isnan(value):
            return False
        if self.best_value is None:
            return True
        if self.mode == "min":
            return value < self.best_value - self.min_delta
        return value > self.best_value + self.min_delta

    def step(self, value: float, epoch: int) -> bool:
        """Record one epoch's metric value.

        Returns:
            True if this epoch is a new best (caller should save the "best" checkpoint).
        """
        value = float(value)
        if self.is_improvement(value):
            self.best_value = value
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            improved = True
        else:
            self.epochs_without_improvement += 1
            improved = False

        if self.epochs_without_improvement >= self.patience:
            self.should_stop = True
        return improved

    def state_dict(self) -> dict[str, Any]:
        """Serializable state, stored in checkpoints so resumed runs keep their patience counter."""
        return {
            "metric": self.metric,
            "mode": self.mode,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "epochs_without_improvement": self.epochs_without_improvement,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore progress from a checkpoint; the metric and direction must match."""
        if state["metric"] != self.metric or state["mode"] != self.mode:
            raise ValueError(
                f"Checkpoint early-stopping monitors {state['metric']!r} ({state['mode']}), "
                f"but current config monitors {self.metric!r} ({self.mode})"
            )
        self.best_value = state["best_value"]
        self.best_epoch = state["best_epoch"]
        self.epochs_without_improvement = state["epochs_without_improvement"]
        self.should_stop = state["should_stop"]
