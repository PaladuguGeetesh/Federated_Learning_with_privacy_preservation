"""
Random Forest model definition for the centralized baseline.

Holds only the model configuration and a builder for the sklearn
RandomForestClassifier. Data loading, fitting, evaluation and saving live in
centralized/train_rf.py.
"""

from dataclasses import asdict, dataclass
from typing import Any

from sklearn.ensemble import RandomForestClassifier


@dataclass(frozen=True)
class RandomForestConfig:
    """Hyperparameters for the centralized Random Forest baseline.

    Defaults match the values used to train models/rf_baseline.joblib.
    """

    n_estimators: int = 200
    max_depth: int | None = None
    class_weight: str | dict | None = "balanced"
    random_state: int = 42
    n_jobs: int = -1

    def to_dict(self) -> dict[str, Any]:
        """Return the config as a plain dict (for logging / JSON metrics files)."""
        return asdict(self)


def build_random_forest(config: RandomForestConfig | None = None) -> RandomForestClassifier:
    """Build an unfitted RandomForestClassifier from a config.

    Args:
        config: Hyperparameters to use. Falls back to RandomForestConfig() defaults.

    Returns:
        An unfitted sklearn RandomForestClassifier.
    """
    config = config or RandomForestConfig()
    if config.n_estimators < 1:
        raise ValueError(f"n_estimators must be >= 1, got {config.n_estimators}")
    if config.max_depth is not None and config.max_depth < 1:
        raise ValueError(f"max_depth must be None or >= 1, got {config.max_depth}")

    return RandomForestClassifier(**config.to_dict())
