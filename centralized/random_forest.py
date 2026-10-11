"""
Random Forest model definition for the centralized baseline.

Holds only the builder for the sklearn RandomForestClassifier. Its
hyperparameters come from config.yaml (random_forest.hyperparameters, with
random_state = seed) via core.config.RandomForestConfig. Data loading, fitting,
evaluation and saving live in centralized/train_rf.py.
"""

from sklearn.ensemble import RandomForestClassifier

from core.config import RandomForestConfig

__all__ = ["RandomForestConfig", "build_random_forest"]


def build_random_forest(config: RandomForestConfig) -> RandomForestClassifier:
    """Build an unfitted RandomForestClassifier from a validated config."""
    return RandomForestClassifier(**config.to_dict())
