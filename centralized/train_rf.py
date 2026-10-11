"""
Centralized Random Forest baseline for malware detection.

Loads the persisted train/test split (<data.dataset_dir>/train.parquet,
test.parquet). If both files are missing, the split is generated once from
cleaned_data.parquet (stratified, data.test_size, seed); an existing split is
never overwritten. Trains the Random Forest defined in
centralized/random_forest.py on all of train.parquet, evaluates it ONCE on the
held-out test set, and saves the model, metrics and plots.

The RF has no validation split to tune a threshold on, so it uses
evaluation.default_threshold (0.5): a row is predicted Attack if
P(attack) >= threshold. Metrics come from core.evaluate, the same code that
scores the ANN and LR models.

Run from the project root:
    python -m centralized.train_rf              # refuses if its test results already exist
    python -m centralized.train_rf --overwrite  # deliberately re-run training + test evaluation
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_curve
from sklearn.model_selection import train_test_split

from centralized.random_forest import RandomForestConfig, build_random_forest
from core.config import DEFAULT_CONFIG_PATH, Config, load_config
from core.dataset import LABEL_COL, TEST_FILENAME
from core.evaluate import BinaryMetrics, compute_metrics, predict_proba

CLEANED_FILENAME = "cleaned_data.parquet"
TRAIN_FILENAME = "train.parquet"
METRICS_FILENAME = "centralized_rf_metrics.json"
CONFUSION_MATRIX_FILENAME = "rf_confusion_matrix.png"
ROC_CURVE_FILENAME = "rf_roc_curve.png"


def create_split(dataset_dir: Path, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate the stratified train/test split from cleaned_data.parquet and save it."""
    cleaned = dataset_dir / CLEANED_FILENAME
    if not cleaned.exists():
        raise FileNotFoundError(f"{cleaned} not found. Run preprocessing/preprocess.py first.")
    df = pd.read_parquet(cleaned)
    print(f"Loaded {cleaned}: {df.shape[0]:,} rows, {df.shape[1]} columns")

    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df[LABEL_COL], random_state=seed)
    train_df.to_parquet(dataset_dir / TRAIN_FILENAME, index=False)
    test_df.to_parquet(dataset_dir / TEST_FILENAME, index=False)
    print(f"Saved train split: {train_df.shape[0]:,} rows; test split: {test_df.shape[0]:,} rows")
    return train_df, test_df


def load_or_create_split(dataset_dir: Path, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the existing train/test split, generating it only if both files are missing.

    If exactly one of the two files exists, this refuses to continue rather than
    regenerating: re-splitting would silently replace the held-out test set.
    """
    train_path, test_path = dataset_dir / TRAIN_FILENAME, dataset_dir / TEST_FILENAME
    train_exists, test_exists = train_path.exists(), test_path.exists()

    if train_exists and test_exists:
        train_df, test_df = pd.read_parquet(train_path), pd.read_parquet(test_path)
        print(f"Loaded existing split: train {train_df.shape[0]:,} rows, test {test_df.shape[0]:,} rows")
        return train_df, test_df
    if train_exists != test_exists:
        missing = test_path if train_exists else train_path
        raise FileNotFoundError(
            f"{missing} is missing but its counterpart exists. Refusing to re-split, since that would "
            f"change the held-out test set. Restore the missing file, or delete both to regenerate them."
        )
    print("No existing train/test split found — generating it once.")
    return create_split(dataset_dir, test_size, seed)


def train_model(train_df: pd.DataFrame, config: RandomForestConfig) -> tuple[RandomForestClassifier, float]:
    model = build_random_forest(config)
    start = time.time()
    model.fit(train_df.drop(columns=[LABEL_COL]), train_df[LABEL_COL])
    elapsed = time.time() - start
    print(f"Training time: {elapsed:.2f} seconds")
    return model, elapsed


def evaluate_model(model: RandomForestClassifier, test_df: pd.DataFrame, threshold: float,
                   batch_size: int, graphs_dir: Path) -> BinaryMetrics:
    """Single test-set evaluation through core.evaluate, plus confusion-matrix and ROC plots."""
    X_test = test_df.drop(columns=[LABEL_COL])
    y_test = test_df[LABEL_COL].to_numpy()
    y_proba = predict_proba(model, X_test, batch_size=batch_size)
    metrics = compute_metrics(y_test, y_proba, threshold)
    print(metrics.format_report("HELD-OUT TEST (centralized RF)"))

    graphs_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(np.asarray(metrics.confusion_matrix), display_labels=["Benign (0)", "Attack (1)"]).plot(
        ax=ax, cmap="Blues", values_format=","
    )
    ax.set_title("Random Forest — Confusion Matrix")
    fig.tight_layout()
    fig.savefig(graphs_dir / CONFUSION_MATRIX_FILENAME, dpi=150)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=metrics.roc_auc).plot(ax=ax)
    ax.set_title("Random Forest — ROC Curve")
    fig.tight_layout()
    fig.savefig(graphs_dir / ROC_CURVE_FILENAME, dpi=150)
    plt.close(fig)
    print(f"Saved plots to {graphs_dir}")
    return metrics


def run(cfg: Config, overwrite: bool = False) -> Path:
    metrics_path = cfg.paths.results_dir / METRICS_FILENAME
    if metrics_path.exists() and not overwrite:
        raise SystemExit(f"{metrics_path} already exists (test set already evaluated for the RF). "
                         f"Pass --overwrite to deliberately retrain and re-evaluate.")

    rf = cfg.random_forest
    train_df, test_df = load_or_create_split(cfg.data.dataset_dir, cfg.data.test_size, cfg.seed)
    model, training_time = train_model(train_df, rf.hyperparameters)
    metrics = evaluate_model(model, test_df, cfg.evaluation.default_threshold,
                             cfg.data.eval_batch_size, cfg.paths.graphs_dir)

    rf.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, rf.model_path)
    print(f"Saved trained model to {rf.model_path}")

    results = {
        "model": "RandomForestClassifier",
        "hyperparameters": rf.hyperparameters.to_dict(),
        "test_size": cfg.data.test_size,
        "training_time_seconds": training_time,
        # Same keys as earlier versions of this file, kept for readers of the old format.
        "metrics": {"accuracy": metrics.accuracy, "precision": metrics.precision, "recall": metrics.recall,
                    "f1_score": metrics.f1, "roc_auc": metrics.roc_auc},
        "test_metrics": metrics.to_dict(),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    return metrics_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Centralized Random Forest baseline (single test evaluation)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--overwrite", action="store_true",
                        help="retrain and re-evaluate on the test set even though results already exist")
    args = parser.parse_args(argv)
    run(load_config(args.config), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
