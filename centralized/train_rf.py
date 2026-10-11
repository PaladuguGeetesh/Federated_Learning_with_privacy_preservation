"""
Centralized Random Forest baseline for malware detection.

Loads the persisted train/test split (dataset/train.parquet, dataset/test.parquet).
If either file is missing, it is generated once from dataset/cleaned_data.parquet
with a deterministic 80/20 stratified split; existing split files are never
overwritten. Trains the Random Forest defined in centralized/random_forest.py and
saves the model, metrics, and evaluation plots.

Run from the project root:
    python -m centralized.train_rf
"""

import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from centralized.random_forest import RandomForestConfig, build_random_forest

# --- Split settings (used only if the split files do not exist yet) ---
SPLIT_RANDOM_STATE = 42
TEST_SIZE = 0.2

BASE_DIR = Path(__file__).resolve().parent.parent
CLEANED_DATA_PATH = BASE_DIR / "dataset" / "cleaned_data.parquet"
TRAIN_PATH = BASE_DIR / "dataset" / "train.parquet"
TEST_PATH = BASE_DIR / "dataset" / "test.parquet"
MODEL_PATH = BASE_DIR / "models" / "rf_baseline.joblib"
METRICS_PATH = BASE_DIR / "results" / "centralized_rf_metrics.json"
CONFUSION_MATRIX_PATH = BASE_DIR / "graphs" / "rf_confusion_matrix.png"
ROC_CURVE_PATH = BASE_DIR / "graphs" / "rf_roc_curve.png"

LABEL_COL = "label"


def create_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate the 80/20 stratified train/test split from cleaned_data.parquet and save it."""
    if not CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"{CLEANED_DATA_PATH} not found. Run preprocessing/preprocess.py first."
        )
    df = pd.read_parquet(CLEANED_DATA_PATH)
    print(f"Loaded {CLEANED_DATA_PATH}: {df.shape[0]:,} rows, {df.shape[1]} columns")

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df[LABEL_COL],
        random_state=SPLIT_RANDOM_STATE,
    )

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(TRAIN_PATH, index=False)
    test_df.to_parquet(TEST_PATH, index=False)
    print(f"Saved train split to {TRAIN_PATH}: {train_df.shape[0]:,} rows")
    print(f"Saved test split to {TEST_PATH}: {test_df.shape[0]:,} rows")

    return train_df, test_df


def load_or_create_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the existing train/test split, generating it only if both files are missing.

    If exactly one of the two files exists, this refuses to continue rather than
    regenerating: re-splitting would silently replace the held-out test set.
    """
    train_exists, test_exists = TRAIN_PATH.exists(), TEST_PATH.exists()

    if train_exists and test_exists:
        train_df = pd.read_parquet(TRAIN_PATH)
        test_df = pd.read_parquet(TEST_PATH)
        print(f"Loaded existing train split {TRAIN_PATH}: {train_df.shape[0]:,} rows")
        print(f"Loaded existing test split {TEST_PATH}: {test_df.shape[0]:,} rows")
        return train_df, test_df

    if train_exists != test_exists:
        missing = TEST_PATH if train_exists else TRAIN_PATH
        raise FileNotFoundError(
            f"{missing} is missing but its counterpart exists. Refusing to re-split, "
            f"since that would change the held-out test set. Restore the missing file, "
            f"or delete both split files to regenerate them."
        )

    print("No existing train/test split found — generating it once.")
    return create_split()


def train_model(
    train_df: pd.DataFrame, config: RandomForestConfig
) -> tuple[RandomForestClassifier, float]:
    X_train = train_df.drop(columns=[LABEL_COL])
    y_train = train_df[LABEL_COL]

    model = build_random_forest(config)

    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"Training time: {elapsed:.2f} seconds")

    return model, elapsed


def evaluate_model(model: RandomForestClassifier, test_df: pd.DataFrame) -> dict:
    X_test = test_df.drop(columns=[LABEL_COL])
    y_test = test_df[LABEL_COL]

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print("Evaluation metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Benign (0)", "Attack (1)"]).plot(
        ax=ax, cmap="Blues", values_format=","
    )
    ax.set_title("Random Forest — Confusion Matrix")
    fig.tight_layout()
    CONFUSION_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix plot to {CONFUSION_MATRIX_PATH}")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=metrics["roc_auc"]).plot(ax=ax)
    ax.set_title("Random Forest — ROC Curve")
    fig.tight_layout()
    fig.savefig(ROC_CURVE_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved ROC curve plot to {ROC_CURVE_PATH}")

    return metrics


def main() -> None:
    config = RandomForestConfig()
    train_df, test_df = load_or_create_split()
    model, training_time = train_model(train_df, config)
    metrics = evaluate_model(model, test_df)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")

    results = {
        "model": "RandomForestClassifier",
        "hyperparameters": config.to_dict(),
        "test_size": TEST_SIZE,
        "training_time_seconds": training_time,
        "metrics": metrics,
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
