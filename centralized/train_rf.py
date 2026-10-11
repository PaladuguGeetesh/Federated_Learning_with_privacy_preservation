"""
Centralized Random Forest baseline for malware detection.

Loads the cleaned CICIDS2017 dataset, performs a single deterministic 80/20
stratified train/test split (persisted for reuse by later federated steps),
trains a RandomForestClassifier, and saves the model, metrics, and evaluation
plots.
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

# --- Hyperparameters (edit here) ---
N_ESTIMATORS = 200
MAX_DEPTH = None
CLASS_WEIGHT = "balanced"
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_JOBS = -1

BASE_DIR = Path(__file__).resolve().parent.parent
CLEANED_DATA_PATH = BASE_DIR / "dataset" / "cleaned_data.parquet"
TRAIN_PATH = BASE_DIR / "dataset" / "train.parquet"
TEST_PATH = BASE_DIR / "dataset" / "test.parquet"
MODEL_PATH = BASE_DIR / "models" / "rf_baseline.joblib"
METRICS_PATH = BASE_DIR / "results" / "centralized_rf_metrics.json"
CONFUSION_MATRIX_PATH = BASE_DIR / "graphs" / "rf_confusion_matrix.png"
ROC_CURVE_PATH = BASE_DIR / "graphs" / "rf_roc_curve.png"

LABEL_COL = "label"


def load_and_split_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(CLEANED_DATA_PATH)
    print(f"Loaded {CLEANED_DATA_PATH}: {df.shape[0]:,} rows, {df.shape[1]} columns")

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df[LABEL_COL],
        random_state=RANDOM_STATE,
    )

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(TRAIN_PATH, index=False)
    test_df.to_parquet(TEST_PATH, index=False)
    print(f"Saved train split to {TRAIN_PATH}: {train_df.shape[0]:,} rows")
    print(f"Saved test split to {TEST_PATH}: {test_df.shape[0]:,} rows")

    return train_df, test_df


def train_model(train_df: pd.DataFrame) -> RandomForestClassifier:
    X_train = train_df.drop(columns=[LABEL_COL])
    y_train = train_df[LABEL_COL]

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        class_weight=CLASS_WEIGHT,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

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
    train_df, test_df = load_and_split_data()
    model, training_time = train_model(train_df)
    metrics = evaluate_model(model, test_df)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")

    results = {
        "model": "RandomForestClassifier",
        "hyperparameters": {
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "class_weight": CLASS_WEIGHT,
            "random_state": RANDOM_STATE,
            "n_jobs": N_JOBS,
        },
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
