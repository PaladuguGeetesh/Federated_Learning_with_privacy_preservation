"""Shared helper functions for the federated pipeline: per-client data loading/scaling, local training, and evaluation."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

LABEL_COL = "label"


def load_and_scale_client_data(path):
    """Load a client's parquet file and fit+apply a StandardScaler using only that client's data.

    Returns (features, labels, scaler) where features/labels are float32 torch tensors.
    """
    df = pd.read_parquet(path)
    X = df.drop(columns=[LABEL_COL]).values
    y = df[LABEL_COL].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    features = torch.tensor(X_scaled, dtype=torch.float32)
    labels = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    return features, labels, scaler


def combine_client_scalers(scalers: list) -> StandardScaler:
    """Combine 3 already-fitted per-client StandardScalers into one, for centralized
    evaluation only — never used in any client's local training.

    Assumes equal sample size per client (true for the IID 3-way split), so the
    combined mean/variance are simple averages (pooled variance) across clients.
    """
    means = np.stack([s.mean_ for s in scalers])
    variances = np.stack([s.var_ for s in scalers])

    combined_mean = means.mean(axis=0)
    combined_var = variances.mean(axis=0)

    combined_scaler = StandardScaler()
    combined_scaler.mean_ = combined_mean
    combined_scaler.var_ = combined_var
    combined_scale = np.sqrt(combined_var)
    # Match sklearn's own StandardScaler behavior: near-zero-variance features get
    # scale 1.0 instead of ~0, avoiding a divide-by-zero -> NaN on transform().
    near_zero = combined_scale < 10 * np.finfo(combined_scale.dtype).eps
    combined_scale[near_zero] = 1.0
    combined_scaler.scale_ = combined_scale
    combined_scaler.n_features_in_ = len(combined_mean)
    combined_scaler.n_samples_seen_ = sum(s.n_samples_seen_ for s in scalers)

    return combined_scaler


def compute_pos_weight(labels):
    num_pos = (labels == 1).sum()
    num_neg = (labels == 0).sum()
    return (num_neg / num_pos).to(torch.float32)


def train_one_epoch(model, features, labels, pos_weight, lr=1e-3, batch_size=512):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    num_examples = features.shape[0]
    permutation = torch.randperm(num_examples)

    total_loss = 0.0
    num_batches = 0
    for start in range(0, num_examples, batch_size):
        batch_idx = permutation[start : start + batch_size]
        batch_features = features[batch_idx]
        batch_labels = labels[batch_idx]

        optimizer.zero_grad()
        logits = model(batch_features)
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, features, labels):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

    y_true = labels.numpy()
    y_pred = preds.numpy()
    y_proba = probs.numpy()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


if __name__ == "__main__":
    from models import get_model

    CLIENT_A_PATH = Path(__file__).resolve().parent.parent / "dataset" / "client_A.parquet"
    NUM_EPOCHS = 3

    features, labels, scaler = load_and_scale_client_data(CLIENT_A_PATH)
    print(f"Loaded {CLIENT_A_PATH}: {features.shape[0]:,} rows, {features.shape[1]} features")

    pos_weight = compute_pos_weight(labels)
    print(f"pos_weight: {pos_weight.item():.4f}")

    model = get_model("mlp")
    for epoch in range(1, NUM_EPOCHS + 1):
        loss = train_one_epoch(model, features, labels, pos_weight)
        print(f"Epoch {epoch}/{NUM_EPOCHS} — loss: {loss:.4f}")

    metrics = evaluate(model, features, labels)
    print(f"Final train-set accuracy: {metrics['accuracy']:.4f}")
