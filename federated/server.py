"""Flower server: configures the FedAvg strategy and orchestrates federated training rounds across Client A, B, and C."""

import json
from pathlib import Path

import flwr as fl
import matplotlib.pyplot as plt
import pandas as pd
import torch

from client import client_fn, set_parameters
from models import get_model
from utils import combine_client_scalers, evaluate, load_and_scale_client_data

NUM_ROUNDS = 15
LOCAL_EPOCHS = 3

BASE_DIR = Path(__file__).resolve().parent.parent
CLIENT_PATHS = [
    BASE_DIR / "dataset" / "client_A.parquet",
    BASE_DIR / "dataset" / "client_B.parquet",
    BASE_DIR / "dataset" / "client_C.parquet",
]
TEST_PATH = BASE_DIR / "dataset" / "test.parquet"
METRICS_PATH = BASE_DIR / "results" / "federated_lr_metrics.json"
CONVERGENCE_PLOT_PATH = BASE_DIR / "graphs" / "federated_lr_convergence.png"

LABEL_COL = "label"


def build_combined_test_scaler():
    scalers = []
    for path in CLIENT_PATHS:
        _, _, scaler = load_and_scale_client_data(path)
        scalers.append(scaler)
    return combine_client_scalers(scalers)


def load_scaled_test_set(scaler):
    df = pd.read_parquet(TEST_PATH)
    X = df.drop(columns=[LABEL_COL]).values
    y = df[LABEL_COL].values

    X_scaled = scaler.transform(X)

    features = torch.tensor(X_scaled, dtype=torch.float32)
    labels = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    return features, labels


def get_evaluate_fn(model_type: str, test_features, test_labels):
    def evaluate_fn(server_round, parameters, config):
        model = get_model(model_type)
        set_parameters(model, parameters)

        metrics = evaluate(model, test_features, test_labels)
        loss = 1 - metrics["accuracy"]

        return float(loss), metrics

    return evaluate_fn


def fit_config(server_round: int):
    return {"LOCAL_EPOCHS": LOCAL_EPOCHS}


def run_simulation(num_rounds: int):
    print("Building combined scaler from client A/B/C statistics...")
    combined_scaler = build_combined_test_scaler()

    print(f"Loading and scaling held-out test set from {TEST_PATH}...")
    test_features, test_labels = load_scaled_test_set(combined_scaler)
    print(f"Test set: {test_features.shape[0]:,} rows, {test_features.shape[1]} features")

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=3,
        min_available_clients=3,
        evaluate_fn=get_evaluate_fn("lr", test_features, test_labels),
        on_fit_config_fn=fit_config,
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=3,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    return history


def save_metrics(history) -> list[dict]:
    losses_by_round = dict(history.losses_centralized)
    metrics_by_round = {
        name: dict(values) for name, values in history.metrics_centralized.items()
    }

    rounds = sorted(losses_by_round.keys())
    records = []
    for round_num in rounds:
        record = {
            "round": round_num,
            "loss": losses_by_round[round_num],
            "accuracy": metrics_by_round["accuracy"][round_num],
            "precision": metrics_by_round["precision"][round_num],
            "recall": metrics_by_round["recall"][round_num],
            "f1_score": metrics_by_round["f1_score"][round_num],
            "roc_auc": metrics_by_round["roc_auc"][round_num],
        }
        records.append(record)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved per-round metrics to {METRICS_PATH}")

    return records


def plot_convergence(records: list[dict]) -> None:
    rounds = [r["round"] for r in records]
    accuracies = [r["accuracy"] for r in records]
    losses = [r["loss"] for r in records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(rounds, accuracies, marker="o", color="#4C72B0")
    ax1.set_title("Federated LR — Accuracy vs Round")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Accuracy (centralized test set)")

    ax2.plot(rounds, losses, marker="o", color="#C44E52")
    ax2.set_title("Federated LR — Loss vs Round")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Loss (1 - accuracy)")

    fig.tight_layout()
    CONVERGENCE_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CONVERGENCE_PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved convergence plot to {CONVERGENCE_PLOT_PATH}")


def main() -> None:
    history = run_simulation(NUM_ROUNDS)
    records = save_metrics(history)
    plot_convergence(records)

    final = records[-1]
    print("Final round metrics:")
    for key, value in final.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
