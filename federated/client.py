"""Flower client implementation: wraps a local model and dataset for one simulated organization, handling fit/evaluate calls from the server."""

from collections import OrderedDict
from pathlib import Path

import flwr as fl
import numpy as np
import torch

from models import get_model
from utils import compute_pos_weight, evaluate, load_and_scale_client_data, train_one_epoch

MODEL_TYPE = "lr"
BATCH_SIZE = 512
DEFAULT_LOCAL_EPOCHS = 3

BASE_DIR = Path(__file__).resolve().parent.parent
CLIENT_PATHS = {
    "A": BASE_DIR / "dataset" / "client_A.parquet",
    "B": BASE_DIR / "dataset" / "client_B.parquet",
    "C": BASE_DIR / "dataset" / "client_C.parquet",
}


def get_parameters(model) -> list[np.ndarray]:
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_parameters(model, parameters: list[np.ndarray]) -> None:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id: str, model_type: str):
        self.client_id = client_id
        self.model_type = model_type

        data_path = CLIENT_PATHS[client_id]
        self.features, self.labels, self.scaler = load_and_scale_client_data(data_path)

        self.model = get_model(model_type)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)

        pos_weight = compute_pos_weight(self.labels)
        local_epochs = config.get("LOCAL_EPOCHS", DEFAULT_LOCAL_EPOCHS)

        final_loss = None
        for _ in range(local_epochs):
            final_loss = train_one_epoch(
                self.model, self.features, self.labels, pos_weight, batch_size=BATCH_SIZE
            )

        return get_parameters(self.model), len(self.features), {"loss": final_loss}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)

        metrics = evaluate(self.model, self.features, self.labels)
        loss = 1 - metrics["accuracy"]

        return float(loss), len(self.features), metrics


def client_fn(cid: str) -> fl.client.Client:
    cid_to_client = {
        "0": "A",
        "1": "B",
        "2": "C",
    }
    client_id = cid_to_client[cid]
    return FlowerClient(client_id=client_id, model_type=MODEL_TYPE).to_client()


if __name__ == "__main__":
    client = FlowerClient(client_id="A", model_type="mlp")

    initial_parameters = client.get_parameters(config={})
    print(f"Retrieved {len(initial_parameters)} parameter arrays from initial model.")

    updated_parameters, num_examples, fit_metrics = client.fit(
        initial_parameters, config={"LOCAL_EPOCHS": 2}
    )
    print(f"fit() — num_examples: {num_examples}, metrics: {fit_metrics}")

    loss, num_examples, eval_metrics = client.evaluate(updated_parameters, config={})
    print(f"evaluate() — loss: {loss:.4f}, num_examples: {num_examples}")
    print("evaluate() metrics:")
    for name, value in eval_metrics.items():
        print(f"  {name}: {value:.4f}")
