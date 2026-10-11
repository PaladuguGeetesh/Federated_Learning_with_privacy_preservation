"""
Flower client for one simulated organization (A, B or C).

Each round the client:
  1. loads the global arrays from the server into its local model,
  2. trains for `local_epochs` epochs on ITS OWN training split only
     (core.dataset.load_client_data — own stratified split, own scaler) using
     core.train's loop (train_one_epoch, build_optimizer, build_criterion):
     raw logits + BCEWithLogitsLoss, no sigmoid in the model,
  3. returns its updated arrays + its local training sample count, which the
     server uses as the FedAvg weight.

Returned arrays follow federated.common.ParameterLayout: trainable parameters
first, then BatchNorm buffers (only present if the model has BatchNorm).
Buffers are running statistics updated during forward passes in train mode —
they are not gradient-trained — and the server combines them on a separate,
sample-count-weighted statistics path (federated/aggregation.py) instead of
FedAvg-ing them like weights.

The optimizer is rebuilt every round (its state is not carried across rounds),
as in standard FedAvg simulations.
"""

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

import flwr as fl
import numpy as np
import torch

from core.config import Config
from core.dataset import CLIENT_IDS, ClientData, IntrusionDataset, get_device, load_client_data, make_dataloader
from core.train import (
    build_criterion,
    build_optimizer,
    compute_pos_weight,
    set_seed,
    train_drop_last,
    train_one_epoch,
)
from federated.common import FederatedSettings, ParameterLayout, build_model, get_arrays, set_arrays


@lru_cache(maxsize=None)
def _client_data(client_id: str, dataset_dir: Path, val_fraction: float, seed: int) -> tuple[ClientData, IntrusionDataset]:
    """Load a client's data once per process (Flower re-creates client objects every round)."""
    data = load_client_data(client_id, dataset_dir, val_fraction, seed)
    return data, data.train.to_dataset()


class IntrusionClient(fl.client.NumPyClient):
    """NumPyClient training the shared model on one organization's local data."""

    def __init__(self, client_id: str, cfg: Config, settings: FederatedSettings, device: torch.device) -> None:
        if client_id not in CLIENT_IDS:
            raise ValueError(f"Unknown client id {client_id!r}")
        self.client_id = client_id
        self.cfg = cfg
        self.settings = settings
        self.device = device
        self.data, self.train_ds = _client_data(client_id, cfg.data.dataset_dir, cfg.data.val_fraction, cfg.seed)
        self.model = build_model(settings, len(self.data.feature_names)).to(device)
        self.layout = ParameterLayout.from_model(self.model)
        # Class weighting uses this client's own training labels only.
        self.pos_weight = compute_pos_weight(
            self.data.train.y, settings.training.class_weighting, settings.training.pos_weight_scale
        )

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        return get_arrays(self.model)

    def fit(self, parameters: list[np.ndarray], config: dict[str, Any]) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        set_arrays(self.model, self.layout, parameters)
        server_round = int(config["server_round"])
        local_epochs = int(config["local_epochs"])
        if local_epochs < 1:
            raise ValueError(f"local_epochs must be >= 1, got {local_epochs}")

        # Different, reproducible shuffling per (round, client).
        seed = self.cfg.seed + 1000 * server_round + CLIENT_IDS.index(self.client_id)
        set_seed(seed)
        training = self.settings.training
        batch_norm = self.settings.ann.batch_norm if self.settings.ann else False
        loader = make_dataloader(
            self.train_ds,
            batch_size=training.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
            device=self.device,
            seed=seed,
            drop_last=train_drop_last(len(self.train_ds), training.batch_size, batch_norm),
        )
        optimizer = build_optimizer(self.model, training)
        criterion = build_criterion(self.pos_weight, self.device)

        losses = [
            train_one_epoch(self.model, loader, criterion, optimizer, self.device, training.grad_clip_norm).loss
            for _ in range(local_epochs)
        ]
        metrics = {
            "client_id": self.client_id,
            "train_loss_first_epoch": losses[0],
            "train_loss": losses[-1],
            "num_trainable_arrays": self.layout.num_trainable,
            "num_buffer_arrays": len(self.layout.buffers),
        }
        return get_arrays(self.model), len(self.train_ds), metrics


def make_client_fn(cfg: Config, settings: FederatedSettings) -> Callable[[str], fl.client.Client]:
    """client_fn for flwr.simulation.start_simulation.

    Flower 1.9's simulation engine requires integer-like client ids, so the
    simulation uses "0", "1", "2", mapped here to organizations A, B, C.
    """

    def client_fn(cid: str) -> fl.client.Client:
        try:
            client_id = CLIENT_IDS[int(cid)]
        except (ValueError, IndexError) as e:
            raise ValueError(f"Unexpected simulation client id {cid!r}; expected 0..{len(CLIENT_IDS) - 1}") from e
        torch.set_num_threads(cfg.federated.client_num_cpus)
        device = get_device(cfg.federated.client_device)
        return IntrusionClient(client_id, cfg, settings, device).to_client()

    return client_fn
