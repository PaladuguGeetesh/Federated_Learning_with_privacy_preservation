"""
Pieces the federated client and server must agree on: which model and local
training settings a run uses, how the model is built, and how its tensors are
laid out in the flat list of numpy arrays Flower transmits.

Parameter layout (both directions, every round):

    [ trainable parameters (model.named_parameters() order) ...,
      buffers              (model.named_buffers() order)    ... ]

Trainable parameters are learned by gradient descent and averaged with FedAvg.
Buffers exist only when BatchNorm is enabled (running_mean, running_var,
num_batches_tracked): they are running statistics, not gradient-trained
weights, so the server aggregates them on a separate path
(federated/aggregation.py). ParameterLayout records which array is which so
the two kinds are never mixed up.
"""

import dataclasses
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from core.ann import ANN
from core.config import ANNConfig, Config, EarlyStoppingConfig, TrainingConfig
from core.logistic_regression import LogisticRegression

BUFFER_KINDS = ("running_mean", "running_var", "num_batches_tracked")


# ---------------------------------------------------------------------------
# Run settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FederatedSettings:
    """Model type plus the architecture / local-training settings for one federated run."""

    model: str  # "ann" | "logistic_regression"
    ann: ANNConfig | None  # None for logistic regression
    training: TrainingConfig  # optimizer, lr, weight decay, batch size, clipping, class weighting
    source: str  # where these settings came from (for traceability in logs/results)

    def describe(self) -> dict[str, Any]:
        training = self.training.to_dict()
        training.pop("checkpoint_dir")
        training.pop("early_stopping")  # local training runs a fixed number of epochs
        training.pop("epochs")  # replaced by federated.local_epochs
        return {
            "model": self.model,
            "ann": self.ann.to_dict() if self.ann else None,
            "local_training": training,
            "source": self.source,
        }


def resolve_settings(cfg: Config, model: str | None = None) -> FederatedSettings:
    """Pick the model and its settings for a federated run.

    ANN: architecture AND local-training settings come from the Optuna result
    file (federated.ann_hyperparameters). That file holds the configuration
    selected by the CENTRALIZED search (trial 14: hidden [312, 289, 201], no
    BatchNorm, dropout ~0.099, AdamW). It must be re-validated under federated
    training before being treated as final for the federated setting.

    Logistic regression: the sanity-check baseline; trained with the training
    section of config.yaml.
    """
    model = model or cfg.federated.model
    if model == "logistic_regression":
        return FederatedSettings(model, None, cfg.training, "config.yaml training section")
    if model != "ann":
        raise ValueError(f"Unknown federated model {model!r}")

    path = cfg.federated.ann_hyperparameters
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run python -m core.optuna_search first")
    with open(path) as f:
        tuned = json.load(f)
    try:
        hp = tuned["hyperparameters"]
        ann_cfg = ANNConfig.from_dict(hp["ann"])
        t = hp["training"]
        training = dataclasses.replace(
            cfg.training,
            optimizer=t["optimizer"],
            learning_rate=float(t["learning_rate"]),
            weight_decay=float(t["weight_decay"]),
            momentum=float(t["momentum"]),
            batch_size=int(t["batch_size"]),
            grad_clip_norm=None if t["grad_clip_norm"] is None else float(t["grad_clip_norm"]),
            class_weighting=t["class_weighting"],
            pos_weight_scale=float(t["pos_weight_scale"]),
            early_stopping=EarlyStoppingConfig(**t["early_stopping"]),
        )
    except (KeyError, TypeError) as e:
        raise ValueError(f"{path} is not a valid best_hyperparameters.json: {e}") from e
    source = f"Optuna-selected config: {path.name}, study {tuned.get('study_name')!r}, trial {tuned.get('best_trial')} (centralized search)"
    return FederatedSettings("ann", ann_cfg, training, source)


def build_model(settings: FederatedSettings, input_dim: int) -> nn.Module:
    """Build the run's model. Every client and the server use this, so shapes always match."""
    if settings.model == "ann":
        return ANN(input_dim, settings.ann)
    return LogisticRegression(input_dim)


# ---------------------------------------------------------------------------
# Parameter layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParameterLayout:
    """Names and shapes of the transmitted arrays: trainable parameters first, then buffers."""

    trainable: tuple[tuple[str, tuple[int, ...]], ...]
    buffers: tuple[tuple[str, tuple[int, ...]], ...]

    @classmethod
    def from_model(cls, model: nn.Module) -> "ParameterLayout":
        trainable = tuple((n, tuple(p.shape)) for n, p in model.named_parameters())
        buffers = tuple((n, tuple(b.shape)) for n, b in model.named_buffers())
        state_keys = set(model.state_dict())
        if {n for n, _ in trainable} | {n for n, _ in buffers} != state_keys:
            raise ValueError("Model state_dict is not exactly its parameters plus buffers")
        for name, _ in buffers:
            if name.rsplit(".", 1)[-1] not in BUFFER_KINDS:
                raise ValueError(f"Unsupported buffer {name!r}: only BatchNorm buffers are handled")
        return cls(trainable, buffers)

    @property
    def num_trainable(self) -> int:
        return len(self.trainable)

    @property
    def has_buffers(self) -> bool:
        return len(self.buffers) > 0

    def buffer_kinds(self) -> list[str]:
        return [name.rsplit(".", 1)[-1] for name, _ in self.buffers]

    def split(self, arrays: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Split a transmitted list into (trainable, buffers), checking every shape."""
        expected = self.trainable + self.buffers
        if len(arrays) != len(expected):
            raise ValueError(f"Expected {len(expected)} arrays, got {len(arrays)}")
        for arr, (name, shape) in zip(arrays, expected):
            if tuple(arr.shape) != shape:
                raise ValueError(f"Shape mismatch for {name}: expected {shape}, got {tuple(arr.shape)}")
        return list(arrays[: self.num_trainable]), list(arrays[self.num_trainable :])


def get_arrays(model: nn.Module) -> list[np.ndarray]:
    """Model -> transmitted arrays (trainable parameters, then buffers)."""
    params = [p.detach().cpu().numpy().copy() for _, p in model.named_parameters()]
    buffers = [b.detach().cpu().numpy().copy() for _, b in model.named_buffers()]
    return params + buffers


def set_arrays(model: nn.Module, layout: ParameterLayout, arrays: list[np.ndarray]) -> None:
    """Transmitted arrays -> model (strict: every key and shape must match)."""
    trainable, buffers = layout.split(arrays)
    names = [n for n, _ in layout.trainable] + [n for n, _ in layout.buffers]
    device = next(model.parameters()).device
    state = OrderedDict(
        (name, torch.as_tensor(arr, device=device)) for name, arr in zip(names, trainable + buffers)
    )
    model.load_state_dict(state, strict=True)
