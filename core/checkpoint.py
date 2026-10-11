"""
Save / load training checkpoints for inference and for resuming training.

A checkpoint bundles everything needed to rebuild and continue a run:
model weights, optimizer state, architecture (model type + input_dim + config),
training config, per-epoch history, epoch number, early-stopping state, RNG
states, feature schema (column order + count), label mapping, and the fitted
StandardScaler statistics (needed to scale new data exactly as in training).

Two files are kept per run directory:
    best.pt    — state at the best epoch by the early-stopping metric
    latest.pt  — state at the most recent epoch (use this to resume)

Checkpoints contain only tensors and plain Python types, so they load with
torch.load(weights_only=True) — no arbitrary pickled objects are executed.
"""

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from core.ann import ANN
from core.config import ANNConfig
from core.logistic_regression import LogisticRegression

CHECKPOINT_FORMAT_VERSION = 1
BEST_FILENAME = "best.pt"
LATEST_FILENAME = "latest.pt"


# ---------------------------------------------------------------------------
# Architecture description <-> model
# ---------------------------------------------------------------------------


def describe_architecture(model: nn.Module) -> dict[str, Any]:
    """Plain-dict description from which build_model() can rebuild an identical model."""
    if isinstance(model, ANN):
        return {"type": "ann", "input_dim": model.input_dim, "ann": model.config.to_dict()}
    if isinstance(model, LogisticRegression):
        return {"type": "logistic_regression", "input_dim": model.input_dim}
    raise TypeError(f"Unsupported model type for checkpointing: {type(model).__name__}")


def build_model(architecture: dict[str, Any]) -> nn.Module:
    """Rebuild an (untrained) model from a describe_architecture() dict."""
    model_type = architecture.get("type")
    if model_type == "ann":
        return ANN(architecture["input_dim"], ANNConfig.from_dict(architecture["ann"]))
    if model_type == "logistic_regression":
        return LogisticRegression(architecture["input_dim"])
    raise ValueError(f"Unknown model type in checkpoint: {model_type!r}")


# ---------------------------------------------------------------------------
# RNG state (so a resumed run continues the same random stream)
# ---------------------------------------------------------------------------


def capture_rng_state(loader_generator: torch.Generator | None = None) -> dict[str, Any]:
    """Snapshot python / numpy / torch (and the train DataLoader's shuffle) RNG states."""
    np_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        # numpy's state holds a uint32 ndarray; store it as an int64 tensor (torch 2.3
        # cannot serialize uint32 tensors) to stay weights_only-safe
        "numpy": (np_state[0], torch.from_numpy(np_state[1].astype(np.int64)), *np_state[2:]),
        "torch": torch.get_rng_state(),
        "loader": loader_generator.get_state() if loader_generator is not None else None,
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(state: dict[str, Any], loader_generator: torch.Generator | None = None) -> None:
    """Inverse of capture_rng_state."""
    random.setstate(state["python"])
    np_state = state["numpy"]
    np.random.set_state((np_state[0], np_state[1].numpy().astype(np.uint32), *np_state[2:]))
    torch.set_rng_state(state["torch"])
    if loader_generator is not None and state.get("loader") is not None:
        loader_generator.set_state(state["loader"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


# ---------------------------------------------------------------------------
# Scaler state (stored as tensors so checkpoints stay weights_only-safe)
# ---------------------------------------------------------------------------


def scaler_to_state(scaler: StandardScaler) -> dict[str, Any]:
    """Serialize a fitted StandardScaler's statistics."""
    return {
        "mean": torch.from_numpy(scaler.mean_.copy()),
        "var": torch.from_numpy(scaler.var_.copy()),
        "scale": torch.from_numpy(scaler.scale_.copy()),
        "n_samples_seen": int(np.asarray(scaler.n_samples_seen_).sum()),
    }


def scaler_from_state(state: dict[str, Any]) -> StandardScaler:
    """Rebuild a fitted StandardScaler from scaler_to_state() output."""
    scaler = StandardScaler()
    scaler.mean_ = state["mean"].numpy().copy()
    scaler.var_ = state["var"].numpy().copy()
    scaler.scale_ = state["scale"].numpy().copy()
    scaler.n_features_in_ = len(scaler.mean_)
    scaler.n_samples_seen_ = state["n_samples_seen"]
    return scaler


# ---------------------------------------------------------------------------
# Checkpoint record
# ---------------------------------------------------------------------------


@dataclass
class Checkpoint:
    """Everything saved for one training state."""

    epoch: int
    model_state: dict[str, torch.Tensor]
    architecture: dict[str, Any]
    training_config: dict[str, Any]
    history: list[dict[str, Any]]
    feature_names: list[str]
    label_mapping: dict[int, str]
    seed: int
    optimizer_state: dict[str, Any] | None = None
    early_stopping_state: dict[str, Any] | None = None
    rng_state: dict[str, Any] | None = None
    scaler_state: dict[str, Any] | None = None  # see scaler_to_state()
    # Free-form extras, e.g. the hyperparameters Optuna selected.
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def num_features(self) -> int:
        return len(self.feature_names)

    def to_payload(self) -> dict[str, Any]:
        return {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "epoch": self.epoch,
            "model_state": self.model_state,
            "architecture": self.architecture,
            "training_config": self.training_config,
            "history": self.history,
            "feature_schema": {"names": list(self.feature_names), "count": len(self.feature_names)},
            "label_mapping": dict(self.label_mapping),
            "seed": self.seed,
            "optimizer_state": self.optimizer_state,
            "early_stopping_state": self.early_stopping_state,
            "rng_state": self.rng_state,
            "scaler_state": self.scaler_state,
            "extra": self.extra,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Checkpoint":
        version = payload.get("format_version")
        if version != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(f"Unsupported checkpoint format_version {version!r}")
        schema = payload["feature_schema"]
        if len(schema["names"]) != schema["count"]:
            raise ValueError("Corrupt checkpoint: feature_schema count does not match names")
        if schema["count"] != payload["architecture"]["input_dim"]:
            raise ValueError("Corrupt checkpoint: feature count does not match model input_dim")
        return cls(
            epoch=payload["epoch"],
            model_state=payload["model_state"],
            architecture=payload["architecture"],
            training_config=payload["training_config"],
            history=payload["history"],
            feature_names=list(schema["names"]),
            label_mapping=dict(payload["label_mapping"]),
            seed=payload["seed"],
            optimizer_state=payload["optimizer_state"],
            early_stopping_state=payload["early_stopping_state"],
            rng_state=payload["rng_state"],
            scaler_state=payload["scaler_state"],
            extra=payload["extra"],
        )

    def build_scaler(self) -> StandardScaler:
        """Rebuild the training-data StandardScaler stored with this checkpoint."""
        if self.scaler_state is None:
            raise ValueError("This checkpoint has no scaler_state")
        return scaler_from_state(self.scaler_state)

    def build_model(self, device: torch.device | None = None) -> nn.Module:
        """Rebuild the model and load its weights (for inference or resuming)."""
        model = build_model(self.architecture)
        model.load_state_dict(self.model_state, strict=True)
        return model.to(device) if device is not None else model


def _to_cpu(obj: Any) -> Any:
    """Recursively copy every tensor in a (nested) state dict to CPU.

    Gives the checkpoint its own snapshot (later training steps cannot mutate it)
    and makes it loadable on any device.
    """
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu(v) for v in obj)
    return obj


def make_checkpoint(
    *,
    epoch: int,
    model: nn.Module,
    training_config: dict[str, Any],
    history: list[dict[str, Any]],
    feature_names: list[str],
    label_mapping: dict[int, str],
    seed: int,
    optimizer: torch.optim.Optimizer | None = None,
    early_stopping_state: dict[str, Any] | None = None,
    rng_state: dict[str, Any] | None = None,
    scaler: StandardScaler | None = None,
    extra: dict[str, Any] | None = None,
) -> Checkpoint:
    """Snapshot the current training state into a Checkpoint (weights copied to CPU)."""
    architecture = describe_architecture(model)
    if scaler is not None and scaler.n_features_in_ != len(feature_names):
        raise ValueError(f"Scaler expects {scaler.n_features_in_} features, data has {len(feature_names)}")
    if architecture["input_dim"] != len(feature_names):
        raise ValueError(
            f"Model input_dim {architecture['input_dim']} != number of features {len(feature_names)}"
        )
    return Checkpoint(
        epoch=epoch,
        model_state=_to_cpu(model.state_dict()),
        architecture=architecture,
        training_config=training_config,
        history=[dict(h) for h in history],
        feature_names=list(feature_names),
        label_mapping=dict(label_mapping),
        seed=seed,
        optimizer_state=_to_cpu(optimizer.state_dict()) if optimizer is not None else None,
        early_stopping_state=early_stopping_state,
        rng_state=rng_state,
        scaler_state=scaler_to_state(scaler) if scaler is not None else None,
        extra=dict(extra or {}),
    )


def save_checkpoint(checkpoint: Checkpoint, path: Path) -> None:
    """Write a checkpoint atomically (temp file + rename), so a crash never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint.to_payload(), tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> Checkpoint:
    """Load a checkpoint written by save_checkpoint."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=True)
    return Checkpoint.from_payload(payload)


class CheckpointManager:
    """Keeps best.pt and latest.pt for one run in a directory."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.best_path = self.directory / BEST_FILENAME
        self.latest_path = self.directory / LATEST_FILENAME

    def save_best(self, checkpoint: Checkpoint) -> None:
        save_checkpoint(checkpoint, self.best_path)

    def save_latest(self, checkpoint: Checkpoint) -> None:
        save_checkpoint(checkpoint, self.latest_path)

    def load_best(self) -> Checkpoint:
        return load_checkpoint(self.best_path)

    def load_latest(self) -> Checkpoint:
        return load_checkpoint(self.latest_path)

    def has_latest(self) -> bool:
        return self.latest_path.exists()

    def has_best(self) -> bool:
        return self.best_path.exists()
