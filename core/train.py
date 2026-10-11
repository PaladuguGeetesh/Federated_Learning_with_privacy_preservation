"""
Reusable training loop for the ANN / Logistic Regression models.

Per batch: forward -> BCEWithLogitsLoss -> zero_grad -> backward ->
[clip_grad_norm_] -> optimizer step. Per epoch: weighted-average train loss,
validation loss in eval mode under torch.no_grad(), optional extra validation
metrics, early stopping, and best/latest checkpoints.

Loss aggregation: each batch's mean loss is multiplied by that batch's size and
the sum is divided by the total number of samples, so a smaller final batch is
weighted correctly (not a plain mean of per-batch means).

Run centralized ANN training from the project root (uses config.yaml):
    python -m core.train                 # fresh run
    python -m core.train --resume        # continue from <checkpoint_dir>/latest.pt
"""

import argparse
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from core.ann import ANN
from core.checkpoint import (
    Checkpoint,
    CheckpointManager,
    capture_rng_state,
    describe_architecture,
    make_checkpoint,
    restore_rng_state,
)
from core.config import DEFAULT_CONFIG_PATH, TrainingConfig, load_config
from core.dataset import (
    LABEL_MAPPING,
    get_device,
    load_centralized_data,
    make_dataloaders,
    to_device,
)
from core.early_stopping import EarlyStopping

# Computes extra per-epoch validation metrics, e.g. {"val_pr_auc": 0.97} (Phase 5/6).
ExtraMetricsFn = Callable[[nn.Module, DataLoader, torch.device], dict[str, float]]


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU + CUDA/MPS) RNGs.

    Note: some GPU kernels (notably on MPS) are not bit-for-bit deterministic,
    so runs on GPU may differ slightly even with identical seeds; CPU runs are
    reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA and MPS generators


def train_drop_last(num_samples: int, batch_size: int, batch_norm: bool) -> bool:
    """Whether the training loader must drop its final batch.

    BatchNorm1d cannot compute batch statistics from a single sample in train
    mode, so the last batch is dropped only when BatchNorm is on AND that batch
    would contain exactly one row. Otherwise no rows are dropped.
    """
    if batch_norm and num_samples == 1:
        raise ValueError("Cannot train a BatchNorm model on a single sample")
    return batch_norm and num_samples % batch_size == 1


def build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """Create the SGD / Adam / AdamW optimizer described by the training config."""
    params = model.parameters()
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            params, lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay
        )
    if config.optimizer == "adam":
        return torch.optim.Adam(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    raise ValueError(f"Unknown optimizer {config.optimizer!r}")


def build_criterion(pos_weight: float | None, device: torch.device) -> nn.BCEWithLogitsLoss:
    """BCEWithLogitsLoss, optionally up-weighting the positive (Attack) class."""
    if pos_weight is None:
        return nn.BCEWithLogitsLoss()
    if pos_weight <= 0:
        raise ValueError(f"pos_weight must be > 0, got {pos_weight}")
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))


@dataclass
class EpochResult:
    """Outcome of one training epoch."""

    loss: float  # sample-weighted mean training loss
    num_samples: int
    num_batches: int
    mean_grad_norm: float | None  # mean pre-clipping total grad norm (only when clipping)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float | None = None,
) -> EpochResult:
    """Run one epoch of training and return the sample-weighted mean loss."""
    model.train()
    # Accumulate on-device to avoid a host sync per batch; read once at the end.
    loss_sum = torch.zeros((), device=device)
    grad_norm_sum = torch.zeros((), device=device)
    num_samples = 0
    num_batches = 0

    for batch in loader:
        features, labels = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)  # mean over this batch
        loss.backward()
        if grad_clip_norm is not None:
            # returns the total norm *before* clipping
            grad_norm_sum += nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        batch_size = features.shape[0]
        loss_sum += loss.detach() * batch_size
        num_samples += batch_size
        num_batches += 1

    if num_samples == 0:
        raise RuntimeError("Training loader produced no batches")
    mean_loss = loss_sum.item() / num_samples
    if not math.isfinite(mean_loss):
        raise RuntimeError(f"Training loss became non-finite ({mean_loss}); try a lower learning rate or gradient clipping")
    return EpochResult(
        loss=mean_loss,
        num_samples=num_samples,
        num_batches=num_batches,
        mean_grad_norm=grad_norm_sum.item() / num_batches if grad_clip_norm is not None else None,
    )


def evaluate_loss(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> float:
    """Sample-weighted mean loss over a loader, in eval mode with gradients disabled."""
    model.eval()
    loss_sum = torch.zeros((), device=device)
    num_samples = 0
    with torch.no_grad():
        for batch in loader:
            features, labels = to_device(batch, device)
            loss_sum += criterion(model(features), labels) * features.shape[0]
            num_samples += features.shape[0]
    if num_samples == 0:
        raise RuntimeError("Evaluation loader produced no batches")
    return loss_sum.item() / num_samples


@dataclass
class FitResult:
    """Summary of a fit() call."""

    history: list[dict[str, Any]]
    metric: str
    best_epoch: int | None
    best_value: float | None
    last_epoch: int
    stopped_early: bool
    optimizer: torch.optim.Optimizer = field(repr=False)


def _check_resume_compatible(checkpoint: Checkpoint, model: nn.Module, feature_names: list[str]) -> None:
    if checkpoint.architecture != describe_architecture(model):
        raise ValueError(
            f"Checkpoint architecture {checkpoint.architecture} does not match the current model "
            f"{describe_architecture(model)}"
        )
    if checkpoint.feature_names != list(feature_names):
        raise ValueError("Checkpoint feature schema (column names/order) does not match the current data")


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    training_config: TrainingConfig,
    device: torch.device,
    seed: int,
    feature_names: list[str],
    scaler: StandardScaler | None = None,
    checkpoint_manager: CheckpointManager | None = None,
    resume_from: Checkpoint | None = None,
    extra_val_metrics_fn: ExtraMetricsFn | None = None,
    pos_weight: float | None = None,
    restore_best: bool = True,
    checkpoint_extra: dict[str, Any] | None = None,
    verbose: bool = True,
) -> FitResult:
    """Train `model` with early stopping, checkpointing and optional resuming.

    Args:
        model: Model already moved to `device`. Seed before constructing it so
            weight initialization is reproducible.
        train_loader / val_loader: From core.dataset.make_dataloaders.
        training_config: Optimizer, epochs, clipping and early-stopping settings.
        device: Device the model lives on (from core.dataset.get_device).
        seed: Seed for a fresh run (ignored when resuming: RNG state is restored instead).
        feature_names: Column order the model was trained on, stored in checkpoints.
        scaler: The StandardScaler fit on this run's training data, stored in checkpoints.
        checkpoint_manager: If given, writes latest.pt every epoch and best.pt on improvement.
        resume_from: A "latest" checkpoint to continue from.
        extra_val_metrics_fn: Optional extra validation metrics merged into each epoch's history.
        pos_weight: Optional BCEWithLogitsLoss positive-class weight (used for train and val loss).
        restore_best: Load the best epoch's weights into `model` before returning.
        checkpoint_extra: Extra info stored in every checkpoint (e.g. tuned hyperparameters).
    """
    criterion = build_criterion(pos_weight, device)
    optimizer = build_optimizer(model, training_config)
    stopper = EarlyStopping.from_config(training_config.early_stopping)
    loader_generator = getattr(train_loader, "generator", None)
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    start_epoch = 1

    if resume_from is not None:
        _check_resume_compatible(resume_from, model, feature_names)
        model.load_state_dict(resume_from.model_state)
        if resume_from.optimizer_state is not None:
            optimizer.load_state_dict(resume_from.optimizer_state)
        if resume_from.early_stopping_state is not None:
            stopper.load_state_dict(resume_from.early_stopping_state)
        if resume_from.rng_state is not None:
            restore_rng_state(resume_from.rng_state, loader_generator)
        history = [dict(h) for h in resume_from.history]
        start_epoch = resume_from.epoch + 1
        if checkpoint_manager is not None and checkpoint_manager.has_best():
            best_state = checkpoint_manager.load_best().model_state
        if verbose:
            print(f"Resuming from epoch {resume_from.epoch} (best {stopper.metric}={stopper.best_value} at epoch {stopper.best_epoch})")
    else:
        set_seed(seed)

    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, training_config.epochs + 1):
        if stopper.should_stop:
            break
        start = time.time()
        train_result = train_one_epoch(
            model, train_loader, criterion, optimizer, device, training_config.grad_clip_norm
        )
        record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_result.loss,
            "val_loss": evaluate_loss(model, val_loader, criterion, device),
        }
        if extra_val_metrics_fn is not None:
            record.update({k: float(v) for k, v in extra_val_metrics_fn(model, val_loader, device).items()})
        if train_result.mean_grad_norm is not None:
            record["mean_grad_norm_preclip"] = train_result.mean_grad_norm
        record["lr"] = optimizer.param_groups[0]["lr"]
        record["epoch_seconds"] = time.time() - start
        history.append(record)
        last_epoch = epoch

        if stopper.metric not in record:
            raise KeyError(
                f"Early-stopping metric {stopper.metric!r} is not in the epoch record "
                f"(available: {sorted(record)}). Provide it via extra_val_metrics_fn or change the config."
            )
        improved = stopper.step(record[stopper.metric], epoch)
        if improved:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if checkpoint_manager is not None:
            checkpoint = make_checkpoint(
                epoch=epoch,
                model=model,
                training_config=training_config.to_dict(),
                history=history,
                feature_names=feature_names,
                label_mapping=LABEL_MAPPING,
                seed=seed,
                optimizer=optimizer,
                early_stopping_state=stopper.state_dict(),
                rng_state=capture_rng_state(loader_generator),
                scaler=scaler,
                extra={**(checkpoint_extra or {}), "pos_weight": pos_weight},
            )
            if improved:
                checkpoint_manager.save_best(checkpoint)
            checkpoint_manager.save_latest(checkpoint)

        if verbose:
            mark = " *" if improved else ""  # * = new best by the early-stopping metric
            monitored = "" if stopper.metric == "val_loss" else f"  {stopper.metric} {record[stopper.metric]:.5f}"
            print(
                f"epoch {epoch:>3}/{training_config.epochs}  train_loss {record['train_loss']:.5f}  "
                f"val_loss {record['val_loss']:.5f}{monitored}{mark}  ({record['epoch_seconds']:.1f}s)"
            )
        if stopper.should_stop:
            if verbose:
                print(f"Early stopping: no {stopper.metric} improvement for {stopper.patience} epochs "
                      f"(best {stopper.best_value:.5f} at epoch {stopper.best_epoch})")
            break

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)

    return FitResult(
        history=history,
        metric=stopper.metric,
        best_epoch=stopper.best_epoch,
        best_value=stopper.best_value,
        last_epoch=last_epoch,
        stopped_early=stopper.should_stop,
        optimizer=optimizer,
    )


def main() -> None:
    """Centralized ANN training on the pooled client train/val data (never touches the test set)."""
    parser = argparse.ArgumentParser(description="Centralized ANN training")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", action="store_true", help="continue from <checkpoint_dir>/latest.pt")
    group.add_argument("--overwrite", action="store_true", help="start fresh, replacing existing checkpoints")
    args = parser.parse_args()

    cfg = load_config(args.config)
    manager = CheckpointManager(cfg.training.checkpoint_dir)
    if manager.has_latest() and not (args.resume or args.overwrite):
        raise SystemExit(
            f"Checkpoints already exist in {manager.directory}. Pass --resume to continue or --overwrite to start over."
        )
    resume_from = manager.load_latest() if args.resume else None

    set_seed(cfg.seed)
    device = get_device(cfg.device)
    print(f"Device: {device}")

    data = load_centralized_data(cfg.data.dataset_dir, cfg.data.val_fraction, cfg.seed)
    loaders = make_dataloaders(
        {"train": data.train.to_dataset(), "val": data.val.to_dataset()},
        batch_size=cfg.training.batch_size,
        eval_batch_size=cfg.data.eval_batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        device=device,
        seed=cfg.seed,
        drop_last_train=train_drop_last(len(data.train.y), cfg.training.batch_size, cfg.ann.batch_norm),
    )
    print(f"Train rows: {len(data.train.y):,}  Val rows: {len(data.val.y):,}  Features: {len(data.feature_names)}")

    model = ANN(len(data.feature_names), cfg.ann).to(device)
    result = fit(
        model,
        loaders["train"],
        loaders["val"],
        training_config=cfg.training,
        device=device,
        seed=cfg.seed,
        feature_names=data.feature_names,
        scaler=data.scaler,
        checkpoint_manager=manager,
        resume_from=resume_from,
    )
    print(
        f"Done. Best {result.metric} = {result.best_value} at epoch {result.best_epoch}; "
        f"last epoch {result.last_epoch}; stopped early: {result.stopped_early}"
    )
    print(f"Checkpoints: {manager.best_path}, {manager.latest_path}")


if __name__ == "__main__":
    main()
