"""Training loop: steps, loss aggregation, drop_last, clipping, early stopping in fit, resume."""

import dataclasses

import numpy as np
import pytest
import torch
from torch import nn

from core.ann import ANN
from core.checkpoint import CheckpointManager
from core.config import ANNConfig, EarlyStoppingConfig
from core.dataset import IntrusionDataset, make_dataloader, make_dataloaders
from core.train import (
    build_optimizer,
    compute_pos_weight,
    evaluate_loss,
    fit,
    set_seed,
    train_drop_last,
    train_one_epoch,
)
from tests.conftest import CPU


def _data(n=256, d=5, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    return (rng.normal(size=(n, d)) + y[:, None]).astype(np.float32), y


def _loaders(X, y, batch_size=32, batch_norm=False):
    ds = IntrusionDataset(X, y)
    return make_dataloaders({"train": ds, "val": ds}, batch_size=batch_size, eval_batch_size=64, num_workers=0,
                            pin_memory=False, device=CPU, seed=0,
                            drop_last_train=train_drop_last(len(y), batch_size, batch_norm))


def test_loss_decreases_over_a_few_steps(cfg):
    X, y = _data(64)
    set_seed(0)
    model = ANN(5, ANNConfig((16,), False, 0.0))
    optimizer = build_optimizer(model, dataclasses.replace(cfg.training, learning_rate=0.01))
    criterion = nn.BCEWithLogitsLoss()
    xb, yb = torch.tensor(X), torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    losses = []
    for _ in range(20):  # forward, loss, zero_grad, backward, step
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.8


def test_loss_is_weighted_by_batch_size():
    X, y = _data(10)
    model = ANN(5, ANNConfig((4,), False, 0.0)).eval()
    criterion = nn.BCEWithLogitsLoss()
    loader = make_dataloader(IntrusionDataset(X, y), batch_size=4, shuffle=False, num_workers=0,
                             pin_memory=False, device=CPU, seed=0)  # batches of 4, 4, 2
    with torch.no_grad():
        full = criterion(model(torch.tensor(X)), torch.tensor(y, dtype=torch.float32).unsqueeze(1)).item()
        naive = np.mean([criterion(model(xb), yb).item() for xb, yb in loader])
    assert evaluate_loss(model, loader, criterion, CPU) == pytest.approx(full, abs=1e-6)
    assert abs(naive - full) > 1e-4  # a plain mean of batch means would be wrong here


def test_train_one_epoch_aggregates_all_samples(cfg):
    X, y = _data(100)
    model = ANN(5, ANNConfig((8,), False, 0.0))
    loader = _loaders(X, y, batch_size=32)["train"]
    result = train_one_epoch(model, loader, nn.BCEWithLogitsLoss(), build_optimizer(model, cfg.training), CPU)
    assert result.num_samples == 100 and result.num_batches == 4 and np.isfinite(result.loss)


@pytest.mark.parametrize("n, batch, bn, expected", [(1025, 512, True, True), (1025, 512, False, False),
                                                    (1026, 512, True, False)])
def test_drop_last_rule(n, batch, bn, expected):
    assert train_drop_last(n, batch, bn) is expected


def test_gradient_clipping_caps_norm(cfg):
    X, y = _data(128)
    set_seed(0)
    model = ANN(5, ANNConfig((16,), False, 0.0))
    optimizer = build_optimizer(model, dataclasses.replace(cfg.training, learning_rate=0.05))
    seen, step = [], optimizer.step

    def recording_step(*args, **kwargs):
        seen.append(torch.norm(torch.stack([p.grad.norm() for p in model.parameters()])).item())
        return step(*args, **kwargs)

    optimizer.step = recording_step
    result = train_one_epoch(model, _loaders(X, y)["train"], nn.BCEWithLogitsLoss(), optimizer, CPU, grad_clip_norm=0.01)
    assert max(seen) <= 0.01 + 1e-6
    assert result.mean_grad_norm > 0.01  # the pre-clip norm was larger


def test_compute_pos_weight():
    y = np.array([0] * 85 + [1] * 15)
    assert compute_pos_weight(y, "none", 1.0) is None
    assert compute_pos_weight(y, "balanced", 1.0) == pytest.approx(85 / 15)
    assert compute_pos_weight(y, "balanced_scaled", 2.0) == pytest.approx(2 * 85 / 15)


def test_fit_stops_early_on_a_metric_that_never_improves(cfg):
    X, y = _data(128)
    tc = dataclasses.replace(cfg.training, epochs=10, early_stopping=EarlyStoppingConfig("val_constant", "max", 2, 0.0))
    set_seed(0)
    loaders = _loaders(X, y)
    result = fit(ANN(5, ANNConfig((8,), False, 0.0)), loaders["train"], loaders["val"], training_config=tc,
                 device=CPU, seed=0, feature_names=list("abcde"),
                 extra_val_metrics_fn=lambda *_: {"val_constant": 0.5}, verbose=False)
    assert result.stopped_early and result.last_epoch == 3 and result.best_epoch == 1


def test_resume_matches_uninterrupted_training(cfg, tmp_path):
    X, y = _data(200)
    tc = dataclasses.replace(cfg.training, epochs=4, early_stopping=EarlyStoppingConfig("val_loss", "min", 10, 0.0))
    arch = ANNConfig((8,), True, 0.2)

    def run(epochs, manager, resume=None):
        set_seed(0)
        loaders = _loaders(X, y, batch_norm=True)
        return fit(ANN(5, arch), loaders["train"], loaders["val"], training_config=dataclasses.replace(tc, epochs=epochs),
                   device=CPU, seed=0, feature_names=list("abcde"), checkpoint_manager=manager, resume_from=resume,
                   verbose=False)

    straight = run(4, CheckpointManager(tmp_path / "straight"))
    manager = CheckpointManager(tmp_path / "resumed")
    run(2, manager)
    resumed = run(4, manager, resume=manager.load_latest())
    key = lambda r: [(h["train_loss"], h["val_loss"]) for h in r.history]  # noqa: E731
    assert key(straight) == key(resumed)
