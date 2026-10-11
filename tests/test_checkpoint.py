"""Checkpoint save/load round trip."""

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from core.ann import ANN
from core.checkpoint import load_checkpoint, make_checkpoint, save_checkpoint
from core.config import ANNConfig
from core.dataset import LABEL_MAPPING
from core.logistic_regression import LogisticRegression
from core.train import build_optimizer

FEATURES = list("abcdef")


def _trained(model, cfg):
    optimizer = build_optimizer(model, cfg.training)
    x, y = torch.randn(64, 6), (torch.rand(64, 1) < 0.3).float()
    for _ in range(3):
        optimizer.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(model(x), y).backward()
        optimizer.step()
    return optimizer


@pytest.mark.parametrize("model_fn", [lambda: ANN(6, ANNConfig((8, 4), True, 0.1)), lambda: LogisticRegression(6)])
def test_round_trip_reproduces_outputs(tmp_path, cfg, model_fn):
    torch.manual_seed(0)
    model = model_fn()
    optimizer = _trained(model, cfg)  # non-trivial weights, BatchNorm statistics and optimizer state
    scaler = StandardScaler().fit(np.random.default_rng(0).normal(size=(50, 6)))
    checkpoint = make_checkpoint(epoch=3, model=model, training_config=cfg.training.to_dict(),
                                 history=[{"epoch": 1, "train_loss": 0.5}], feature_names=FEATURES,
                                 label_mapping=LABEL_MAPPING, seed=cfg.seed, optimizer=optimizer, scaler=scaler)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(checkpoint, path)
    loaded = load_checkpoint(path)  # torch.load(weights_only=True) inside

    restored = loaded.build_model()
    model.eval(), restored.eval()
    x = torch.randn(32, 6)
    with torch.no_grad():
        assert torch.equal(model(x), restored(x))
    assert loaded.epoch == 3 and loaded.feature_names == FEATURES and loaded.label_mapping == LABEL_MAPPING
    assert np.allclose(loaded.build_scaler().scale_, scaler.scale_)
    opt2 = build_optimizer(restored, cfg.training)
    opt2.load_state_dict(loaded.optimizer_state)
    for state_a, state_b in zip(optimizer.state_dict()["state"].values(), opt2.state_dict()["state"].values()):
        assert torch.equal(state_a["exp_avg"], state_b["exp_avg"])


def test_feature_count_must_match_model(cfg):
    with pytest.raises(ValueError, match="input_dim"):
        make_checkpoint(epoch=1, model=LogisticRegression(6), training_config={}, history=[],
                        feature_names=["only", "three", "names"], label_mapping=LABEL_MAPPING, seed=0)
