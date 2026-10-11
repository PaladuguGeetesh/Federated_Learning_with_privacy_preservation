"""
Shared fixtures. Every test runs on tiny synthetic data in a temporary
directory — never on the real dataset — with a temporary copy of the real
config.yaml whose paths point at that directory (so the real config file's
structure is exercised too).
"""

import copy
import json
import warnings
from pathlib import Path

# matplotlib 3.9 calls pyparsing APIs that newer pyparsing versions mark deprecated; the
# warnings fire once at import time, before pytest's ini filters apply. Library noise only.
warnings.filterwarnings("ignore", message=r".*deprecated - use")
import matplotlib.pyplot  # noqa: E402,F401  (import now, under the filter above)

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from core.config import DEFAULT_CONFIG_PATH, Config, load_config
from core.dataset import CLIENT_IDS, LABEL_COL

FEATURES = [f"feature_{i}" for i in range(6)]
CPU = torch.device("cpu")


def synthetic_frame(n: int, seed: int, attack_rate: float = 0.15) -> pd.DataFrame:
    """Separable-ish binary data: attacks are shifted by +1.5 on every feature."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < attack_rate).astype(int)
    X = rng.normal(size=(n, len(FEATURES))) + y[:, None] * 1.5
    df = pd.DataFrame(X, columns=FEATURES)
    df[LABEL_COL] = y
    return df


@pytest.fixture(scope="session")
def dataset_dir(tmp_path_factory) -> Path:
    """client_A/B/C.parquet (600 rows each) + test.parquet (400 rows)."""
    d = tmp_path_factory.mktemp("dataset")
    for i, cid in enumerate(CLIENT_IDS):
        synthetic_frame(600, seed=i).to_parquet(d / f"client_{cid}.parquet", index=False)
    synthetic_frame(400, seed=99).to_parquet(d / "test.parquet", index=False)
    return d


def _deep_update(base: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


@pytest.fixture
def make_config(tmp_path, dataset_dir):
    """make_config(**overrides) -> Config, built from the real config.yaml with test-safe paths."""

    def _make(overrides: dict | None = None, dataset: Path | None = None) -> Config:
        with open(DEFAULT_CONFIG_PATH) as f:
            raw = yaml.safe_load(f)
        raw = copy.deepcopy(raw)
        _deep_update(raw, {
            "device": "cpu",
            "paths": {"results_dir": str(tmp_path / "results"), "graphs_dir": str(tmp_path / "graphs")},
            "data": {"dataset_dir": str(dataset or dataset_dir), "eval_batch_size": 256},
            "training": {"checkpoint_dir": str(tmp_path / "ckpt"), "batch_size": 64, "epochs": 2},
            "random_forest": {"model_path": str(tmp_path / "models" / "rf.joblib"),
                              "hyperparameters": {"n_estimators": 5, "n_jobs": 1}},
            "optuna": {"storage": str(tmp_path / "optuna" / "study.db"),
                       "final_checkpoint_dir": str(tmp_path / "ckpt_tuned"),
                       "pruner": {"n_startup_trials": 1, "n_warmup_epochs": 0},
                       "search_space": {"n_layers": [1, 2], "layer_size": [8, 16], "epochs": [1, 2],
                                        "batch_size": [64, 128]}},
            "federated": {"ann_hyperparameters": str(tmp_path / "results" / "best_hyperparameters.json"),
                          "checkpoint_dir": str(tmp_path / "ckpt_fed")},
        })
        _deep_update(raw, overrides or {})
        path = tmp_path / "config.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(raw, f)
        return load_config(path)

    return _make


@pytest.fixture
def cfg(make_config) -> Config:
    return make_config()


@pytest.fixture
def best_hyperparameters_file(cfg) -> Path:
    """A minimal best_hyperparameters.json (the shape core.optuna_search writes)."""
    path = cfg.federated.ann_hyperparameters
    path.parent.mkdir(parents=True, exist_ok=True)
    training = {k: v for k, v in cfg.training.to_dict().items() if k != "checkpoint_dir"}
    training.update(optimizer="adamw", learning_rate=0.003, batch_size=64)
    with open(path, "w") as f:
        json.dump({"study_name": "test", "best_trial": 0,
                   "hyperparameters": {"ann": {"hidden_sizes": [16, 8], "batch_norm": False, "dropout": 0.1},
                                       "training": training, "pos_weight": None}}, f)
    return path


@pytest.fixture
def test_set_trap(monkeypatch):
    """Make every way of reading the test set raise; returns the list of parquet files read."""
    import core.dataset as dataset_module
    import core.evaluate as evaluate_module

    reads: list[str] = []
    real_read = pd.read_parquet

    def guarded_read(path, *args, **kwargs):
        reads.append(Path(path).name)
        if Path(path).name == "test.parquet":
            raise AssertionError(f"test set read: {path}")
        return real_read(path, *args, **kwargs)

    def trap(*args, **kwargs):
        raise AssertionError("test-set function called")

    monkeypatch.setattr(pd, "read_parquet", guarded_read)
    monkeypatch.setattr(dataset_module, "load_test_set", trap)
    monkeypatch.setattr(evaluate_module, "load_test_set", trap)
    monkeypatch.setattr(evaluate_module, "evaluate_on_test", trap)
    return reads
