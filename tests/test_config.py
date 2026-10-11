"""config.yaml loading and strict validation."""

import copy

import pytest
import yaml

from core.config import DEFAULT_CONFIG_PATH, load_config


def test_real_config_loads():
    cfg = load_config()
    assert cfg.optuna.search_space.optimizer == ("sgd", "adam", "adamw")
    assert cfg.federated.local_epochs == 3
    assert cfg.random_forest.hyperparameters.random_state == cfg.seed
    assert cfg.paths.results_dir.name == "results"


def _write(tmp_path, raw):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.mark.parametrize("mutate, message", [
    (lambda r: r["model"]["ann"].update(hiden_sizes=[1]), "unknown keys"),
    (lambda r: r.update(extra_section={}), "unknown keys"),
    (lambda r: r["data"].pop("val_fraction"), "missing keys"),
    (lambda r: r["model"]["ann"].update(dropout=1.0), "dropout"),
    (lambda r: r["training"].update(optimizer="rmsprop"), "optimizer"),
    (lambda r: r["training"]["early_stopping"].update(mode="up"), "mode"),
    (lambda r: r["evaluation"].update(threshold_strategy="best"), "threshold_strategy"),
    (lambda r: r["random_forest"]["hyperparameters"].update(n_estimators=0), "n_estimators"),
    (lambda r: r["optuna"]["search_space"].update(learning_rate=[0.1, 0.01]), "low"),
    (lambda r: r["federated"].update(model="svm"), "federated.model"),
    (lambda r: r.update(device="tpu"), "device"),
])
def test_bad_config_rejected(tmp_path, mutate, message):
    raw = copy.deepcopy(yaml.safe_load(DEFAULT_CONFIG_PATH.read_text()))
    mutate(raw)
    with pytest.raises(ValueError, match=message):
        load_config(_write(tmp_path, raw))
