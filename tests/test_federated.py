"""FedAvg aggregation math, parameter layout, client fit and server-side round evaluation (no Ray)."""

import inspect

import numpy as np
import pytest
import torch
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays

import federated.server as S
from core.ann import ANN
from core.checkpoint import CheckpointManager, load_checkpoint
from core.config import ANNConfig
from core.dataset import CLIENT_IDS, SplitArrays, combine_scalers, load_all_clients, load_raw_validation, apply_scaler
from federated.aggregation import aggregate_bn_buffers, fedavg_trainable
from federated.client import IntrusionClient
from federated.common import FederatedSettings, ParameterLayout, build_model, get_arrays, resolve_settings, set_arrays
from tests.conftest import CPU


def test_fedavg_is_sample_weighted():
    a = ([np.array([1.0, 2.0]), np.array([[0.0]])], 100)
    b = ([np.array([3.0, 6.0]), np.array([[10.0]])], 300)
    out = fedavg_trainable([a, b])
    assert np.allclose(out[0], [2.5, 5.0]) and np.allclose(out[1], [[7.5]])


def test_bn_buffers_combine_as_pooled_statistics():
    rng = np.random.default_rng(0)
    xa, xb = rng.normal(0, 1, (400, 3)), rng.normal(2, 3, (1200, 3))
    ua = ([xa.mean(0), xa.var(0), np.array(7)], 400)
    ub = ([xb.mean(0), xb.var(0), np.array(9)], 1200)
    mean, var, count = aggregate_bn_buffers([ua, ub], ["running_mean", "running_var", "num_batches_tracked"])
    pooled = np.vstack([xa, xb])
    assert np.allclose(mean, pooled.mean(0))
    assert np.allclose(var, pooled.var(0))  # includes the between-client spread of means
    assert not np.allclose(var, 0.25 * xa.var(0) + 0.75 * xb.var(0))  # a plain weighted mean would be wrong
    assert int(count) == 9


@pytest.mark.parametrize("updates, message", [
    ([([np.ones(2)], 0), ([np.ones(2)], 5)], "positive sample count"),
    ([([np.ones(2)], 1), ([np.ones(2), np.ones(2)], 1)], "different numbers of arrays"),
    ([], "No client updates"),
])
def test_aggregation_rejects_bad_updates(updates, message):
    with pytest.raises(ValueError, match=message):
        fedavg_trainable(updates)


def test_layout_round_trip_and_shape_check():
    model = ANN(6, ANNConfig((8,), True, 0.0))
    layout = ParameterLayout.from_model(model)
    assert layout.num_trainable == 6 and layout.buffer_kinds() == ["running_mean", "running_var", "num_batches_tracked"]
    arrays = [a + 1 for a in get_arrays(model)]
    other = ANN(6, ANNConfig((8,), True, 0.0))
    set_arrays(other, layout, arrays)
    assert all(np.array_equal(x, y) for x, y in zip(get_arrays(other), arrays))
    with pytest.raises(ValueError, match="Shape mismatch"):
        layout.split([np.zeros((1, 1))] + arrays[1:])


def _fit_res(arrays, n):
    return FitRes(status=Status(Code.OK, ""), parameters=ndarrays_to_parameters(arrays), num_examples=n,
                  metrics={"client_id": "x", "train_loss": 0.1})


@pytest.mark.parametrize("batch_norm, expected_buffers", [(False, 0), (True, 3)])
def test_strategy_uses_buffer_path_only_with_batchnorm(batch_norm, expected_buffers):
    model = ANN(6, ANNConfig((8,), batch_norm, 0.0))
    layout = ParameterLayout.from_model(model)
    strategy = S.FedAvgWithBuffers(layout)
    arrays = get_arrays(model)
    params, metrics = strategy.aggregate_fit(1, [(None, _fit_res(arrays, 10)), (None, _fit_res(arrays, 30))], [])
    assert metrics["bn_buffer_arrays_aggregated"] == expected_buffers
    assert all(np.allclose(x, y) for x, y in zip(parameters_to_ndarrays(params), arrays))
    with pytest.raises(RuntimeError, match="failed"):
        strategy.aggregate_fit(2, [], [RuntimeError("client crashed")])


def test_resolve_settings_reads_tuned_ann(cfg, best_hyperparameters_file):
    settings = resolve_settings(cfg, "ann")
    assert settings.ann.hidden_sizes == (16, 8) and settings.training.optimizer == "adamw"
    assert "Optuna-selected" in settings.source
    assert resolve_settings(cfg, "logistic_regression").ann is None


def test_client_fit_returns_arrays_and_local_sample_count(cfg):
    settings = FederatedSettings("ann", ANNConfig((8,), True, 0.0), cfg.training, "test")
    client = IntrusionClient("B", cfg, settings, CPU)
    start = client.get_parameters({})
    arrays, n, metrics = client.fit(start, {"server_round": 1, "local_epochs": 1})
    assert n == len(client.data.train.y) == 540
    assert [a.shape for a in arrays] == [a.shape for a in start]
    assert not np.allclose(arrays[0], start[0])  # weights were trained
    assert metrics["num_buffer_arrays"] == 3 and np.isfinite(metrics["train_loss"])


def test_round_evaluator_scores_validation_without_touching_test(cfg, tmp_path, test_set_trap):
    settings = resolve_settings(cfg, "logistic_regression")
    clients = load_all_clients(cfg.data.dataset_dir, cfg.data.val_fraction, cfg.seed)
    names = clients["A"].feature_names
    scaler = combine_scalers([clients[c].scaler for c in CLIENT_IDS])
    X_raw, y, _ = load_raw_validation(cfg.data.dataset_dir, cfg.data.val_fraction, cfg.seed)
    model = build_model(settings, len(names))
    layout = ParameterLayout.from_model(model)
    S.check_client_shapes(settings, {c: names for c in CLIENT_IDS}, model)
    manager = CheckpointManager(tmp_path / "fed")
    evaluator = S.RoundEvaluator(model, layout, SplitArrays(apply_scaler(scaler, X_raw), y), cfg, CPU, manager, names,
                                 scaler, {"settings": settings.describe()})
    loss, metrics = evaluator(1, get_arrays(model), {})
    assert np.isfinite(loss) and 0 <= metrics["val_pr_auc"] <= 1
    assert manager.has_best() and np.allclose(load_checkpoint(manager.best_path).build_scaler().var_, scaler.var_)
    assert "test.parquet" not in test_set_trap
    assert "load_test_set" not in inspect.getsource(S) and "TEST_FILENAME" not in inspect.getsource(S)
