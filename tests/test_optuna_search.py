"""Optuna objective / study plumbing on tiny synthetic data (not a real search)."""

import json

import optuna
import pytest

import core.optuna_search as OS
from tests.conftest import CPU


@pytest.fixture
def prepared(cfg):
    return OS.prepare_data(cfg)


def test_objective_runs_and_reports_a_score(cfg, prepared, test_set_trap):
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(OS.make_objective(prepared, cfg, CPU), n_trials=2)
    for trial in study.trials:
        assert trial.state == optuna.trial.TrialState.COMPLETE
        assert 0.0 <= trial.value <= 1.0
        assert trial.intermediate_values  # val_pr_auc was reported every epoch (for pruning)
        assert trial.user_attrs["best_epoch"] >= 1
    assert "test.parquet" not in test_set_trap


def test_suggested_configs_respect_conditional_params(cfg):
    fixed = optuna.trial.FixedTrial({"n_layers": 1, "layer_0_size": 8, "dropout": 0.1, "batch_norm": False,
                                     "optimizer": "adam", "grad_clip": False, "class_weighting": "none",
                                     "learning_rate": 0.001, "weight_decay": 1e-5, "batch_size": 64, "epochs": 1})
    ann_cfg, train_cfg = OS.suggest_configs(fixed, cfg)
    assert ann_cfg.hidden_sizes == (8,)
    assert train_cfg.grad_clip_norm is None and train_cfg.momentum == cfg.training.momentum
    assert train_cfg.early_stopping.metric == "val_pr_auc" and train_cfg.early_stopping.mode == "max"


def test_study_persists_resumes_and_final_step_runs_once(cfg, prepared):
    storage = cfg.optuna.storage_path
    study = OS.run_study(prepared, cfg, CPU, n_trials=2, storage_path=storage)
    assert len(OS.finished_trials(study)) == 2
    resumed = OS.run_study(prepared, cfg, CPU, n_trials=3, storage_path=storage)  # adds only 1 more
    assert [t.number for t in resumed.trials] == [0, 1, 2]

    results_dir = cfg.paths.results_dir
    OS.save_best_hyperparameters(resumed, cfg, prepared.data.train.y, storage, results_dir)
    saved = json.loads((results_dir / OS.BEST_PARAMS_FILENAME).read_text())
    assert saved["test_set_used"] is False and saved["best_trial"] == resumed.best_trial.number

    OS.final_retrain_and_test(resumed, prepared, cfg, CPU, results_dir)
    final = json.loads((results_dir / OS.FINAL_METRICS_FILENAME).read_text())
    assert final["threshold_selection"]["selected_on"] == "validation"
    assert final["test_metrics"]["threshold"] == final["threshold_selection"]["threshold"]
    with pytest.raises(SystemExit, match="already"):
        OS.final_retrain_and_test(resumed, prepared, cfg, CPU, results_dir)
