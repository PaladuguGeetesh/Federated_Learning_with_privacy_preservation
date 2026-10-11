"""Metrics, threshold selection and test-set isolation (core/evaluate.py)."""

import inspect

import numpy as np
import pytest
import torch
from sklearn import metrics as skm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import core.evaluate as E
from core.ann import ANN
from core.config import ANNConfig, EvaluationConfig
from core.dataset import load_client_data
from core.logistic_regression import LogisticRegression

# Known-answer example (hand-computed): threshold 0.5 -> TP 3, FN 1, FP 2, TN 4.
Y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
P = np.array([0.9, 0.8, 0.4, 0.6, 0.7, 0.2, 0.1, 0.3, 0.05, 0.55])
EXPECTED = {"tp": 3, "fn": 1, "fp": 2, "tn": 4, "accuracy": 0.7, "precision": 0.6, "recall": 0.75, "f1": 2 / 3,
            "macro_f1": (2 / 3 + 8 / 11) / 2, "fpr": 2 / 6, "fnr": 0.25, "specificity": 4 / 6,
            "roc_auc": 21 / 24,                           # 21 of 24 (attack, benign) pairs ranked correctly
            "pr_auc": (1 + 1 + 3 / 4 + 4 / 6) / 4}         # average precision at each attack's rank


def test_known_answer_metrics():
    m = E.compute_metrics(Y, P, 0.5)
    for key, value in EXPECTED.items():
        assert getattr(m, key) == pytest.approx(value), key
    assert m.confusion_matrix == [[4, 2], [1, 3]]
    assert m.attack_recall == m.recall and m.false_negatives == m.fn
    assert m.tp + m.fn == Y.sum()


@pytest.mark.parametrize("threshold", [0.0, 0.25, 0.5, 0.73, 1.0])
def test_metrics_match_sklearn(threshold):
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.15).astype(int)
    p = np.round(np.clip(rng.normal(0.3 + 0.4 * y, 0.2), 0, 1), 2)
    m, pred = E.compute_metrics(y, p, threshold), (p >= threshold).astype(int)
    assert m.accuracy == pytest.approx(skm.accuracy_score(y, pred))
    assert m.precision == pytest.approx(skm.precision_score(y, pred, zero_division=0))
    assert m.recall == pytest.approx(skm.recall_score(y, pred))
    assert m.f1 == pytest.approx(skm.f1_score(y, pred, zero_division=0))
    assert m.macro_f1 == pytest.approx(skm.f1_score(y, pred, average="macro", zero_division=0))
    assert m.confusion_matrix == skm.confusion_matrix(y, pred, labels=[0, 1]).tolist()
    assert m.roc_auc == pytest.approx(skm.roc_auc_score(y, p))
    assert m.pr_auc == pytest.approx(skm.average_precision_score(y, p))


def test_metrics_dict_round_trip():
    m = E.compute_metrics(Y, P, 0.5)
    assert E.BinaryMetrics.from_dict(m.to_dict()) == m


@pytest.mark.parametrize("strategy, score", [("max_f1", "f1"), ("max_macro_f1", "macro_f1")])
def test_select_threshold_matches_brute_force(strategy, score):
    rng = np.random.default_rng(1)
    y = (rng.random(3000) < 0.15).astype(int)
    p = np.round(np.clip(rng.normal(0.3 + 0.4 * y, 0.2), 0, 1), 2)
    selection = E.select_threshold(y, p, EvaluationConfig(0.5, strategy, 0.99))
    brute = max(getattr(E.compute_metrics(y, p, t), score) for t in np.unique(p))
    assert getattr(selection.validation_metrics, score) == pytest.approx(brute)


def test_select_threshold_target_recall_and_fixed():
    sel = E.select_threshold(Y, P, EvaluationConfig(0.5, "target_recall", 0.99))
    assert sel.validation_metrics.recall >= 0.99
    assert E.select_threshold(Y, P, EvaluationConfig(0.42, "fixed", 0.99)).threshold == 0.42


def test_select_threshold_never_touches_the_test_set(dataset_dir, test_set_trap):
    """select_threshold runs on validation predictions with every test-set path booby-trapped."""
    assert list(inspect.signature(E.select_threshold).parameters) == ["y_val", "p_val", "config"]
    source = inspect.getsource(E.select_threshold) + inspect.getsource(E._counts_per_candidate_threshold)
    assert not any(s in source for s in ("load_test_set", "test.parquet", "read_parquet", "TEST_FILENAME"))

    client = load_client_data("A", dataset_dir, 0.1, 42)  # reads client_A only (allowed)
    model = LogisticRegression(client.train.X.shape[1])
    p_val = E.predict_proba(model, client.val.X, batch_size=64)
    selection = E.select_threshold(client.val.y, p_val, EvaluationConfig(0.5, "max_f1", 0.99))
    assert 0.0 <= selection.threshold <= 1.0
    assert "test.parquet" not in test_set_trap


def test_evaluate_on_test_requires_chosen_threshold_and_fitted_scaler(dataset_dir):
    signature = inspect.signature(E.evaluate_on_test)
    assert signature.parameters["threshold"].default is inspect.Parameter.empty
    assert signature.parameters["scaler"].default is inspect.Parameter.empty
    model = LogisticRegression(6)
    with pytest.raises(ValueError, match="already-fitted scaler"):
        E.evaluate_on_test(model, scaler=StandardScaler(), threshold=0.5, feature_names=[], dataset_dir=dataset_dir,
                           batch_size=64)
    with pytest.raises(ValueError, match="threshold"):
        E.evaluate_on_test(model, scaler=None, threshold=1.5, feature_names=[], dataset_dir=dataset_dir, batch_size=64)


def test_predict_proba_works_for_torch_and_sklearn_models():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(100, 6)).astype(np.float32), np.array([0, 1] * 50)
    for model in (ANN(6, ANNConfig((4,), False, 0.0)), LogisticRegression(6),
                  RandomForestClassifier(n_estimators=3, random_state=0).fit(X, y)):
        p = E.predict_proba(model, X, batch_size=32, device=torch.device("cpu"))
        assert p.shape == (100,) and p.dtype == np.float64 and ((p >= 0) & (p <= 1)).all()


def test_final_test_cli_runs_once(cfg, tmp_path):
    """python -m core.evaluate --final-test on a synthetic checkpoint; a second run refuses."""
    from core.checkpoint import make_checkpoint, save_checkpoint
    from core.dataset import LABEL_MAPPING, load_centralized_data

    data = load_centralized_data(cfg.data.dataset_dir, cfg.data.val_fraction, cfg.seed)
    ckpt = make_checkpoint(epoch=1, model=LogisticRegression(6), training_config={}, history=[],
                           feature_names=data.feature_names, label_mapping=LABEL_MAPPING, seed=0, scaler=data.scaler)
    path = tmp_path / "lr.pt"
    save_checkpoint(ckpt, path)
    argv = ["--config", str(cfg.source_path), "--checkpoint", str(path), "--name", "lr_test"]
    with pytest.raises(SystemExit, match="--final-test"):
        E.main(argv)
    E.main(argv + ["--final-test"])
    assert (cfg.paths.results_dir / "lr_test_test_metrics.json").exists()
    with pytest.raises(SystemExit, match="already exists"):
        E.main(argv + ["--final-test"])
