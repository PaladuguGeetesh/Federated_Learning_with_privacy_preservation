"""Centralized RF script, client partitioning, and the result-file loaders."""

import json

import numpy as np
import pandas as pd
import pytest

from centralized import train_rf
from core.evaluate import BinaryMetrics
from federated import partition_data
from results.sources import _rf_metrics
from tests.conftest import synthetic_frame


def test_train_rf_creates_split_once_and_evaluates_once(make_config, tmp_path):
    data_dir = tmp_path / "rf_dataset"
    data_dir.mkdir()
    synthetic_frame(500, seed=5).to_parquet(data_dir / "cleaned_data.parquet", index=False)
    cfg = make_config(dataset=data_dir)

    path = train_rf.run(cfg)
    assert (data_dir / "train.parquet").exists() and (data_dir / "test.parquet").exists()
    result = json.loads(path.read_text())
    assert BinaryMetrics.from_dict(result["test_metrics"]).n_samples == 100  # test_size 0.2 of 500
    assert cfg.random_forest.model_path.exists()
    with pytest.raises(SystemExit, match="already exists"):
        train_rf.run(cfg)

    (data_dir / "train.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="Refusing to re-split"):
        train_rf.load_or_create_split(data_dir, 0.2, cfg.seed)


def test_partition_data_is_stratified_and_refuses_overwrite(make_config, tmp_path):
    data_dir = tmp_path / "part_dataset"
    data_dir.mkdir()
    synthetic_frame(900, seed=3).to_parquet(data_dir / "train.parquet", index=False)
    cfg = make_config(dataset=data_dir)
    partition_data.main(["--config", str(cfg.source_path)])
    sizes = [len(pd.read_parquet(data_dir / f"client_{c}.parquet")) for c in "ABC"]
    assert sum(sizes) == 900 and max(sizes) - min(sizes) <= 1
    with pytest.raises(SystemExit, match="already exist"):
        partition_data.main(["--config", str(cfg.source_path)])


def test_legacy_rf_metrics_recovery_is_exact():
    """The original RF result stored only five metrics; confusion counts are recovered exactly."""
    truth = BinaryMetrics.from_counts(tp=66969, fp=189, tn=378874, fn=330, threshold=0.5, roc_auc=0.9999, pr_auc=np.nan)
    legacy = {"metrics": {"accuracy": truth.accuracy, "precision": truth.precision, "recall": truth.recall,
                          "f1_score": truth.f1, "roc_auc": truth.roc_auc}}
    recovered = _rf_metrics(legacy, truth.n_attack, truth.n_benign)
    assert (recovered.tp, recovered.fp, recovered.tn, recovered.fn) == (66969, 189, 378874, 330)
