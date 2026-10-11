"""IntrusionDataset, splits, scaling and DataLoaders."""

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from core import dataset as D
from tests.conftest import CPU, FEATURES


def test_dataset_shapes_and_dtypes():
    ds = D.IntrusionDataset(np.random.randn(10, 4), np.array([0, 1] * 5))
    x, y = ds[0]
    assert len(ds) == 10 and ds.num_features == 4
    assert x.shape == (4,) and x.dtype == torch.float32
    assert y.shape == (1,) and y.dtype == torch.float32  # float targets for BCEWithLogitsLoss


@pytest.mark.parametrize("features, labels, message", [
    (np.array([[np.nan, 1.0]]), np.array([0]), "NaN or infinite"),
    (np.array([[np.inf, 1.0]]), np.array([0]), "NaN or infinite"),
    (np.zeros((2, 2)), np.array([0, 2]), "binary"),
    (np.zeros((2, 2)), np.array([0]), "length mismatch"),
    (np.zeros(3), np.zeros(3), "2-D"),
])
def test_dataset_rejects_bad_input(features, labels, message):
    with pytest.raises(ValueError, match=message):
        D.IntrusionDataset(features, labels)


def test_dataloaders_batching_shuffle_and_drop_last():
    ds = D.IntrusionDataset(np.random.randn(1025, 3), np.random.randint(0, 2, 1025))
    loaders = D.make_dataloaders({"train": ds, "val": ds}, batch_size=512, eval_batch_size=400, num_workers=0,
                                 pin_memory=False, device=CPU, seed=0, drop_last_train=True)
    assert [b[0].shape[0] for b in loaders["train"]] == [512, 512]  # 1-row last batch dropped
    assert [b[0].shape[0] for b in loaders["val"]] == [400, 400, 225]  # eval never drops rows
    assert type(loaders["train"].sampler).__name__ == "RandomSampler"
    assert type(loaders["val"].sampler).__name__ == "SequentialSampler"
    with pytest.raises(ValueError, match="drop_last"):
        D.make_dataloader(ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=False, device=CPU,
                          seed=0, drop_last=True)


def test_client_split_is_stratified_and_scaler_fit_on_train_only(dataset_dir):
    c = D.load_client_data("A", dataset_dir, val_fraction=0.1, seed=42)
    assert len(c.train.y) == 540 and len(c.val.y) == 60
    assert abs(c.train.attack_fraction() - c.val.attack_fraction()) < 0.02
    assert c.scaler.n_samples_seen_ == len(c.train.y)
    assert np.allclose(c.train.X.mean(axis=0), 0, atol=1e-5)  # train is centred by its own scaler
    assert c.feature_names == FEATURES


def test_centralized_and_federated_use_the_same_validation_rows(dataset_dir):
    cen = D.load_centralized_data(dataset_dir, 0.1, 42)
    X_raw, y, _ = D.load_raw_validation(dataset_dir, 0.1, 42)
    assert np.array_equal(cen.val.y, y)
    assert np.allclose(cen.scaler.inverse_transform(cen.val.X.astype(np.float64)), X_raw, atol=1e-4)


def test_combine_scalers_equals_pooled_scaler(dataset_dir):
    clients = D.load_all_clients(dataset_dir, 0.1, 42)
    combined = D.combine_scalers([clients[c].scaler for c in D.CLIENT_IDS])
    raw = np.concatenate([clients[c].scaler.inverse_transform(clients[c].train.X.astype(np.float64))
                          for c in D.CLIENT_IDS])
    pooled = StandardScaler().fit(raw)
    assert np.allclose(combined.mean_, pooled.mean_, atol=1e-5)
    assert np.allclose(combined.var_, pooled.var_, rtol=1e-4)
    assert combined.n_samples_seen_ == len(raw)
