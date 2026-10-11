"""
Dataset, train/validation/test splitting, feature scaling and DataLoaders.

Split layout (shared by the centralized and federated pipelines):

    dataset/client_{A,B,C}.parquet   (on-disk files are never modified)
        └─ per client, in memory: stratified split into
             client train (1 - val_fraction)  and  client validation (val_fraction)

    federated:    each client trains on its own train part; the server's
                  validation set is the 3 client validation slices combined.
    centralized:  train = the 3 client train parts pooled,
                  validation = the 3 client validation slices pooled.

    dataset/test.parquet is the held-out test set. It is never split, and is
    only loaded through load_test_set(), which no training or tuning code calls.

Because the splits use a fixed seed, centralized and federated runs see exactly
the same training rows and the same validation rows.

Scaling: a StandardScaler is fit on training data only — per client for the
federated path, on the pooled training data for the centralized path — and that
same fitted scaler is applied to the matching validation (and test) data.
"""

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

LABEL_COL = "label"
LABEL_MAPPING = {0: "Benign", 1: "Attack"}
CLIENT_IDS = ("A", "B", "C")

TEST_FILENAME = "test.parquet"

# Split fraction, seed, dataset location and loader settings are not defaulted
# here: callers pass them from config.yaml (see core/config.py).


def client_path(client_id: str, dataset_dir: Path) -> Path:
    """Return the parquet path for a client partition (A, B or C)."""
    if client_id not in CLIENT_IDS:
        raise ValueError(f"Unknown client_id {client_id!r}; expected one of {CLIENT_IDS}")
    return dataset_dir / f"client_{client_id}.parquet"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class IntrusionDataset(Dataset):
    """In-memory tabular dataset of network-flow features and binary labels.

    Features are stored as float32 of shape [N, num_features]. Labels are stored
    as float32 of shape [N, 1] (not long), matching the [batch, 1] logit output
    of the models and the float targets BCEWithLogitsLoss expects.

    WARNING — always build DataLoaders for this class with make_dataloader() /
    make_dataloaders(). This class defines __getitems__ (a batched fast path)
    which returns already-stacked (features, labels) tensors, so it only works
    with the pass-through collate_prebatched collate function those helpers set.
    A plain DataLoader(ds, batch_size=...) uses the default collate, which will
    try to torch.stack the features and labels tensors together and fail with a
    shape error (or, for a single-feature dataset, silently produce wrong batches).
    Indexing with ds[i] (__getitem__) is unaffected.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        Args:
            features: Array of shape [N, num_features] (already scaled).
            labels: Array of shape [N] or [N, 1] containing only 0/1.
        """
        features = np.asarray(features)
        labels = np.asarray(labels).reshape(-1)

        if features.ndim != 2:
            raise ValueError(f"features must be 2-D, got shape {features.shape}")
        if len(features) != len(labels):
            raise ValueError(
                f"features and labels length mismatch: {len(features)} vs {len(labels)}"
            )
        if len(features) == 0:
            raise ValueError("Cannot build a dataset with zero samples")
        if not np.isfinite(features).all():
            raise ValueError("features contain NaN or infinite values")
        if not np.isin(labels, (0, 1)).all():
            raise ValueError("labels must be binary (0 = Benign, 1 = Attack)")

        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (features [num_features], label [1]) for one sample."""
        return self.features[idx], self.labels[idx]

    def __getitems__(self, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched fetch used by DataLoader: one tensor-index op instead of one per row.

        Must be paired with collate_prebatched (make_dataloader does this).
        """
        idx = torch.as_tensor(indices, dtype=torch.long)
        return self.features[idx], self.labels[idx]

    @property
    def num_features(self) -> int:
        return self.features.shape[1]


def collate_prebatched(batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate function for IntrusionDataset.__getitems__, which already returns stacked tensors."""
    return batch


# ---------------------------------------------------------------------------
# Loading, splitting, scaling
# ---------------------------------------------------------------------------


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a split parquet file and check it has a binary label column."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_parquet(path)
    if LABEL_COL not in df.columns:
        raise ValueError(f"{path} has no {LABEL_COL!r} column")
    return df


def split_features_labels(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (features as float64, labels as int64, feature column names)."""
    feature_names = [c for c in df.columns if c != LABEL_COL]
    X = df[feature_names].to_numpy(dtype=np.float64)
    y = df[LABEL_COL].to_numpy(dtype=np.int64)
    return X, y, feature_names


def stratified_train_val_split(
    df: pd.DataFrame, val_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified (on label) split of one DataFrame into train and validation parts."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    train_df, val_df = train_test_split(
        df, test_size=val_fraction, stratify=df[LABEL_COL], random_state=seed
    )
    return train_df, val_df


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit a StandardScaler on training features only."""
    return StandardScaler().fit(X_train)


def apply_scaler(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    """Scale features with an already-fitted scaler and cast to float32."""
    return scaler.transform(X).astype(np.float32)


@dataclass
class SplitArrays:
    """Scaled features + labels for one split."""

    X: np.ndarray  # float32, [N, num_features]
    y: np.ndarray  # int64, [N]

    def to_dataset(self) -> IntrusionDataset:
        return IntrusionDataset(self.X, self.y)

    def attack_fraction(self) -> float:
        return float(self.y.mean())


@dataclass
class ClientData:
    """One federated client's local train/validation data, scaled with its own scaler."""

    client_id: str
    train: SplitArrays
    val: SplitArrays
    scaler: StandardScaler  # fit on this client's train part only
    feature_names: list[str]


@dataclass
class CentralizedData:
    """Pooled train/validation data for centralized training and tuning (no test set)."""

    train: SplitArrays
    val: SplitArrays
    scaler: StandardScaler  # fit on pooled train only
    feature_names: list[str]
    val_client_ids: np.ndarray = field(repr=False)  # which client each val row came from


def _load_raw_client_split(
    client_id: str, dataset_dir: Path, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load one client file and split it in memory into unscaled train/val arrays."""
    df = load_parquet(client_path(client_id, dataset_dir))
    train_df, val_df = stratified_train_val_split(df, val_fraction, seed)
    X_tr, y_tr, names = split_features_labels(train_df)
    X_va, y_va, _ = split_features_labels(val_df)
    return X_tr, y_tr, X_va, y_va, names


def load_client_data(
    client_id: str,
    dataset_dir: Path,
    val_fraction: float,
    seed: int,
) -> ClientData:
    """Load one client's partition, split off its validation slice, and scale both parts.

    The scaler is fit on this client's train part only and reused for its
    validation slice. The client parquet file on disk is not modified.
    """
    X_tr, y_tr, X_va, y_va, names = _load_raw_client_split(client_id, dataset_dir, val_fraction, seed)
    scaler = fit_scaler(X_tr)
    return ClientData(
        client_id=client_id,
        train=SplitArrays(apply_scaler(scaler, X_tr), y_tr),
        val=SplitArrays(apply_scaler(scaler, X_va), y_va),
        scaler=scaler,
        feature_names=names,
    )


def load_all_clients(
    dataset_dir: Path,
    val_fraction: float,
    seed: int,
) -> dict[str, ClientData]:
    """Load ClientData for every client and check they share the same feature schema."""
    clients = {cid: load_client_data(cid, dataset_dir, val_fraction, seed) for cid in CLIENT_IDS}
    reference = clients[CLIENT_IDS[0]].feature_names
    for cid, data in clients.items():
        if data.feature_names != reference:
            raise ValueError(f"Client {cid} feature columns differ from client {CLIENT_IDS[0]}")
    return clients


def combined_federated_validation(clients: dict[str, ClientData]) -> SplitArrays:
    """Server-side validation set: every client's validation slice, each scaled by its own client's scaler."""
    return SplitArrays(
        X=np.concatenate([clients[cid].val.X for cid in CLIENT_IDS]),
        y=np.concatenate([clients[cid].val.y for cid in CLIENT_IDS]),
    )


def load_centralized_data(
    dataset_dir: Path,
    val_fraction: float,
    seed: int,
) -> CentralizedData:
    """Pooled train/val built from the same per-client splits the federated path uses.

    One scaler is fit on the pooled training rows and applied to the pooled
    validation rows. The test set is deliberately not loaded here.
    """
    X_tr_parts, y_tr_parts, X_va_parts, y_va_parts, val_ids = [], [], [], [], []
    names: list[str] | None = None
    for cid in CLIENT_IDS:
        X_tr, y_tr, X_va, y_va, client_names = _load_raw_client_split(
            cid, dataset_dir, val_fraction, seed
        )
        if names is not None and client_names != names:
            raise ValueError(f"Client {cid} feature columns differ from earlier clients")
        names = client_names
        X_tr_parts.append(X_tr)
        y_tr_parts.append(y_tr)
        X_va_parts.append(X_va)
        y_va_parts.append(y_va)
        val_ids.append(np.full(len(y_va), cid))

    X_train = np.concatenate(X_tr_parts)
    scaler = fit_scaler(X_train)
    return CentralizedData(
        train=SplitArrays(apply_scaler(scaler, X_train), np.concatenate(y_tr_parts)),
        val=SplitArrays(apply_scaler(scaler, np.concatenate(X_va_parts)), np.concatenate(y_va_parts)),
        scaler=scaler,
        feature_names=names,
        val_client_ids=np.concatenate(val_ids),
    )


def load_test_set(
    scaler: StandardScaler, expected_features: list[str], dataset_dir: Path
) -> SplitArrays:
    """Load and scale the held-out test set with an already-fitted training scaler.

    Only the final, post-tuning evaluation should call this. Hyperparameter search
    and training code must never import or call it.
    """
    X, y, names = split_features_labels(load_parquet(dataset_dir / TEST_FILENAME))
    if names != expected_features:
        raise ValueError("Test set feature columns do not match the training feature schema")
    return SplitArrays(apply_scaler(scaler, X), y)


# ---------------------------------------------------------------------------
# Device + DataLoaders
# ---------------------------------------------------------------------------


def get_device(preference: str) -> torch.device:
    """Select a torch device.

    Args:
        preference: "auto" (cuda > mps > cpu), or one of "cuda", "mps", "cpu".
    """
    preference = preference.lower()
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but CUDA is not available")
    if preference == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device 'mps' requested but MPS is not available")
    if preference not in ("cuda", "mps", "cpu"):
        raise ValueError(f"Unknown device preference {preference!r}")
    return torch.device(preference)


def make_dataloader(
    dataset: IntrusionDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    device: torch.device,
    seed: int,
) -> DataLoader:
    """Build a DataLoader for an IntrusionDataset.

    pin_memory only helps host->CUDA copies, so it is turned off (with a warning)
    for any other device. Shuffling uses a seeded generator for reproducibility.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}")
    if pin_memory and device.type != "cuda":
        warnings.warn("pin_memory is only used with a CUDA device; disabling it", stacklevel=2)
        pin_memory = False

    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=generator,
        collate_fn=collate_prebatched,
    )


def make_dataloaders(
    datasets: dict[str, IntrusionDataset],
    *,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    pin_memory: bool,
    device: torch.device,
    seed: int,
) -> dict[str, DataLoader]:
    """Build one DataLoader per split: shuffle=True only for "train", False for every other split.

    "train" uses batch_size; every other split uses eval_batch_size.
    """
    return {
        name: make_dataloader(
            ds,
            batch_size=batch_size if name == "train" else eval_batch_size,
            shuffle=name == "train",
            num_workers=num_workers,
            pin_memory=pin_memory,
            device=device,
            seed=seed,
        )
        for name, ds in datasets.items()
    }


def to_device(
    batch: tuple[torch.Tensor, torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move a (features, labels) batch to the target device."""
    non_blocking = device.type == "cuda"
    features, labels = batch
    return features.to(device, non_blocking=non_blocking), labels.to(device, non_blocking=non_blocking)
