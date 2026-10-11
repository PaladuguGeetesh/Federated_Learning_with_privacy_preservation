"""
Load and validate config.yaml into typed, immutable config objects.

Unknown keys are rejected so a typo in config.yaml fails loudly instead of
silently falling back to some other value.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

VALID_DEVICES = ("auto", "cuda", "mps", "cpu")


def _check_keys(section: str, given: dict[str, Any], expected: set[str]) -> None:
    """Raise if a config section has missing or unknown keys."""
    if not isinstance(given, dict):
        raise ValueError(f"config section {section!r} must be a mapping, got {type(given).__name__}")
    missing = expected - given.keys()
    unknown = given.keys() - expected
    if missing:
        raise ValueError(f"config section {section!r} is missing keys: {sorted(missing)}")
    if unknown:
        raise ValueError(f"config section {section!r} has unknown keys: {sorted(unknown)}")


@dataclass(frozen=True)
class DataConfig:
    """Data split and DataLoader settings."""

    dataset_dir: Path
    val_fraction: float
    eval_batch_size: int
    num_workers: int
    pin_memory: bool

    def __post_init__(self) -> None:
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError(f"data.val_fraction must be in (0, 1), got {self.val_fraction}")
        if self.eval_batch_size < 1:
            raise ValueError(f"data.eval_batch_size must be >= 1, got {self.eval_batch_size}")
        if self.num_workers < 0:
            raise ValueError(f"data.num_workers must be >= 0, got {self.num_workers}")

    @classmethod
    def from_dict(cls, d: dict[str, Any], base_dir: Path) -> "DataConfig":
        _check_keys("data", d, {f for f in cls.__dataclass_fields__})
        dataset_dir = Path(d["dataset_dir"])
        if not dataset_dir.is_absolute():
            dataset_dir = (base_dir / dataset_dir).resolve()
        return cls(
            dataset_dir=dataset_dir,
            val_fraction=float(d["val_fraction"]),
            eval_batch_size=int(d["eval_batch_size"]),
            num_workers=int(d["num_workers"]),
            pin_memory=bool(d["pin_memory"]),
        )


@dataclass(frozen=True)
class ANNConfig:
    """Architecture settings for core.ann.ANN (input_dim is inferred from data, not configured)."""

    hidden_sizes: tuple[int, ...]
    batch_norm: bool
    dropout: float

    def __post_init__(self) -> None:
        if len(self.hidden_sizes) == 0:
            raise ValueError("model.ann.hidden_sizes must contain at least one layer")
        if any(int(h) != h or h < 1 for h in self.hidden_sizes):
            raise ValueError(f"model.ann.hidden_sizes must be positive integers, got {self.hidden_sizes}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"model.ann.dropout must be in [0, 1), got {self.dropout}")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ANNConfig":
        _check_keys("model.ann", d, {f for f in cls.__dataclass_fields__})
        if not isinstance(d["hidden_sizes"], (list, tuple)):
            raise ValueError("model.ann.hidden_sizes must be a list of integers")
        return cls(
            hidden_sizes=tuple(d["hidden_sizes"]),
            batch_norm=bool(d["batch_norm"]),
            dropout=float(d["dropout"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form (hidden_sizes as a list) for checkpoints and logs."""
        d = asdict(self)
        d["hidden_sizes"] = list(self.hidden_sizes)
        return d


VALID_OPTIMIZERS = ("sgd", "adam", "adamw")
VALID_MODES = ("min", "max")


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Early stopping on a validation metric.

    mode is explicit so both lower-is-better (val_loss) and higher-is-better
    (val_pr_auc, val_f1) metrics are supported.
    """

    metric: str
    mode: str
    patience: int
    min_delta: float

    def __post_init__(self) -> None:
        if not self.metric:
            raise ValueError("training.early_stopping.metric must be a non-empty string")
        if self.mode not in VALID_MODES:
            raise ValueError(f"training.early_stopping.mode must be one of {VALID_MODES}, got {self.mode!r}")
        if self.patience < 1:
            raise ValueError(f"training.early_stopping.patience must be >= 1, got {self.patience}")
        if self.min_delta < 0:
            raise ValueError(f"training.early_stopping.min_delta must be >= 0, got {self.min_delta}")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EarlyStoppingConfig":
        _check_keys("training.early_stopping", d, {f for f in cls.__dataclass_fields__})
        return cls(
            metric=str(d["metric"]),
            mode=str(d["mode"]).lower(),
            patience=int(d["patience"]),
            min_delta=float(d["min_delta"]),
        )


@dataclass(frozen=True)
class TrainingConfig:
    """Optimizer and training-loop settings."""

    optimizer: str
    learning_rate: float
    weight_decay: float
    momentum: float  # used only by SGD
    batch_size: int
    epochs: int
    grad_clip_norm: float | None  # None disables gradient clipping
    early_stopping: EarlyStoppingConfig
    checkpoint_dir: Path

    def __post_init__(self) -> None:
        if self.optimizer not in VALID_OPTIMIZERS:
            raise ValueError(f"training.optimizer must be one of {VALID_OPTIMIZERS}, got {self.optimizer!r}")
        if self.learning_rate <= 0:
            raise ValueError(f"training.learning_rate must be > 0, got {self.learning_rate}")
        if self.weight_decay < 0:
            raise ValueError(f"training.weight_decay must be >= 0, got {self.weight_decay}")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError(f"training.momentum must be in [0, 1), got {self.momentum}")
        if self.batch_size < 1:
            raise ValueError(f"training.batch_size must be >= 1, got {self.batch_size}")
        if self.epochs < 1:
            raise ValueError(f"training.epochs must be >= 1, got {self.epochs}")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError(f"training.grad_clip_norm must be null or > 0, got {self.grad_clip_norm}")

    @classmethod
    def from_dict(cls, d: dict[str, Any], base_dir: Path) -> "TrainingConfig":
        _check_keys("training", d, {f for f in cls.__dataclass_fields__})
        checkpoint_dir = Path(d["checkpoint_dir"])
        if not checkpoint_dir.is_absolute():
            checkpoint_dir = (base_dir / checkpoint_dir).resolve()
        clip = d["grad_clip_norm"]
        return cls(
            optimizer=str(d["optimizer"]).lower(),
            # float() also accepts strings like "1e-3", which PyYAML does not parse as numbers
            learning_rate=float(d["learning_rate"]),
            weight_decay=float(d["weight_decay"]),
            momentum=float(d["momentum"]),
            batch_size=int(d["batch_size"]),
            epochs=int(d["epochs"]),
            grad_clip_norm=None if clip is None else float(clip),
            early_stopping=EarlyStoppingConfig.from_dict(d["early_stopping"]),
            checkpoint_dir=checkpoint_dir,
        )

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form (paths as strings) for checkpoints and logs."""
        d = asdict(self)
        d["checkpoint_dir"] = str(self.checkpoint_dir)
        return d


@dataclass(frozen=True)
class Config:
    """Top-level project configuration."""

    seed: int
    device: str
    data: DataConfig
    ann: ANNConfig
    training: TrainingConfig
    source_path: Path


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Config:
    """Read config.yaml and return a validated Config.

    Relative paths inside the file are resolved against the file's directory.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f)

    _check_keys("<root>", raw, {"seed", "device", "data", "model", "training"})
    _check_keys("model", raw["model"], {"ann"})

    device = str(raw["device"]).lower()
    if device not in VALID_DEVICES:
        raise ValueError(f"device must be one of {VALID_DEVICES}, got {raw['device']!r}")

    return Config(
        seed=int(raw["seed"]),
        device=device,
        data=DataConfig.from_dict(raw["data"], base_dir=path.parent),
        ann=ANNConfig.from_dict(raw["model"]["ann"]),
        training=TrainingConfig.from_dict(raw["training"], base_dir=path.parent),
        source_path=path,
    )
