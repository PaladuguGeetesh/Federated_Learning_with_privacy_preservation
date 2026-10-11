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


@dataclass(frozen=True)
class Config:
    """Top-level project configuration."""

    seed: int
    device: str
    data: DataConfig
    ann: ANNConfig
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

    _check_keys("<root>", raw, {"seed", "device", "data", "model"})
    _check_keys("model", raw["model"], {"ann"})

    device = str(raw["device"]).lower()
    if device not in VALID_DEVICES:
        raise ValueError(f"device must be one of {VALID_DEVICES}, got {raw['device']!r}")

    return Config(
        seed=int(raw["seed"]),
        device=device,
        data=DataConfig.from_dict(raw["data"], base_dir=path.parent),
        ann=ANNConfig.from_dict(raw["model"]["ann"]),
        source_path=path,
    )
