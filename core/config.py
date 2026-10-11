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
VALID_CLASS_WEIGHTING = ("none", "balanced", "balanced_scaled")


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
    class_weighting: str  # none | balanced | balanced_scaled (see core.train.compute_pos_weight)
    pos_weight_scale: float  # used only by balanced_scaled
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
        if self.class_weighting not in VALID_CLASS_WEIGHTING:
            raise ValueError(
                f"training.class_weighting must be one of {VALID_CLASS_WEIGHTING}, got {self.class_weighting!r}"
            )
        if self.pos_weight_scale <= 0:
            raise ValueError(f"training.pos_weight_scale must be > 0, got {self.pos_weight_scale}")

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
            class_weighting=str(d["class_weighting"]).lower(),
            pos_weight_scale=float(d["pos_weight_scale"]),
            early_stopping=EarlyStoppingConfig.from_dict(d["early_stopping"]),
            checkpoint_dir=checkpoint_dir,
        )

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form (paths as strings) for checkpoints and logs."""
        d = asdict(self)
        d["checkpoint_dir"] = str(self.checkpoint_dir)
        return d


VALID_THRESHOLD_STRATEGIES = ("max_f1", "max_macro_f1", "target_recall", "fixed")


@dataclass(frozen=True)
class EvaluationConfig:
    """Decision-threshold settings (see core/evaluate.py for how each strategy works)."""

    default_threshold: float
    threshold_strategy: str
    target_recall: float  # used only by the target_recall strategy

    def __post_init__(self) -> None:
        if not 0.0 < self.default_threshold < 1.0:
            raise ValueError(f"evaluation.default_threshold must be in (0, 1), got {self.default_threshold}")
        if self.threshold_strategy not in VALID_THRESHOLD_STRATEGIES:
            raise ValueError(
                f"evaluation.threshold_strategy must be one of {VALID_THRESHOLD_STRATEGIES}, "
                f"got {self.threshold_strategy!r}"
            )
        if not 0.0 < self.target_recall <= 1.0:
            raise ValueError(f"evaluation.target_recall must be in (0, 1], got {self.target_recall}")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvaluationConfig":
        _check_keys("evaluation", d, {f for f in cls.__dataclass_fields__})
        return cls(
            default_threshold=float(d["default_threshold"]),
            threshold_strategy=str(d["threshold_strategy"]).lower(),
            target_recall=float(d["target_recall"]),
        )


def _range(section: str, value: Any, cast: type, positive: bool = False) -> tuple[Any, Any]:
    """Parse a [low, high] list."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{section} must be a [low, high] list, got {value!r}")
    low, high = cast(value[0]), cast(value[1])
    if low > high:
        raise ValueError(f"{section}: low {low} > high {high}")
    if positive and low <= 0:
        raise ValueError(f"{section}: values must be > 0 (log-scale range), got low {low}")
    return low, high


def _choices(section: str, value: Any, cast: type, allowed: tuple | None = None) -> tuple[Any, ...]:
    """Parse a non-empty list of choices."""
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        raise ValueError(f"{section} must be a non-empty list, got {value!r}")
    if cast is bool and not all(isinstance(v, bool) for v in value):
        raise ValueError(f"{section} must contain only true/false, got {value!r}")
    out = tuple(cast(v) for v in value)
    if allowed is not None and any(v not in allowed for v in out):
        raise ValueError(f"{section} values must be in {allowed}, got {out}")
    return out


@dataclass(frozen=True)
class SearchSpaceConfig:
    """Optuna search ranges ([low, high]) and choice lists for the centralized ANN."""

    n_layers: tuple[int, int]
    layer_size: tuple[int, int]
    optimizer: tuple[str, ...]
    epochs: tuple[int, int]
    learning_rate: tuple[float, float]
    batch_size: tuple[int, ...]
    dropout: tuple[float, float]
    weight_decay: tuple[float, float]
    batch_norm: tuple[bool, ...]
    momentum: tuple[float, float]
    grad_clip: tuple[bool, ...]
    grad_clip_norm: tuple[float, float]
    class_weighting: tuple[str, ...]
    pos_weight_scale: tuple[float, float]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SearchSpaceConfig":
        sec = "optuna.search_space"
        _check_keys(sec, d, {f for f in cls.__dataclass_fields__})
        space = cls(
            n_layers=_range(f"{sec}.n_layers", d["n_layers"], int, positive=True),
            layer_size=_range(f"{sec}.layer_size", d["layer_size"], int, positive=True),
            optimizer=_choices(f"{sec}.optimizer", d["optimizer"], str, VALID_OPTIMIZERS),
            epochs=_range(f"{sec}.epochs", d["epochs"], int, positive=True),
            learning_rate=_range(f"{sec}.learning_rate", d["learning_rate"], float, positive=True),
            batch_size=_choices(f"{sec}.batch_size", d["batch_size"], int),
            dropout=_range(f"{sec}.dropout", d["dropout"], float),
            weight_decay=_range(f"{sec}.weight_decay", d["weight_decay"], float, positive=True),
            batch_norm=_choices(f"{sec}.batch_norm", d["batch_norm"], bool),
            momentum=_range(f"{sec}.momentum", d["momentum"], float),
            grad_clip=_choices(f"{sec}.grad_clip", d["grad_clip"], bool),
            grad_clip_norm=_range(f"{sec}.grad_clip_norm", d["grad_clip_norm"], float, positive=True),
            class_weighting=_choices(f"{sec}.class_weighting", d["class_weighting"], str, VALID_CLASS_WEIGHTING),
            pos_weight_scale=_range(f"{sec}.pos_weight_scale", d["pos_weight_scale"], float, positive=True),
        )
        if any(b < 1 for b in space.batch_size):
            raise ValueError(f"{sec}.batch_size values must be >= 1")
        if not (0.0 <= space.dropout[0] and space.dropout[1] < 1.0):
            raise ValueError(f"{sec}.dropout must lie within [0, 1)")
        if not (0.0 <= space.momentum[0] and space.momentum[1] < 1.0):
            raise ValueError(f"{sec}.momentum must lie within [0, 1)")
        return space


@dataclass(frozen=True)
class OptunaConfig:
    """Study, pruning and per-trial early-stopping settings for core/optuna_search.py."""

    study_name: str
    storage_path: Path
    n_trials: int
    timeout_seconds: float | None
    sampler_seed: int
    pruner_startup_trials: int
    pruner_warmup_epochs: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    final_checkpoint_dir: Path
    search_space: SearchSpaceConfig

    def __post_init__(self) -> None:
        if not self.study_name:
            raise ValueError("optuna.study_name must be non-empty")
        if self.n_trials < 1:
            raise ValueError(f"optuna.n_trials must be >= 1, got {self.n_trials}")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError(f"optuna.timeout_seconds must be null or > 0, got {self.timeout_seconds}")
        if self.pruner_startup_trials < 0 or self.pruner_warmup_epochs < 0:
            raise ValueError("optuna.pruner values must be >= 0")
        if self.early_stopping_patience < 1:
            raise ValueError("optuna.early_stopping.patience must be >= 1")
        if self.early_stopping_min_delta < 0:
            raise ValueError("optuna.early_stopping.min_delta must be >= 0")

    @classmethod
    def from_dict(cls, d: dict[str, Any], base_dir: Path) -> "OptunaConfig":
        _check_keys("optuna", d, {"study_name", "storage", "n_trials", "timeout_seconds", "sampler_seed",
                                  "pruner", "early_stopping", "final_checkpoint_dir", "search_space"})
        _check_keys("optuna.pruner", d["pruner"], {"n_startup_trials", "n_warmup_epochs"})
        _check_keys("optuna.early_stopping", d["early_stopping"], {"patience", "min_delta"})

        def resolve(p: str) -> Path:
            path = Path(p)
            return path if path.is_absolute() else (base_dir / path).resolve()

        timeout = d["timeout_seconds"]
        return cls(
            study_name=str(d["study_name"]),
            storage_path=resolve(d["storage"]),
            n_trials=int(d["n_trials"]),
            timeout_seconds=None if timeout is None else float(timeout),
            sampler_seed=int(d["sampler_seed"]),
            pruner_startup_trials=int(d["pruner"]["n_startup_trials"]),
            pruner_warmup_epochs=int(d["pruner"]["n_warmup_epochs"]),
            early_stopping_patience=int(d["early_stopping"]["patience"]),
            early_stopping_min_delta=float(d["early_stopping"]["min_delta"]),
            final_checkpoint_dir=resolve(d["final_checkpoint_dir"]),
            search_space=SearchSpaceConfig.from_dict(d["search_space"]),
        )


@dataclass(frozen=True)
class Config:
    """Top-level project configuration."""

    seed: int
    device: str
    data: DataConfig
    ann: ANNConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    optuna: OptunaConfig
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

    _check_keys("<root>", raw, {"seed", "device", "data", "model", "training", "evaluation", "optuna"})
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
        evaluation=EvaluationConfig.from_dict(raw["evaluation"]),
        optuna=OptunaConfig.from_dict(raw["optuna"], base_dir=path.parent),
        source_path=path,
    )
