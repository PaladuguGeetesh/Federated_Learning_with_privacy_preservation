"""
Optuna hyperparameter search for the CENTRALIZED ANN.

Objective: validation PR-AUC for the attack class (maximize). PR-AUC is
threshold-free and, unlike accuracy or ROC-AUC, stays sensitive to errors on
the ~15% attack minority class.

Data used by the search:
    train       load_centralized_data().train  (pooled client train parts, pooled scaler)
    validation  load_centralized_data().val    — the same rows, in the same order, as
                combined_federated_validation(), but scaled with the centralized
                model's own (pooled) training scaler. Using the per-client-scaled
                combined_federated_validation() arrays here would feed the model
                inputs scaled differently from its training data (per-client vs
                pooled statistics differ by up to ~120 std units on a few
                heavy-tailed features).
    test        NEVER touched by the search. dataset/test.parquet is read exactly
                once, in final_retrain_and_test(), after the study is finished,
                and only when --final-test is passed.

Per trial:
    * values are sampled into the existing ANNConfig / TrainingConfig (via
      dataclasses.replace on the model.ann and training sections of config.yaml),
      so trials and normal training share one config path;
    * set_seed() is called inside the objective, before the trial's model is
      built (torch's startup seed is random, so seeding once at startup is not enough);
    * early stopping watches val_pr_auc in max mode (a per-trial override; the
      project's default training config keeps val_loss/min);
    * val_pr_auc is reported every epoch so the MedianPruner can stop weak trials.

Not searched (prerequisite not implemented): activation function (ann.py is
ReLU-only) and learning-rate scheduler (none implemented).

Usage (from the project root):
    python -m core.optuna_search                     # run / resume the study, save best hyperparameters
    python -m core.optuna_search --n-trials 60       # override optuna.n_trials
    python -m core.optuna_search --final-test        # also: retrain best config once + single test evaluation
"""

import argparse
import dataclasses
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import optuna
import torch
from optuna.trial import FrozenTrial, TrialState

from core.ann import ANN
from core.checkpoint import CheckpointManager
from core.config import (
    ANNConfig,
    Config,
    DEFAULT_CONFIG_PATH,
    EarlyStoppingConfig,
    OptunaConfig,
    TrainingConfig,
    load_config,
)
from core.dataset import (
    CentralizedData,
    IntrusionDataset,
    get_device,
    load_centralized_data,
    make_dataloaders,
)
from core.evaluate import evaluate_on_test, make_val_metrics_fn, predict_proba, select_threshold
from core.train import TrainingDivergedError, compute_pos_weight, fit, set_seed, train_drop_last

OBJECTIVE_METRIC = "val_pr_auc"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
BEST_PARAMS_FILENAME = "best_hyperparameters.json"
FINAL_METRICS_FILENAME = "ann_centralized_metrics.json"


# ---------------------------------------------------------------------------
# Search space -> existing config objects
# ---------------------------------------------------------------------------


def suggest_configs(
    trial: optuna.trial.BaseTrial, cfg: Config
) -> tuple[ANNConfig, TrainingConfig]:
    """Sample one trial's values into ANNConfig / TrainingConfig.

    Works with a live Trial (during the search) and with optuna.trial.FixedTrial
    (to rebuild the best trial's configs for the final retrain).
    """
    space = cfg.optuna.search_space

    n_layers = trial.suggest_int("n_layers", *space.n_layers)
    hidden_sizes = tuple(
        trial.suggest_int(f"layer_{i}_size", *space.layer_size, log=True) for i in range(n_layers)
    )
    ann_cfg = dataclasses.replace(
        cfg.ann,
        hidden_sizes=hidden_sizes,
        dropout=trial.suggest_float("dropout", *space.dropout),
        batch_norm=trial.suggest_categorical("batch_norm", list(space.batch_norm)),
    )

    optimizer = trial.suggest_categorical("optimizer", list(space.optimizer))
    # Conditional parameters: only sampled when they have an effect.
    momentum = trial.suggest_float("momentum", *space.momentum) if optimizer == "sgd" else cfg.training.momentum
    clip_enabled = trial.suggest_categorical("grad_clip", list(space.grad_clip))
    grad_clip_norm = trial.suggest_float("grad_clip_norm", *space.grad_clip_norm, log=True) if clip_enabled else None
    class_weighting = trial.suggest_categorical("class_weighting", list(space.class_weighting))
    pos_weight_scale = (
        trial.suggest_float("pos_weight_scale", *space.pos_weight_scale, log=True)
        if class_weighting == "balanced_scaled"
        else 1.0
    )

    train_cfg = dataclasses.replace(
        cfg.training,
        optimizer=optimizer,
        learning_rate=trial.suggest_float("learning_rate", *space.learning_rate, log=True),
        weight_decay=trial.suggest_float("weight_decay", *space.weight_decay, log=True),
        momentum=momentum,
        batch_size=trial.suggest_categorical("batch_size", list(space.batch_size)),
        epochs=trial.suggest_int("epochs", *space.epochs),
        grad_clip_norm=grad_clip_norm,
        class_weighting=class_weighting,
        pos_weight_scale=pos_weight_scale,
        # Per-trial override: early stopping on the objective metric, higher is better.
        early_stopping=EarlyStoppingConfig(
            metric=OBJECTIVE_METRIC,
            mode="max",
            patience=cfg.optuna.early_stopping_patience,
            min_delta=cfg.optuna.early_stopping_min_delta,
        ),
    )
    return ann_cfg, train_cfg


def resolved_hyperparameters(ann_cfg: ANNConfig, train_cfg: TrainingConfig, pos_weight: float | None) -> dict[str, Any]:
    """The full hyperparameter set a trial trains with (sampled + fixed values)."""
    training = train_cfg.to_dict()
    training.pop("checkpoint_dir")  # a path, not a hyperparameter
    return {"ann": ann_cfg.to_dict(), "training": training, "pos_weight": pos_weight}


# ---------------------------------------------------------------------------
# Training one configuration (shared by trials and the final retrain)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PreparedData:
    """Centralized data plus tensors built once and reused by every trial."""

    data: CentralizedData
    train_ds: IntrusionDataset
    val_ds: IntrusionDataset


def prepare_data(cfg: Config) -> PreparedData:
    """Load centralized train + validation (never the test set)."""
    data = load_centralized_data(cfg.data.dataset_dir, cfg.data.val_fraction, cfg.seed)
    return PreparedData(data=data, train_ds=data.train.to_dataset(), val_ds=data.val.to_dataset())


def train_configuration(
    prepared: PreparedData,
    cfg: Config,
    ann_cfg: ANNConfig,
    train_cfg: TrainingConfig,
    device: torch.device,
    *,
    epoch_callback=None,
    checkpoint_manager: CheckpointManager | None = None,
    checkpoint_extra: dict[str, Any] | None = None,
    verbose: bool = False,
):
    """Seed, build a fresh model, and fit it. Returns (model, FitResult, pos_weight)."""
    data = prepared.data
    pos_weight = compute_pos_weight(data.train.y, train_cfg.class_weighting, train_cfg.pos_weight_scale)
    loaders = make_dataloaders(
        {"train": prepared.train_ds, "val": prepared.val_ds},
        batch_size=train_cfg.batch_size,
        eval_batch_size=cfg.data.eval_batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        device=device,
        seed=cfg.seed,
        drop_last_train=train_drop_last(len(data.train.y), train_cfg.batch_size, ann_cfg.batch_norm),
    )
    # Seed immediately before building this configuration's model (see module docstring).
    set_seed(cfg.seed)
    model = ANN(len(data.feature_names), ann_cfg).to(device)
    result = fit(
        model,
        loaders["train"],
        loaders["val"],
        training_config=train_cfg,
        device=device,
        seed=cfg.seed,
        feature_names=data.feature_names,
        scaler=data.scaler,
        checkpoint_manager=checkpoint_manager,
        extra_val_metrics_fn=make_val_metrics_fn(cfg.evaluation.default_threshold),
        pos_weight=pos_weight,
        checkpoint_extra=checkpoint_extra,
        epoch_callback=epoch_callback,
        verbose=verbose,
    )
    return model, result, pos_weight


# ---------------------------------------------------------------------------
# Study
# ---------------------------------------------------------------------------


def make_objective(prepared: PreparedData, cfg: Config, device: torch.device):
    """Objective: best validation PR-AUC reached by one sampled configuration."""

    def objective(trial: optuna.Trial) -> float:
        ann_cfg, train_cfg = suggest_configs(trial, cfg)

        def report(record: dict[str, Any]) -> None:
            epoch = record["epoch"]
            trial.set_user_attr("epochs_run", epoch)
            trial.report(record[OBJECTIVE_METRIC], step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(f"pruned at epoch {epoch} ({OBJECTIVE_METRIC}={record[OBJECTIVE_METRIC]:.5f})")

        result = train_configuration(prepared, cfg, ann_cfg, train_cfg, device, epoch_callback=report)[1]
        best = next(h for h in result.history if h["epoch"] == result.best_epoch)
        trial.set_user_attr("best_epoch", result.best_epoch)
        trial.set_user_attr("stopped_early", result.stopped_early)
        trial.set_user_attr("val_metrics_at_best_epoch", {k: v for k, v in best.items() if k.startswith("val_")})
        return float(result.best_value)

    return objective


def storage_url(path: Path) -> str:
    return f"sqlite:///{path}"


def open_study(opt: OptunaConfig, storage_path: Path) -> optuna.Study:
    """Create or resume the persistent study.

    The heartbeat lets Optuna mark trials left RUNNING by an interrupted run as
    FAIL when the study is resumed, instead of leaving them stuck.
    """
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.RDBStorage(storage_url(storage_path), heartbeat_interval=60, grace_period=180)
    existing = 0
    if opt.study_name in [s.study_name for s in optuna.get_all_study_summaries(storage)]:
        existing = len(optuna.load_study(study_name=opt.study_name, storage=storage).trials)
    return optuna.create_study(
        study_name=opt.study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        # Offset by the number of existing trials so a resumed study does not
        # replay the same random start-up samples.
        sampler=optuna.samplers.TPESampler(seed=opt.sampler_seed + existing),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=opt.pruner_startup_trials, n_warmup_steps=opt.pruner_warmup_epochs
        ),
    )


def finished_trials(study: optuna.Study) -> list[FrozenTrial]:
    return [t for t in study.trials if t.state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)]


def log_trial(study: optuna.Study, trial: FrozenTrial) -> None:
    """Per-trial log line: state, score, duration, epochs, parameters."""
    duration = trial.duration.total_seconds() if trial.duration else float("nan")
    if trial.value is not None:
        score = f"{OBJECTIVE_METRIC}={trial.value:.5f}"
    elif trial.intermediate_values:
        last_step = max(trial.intermediate_values)
        score = f"last {OBJECTIVE_METRIC}={trial.intermediate_values[last_step]:.5f} @epoch {last_step}"
    else:
        score = "no score"
    print(
        f"[trial {trial.number:>3}] {trial.state.name:<8} {score:<38} {duration:7.1f}s "
        f"epochs={trial.user_attrs.get('epochs_run', 0)} params={trial.params}",
        flush=True,
    )


def run_study(
    prepared: PreparedData, cfg: Config, device: torch.device, n_trials: int, storage_path: Path
) -> optuna.Study:
    """Run (or resume) the study until it has n_trials finished trials."""
    opt = cfg.optuna
    study = open_study(opt, storage_path)
    remaining = n_trials - len(finished_trials(study))
    print(f"Study {opt.study_name!r} at {storage_path}: {len(finished_trials(study))} finished trials, "
          f"running {max(remaining, 0)} more (target {n_trials})")
    if remaining > 0:
        study.optimize(
            make_objective(prepared, cfg, device),
            n_trials=remaining,
            timeout=opt.timeout_seconds,
            # A diverging configuration (non-finite loss) is recorded as FAIL and the study continues.
            catch=(TrainingDivergedError,),
            callbacks=[log_trial],
            gc_after_trial=True,
        )
    return study


def summarize_study(study: optuna.Study) -> dict[str, int]:
    counts = {state.name.lower(): 0 for state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)}
    for t in study.trials:
        if t.state.name.lower() in counts:
            counts[t.state.name.lower()] += 1
    return counts


def save_best_hyperparameters(
    study: optuna.Study, cfg: Config, y_train, storage_path: Path, results_dir: Path
) -> Path:
    """Write the best trial's full hyperparameter set to results/best_hyperparameters.json (config.yaml is not modified)."""
    best = study.best_trial  # raises ValueError if no trial completed
    ann_cfg, train_cfg = suggest_configs(optuna.trial.FixedTrial(best.params), cfg)
    pos_weight = compute_pos_weight(y_train, train_cfg.class_weighting, train_cfg.pos_weight_scale)
    payload = {
        "study_name": study.study_name,
        "storage": str(storage_path),
        "objective": f"{OBJECTIVE_METRIC} (maximize)",
        "best_trial": best.number,
        "best_value": best.value,
        "best_epoch": best.user_attrs.get("best_epoch"),
        "val_metrics_at_best_epoch": best.user_attrs.get("val_metrics_at_best_epoch"),
        "sampled_params": best.params,
        "hyperparameters": resolved_hyperparameters(ann_cfg, train_cfg, pos_weight),
        "trial_counts": summarize_study(study),
        "validation_data": "load_centralized_data().val (combined client validation rows, pooled-train scaler)",
        "test_set_used": False,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / BEST_PARAMS_FILENAME
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    trials_csv = storage_path.with_name(f"{study.study_name}_trials.csv")
    study.trials_dataframe().to_csv(trials_csv, index=False)
    print(f"Saved {path}\nSaved trial log {trials_csv}")
    return path


# ---------------------------------------------------------------------------
# Final step: retrain best config once, select threshold on validation, test once
# ---------------------------------------------------------------------------


def final_retrain_and_test(
    study: optuna.Study,
    prepared: PreparedData,
    cfg: Config,
    device: torch.device,
    results_dir: Path,
    overwrite: bool = False,
) -> Path:
    """Retrain the best trial's configuration (no pruning) and evaluate it on the test set ONCE.

    Refuses to run if the results file or final checkpoints already exist, so the
    test set is not evaluated repeatedly by accident.
    """
    out_path = results_dir / FINAL_METRICS_FILENAME
    manager = CheckpointManager(cfg.optuna.final_checkpoint_dir)
    if not overwrite and (out_path.exists() or manager.has_best() or manager.has_latest()):
        raise SystemExit(
            f"{out_path} or checkpoints in {manager.directory} already exist — the final test evaluation "
            f"has already been run. Pass --overwrite to redo it."
        )

    best = study.best_trial
    ann_cfg, train_cfg = suggest_configs(optuna.trial.FixedTrial(best.params), cfg)
    hyperparameters = resolved_hyperparameters(
        ann_cfg, train_cfg,
        compute_pos_weight(prepared.data.train.y, train_cfg.class_weighting, train_cfg.pos_weight_scale),
    )
    print(f"\nRetraining best trial #{best.number} ({OBJECTIVE_METRIC}={best.value:.5f}) with full training (no pruning)")
    start = time.time()
    model, result, _ = train_configuration(
        prepared,
        cfg,
        ann_cfg,
        train_cfg,
        device,
        checkpoint_manager=manager,
        checkpoint_extra={
            "optuna": {"study_name": study.study_name, "trial": best.number, "params": best.params},
            "hyperparameters": hyperparameters,
        },
        verbose=True,
    )
    retrain_seconds = time.time() - start

    # Threshold from validation only.
    data = prepared.data
    p_val = predict_proba(model, data.val.X, batch_size=cfg.data.eval_batch_size, device=device)
    selection = select_threshold(data.val.y, p_val, cfg.evaluation)
    print(selection.validation_metrics.format_report(f"Validation ({selection.strategy})"))

    # The single test-set evaluation.
    test_metrics = evaluate_on_test(
        model,
        scaler=data.scaler,
        threshold=selection.threshold,
        feature_names=data.feature_names,
        dataset_dir=cfg.data.dataset_dir,
        batch_size=cfg.data.eval_batch_size,
        device=device,
    )
    print()
    print(test_metrics.format_report("HELD-OUT TEST (centralized ANN, Optuna-tuned)"))

    payload = {
        "model": "ANN (centralized, Optuna-tuned)",
        "study_name": study.study_name,
        "best_trial": best.number,
        "trial_val_pr_auc": best.value,
        "retrain": {
            "best_epoch": result.best_epoch,
            "last_epoch": result.last_epoch,
            "stopped_early": result.stopped_early,
            "best_val_pr_auc": result.best_value,
            "seconds": retrain_seconds,
            "history": result.history,
        },
        "hyperparameters": hyperparameters,
        "checkpoint": str(manager.best_path),
        "threshold_selection": selection.to_dict(),
        "test_metrics": test_metrics.to_dict(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {out_path}")
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Optuna search for the centralized ANN")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--n-trials", type=int, default=None, help="target number of finished trials (default: optuna.n_trials)")
    parser.add_argument("--storage", type=Path, default=None, help="SQLite file (default: optuna.storage)")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--final-test", action="store_true",
                        help="after the study: retrain the best config once and evaluate it on the test set once")
    parser.add_argument("--overwrite", action="store_true", help="allow redoing the final retrain + test evaluation")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    n_trials = args.n_trials if args.n_trials is not None else cfg.optuna.n_trials
    if n_trials < 1:
        raise SystemExit("--n-trials must be >= 1")
    storage_path = (args.storage or cfg.optuna.storage_path).resolve()

    final_out = args.results_dir / FINAL_METRICS_FILENAME
    if args.final_test and final_out.exists() and not args.overwrite:
        raise SystemExit(f"{final_out} already exists — the final test evaluation has already been run.")

    device = get_device(cfg.device)
    print(f"Device: {device}")
    prepared = prepare_data(cfg)
    print(f"Train rows: {len(prepared.data.train.y):,}  Validation rows: {len(prepared.data.val.y):,}")

    study = run_study(prepared, cfg, device, n_trials, storage_path)
    counts = summarize_study(study)
    print(f"Study finished: {counts}")
    if counts["complete"] == 0:
        raise SystemExit("No trial completed (all pruned/failed); nothing to save.")
    print(f"Best trial #{study.best_trial.number}: {OBJECTIVE_METRIC}={study.best_value:.5f}")
    save_best_hyperparameters(study, cfg, prepared.data.train.y, storage_path, args.results_dir)

    if args.final_test:
        final_retrain_and_test(study, prepared, cfg, device, args.results_dir, overwrite=args.overwrite)
    else:
        print("Skipping the final retrain + test evaluation (pass --final-test once tuning is complete).")


if __name__ == "__main__":
    main()
