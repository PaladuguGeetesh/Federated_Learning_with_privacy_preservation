"""
Federated round orchestration (Flower simulation, FedAvg) across clients A, B, C.

Per round: send the global arrays to every client -> clients train locally ->
collect (arrays, local sample count) -> aggregate via federated/aggregation.py
-> evaluate the new global model on the combined VALIDATION set (every
client's validation slice), scaled with the sample-count-weighted combined
client scaler (core.dataset.combine_scalers). That is the same scaler the
one-time test evaluation uses, so the per-round validation trajectory and the
final test metric score the same model under the same input scaling.

The held-out test set is never read here. After the run, the global model is
saved as a checkpoint together with the sample-count-weighted combined client
scaler (core.dataset.combine_scalers); the one-time federated test evaluation
is done afterwards with:
    python -m core.evaluate --checkpoint models/federated/ann/best.pt --name ann_federated --final-test

Usage (from the project root):
    python -m federated.server                                  # settings from config.yaml (federated:)
    python -m federated.server --model logistic_regression --rounds 5 --local-epochs 1
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import flwr as fl
import numpy as np
import torch
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from sklearn.preprocessing import StandardScaler
from torch import nn

from core.ann import state_dict_shapes
from core.checkpoint import CheckpointManager, make_checkpoint
from core.config import DEFAULT_CONFIG_PATH, Config, load_config
from core.dataset import (
    CLIENT_IDS,
    LABEL_MAPPING,
    SplitArrays,
    apply_scaler,
    combine_scalers,
    get_device,
    load_all_clients,
    load_raw_validation,
    make_dataloader,
)
from core.evaluate import compute_metrics, predict_proba_loader
from core.train import evaluate_loss, set_seed
from federated.aggregation import aggregate_bn_buffers, fedavg_trainable
from federated.client import make_client_fn
from federated.common import FederatedSettings, ParameterLayout, build_model, get_arrays, resolve_settings, set_arrays

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Strategy: FedAvg with a separate BatchNorm-buffer path
# ---------------------------------------------------------------------------


class FedAvgWithBuffers(fl.server.strategy.FedAvg):
    """Flower FedAvg whose aggregation is delegated to federated/aggregation.py."""

    def __init__(self, layout: ParameterLayout, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.layout = layout
        self.round_log: list[dict[str, Any]] = []

    def aggregate_fit(self, server_round, results, failures):
        if failures:
            # Never aggregate a partial round silently.
            raise RuntimeError(f"Round {server_round}: {len(failures)} client(s) failed: {failures}")
        trainable_updates, buffer_updates, client_info = [], [], []
        for _, fit_res in results:
            # split() also checks every array's shape against the layout.
            trainable, buffers = self.layout.split(parameters_to_ndarrays(fit_res.parameters))
            trainable_updates.append((trainable, fit_res.num_examples))
            buffer_updates.append((buffers, fit_res.num_examples))
            client_info.append({"client": fit_res.metrics.get("client_id"), "num_examples": fit_res.num_examples,
                                "train_loss": fit_res.metrics.get("train_loss")})

        new_trainable = fedavg_trainable(trainable_updates)
        if self.layout.has_buffers:
            new_buffers = aggregate_bn_buffers(buffer_updates, self.layout.buffer_kinds())
        else:
            new_buffers = []

        total = sum(c["num_examples"] for c in client_info)
        train_loss = sum(c["train_loss"] * c["num_examples"] for c in client_info) / total
        self.round_log.append({
            "round": server_round,
            "clients": sorted(client_info, key=lambda c: c["client"]),
            "client_train_loss_weighted": train_loss,
            "bn_buffer_arrays_aggregated": len(new_buffers),
        })
        metrics = {"client_train_loss": train_loss, "bn_buffer_arrays_aggregated": len(new_buffers)}
        # Machine-readable progress line (read by the dashboard backend); logging only, no effect on training.
        print("FEDLOG " + json.dumps({"event": "fit", "round": server_round, "clients": self.round_log[-1]["clients"],
                                      "train_loss": train_loss, "arrays": len(new_trainable),
                                      "values_per_client": int(sum(a.size for a in new_trainable))}), flush=True)
        return ndarrays_to_parameters(new_trainable + new_buffers), metrics


# ---------------------------------------------------------------------------
# Server-side per-round validation
# ---------------------------------------------------------------------------


class RoundEvaluator:
    """evaluate_fn: scores the global model on the combined validation set every round.

    Loss is plain (unweighted) BCE so it is comparable across runs regardless of
    the clients' class weighting. Threshold-dependent metrics use
    evaluation.default_threshold; PR-AUC / ROC-AUC are threshold-free.
    Saves latest.pt every round and best.pt on the best validation PR-AUC.
    """

    def __init__(self, model: nn.Module, layout: ParameterLayout, val: SplitArrays, cfg: Config,
                 device: torch.device, checkpoint_manager: CheckpointManager, feature_names: list[str],
                 test_scaler: StandardScaler, checkpoint_extra: dict[str, Any]) -> None:
        self.model = model
        self.layout = layout
        self.device = device
        self.cfg = cfg
        self.loader = make_dataloader(val.to_dataset(), batch_size=cfg.data.eval_batch_size, shuffle=False,
                                      num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
                                      device=device, seed=cfg.seed)
        self.criterion = nn.BCEWithLogitsLoss()
        self.manager = checkpoint_manager
        self.feature_names = feature_names
        self.test_scaler = test_scaler
        self.checkpoint_extra = checkpoint_extra
        self.history: list[dict[str, Any]] = []
        self.best_round: int | None = None
        self.best_pr_auc = -np.inf
        self._start = time.time()

    def __call__(self, server_round: int, parameters: list[np.ndarray], config: dict[str, Any]):
        set_arrays(self.model, self.layout, parameters)
        y_true, y_prob = predict_proba_loader(self.model, self.loader, self.device)
        m = compute_metrics(y_true, y_prob, self.cfg.evaluation.default_threshold)
        loss = evaluate_loss(self.model, self.loader, self.criterion, self.device)
        record = {
            "round": server_round,
            "val_loss": loss,
            "val_pr_auc": m.pr_auc,
            "val_roc_auc": m.roc_auc,
            "val_macro_f1": m.macro_f1,
            "val_f1": m.f1,
            "val_attack_recall": m.recall,
            "val_false_negatives": m.fn,
            "val_fpr": m.fpr,
            "elapsed_seconds": time.time() - self._start,
        }
        self.history.append(record)

        if server_round > 0:  # round 0 = untrained initial weights
            improved = m.pr_auc > self.best_pr_auc
            checkpoint = make_checkpoint(
                epoch=server_round,
                model=self.model,
                training_config=self.checkpoint_extra["settings"]["local_training"],
                history=self.history,
                feature_names=self.feature_names,
                label_mapping=LABEL_MAPPING,
                seed=self.cfg.seed,
                scaler=self.test_scaler,
                extra={**self.checkpoint_extra, "round": server_round},
            )
            if improved:
                self.best_pr_auc, self.best_round = m.pr_auc, server_round
                self.manager.save_best(checkpoint)
            self.manager.save_latest(checkpoint)

        print("FEDLOG " + json.dumps({"event": "val", "round": server_round, "val_loss": loss, "pr_auc": m.pr_auc,
                                      "roc_auc": m.roc_auc, "f1": m.f1, "accuracy": m.accuracy, "precision": m.precision,
                                      "recall": m.recall, "fpr": m.fpr, "tp": m.tp, "fp": m.fp, "tn": m.tn, "fn": m.fn,
                                      "seconds": record["elapsed_seconds"]}), flush=True)
        print(f"[round {server_round:>3}] val_loss {loss:.5f}  val_pr_auc {m.pr_auc:.5f}  macro_f1 {m.macro_f1:.5f}  "
              f"attack_recall {m.recall:.5f}  FN {m.fn:,}  FPR {m.fpr:.5f}", flush=True)
        return loss, {k: v for k, v in record.items() if k not in ("round",)}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def check_client_shapes(settings: FederatedSettings, client_feature_names: dict[str, list[str]],
                        server_model: nn.Module) -> dict[str, tuple[int, ...]]:
    """Build each client's model from its own feature schema and confirm identical shapes everywhere."""
    reference = state_dict_shapes(server_model)
    for cid, names in client_feature_names.items():
        set_seed(CLIENT_IDS.index(cid))  # different init per client: only shapes must match
        shapes = state_dict_shapes(build_model(settings, len(names)))
        if shapes != reference:
            raise ValueError(f"Client {cid} model shapes differ from the server model: {shapes} vs {reference}")
    return reference


def run_federated(cfg: Config, settings: FederatedSettings, *, rounds: int, local_epochs: int,
                  checkpoint_dir: Path, results_path: Path, overwrite: bool = False) -> dict[str, Any]:
    """Run one federated simulation and write per-round validation results + checkpoints."""
    manager = CheckpointManager(checkpoint_dir)
    if not overwrite and (results_path.exists() or manager.has_best() or manager.has_latest()):
        raise SystemExit(f"{results_path} or checkpoints in {checkpoint_dir} already exist. Pass --overwrite to replace them.")

    device = get_device(cfg.device)
    print(f"Model: {settings.model} — {settings.source}")
    print(f"Rounds: {rounds}, local epochs: {local_epochs}, server device: {device}, "
          f"client device: {cfg.federated.client_device}")

    # Server-side validation data and client scaler statistics. (In this simulation the
    # server reads the client files directly; in a real deployment each client would
    # send its validation predictions/metrics and its scaler's (n, mean, var).)
    clients = load_all_clients(cfg.data.dataset_dir, cfg.data.val_fraction, cfg.seed)
    feature_names = clients[CLIENT_IDS[0]].feature_names
    client_features = {cid: c.feature_names for cid, c in clients.items()}
    test_scaler = combine_scalers([clients[cid].scaler for cid in CLIENT_IDS])
    client_sizes = {cid: len(c.train.y) for cid, c in clients.items()}
    del clients  # free the training arrays held here; clients load their own data

    # Combined validation rows (same rows and order as combined_federated_validation()),
    # scaled with the combined scaler rather than each client's own scaler.
    X_val_raw, y_val, val_features = load_raw_validation(cfg.data.dataset_dir, cfg.data.val_fraction, cfg.seed)
    if val_features != feature_names:
        raise ValueError("Validation feature schema differs from the clients' training schema")
    val = SplitArrays(apply_scaler(test_scaler, X_val_raw), y_val)
    del X_val_raw

    set_seed(cfg.seed)
    global_model = build_model(settings, len(feature_names)).to(device)
    layout = ParameterLayout.from_model(global_model)
    shapes = check_client_shapes(settings, client_features, global_model)
    print(f"Identical parameter shapes on server and clients {', '.join(CLIENT_IDS)}: "
          f"{len(layout.trainable)} trainable arrays, {len(layout.buffers)} BatchNorm buffer arrays")

    checkpoint_extra = {"federated": {"rounds": rounds, "local_epochs": local_epochs, "clients": list(CLIENT_IDS),
                                      "client_train_sizes": client_sizes},
                        "settings": settings.describe(),
                        "scaler": "sample-count-weighted combination of client scalers (core.dataset.combine_scalers)"}
    evaluator = RoundEvaluator(global_model, layout, val, cfg, device, manager, feature_names, test_scaler, checkpoint_extra)
    strategy = FedAvgWithBuffers(
        layout,
        fraction_fit=1.0,
        fraction_evaluate=0.0,  # no client-side evaluation; the server validates each round
        min_fit_clients=len(CLIENT_IDS),
        min_available_clients=len(CLIENT_IDS),
        accept_failures=False,
        initial_parameters=ndarrays_to_parameters(get_arrays(global_model)),
        on_fit_config_fn=lambda server_round: {"server_round": server_round, "local_epochs": local_epochs},
        evaluate_fn=evaluator,
    )

    # Ray worker processes must be able to import core/ and federated/. They inherit
    # this process's environment, so the project root goes on PYTHONPATH here.
    # (Ray's runtime_env={"env_vars": ...} is NOT used: it launches workers through
    # an unquoted shell command, which breaks when the project path contains a
    # space — workers then fail to start and Ray retries indefinitely.)
    existing = os.environ.get("PYTHONPATH")
    if str(PROJECT_ROOT) not in (existing or "").split(os.pathsep):
        os.environ["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + existing if existing else "")

    start = time.time()
    fl.simulation.start_simulation(
        client_fn=make_client_fn(cfg, settings),
        num_clients=len(CLIENT_IDS),  # simulation ids 0..2 -> A, B, C (see federated.client.make_client_fn)
        client_resources={"num_cpus": cfg.federated.client_num_cpus, "num_gpus": 0.0},
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
        ray_init_args={"ignore_reinit_error": True, "include_dashboard": False},
    )
    seconds = time.time() - start

    rounds_out = []
    fit_by_round = {r["round"]: r for r in strategy.round_log}
    for rec in evaluator.history:
        rounds_out.append({**rec, **({"fit": fit_by_round[rec["round"]]} if rec["round"] in fit_by_round else {})})
    payload = {
        "model": settings.model,
        "settings": settings.describe(),
        "rounds": rounds,
        "local_epochs": local_epochs,
        "parameter_shapes": {k: list(v) for k, v in shapes.items()},
        "validation_data": "all clients' validation slices, scaled with the combined (sample-weighted) client scaler",
        "validation_scaler": "core.dataset.combine_scalers (same scaler stored in the checkpoints for the test evaluation)",
        "best_round_by_val_pr_auc": evaluator.best_round,
        "best_val_pr_auc": evaluator.best_pr_auc,
        "wall_clock_seconds": seconds,
        "checkpoints": {"best": str(manager.best_path), "latest": str(manager.latest_path)},
        "test_set_used": False,
        "history": rounds_out,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Done in {seconds:.1f}s. Best round {evaluator.best_round} (val_pr_auc {evaluator.best_pr_auc:.5f}). Saved {results_path}")
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Federated simulation (Flower FedAvg)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model", choices=["ann", "logistic_regression"], default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    settings = resolve_settings(cfg, args.model)
    rounds = args.rounds or cfg.federated.rounds
    local_epochs = args.local_epochs or cfg.federated.local_epochs
    if rounds < 1 or local_epochs < 1:
        raise SystemExit("--rounds and --local-epochs must be >= 1")
    checkpoint_dir = args.checkpoint_dir or (cfg.federated.checkpoint_dir / settings.model)
    results_dir = args.results_dir or cfg.paths.results_dir
    run_federated(cfg, settings, rounds=rounds, local_epochs=local_epochs, checkpoint_dir=checkpoint_dir,
                  results_path=results_dir / f"federated_{settings.model}_validation_rounds.json",
                  overwrite=args.overwrite)


if __name__ == "__main__":
    main()
