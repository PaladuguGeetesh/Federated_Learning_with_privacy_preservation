"""Small API that lets the React dashboard call the trained models and re-run the quick tasks.

    source venv/bin/activate
    uvicorn backend.main:app --port 8000          # from the project root

What it offers (everything here is fast; nothing re-trains the big models):
  GET  /api/health         models loaded, device, versions
  GET  /api/samples        the demo sample bank (real flows from the held-out test file)
  POST /api/predict        score one flow (a sample id or edited feature values) with all four models
  POST /api/predict-batch  score several samples with all four models
  POST /api/quick-eval     re-score N random *validation* rows (cached) with adjustable thresholds
  POST /api/jobs           start a background job: "tests" (pytest) or "federated" (mini FedAvg run)
  GET  /api/jobs/{id}      status, live log and result of a job

The held-out test set is never scored in bulk here: the official one-time test evaluation is stored in results/.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.checkpoint import load_checkpoint
from core.config import PROJECT_ROOT, load_config
from core.dataset import apply_scaler, load_raw_validation
from core.evaluate import compute_metrics, predict_proba

ROOT = PROJECT_ROOT
RUNTIME = ROOT / "backend" / ".runtime"
SAMPLES = json.loads((ROOT / "backend" / "demo_samples.json").read_text())
FEATURES: list[str] = SAMPLES["feature_names"]
SAMPLE_BY_ID = {s["id"]: s for s in SAMPLES["samples"]}
CFG = load_config()

MODEL_FILES = {
    "Centralized RF": ("rf", ROOT / "models" / "rf_baseline.joblib"),
    "Centralized ANN": ("ckpt", ROOT / "models" / "ann_centralized_tuned" / "best.pt"),
    "Federated ANN": ("ckpt", ROOT / "models" / "federated" / "ann" / "best.pt"),
    "Federated LR": ("ckpt", ROOT / "models" / "federated" / "logistic_regression" / "best.pt"),
}
THRESHOLD_NAME = {"Centralized RF": "Centralized RF", "Centralized ANN": "Centralized ANN (tuned)",
                  "Federated ANN": "Federated ANN", "Federated LR": "Federated LR"}

app = FastAPI(title="Federated malware detection API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


# ------------------------------------------------------------------ models
class Loaded:
    def __init__(self, name: str, kind: str, model: Any, scaler: Any, threshold: float, source: str):
        self.name, self.kind, self.model, self.scaler, self.threshold, self.source = name, kind, model, scaler, threshold, source
        # the Random Forest was fitted on every training row, which includes the validation slices
        self.saw_validation = kind == "rf"

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.kind == "rf":
            return predict_proba(self.model, X.astype(np.float32), batch_size=4096)
        return predict_proba(self.model, apply_scaler(self.scaler, X).astype(np.float32), batch_size=4096, device=torch.device("cpu"))


MODELS: dict[str, Loaded] = {}
LOAD_ERRORS: dict[str, str] = {}


def load_models() -> None:
    final = json.loads((ROOT / "results" / "final_comparison.json").read_text().replace("NaN", "null"))
    thr = {m["model"]: m["threshold"] for m in final["table"]}
    for name, (kind, path) in MODEL_FILES.items():
        try:
            if kind == "rf":
                model = joblib.load(path)
                if list(getattr(model, "feature_names_in_", FEATURES)) != FEATURES:
                    raise ValueError("feature order of the Random Forest differs from the sample bank")
                MODELS[name] = Loaded(name, kind, model, None, float(thr[THRESHOLD_NAME[name]]), path.name)
            else:
                ck = load_checkpoint(path)
                if ck.feature_names != FEATURES:
                    raise ValueError("feature order of the checkpoint differs from the sample bank")
                model = ck.build_model(torch.device("cpu")).eval()
                MODELS[name] = Loaded(name, kind, model, ck.build_scaler(), float(thr[THRESHOLD_NAME[name]]),
                                      str(path.relative_to(ROOT)))
        except Exception as exc:  # keep the API up and report which model is missing
            LOAD_ERRORS[name] = f"{type(exc).__name__}: {exc}"


@app.on_event("startup")
def _startup() -> None:
    load_models()


def need_models() -> None:
    if not MODELS:
        raise HTTPException(503, f"No model could be loaded: {LOAD_ERRORS}")


# ------------------------------------------------------------------ schemas
class PredictRequest(BaseModel):
    sample_id: int | None = None
    features: dict[str, float] | None = None
    thresholds: dict[str, float] | None = None


class BatchRequest(BaseModel):
    sample_ids: list[int] | None = None  # None = the whole bank
    thresholds: dict[str, float] | None = None


class QuickEvalRequest(BaseModel):
    n: int = Field(20000, ge=500, le=100000)
    seed: int = Field(0, ge=0, le=10**6)
    thresholds: dict[str, float] | None = None
    models: list[str] | None = None


class JobRequest(BaseModel):
    kind: str  # "tests" | "federated"
    model: str = "logistic_regression"  # federated only: "logistic_regression" | "ann"
    rounds: int = Field(3, ge=1, le=5)
    local_epochs: int = Field(1, ge=1, le=2)


# ------------------------------------------------------------------ endpoints
@app.get("/api/health")
def health() -> dict[str, Any]:
    import sklearn
    return {
        "status": "ok", "loaded": {k: {"threshold": v.threshold, "source": v.source} for k, v in MODELS.items()},
        "errors": LOAD_ERRORS, "device": "cpu (inference)", "torch": torch.__version__, "sklearn": sklearn.__version__,
        "n_samples": len(SAMPLES["samples"]), "n_features": len(FEATURES),
    }


@app.get("/api/samples")
def samples() -> dict[str, Any]:
    return {"source": SAMPLES["source"], "feature_names": FEATURES, "samples": SAMPLES["samples"]}


def _verdict(p: float, thr: float) -> str:
    return "Attack" if p >= thr else "Benign"


def _predict_rows(X: np.ndarray, thresholds: dict[str, float] | None) -> dict[str, dict[str, Any]]:
    out = {}
    for name, m in MODELS.items():
        t0 = time.perf_counter()
        p = m.score(X)
        ms = (time.perf_counter() - t0) * 1000
        thr = (thresholds or {}).get(name, m.threshold)
        out[name] = {"prob": p.tolist(), "threshold": thr, "default_threshold": m.threshold, "ms": ms,
                     "pred": [1 if v >= thr else 0 for v in p]}
    return out


@app.post("/api/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    need_models()
    truth = None
    if req.features is not None:
        missing = [f for f in FEATURES if f not in req.features]
        if missing:
            raise HTTPException(422, f"Missing features: {missing[:5]}{'…' if len(missing) > 5 else ''}")
        x = np.array([[req.features[f] for f in FEATURES]], dtype=np.float64)
        if req.sample_id is not None and req.sample_id in SAMPLE_BY_ID:
            truth = SAMPLE_BY_ID[req.sample_id]["label"]
    elif req.sample_id is not None:
        if req.sample_id not in SAMPLE_BY_ID:
            raise HTTPException(404, f"Unknown sample id {req.sample_id}")
        s = SAMPLE_BY_ID[req.sample_id]
        x, truth = np.array([[s["features"][f] for f in FEATURES]], dtype=np.float64), s["label"]
    else:
        raise HTTPException(422, "Send sample_id or features")
    res = _predict_rows(x, req.thresholds)
    models = {}
    for name, r in res.items():
        models[name] = {"prob": r["prob"][0], "pred": r["pred"][0], "verdict": _verdict(r["prob"][0], r["threshold"]),
                        "threshold": r["threshold"], "default_threshold": r["default_threshold"], "ms": r["ms"]}
    return {"truth": truth, "models": models}


@app.post("/api/predict-batch")
def predict_batch(req: BatchRequest) -> dict[str, Any]:
    need_models()
    ids = req.sample_ids or [s["id"] for s in SAMPLES["samples"]]
    rows = [SAMPLE_BY_ID[i] for i in ids if i in SAMPLE_BY_ID]
    X = np.array([[s["features"][f] for f in FEATURES] for s in rows], dtype=np.float64)
    y = [s["label"] for s in rows]
    t0 = time.perf_counter()
    res = _predict_rows(X, req.thresholds)
    total_ms = (time.perf_counter() - t0) * 1000
    summary = {}
    for name, r in res.items():
        pred = r["pred"]
        tp = sum(1 for a, b in zip(y, pred) if a == 1 and b == 1)
        tn = sum(1 for a, b in zip(y, pred) if a == 0 and b == 0)
        fp = sum(1 for a, b in zip(y, pred) if a == 0 and b == 1)
        fn = sum(1 for a, b in zip(y, pred) if a == 1 and b == 0)
        summary[name] = {"correct": tp + tn, "total": len(y), "tp": tp, "tn": tn, "fp": fp, "fn": fn, "ms": r["ms"],
                         "threshold": r["threshold"]}
    return {"ids": [s["id"] for s in rows], "truth": y, "total_ms": total_ms, "summary": summary,
            "rows": {name: {"prob": r["prob"], "pred": r["pred"]} for name, r in res.items()}}


_VAL: dict[str, Any] = {}
_PROBS: dict[tuple, np.ndarray] = {}
_LOCK = threading.Lock()


@app.post("/api/quick-eval")
def quick_eval(req: QuickEvalRequest) -> dict[str, Any]:
    need_models()
    t_start = time.perf_counter()
    with _LOCK:
        if "X" not in _VAL:
            t0 = time.perf_counter()
            X, y, names = load_raw_validation(CFG.data.dataset_dir, CFG.data.val_fraction, CFG.seed)
            if names != FEATURES:
                raise HTTPException(500, "validation feature order differs from the sample bank")
            _VAL.update(X=X, y=y, load_s=time.perf_counter() - t0)
        X_all, y_all = _VAL["X"], _VAL["y"]
        n = min(req.n, len(y_all))
        idx = np.sort(np.random.RandomState(req.seed).choice(len(y_all), size=n, replace=False))
        X, y = X_all[idx], y_all[idx]
        wanted = [m for m in (req.models or list(MODELS)) if m in MODELS]
        out = {}
        for name in wanted:
            m = MODELS[name]
            key = (name, n, req.seed)
            cached = key in _PROBS
            t0 = time.perf_counter()
            if not cached:
                if len(_PROBS) > 24:
                    _PROBS.clear()
                _PROBS[key] = m.score(X)
            score_s = time.perf_counter() - t0
            thr = (req.thresholds or {}).get(name, m.threshold)
            met = compute_metrics(y, _PROBS[key], thr).to_dict()
            met = {k: (None if isinstance(v, float) and v != v else v) for k, v in met.items()}
            out[name] = {"metrics": met, "threshold": thr, "default_threshold": m.threshold, "seconds": score_s, "cached": cached,
                         "saw_these_rows_in_training": m.saw_validation}
    return {"n": int(n), "attacks": int(y.sum()), "benign": int(len(y) - y.sum()), "seed": req.seed,
            "seconds": time.perf_counter() - t_start, "validation_rows_total": int(len(y_all)), "models": out,
            "note": "Validation rows, not the held-out test set. The Random Forest was trained on all training rows, which "
                    "include these rows, so its score here is optimistic."}


# ------------------------------------------------------------------ background jobs
ANSI = re.compile(r"\x1b\[[0-9;]*m")
JOBS: dict[str, dict[str, Any]] = {}
_JOB_LOCK = threading.Lock()


def _run_job(job: dict[str, Any], cmd: list[str], finish) -> None:
    try:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        job["pid"] = proc.pid
        for line in proc.stdout:  # type: ignore[union-attr]
            job["log"].append(ANSI.sub("", line.rstrip("\n")))
            job["log"] = job["log"][-400:]
        code = proc.wait()
        job["exit_code"] = code
        job["result"] = finish(job, code)
        job["status"] = "done" if code == 0 else "failed"
    except Exception as exc:
        job["log"].append(f"{type(exc).__name__}: {exc}")
        job["status"] = "failed"
    job["seconds"] = time.time() - job["started"]


def _finish_tests(job: dict[str, Any], code: int) -> dict[str, Any]:
    text = "\n".join(job["log"])
    m = re.search(r"(\d+) passed(?:, (\d+) failed)?[^\n]*? in ([\d.]+)s", text)
    return {"passed": int(m.group(1)) if m else None, "failed": int(m.group(2) or 0) if m else None,
            "pytest_seconds": float(m.group(3)) if m else None}


def _finish_fed(job: dict[str, Any], code: int) -> dict[str, Any] | None:
    f = Path(job["out_dir"]) / f"federated_{job['params']['model']}_validation_rounds.json"
    if code != 0 or not f.exists():
        return None
    d = json.loads(f.read_text())
    hist = []
    for h in d["history"]:
        hist.append({k: h.get(k) for k in ("round", "val_loss", "val_pr_auc", "val_f1", "val_attack_recall", "val_fpr", "val_false_negatives")})
    return {"rounds": d["rounds"], "local_epochs": d["local_epochs"], "seconds": d["wall_clock_seconds"],
            "best_round": d["best_round_by_val_pr_auc"], "best_val_pr_auc": d["best_val_pr_auc"], "history": hist,
            "note": "Written to backend/.runtime only; the saved project results and models are untouched."}


@app.post("/api/jobs")
def start_job(req: JobRequest) -> dict[str, Any]:
    with _JOB_LOCK:
        if any(j["status"] == "running" for j in JOBS.values()):
            raise HTTPException(409, "Another job is still running")
        jid = uuid.uuid4().hex[:8]
        job: dict[str, Any] = {"id": jid, "kind": req.kind, "status": "running", "started": time.time(), "log": [],
                               "params": req.model_dump(), "result": None}
        if req.kind == "tests":
            cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
            finish = _finish_tests
        elif req.kind == "federated":
            if req.model not in ("logistic_regression", "ann"):
                raise HTTPException(422, "model must be logistic_regression or ann")
            out = RUNTIME / f"fed_{jid}"
            job["out_dir"] = str(out)
            cmd = [sys.executable, "-m", "federated.server", "--model", req.model, "--rounds", str(req.rounds),
                   "--local-epochs", str(req.local_epochs), "--checkpoint-dir", str(out / "ckpt"),
                   "--results-dir", str(out), "--overwrite"]
            finish = _finish_fed
        else:
            raise HTTPException(422, "kind must be tests or federated")
        JOBS[jid] = job
    threading.Thread(target=_run_job, args=(job, cmd, finish), daemon=True).start()
    return _public(job)


def _public(job: dict[str, Any]) -> dict[str, Any]:
    return {"id": job["id"], "kind": job["kind"], "status": job["status"], "params": job["params"],
            "elapsed": (time.time() - job["started"]) if job["status"] == "running" else job.get("seconds"),
            "log": job["log"][-60:], "result": job["result"], "exit_code": job.get("exit_code")}


@app.get("/api/jobs/{jid}")
def get_job(jid: str) -> dict[str, Any]:
    if jid not in JOBS:
        raise HTTPException(404, "Unknown job")
    return _public(JOBS[jid])
