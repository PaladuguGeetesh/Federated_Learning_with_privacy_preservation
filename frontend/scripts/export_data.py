"""Export the project's real data and results to JSON files for the React UI.

Read-only: it reads dataset/raw/*.parquet, dataset/*.parquet metadata and the saved files in
results/. It never trains or evaluates a model, so the held-out test set is not touched.

    source venv/bin/activate
    python frontend/scripts/export_data.py

Writes frontend/src/data/dataset.json and frontend/src/data/results.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "frontend" / "src" / "data"
RAW = ROOT / "dataset" / "raw"
RES = ROOT / "results"
SEED = 42

RAW_FILES = [
    "Benign-Monday-no-metadata.parquet",
    "Bruteforce-Tuesday-no-metadata.parquet",
    "DoS-Wednesday-no-metadata.parquet",
    "WebAttacks-Thursday-no-metadata.parquet",
    "Infiltration-Thursday-no-metadata.parquet",
    "Botnet-Friday-no-metadata.parquet",
    "Portscan-Friday-no-metadata.parquet",
    "DDoS-Friday-no-metadata.parquet",
]
# (attack-type label, how many rows) shown in the sample table: some normal, some attacks
SAMPLE_PLAN = [("Benign", 2), ("DDoS", 1), ("PortScan", 1), ("DoS Hulk", 1)]


def clean(o):
    """JSON-safe: NaN/inf -> None, numpy scalars -> python."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if hasattr(o, "item"):
        o = o.item()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


def num(x):
    x = x.item() if hasattr(x, "item") else x
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return int(x) if x == int(x) and abs(x) < 1e15 else float(f"{x:.6g}")
    return x


def build_dataset() -> dict:
    print("loading raw CICIDS2017 files ...")
    df = pd.concat([pd.read_parquet(RAW / f) for f in RAW_FILES], ignore_index=True)
    df["Label"] = df["Label"].astype(str).str.replace("\ufffd", "\u2013", regex=False)
    raw_rows = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"raw {raw_rows:,} -> {len(df):,} after removing duplicates")
    assert len(df) == 2_231_806, "does not match the cleaned data set used in the report"

    features = [c for c in df.columns if c != "Label"]
    assert len(features) == 77
    attack = df["Label"] != "Benign"

    cols = []
    for i, c in enumerate(features, 1):
        s = df[c]
        cols.append({
            "index": i, "name": c, "dtype": str(s.dtype),
            "min": num(s.min()), "max": num(s.max()), "mean": num(s.mean()),
            "benign_mean": num(s[~attack].mean()), "attack_mean": num(s[attack].mean()),
        })

    counts = df["Label"].value_counts()
    total = len(df)
    types = [{"label": k, "count": int(v), "share": float(v) / total,
              "binary": 0 if k == "Benign" else 1} for k, v in counts.items()]

    # splits (row counts from parquet metadata; attacks from the binary label column)
    splits = []
    desc = {
        "cleaned_data.parquet": "Cleaned data (all flows)",
        "train.parquet": "Training pool (80%)",
        "test.parquet": "Held-out test set (20%)",
        "client_A.parquet": "Client A (IID share of train)",
        "client_B.parquet": "Client B (IID share of train)",
        "client_C.parquet": "Client C (IID share of train)",
    }
    for f, d in desc.items():
        p = ROOT / "dataset" / f
        n = pq.ParquetFile(p).metadata.num_rows
        a = int(pd.read_parquet(p, columns=["label"])["label"].sum())
        splits.append({"file": f, "description": d, "rows": n, "attacks": a, "benign": n - a})

    # sample rows: seeded random draw per label from the same data
    samples = []
    for label, k in SAMPLE_PLAN:
        pick = df[df["Label"] == label].sample(n=k, random_state=SEED)
        for idx, row in pick.iterrows():
            samples.append({
                "id": int(idx), "type": label, "label": 0 if label == "Benign" else 1,
                "values": {c: num(row[c]) for c in features},
            })
    return {
        "name": "CICIDS2017 (dhoogla cleaned parquet)",
        "raw_rows": raw_rows, "rows": total, "duplicates_removed": raw_rows - total,
        "n_features": len(features), "label_column": "label (Benign = 0, Attack = 1)",
        "columns": cols, "attack_types": types, "splits": splits, "samples": samples,
        "sample_note": f"Rows drawn at random (seed {SEED}) from the cleaned data; "
                       "the attack-type name comes from the original CICIDS2017 Label column.",
    }


def build_results() -> dict:
    fc = json.loads((RES / "final_comparison.json").read_text().replace("NaN", "null"))
    tests = {}
    for key, f in [("Federated ANN", "ann_federated_test_metrics"),
                   ("Federated LR", "lr_federated_test_metrics")]:
        tests[key] = json.loads((RES / f"{f}.json").read_text())["test_metrics"]
    tests["Centralized ANN (tuned)"] = json.loads((RES / "ann_centralized_metrics.json").read_text())["test_metrics"]
    n_attack, n_benign = 67299, 379063
    models = []
    for m in fc["table"]:
        t = tests.get(m["model"])
        fn, fp = m["false_negatives"], m["false_positives"]
        m = dict(m)
        m.update({"tp": n_attack - fn, "fn": fn, "fp": fp, "tn": n_benign - fp})
        if t:
            assert (t["tp"], t["fp"], t["tn"], t["fn"]) == (m["tp"], fp, m["tn"], fn)
        models.append(m)
    rf = json.loads((RES / "centralized_rf_metrics.json").read_text())

    def rounds(path):
        d = json.loads(Path(path).read_text())
        hist = []
        for h in d["history"]:
            row = {k: h[k] for k in ("round", "val_loss", "val_pr_auc", "val_roc_auc", "val_f1",
                                      "val_attack_recall", "val_false_negatives", "val_fpr", "elapsed_seconds")}
            if "fit" in h:
                row["clients"] = {c["client"]: c["train_loss"] for c in h["fit"]["clients"]}
                row["train_loss"] = h["fit"]["client_train_loss_weighted"]
            hist.append(row)
        return {"rounds": d["rounds"], "local_epochs": d["local_epochs"],
                "best_round": d["best_round_by_val_pr_auc"], "best_val_pr_auc": d["best_val_pr_auc"],
                "seconds": d["wall_clock_seconds"], "history": hist}

    fed = {
        "ann": rounds(RES / "federated_ann_validation_rounds.json"),
        "lr": rounds(RES / "federated_logistic_regression_validation_rounds.json"),
        "ann_local1": rounds(RES / "federated_ann_local1" / "federated_ann_validation_rounds.json"),
    }
    tr = pd.read_csv(RES / "optuna" / "ann_centralized_trials.csv")
    trials = [{
        "trial": int(r["number"]), "state": r["state"], "pr_auc": num(r["value"]),
        "minutes": round(pd.Timedelta(r["duration"]).total_seconds() / 60, 2),
        "layers": num(r["params_n_layers"]), "optimizer": r["params_optimizer"],
        "dropout": num(r["params_dropout"]), "lr": num(r["params_learning_rate"]),
        "batch": num(r["params_batch_size"]),
    } for _, r in tr.iterrows()]
    best = json.loads((RES / "best_hyperparameters.json").read_text())
    return clean({
        "test_set": {"flows": 446362, "attacks": n_attack, "benign": n_benign},
        "models": models, "gaps_ann": fc["gap_pp_centralized_ann_minus_federated_ann"],
        "gaps_rf": fc["gap_pp_centralized_rf_minus_federated_ann"], "notes": fc["notes"],
        "rf": {"params": rf["hyperparameters"], "train_seconds": rf["training_time_seconds"]},
        "federated": fed, "optuna": {"trials": trials, "best_trial": best["best_trial"],
                                     "best_value": best["best_value"], "params": best["sampled_params"]},
    })


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    ds = clean(build_dataset())
    (OUT / "dataset.json").write_text(json.dumps(ds, indent=1))
    rs = build_results()
    (OUT / "results.json").write_text(json.dumps(rs, indent=1))
    print("wrote", OUT / "dataset.json", OUT / "results.json")
