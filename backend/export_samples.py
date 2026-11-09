"""Export the demo sample bank used by the Live Test page.

Draws a small, seeded set of real flows from the held-out test file (dataset/test.parquet) and attaches the
original CICIDS2017 traffic type (the test file only has the binary label; the type is recovered by an exact
match of the 77 feature values against the cleaned raw files). The rows are used only for single-row inference
demonstrations in the UI; no metric is computed from them for model selection.

    source venv/bin/activate
    python backend/export_samples.py          # writes backend/demo_samples.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "dataset" / "raw"
SEED = 42
# (traffic type, how many samples); "Benign" gets the most. Types with fewer test rows are capped automatically.
PLAN = [("Benign", 14), ("DoS Hulk", 3), ("DDoS", 3), ("PortScan", 3), ("DoS GoldenEye", 2), ("FTP-Patator", 2),
        ("SSH-Patator", 2), ("Bot", 2), ("DoS slowloris", 2), ("DoS Slowhttptest", 2), ("Web Attack – Brute Force", 2),
        ("Web Attack – XSS", 1), ("Web Attack – Sql Injection", 1), ("Infiltration", 1), ("Heartbleed", 1)]
RAW_FILES = ["Benign-Monday", "Bruteforce-Tuesday", "DoS-Wednesday", "WebAttacks-Thursday", "Infiltration-Thursday",
             "Botnet-Friday", "Portscan-Friday", "DDoS-Friday"]


def num(x):
    x = x.item() if hasattr(x, "item") else x
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return int(x) if x == int(x) and abs(x) < 1e15 else float(f"{x:.9g}")
    return x


def row_keys(df: pd.DataFrame) -> pd.Series:
    return pd.util.hash_pandas_object(df.astype("float64"), index=False)


def main() -> None:
    test = pd.read_parquet(ROOT / "dataset" / "test.parquet")
    features = [c for c in test.columns if c != "label"]
    raw = pd.concat([pd.read_parquet(RAW / f"{f}-no-metadata.parquet") for f in RAW_FILES], ignore_index=True)
    raw["Label"] = raw["Label"].astype(str).str.replace("�", "–", regex=False)
    raw = raw.drop_duplicates().reset_index(drop=True)
    assert list(raw.columns[:-1]) == features, "raw and test feature columns differ"
    raw["_k"] = row_keys(raw[features]).values
    # feature vectors that occur with more than one label are ambiguous and are excluded from the sample bank
    ambiguous = set(raw.groupby("_k")["Label"].nunique().loc[lambda s: s > 1].index)
    print(f"{len(ambiguous):,} ambiguous feature vectors excluded")
    lookup = dict(zip(raw["_k"].values, raw["Label"].values))
    keys = row_keys(test[features]).values
    test["type"] = [None if k in ambiguous else lookup.get(k) for k in keys]
    missing = int(test["type"].isna().sum())
    print(f"test rows {len(test):,}; traffic type found for {len(test) - missing:,}; missing {missing}")
    test = test.dropna(subset=["type"])
    assert ((test["type"] != "Benign").astype(int) == test["label"]).all(), "type does not agree with binary label"

    samples, rng = [], np.random.RandomState(SEED)
    for label, k in PLAN:
        pool = test[test["type"] == label]
        k = min(k, len(pool))
        pick = pool.sample(n=k, random_state=int(rng.randint(1 << 30)))
        for idx, r in pick.iterrows():
            samples.append({"id": int(idx), "type": label, "label": int(r["label"]),
                            "features": {c: num(r[c]) for c in features}})
        print(f"{label:28s} {k} of {len(pool):,}")
    rng.shuffle(samples)
    out = ROOT / "backend" / "demo_samples.json"
    out.write_text(json.dumps({"source": "dataset/test.parquet (held-out test rows, seed 42)", "feature_names": features,
                               "samples": samples}, indent=1))
    print("wrote", out, len(samples), "samples")


if __name__ == "__main__":
    main()
