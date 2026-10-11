# results/print_final_metrics.py
import json

MODELS = [
    ("Centralized RF", "results/centralized_rf_metrics.json", None),
    ("Federated LR", "results/federated_lr_metrics.json", -1),
    ("Federated MLP", "results/federated_mlp_metrics.json", -1),
]

# canonical name -> acceptable key variants (checked case-insensitively)
METRIC_ALIASES = {
    "accuracy": ["accuracy"],
    "precision": ["precision"],
    "recall": ["recall"],
    "f1": ["f1", "f1_score", "f1score"],
    "roc_auc": ["roc_auc", "rocauc", "auc"],
}

def load_metrics(path, index):
    with open(path) as f:
        data = json.load(f)
    record = data if index is None else data[index]
    # Single-record files (e.g. the centralized RF file) nest the actual
    # metric values one level down under a "metrics" key rather than at
    # the top level — unwrap that here if present.
    if isinstance(record, dict) and "metrics" in record and isinstance(record["metrics"], dict):
        record = record["metrics"]
    return record

def get_metric(record, canonical_name):
    lower_record = {k.lower(): v for k, v in record.items()}
    for variant in METRIC_ALIASES[canonical_name]:
        if variant.lower() in lower_record:
            return lower_record[variant.lower()]
    raise KeyError(f"None of {METRIC_ALIASES[canonical_name]} found in keys: {list(record.keys())}")

rows = []
for name, path, index in MODELS:
    m = load_metrics(path, index)
    rows.append((name, [get_metric(m, k) for k in METRIC_ALIASES]))

col_w = 14
labels = list(METRIC_ALIASES.keys())
header = f"{'Model':<20}" + "".join(f"{lbl.capitalize():>{col_w}}" for lbl in labels)
print("=" * len(header))
print("FINAL MODEL METRICS")
print("=" * len(header))
print(header)
print("-" * len(header))
for name, values in rows:
    print(f"{name:<20}" + "".join(f"{v:>{col_w}.4f}" for v in values))
print("=" * len(header))

rf_acc = rows[0][1][0]
print("\nGap to centralized RF ceiling (accuracy):")
for name, values in rows[1:]:
    gap = (rf_acc - values[0]) * 100
    print(f"  {name}: {gap:.2f} percentage points behind")