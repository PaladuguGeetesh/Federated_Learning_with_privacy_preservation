# Privacy-Preserving Federated Learning Framework for Collaborative Malware Detection Across Organizations

Binary network-intrusion detection (Benign = 0, Attack = 1) on CICIDS2017. Three
simulated organizations (clients A, B, C) train a shared model with FedAvg
(Flower) without exchanging raw data. The federated models are compared with
centralized models trained on the pooled data:

| Model | Setting | Code |
|---|---|---|
| Random Forest | centralized baseline | `centralized/` |
| ANN (MLP), Optuna-tuned | centralized | `core/`, `core/optuna_search.py` |
| Logistic Regression | federated (sanity-check baseline) | `federated/` |
| ANN, same architecture as the tuned centralized ANN | federated | `federated/` |

## Layout

```
config.yaml              all configurable values (strictly validated by core/config.py)
preprocessing/           merge raw files, dedupe, binary label; EDA
centralized/             random_forest.py (model), train_rf.py (training + one-time test evaluation)
core/                    shared code
  config.py              loads/validates config.yaml
  dataset.py             Dataset, splits, scaling, DataLoaders, test-set loader
  ann.py                 configurable MLP          logistic_regression.py   single Linear layer
  train.py               training loop (+ python -m core.train)
  early_stopping.py      checkpoint.py           evaluate.py (metrics, threshold selection, final test CLI)
  optuna_search.py       hyperparameter search for the centralized ANN
federated/               partition_data.py, client.py, aggregation.py, server.py, common.py
results/                 result files + reporting scripts (sources.py, compare_results.py, print_*.py, plot_comparison.py)
graphs/                  generated plots
models/                  trained models / checkpoints (not in git)
dataset/                 parquet data (not in git)
tests/                   pytest suite (synthetic data only)
backend/, frontend/      empty placeholders — not implemented
```

## Data splits

- `dataset/test.parquet` — the held-out test set (20%). Never used for training, tuning,
  threshold selection or model selection.
- `dataset/train.parquet` — the remaining 80%, partitioned IID into
  `client_A/B/C.parquet`.
- Validation — a stratified 10% slice of each client file, taken in memory (the files are
  not modified). Centralized and federated runs use exactly the same training and
  validation rows. Decision thresholds are chosen on validation only.

## Run everything with one command

```bash
source venv/bin/activate
python run_project.py            # runs what is missing, then reports + tests; safe to re-run
```

It prints each stage's output in the terminal, writes `results/run_report.md`, and keeps full logs in `logs/`.
Stages whose outputs already exist are skipped, so the held-out test set is never re-evaluated by accident.
See [HOW_IT_WORKS.md](HOW_IT_WORKS.md) for the data flow, every stage and its options.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

All commands below are run **from the project root** with the virtual environment active.
Settings come from `config.yaml`; every command accepts `--config <path>` to use another file.

## Pipeline

### 1. Preprocessing (already done for this project)

Place the 8 dhoogla-cleaned CICIDS2017 parquet files in `dataset/raw/` (exact original names:
`Benign-Monday-no-metadata.parquet`, `Bruteforce-Tuesday-…`, `DoS-Wednesday-…`,
`WebAttacks-Thursday-…`, `Infiltration-Thursday-…`, `Botnet-Friday-…`, `Portscan-Friday-…`,
`DDoS-Friday-no-metadata.parquet`), then:

```bash
python preprocessing/preprocess.py   # -> dataset/cleaned_data.parquet
python preprocessing/eda.py          # class-distribution summary + graphs/label_distribution.png
```

### 2. Centralized Random Forest (creates the train/test split)

```bash
python -m centralized.train_rf
```

Creates `dataset/train.parquet` / `test.parquet` only if both are missing (an existing split is
never re-split), trains the RF on all of `train.parquet`, evaluates it once on the test set at
`evaluation.default_threshold`, and writes `models/rf_baseline.joblib`,
`results/centralized_rf_metrics.json`, `graphs/rf_confusion_matrix.png`, `graphs/rf_roc_curve.png`.

### 3. Client partitions

```bash
python -m federated.partition_data   # train.parquet -> dataset/client_{A,B,C}.parquet
```

### 4. Centralized ANN: Optuna search, then one final test evaluation

```bash
python -m core.optuna_search                 # run/resume the study (optuna.n_trials total trials)
python -m core.optuna_search --final-test    # once tuning is finished: retrain the best trial + single test evaluation
```

- The search uses validation PR-AUC (attack class) as its objective, prunes weak trials, and
  stores the study in `results/optuna/ann_centralized.db` (resumable; re-running continues up to
  `optuna.n_trials` finished trials and does not repeat finished ones).
- Outputs: `results/best_hyperparameters.json`, `results/optuna/ann_centralized_trials.csv`;
  with `--final-test` also `models/ann_centralized_tuned/{best,latest}.pt` and
  `results/ann_centralized_metrics.json`.
- `config.yaml` is not modified by the search. A plain, untuned training run with the
  `model.ann` / `training` defaults is available as `python -m core.train` (validation only).

### 5. Federated training (validation only)

```bash
python -m federated.server --model logistic_regression
python -m federated.server --model ann
```

- `federated.rounds` × `federated.local_epochs` (default 15 × 3); `--rounds` / `--local-epochs`
  override them for a single run.
- The federated ANN uses the architecture **and** local-training settings in
  `results/best_hyperparameters.json` (the centralized Optuna result).
- Each round the global model is scored on the combined validation set, scaled with the
  sample-count-weighted combination of the clients' scalers — the same scaler used for its test
  evaluation.
- Outputs: `results/federated_<model>_validation_rounds.json` and
  `models/federated/<model>/best.pt` (best round by validation PR-AUC) and `latest.pt`.

### 6. Federated models: one final test evaluation each

```bash
python -m core.evaluate --checkpoint models/federated/ann/best.pt --name ann_federated --final-test
python -m core.evaluate --checkpoint models/federated/logistic_regression/best.pt --name lr_federated --final-test
```

Selects the decision threshold on validation (`evaluation.threshold_strategy`, default: maximize
attack-class F1), then scores the test set once. Writes `results/<name>_test_metrics.json`.

### 7. Reports and plots

```bash
python -m results.compare_results       # results/final_comparison.csv / .json (+ gaps in percentage points)
python -m results.print_final_metrics   # full test report for every model
python -m results.print_convergence     # federated per-round validation tables
python -m results.plot_comparison       # graphs/final_comparison_bar.png, final_comparison_metrics.png,
                                        # convergence_overlay.png, federated_{lr,ann}_convergence.png
python -m results.plot_diagrams         # graphs/system_architecture.png, data_pipeline.png, federated_round_flow.png
```

These read the saved result files only; they never evaluate a model. (`plot_diagrams` reads only dataset
file metadata and `config.yaml`.)

## The test set is evaluated once per model

Every command that touches `dataset/test.parquet` — `centralized.train_rf`,
`core.optuna_search --final-test`, `core.evaluate --final-test` — refuses to run when that
model's test result already exists. `core.evaluate` additionally refuses without `--final-test`.
Each accepts `--overwrite` to redo the evaluation; use it only deliberately (for example after
fixing a bug), never to try alternatives against the test set. Tuning, threshold selection and
checkpoint selection use validation data only.

## Tests

```bash
pytest
```

Runs in a few seconds on small synthetic data in temporary directories; it never reads the real
dataset or writes to `results/`, `models/` or `graphs/`.
