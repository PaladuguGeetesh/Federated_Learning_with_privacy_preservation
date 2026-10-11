# Privacy-Preserving Federated Learning Framework for Collaborative Malware Detection Across Organizations

This project explores privacy-preserving collaborative malware detection using federated
learning. Three simulated client organizations (Client A, B, C) train models locally
without sharing raw data, coordinated via the Flower (FLWR) framework using FedAvg. The
project compares a centralized Random Forest baseline (upper-bound reference, pooled
data) against two federated models — a Logistic Regression sanity-check model and a
small PyTorch MLP (the main result) — to evaluate the trade-offs between privacy
preservation and detection performance. A FastAPI backend and React frontend dashboard
will expose training results and visualizations.

## Folder Structure

```
malware-fl-project/
├── dataset/            # CICIDS2017 (dhoogla-cleaned) parquet files
│   └── raw/             # Place the 8 raw parquet files here (see "How to run")
├── preprocessing/       # Dataset-agnostic preprocessing pipeline
├── centralized/         # Centralized Random Forest baseline
├── federated/           # Federated learning code (Flower)
│   ├── client.py        # Flower client implementation
│   ├── server.py        # Flower server / FedAvg strategy
│   ├── models.py        # Logistic Regression + MLP model definitions
│   └── utils.py         # Shared federated helper functions
├── backend/             # FastAPI backend
├── frontend/            # React dashboard
├── models/              # Saved/trained model artifacts
├── results/             # Evaluation results (metrics, logs)
├── graphs/              # Generated plots and figures
├── requirements.txt     # Pinned Python dependencies
└── README.md
```

## How to Run

### 1. Dataset

This project uses the CICIDS2017 dataset (dhoogla-cleaned mirror on Kaggle), provided
as 8 pre-cleaned parquet files (no nulls, no infinite values, no identifier columns).
Place all 8 files in `dataset/raw/` using their exact original filenames:

```
dataset/raw/
├── Benign-Monday-no-metadata.parquet
├── Bruteforce-Tuesday-no-metadata.parquet
├── DoS-Wednesday-no-metadata.parquet
├── WebAttacks-Thursday-no-metadata.parquet
├── Infiltration-Thursday-no-metadata.parquet
├── Botnet-Friday-no-metadata.parquet
├── Portscan-Friday-no-metadata.parquet
└── DDoS-Friday-no-metadata.parquet
```

### 2. Activate the virtual environment

```bash
source venv/bin/activate
```

### 3. Run preprocessing

Merges the 8 files, removes duplicates, encodes the binary `label` column
(Benign=0, Attack=1), and saves `dataset/cleaned_data.parquet`:

```bash
python preprocessing/preprocess.py
```

### 4. Run EDA

Prints a before/after-dedup and shape summary, and saves a class distribution
bar chart to `graphs/label_distribution.png`:

```bash
python preprocessing/eda.py
```
