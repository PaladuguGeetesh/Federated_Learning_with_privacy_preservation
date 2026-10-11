"""Model definitions for federated training: the Logistic Regression and small PyTorch MLP used with FedAvg."""

import torch.nn as nn
import torch as t
from torch.utils.data import Dataset
NUM_FEATURES = 77



import pandas as pd


if t.backends.mps.is_available():
    dev = t.device("mps")
elif t.cuda.is_available():
    dev = t.device("cuda")
else:
    dev = t.device("cpu")

print("Using device:", dev)

class CustomDS(Dataset):

    def __init__(self, f, l):
        self.f = t.tensor(f, dtype=t.float32)
        self.l = t.tensor(l, dtype=t.long)

    def __len__(self):
        return len(self.f)

    def __getitem__(self, idx):
        return self.f[idx], self.l[idx]
train_df = pd.read_parquet("dataset/train.parquet")
test_df = pd.read_parquet("dataset/test.parquet")

X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]

X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]   
train_ds = CustomDS(X_train.values, y_train.values)
test_ds = CustomDS(X_test.values, y_test.values)
