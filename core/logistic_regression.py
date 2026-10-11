"""
Logistic Regression model for binary intrusion detection.

A single nn.Linear layer that outputs one raw logit per sample. No sigmoid is
applied inside the model: training uses BCEWithLogitsLoss, and sigmoid is
applied only at evaluation time when converting logits to probabilities.

Because it is a plain nn.Module with a fixed parameter layout (weight
[1, input_dim], bias [1]) and no buffers, every federated client builds an
identically shaped model, so its state_dict can be averaged directly by FedAvg.
"""

import torch
from torch import nn


class LogisticRegression(nn.Module):
    """Linear binary classifier producing a single logit per sample."""

    def __init__(self, input_dim: int) -> None:
        """
        Args:
            input_dim: Number of input features (inferred from the data by the caller).
        """
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map features of shape [batch, input_dim] to logits of shape [batch, 1]."""
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input of shape [batch, {self.input_dim}], got {tuple(x.shape)}"
            )
        return self.linear(x)
