"""
Configurable feed-forward network (MLP) for binary intrusion detection.

Architecture, per hidden layer i:  Linear -> [BatchNorm1d] -> ReLU -> [Dropout]
followed by a final Linear layer producing one raw logit per sample. No sigmoid
is applied inside the model (same convention as core/logistic_regression.py):
training uses BCEWithLogitsLoss, and sigmoid is applied only at evaluation time.

Federated compatibility: the architecture is a pure function of
(input_dim, ANNConfig), so every client that builds the model from the same
config.yaml and the same feature schema gets identical state_dict keys and
shapes. Hidden layers are named sub-modules (blocks.<i>.linear, blocks.<i>.bn)
rather than positional nn.Sequential indices, so the keys stay readable and do
not shift when dropout is turned on or off.

If batch_norm is enabled, each block's BatchNorm1d holds trainable parameters
(bn.weight, bn.bias) AND non-trainable running-statistics buffers
(bn.running_mean, bn.running_var, bn.num_batches_tracked). Both appear in
state_dict(), but only the former appear in parameters(). Federated
aggregation must treat the buffers separately (see federated/aggregation.py).
"""

import torch
from torch import nn

from core.config import ANNConfig


class HiddenBlock(nn.Module):
    """One hidden layer: Linear -> optional BatchNorm1d -> ReLU -> optional Dropout."""

    def __init__(self, in_features: int, out_features: int, batch_norm: bool, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features) if batch_norm else nn.Identity()
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.act(self.bn(self.linear(x))))


class ANN(nn.Module):
    """Feed-forward binary classifier producing a single logit per sample."""

    def __init__(self, input_dim: int, config: ANNConfig) -> None:
        """
        Args:
            input_dim: Number of input features, inferred from the data by the caller
                (e.g. len(feature_names)).
            config: Architecture settings loaded from config.yaml (model.ann).
        """
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        self.input_dim = input_dim
        self.config = config

        sizes = [input_dim, *config.hidden_sizes]
        self.blocks = nn.ModuleList(
            HiddenBlock(sizes[i], sizes[i + 1], config.batch_norm, config.dropout)
            for i in range(len(config.hidden_sizes))
        )
        self.output = nn.Linear(sizes[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map features of shape [batch, input_dim] to logits of shape [batch, 1]."""
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input of shape [batch, {self.input_dim}], got {tuple(x.shape)}"
            )
        for block in self.blocks:
            x = block(x)
        return self.output(x)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count trainable parameter elements and non-trainable buffer elements separately."""
    return {
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "buffers": sum(b.numel() for b in model.buffers()),
    }


def state_dict_shapes(model: nn.Module) -> dict[str, tuple[int, ...]]:
    """Map every state_dict key to its tensor shape (used to verify clients' models match)."""
    return {k: tuple(v.shape) for k, v in model.state_dict().items()}
