"""
Server-side aggregation of client updates (kept separate from round
orchestration in federated/server.py).

Two deliberately separate paths:

1. fedavg_trainable() — trainable parameters (weights, biases, BatchNorm
   gamma/beta). Standard FedAvg: the sample-count-weighted mean of the clients'
   locally trained values,  w = sum(n_k * w_k) / sum(n_k).

2. aggregate_bn_buffers() — BatchNorm running statistics. These are NOT
   gradient-trained weights; each client's running_mean / running_var estimate
   the mean / variance of a layer's activations on that client's data. They
   are combined as statistics, also weighted by sample count:
     running_mean: weighted mean of client means
     running_var:  pooled variance = weighted mean of client variances
                   + weighted spread of client means around the global mean
                   (law of total variance) — i.e. the variance of the pooled
                   activations, not just an average of variances
     num_batches_tracked: integer counter, not averaged; the max is kept.
                   (Only used by BatchNorm when momentum=None; core/ann.py
                   uses the default momentum=0.1, so it does not affect outputs.)
   This path only runs when the model actually has BatchNorm buffers.

Known caveat: averaging BatchNorm statistics across clients assumes their
activation distributions are similar (true here: the three clients are IID
stratified splits). Under strongly non-IID data, keeping BatchNorm local per
client (FedBN) is the usual alternative.
"""

import numpy as np

# One client's contribution: (arrays, number of local training samples).
ClientArrays = tuple[list[np.ndarray], int]


def _weights(updates: list[ClientArrays]) -> np.ndarray:
    if not updates:
        raise ValueError("No client updates to aggregate")
    counts = np.array([n for _, n in updates], dtype=np.float64)
    if (counts <= 0).any():
        raise ValueError(f"Every client must report a positive sample count, got {counts.tolist()}")
    lengths = {len(arrays) for arrays, _ in updates}
    if len(lengths) != 1:
        raise ValueError(f"Clients sent different numbers of arrays: {sorted(lengths)}")
    return counts / counts.sum()


def fedavg_trainable(updates: list[ClientArrays]) -> list[np.ndarray]:
    """Sample-count-weighted FedAvg of trainable parameters."""
    weights = _weights(updates)
    num_arrays = len(updates[0][0])
    aggregated = []
    for i in range(num_arrays):
        stacked = np.stack([arrays[i].astype(np.float64) for arrays, _ in updates])
        mean = np.tensordot(weights, stacked, axes=1)
        aggregated.append(mean.astype(updates[0][0][i].dtype))
    return aggregated


def aggregate_bn_buffers(updates: list[ClientArrays], kinds: list[str]) -> list[np.ndarray]:
    """Combine BatchNorm running statistics across clients (see module docstring).

    Args:
        updates: Per client, (buffer arrays in layout order, local sample count).
        kinds: For each buffer array, its kind: running_mean | running_var | num_batches_tracked.
            Each running_var must belong to the same layer as the running_mean
            directly before it (the order nn.BatchNorm1d registers them in).
    """
    weights = _weights(updates)
    if len(kinds) != len(updates[0][0]):
        raise ValueError(f"{len(kinds)} kinds given for {len(updates[0][0])} buffer arrays")

    aggregated: list[np.ndarray] = []
    global_means: dict[int, np.ndarray] = {}  # index of running_mean -> aggregated mean
    for i, kind in enumerate(kinds):
        stacked = np.stack([arrays[i] for arrays, _ in updates])
        dtype = updates[0][0][i].dtype
        if kind == "running_mean":
            mean = np.tensordot(weights, stacked.astype(np.float64), axes=1)
            global_means[i] = mean
            aggregated.append(mean.astype(dtype))
        elif kind == "running_var":
            if i == 0 or kinds[i - 1] != "running_mean":
                raise ValueError(f"running_var at index {i} is not preceded by its running_mean")
            client_means = np.stack([arrays[i - 1] for arrays, _ in updates]).astype(np.float64)
            within = np.tensordot(weights, stacked.astype(np.float64), axes=1)
            between = np.tensordot(weights, (client_means - global_means[i - 1]) ** 2, axes=1)
            aggregated.append((within + between).astype(dtype))
        elif kind == "num_batches_tracked":
            aggregated.append(stacked.max(axis=0).astype(dtype))
        else:
            raise ValueError(f"Unknown buffer kind {kind!r}")
    return aggregated
