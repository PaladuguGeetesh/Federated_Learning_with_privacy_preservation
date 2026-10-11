"""ANN and Logistic Regression model structure."""

import pytest
import torch

from core.ann import ANN, count_parameters, state_dict_shapes
from core.config import ANNConfig
from core.logistic_regression import LogisticRegression


@pytest.mark.parametrize("batch_norm", [False, True])
def test_ann_forward_shape_and_raw_logits(batch_norm):
    model = ANN(7, ANNConfig((16, 8), batch_norm, 0.1)).eval()
    out = model(torch.randn(5, 7) * 10)
    assert out.shape == (5, 1)
    assert ((out < 0) | (out > 1)).any()  # no sigmoid inside the model


def test_ann_parameter_count_matches_formula():
    model = ANN(7, ANNConfig((16, 8), False, 0.0))
    assert count_parameters(model) == {"trainable": 7 * 16 + 16 + 16 * 8 + 8 + 8 + 1, "buffers": 0}


def test_batchnorm_buffers_are_separate_from_parameters():
    model = ANN(7, ANNConfig((16, 8), True, 0.0))
    params = {n for n, _ in model.named_parameters()}
    buffers = {n for n, _ in model.named_buffers()}
    assert buffers == {f"blocks.{i}.bn.{k}" for i in (0, 1) for k in ("running_mean", "running_var", "num_batches_tracked")}
    assert not params & buffers
    assert not any(b.requires_grad for b in model.buffers())
    assert set(model.state_dict()) == params | buffers


def test_independently_built_models_have_identical_shapes():
    shapes = []
    for seed in range(3):
        torch.manual_seed(seed)
        shapes.append(state_dict_shapes(ANN(7, ANNConfig((16, 8), True, 0.2))))
    assert shapes[0] == shapes[1] == shapes[2]


def test_wrong_input_width_rejected():
    with pytest.raises(ValueError, match="Expected input"):
        ANN(7, ANNConfig((4,), False, 0.0))(torch.randn(2, 3))
    with pytest.raises(ValueError, match="Expected input"):
        LogisticRegression(7)(torch.randn(2, 3))


def test_logistic_regression_shape():
    model = LogisticRegression(7)
    assert model(torch.randn(4, 7)).shape == (4, 1)
    assert {k: tuple(v.shape) for k, v in model.state_dict().items()} == {"linear.weight": (1, 7), "linear.bias": (1,)}
