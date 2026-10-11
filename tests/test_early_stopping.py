"""EarlyStopping in min and max mode."""

import pytest

from core.early_stopping import EarlyStopping


def _run(stopper, values):
    return [stopper.step(v, epoch) for epoch, v in enumerate(values, start=1)]


def test_min_mode_with_min_delta():
    stopper = EarlyStopping("val_loss", "min", patience=2, min_delta=0.01)
    # 0.945 and 0.944 are within min_delta of the best (0.95), so they do not count.
    assert _run(stopper, [1.0, 0.95, 0.945, 0.944]) == [True, True, False, False]
    assert stopper.should_stop and stopper.best_epoch == 2 and stopper.best_value == 0.95


def test_max_mode():
    stopper = EarlyStopping("val_pr_auc", "max", patience=2)
    assert _run(stopper, [0.5, 0.7, 0.69, 0.71]) == [True, True, False, True]
    assert not stopper.should_stop and stopper.best_value == 0.71


def test_nan_never_improves():
    stopper = EarlyStopping("val_pr_auc", "max", patience=1)
    stopper.step(0.5, 1)
    assert stopper.step(float("nan"), 2) is False
    assert stopper.should_stop


def test_state_round_trip_and_mismatch():
    stopper = EarlyStopping("val_loss", "min", patience=3)
    _run(stopper, [1.0, 1.1])
    restored = EarlyStopping("val_loss", "min", patience=3)
    restored.load_state_dict(stopper.state_dict())
    assert restored.state_dict() == stopper.state_dict()
    with pytest.raises(ValueError, match="monitors"):
        EarlyStopping("val_pr_auc", "max", patience=3).load_state_dict(stopper.state_dict())


@pytest.mark.parametrize("kwargs", [{"mode": "up"}, {"patience": 0}, {"min_delta": -1.0}])
def test_invalid_arguments(kwargs):
    args = {"metric": "val_loss", "mode": "min", "patience": 1, "min_delta": 0.0, **kwargs}
    with pytest.raises(ValueError):
        EarlyStopping(**args)
