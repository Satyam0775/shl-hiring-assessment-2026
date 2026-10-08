import numpy as np
import pytest

from src.evaluation.cross_validation import cross_validate, make_folds
from src.evaluation.metrics import clip_predictions, evaluate, pearson, rmse


def test_rmse_known_value():
    assert rmse([1, 2, 3], [1, 2, 3]) == 0.0
    assert rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))


def test_pearson_known_values():
    assert pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)
    assert pearson([1, 2, 3, 4], [8, 6, 4, 2]) == pytest.approx(-1.0)


def test_pearson_constant_prediction_is_zero():
    assert pearson([1, 2, 3], [2, 2, 2]) == 0.0
    assert pearson([2, 2, 2], [1, 2, 3]) == 0.0


@pytest.mark.parametrize("fn", [rmse, pearson])
def test_edge_cases_raise(fn):
    with pytest.raises(ValueError):
        fn([], [])
    with pytest.raises(ValueError):
        fn([1, 2], [1])
    with pytest.raises(ValueError):
        fn([1, np.nan], [1, 2])
    with pytest.raises(ValueError):
        fn([1, 2], [np.inf, 2])


def test_pearson_needs_two_points():
    with pytest.raises(ValueError):
        pearson([1], [1])


def test_evaluate_and_clip():
    out = evaluate([1, 2, 3], [1, 2, 4])
    assert set(out) == {"rmse", "pearson"}
    assert clip_predictions([-1, 2.5, 7]).tolist() == [0.0, 2.5, 5.0]


def test_folds_partition_all_rows_once():
    y = np.random.default_rng(0).choice([0, 1.5, 2, 3, 4, 5], 120)
    folds = make_folds(y, 5, 42)
    val = np.concatenate([v for _, v in folds])
    assert sorted(val) == list(range(120))
    for tr, va in folds:
        assert not set(tr) & set(va)


def test_cross_validate_clips_and_is_per_fold():
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=100)})
    y = np.clip(2.5 + X["a"].to_numpy() * 3, 0, 5)
    res = cross_validate(LinearRegression(), X, y, make_folds(y, 5, 0))
    assert len(res.fold_rmse) == 5 and res.oof.min() >= 0 and res.oof.max() <= 5
    assert res.pearson_mean > 0.8
