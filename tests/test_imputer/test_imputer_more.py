"""
File Name: test_imputer_more.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.6
Relative Path: /tests/test_imputer/test_imputer_more.py
File Created: Monday, 16th November 2026 12:00:00 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 16th November 2026 12:00:00 pm
Modified By: Panyi Dong (panyid2@illinois.edu)
-----
MIT License

Copyright (c) 2022 - 2026, Panyi Dong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import pandas as pd
import pytest

from InsurAutoML.imputation import (
    DummyImputer,
    ExpectationMaximization,
    JointImputer,
    KNNImputer,
    MeanModeImputer,
    MICE,
    MissForestImputer,
    SimpleImputer,
)
from InsurAutoML.imputation.base import BaseImputer


def _numeric_data(n=100, p=4, seed=0, p_missing=0.1):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"x{i}" for i in range(p)])
    X["x1"] = X["x0"] + rng.normal(scale=0.2, size=n)
    mask = rng.random(X.shape) < p_missing
    mask[0, 0] = True  # at least one missing value
    return X, X.mask(mask), mask


def _assert_observed_kept(original, filled, mask):
    assert not filled.isnull().values.any()
    observed = ~mask
    np.testing.assert_allclose(
        filled[original.columns].values.astype(float)[observed],
        original.values.astype(float)[observed],
    )


# ---------------------------------------------------------------------------
# base.py
# ---------------------------------------------------------------------------


def test_base_imputer_not_implemented():
    with pytest.raises(NotImplementedError):
        BaseImputer().fill(pd.DataFrame({"a": [1.0]}))


@pytest.mark.parametrize(
    "method, expected",
    [
        ("mean", 2.0),
        ("zero", 0.0),
        ("median", 1.0),
        ("most frequent", 1.0),
        (-9, -9.0),
    ],
)
def test_simple_imputer_methods(method, expected):
    X = pd.DataFrame({"a": [1.0, 1.0, 4.0, np.nan], "b": [1.0, 2.0, 3.0, 4.0]})
    imp = SimpleImputer(method=method)
    filled = imp.fill(X)

    assert imp._fitted
    assert filled.loc[3, "a"] == pytest.approx(expected)
    # observed values and non-missing columns untouched
    pd.testing.assert_series_equal(filled["b"], X["b"])
    assert filled.loc[:2, "a"].tolist() == [1.0, 1.0, 4.0]
    # input is not modified in place
    assert np.isnan(X.loc[3, "a"])


def test_simple_imputer_no_missing():
    X = pd.DataFrame({"a": [1.0, 2.0]})
    filled = SimpleImputer().fill(X)
    pd.testing.assert_frame_equal(filled, X)


def test_mean_mode_imputer_mixed_types():
    X = pd.DataFrame(
        {
            "num": [1.0, 3.0, np.nan, 2.0],
            "cat": ["u", "v", "u", None],
            "empty_cat": pd.Series([None, None, None, None], dtype=object),
        }
    )
    imp = MeanModeImputer()
    filled = imp.fill(X)

    assert imp._fitted
    assert filled.loc[2, "num"] == pytest.approx(2.0)
    assert filled.loc[3, "cat"] == "u"
    assert filled[["num", "cat"]].notnull().all().all()
    # all-missing categorical column has no mode, so it stays missing
    assert filled["empty_cat"].isnull().all()
    # original untouched
    assert np.isnan(X.loc[2, "num"])


def test_dummy_imputer_creates_indicator_when_response_differs():
    X = pd.DataFrame(
        {"a": [1.0, np.nan, 3.0, np.nan, 5.0, 6.0], "b": [1.0, 2, 3, 4, 5, 6]}
    )
    # response is much larger where "a" is missing
    y = pd.Series([1.0, 10.0, 1.0, 10.0, 1.0, 1.0])
    imp = DummyImputer(threshold=0.1)
    filled = imp.fill(X, y)

    assert imp._fitted
    assert "a_nan" in filled.columns
    assert filled["a_nan"].tolist() == [0, 1, 0, 1, 0, 0]
    assert filled.loc[[1, 3], "a"].tolist() == [0.0, 0.0]
    assert not filled.isnull().values.any()


@pytest.mark.parametrize(
    "method, expected",
    [("mean", 3.5), ("zero", 0.0), ("median", 3.5), ("most frequent", 2.0), (-1, -1.0)],
)
def test_dummy_imputer_fill_methods_below_threshold(method, expected):
    X = pd.DataFrame({"a": [2.0, 2.0, np.nan, 5.0, 5.0], "b": [1.0, 2, 3, 4, 5]})
    y = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0])
    filled = DummyImputer(threshold=0.5, method=method).fill(X, y)

    assert "a_nan" not in filled.columns
    assert filled.loc[2, "a"] == pytest.approx(expected)


def test_dummy_imputer_no_missing():
    X = pd.DataFrame({"a": [1.0, 2.0]})
    imp = DummyImputer()
    pd.testing.assert_frame_equal(imp.fill(X, pd.Series([0, 1])), X)
    assert imp._fitted


def test_joint_imputer():
    np.random.seed(0)
    X, Xm, mask = _numeric_data(n=80, p=3, p_missing=0.05)
    imp = JointImputer()
    filled = imp.fill(Xm)

    assert imp._fitted
    _assert_observed_kept(X, filled, mask)
    # no missing -> unchanged
    pd.testing.assert_frame_equal(JointImputer().fill(X), X)


# ---------------------------------------------------------------------------
# multiple.py: ExpectationMaximization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("iterations", [5, 30])
def test_expectation_maximization(iterations):
    X, Xm, mask = _numeric_data()
    imp = ExpectationMaximization(iterations=iterations, threshold=0.05, seed=1)
    filled = imp.fill(Xm)

    assert imp._fitted
    _assert_observed_kept(X, filled, mask)
    assert set(imp._missing_table.columns) == set(Xm.columns[Xm.isnull().any()])
    # imputed values stay in a plausible range of the column
    assert np.abs(filled.values).max() < 10

    pd.testing.assert_frame_equal(ExpectationMaximization().fill(X), X)


# ---------------------------------------------------------------------------
# multiple.py: KNNImputer
# ---------------------------------------------------------------------------


def _knn_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 4)), columns=list("abcd"))
    X["b"] = X["a"] * 2
    Xm = X.copy()
    # different number of missing values in each column
    Xm.loc[[3, 7, 20, 50], "a"] = np.nan
    Xm.loc[[5, 9], "c"] = np.nan
    Xm.loc[[11], "b"] = np.nan
    return X, Xm


def test_knn_imputer_fixed_k_multiple_columns():
    X, Xm = _knn_data()
    imp = KNNImputer(n_neighbors=3, seed=1)
    filled = imp.fill(Xm)

    assert imp._fitted
    _assert_observed_kept(X, filled, Xm.isnull().values)
    assert sorted(imp._missing_feature) == ["a", "b", "c"]
    assert list(imp._missing_table["a"]) == [3, 7, 20, 50]
    assert list(imp._missing_table["c"]) == [5, 9]
    # b = 2 * a is recoverable from its neighbours, much better than the mean
    err = np.abs(filled.loc[[3, 7, 20, 50], "a"] - X.loc[[3, 7, 20, 50], "a"])
    assert err.mean() < np.abs(X["a"] - X["a"].mean()).mean()


def test_knn_imputer_cross_validated_k():
    X, Xm = _knn_data()
    imp = KNNImputer(n_neighbors=None, fold=5, seed=1)
    filled = imp.fill(Xm)

    assert imp._fitted
    assert isinstance(imp.optimial_k, int)
    assert 1 <= imp.optimial_k <= 10
    _assert_observed_kept(X, filled, Xm.isnull().values)
    # input not modified, all rows retained
    assert Xm.isnull().sum().sum() == 7
    assert filled.shape == X.shape


def test_knn_imputer_categorical_and_no_missing():
    X = pd.DataFrame({"a": [0, 1] * 50, "b": np.arange(100.0)})
    with pytest.raises(ValueError, match="categorical"):
        KNNImputer(n_neighbors=2).fill(X)

    X = pd.DataFrame({"a": np.arange(100.0), "b": np.arange(100.0) ** 2})
    imp = KNNImputer(n_neighbors=2)
    with pytest.warns(UserWarning, match="No nan values"):
        filled = imp.fill(X)
    pd.testing.assert_frame_equal(filled, X)
    assert imp._fitted


@pytest.mark.parametrize("n_neighbors, n_expected", [(None, 10), (2, 1), ([1, 3], 2)])
def test_knn_imputer_cross_validation(n_neighbors, n_expected):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, 3)), columns=list("abc"))
    X["b"] = X["a"]
    imp = KNNImputer(n_neighbors=n_neighbors)
    err = imp._cross_validation_knn(X.iloc[15:], X.iloc[:15], ["b"])

    assert len(err) == n_expected
    for e in err:
        # only the regressed feature can have an error
        assert e["a"] == 0 and e["c"] == 0
        assert e["b"] >= 0


# ---------------------------------------------------------------------------
# multiple.py: MissForestImputer
# ---------------------------------------------------------------------------


def _mixed_data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "x0": x0,
            "x1": 3 * x0 + rng.normal(scale=0.1, size=n),
            "k": (x0 > 0).astype(float),
        }
    )
    mask = np.zeros(X.shape, dtype=bool)
    mask[rng.choice(n, 10, replace=False), 1] = True
    mask[rng.choice(n, 6, replace=False), 2] = True
    return X, X.mask(mask), mask


def test_missforest_fill_and_refill():
    X, Xm, mask = _mixed_data()
    imp = MissForestImputer(
        max_iter=3, n_estimators=20, random_state=0, categorical_threshold=10
    )
    with pytest.raises(RuntimeError):
        imp.refill(Xm)

    filled = imp.fill(Xm)

    assert imp._fitted
    _assert_observed_kept(X, filled, mask)
    assert imp.feature_types_["numerical"] == ["x0", "x1"]
    assert imp.feature_types_["categorical"] == ["k"]
    assert len(imp.models_) > 0
    # dtypes are restored
    assert (filled.dtypes == Xm.dtypes).all()
    # categorical column only takes observed classes
    assert set(filled["k"].unique()) <= {0.0, 1.0}
    # x1 = 3 * x0 is learnable by the forest
    miss1 = mask[:, 1]
    err = np.abs(filled.loc[miss1, "x1"] - X.loc[miss1, "x1"]).mean()
    assert err < np.abs(X["x1"] - X["x1"].mean()).mean()

    refilled = imp.refill(Xm)
    _assert_observed_kept(X, refilled, mask)

    # a column without missing values at fit time has no model: refill only
    # falls back to the initial (mean) imputation for it
    Xnew = Xm.copy()
    Xnew.loc[[0, 1], "x0"] = np.nan
    refilled = imp.refill(Xnew)
    assert not refilled.isnull().values.any()
    assert refilled.loc[0, "x0"] == pytest.approx(Xnew["x0"].mean())


def test_missforest_string_column_and_all_missing_column():
    X, Xm, mask = _mixed_data()
    rng = np.random.default_rng(1)
    Xm["s"] = pd.Series(rng.choice(["u", "v"], len(Xm))).mask(rng.random(len(Xm)) < 0.1)
    Xm["allnan"] = np.nan

    imp = MissForestImputer(
        max_iter=2, n_estimators=10, random_state=0, categorical_threshold=10
    )
    with pytest.warns(UserWarning, match="Error imputing"):
        filled = imp.fill(Xm)

    assert "s" in imp.feature_types_["categorical"]
    assert (
        "allnan"
        not in imp.feature_types_["numerical"] + imp.feature_types_["categorical"]
    )
    assert filled["s"].notnull().all()
    assert set(filled["s"]) <= {"u", "v"}
    assert filled[["x0", "x1", "k"]].notnull().all().all()


def test_missforest_no_missing():
    X, _, _ = _mixed_data()
    imp = MissForestImputer()
    with pytest.warns(UserWarning, match="No missing values"):
        out = imp.fill(X)
    assert out is X
    assert imp._fitted


def test_missforest_calculate_delta():
    imp = MissForestImputer()
    imp.feature_types_ = {"numerical": ["n"], "categorical": ["c"]}
    old = pd.DataFrame({"n": [1.0, 2.0, 3.0], "c": ["a", "b", "a"]})
    new = pd.DataFrame({"n": [1.0, 2.0, 4.0], "c": ["a", "a", "a"]})
    delta = imp._calculate_delta(new, old, ["n", "c"])
    expected = 1.0 / new["n"].var() + 1 / 3
    assert delta == pytest.approx(expected)
    assert imp._calculate_delta(old, old, ["n", "c"]) == 0


# ---------------------------------------------------------------------------
# multiple.py: MICE
# ---------------------------------------------------------------------------


def test_mice_numeric_and_binary():
    X, Xm, mask = _numeric_data(n=100, p=3)
    rng = np.random.default_rng(3)
    X["bin"] = (X["x0"] > 0).astype(float)
    Xm["bin"] = X["bin"]
    Xm.loc[[3, 10], "bin"] = np.nan
    mask = Xm.isnull().values

    imp = MICE(cycle=2, seed=1)
    filled = imp.fill(Xm)

    assert imp._fitted
    _assert_observed_kept(X, filled, mask)
    assert set(imp._missing_feature) == set(Xm.columns[Xm.isnull().any()])
    # logistic regression used for the binary column -> imputed classes are 0/1
    assert set(filled["bin"].unique()) <= {0.0, 1.0}
    assert set(filled.loc[[3, 10], "bin"]) <= {0.0, 1.0}


def test_mice_wide_data_uses_lasso():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, 16)), columns=[f"c{i}" for i in range(16)])
    mask = rng.random(X.shape) < 0.05
    mask[0, 0] = True
    filled = MICE(cycle=1, seed=1).fill(X.mask(mask))
    _assert_observed_kept(X, filled, mask)


def test_mice_errors_and_no_missing():
    X, Xm, _ = _numeric_data(n=50, p=3)
    Xs = Xm.copy()
    Xs["s"] = ["u", "v"] * 25
    with pytest.raises(ValueError, match="numerical"):
        MICE(cycle=1).fill(Xs)

    Xc = Xm.copy()
    Xc["s"] = pd.Categorical(["u", "v"] * 25)
    with pytest.raises(ValueError, match="numerical"):
        MICE(cycle=1).fill(Xc)

    with pytest.warns(UserWarning, match="No nan values"):
        out = MICE(cycle=1).fill(X)
    pd.testing.assert_frame_equal(out, X)
