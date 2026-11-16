"""
File Name: test_utils_stats_extra.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.6
Relative Path: /tests/test_utils/test_utils_stats_extra.py
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

import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.stats

from InsurAutoML.utils import stats


@pytest.fixture
def clf_data():
    rng = np.random.default_rng(0)
    n = 120
    y = pd.Series(np.repeat([0, 1, 2], n // 3), name="y")
    X = pd.DataFrame(
        {
            "a": rng.normal(size=n) + y.values,
            "b": rng.normal(size=n),
        }
    )
    return X, y


def test_merge_helpers():
    a, b = np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0])
    full = np.concatenate([a, b])
    assert stats.merge_mean((3, a.mean()), (2, b.mean())) == pytest.approx(full.mean())
    assert stats.merge_std(
        (3, a.mean(), a.std()), (2, b.mean(), b.std())
    ) == pytest.approx(full.std())
    assert stats.merge_dict({"a": 1, "b": 2}, {"b": 3, "c": 4}) == {
        "a": 1,
        "b": 5,
        "c": 4,
    }


def test_nan_cov_variants():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 2))
    y = rng.normal(size=30)
    ref = np.cov(np.column_stack([X, y]).T)

    # y as ndarray / Series / DataFrame
    for _y in [y, pd.Series(y), pd.DataFrame({"y": y})]:
        cov = stats.nan_cov(X, _y)
        assert cov.shape == (2, 1)
        np.testing.assert_allclose(cov[:, 0], ref[:2, 2])

    # y all missing -> covariance of X itself
    for _y in [pd.Series([np.nan] * 30), pd.DataFrame({"y": [np.nan] * 30}),
               np.full(30, np.nan)]:
        np.testing.assert_allclose(stats.nan_cov(X, _y), ref[:2, :2])

    # 1d X
    np.testing.assert_allclose(stats.nan_cov(X[:, 0]), [[ref[0, 0]]])

    # axis = 1: covariance between rows
    np.testing.assert_allclose(stats.nan_cov(X.T, axis=1), ref[:2, :2])

    with pytest.raises(ValueError, match="same length of rows"):
        stats.nan_cov(X, y[:10])
    with pytest.raises(ValueError, match="same length of columns"):
        stats.nan_cov(X.T, X.T[:, :10], axis=1)


def test_class_means_and_cov(clf_data):
    X, y = clf_data
    means = stats.class_means(X, y)
    assert len(means) == 3
    for cl, m in zip([0, 1, 2], means):
        np.testing.assert_allclose(m, X[y == cl].mean().values)

    priors = np.array([1 / 3] * 3)
    cov = stats.class_cov(X.values, y.values, priors)
    expected = sum(
        p * np.cov(X[y == cl].values.T, bias=1) for p, cl in zip(priors, [0, 1, 2])
    )
    np.testing.assert_allclose(cov, expected)


def test_empirical_covariance():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 3))
    np.testing.assert_allclose(stats.empirical_covariance(X), np.cov(X.T, bias=1))
    np.testing.assert_allclose(
        stats.empirical_covariance(X, assume_centered=True), X.T @ X / 40
    )
    # single column -> 2d matrix
    single = stats.empirical_covariance(X[:, :1])
    assert single.shape == (1, 1)
    assert single[0, 0] == pytest.approx(X[:, 0].var())
    # 1d input is treated as a single sample
    with pytest.warns(UserWarning, match="Only one data sample"):
        out = stats.empirical_covariance(X[0])
    np.testing.assert_allclose(out, np.zeros((3, 3)))


def test_pearson_and_mi(clf_data):
    X, y = clf_data
    corr = stats.Pearson_Corr(X.values, y)
    for i, col in enumerate(X.columns):
        assert corr[i] == pytest.approx(np.corrcoef(X[col], y)[0, 1])

    Xd = pd.DataFrame({"d": np.tile([0, 1], 60), "e": y.values})
    mi = stats.MI(Xd, y)
    # ndarray features are converted to a dataframe
    assert stats.MI(Xd.values[:, :1], y.to_frame())[0] == pytest.approx(mi[0])
    # feature identical to y -> MI equals entropy of y
    assert mi[1] == pytest.approx(np.log(3))
    # balanced feature independent of y -> MI ~ 0
    assert mi[0] == pytest.approx(0, abs=1e-10)

    with pytest.raises(ValueError, match="not same size"):
        stats.MI(X, y[:10])


def test_t_score_and_anova(clf_data):
    X, y = clf_data
    two = y < 2
    X2, y2 = X[two], y[two]
    ref = scipy.stats.ttest_ind(X2.loc[y2 == 0, "a"], X2.loc[y2 == 1, "a"])

    f = stats.t_score(X2, y2)
    assert f[0] == pytest.approx(ref[0])
    p = stats.t_score(X2, y2.to_frame(), fvalue=False, pvalue=True)
    assert p[0] == pytest.approx(ref[1])
    f, p = stats.t_score(X2, y2, pvalue=True)
    assert len(f) == len(p) == 2

    with pytest.raises(ValueError, match="not same size"):
        stats.t_score(X, y[:10])
    with pytest.raises(ValueError, match="Only 2 group"):
        stats.t_score(X, y)

    ref = scipy.stats.f_oneway(*[X.loc[y == g, "a"] for g in [0, 1, 2]])
    assert stats.ANOVA(X, y)[0] == pytest.approx(ref[0])
    assert stats.ANOVA(X, y.to_frame(), fvalue=False, pvalue=True)[0] == pytest.approx(
        ref[1]
    )
    f, p = stats.ANOVA(X, y, pvalue=True)
    assert f[0] == pytest.approx(ref[0]) and p[0] == pytest.approx(ref[1])
    with pytest.raises(ValueError, match="not same size"):
        stats.ANOVA(X, y[:10])


def test_neg_metrics():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])

    assert stats.neg_R2(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) == -1
    # R2 below -1 -> inf
    assert stats.neg_R2(np.array([1.0, 2.0, 3.0]), np.array([10.0, -5.0, 30.0])) == np.inf
    assert stats.neg_accuracy(y_true, y_pred) == pytest.approx(-0.8)
    assert stats.neg_precision(y_true, y_pred) == pytest.approx(-1.0)
    assert stats.neg_f1(y_true, y_pred) == pytest.approx(-0.8)
    assert stats.neg_hinge(np.array([-1, 1]), np.array([-2.0, 2.0])) == 0

    score = np.array([0.1, 0.9, 0.8, 0.2, 0.7])
    assert stats.neg_auc(y_true, score) == -1
    assert stats.neg_auc(y_true, np.column_stack([1 - score, score])) == -1
    y3 = np.array([0, 1, 2, 0, 1, 2])
    prob3 = np.eye(3)[y3] * 0.8 + 0.2 / 3
    assert stats.neg_auc(y3, prob3) == pytest.approx(-1)


def test_rank_and_nn_helpers():
    values = np.array([3, 1, 1, 2])
    np.testing.assert_array_equal(stats._rankdata(values), [4, 1.5, 1.5, 3])
    r = stats._rankdata(values, ties_method="random", seed=0)
    assert sorted(r) == [1, 2, 3, 4] and r[0] == 4 and r[3] == 3
    np.testing.assert_array_equal(
        stats._rankdata(values, ties_method="random", seed=0), r
    )

    idx = np.array([5, 7, 9])
    np.testing.assert_array_equal(stats._NN(idx, "first"), [5, 5, 5])
    np.testing.assert_array_equal(stats._NN(idx, "last"), [9, 9, 9])
    np.testing.assert_array_equal(stats._NN(idx, "ordinal"), idx)
    np.random.seed(0)
    rnd = stats._NN(idx, "random")
    assert all(a != b for a, b in zip(rnd, idx))


def test_ccc():
    rng = np.random.default_rng(3)
    x = rng.normal(size=200)
    # perfectly dependent -> close to 1, independent -> close to 0
    assert stats.CCC(x, x**2, X_ties_method="ordinal", y_ties_method="ordinal") > 0.9
    noise = rng.normal(size=200)
    assert abs(stats.CCC(x, noise, fold=3, seed=0)) < 0.2
    # pandas inputs are accepted
    assert stats.CCC(pd.Series(x), pd.Series(x**2), X_ties_method="ordinal") > 0.9
    with pytest.raises(ValueError, match="same length"):
        stats._CCC(x, x[:10])


def test_accc():
    rng = np.random.default_rng(4)
    n = 150
    X = rng.normal(size=(n, 1))
    Z = rng.normal(size=(n, 1))
    # include repeated points to exercise the tie handling
    Z[:10] = Z[0]
    X[:10] = X[0]
    y = Z[:, 0] + 0.01 * rng.normal(size=n)

    t = stats.ACCC(Z, y, seed=0)
    assert t > 0.5
    t_fixed = stats.ACCC(
        Z[:, 0], y, Z_ties_method="first", X_ties_method="last", seed=0
    )
    assert t_fixed > 0.5
    q = stats.ACCC(Z, y, mode="Q", fold=2, seed=0)
    assert np.isfinite(q)

    # equidistant grid -> nearest neighbor ties broken at random
    grid = np.arange(n, dtype=float)
    assert stats.ACCC(grid, grid**2, seed=0) > 0.9

    # conditional on X: y depends on Z given X
    t_cond = stats.ACCC(Z, y, X=X[:, 0], seed=0)
    assert t_cond > 0.5
    # y depends on X only -> Z adds little information given X
    t_none = stats.ACCC(Z, X[:, 0] + 0.01 * rng.normal(size=n), X=X,
                        Z_ties_method="ordinal", X_ties_method="first", seed=0)
    assert t_none < t_cond

    # y constant -> S_n == 0 -> returns 1 for conditional measure
    y_const = np.r_[np.zeros(n)]
    assert stats._ACCC(Z, y_const, X=X, y_ties_method="min") == 1

    with pytest.raises(ValueError, match="Z and y must have the same length"):
        stats._ACCC(Z, y[:10])
    with pytest.raises(ValueError, match="X and y must have the same length"):
        stats._ACCC(Z, y, X=X[:10])
