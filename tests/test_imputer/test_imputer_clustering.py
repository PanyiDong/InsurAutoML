"""
File Name: test_imputer_clustering.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.6
Relative Path: /tests/test_imputer/test_imputer_clustering.py
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

import InsurAutoML.imputation.clustering as clustering
from InsurAutoML.imputation.clustering import AAI_kNN, CMI, KMI, k_Prototype_NN


class _SerialPool:
    """In-process replacement of multiprocessing.Pool (keeps coverage tracing)."""

    def __init__(self, processes=None, *args, **kwargs):
        self.processes = processes

    def map(self, func, iterable):
        return list(map(func, iterable))

    def close(self):
        pass

    def join(self):
        pass


@pytest.fixture
def serial_pool(monkeypatch):
    monkeypatch.setattr(clustering, "Pool", _SerialPool)


def _numeric_data(n=40, seed=0, p_missing=0.1):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.normal(size=n),
        }
    )
    X["b"] = 2 * X["a"] + rng.normal(scale=0.1, size=n)
    mask = rng.random(X.shape) < p_missing
    # make sure every column has at least one missing value
    mask[0, :] = [True, False, False]
    mask[1, :] = [False, True, False]
    mask[2, :] = [False, False, True]
    return X, X.mask(mask), mask


def _mixed_data(n=40, seed=0, p_missing=0.1):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "cat": rng.choice(["x", "y", "z"], size=n),
        }
    )
    mask = rng.random(X.shape) < p_missing
    mask[0, :] = [True, False, False]
    mask[1, :] = [False, False, True]
    return X, X.mask(mask), mask


def _assert_observed_kept(original, filled, mask):
    assert not filled.isnull().values.any()
    assert list(filled.columns) == list(original.columns)
    for j, col in enumerate(original.columns):
        observed = ~mask[:, j]
        if pd.api.types.is_numeric_dtype(original[col]):
            np.testing.assert_allclose(
                filled[col].values[observed].astype(float),
                original[col].values[observed].astype(float),
            )
        else:
            assert (filled[col].values[observed] == original[col].values[observed]).all()
            # imputed categories must come from the observed categories
            assert set(filled[col]) <= set(original[col])


# ---------------------------------------------------------------------------
# AAI_kNN
# ---------------------------------------------------------------------------


def test_aai_similarity_functions():
    imp = AAI_kNN()
    x = np.array([1.0, 2.0, 3.0])

    assert imp.Pearson_Correlation_Coefficient(x, 2 * x) == pytest.approx(1.0)
    assert imp.Pearson_Correlation_Coefficient(x, -x) == pytest.approx(-1.0)
    # constant vector gives zero denominator -> defined as 1
    assert imp.Pearson_Correlation_Coefficient(x, np.ones(3)) == 1

    assert imp.Cosine_based_similarity(x, 3 * x) == pytest.approx(1.0)
    assert imp.Cosine_based_similarity(
        np.array([1.0, 0.0]), np.array([0.0, 1.0])
    ) == pytest.approx(0.0)
    assert imp.Cosine_based_similarity(np.zeros(3), x) == 1


def test_aai_get_k_neighbors():
    train = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [0.0, 1.0, 2.0, 3.0]})
    test = pd.Series({"a": 2.0, "b": 2.0})
    for similarity in ["PCC", "COS"]:
        imp = AAI_kNN(k=2, similarity=similarity)
        values, index, sim = imp._get_k_neighbors(test, train, "a")
        assert len(values) == len(index) == len(sim) == 2
        assert set(index) <= set(train.index)
        # returned similarities are sorted ascending (largest last)
        assert sim[0] <= sim[1]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(k=3, similarity="PCC", AutAI=True, AutAI_tmp=True),
        dict(k=3, similarity="COS", AutAI=True, AutAI_tmp=False, scaling=False),
        dict(k=3, similarity="PCC", AutAI=False),
    ],
)
def test_aai_knn_fill_numeric(serial_pool, kwargs):
    X, Xm, mask = _numeric_data()
    imp = AAI_kNN(threads=2, **kwargs)
    filled = imp.fill(Xm.copy())

    _assert_observed_kept(X, filled, mask)
    assert imp._fitted
    assert not imp.train.empty
    assert not imp.train.isnull().values.any()

    # second call reuses the stored (fitted) train set
    train_before = imp.train.copy()
    refilled = imp.fill(Xm.copy())
    _assert_observed_kept(X, refilled, mask)
    pd.testing.assert_frame_equal(imp.train, train_before)


def test_aai_knn_fill_mixed(serial_pool):
    X, Xm, mask = _mixed_data()
    filled = AAI_kNN(k=3, threads=2).fill(Xm.copy())
    _assert_observed_kept(X, filled, mask)


def test_aai_knn_ndarray_input_and_no_missing(serial_pool):
    X, Xm, mask = _numeric_data()
    filled = AAI_kNN(k=3, threads=1).fill(Xm.values)
    assert isinstance(filled, pd.DataFrame)
    assert not filled.isnull().values.any()

    with pytest.warns(UserWarning, match="No missing values"):
        out = AAI_kNN(threads=1).fill(X)
    pd.testing.assert_frame_equal(out, X)


def test_aai_knn_invalid_input():
    with pytest.raises(TypeError):
        AAI_kNN(threads=1).fill(5)


def test_aai_knn_real_pool():
    # smoke test with the real multiprocessing pool
    X, Xm, mask = _numeric_data(n=20)
    filled = AAI_kNN(k=3, threads=2, AutAI=False).fill(Xm.copy())
    _assert_observed_kept(X, filled, mask)


# ---------------------------------------------------------------------------
# KMI
# ---------------------------------------------------------------------------


def test_kmi_not_implemented():
    with pytest.raises(NotImplementedError):
        KMI()


# ---------------------------------------------------------------------------
# CMI
# ---------------------------------------------------------------------------


def test_cmi_helpers():
    imp = CMI(k=2, seed=1)
    imp.k_means = [np.array([0.0, 0.0]), np.array([3.0, 4.0])]
    row = np.array([0.0, 0.0])
    imp.distance = "l2"
    assert imp._distance(row, 1) == pytest.approx(5.0)
    imp.distance = "l1"
    assert imp._distance(row, 1) == pytest.approx(7.0)

    imp.bandwidth = 1.0
    assert imp._kernel(row, row) == pytest.approx(1 / np.sqrt(2 * np.pi) ** 2)
    assert imp._kernel(row, row + 5) < imp._kernel(row, row + 1)

    data = pd.DataFrame({"a": [0.0, 1.0, 10.0], "b": [0.0, 1.0, 10.0]})
    imp.group_assign = np.array([0, 0, 1])
    np.testing.assert_allclose(imp._get_k_means(data, 0), [0.5, 0.5])
    # empty group gives a None array
    assert imp._get_k_means(data, 5)[0] is None

    imp.distance = "l2"
    imp.k_means = [np.array([0.5, 0.5]), np.array([10.0, 10.0])]
    assert imp._get_group_assign(data, [0, 1, 2]) == [0, 0, 1]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(k=3, distance="l2", scaling=True),
        dict(k=3, distance="l1", scaling=False),
    ],
)
def test_cmi_fill_numeric(serial_pool, kwargs):
    X, Xm, mask = _numeric_data()
    imp = CMI(seed=1, threads=2, **kwargs)
    filled = imp.fill(Xm.copy())

    _assert_observed_kept(X, filled, mask)
    assert imp._fitted
    assert not imp.train.empty
    assert len(imp.group_assign) == len(X)
    assert set(np.unique(imp.group_assign)) <= set(range(imp.k))

    # second call uses the trained k_means/train set
    refilled = imp.fill(Xm.copy())
    _assert_observed_kept(X, refilled, mask)
    assert len(imp.group_assign_train) == len(X)


def test_cmi_fill_mixed(serial_pool):
    X, Xm, mask = _mixed_data()
    filled = CMI(k=2, seed=1, threads=2).fill(Xm.copy())
    _assert_observed_kept(X, filled, mask)


def test_cmi_empty_cluster_warning(serial_pool):
    # many more clusters than distinct points -> empty clusters are removed
    X, Xm, mask = _numeric_data(n=12)
    imp = CMI(k=12, seed=1, threads=1)
    with pytest.warns(UserWarning, match="Empty cluster"):
        filled = imp.fill(Xm.copy())
    assert imp.k < 12
    assert not filled.isnull().values.any()


def test_cmi_no_missing_and_invalid_input():
    X, _, _ = _numeric_data()
    with pytest.warns(UserWarning, match="No missing values"):
        out = CMI(threads=1).fill(X)
    pd.testing.assert_frame_equal(out, X)

    with pytest.raises(TypeError):
        CMI(threads=1).fill(5)


def test_cmi_real_pool():
    X, Xm, mask = _numeric_data(n=20)
    filled = CMI(k=2, seed=1, threads=2).fill(Xm.copy())
    _assert_observed_kept(X, filled, mask)


# ---------------------------------------------------------------------------
# k_Prototype_NN
# ---------------------------------------------------------------------------


def test_k_prototype_helpers():
    imp = k_Prototype_NN(k=2, seed=1)
    centroids = pd.DataFrame({"a": [0.0, 3.0], "b": [0.0, 4.0]})
    row = pd.Series({"a": 0.0, "b": 0.0})
    imp.distance = "l2"
    np.testing.assert_allclose(imp._distance(row, centroids), [0.0, 5.0])
    imp.distance = "l1"
    np.testing.assert_allclose(imp._distance(row, centroids), [0.0, 7.0])

    cat_centroids = pd.DataFrame({"c": [0, 1], "d": [1, 1]})
    cat_row = pd.Series({"c": 0, "d": 0})
    imp.dissimilarity = "simple"
    np.testing.assert_array_equal(imp._dissimilarity(cat_row, cat_centroids), [1, 2])

    imp.dissimilarity = "weighted"
    imp.categorical_table = {"c": {0: 3, 1: 2}, "d": {0: 1, 1: 4}}
    weighted = imp._dissimilarity(cat_row, cat_centroids)
    assert weighted.shape == (2,)
    assert np.all(weighted >= 0)

    imp._k_modes_clustering(pd.DataFrame({"a": [1.0]}))
    assert imp.k == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(distance="l2", dissimilarity="weighted", scaling=True),
        dict(distance="l1", dissimilarity="simple", scaling=False),
    ],
)
def test_k_prototype_fill_mixed(serial_pool, kwargs):
    X, Xm, mask = _mixed_data()
    imp = k_Prototype_NN(k=3, seed=1, threads=2, **kwargs)
    filled = imp.fill(Xm.copy())

    _assert_observed_kept(X, filled, mask)
    assert imp._fitted
    assert len(imp.models) == imp.k
    assert "cat" in imp.categorical_table

    # refill with fitted centroids/models
    refilled = imp.fill(Xm.copy())
    _assert_observed_kept(X, refilled, mask)


@pytest.mark.parametrize("dissimilarity", ["weighted", "simple"])
def test_k_prototype_fill_all_numeric(serial_pool, dissimilarity):
    # no categorical columns at all
    X, Xm, mask = _numeric_data()
    imp = k_Prototype_NN(k=3, seed=1, threads=2, dissimilarity=dissimilarity)
    filled = imp.fill(Xm.copy())

    _assert_observed_kept(X, filled, mask)
    assert imp.categorical_table == {}
    assert len(imp.models) == imp.k
    refilled = imp.fill(Xm.copy())
    _assert_observed_kept(X, refilled, mask)


def test_k_prototype_no_missing_and_invalid_input():
    X, _, _ = _mixed_data()
    with pytest.warns(UserWarning, match="No missing values"):
        out = k_Prototype_NN(threads=1).fill(X)
    pd.testing.assert_frame_equal(out, X)

    with pytest.raises(TypeError):
        k_Prototype_NN(threads=1).fill(5)


def test_k_prototype_real_pool():
    X, Xm, mask = _mixed_data(n=20)
    filled = k_Prototype_NN(k=2, seed=1, threads=2).fill(Xm.copy())
    _assert_observed_kept(X, filled, mask)
