"""
File Name: test_feature_selection_more.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.6
Relative Path: /tests/test_feature_selection/test_feature_selection_more.py
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

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

import InsurAutoML.feature_selection.filter as fs_filter
from InsurAutoML.feature_selection import (
    ASFFS,
    CBFS,
    CCCFilter,
    FOCI,
    FeatureFilter,
    GeneticAlgorithm,
    SFS,
    mRMR,
)
from InsurAutoML.feature_selection.wrapper import ExhaustiveFS
from InsurAutoML.feature_selection import sklearn as fs_sk


def _regression_data(n=150, p=4, seed=0):
    """y depends (almost) only on x0; other features are pure noise."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"x{i}" for i in range(p)])
    y = pd.Series(5 * X["x0"] + rng.normal(scale=0.1, size=n), name="y")
    return X, y


def _discrete_data(n=150, p=3, seed=0):
    """Discrete features; y is a copy of x0 (needed by the discrete MI)."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.integers(0, 4, size=(n, p)), columns=[f"x{i}" for i in range(p)]
    )
    return X, X["x0"].rename("y")


def _classification_data(n=150, p=4, seed=0, n_classes=2):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"x{i}" for i in range(p)])
    if n_classes == 2:
        y = (X["x0"] > 0).astype(int)
    else:
        y = pd.Series(np.digitize(X["x0"], [-0.5, 0.5]), name="y")
    return X, y.rename("y")


# ---------------------------------------------------------------------------
# filter.py
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("criteria", ["Pearson", "MI", "CCC"])
def test_feature_filter_selects_informative_feature(criteria):
    X, y = _discrete_data(p=4) if criteria == "MI" else _regression_data()
    fs = FeatureFilter(criteria=criteria, n_components=1)
    out = fs.fit_transform(X, y)

    assert fs._fitted
    assert list(out.columns) == ["x0"]
    assert fs.feature_names_in_ == list(X.columns)


def test_feature_filter_n_prop_default_and_ndarray():
    X, y = _regression_data()
    fs = FeatureFilter(n_prop=0.5).fit(X, y)
    assert fs.n_components == 2
    assert "x0" in fs.select_features

    fs = FeatureFilter().fit(X.values, y.values)
    assert fs.n_components == X.shape[1]
    out = fs.transform(X.values)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == X.shape
    assert fs.feature_names_in_ == [f"feature_{i}" for i in range(X.shape[1])]

    # n_components larger than number of features is clipped
    assert FeatureFilter(n_components=100).fit(X, y).n_components == X.shape[1]


@pytest.mark.parametrize(
    "y",
    [
        None,
        pd.Series([np.nan] * 150),
        pd.DataFrame({"y": [np.nan] * 150}),
        np.full(150, np.nan),
    ],
)
def test_feature_filter_requires_response(y):
    X, _ = _regression_data()
    with pytest.raises(ValueError, match="Must have response"):
        FeatureFilter().fit(X, y)


def test_feature_filter_invalid_inputs():
    X, y = _regression_data()
    with pytest.raises(ValueError, match="same number"):
        FeatureFilter().fit(X, y.iloc[:10])
    with pytest.raises(ValueError, match="Unsupported criteria"):
        FeatureFilter(criteria="unknown").fit(X, y)


def test_mrmr_selects_informative_feature_first():
    X, y = _discrete_data(p=3)
    fs = mRMR(n_components=2).fit(X, y)
    assert fs._fitted
    assert fs.select_features[0] == 0
    assert len(fs.select_features) == 2
    out = fs.transform(X)
    assert list(out.columns)[0] == "x0"

    assert len(mRMR(n_prop=0.4).fit(X, y).select_features) == 1
    # ndarray inputs: response name must not collide with feature 0
    fs = mRMR(n_components=2).fit(X.values, y.values)
    assert fs.select_features[0] == 0
    assert fs.transform(X).shape == (len(X), 2)
    # response named like a feature
    fs = mRMR(n_components=1).fit(X, y.rename("x1"))
    assert fs.select_features == [0]
    assert len(mRMR().fit(X, y).select_features) == 3


def test_ccc_filter():
    X, y = _regression_data()
    X["const"] = 1.0
    fs = CCCFilter(threshold=0.3)
    with pytest.raises(ValueError, match="fitted"):
        fs.transform(X)

    fs.fit(X, y)
    assert fs._fitted
    assert fs.selected_features_ == ["x0"]
    assert fs.ccc_scores_.index[0] == "x0"
    assert list(fs.transform(X).columns) == ["x0"]


def test_ccc_filter_failing_score(monkeypatch):
    X, y = _regression_data()

    def _broken(*args, **kwargs):
        raise RuntimeError("fail")

    monkeypatch.setattr(fs_filter, "CCC", _broken)
    fs = CCCFilter(threshold=0.0).fit(X, y)
    # every failing score falls back to 0
    assert (fs.ccc_scores_ == 0).all()
    assert len(fs.selected_features_) == X.shape[1]


@pytest.fixture
def thread_executor(monkeypatch):
    # run FOCI's parallel map in threads so coverage is traced
    monkeypatch.setattr(fs_filter, "ProcessPoolExecutor", ThreadPoolExecutor)


@pytest.mark.parametrize(
    "standardize, conditional",
    [(None, False), ("standardize", False), ("minmax", True)],
)
def test_foci(thread_executor, standardize, conditional):
    X, y = _regression_data(n=120, p=3)
    fs = FOCI(standardize=standardize, conditional=conditional)
    fs.fit(X, y)

    assert fs._fitted
    assert fs.select_features[0] == "x0"
    out = fs.transform(X)
    assert list(out.columns) == fs.select_features


def test_foci_ndarray_and_static_helpers(thread_executor):
    X, y = _regression_data(n=120, p=3)
    fs = FOCI(num_features=1).fit(X.values, y.values)
    assert fs.select_features == [0]
    assert fs.transform(X.values).shape == (120, 1)

    Xs, ys = X, pd.DataFrame(y)
    single = FOCI.compute_accc("x0", Xs, ys, [])
    noise = FOCI.compute_accc("x1", Xs, ys, [])
    joint = FOCI.compute_accc("x1", Xs, ys, ["x0"])
    cond = FOCI.compute_conditional_accc("x1", Xs, ys, ["x0"])
    cond0 = FOCI.compute_conditional_accc("x0", Xs, ys, [])
    assert single > 5 * noise
    assert cond0 == pytest.approx(single)
    # given x0, the noise feature adds (almost) nothing
    assert joint <= single + 0.01
    assert cond < 0.05


def test_foci_keeps_top_feature_when_nothing_selected(thread_executor, monkeypatch):
    X, y = _regression_data(n=60, p=3)
    monkeypatch.setattr(
        FOCI,
        "compute_accc",
        staticmethod(lambda feature, X, y, selected_features: -1.0),
    )
    with pytest.warns(UserWarning, match="selected no feature"):
        fs = FOCI().fit(X, y)
    assert len(fs.select_features) == 1


# ---------------------------------------------------------------------------
# wrapper.py
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("estimator", ["Lasso", "Ridge"])
def test_exhaustive_fs_regression(estimator):
    X, y = _regression_data(p=3)
    fs = ExhaustiveFS(estimator=estimator, criteria="MSE").fit(X, y)
    assert fs._fitted
    assert 0 in fs.selected_features
    assert fs.transform(X).shape[1] == len(fs.selected_features)


def test_exhaustive_fs_custom_estimator_and_metrics():
    from sklearn.linear_model import LogisticRegression

    X, y = _classification_data(p=3)
    fs = ExhaustiveFS(estimator=LogisticRegression()).fit(X, y)
    assert 0 in fs.selected_features

    # callable criteria
    from sklearn.metrics import mean_absolute_error

    Xr, yr = _regression_data(p=3)
    fs = ExhaustiveFS(estimator="Ridge", criteria=mean_absolute_error).fit(Xr, yr)
    assert 0 in fs.selected_features


def test_exhaustive_fs_invalid_arguments():
    X, y = _regression_data(p=2)
    with pytest.raises(ValueError, match="fit/predict"):
        ExhaustiveFS(estimator=object()).fit(X, y)
    with pytest.raises(ValueError, match="criteria"):
        ExhaustiveFS(estimator="Ridge", criteria="unknown").fit(X, y)


def test_sfs_regression_and_classification():
    X, y = _regression_data(p=4)
    fs = SFS(estimator="Lasso", n_components=2, criteria="MSE").fit(X, y)
    assert fs._fitted
    assert fs.selected_features[0] == 0
    assert 1 <= len(fs.selected_features) <= 2
    assert fs.transform(X).shape[1] == len(fs.selected_features)

    fs = SFS(estimator="Ridge", n_prop=0.5, criteria="MSE").fit(X.values, y.values)
    assert fs.n_components == 2

    Xc, yc = _classification_data(p=3)
    fs = SFS(estimator="LogisticRegression", criteria="neg_accuracy").fit(Xc, yc)
    assert fs.selected_features[0] == 0
    # adding noise features cannot beat the perfect single feature by much,
    # the forward search stops once loss increases
    assert len(fs.selected_features) <= 3


@pytest.mark.parametrize(
    "y",
    [
        None,
        pd.Series([np.nan] * 150),
        pd.DataFrame({"y": [np.nan] * 150}),
        np.full(150, np.nan),
    ],
)
def test_sfs_requires_response(y):
    X, _ = _regression_data()
    with pytest.raises(ValueError, match="Must have response"):
        SFS().fit(X, y)


def test_asffs_generalization_limit():
    fs = ASFFS(r_max=3)
    assert fs.generalization_limit(k=5, d=5, b=2) == 3
    assert fs.generalization_limit(k=2, d=5, b=2) == 2
    assert fs.generalization_limit(k=0, d=10, b=2) == 1


@pytest.mark.parametrize(
    "model, objective", [("Linear", "MSE"), ("Lasso", "MAE"), ("Ridge", "MSE")]
)
def test_asffs_objectives(model, objective):
    X, y = _regression_data(p=4)
    fs = ASFFS(n_components=2, model=model, objective=objective)
    feat, obj = fs._Forward_Objective([], list(X.columns), 1, X, y)
    assert feat == ("x0",)
    assert obj > 0

    feat, obj = fs._Backward_Objective(["x0", "x2"], 1, X, y)
    # removing the noise feature is the best backward step
    assert feat == ("x2",)


def test_asffs_floating_backward_step():
    # forward search greedily picks the proxy x0 first, but {x1, x2} explains
    # y almost perfectly -> the backward (floating) step must drop x0 again
    rng = np.random.default_rng(0)
    n = 200
    x1, x2, x3 = rng.normal(size=(3, n))
    X = pd.DataFrame(
        {"x0": x1 + x2 + rng.normal(scale=0.5, size=n), "x1": x1, "x2": x2, "x3": x3}
    )
    y = pd.Series(x1 + x2 + rng.normal(scale=0.05, size=n))

    fs = ASFFS(n_components=3, Delta=1, r_max=2).fit(X, y)
    assert fs._subset_max[1] == ["x0"]
    assert sorted(fs._subset_max[2]) == ["x1", "x2"]
    assert {"x1", "x2"} <= set(fs.selected_)


def test_asffs_explicit_b():
    X, y = _regression_data(p=4)
    fs = ASFFS(n_components=2, b=1, r_max=2).fit(X, y)
    assert fs._fitted
    assert "x0" in fs.selected_
    assert len(fs.selected_) >= 2


def test_asffs_fit_and_invalid_model():
    X, y = _regression_data(p=4)
    fs = ASFFS(n_components=2).fit(X, y)
    assert fs._fitted
    assert "x0" in fs.selected_
    assert fs.transform(X).shape[1] == len(fs.selected_)

    fs = ASFFS().fit(X.values, y.values)  # default n_components
    assert 0 in fs.selected_
    assert fs.transform(X.values).shape == (len(X), len(fs.selected_))

    with pytest.raises(ValueError, match="model"):
        ASFFS(model="Unknown")._Forward_Objective([], ["x0"], 1, X, y)
    with pytest.raises(ValueError, match="model"):
        ASFFS(model="Unknown")._Backward_Objective(["x0", "x1"], 1, X, y)


# ---------------------------------------------------------------------------
# hybrid.py
# ---------------------------------------------------------------------------


def test_cbfs_empirical_copula():
    fs = CBFS()
    p = fs.Empirical_Copula(np.array([[1, 1], [2, 2], [3, 3]]))
    assert p == pytest.approx([1 / 3, 2 / 3, 1.0])
    with pytest.raises(ValueError):
        fs.Empirical_Copula(5)


def test_cbfs_fit_transform():
    X, y = _regression_data(n=60, p=3)
    fs = CBFS(n_components=2).fit(X, y)
    assert fs._fitted
    assert 1 <= len(fs.selected_features) <= 2
    assert fs.transform(X.values).shape == (60, len(fs.selected_features))

    fs = CBFS(n_prop=0.4).fit(X, y)
    assert fs.n_components == 1 and len(fs.selected_features) == 1
    assert CBFS().fit(X, y).n_components == 3


def test_ga_selection_helpers():
    X, y = _classification_data(p=6, n_classes=2)
    ga = GeneticAlgorithm(seed=1)

    sel = ga._random(X, y, 3)
    assert sum(sel) == 3 and len(sel) == 6
    with pytest.raises(ValueError, match="larger"):
        ga._random(X, y, 10)

    Xd, yd = _discrete_data(p=4)
    assert ga._entropy(Xd, yd, 1) == [1, 0, 0, 0]

    assert sum(ga._t_statistics(X, y, 2)) == 2
    X3, y3 = _classification_data(p=6, n_classes=3)
    assert sum(ga._t_statistics(X3, y3, 2)) == 2
    with pytest.raises(ValueError, match="1 group"):
        ga._t_statistics(X, pd.Series(np.zeros(len(X))), 2)

    sel = ga._SVM_RFE(X, pd.DataFrame(y), 1)
    assert sel == [1, 0, 0, 0, 0, 0]
    Xr, yr = _regression_data(p=3)
    sel = ga._SVM_RFE(Xr, pd.DataFrame(yr), 1)
    assert sel == [1, 0, 0]


@pytest.mark.parametrize(
    "fitness_fit", ["Linear", "Decision Tree", "Random Forest", "SVM"]
)
def test_ga_fitness_functions(fitness_fit):
    Xc, yc = _classification_data(n=80, p=3)
    # more unique responses than UNI_CLASS -> regression models
    Xr, yr = _regression_data(n=150, p=3)
    ga = GeneticAlgorithm(fitness_fit=fitness_fit, fitness_weight=0.9)

    fit_c = ga._cal_fitness(Xc, pd.DataFrame(yc), [1, 0, 0])
    assert 0 < fit_c <= 0.9 + 0.1
    fit_r = ga._cal_fitness(Xr, pd.DataFrame(yr), [1, 1, 0])
    assert np.isfinite(fit_r) and fit_r >= 0
    # no feature selected -> mean prediction
    fit_none = ga._cal_fitness(Xr, pd.DataFrame(yr), [0, 0, 0])
    assert np.isfinite(fit_none)


def test_ga_custom_and_invalid_fitness():
    X, y = _classification_data(n=60, p=3)
    ga = GeneticAlgorithm(fitness_func=lambda X, y, s: float(sum(s)))
    assert ga._cal_fitness(X, pd.DataFrame(y), [1, 1, 0]) == 2.0

    with pytest.raises(ValueError, match="Only support"):
        GeneticAlgorithm(fitness_fit="KNN")._cal_fitness(X, pd.DataFrame(y), [1, 0, 0])


def test_ga_fit_random_and_methods():
    X, y = _classification_data(n=100, p=6)
    ga = GeneticAlgorithm(
        n_components=2,
        fs_method="random",
        n_initial=4,
        n_generations=3,
        n_pair=2,
        fitness_fit="Linear",
        p_mutation=1.0,
        crossover_n=2,
        mutation_n=2,
        seed=1,
    )
    ga.fit(X, y)
    assert ga._fitted
    assert len(ga.selection_) == 6
    out = ga.transform(X.values)
    assert out.shape[1] == sum(ga.selection_)

    ga = GeneticAlgorithm(
        n_prop=0.5,
        fs_method=["Entropy", "t_statistics"],
        n_generations=2,
        n_pair=2,
        fitness_fit="Decision Tree",
        seed=1,
    ).fit(X, y)
    assert ga.n_components == 3
    assert set(ga._feature_sel_methods) == {"Entropy", "t_statistics"}

    ga = GeneticAlgorithm(
        n_components=2,
        fs_method="auto",
        n_generations=2,
        n_pair=2,
        fitness_fit="Linear",
        seed=1,
    ).fit(X, y)
    assert set(ga._feature_sel_methods) == {"Entropy", "t_statistics"}


def test_ga_all_features_selected():
    X, y = _classification_data(n=60, p=3)
    with pytest.warns(UserWarning, match="All features selected"):
        ga = GeneticAlgorithm().fit(X, y)
    assert ga.selection_ == [1, 1, 1]
    with pytest.warns(UserWarning, match="No feature removed"):
        out = ga.transform(X)
    pd.testing.assert_frame_equal(out, X)

    ga.selection_ = [0, 0, 0]
    with pytest.warns(UserWarning, match="All features removed"):
        assert ga.transform(X).shape == (60, 0)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(p_crossover=1.5), "crossover"),
        (dict(p_mutation=-0.1), "mutation"),
        (dict(fs_method="unknown"), "Not recognizing"),
        (dict(crossover_n=10), "cross points"),
        (dict(mutation_n=10, p_mutation=1.0), "mutation points"),
    ],
)
def test_ga_invalid_arguments(kwargs, match):
    X, y = _classification_data(n=60, p=4)
    params = dict(
        n_components=2,
        n_generations=1,
        n_initial=2,
        n_pair=1,
        fitness_fit="Linear",
        seed=1,
    )
    params.update(kwargs)
    with pytest.raises(ValueError, match=match):
        GeneticAlgorithm(**params).fit(X, y)


def test_ga_early_stopping():
    ga = GeneticAlgorithm()
    ga._fitness = np.ones(5)
    assert not ga._early_stopping()
    ga._fitness = np.ones(12) / 12
    assert ga._early_stopping()
    ga._fitness = np.linspace(0, 1, 12)
    assert not ga._early_stopping()


# ---------------------------------------------------------------------------
# sklearn.py
# ---------------------------------------------------------------------------


_UNFITTED = [
    fs_sk.extra_trees_preproc_for_classification,
    fs_sk.extra_trees_preproc_for_regression,
    fs_sk.fast_ica,
    fs_sk.feature_agglomeration,
    fs_sk.kernel_pca,
    fs_sk.kitchen_sinks,
    fs_sk.liblinear_svc_preprocessor,
    fs_sk.nystroem_sampler,
    fs_sk.pca,
    fs_sk.polynomial,
    fs_sk.random_trees_embedding,
    fs_sk.select_percentile_classification,
    fs_sk.select_percentile_regression,
    fs_sk.select_rates_classification,
    fs_sk.select_rates_regression,
    fs_sk.truncatedSVD,
]


@pytest.mark.parametrize("cls", _UNFITTED)
def test_sklearn_transform_before_fit(cls):
    X, _ = _regression_data(n=20, p=3)
    with pytest.raises(NotImplementedError):
        cls().transform(X.values)


def test_kernel_pca_sparse_input():
    X, y = _regression_data(n=60, p=5)
    Xs = scipy.sparse.csr_matrix(np.abs(X.values).astype(np.float32))
    out_sparse = fs_sk.kernel_pca(n_components=3).fit(Xs).transform(Xs)
    dense = np.abs(X.values)
    out_dense = fs_sk.kernel_pca(n_components=3).fit(dense).transform(dense)
    assert out_sparse.shape == (60, 3)
    # same projection as dense input (up to the sign of each component)
    np.testing.assert_allclose(np.abs(out_sparse), np.abs(out_dense), atol=1e-5)


def test_densifier():
    dense = np.arange(6.0).reshape(3, 2)
    d = fs_sk.densifier().fit(dense)
    assert d._fitted
    assert d.transform(dense) is dense
    out = d.transform(scipy.sparse.csr_matrix(dense))
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, dense)


def test_extra_trees_preprocessors_keep_informative_feature():
    Xc, yc = _classification_data(n=150, p=5)
    fs = fs_sk.extra_trees_preproc_for_classification(n_estimators=20, max_features=1.0)
    out = fs.fit(Xc, yc).transform(Xc)
    assert fs._fitted
    assert 1 <= out.shape[1] < 5
    assert fs.selector.get_support()[0]

    Xr, yr = _regression_data(n=150, p=5)
    fs = fs_sk.extra_trees_preproc_for_regression(
        n_estimators=20, max_depth=5, max_leaf_nodes=20
    )
    out = fs.fit(Xr, yr).transform(Xr)
    assert 1 <= out.shape[1] < 5
    assert fs.selector.get_support()[0]


def test_liblinear_svc_preprocessor():
    X, y = _classification_data(n=150, p=5)
    fs = fs_sk.liblinear_svc_preprocessor(C=1.0)
    out = fs.fit(X, y).transform(X)
    assert fs.selector.get_support()[0]
    assert out.shape[1] < 5


def test_decomposition_wrappers_shapes():
    X, y = _regression_data(n=60, p=5)
    Xpos = X.values - X.values.min()

    assert fs_sk.fast_ica(n_components=3, whiten="unit-variance").fit(X, y).transform(
        X
    ).shape == (60, 3)
    # arguments take effect and n_clusters is clipped to the number of features
    fa = fs_sk.feature_agglomeration(n_clusters=10, pooling_func="max").fit(X, y)
    assert fa._fitted
    assert fa.n_clusters == 5
    assert fa.pooling_func is np.max
    assert fa.transform(X).shape == (60, 5)
    fa = fs_sk.feature_agglomeration(
        n_clusters=3, linkage="average", metric="manhattan"
    )
    assert (fa.n_clusters, fa.linkage, fa.metric) == (3, "average", "manhattan")
    assert fa.fit(X, y).transform(X).shape == (60, 3)
    assert fs_sk.feature_agglomeration().fit(X, y).transform(X).shape == (60, 5)
    assert fs_sk.feature_agglomeration(n_clusters=2).fit(X, y).transform(X).shape == (
        60,
        2,
    )
    assert fs_sk.kernel_pca(n_components=3).fit(X, y).transform(X).shape == (60, 3)
    assert fs_sk.kitchen_sinks(n_components=7).fit(X, y).transform(X).shape == (60, 7)
    assert fs_sk.pca(keep_variance=0.99).fit(X, y).transform(X).shape[0] == 60
    assert (
        fs_sk.random_trees_embedding(n_estimators=3).fit(X, y).transform(X).shape[0]
        == 60
    )
    assert fs_sk.truncatedSVD(target_dim=10).fit(X, y).transform(X).shape == (60, 4)

    poly = fs_sk.polynomial(degree=2, include_bias=False).fit(X, y)
    assert poly.transform(X).shape == (60, 5 + 15)

    ny = fs_sk.nystroem_sampler(kernel="rbf", n_components=10).fit(X.values)
    assert ny.transform(X.values).shape == (60, 10)
    ny = fs_sk.nystroem_sampler(kernel="chi2", n_components=10)
    Xneg = X.values.copy()
    ny.fit(Xneg)
    assert (Xneg >= 0).all()  # negatives clipped for chi2
    assert ny.transform(X.values.copy()).shape == (60, 10)
    assert Xpos.min() == 0


@pytest.mark.parametrize("score_func", ["chi2", "f_classif", "mutual_info"])
def test_select_percentile_classification(score_func):
    X, y = _classification_data(n=150, p=5)
    fs = fs_sk.select_percentile_classification(percentile=20, score_func=score_func)
    Xin = X.values - (X.values.min() if score_func == "chi2" else 0)
    out = fs.fit(Xin.copy(), y).transform(Xin.copy())
    assert out.shape == (150, 1)
    assert fs.get_support()[0]


def test_chi2_selectors_clip_negative_values():
    X, y = _classification_data(n=150, p=4)
    for fs in [
        fs_sk.select_percentile_classification(percentile=50, score_func="chi2"),
        fs_sk.select_rates_classification(alpha=0.5, score_func="chi2"),
    ]:
        Xd = X.values.copy()
        fs.fit(Xd, y)
        assert (Xd >= 0).all()
        assert fs.transform(X.values.copy()).shape[0] == 150

        Xs = scipy.sparse.csr_matrix(X.values)
        fs.fit(Xs, y)
        assert (Xs.data >= 0).all()
        assert fs.transform(scipy.sparse.csr_matrix(X.values)).shape[0] == 150


@pytest.mark.parametrize("score_func", ["f_regression", "mutual_info"])
def test_select_regression_selectors(score_func):
    X, y = _regression_data(n=150, p=5)
    fs = fs_sk.select_percentile_regression(percentile=20, score_func=score_func)
    out = fs.fit(X, y).transform(X)
    assert out.shape == (150, 1)
    assert fs.get_support()[0]

    fs = fs_sk.select_rates_regression(
        alpha=0.01 if score_func == "f_regression" else 20, score_func=score_func
    )
    out = fs.fit(X, y).transform(X)
    assert fs.get_support()[0]
    assert out.shape[1] >= 1


@pytest.mark.parametrize("score_func", ["f_classif", "mutual_info"])
def test_select_rates_classification(score_func):
    X, y = _classification_data(n=150, p=5)
    alpha = 0.01 if score_func == "f_classif" else 20
    fs = fs_sk.select_rates_classification(alpha=alpha, score_func=score_func)
    if score_func == "mutual_info":
        assert fs.mode == "percentile"
    out = fs.fit(X, y).transform(X)
    assert fs.get_support()[0]
    assert out.shape[1] >= 1


@pytest.mark.parametrize(
    "cls",
    [
        fs_sk.select_percentile_classification,
        fs_sk.select_percentile_regression,
        fs_sk.select_rates_classification,
        fs_sk.select_rates_regression,
    ],
)
def test_select_invalid_score_func(cls):
    with pytest.raises(ValueError, match="score_func"):
        cls(score_func="unknown")
