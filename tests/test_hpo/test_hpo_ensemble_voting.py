"""
File: test_hpo_ensemble_voting.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Relative Path: /tests/test_hpo/test_hpo_ensemble_voting.py

Voting methods, ensemble strategies and input validation of Pipeline,
ClassifierEnsemble and RegressorEnsemble.
"""

import numpy as np
import pandas as pd
import pytest

from InsurAutoML.encoding import DataEncoding
from InsurAutoML.hpo.ensemble import ClassifierEnsemble, Pipeline, RegressorEnsemble
from InsurAutoML.imputation import SimpleImputer
from InsurAutoML.model import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    LinearRegression,
    LogisticRegression,
)


@pytest.fixture
def classification():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=["a", "b", "c"])
    y = pd.DataFrame({"y": (X["a"] + 0.5 * X["b"] > 0).astype(int)})
    return X, y


@pytest.fixture
def regression():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=["a", "b", "c"])
    y = pd.DataFrame({"y": 2.0 * X["a"] - X["b"] + rng.normal(scale=0.1, size=120)})
    return X, y


def _classifiers():
    return [
        ("pipe_1", Pipeline(model=LogisticRegression())),
        ("pipe_2", Pipeline(model=DecisionTreeClassifier(max_depth_factor=0.5))),
        ("pipe_3", Pipeline(model=LogisticRegression(C=0.1))),
    ]


def _regressors():
    return [
        ("pipe_1", Pipeline(model=LinearRegression())),
        ("pipe_2", Pipeline(model=DecisionTreeRegressor(max_depth_factor=0.5))),
    ]


# Pipeline


def test_pipeline_fits_all_components():
    X = pd.DataFrame(
        {
            "num": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0],
            "cat": ["a", "b", "a", "b", "a", "b"],
        }
    )
    y = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})

    pipeline = Pipeline(
        encoder=DataEncoding(),
        imputer=SimpleImputer(method="mean"),
        model=LinearRegression(),
    ).fit(X, y)

    assert pipeline._fitted
    assert pipeline.encoder._fitted and pipeline.imputer._fitted
    assert pipeline.predict(X.copy()).shape[0] == len(X)


def test_pipeline_errors(classification):
    X, y = classification
    with pytest.raises(ValueError):
        Pipeline().fit(X, y)  # no model
    with pytest.raises(ValueError):
        Pipeline(model=LogisticRegression()).predict(X)
    with pytest.raises(ValueError):
        Pipeline(model=LogisticRegression()).predict_proba(X)


# ClassifierEnsemble


@pytest.mark.parametrize("voting", ["hard", "soft"])
@pytest.mark.parametrize("strategy", ["stacking", "bagging"])
def test_classifier_voting(classification, voting, strategy):
    X, y = classification
    features = (
        [["a", "b"], ["a", "c"], ["a", "b", "c"]] if strategy == "bagging" else []
    )

    ensemble = ClassifierEnsemble(
        _classifiers(), voting=voting, features=features, strategy=strategy
    ).fit(X, y)
    y_pred = ensemble.predict(X)

    assert y_pred.shape == (len(X), 1) and list(y_pred.columns) == ["y"]
    assert set(np.unique(y_pred)) <= {0, 1}
    # informative feature "a": the ensemble is clearly better than chance
    assert (y_pred["y"].values == y["y"].values).mean() > 0.8

    proba = ensemble.predict_proba(X)
    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_classifier_weights(classification):
    X, y = classification
    estimators = _classifiers()
    for _, pipeline in estimators:
        pipeline.fit(X, y)

    # all weight on the first pipeline: hard voting returns its predictions
    ensemble = ClassifierEnsemble(estimators, voting="hard", weights=[1.0, 0.0, 0.0])
    ensemble.fit(X, y["y"])
    np.testing.assert_array_equal(
        ensemble.predict(X)["y"].values, np.ravel(estimators[0][1].predict(X))
    )


def test_classifier_numpy_response(classification):
    X, y = classification
    ensemble = ClassifierEnsemble(_classifiers()).fit(X, y.values)
    assert list(ensemble.predict(X).columns) == ["response"]


def test_classifier_validation(classification):
    X, y = classification
    with pytest.raises(ValueError):
        ClassifierEnsemble(_classifiers(), voting="mean").fit(X, y)
    with pytest.raises(ValueError):
        ClassifierEnsemble(_classifiers(), strategy="bagging").fit(X, y)
    with pytest.raises(TypeError):
        ClassifierEnsemble(tuple(_classifiers())).fit(X, y)
    with pytest.raises(TypeError):
        ClassifierEnsemble([pipeline for _, pipeline in _classifiers()]).fit(X, y)
    with pytest.raises(TypeError):
        ClassifierEnsemble([("model", LogisticRegression())]).fit(X, y)
    with pytest.raises(ValueError):
        ClassifierEnsemble(_classifiers()).predict(X)
    with pytest.raises(ValueError):
        ClassifierEnsemble(_classifiers()).predict_proba(X)


def test_classifier_predict_proba_unsupported(regression):
    X, y = regression
    y = (y > 0).astype(int)
    # LinearRegression has no predict_proba
    ensemble = ClassifierEnsemble([("pipe_1", Pipeline(model=LinearRegression()))])
    ensemble.fit(X, y)
    with pytest.raises(ValueError):
        ensemble.predict_proba(X)


# RegressorEnsemble


@pytest.mark.parametrize("voting", ["mean", "median", "max", "min", np.mean])
def test_regressor_voting(regression, voting):
    X, y = regression
    estimators = _regressors()
    ensemble = RegressorEnsemble(estimators, voting=voting).fit(X, y)
    y_pred = ensemble.predict(X)

    assert y_pred.shape == (len(X), 1) and list(y_pred.columns) == ["y"]
    preds = np.column_stack([np.ravel(p.predict(X)) for _, p in estimators])
    expected = {
        "mean": preds.mean(axis=1),
        "median": np.median(preds, axis=1),
        "max": preds.max(axis=1),
        "min": preds.min(axis=1),
    }.get(voting if isinstance(voting, str) else "mean")
    np.testing.assert_allclose(y_pred["y"].values, expected)


def test_regressor_weights_and_strategies(regression):
    X, y = regression
    estimators = _regressors()
    preds = None

    # weighted mean
    weighted = RegressorEnsemble(estimators, voting="mean", weights=[3.0, 1.0]).fit(
        X, y
    )
    preds = np.column_stack([np.ravel(p.predict(X)) for _, p in estimators])
    np.testing.assert_allclose(
        weighted.predict(X)["y"].values, np.average(preds, axis=1, weights=[3.0, 1.0])
    )

    # median does not support weights: falls back to the unweighted median
    median = RegressorEnsemble(estimators, voting="median", weights=[3.0, 1.0]).fit(
        X, y
    )
    np.testing.assert_allclose(median.predict(X)["y"].values, np.median(preds, axis=1))

    # boosting: pipelines fitted on residuals, predictions are summed
    boosting = RegressorEnsemble(estimators, strategy="boosting").fit(X, y)
    np.testing.assert_allclose(boosting.predict(X)["y"].values, preds.sum(axis=1))

    # bagging on feature subsets
    bagging = RegressorEnsemble(
        _regressors(), features=[["a", "b"], ["a", "c"]], strategy="bagging"
    ).fit(X, y)
    assert bagging.predict(X).shape == (len(X), 1)


def test_regressor_validation(regression):
    X, y = regression
    with pytest.raises(ValueError):
        RegressorEnsemble(_regressors(), voting="hard").fit(X, y)
    with pytest.raises(ValueError):
        RegressorEnsemble(_regressors(), strategy="bagging").fit(X, y)
    with pytest.raises(TypeError):
        RegressorEnsemble(tuple(_regressors())).fit(X, y)
    with pytest.raises(TypeError):
        RegressorEnsemble([p for _, p in _regressors()]).fit(X, y)
    with pytest.raises(TypeError):
        RegressorEnsemble([("model", LinearRegression())]).fit(X, y)
    with pytest.raises(ValueError):
        RegressorEnsemble(_regressors()).predict(X)
    with pytest.raises(NotImplementedError):
        RegressorEnsemble(_regressors()).predict_proba(X)

    ensemble = RegressorEnsemble(_regressors()).fit(X, y.values)
    assert list(ensemble.predict(X).columns) == ["response"]
