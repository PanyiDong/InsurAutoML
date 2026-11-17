"""
File: test_informed_ensemble.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Relative Path: /tests/test_informed/test_informed_ensemble.py

Voting methods, ensemble strategies and input validation of
InformedClassifierEnsemble and InformedRegressorEnsemble, using informed pipelines
with simple preparation steps and fitted models.
"""

import numpy as np
import pandas as pd
import pytest

from InsurAutoML.hpo.informed.informed_ensemble import (
    InformedClassifierEnsemble,
    InformedPipeline,
    InformedRegressorEnsemble,
)
from InsurAutoML.model import LinearRegression, LogisticRegression


class _CompleteCases:
    """Rows without missing values."""

    _fitted = True

    def transform(self, X, y=None):
        return X.dropna(), None


class _MeanImputation:
    """All rows, missing values filled by the column mean."""

    _fitted = True

    def transform(self, X, y=None):
        return X.fillna(X.mean()), None


def _pipeline(model_class, X, y, **kwargs):
    model_complete = model_class(**kwargs).fit(X, y.values.ravel())
    model_missing = model_class(**kwargs).fit(X, y.values.ravel())
    return InformedPipeline(
        complete_prep=_CompleteCases(),
        missing_prep=_MeanImputation(),
        model_complete=model_complete,
        model_missing=model_missing,
    )


@pytest.fixture
def classification():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(150, 3)), columns=["a", "b", "c"])
    y = pd.DataFrame({"y": (X["a"] - 0.5 * X["b"] > 0).astype(int)})
    X_missing = X.copy()
    X_missing.loc[rng.random(150) < 0.2, "c"] = np.nan
    return X, y, X_missing


@pytest.fixture
def regression():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(150, 3)), columns=["a", "b", "c"])
    y = pd.DataFrame({"y": 2.0 * X["a"] - X["b"] + rng.normal(scale=0.1, size=150)})
    return X, y


def _classifiers(X, y):
    return [
        ("pipe_1", _pipeline(LogisticRegression, X, y)),
        ("pipe_2", _pipeline(LogisticRegression, X, y, C=0.1)),
    ]


def _regressors(X, y):
    return [
        ("pipe_1", _pipeline(LinearRegression, X, y)),
        ("pipe_2", _pipeline(LinearRegression, X[["a"]], y)),
    ]


@pytest.mark.parametrize("voting", ["hard", "soft"])
@pytest.mark.parametrize("strategy", ["stacking", "bagging"])
def test_classifier_voting(classification, voting, strategy):
    X, y, X_missing = classification
    features = [["a", "b", "c"], ["a", "b", "c"]] if strategy == "bagging" else []

    ensemble = InformedClassifierEnsemble(
        [(name, pipe.fit(X, y)) for name, pipe in _classifiers(X, y)],
        voting=voting,
        features=features,
        strategy=strategy,
    ).fit(X, y)

    # rows with missing values are predicted by the missing-data models
    y_pred = ensemble.predict(X_missing)
    assert y_pred.shape == (len(X), 1)
    assert set(np.unique(y_pred)) <= {0, 1}
    assert (y_pred["y"].values == y["y"].values).mean() > 0.8

    proba = np.asarray(ensemble.predict_proba(X_missing))
    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_classifier_validation(classification):
    X, y, _ = classification
    pipelines = [(name, pipe.fit(X, y)) for name, pipe in _classifiers(X, y)]
    with pytest.raises(ValueError):
        InformedClassifierEnsemble(pipelines, voting="mean").fit(X, y)
    with pytest.raises(ValueError):
        InformedClassifierEnsemble(pipelines, strategy="bagging").fit(X, y)
    with pytest.raises(TypeError):
        InformedClassifierEnsemble(tuple(pipelines)).fit(X, y)
    with pytest.raises(TypeError):
        InformedClassifierEnsemble([pipe for _, pipe in pipelines]).fit(X, y)
    with pytest.raises(TypeError):
        InformedClassifierEnsemble([("model", LogisticRegression())]).fit(X, y)
    with pytest.raises(ValueError):
        InformedClassifierEnsemble(pipelines).predict(X)

    ensemble = InformedClassifierEnsemble(pipelines).fit(X, y.values)
    assert list(ensemble.predict(X).columns) == ["response"]


@pytest.mark.parametrize("voting", ["mean", "median", "max", "min", np.mean])
def test_regressor_voting(regression, voting):
    X, y = regression
    pipelines = [(name, pipe.fit(X, y)) for name, pipe in _regressors(X, y)]
    features = [["a", "b", "c"], ["a"]]

    ensemble = InformedRegressorEnsemble(pipelines, voting=voting, features=features)
    y_pred = ensemble.fit(X, y).predict(X)

    assert y_pred.shape == (len(X), 1)
    preds = np.column_stack(
        [np.ravel(pipe.predict(X[f])) for (_, pipe), f in zip(pipelines, features)]
    )
    expected = {
        "mean": preds.mean(axis=1),
        "median": np.median(preds, axis=1),
        "max": preds.max(axis=1),
        "min": preds.min(axis=1),
    }.get(voting if isinstance(voting, str) else "mean")
    np.testing.assert_allclose(np.ravel(y_pred), expected)


def test_regressor_weights_boosting_validation(regression):
    X, y = regression
    pipelines = [(name, pipe.fit(X, y)) for name, pipe in _regressors(X, y)]
    features = [["a", "b", "c"], ["a"]]
    preds = np.column_stack(
        [np.ravel(pipe.predict(X[f])) for (_, pipe), f in zip(pipelines, features)]
    )

    weighted = InformedRegressorEnsemble(
        pipelines, voting="mean", weights=[3.0, 1.0], features=features
    ).fit(X, y)
    np.testing.assert_allclose(
        np.ravel(weighted.predict(X)), np.average(preds, axis=1, weights=[3.0, 1.0])
    )

    boosting = InformedRegressorEnsemble(
        pipelines, features=features, strategy="boosting"
    ).fit(X, y)
    np.testing.assert_allclose(np.ravel(boosting.predict(X)), preds.sum(axis=1))

    with pytest.raises(ValueError):
        InformedRegressorEnsemble(pipelines, voting="hard").fit(X, y)
    with pytest.raises(ValueError):
        InformedRegressorEnsemble(pipelines, strategy="bagging").fit(X, y)
    with pytest.raises(TypeError):
        InformedRegressorEnsemble(tuple(pipelines)).fit(X, y)
    with pytest.raises(TypeError):
        InformedRegressorEnsemble([("model", LinearRegression())]).fit(X, y)
    with pytest.raises(ValueError):
        InformedRegressorEnsemble(pipelines).predict(X)
    with pytest.raises(NotImplementedError):
        InformedRegressorEnsemble(pipelines).predict_proba(X)
