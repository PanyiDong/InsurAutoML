"""
A stored model (and the experiments of its run folder) is only reused for the same
data and settings.
"""

import numpy as np
import pandas as pd
import pytest
from ray import tune

import InsurAutoML
from InsurAutoML import load_data
from InsurAutoML.hpo.base import AutoTabularBase


def _heart():
    data = load_data().load("example/example_data", "heart")["heart"]
    return data.drop(columns=["HeartDisease"]), data[["HeartDisease"]]


def _model(**kwargs):
    settings = dict(
        model_name="fingerprint",
        n_estimators=2,
        timeout=30,
        max_evals=3,
        objective="accuracy",
        seed=1,
    )
    settings.update(kwargs)
    return InsurAutoML.AutoTabular(**settings)


def test_changed_data_searches_again(monkeypatch):
    X, y = _heart()
    first = _model().fit(X, y)
    assert len(first.ensemble.estimators) <= 2

    # other data: the stored model and the experiments in tmp/fingerprint are not
    # reused (no restore of the finished Ray Tune experiment)
    def no_restore(*args, **kwargs):
        raise AssertionError("old experiment restored")

    monkeypatch.setattr(tune.Tuner, "restore", no_restore)
    with pytest.warns(UserWarning, match="other data or settings"):
        first.fit(X.iloc[:-20], y.iloc[:-20])
    # the same object: a new ensemble, not added to the previous one
    assert len(first.ensemble.estimators) <= 2


def test_data_hash_and_callable_settings():
    # cells that can not be hashed
    data = pd.DataFrame({"a": [[1, 2], [3]], "b": [{"k": 1}, {"k": 2}]})
    assert AutoTabularBase._data_hash(data) == AutoTabularBase._data_hash(data.copy())
    assert AutoTabularBase._data_hash(None) is None

    def model(objective):
        mol = AutoTabularBase(task_mode="classification")
        mol.objective = objective
        return mol._fit_fingerprint(pd.DataFrame({"a": [1]}))

    # lambdas have no stable name: never the same fingerprint
    assert model(lambda y, p: 0) != model(lambda y, p: 1)
    # module level functions: the same
    assert model(np.mean) == model(np.mean)
    assert model("accuracy") == model("accuracy")


def test_unlimited_evaluations(tmp_path):
    # bagging/boosting split max_evals over the rounds
    X, y = _heart()
    with pytest.raises(ValueError, match="set max_evals > 0"):
        _model(model_name="unlimited", ensemble_strategy="bagging", max_evals=-1).fit(
            X, y
        )

    # no trial limit: the time limit of a trial is the time budget
    mol = AutoTabularBase(
        task_mode="classification",
        max_evals=-1,
        timeout=50,
        temp_directory=str(tmp_path),
        model_name="unlimited_stacking",
    )
    mol._init_fit()
    assert mol.timeout_per_trial == 50
