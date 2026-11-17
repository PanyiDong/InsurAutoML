"""
File Name: test_informed_automl.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Actuarial and Risk Management Sciences, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.8
Relative Path: /tests/test_informed/test_informed_automl.py
File Created: Monday, 16th November 2026 9:52:23 am
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 16th November 2026 9:57:06 am
Modified By: Panyi Dong (panyid2@illinois.edu)

-----
MIT License

Copyright (c) 2026, Panyi Dong

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

import os
from typing import Any, Dict

import numpy as np
import pytest
import pandas as pd

from InsurAutoML import (
    InformedAutoTabular,
    InformedAutoTabularClassifier,
    InformedAutoTabularRegressor,
)
from InsurAutoML.hpo.informed.fixed import (
    InformedAutoTabular as FixedInformedAutoTabular,
)

# generous budget: MissForest imputation in every trial is slow on loaded machines,
# the search still ends after max_evals trials
SEARCH: Dict[str, Any] = dict(max_evals=4, timeout=300, timeout_per_trial=300, seed=1)


def _check_outputs(model_name):
    assert os.path.exists(os.path.join("tmp", model_name, "init.txt"))
    assert os.path.exists(os.path.join("tmp", model_name, "optimal_setting.txt"))


def test_informed_classifier_stacking(in_tmp_dir, heart_missing):
    X, y = heart_missing

    mol = InformedAutoTabularClassifier(
        model_name="informed_clf",
        ensemble_strategy="stacking",
        n_estimators=2,
        **SEARCH,
    )
    mol.fit(X, y)

    assert mol._fitted
    _check_outputs("informed_clf")

    # the dashboard records the informed search space and the trials
    from InsurAutoML.dashboard import load_state

    state = load_state(os.path.join("tmp", "informed_clf", "dashboard"))
    assert state.finished
    assert state.schema["steps"] == [
        "encoder",
        "complete_prep",
        "missing_prep",
        "model",
    ]
    assert all(
        trial["config"]["model"]["method"] in state.schema["methods"]["model"]
        for trial in state.trials.values()
    )

    y_pred = np.asarray(mol.predict(X.copy())).ravel()
    assert len(y_pred) == len(X)
    assert set(np.unique(y_pred)) <= {0, 1}

    proba = np.asarray(mol.predict_proba(X.copy()))
    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_informed_regressor_bagging(in_tmp_dir, insurance_missing):
    X, y = insurance_missing

    mol = InformedAutoTabularRegressor(
        model_name="informed_reg_bagging",
        ensemble_strategy="bagging",
        n_estimators=2,
        objective="MAE",
        **SEARCH,
    )
    mol.fit(X, y)

    assert mol._fitted
    _check_outputs("informed_reg_bagging")
    # one feature subset per estimator
    assert len(mol.feature_list) == 2

    y_pred = np.asarray(mol.predict(X.copy()), dtype=float).ravel()
    assert len(y_pred) == len(X)
    assert np.isfinite(y_pred).all()


def test_informed_regressor_boosting(in_tmp_dir, insurance_missing):
    X, y = insurance_missing

    mol = InformedAutoTabularRegressor(
        model_name="informed_reg_boosting",
        ensemble_strategy="boosting",
        n_estimators=2,
        objective="MSE",
        **SEARCH,
    )
    mol.fit(X, y)

    assert mol._fitted
    _check_outputs("informed_reg_boosting")

    y_pred = np.asarray(mol.predict(X.copy()), dtype=float).ravel()
    assert len(y_pred) == len(X)
    assert np.isfinite(y_pred).all()


def test_informed_autotabular_detects_classification(in_tmp_dir, heart_missing):
    X, y = heart_missing

    mol = InformedAutoTabular(model_name="informed_auto", n_estimators=1, **SEARCH)
    mol.fit(X, y)

    assert mol.task_mode == "classification"
    y_pred = np.asarray(mol.predict(X.copy())).ravel()
    assert len(y_pred) == len(X)
    assert set(np.unique(y_pred)) <= {0, 1}


def test_fixed_informed_autotabular(in_tmp_dir, insurance_missing):
    X, y = insurance_missing

    mol = FixedInformedAutoTabular(
        model_name="fixed_informed", max_evals=4, timeout=300, n_estimators=1
    )
    mol.fit(X, y["expenses"])

    # one model on complete cases, one on imputed data
    assert mol.complete_model is not None and mol.missing_model is not None

    y_pred = mol.predict(X.copy())
    assert isinstance(y_pred, pd.DataFrame)
    assert len(y_pred) == len(X)
    assert not y_pred.isna().any().any()


def test_fixed_informed_autotabular_saved_and_seeded(
    in_tmp_dir, insurance_missing, monkeypatch
):
    X, y = insurance_missing

    mol = FixedInformedAutoTabular(
        model_name="fixed_saved", max_evals=2, timeout=60, n_estimators=1, seed=1
    )
    mol.fit(X, y["expenses"])
    y_pred = mol.predict(X.copy())
    # the wrapper (models with their preparation) is saved, the sub-models alone not
    assert os.path.isfile("fixed_saved")
    assert not os.path.exists("fixed_saved_complete")
    assert not os.path.exists("fixed_saved_missing")

    # second fit: loaded, no new preparation or search
    def no_fit(*args, **kwargs):
        raise AssertionError("fitted again")

    import InsurAutoML.hpo.informed.fixed as fixed_module

    prep_fit = fixed_module.CompletePrepPipeline.fit
    auto_fit = fixed_module.AutoTabular.fit
    monkeypatch.setattr(fixed_module.CompletePrepPipeline, "fit", no_fit)
    monkeypatch.setattr(fixed_module.AutoTabular, "fit", no_fit)
    loaded = FixedInformedAutoTabular(
        model_name="fixed_saved", max_evals=2, timeout=60, n_estimators=1, seed=1
    ).fit(X, y["expenses"])
    pd.testing.assert_frame_equal(loaded.predict(X.copy()), y_pred)
    assert list(loaded.complete_prep.filter.select_features) == list(
        mol.complete_prep.filter.select_features
    )

    # other data: fitted again (the saved wrapper is not reused)
    monkeypatch.setattr(fixed_module.CompletePrepPipeline, "fit", prep_fit)
    monkeypatch.setattr(fixed_module.AutoTabular, "fit", auto_fit)
    with pytest.warns(UserWarning, match="other data or settings"):
        refit = FixedInformedAutoTabular(
            model_name="fixed_saved", max_evals=2, timeout=60, n_estimators=1, seed=1
        ).fit(X.iloc[:-20], y["expenses"].iloc[:-20])
    assert refit.complete_model is not None

    # not a saved wrapper
    with open("not_a_model", "wb") as out_f:
        __import__("pickle").dump([1], out_f)
    with pytest.raises(ValueError, match="not a saved InformedAutoTabular"):
        FixedInformedAutoTabular(model_name="not_a_model").fit(X, y["expenses"])


def test_fixed_informed_autotabular_without_missing(in_tmp_dir, insurance_missing):
    X, y = insurance_missing
    X = X.fillna(X.median(numeric_only=True))

    mol = FixedInformedAutoTabular(
        model_name="fixed_complete", max_evals=4, timeout=300, n_estimators=1
    )
    mol.fit(X, y["expenses"])

    # no missing values: only the complete-case model is fitted
    assert mol.complete_model is not None and mol.missing_model is None
    y_pred = mol.predict(X.copy())
    assert len(y_pred) == len(X)
