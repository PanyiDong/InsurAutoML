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

import numpy as np
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
SEARCH = dict(max_evals=4, timeout=300, timeout_per_trial=300, seed=1)


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
