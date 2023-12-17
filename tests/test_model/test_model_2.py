"""
File: test_model_2.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.5
Relative Path: /tests/test_model/test_model_2.py
File: test_model_2.py
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Saturday, 16th December 2023 8:30:56 pm
Modified By: Panyi Dong (panyid2@illinois.edu)

-----
MIT License

Copyright (c) 2022 - 2022, Panyi Dong

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

import pandas as pd
from InsurAutoML.utils import formatting


def test_add_classifier():
    from InsurAutoML.model.sklearn_classifiers import ComplementNB

    data = pd.read_csv("example/example_data/heart.csv")
    # encoding categorical features
    encoder = formatting()
    encoder.fit(data)

    # X/y split
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    mol = ComplementNB()
    mol.fit(X.abs(), y)  # make sure no negative values
    y_pred = mol.predict(X)
    y_prob = mol.predict_proba(X)

    assert mol._fitted, "Model ComplementNB has not been fitted."


def test_add_regressor():
    from InsurAutoML.model import LibSVM_SVR, MLPRegressor, SGDRegressor

    data = pd.read_csv("example/example_data/insurance.csv")

    # encoding categorical features
    encoder = formatting()
    encoder.fit(data)

    # X/y split
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # model = HistGradientBoostingRegressor()

    # model.fit(X, y)
    # y_pred = model.predict(X)

    # assert (
    #     model._fitted == True
    # ), "Model HistGradientBoostingRegressor has not been fitted."

    model = LibSVM_SVR()
    model.fit(X, y)
    y_pred = model.predict(X)
    try:
        y_prob = model.predict_proba(X)
    except NotImplementedError:
        pass

    assert model._fitted, "Model LibSVM_SVR has not been fitted."

    model = MLPRegressor()

    model.fit(X, y)
    y_pred = model.predict(X)
    try:
        y_prob = model.predict_proba(X)
    except NotImplementedError:
        pass

    assert model._fitted, "Model MLPRegressor has not been fitted."

    model = SGDRegressor()

    model.fit(X, y)
    y_pred = model.predict(X)
    try:
        y_prob = model.predict_proba(X)
    except NotImplementedError:
        pass

    assert model._fitted, "Model SGDRegressor has not been fitted."


def test_lightgbm_classifier():
    from InsurAutoML.model import LightGBM_Classifier

    data = pd.read_csv("example/example_data/heart.csv")

    # encoding categorical features
    encoder = formatting()
    encoder.fit(data)

    # X/y split
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    model = LightGBM_Classifier(
        objective="binary",
        boosting="gbdt",
        max_depth=-1,
        num_leaves=31,
        min_data_in_leaf=20,
        learning_rate=0.1,
        tree_learner="serial",
        num_iterations=100,
        min_gain_to_split=0.0,
        early_stopping_round=0,
        max_bin=255,
        feature_fraction=1.0,
        seed=1,
    )

    model.fit(X, y)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    assert model._fitted, "Model has not been fitted."


def test_lightgbm_regressor():
    from InsurAutoML.model import LightGBM_Regressor

    data = pd.read_csv("example/example_data/insurance.csv")

    # encoding categorical features
    encoder = formatting()
    encoder.fit(data)

    # X/y split
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    model = LightGBM_Regressor(
        objective="regression",
        boosting="gbdt",
        max_depth=-1,
        num_leaves=31,
        min_data_in_leaf=20,
        learning_rate=0.1,
        tree_learner="serial",
        num_iterations=100,
        min_gain_to_split=0.0,
        early_stopping_round=0,
        max_bin=255,
        feature_fraction=1.0,
        seed=1,
    )

    model.fit(X, y)
    y_pred = model.predict(X)
    try:
        y_prob = model.predict_proba(X)
    except NotImplementedError:
        pass

    assert model._fitted, "Model has not been fitted."


def test_xgboost_classifier():
    from InsurAutoML.model import XGBoost_Classifier

    data = pd.read_csv("example/example_data/heart.csv")

    # encoding categorical features
    encoder = formatting()
    encoder.fit(data)

    # X/y split
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    model = XGBoost_Classifier(
        eta=0.3,
        gamma=0,
        max_depth=6,
        min_child_weight=1,
        max_delta_step=0,
        reg_lambda=1,
        reg_alpha=0,
    )

    model.fit(X, y)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    assert model._fitted, "Model has not been fitted."


def test_xgboost_regressor():
    from InsurAutoML.model import XGBoost_Regressor

    data = pd.read_csv("example/example_data/insurance.csv")

    # encoding categorical features
    encoder = formatting()
    encoder.fit(data)

    # X/y split
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    model = XGBoost_Regressor(
        eta=0.3,
        gamma=0,
        max_depth=6,
        min_child_weight=1,
        max_delta_step=0,
        reg_lambda=1,
        reg_alpha=0,
    )

    model.fit(X, y)
    y_pred = model.predict(X)
    try:
        y_prob = model.predict_proba(X)
    except NotImplementedError:
        pass

    assert model._fitted, "Model has not been fitted."


def test_gam_classifier():
    from InsurAutoML.model import GAM_Classifier

    data = pd.read_csv("example/example_data/heart.csv")

    # encoding categorical features
    encoder = formatting()
    encoder.fit(data)

    # X/y split
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    model = GAM_Classifier(
        type="logistic",
        tol=1e-4,
    )

    model.fit(X, y)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    assert model._fitted, "Model has not been fitted."


def test_gam_regressor():
    from InsurAutoML.model import GAM_Regressor

    data = pd.read_csv("example/example_data/insurance.csv")

    # encoding categorical features
    encoder = formatting()
    encoder.fit(data)

    # X/y split
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    model = GAM_Regressor(
        type="linear",
        tol=1e-4,
    )

    model.fit(X, y)
    y_pred = model.predict(X)
    try:
        y_prob = model.predict_proba(X)
    except NotImplementedError:
        pass

    assert model._fitted, "Model GAM_Regressor Linear has not been fitted."

    model = GAM_Regressor(
        type="gamma",
        tol=1e-4,
    )

    model.fit(X, y)
    y_pred = model.predict(X)
    try:
        y_prob = model.predict_proba(X)
    except NotImplementedError:
        pass

    assert model._fitted, "Model GAM_Regressor Gamma has not been fitted."

    model = GAM_Regressor(
        type="poisson",
        tol=1e-4,
    )

    model.fit(X, y)
    y_pred = model.predict(X)
    try:
        y_prob = model.predict_proba(X)
    except NotImplementedError:
        pass

    assert model._fitted, "Model GAM_Regressor Poisson has not been fitted."

    model = GAM_Regressor(
        type="inverse_gaussian",
        tol=1e-4,
    )

    model.fit(X, y)
    y_pred = model.predict(X)
    try:
        y_prob = model.predict_proba(X)
    except NotImplementedError:
        pass

    assert model._fitted, "Model GAM_Regressor Inverse Gaussian has not been fitted."
