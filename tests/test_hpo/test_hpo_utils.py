"""
File Name: test_utils.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /tests/test_hpo/test_utils.py
File Created: Monday, 7th November 2022 11:38:53 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 14th November 2022 8:22:44 pm
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

import numpy as np
from InsurAutoML import load_data


def test_pipeline():

    from InsurAutoML._hpo._utils import Pipeline
    from InsurAutoML._encoding import DataEncoding
    from InsurAutoML._imputation import SimpleImputer
    from InsurAutoML._base import no_processing
    from InsurAutoML._scaling import Standardize
    from InsurAutoML._model import LogisticRegression

    # test load_data here
    data = load_data().load("example/example_data", "heart")
    data = data["heart"]

    features = list(data.columns)
    features.remove("HeartDisease")
    response = ["HeartDisease"]

    pip = Pipeline(
        encoder=DataEncoding(),
        imputer=SimpleImputer(method="mean"),
        balancing=no_processing(),
        scaling=Standardize(),
        feature_selection=no_processing(),
        model=LogisticRegression(
            penalty="l2",
            tol=1e-4,
            C=1,
        ),
    )

    pip.fit(data[features], data[response])

    assert pip._fitted, "Pipeline is not fitted."

    pred = pip.predict_proba(data[features])

    assert pred.shape == (data.shape[0], 2), "Prediction shape is not correct."


def test_ClassifierEnsemble():

    from InsurAutoML._hpo._utils import ClassifierEnsemble
    from InsurAutoML._hpo._utils import Pipeline
    from InsurAutoML._encoding import DataEncoding
    from InsurAutoML._imputation import SimpleImputer
    from InsurAutoML._base import no_processing
    from InsurAutoML._scaling import Standardize
    from InsurAutoML._model import LogisticRegression

    # test load_data here
    data = load_data().load("example/example_data", "heart")
    data = data["heart"]

    features = list(data.columns)
    features.remove("HeartDisease")
    response = ["HeartDisease"]

    pip = Pipeline(
        encoder=DataEncoding(),
        imputer=SimpleImputer(method="mean"),
        balancing=no_processing(),
        scaling=Standardize(),
        feature_selection=no_processing(),
        model=LogisticRegression(
            penalty="l2",
            tol=1e-4,
            C=1,
        ),
    )

    pip.fit(data[features], data[response])

    ens1 = ClassifierEnsemble(
        [("pip_1", pip)],
        voting="hard",
        strategy="stacking",
    )
    ens1.fit(data[features], data[response].squeeze())
    pred = ens1.predict(data[features])

    assert pred.shape == (data.shape[0], 1), "Prediction shape is not correct."

    ens2 = ClassifierEnsemble(
        [("pip_1", pip)],
        voting="hard",
        strategy="boosting",
    )
    ens2.fit(data[features], data[response].values)
    pred = ens2.predict(data[features])

    assert pred.shape == (data.shape[0], 1), "Prediction shape is not correct."

    ens3 = ClassifierEnsemble(
        [("pip_1", pip)],
        voting="soft",
        strategy="stacking",
    )
    ens3.fit(data[features], data[response])
    pred = ens3.predict(data[features])

    assert pred.shape == (data.shape[0], 1), "Prediction shape is not correct."

    ens4 = ClassifierEnsemble(
        [("pip_1", pip)],
        voting="soft",
        strategy="boosting",
    )
    ens4.fit(data[features], data[response])
    pred = ens4.predict(data[features])

    assert pred.shape == (data.shape[0], 1), "Prediction shape is not correct."


def test_RegressorEnsemble():

    from InsurAutoML._hpo._utils import RegressorEnsemble
    from InsurAutoML._hpo._utils import Pipeline
    from InsurAutoML._encoding import DataEncoding
    from InsurAutoML._imputation import SimpleImputer
    from InsurAutoML._base import no_processing
    from InsurAutoML._scaling import Standardize
    from InsurAutoML._model import LinearRegression

    # test load_data here
    data = load_data().load("example/example_data", "insurance")
    data = data["insurance"]

    features = list(data.columns)
    features.remove("expenses")
    response = ["expenses"]

    pip = Pipeline(
        encoder=DataEncoding(),
        imputer=SimpleImputer(method="mean"),
        balancing=no_processing(),
        scaling=Standardize(),
        feature_selection=no_processing(),
        model=LinearRegression(),
    )

    pip.fit(data[features], data[response])

    ens1 = RegressorEnsemble(
        [("pip_1", pip), ("pip_2", pip)],
        voting="mean",
        strategy="stacking",
    )
    ens1.fit(data[features], data[response].squeeze())
    pred = ens1.predict(data[features])

    assert pred.shape == (data.shape[0], 1), "Prediction shape is not correct."

    ens2 = RegressorEnsemble(
        [("pip_1", pip), ("pip_2", pip)],
        voting=np.mean,
        strategy="boosting",
    )
    ens2.fit(data[features], data[response].values)
    pred = ens2.predict(data[features])

    assert pred.shape == (data.shape[0], 1), "Prediction shape is not correct."

    ens3 = RegressorEnsemble(
        [("pip_1", pip), ("pip_2", pip)],
        voting="mean",
        strategy="boosting",
    )
    ens3.fit(data[features], data[response])
    pred = ens3.predict(data[features])

    assert pred.shape == (data.shape[0], 1), "Prediction shape is not correct."
