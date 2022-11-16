"""
File: test_model.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /tests/test_model/test_model.py
File: test_model.py
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Wednesday, 16th November 2022 11:45:11 am
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
from InsurAutoML._utils import formatting


def test_classifiers():

    from InsurAutoML._model import classifiers

    for method_name, method in classifiers.items():

        # pass these methods as they are tested individually
        if method_name not in [
            "LightGBM_Classifier",
            "XGBoost_Classifier",
            "GAM_Classifier",
            "MLP_Classifier",
            "RNN_Classifier",
        ]:

            data = pd.read_csv("example/example_data/heart.csv")
            # encoding categorical features
            encoder = formatting()
            encoder.fit(data)

            # X/y split
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]

            mol = method()
            mol.fit(X, y)
            y_pred = mol.predict(X)
            y_prob = mol.predict_proba(X)

            assert mol._fitted == True, "Model {} has not been fitted.".format(
                method_name
            )

    from InsurAutoML._model import (
        AdaboostClassifier,
        BernoulliNB,
        DecisionTreeClassifier,
        ExtraTreesClassifier,
        GaussianNB,
        HistGradientBoostingClassifier,
        KNearestNeighborsClassifier,
        LDA,
        LibLinear_SVC,
        LibSVM_SVC,
        MLPClassifier,
        MultinomialNB,
        PassiveAggressive,
        QDA,
        RandomForestClassifier,
        SGDClassifier,
    )

    sklearn_classifiers = {
        "AdaboostClassifier": AdaboostClassifier,
        "BernoulliNB": BernoulliNB,
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "GaussianNB": GaussianNB,
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
        "KNearestNeighborsClassifier": KNearestNeighborsClassifier,
        "LDA": LDA,
        "LibLinear_SVC": LibLinear_SVC,
        "LibSVM_SVC": LibSVM_SVC,
        "MLPClassifier": MLPClassifier,
        "MultinomialNB": MultinomialNB,
        "PassiveAggressive": PassiveAggressive,
        "QDA": QDA,
        "RandomForestClassifier": RandomForestClassifier,
        "SGDClassifier": SGDClassifier,
    }

    for method_name, method in sklearn_classifiers.items():

        data = pd.read_csv("example/example_data/heart.csv")
        # encoding categorical features
        encoder = formatting()
        encoder.fit(data)

        # X/y split
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        mol = method()
        mol.fit(X, y)
        y_pred = mol.predict(X)
        y_prob = mol.predict_proba(X)

        assert mol._fitted == True, "Model {} has not been fitted.".format(method_name)


def test_regressors():

    from InsurAutoML._model import regressors

    for method_name, method in regressors.items():

        # pass these methods as they are tested individually
        if method_name not in [
            "LightGBM_Regressor",
            "XGBoost_Regressor",
            "GAM_Regressor",
            "MLP_Regressor",
            "RNN_Regressor",
        ]:

            data = pd.read_csv("example/example_data/insurance.csv")
            # encoding categorical features
            encoder = formatting()
            encoder.fit(data)

            # X/y split
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]

            mol = method()
            mol.fit(X, y)
            y_pred = mol.predict(X)
            try:
                y_prob = mol.predict_proba(X)
            except NotImplementedError:
                pass

            assert mol._fitted == True, "Model {} has not been fitted.".format(
                method_name
            )

    from InsurAutoML._model import (
        AdaboostRegressor,
        ARDRegression,
        DecisionTreeRegressor,
        ExtraTreesRegressor,
        GaussianProcess,
        HistGradientBoostingRegressor,
        KNearestNeighborsRegressor,
        LibLinear_SVR,
        LibSVM_SVR,
        MLPRegressor,
        RandomForestRegressor,
        SGDRegressor,
    )

    sklearn_regressors = {
        "AdaboostRegressor": AdaboostRegressor,
        "ARDRegression": ARDRegression,
        "DecisionTreeRegressor": DecisionTreeRegressor,
        "ExtraTreesRegressor": ExtraTreesRegressor,
        "GaussianProcess": GaussianProcess,
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
        "KNearestNeighborsRegressor": KNearestNeighborsRegressor,
        "LibLinear_SVR": LibLinear_SVR,
        "LibSVM_SVR": LibSVM_SVR,
        "MLPRegressor": MLPRegressor,
        "RandomForestRegressor": RandomForestRegressor,
        "SGDRegressor": SGDRegressor,
    }

    for method_name, method in sklearn_regressors.items():

        data = pd.read_csv("example/example_data/insurance.csv")
        # encoding categorical features
        encoder = formatting()
        encoder.fit(data)

        # X/y split
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        mol = method()
        mol.fit(X, y)
        y_pred = mol.predict(X)
        try:
            y_prob = mol.predict_proba(X)
        except NotImplementedError:
            pass

        assert mol._fitted == True, "Model {} has not been fitted.".format(method_name)
