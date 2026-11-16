"""
File Name: test_feature_selection.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.6
Relative Path: /tests/test_feature_selection/test_feature_selection.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Tuesday, 20th January 2026 12:24:29 pm
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
import pytest
import pandas as pd
from InsurAutoML.feature_selection import feature_selections


def test_feature_selection():
    # loop through all feature selection methods
    for method_name, method in zip(
        feature_selections.keys(), feature_selections.values()
    ):
        data = pd.read_csv("data/Medicalpremium.csv")
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        if method_name in ["FeatureFilter", "ASFFS", "GeneticAlgorithm", "RBFSampler"]:
            pass
        elif method_name == "SFS":
            feature_selection = method(
                estimator="Lasso",
                n_components=5,
                criteria="MSE",
            )
        elif method_name in ["mRMR", "CBFS"]:
            feature_selection = method(n_components=5)
        else:
            feature_selection = method()

        feature_selection.fit(X, y)
        _X = feature_selection.transform(X)

        assert feature_selection._fitted, "Fitted should be True"
        if method_name != "polynomial":
            assert (
                _X.shape[1] <= X.shape[1]
            ), "Feature selection method {} failed".format(method_name)


def test_FeatureFilter():
    from InsurAutoML.feature_selection import FeatureFilter

    data = pd.read_csv("data/Medicalpremium.csv")
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    feature_selection = FeatureFilter(
        n_components=5,
        criteria="Pearson",
    )
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted, "Fitted should be True"
    assert _X.shape[1] <= X.shape[1], "Feature selection method FeatureFilter failed"

    feature_selection = FeatureFilter(
        n_components=5,
        criteria="MI",
    )
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted, "Fitted should be True"
    assert _X.shape[1] <= X.shape[1], "Feature selection method FeatureFilter failed"

    feature_selection = FeatureFilter(
        n_components=5,
        criteria="CCC",
    )
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted, "Fitted should be True"
    assert _X.shape[1] <= X.shape[1], "Feature selection method FeatureFilter failed"


def test_ASFFS():
    from InsurAutoML.feature_selection import ASFFS

    data = pd.read_csv("data/Medicalpremium.csv")
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    feature_selection = ASFFS(
        n_components=5,
        model="Linear",
    )
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted, "Fitted should be True"
    assert _X.shape[1] <= X.shape[1], "Feature selection method ASFFS failed"

    feature_selection = ASFFS(
        n_components=5,
        model="Lasso",
    )
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted, "Fitted should be True"
    assert _X.shape[1] <= X.shape[1], "Feature selection method ASFFS failed"

    feature_selection = ASFFS(n_components=5, model="Ridge", objective="MAE")
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted, "Fitted should be True"
    assert _X.shape[1] <= X.shape[1], "Feature selection method ASFFS failed"


def test_GA():
    from InsurAutoML.encoding import DataEncoding
    from InsurAutoML.feature_selection import GeneticAlgorithm

    data = pd.read_csv("data/heart.csv")
    formatter = DataEncoding()

    # to numerical
    formatter.fit(data)
    data = formatter.refit(data)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    feature_selection = GeneticAlgorithm(
        n_components=5,
        fs_method="random",
        fitness_fit="Linear",
        n_generations=50,
        p_mutation=0.1,
    )
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted, "Fitted should be True"
    assert _X.shape[1] <= X.shape[1], "Feature selection method GeneticAlgorithm failed"

    feature_selection = GeneticAlgorithm(
        n_components=5, fs_method=["Entropy"], fitness_fit="Decision Tree"
    )
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted, "Fitted should be True"
    assert _X.shape[1] <= X.shape[1], "Feature selection method GeneticAlgorithm failed"

    feature_selection = GeneticAlgorithm(
        n_components=5, fs_method=["t_statistics"], fitness_fit="Random Forest"
    )
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted, "Fitted should be True"
    assert _X.shape[1] <= X.shape[1], "Feature selection method GeneticAlgorithm failed"

    feature_selection = GeneticAlgorithm(
        n_components=5, fs_method="auto", fitness_fit="SVM"
    )
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted, "Fitted should be True"
    assert _X.shape[1] <= X.shape[1], "Feature selection method GeneticAlgorithm failed"


def test_FOCI():
    from InsurAutoML.feature_selection.filter import FOCI

    data = pd.read_csv("data/Medicalpremium.csv")
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    feature_selection = FOCI()
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted, "Fitted should be True"
    assert _X.shape[1] <= X.shape[1], "Feature selection method GeneticAlgorithm failed"


def test_ExhaustiveFS():
    from InsurAutoML.feature_selection.wrapper import ExhaustiveFS
    from sklearn.linear_model import Ridge

    data = pd.read_csv("data/Medicalpremium.csv")
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    feature_selection = ExhaustiveFS(criteria="MSE")
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted, "Fitted should be True"
    assert _X.shape[1] <= X.shape[1], "Feature selection method GeneticAlgorithm failed"

    feature_selection = ExhaustiveFS(estimator=Ridge(), criteria="MSE")
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted, "Fitted should be True"
    assert _X.shape[1] <= X.shape[1], "Feature selection method GeneticAlgorithm failed"


def test_GA_fitness_without_selected_features():
    # an empty (random) selection predicts a constant: the most frequent class for
    # classification (accuracy cannot score a continuous mean), the mean otherwise
    from InsurAutoML.feature_selection import GeneticAlgorithm

    X = pd.DataFrame({"a": [0.1, 0.4, 0.2, 0.9], "b": [1.0, 0.0, 1.0, 0.0]})
    selector = GeneticAlgorithm(n_components=1, fitness_fit="Linear", fitness_weight=1)

    y_class = pd.DataFrame({"y": [1.0, 0.0, 1.0, 1.0]})
    assert selector._cal_fitness(X, y_class, [0, 0]) == 0.75

    # more unique values than UNI_CLASS: regression (fitness is the MSE of the mean)
    rng = np.random.default_rng(0)
    X_reg = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    y_reg = pd.DataFrame({"y": rng.normal(size=200)})
    assert selector._cal_fitness(X_reg, y_reg, [0, 0]) == pytest.approx(np.var(y_reg["y"]))
