"""
File Name: test_feature_selection.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /tests/test_feature_selection/test_feature_selection.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 14th November 2022 8:22:16 pm
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
import pandas as pd
from InsurAutoML._feature_selection import feature_selections


def test_feature_selection():

    # loop through all feature selection methods
    for method_name, method in zip(
        feature_selections.keys(), feature_selections.values()
    ):
        data = pd.read_csv("Appendix/Medicalpremium.csv")
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

        assert feature_selection._fitted == True, "Fitted should be True"
        if method_name != "polynomial":
            assert (
                _X.shape[1] <= X.shape[1]
            ), "Feature selection method {} failed".format(method_name)

    # test sklearn version if autosklearn is installed
    import importlib

    autosklearn_spec = importlib.util.find_spec("autosklearn")
    if autosklearn_spec is not None:
        from InsurAutoML._feature_selection._sklearn import (
            extra_trees_preproc_for_classification,
            extra_trees_preproc_for_regression,
            liblinear_svc_preprocessor,
            polynomial,
            select_percentile_classification,
            select_percentile_regression,
            select_rates_classification,
            select_rates_regression,
            truncatedSVD,
        )

        methods = {
            "extra_trees_preproc_for_classification": extra_trees_preproc_for_classification,
            "extra_trees_preproc_for_regression": extra_trees_preproc_for_regression,
            "liblinear_svc_preprocessor": liblinear_svc_preprocessor,
            "polynomial": polynomial,
            "select_percentile_classification": select_percentile_classification,
            "select_percentile_regression": select_percentile_regression,
            "select_rates_classification": select_rates_classification,
            "select_rates_regression": select_rates_regression,
            "truncatedSVD": truncatedSVD,
        }
        for method_name, method in zip(methods.keys(), methods.values()):
            data = pd.read_csv("Appendix/Medicalpremium.csv")
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]

            feature_selection = method()
            feature_selection.fit(X, y)
            _X = feature_selection.transform(X)

            assert feature_selection._fitted == True, "Fitted should be True"


def test_FeatureFilter():

    from InsurAutoML._feature_selection import FeatureFilter

    data = pd.read_csv("Appendix/Medicalpremium.csv")
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    feature_selection = FeatureFilter(
        n_components=5,
        criteria="Pearson",
    )
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted == True, "Fitted should be True"
    assert _X.shape[1] <= X.shape[1], "Feature selection method FeatureFilter failed"

    feature_selection = FeatureFilter(
        n_components=5,
        criteria="MI",
    )
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted == True, "Fitted should be True"
    assert _X.shape[1] <= X.shape[1], "Feature selection method FeatureFilter failed"


def test_ASFFS():

    from InsurAutoML._feature_selection import ASFFS

    data = pd.read_csv("Appendix/Medicalpremium.csv")
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    feature_selection = ASFFS(
        n_components=5,
        model="Linear",
    )
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted == True, "Fitted should be True"
    assert _X.shape[1] <= X.shape[1], "Feature selection method ASFFS failed"

    feature_selection = ASFFS(
        n_components=5,
        model="Lasso",
    )
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted == True, "Fitted should be True"
    assert _X.shape[1] <= X.shape[1], "Feature selection method ASFFS failed"

    feature_selection = ASFFS(n_components=5, model="Ridge", objective="MAE")
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted == True, "Fitted should be True"
    assert _X.shape[1] <= X.shape[1], "Feature selection method ASFFS failed"


def test_GA():

    from InsurAutoML._encoding import DataEncoding
    from InsurAutoML._feature_selection import GeneticAlgorithm

    data = pd.read_csv("Appendix/heart.csv")
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

    assert feature_selection._fitted == True, "Fitted should be True"
    assert _X.shape[1] <= X.shape[1], "Feature selection method GeneticAlgorithm failed"

    feature_selection = GeneticAlgorithm(
        n_components=5, fs_method=["Entropy"], fitness_fit="Decision Tree"
    )
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted == True, "Fitted should be True"
    assert _X.shape[1] <= X.shape[1], "Feature selection method GeneticAlgorithm failed"

    feature_selection = GeneticAlgorithm(
        n_components=5, fs_method=["t_statistics"], fitness_fit="Random Forest"
    )
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted == True, "Fitted should be True"
    assert _X.shape[1] <= X.shape[1], "Feature selection method GeneticAlgorithm failed"

    feature_selection = GeneticAlgorithm(
        n_components=5, fs_method="auto", fitness_fit="SVM"
    )
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted == True, "Fitted should be True"
    assert _X.shape[1] <= X.shape[1], "Feature selection method GeneticAlgorithm failed"


def test_FOCI():

    from InsurAutoML._feature_selection import FOCI

    data = pd.read_csv("Appendix/Medicalpremium.csv")
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    feature_selection = FOCI()
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted == True, "Fitted should be True"
    assert _X.shape[1] <= X.shape[1], "Feature selection method GeneticAlgorithm failed"


def test_ExhaustiveFS():

    from InsurAutoML._feature_selection._wrapper import ExhaustiveFS
    from sklearn.linear_model import Ridge

    data = pd.read_csv("Appendix/Medicalpremium.csv")
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    feature_selection = ExhaustiveFS(criteria="MSE")
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted == True, "Fitted should be True"
    assert _X.shape[1] <= X.shape[1], "Feature selection method GeneticAlgorithm failed"

    feature_selection = ExhaustiveFS(estimator=Ridge(), criteria="MSE")
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted == True, "Fitted should be True"
    assert _X.shape[1] <= X.shape[1], "Feature selection method GeneticAlgorithm failed"
