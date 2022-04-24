"""
File: test_feature_selection.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Latest Version: 0.2.0
Relative Path: /tests/test_feature_selection/test_feature_selection.py
File Created: Friday, 15th April 2022 12:27:07 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Sunday, 24th April 2022 5:58:33 pm
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
from My_AutoML._feature_selection import feature_selections
from My_AutoML._feature_selection._base import (
    PCA_FeatureSelection,
    RBFSampler,
)


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


def test_FeatureFilter():

    from My_AutoML._feature_selection import FeatureFilter

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

    from My_AutoML._feature_selection import ASFFS

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
        model="lasso",
    )
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted == True, "Fitted should be True"
    assert _X.shape[1] <= X.shape[1], "Feature selection method ASFFS failed"

    feature_selection = ASFFS(n_components=5, model="ridge", objective="MAE")
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted == True, "Fitted should be True"
    assert _X.shape[1] <= X.shape[1], "Feature selection method ASFFS failed"


def test_GA():

    from My_AutoML._encoding import DataEncoding
    from My_AutoML._feature_selection import GeneticAlgorithm

    data = pd.read_csv("Appendix/heart.csv")
    formatter = DataEncoding()

    # to numerical
    formatter.fit(data)
    data = formatter.refit(data)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    feature_selection = GeneticAlgorithm(
        n_components=5,
        feature_selection="random",
        fitness_fit="Linear",
        n_generations=50,
        p_mutation=0.1,
    )
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted == True, "Fitted should be True"
    assert _X.shape[1] <= X.shape[1], "Feature selection method GeneticAlgorithm failed"

    feature_selection = GeneticAlgorithm(
        n_components=5, feature_selection=["Entropy"], fitness_fit="Logistic"
    )
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted == True, "Fitted should be True"
    assert _X.shape[1] <= X.shape[1], "Feature selection method GeneticAlgorithm failed"

    feature_selection = GeneticAlgorithm(
        n_components=5, feature_selection=["t_statistics"], fitness_fit="Random Forest"
    )
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted == True, "Fitted should be True"
    assert _X.shape[1] <= X.shape[1], "Feature selection method GeneticAlgorithm failed"

    feature_selection = GeneticAlgorithm(
        n_components=5, feature_selection="auto", fitness_fit="SVM"
    )
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted == True, "Fitted should be True"
    assert _X.shape[1] <= X.shape[1], "Feature selection method GeneticAlgorithm failed"


def test_feature_selection_PCA_FeatureSelection():

    data = pd.read_csv("Appendix/Medicalpremium.csv")
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    feature_selection = PCA_FeatureSelection(
        n_components=5,
        solver="auto",
    )
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted == True, "Fitted should be True"

    feature_selection = PCA_FeatureSelection(
        n_components=5,
        solver="full",
    )
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted == True, "Fitted should be True"

    feature_selection = PCA_FeatureSelection(
        n_components=5,
        solver="truncated",
    )
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted == True, "Fitted should be True"

    feature_selection = PCA_FeatureSelection(
        n_components=5,
        solver="randomized",
    )
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted == True, "Fitted should be True"


# def test_feature_selection_LDASelection():

#     data = pd.read_csv("Appendix/Medicalpremium.csv")
#     X = data.iloc[:, :-1]
#     y = data.iloc[:, -1]

#     feature_selection = LDASelection(n_components=5)
#     feature_selection.fit(X, y)

#     assert feature_selection._fitted == True, "Fitted should be True"


def test_feature_selection_RBFSampler():

    data = pd.read_csv("Appendix/Medicalpremium.csv")
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    feature_selection = RBFSampler(n_components=5)
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted == True, "Fitted should be True"


# test decrepted methods


def test_feature_selection_densifier():

    from My_AutoML._feature_selection._autosklearn import densifier

    data = pd.read_csv("Appendix/Medicalpremium.csv")
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    feature_selection = densifier()
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted == True, "Fitted should be True"


# def test_feature_selection_fast_ica():

#     from My_AutoML._feature_selection._autosklearn import fast_ica

#     data = pd.read_csv("Appendix/Medicalpremium.csv")
#     X = data.iloc[:, :-1]
#     y = data.iloc[:, -1]

#     feature_selection = fast_ica()
#     feature_selection.fit(X, y)
#     _X = feature_selection.transform(X)

#     assert feature_selection._fitted == True, "Fitted should be True"


def test_feature_selection_feature_agglomeration():

    from My_AutoML._feature_selection._autosklearn import feature_agglomeration

    data = pd.read_csv("Appendix/Medicalpremium.csv")
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    feature_selection = feature_agglomeration()
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted == True, "Fitted should be True"


def test_feature_selection_kernel_pca():

    from My_AutoML._feature_selection._autosklearn import kernel_pca

    data = pd.read_csv("Appendix/Medicalpremium.csv")
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    feature_selection = kernel_pca()
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted == True, "Fitted should be True"


def test_feature_selection_kitchen_sinks():

    from My_AutoML._feature_selection._autosklearn import kitchen_sinks

    data = pd.read_csv("Appendix/Medicalpremium.csv")
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    feature_selection = kitchen_sinks()
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted == True, "Fitted should be True"


def test_feature_selection_nystroem_sampler():

    from My_AutoML._feature_selection._autosklearn import nystroem_sampler

    data = pd.read_csv("Appendix/Medicalpremium.csv")
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    feature_selection = nystroem_sampler()
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted == True, "Fitted should be True"


def test_feature_selection_pca():

    from My_AutoML._feature_selection._autosklearn import pca

    data = pd.read_csv("Appendix/Medicalpremium.csv")
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    feature_selection = pca()
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted == True, "Fitted should be True"


def test_feature_selection_random_trees_embedding():

    from My_AutoML._feature_selection._autosklearn import random_trees_embedding

    data = pd.read_csv("Appendix/Medicalpremium.csv")
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    feature_selection = random_trees_embedding()
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted == True, "Fitted should be True"
