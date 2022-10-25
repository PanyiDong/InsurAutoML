"""
File: test_feature_selection_wrapper.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Last Version: 0.2.1
Relative Path: /tests/test_feature_selection/test_feature_selection_wrapper.py
File Created: Sunday, 25th September 2022 11:22:59 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 24th October 2022 10:57:01 pm
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
from InsurAutoML._feature_selection import (
    PCA_FeatureSelection,
    RBFSampler,
)


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

    import importlib

    autosklearn_spec = importlib.util.find_spec("autosklearn")
    if autosklearn_spec is None:
        from InsurAutoML._feature_selection._sklearn import densifier
    else:
        from InsurAutoML._feature_selection._autosklearn import densifier

    data = pd.read_csv("Appendix/Medicalpremium.csv")
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    feature_selection = densifier()
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted == True, "Fitted should be True"

    # add tests for sklearn methods when autosklearn is installed
    if autosklearn_spec is not None:
        from InsurAutoML._feature_selection._sklearn import densifier

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

    import importlib

    autosklearn_spec = importlib.util.find_spec("autosklearn")
    if autosklearn_spec is None:
        from InsurAutoML._feature_selection._sklearn import feature_agglomeration
    else:
        from InsurAutoML._feature_selection._autosklearn import feature_agglomeration

    data = pd.read_csv("Appendix/Medicalpremium.csv")
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    feature_selection = feature_agglomeration()
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted == True, "Fitted should be True"

    # add tests for sklearn methods when autosklearn is installed
    if autosklearn_spec is not None:
        from InsurAutoML._feature_selection._sklearn import feature_agglomeration

        data = pd.read_csv("Appendix/Medicalpremium.csv")
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        feature_selection = feature_agglomeration()
        feature_selection.fit(X, y)
        _X = feature_selection.transform(X)

        assert feature_selection._fitted == True, "Fitted should be True"


def test_feature_selection_kernel_pca():

    import importlib

    autosklearn_spec = importlib.util.find_spec("autosklearn")
    if autosklearn_spec is None:
        from InsurAutoML._feature_selection._sklearn import kernel_pca
    else:
        from InsurAutoML._feature_selection._autosklearn import kernel_pca

    data = pd.read_csv("Appendix/Medicalpremium.csv")
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    feature_selection = kernel_pca()
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted == True, "Fitted should be True"

    # add tests for sklearn methods when autosklearn is installed
    if autosklearn_spec is not None:
        from InsurAutoML._feature_selection._sklearn import kernel_pca

        data = pd.read_csv("Appendix/Medicalpremium.csv")
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        feature_selection = kernel_pca()
        feature_selection.fit(X, y)
        _X = feature_selection.transform(X)

        assert feature_selection._fitted == True, "Fitted should be True"


def test_feature_selection_kitchen_sinks():

    import importlib

    autosklearn_spec = importlib.util.find_spec("autosklearn")
    if autosklearn_spec is None:
        from InsurAutoML._feature_selection._sklearn import kitchen_sinks
    else:
        from InsurAutoML._feature_selection._autosklearn import kitchen_sinks

    data = pd.read_csv("Appendix/Medicalpremium.csv")
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    feature_selection = kitchen_sinks()
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted == True, "Fitted should be True"

    # add tests for sklearn methods when autosklearn is installed
    if autosklearn_spec is not None:
        from InsurAutoML._feature_selection._sklearn import kitchen_sinks

        data = pd.read_csv("Appendix/Medicalpremium.csv")
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        feature_selection = kitchen_sinks()
        feature_selection.fit(X, y)
        _X = feature_selection.transform(X)

        assert feature_selection._fitted == True, "Fitted should be True"


def test_feature_selection_nystroem_sampler():

    import importlib

    autosklearn_spec = importlib.util.find_spec("autosklearn")
    if autosklearn_spec is None:
        from InsurAutoML._feature_selection._sklearn import nystroem_sampler
    else:
        from InsurAutoML._feature_selection._autosklearn import nystroem_sampler

    data = pd.read_csv("Appendix/Medicalpremium.csv")
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    feature_selection = nystroem_sampler()
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted == True, "Fitted should be True"

    # add tests for sklearn methods when autosklearn is installed
    if autosklearn_spec is not None:
        from InsurAutoML._feature_selection._sklearn import nystroem_sampler

        data = pd.read_csv("Appendix/Medicalpremium.csv")
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        feature_selection = nystroem_sampler()
        feature_selection.fit(X, y)
        _X = feature_selection.transform(X)

        assert feature_selection._fitted == True, "Fitted should be True"


def test_feature_selection_pca():

    import importlib

    autosklearn_spec = importlib.util.find_spec("autosklearn")
    if autosklearn_spec is None:
        from InsurAutoML._feature_selection._sklearn import pca
    else:
        from InsurAutoML._feature_selection._autosklearn import pca

    data = pd.read_csv("Appendix/Medicalpremium.csv")
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    feature_selection = pca()
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted == True, "Fitted should be True"

    # add tests for sklearn methods when autosklearn is installed
    if autosklearn_spec is not None:
        from InsurAutoML._feature_selection._sklearn import pca

        data = pd.read_csv("Appendix/Medicalpremium.csv")
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        feature_selection = pca()
        feature_selection.fit(X, y)
        _X = feature_selection.transform(X)

        assert feature_selection._fitted == True, "Fitted should be True"


def test_feature_selection_random_trees_embedding():

    import importlib

    autosklearn_spec = importlib.util.find_spec("autosklearn")
    if autosklearn_spec is None:
        from InsurAutoML._feature_selection._sklearn import random_trees_embedding
    else:
        from InsurAutoML._feature_selection._autosklearn import random_trees_embedding

    data = pd.read_csv("Appendix/Medicalpremium.csv")
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    feature_selection = random_trees_embedding()
    feature_selection.fit(X, y)
    _X = feature_selection.transform(X)

    assert feature_selection._fitted == True, "Fitted should be True"

    # add tests for sklearn methods when autosklearn is installed
    if autosklearn_spec is not None:
        from InsurAutoML._feature_selection._sklearn import random_trees_embedding

        data = pd.read_csv("Appendix/Medicalpremium.csv")
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        feature_selection = random_trees_embedding()
        feature_selection.fit(X, y)
        _X = feature_selection.transform(X)

        assert feature_selection._fitted == True, "Fitted should be True"
