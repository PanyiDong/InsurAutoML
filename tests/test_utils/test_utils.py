"""
File Name: test_utils.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /tests/test_utils/test_utils.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 14th November 2022 8:23:45 pm
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

import os
import numpy as np


def test_save_model():

    from InsurAutoML._utils._file import save_model

    save_model(
        "encoder",
        "encoder_hyperparameters",
        "imputer",
        "imputer_hyperparameters",
        "balancing",
        "balancing_hyperparameters",
        "scaling",
        "scaling_hyperparameters",
        "feature_selection",
        "feature_selection_hyperparameters",
        "model",
        "model_hyperparameters",
        "model_name",
    )

    assert os.path.exists("model_name"), "The model is not saved."


def test_EDA():

    from InsurAutoML._utils._eda import EDA
    from InsurAutoML._datasets import PROD, HEART

    features, label = PROD(split="train")

    EDA(features, plot=False)

    assert os.path.exists(
        "tmp/EDA/data_type.csv"), "EDA data type not created."
    assert os.path.exists("tmp/EDA/summary.txt"), "EDA summary not created."

    features, label = HEART()

    EDA(features, label, plot=False)

    assert os.path.exists(
        "tmp/EDA/data_type.csv"), "EDA data type not created."
    assert os.path.exists("tmp/EDA/summary.txt"), "EDA summary not created."


def test_feature_type():

    from InsurAutoML._utils._data import feature_type
    from InsurAutoML._datasets import PROD, HEART

    data, label = PROD(split="test")
    data_type = {}
    for column in data.columns:
        data_type[column] = feature_type(data[column])

    assert isinstance(
        data_type, dict), "The feature_type function is not correct."

    data, label = HEART()
    data_type = {}
    for column in data.columns:
        data_type[column] = feature_type(data[column])

    assert isinstance(
        data_type, dict), "The feature_type function is not correct."


def test_plotHighDimCluster():

    from InsurAutoML._utils._data import plotHighDimCluster

    X = np.random.randint(0, 100, size=(1000, 200))
    y = np.random.randint(0, 5, size=(1000,))

    plotHighDimCluster(X, y, plot=False, method="PCA", dim=2, save=True)

    plotHighDimCluster(X, y, plot=False, method="TSNE", dim=3, save=True)

    assert True, "The plotHighDimCluster function is not correct."


def test_word2vec():

    from InsurAutoML._datasets import PROD
    from InsurAutoML._utils._data import text2vec

    features, labels = PROD(split="test")
    features = features["Product_Description"]
    vec_df = text2vec(features, method="Word2Vec", dim=20)

    assert vec_df.shape[1] == 20, "The word2vec function is not correct."
