"""
File Name: test_utils_stat.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /tests/test_utils/test_utils_stat.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Tuesday, 6th December 2022 11:27:36 pm
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


def test_nan_cov():

    from InsurAutoML.utils.stats import nan_cov

    assert (
        nan_cov(pd.DataFrame([4, 5, 6, np.nan, 1, np.nan]))[0, 0] == 2.8
    ), "nan_cov returns not as expected."


def test_class_means():

    from InsurAutoML.utils.stats import class_means

    X = pd.DataFrame(
        {
            "col_1": [1, 2, 3, 4, 5],
            "col_2": [1, 2, 3, 4, 5],
        }
    )

    y = pd.Series([1, 1, 1, 0, 0])

    assert isinstance(
        class_means(
            X, y), list), "class_means should return a list, get {}".format(
        type(
            class_means(
                X, y)))


def test_empirical_covariance():

    from InsurAutoML.utils import empirical_covariance

    cov = empirical_covariance(10 * np.random.random(size=(10, 10)))

    assert isinstance(
        cov, np.ndarray
    ), "empirical_covariance should return a np.ndarray, get {}".format(type(cov))


def test_class_cov():

    from InsurAutoML.utils.stats import class_cov

    X = np.arange(10)
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

    cov = class_cov(X, y, priors=[0.2, 0.8])

    assert isinstance(
        cov, np.ndarray
    ), "class_cov should return a numpy array, get {}".format(type(cov))


def test_MI():

    from InsurAutoML.utils.stats import MI

    X = pd.DataFrame(np.arange(20).reshape(10, 2), columns=["X_1", "X_2"])
    y = pd.DataFrame(
        np.random.randint(
            0, 2, size=(
                10, 2)), columns=[
            "y_1", "y_2"])

    mi = MI(X, y)

    assert len(mi) == 2, "MI should return a list of length 2, get {}".format(
        len(mi))


def test_ACCC():

    from InsurAutoML.utils.stats import ACCC

    Z = pd.DataFrame(
        np.random.normal(
            0, 2, size=(
                10, 2)), columns=[
            "X_1", "X_2"])
    X = pd.DataFrame(
        np.random.normal(
            0, 2, size=(
                10, 2)), columns=[
            "X_1", "X_2"])
    y = pd.DataFrame(np.random.normal(0, 2, size=(10, 1)), columns=["y"])

    accc = ACCC(Z, y)
    accc = ACCC(Z, y, Z)

    assert isinstance(
        accc, float), "MI should return a float value, get {}".format(
        type(float))


def test_t_score():

    from InsurAutoML.utils.stats import t_score

    X = pd.DataFrame(np.arange(20).reshape(10, 2), columns=["X_1", "X_2"])
    y = pd.DataFrame(
        np.random.randint(
            0, 2, size=(
                10, 2)), columns=[
            "y_1", "y_2"])

    score = t_score(X, y)

    fvalue, pvalue = t_score(X, y, pvalue=True)

    assert len(score) == 2, "t_score should return a list of length 2, get {}".format(
        len(score))


def test_ANOVA():

    from InsurAutoML.utils.stats import ANOVA

    X = pd.DataFrame(np.arange(20).reshape(10, 2), columns=["X_1", "X_2"])
    y = pd.DataFrame(
        np.random.randint(
            0, 5, size=(
                10, 2)), columns=[
            "y_1", "y_2"])

    score = ANOVA(X, y)

    fvalue, pvalue = ANOVA(X, y, pvalue=True)

    assert len(score) == 2, "ANOVA should return a list of length 2, get {}".format(
        len(score))


def test_neg_metrics():

    from InsurAutoML.utils.stats import (
        neg_R2,
        neg_accuracy,
        neg_precision,
        neg_auc,
        neg_hinge,
        neg_f1,
    )
    from sklearn.metrics import (
        r2_score,
        accuracy_score,
        precision_score,
        roc_auc_score,
        hinge_loss,
        f1_score,
    )

    y_true = np.random.randint(0, 2, size=(100,))
    y_pred = np.random.randint(0, 2, size=(100,))
    assert neg_R2(y_true, y_pred) == -1 * r2_score(
        y_true, y_pred
    ), "The neg_R2 function is not correct."
    assert neg_accuracy(y_true, y_pred) == -1 * accuracy_score(
        y_true, y_pred
    ), "The neg_accuracy function is not correct."
    assert neg_precision(y_true, y_pred) == -1 * precision_score(
        y_true, y_pred
    ), "The neg_precision function is not correct."
    assert neg_auc(y_true, y_pred) == -1 * roc_auc_score(
        y_true, y_pred
    ), "The neg_auc function is not correct."
    assert neg_hinge(y_true, y_pred) == -1 * hinge_loss(
        y_true, y_pred
    ), "The neg_hinge function is not correct."
    assert neg_f1(y_true, y_pred) == -1 * f1_score(
        y_true, y_pred
    ), "The neg_f1 function is not correct."
