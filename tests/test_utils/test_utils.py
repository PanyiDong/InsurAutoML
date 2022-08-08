"""
File: test_utils.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Latest Version: 0.2.0
Relative Path: /tests/test_utils/test_utils.py
File Created: Friday, 15th April 2022 7:42:15 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Sunday, 7th August 2022 11:39:50 pm
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
import pandas as pd


def test_load_data():

    from My_AutoML import load_data

    data = load_data().load("Appendix", "insurance")

    assert isinstance(
        data, dict
    ), "load_data should return a dict database, get {}".format(type(data))

    assert isinstance(
        data["insurance"], pd.DataFrame
    ), "load_data should return a dict database containing dataframes, get {}".format(
        type(data["insurance"])
    )


def test_random_guess():

    from My_AutoML._utils._base import random_guess

    assert random_guess(1) == 1, "random_guess(1) should be 1, get {}".format(
        random_guess(1)
    )
    assert random_guess(0) == 0, "random_guess(0) should be 0, get {}".format(
        random_guess(0)
    )
    assert (
        random_guess(0.5) == 0 or random_guess(0.5) == 1
    ), "random_guess(0.5) should be either 0 or 1, get {}".format(random_guess(0.5))


def test_random_index():

    from My_AutoML._utils._base import random_index

    assert (
        np.sort(random_index(5)) == np.array([0, 1, 2, 3, 4])
    ).all(), "random_index(5) should contain [0, 1, 2, 3, 4], get {}".format(
        random_index(5)
    )


def test_random_list():

    from My_AutoML._utils._base import random_list

    assert (
        np.sort(random_list([0, 1, 2, 3, 4])) == np.array([0, 1, 2, 3, 4])
    ).all(), "random_index(5) should contain [0, 1, 2, 3, 4], get {}".format(
        random_list([0, 1, 2, 3, 4])
    )


def test_is_date():

    from My_AutoML._utils._base import is_date

    test = pd.DataFrame(
        {
            "col_1": [1, 2, 3, 4, 5],
            "col_2": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-04",
                "2020-01-05",
            ],
        }
    )

    assert is_date(test, rule="all"), "The is_date method is not correctly done."


def test_feature_rounding():

    from My_AutoML._utils._base import feature_rounding

    test = pd.DataFrame(
        {
            "col_1": [1, 2, 3, 4, 5],
            "col_2": [1.2, 2.2, 3.2, 4.2, 5.2],
        }
    )

    target_data = pd.DataFrame(
        {
            "col_1": [1, 2, 3, 4, 5],
            "col_2": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    assert (
        feature_rounding(test) == target_data
    ).all().all() == True, "The feature_rounding method is not correctly done."


def test_timer():

    from My_AutoML._utils._base import Timer
    import time

    timer = Timer()
    timer.start()
    time.sleep(4)
    timer.stop()
    timer.start()
    time.sleep(3)
    timer.stop()

    assert timer.sum() / timer.avg() == 2.0, "The timer is not correctly done."
    assert timer.cumsum()[-1] == timer.sum(), "The timer is not correctly done."


def test_minloc():

    from My_AutoML._utils._base import minloc

    assert (
        minloc([4, 2, 6, 2, 1]) == 4
    ), "minloc([4, 2, 6, 2, 1]) should be 5, get {}".format(minloc([4, 2, 6, 2, 1]))


def test_maxloc():

    from My_AutoML._utils._base import maxloc

    assert (
        maxloc([4, 2, 6, 2, 1]) == 2
    ), "maxloc([4, 2, 6, 2, 1]) should be 5, get {}".format(maxloc([4, 2, 6, 2, 1]))


def test_True_index():

    from My_AutoML._utils._base import True_index

    assert True_index([True, False, 1, 0, "hello", 5]) == [
        0,
        2,
    ], "True_index([True, False, 1, 0, 'hello', 5]) should be [0, 2], get {}".format(
        True_index([True, False, 1, 0, "hello", 5])
    )


def test_type_of_script():

    from My_AutoML._utils._base import type_of_script

    assert (
        type_of_script() == "terminal"
    ), "type_of_script() should be 'terminal', get {}".format(type_of_script())


def test_as_dataframe():

    from My_AutoML._utils._data import as_dataframe

    converter = as_dataframe()

    _array = converter.to_array(pd.DataFrame([1, 2, 3, 4]))

    _df = converter.to_df(_array)

    assert isinstance(
        _array, np.ndarray
    ), "as_dataframe.to_array should return a np.ndarray, get {}".format(type(_array))

    assert isinstance(
        _df, pd.DataFrame
    ), "as_dataframe.to_df should return a pd.DataFrame, get {}".format(type(_df))


def test_unify_nan():

    from My_AutoML._utils._data import unify_nan

    data = np.arange(15).reshape(5, 3)
    data = pd.DataFrame(data, columns=["column_1", "column_2", "column_3"])
    data.loc[:, "column_1"] = "novalue"
    data.loc[3, "column_2"] = "None"

    target_data = pd.DataFrame(
        {
            "column_1": ["novalue", "novalue", "novalue", "novalue", "novalue"],
            "column_2": [1, 4, 7, "None", 13],
            "column_3": [2, 5, 8, 11, 14],
            "column_1_useNA": [np.nan, np.nan, np.nan, np.nan, np.nan],
            "column_2_useNA": [1, 4, 7, np.nan, 13],
        }
    )

    assert (
        (unify_nan(data).astype(str) == target_data.astype(str)).all().all()
    ), "unify_nan should return target dataframe {}, get {}".format(
        target_data, unify_nan(data)
    )


def test_remove_index_columns():

    from My_AutoML._utils._data import remove_index_columns

    data = pd.DataFrame(
        {
            "col_1": [1, 1, 1, 1, 1],
            "col_2": [1, 2, 3, 4, 5],
            "col_3": [1, 2, 3, 4, 5],
            "col_4": [1, 2, 3, 4, 5],
            "col_5": [1, 2, 3, 4, 5],
        }
    )

    remove_data_0 = remove_index_columns(data.values, axis=0, threshold=0.8)
    remove_data_1 = remove_index_columns(data, axis=1, threshold=0.8, save=True)

    assert isinstance(
        remove_data_0, pd.DataFrame
    ), "remove_index_columns should return a pd.DataFrame, get {}".format(
        type(remove_data_0)
    )
    assert isinstance(
        remove_data_1, pd.DataFrame
    ), "remove_index_columns should return a pd.DataFrame, get {}".format(
        type(remove_data_1)
    )

    remove_data_0 = remove_index_columns(
        data, axis=0, threshold=[0.8, 0.8, 0.8, 0.8, 0.8]
    )
    remove_data_1 = remove_index_columns(
        data, axis=1, threshold=[0.8, 0.8, 0.8, 0.8, 0.8], save=True
    )

    assert isinstance(
        remove_data_0, pd.DataFrame
    ), "remove_index_columns should return a pd.DataFrame, get {}".format(
        type(remove_data_0)
    )
    assert isinstance(
        remove_data_1, pd.DataFrame
    ), "remove_index_columns should return a pd.DataFrame, get {}".format(
        type(remove_data_1)
    )


def test_nan_cov():

    from My_AutoML._utils._stat import nan_cov

    assert (
        nan_cov(pd.DataFrame([4, 5, 6, np.nan, 1, np.nan]))[0, 0] == 2.8
    ), "nan_cov returns not as expected."


def test_class_means():

    from My_AutoML._utils._stat import class_means

    X = pd.DataFrame(
        {
            "col_1": [1, 2, 3, 4, 5],
            "col_2": [1, 2, 3, 4, 5],
        }
    )

    y = pd.Series([1, 1, 1, 0, 0])

    assert isinstance(
        class_means(X, y), list
    ), "class_means should return a list, get {}".format(type(class_means(X, y)))


def test_empirical_covariance():

    from My_AutoML._utils import empirical_covariance

    cov = empirical_covariance(10 * np.random.random(size=(10, 10)))

    assert isinstance(
        cov, np.ndarray
    ), "empirical_covariance should return a np.ndarray, get {}".format(type(cov))


def test_class_cov():

    from My_AutoML._utils._stat import class_cov

    X = np.arange(10)
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

    cov = class_cov(X, y, priors=[0.2, 0.8])

    assert isinstance(
        cov, np.ndarray
    ), "class_cov should return a numpy array, get {}".format(type(cov))


def test_MI():

    from My_AutoML._utils._stat import MI

    X = pd.DataFrame(np.arange(20).reshape(10, 2), columns=["X_1", "X_2"])
    y = pd.DataFrame(np.random.randint(0, 2, size=(10, 2)), columns=["y_1", "y_2"])

    mi = MI(X, y)

    assert len(mi) == 2, "MI should return a list of length 2, get {}".format(len(mi))


def test_t_score():

    from My_AutoML._utils._stat import t_score

    X = pd.DataFrame(np.arange(20).reshape(10, 2), columns=["X_1", "X_2"])
    y = pd.DataFrame(np.random.randint(0, 2, size=(10, 2)), columns=["y_1", "y_2"])

    score = t_score(X, y)

    fvalue, pvalue = t_score(X, y, pvalue=True)

    assert len(score) == 2, "t_score should return a list of length 2, get {}".format(
        len(score)
    )


def test_ANOVA():

    from My_AutoML._utils._stat import ANOVA

    X = pd.DataFrame(np.arange(20).reshape(10, 2), columns=["X_1", "X_2"])
    y = pd.DataFrame(np.random.randint(0, 5, size=(10, 2)), columns=["y_1", "y_2"])

    score = ANOVA(X, y)

    fvalue, pvalue = ANOVA(X, y, pvalue=True)

    assert len(score) == 2, "ANOVA should return a list of length 2, get {}".format(
        len(score)
    )


def test_get_algo():

    from My_AutoML._utils._optimize import get_algo

    get_algo("GridSearch")
    get_algo("HyperOpt")
    get_algo("Repeater")
    get_algo("ConcurrencyLimiter")

    try:
        get_algo("AxSearch")
    except ImportError:
        pass

    try:
        get_algo("BlendSearch")
    except ImportError:
        pass

    try:
        get_algo("CFO")
    except ImportError:
        pass

    try:
        get_algo("HEBO")
    except ImportError:
        pass

    try:
        get_algo("Nevergrad")
    except ImportError:
        pass

    get_algo(get_algo)

    assert True, "The get_algo method is not correctly done."


def test_get_scheduler():

    from My_AutoML._utils._optimize import get_scheduler

    get_scheduler("FIFOScheduler")
    get_scheduler("ASHAScheduler")
    get_scheduler("HyperBandScheduler")
    get_scheduler("MedianStoppingRule")
    get_scheduler("PopulationBasedTraining")
    get_scheduler("PopulationBasedTrainingReplay")

    try:
        get_scheduler("PB2")
    except ImportError:
        pass

    try:
        get_scheduler("HyperBandForBOHB")
    except ImportError:
        pass

    get_scheduler(get_scheduler)

    assert True, "The get_scheduler method is not correctly done."


def test_get_progress_reporter():

    from My_AutoML._utils._optimize import get_progress_reporter

    get_progress_reporter("CLIReporter", max_evals=64, max_error=4)
    get_progress_reporter("JupyterNotebookReporter", max_evals=64, max_error=4)


def test_get_logger():

    from My_AutoML._utils._optimize import get_logger

    get_logger(["Logger", "TBX", "JSON", "CSV", "MLflow"])


def test_save_model():

    from My_AutoML._utils._file import save_model

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

    assert os.path.exists("model_name") == True, "The model is not saved."


def test_formatting():

    from My_AutoML._utils._data import formatting

    data = pd.read_csv("Appendix/insurance.csv")
    train = data.iloc[100:, :]
    test = data.iloc[:100, :]

    formatter = formatting()
    formatter.fit(train)
    formatter.refit(train)
    formatter.refit(test)

    assert True, "The formatting is not correctly done."


def test_get_missing_matrix():

    from My_AutoML._utils._data import get_missing_matrix

    test = pd.DataFrame(
        {
            "col_1": [1, 2, 3, np.nan, 4, "NA"],
            "col_2": [7, "novalue", "none", 10, 11, None],
            "col_3": [
                np.nan,
                "3/12/2000",
                "3/13/2000",
                np.nan,
                "3/12/2000",
                "3/13/2000",
            ],
        }
    )
    test["col_3"] = pd.to_datetime(test["col_3"])

    target_test = pd.DataFrame(
        {
            "col_1": [0, 0, 0, 1, 0, 1],
            "col_2": [0, 1, 1, 0, 0, 1],
            "col_3": [1, 0, 0, 1, 0, 0],
        }
    )

    assert (
        (get_missing_matrix(test) == target_test).all().all()
    ), "The missing matrix is not correct."


def test_extremeclass():

    from My_AutoML._utils._data import ExtremeClass

    cutter = ExtremeClass(extreme_threshold=0.9)
    test = pd.DataFrame(
        np.random.randint(0, 10, size=(100, 10)),
        columns=["col_" + str(i) for i in range(10)],
    )
    test = cutter.cut(test)

    assert True, "The extreme class is not correctly done."


def test_assign_classes():

    from My_AutoML._utils._data import assign_classes

    test = [[0.9, 0.1], [0.2, 0.8]]

    assert (
        assign_classes(test) == np.array([0, 1])
    ).all(), "The classes are not correctly assigned."


def test_has_method():

    from My_AutoML._utils._base import has_method
    from sklearn.linear_model import LogisticRegression

    mol = LogisticRegression()

    assert has_method(mol, "fit") == True, "The has_method function is not correct."
    assert has_method(mol, "__fit") == False, "The has_method function is not correct."


def test_neg_metrics():

    from My_AutoML._utils._stat import (
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


def test_get_estimator():

    from My_AutoML._utils._optimize import get_estimator
    from sklearn.linear_model import LinearRegression
    from My_AutoML._utils._base import has_method

    test_list = [
        "Lasso",
        "Ridge",
        "ExtraTreeRegressor",
        "RandomForestRegressor",
        "LogisticRegression",
        "ExtraTreeClassifier",
        "RandomForestClassifier",
        LinearRegression,
    ]

    for item in test_list:
        estimator = get_estimator(item)
        assert has_method(estimator, "fit") and has_method(
            estimator, "predict"
        ), "The estimator is not correctly called."


def test_get_metrics():

    from My_AutoML._utils._optimize import get_metrics
    from sklearn.metrics import accuracy_score
    from typing import Callable

    test_list = [
        "neg_accuracy",
        "accuracy",
        "neg_precision",
        "precision",
        "neg_auc",
        "auc",
        "neg_hinge",
        "hinge",
        "neg_f1",
        "f1",
        "MSE",
        "MAE",
        "MSLE",
        "neg_R2",
        "R2",
        "MAX",
        accuracy_score,
    ]

    for item in test_list:
        assert isinstance(
            get_metrics(item), Callable
        ), "The metrics are not correctly called."


def test_is_none():

    from My_AutoML._utils._base import is_none

    assert is_none(None) == True, "The is_none function is not correct."
    assert is_none("not none") == False, "The is_none function is not correct."


def test_softmax():

    from My_AutoML._utils._data import softmax

    a = np.array([0.1, -0.1, 1])

    assert (
        softmax(a).sum(axis=1) == np.ones(3)
    ).all(), "The softmax function is not correct."

    a = np.array([[0.1, 0.2], [0.1, -0.2], [1, 0]])

    assert (
        softmax(a).sum(axis=1) == np.ones(3)
    ).all(), "The softmax function is not correct."


def test_EDA():

    from My_AutoML._utils._eda import EDA
    from My_AutoML._datasets import PROD, HEART

    features, label = PROD(split="train")

    EDA(features)

    assert os.path.exists("tmp/EDA/data_type.csv"), "EDA data type not created."
    assert os.path.exists("tmp/EDA/summary.txt"), "EDA summary not created."

    features, label = HEART()

    EDA(features, label)

    assert os.path.exists("tmp/EDA/data_type.csv"), "EDA data type not created."
    assert os.path.exists("tmp/EDA/summary.txt"), "EDA summary not created."


def test_feature_type():

    from My_AutoML._utils._data import feature_type
    from My_AutoML._datasets import PROD, HEART

    data, label = PROD(split="test")
    data_type = {}
    for column in data.columns:
        data_type[column] = feature_type(data[column])

    assert isinstance(data_type, dict), "The feature_type function is not correct."

    data, label = HEART()
    data_type = {}
    for column in data.columns:
        data_type[column] = feature_type(data[column])

    assert isinstance(data_type, dict), "The feature_type function is not correct."


def test_plotHighDimCluster():

    from My_AutoML._utils._data import plotHighDimCluster

    X = np.random.randint(0, 100, size=(1000, 200))
    y = np.random.randint(0, 5, size=(1000,))

    plotHighDimCluster(X, y, method="PCA", dim=2)

    plotHighDimCluster(X, y, method="TSNE", dim=3)

    assert True, "The plotHighDimCluster function is not correct."


def test_word2vec():

    from My_AutoML._datasets import PROD
    from My_AutoML._utils._data import text2vec

    features, labels = PROD(split="test")
    features = features["Product_Description"]
    vec_df = text2vec(features, method="Word2Vec", dim=20)

    assert vec_df.shape[1] == 20, "The word2vec function is not correct."