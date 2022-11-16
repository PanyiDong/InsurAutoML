"""
File: test_utils_optimize.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /tests/test_utils/test_utils_optimize.py
File: test_utils_optimize.py
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Tuesday, 15th November 2022 11:30:31 pm
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


def test_get_algo():

    from InsurAutoML._utils._optimize import get_algo

    get_algo("GridSearch")
    get_algo("HyperOpt")
    get_algo("Repeater")
    get_algo("ConcurrencyLimiter")

    # try:
    #     get_algo("AxSearch")
    # except ImportError:
    #     pass

    # try:
    #     get_algo("BlendSearch")
    # except ImportError:
    #     pass

    try:
        get_algo("CFO")
    except ImportError:
        pass

    # try:
    #     get_algo("HEBO")
    # except ImportError:
    #     pass

    try:
        get_algo("Nevergrad")
    except ImportError:
        pass

    # get_algo(get_algo)

    assert True, "The get_algo method is not correctly done."


def test_get_scheduler():

    from InsurAutoML._utils._optimize import get_scheduler

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

    from InsurAutoML._utils._optimize import get_progress_reporter

    get_progress_reporter("CLIReporter", max_evals=64, max_error=4)
    get_progress_reporter("JupyterNotebookReporter", max_evals=64, max_error=4)


def test_get_logger():

    from InsurAutoML._utils._optimize import get_logger

    get_logger(["Logger", "TBX", "JSON", "CSV", "MLflow"])


def test_get_estimator():

    from InsurAutoML._utils._optimize import get_estimator
    from sklearn.linear_model import LinearRegression
    from InsurAutoML._utils._base import has_method

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

    from InsurAutoML._utils._optimize import get_metrics
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


def test_check_func():

    from InsurAutoML._utils._optimize import check_func
    from InsurAutoML._model import LogisticRegression

    assert (
        check_func(LogisticRegression, ref="model") == None
    ), "The check_func method is not correctly done."


def test_check_status():

    from InsurAutoML._utils._optimize import check_status
    from InsurAutoML._hyperparameters import regressor_hyperparameter
    from InsurAutoML._model import regressors

    assert (
        check_status(regressors, regressor_hyperparameter, ref="model") == None
    ), "The check_status method is not correctly done."
