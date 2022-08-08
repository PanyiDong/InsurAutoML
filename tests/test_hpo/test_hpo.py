"""
File: test_regression.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Latest Version: 0.2.0
Relative Path: /tests/test_hpo/test_regression.py
File Created: Sunday, 10th April 2022 12:00:04 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 8th August 2022 12:10:51 am
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
from ray import tune

import My_AutoML
from My_AutoML import load_data

# use command line interaction to run the model
# apparently, same class object called in one test case will not be able
# to run the model correctly after the first time
# detect whether optimal setting exists as method of determining whether
# the model is fitted correctly


# def test_stroke():

#     os.system(
#         "python main.py --data_folder Appendix --train_data healthcare-dataset-stroke-data --response stroke"
#     )

#     assert (
#         os.path.exists("tmp/healthcare-dataset-stroke-data_model/init.txt") == True
#     ), "Classification for Stroke data failed to initiated."
#     # assert (
#     #     mol_heart._fitted == True
#     # ), "Classification for Heart data failed to fit."
#     assert (
#         os.path.exists("tmp/healthcare-dataset-stroke-data_model/optimal_setting.txt")
#         == True
#     ), "Classification for Stroke data failed to find optimal setting."


def test_objective_1():

    from My_AutoML._hpo._utils import TabularObjective
    from My_AutoML._encoding import DataEncoding
    from My_AutoML._imputation import SimpleImputer
    from My_AutoML._base import no_processing
    from My_AutoML._scaling import Standardize
    from My_AutoML._model import LogisticRegression

    # test load_data here
    data = load_data().load("example/example_data", "heart")
    data = data["heart"]

    features = list(data.columns)
    features.remove("HeartDisease")
    response = ["HeartDisease"]

    encoder = {"DataEncoding": DataEncoding}
    imputer = {"SimpleImputer": SimpleImputer}
    balancing = {"no_processing": no_processing}
    scaling = {"Standardize": Standardize}
    feature_selection = {"no_processing": no_processing}
    models = {"LogisticRegression": LogisticRegression}

    params = {
        "encoder": {
            "encoder_1": "DataEncoding",
        },
        "imputer": {
            "imputer_1": "SimpleImputer",
            "SimpleImputer_method": "mean",
        },
        "balancing": {"balancing_1": "no_processing"},
        "scaling": {"scaling_2": "Standardize"},
        "feature_selection": {"feature_selection_1": "no_processing"},
        "model": {
            "model_17": "LogisticRegression",
            "LogisticRegression_penalty": "l2",
            "LogisticRegression_tol": 1e-4,
            "LogisticRegression_C": 1,
        },
    }

    clf = TabularObjective(
        params,
    )
    clf.setup(
        params,
        _X=data[features],
        _y=data[response],
        encoder=encoder,
        imputer=imputer,
        balancing=balancing,
        scaling=scaling,
        feature_selection=feature_selection,
        models=models,
        model_name="obj_1",
        task_mode="classification",
        objective="accuracy",
        validation=True,
        valid_size=0.15,
        full_status=False,
        reset_index=True,
        _iter=1,
        seed=1,
    )
    result = clf.step()
    clf.reset_config(params)

    assert isinstance(result, dict), "Objective function should return a dict."
    assert "loss" in result.keys(), "Objective function should return loss."
    assert (
        "fitted_model" in result.keys()
    ), "Objective function should return fitted model."
    assert (
        "training_status" in result.keys()
    ), "Objective function should return training status."


def test_objective_2():

    from My_AutoML._hpo._utils import TabularObjective
    from My_AutoML._encoding import DataEncoding
    from My_AutoML._imputation import SimpleImputer
    from My_AutoML._base import no_processing
    from My_AutoML._scaling import Standardize
    from My_AutoML._model import LogisticRegression

    # test load_data here
    data = load_data().load("example/example_data", "heart")
    data = data["heart"]

    features = list(data.columns)
    features.remove("HeartDisease")
    response = ["HeartDisease"]

    encoder = {"DataEncoding": DataEncoding}
    imputer = {"SimpleImputer": SimpleImputer}
    balancing = {"no_processing": no_processing}
    scaling = {"Standardize": Standardize}
    feature_selection = {"no_processing": no_processing}
    models = {"LogisticRegression": LogisticRegression}

    params = {
        "encoder": {
            "encoder_1": "DataEncoding",
        },
        "imputer": {
            "imputer_1": "SimpleImputer",
            "SimpleImputer_method": "mean",
        },
        "balancing": {"balancing_1": "no_processing"},
        "scaling": {"scaling_2": "Standardize"},
        "feature_selection": {"feature_selection_1": "no_processing"},
        "model": {
            "model_17": "LogisticRegression",
            "LogisticRegression_penalty": "l2",
            "LogisticRegression_tol": 1e-4,
            "LogisticRegression_C": 1,
        },
    }

    clf = TabularObjective(
        params,
    )
    clf.setup(
        params,
        _X=data[features],
        _y=data[response],
        encoder=encoder,
        imputer=imputer,
        balancing=balancing,
        scaling=scaling,
        feature_selection=feature_selection,
        models=models,
        model_name="obj_2",
        task_mode="classification",
        objective="auc",
        validation=False,
        valid_size=0.15,
        full_status=False,
        reset_index=True,
        _iter=1,
        seed=1,
    )
    result = clf.step()
    clf.reset_config(params)

    assert isinstance(result, dict), "Objective function should return a dict."
    assert "loss" in result.keys(), "Objective function should return loss."
    assert (
        "fitted_model" in result.keys()
    ), "Objective function should return fitted model."
    assert (
        "training_status" in result.keys()
    ), "Objective function should return training status."


def test_objective_3():

    from My_AutoML._hpo._utils import TabularObjective
    from My_AutoML._encoding import DataEncoding
    from My_AutoML._imputation import SimpleImputer
    from My_AutoML._base import no_processing
    from My_AutoML._scaling import Standardize
    from My_AutoML._model import LinearRegression

    # test load_data here
    data = load_data().load("example/example_data", "insurance")
    data = data["insurance"]

    features = list(data.columns)
    features.remove("expenses")
    response = ["expenses"]

    encoder = {"DataEncoding": DataEncoding}
    imputer = {"SimpleImputer": SimpleImputer}
    balancing = {"no_processing": no_processing}
    scaling = {"Standardize": Standardize}
    feature_selection = {"no_processing": no_processing}
    models = {"LinearRegression": LinearRegression}

    params = {
        "encoder": {
            "encoder_1": "DataEncoding",
        },
        "imputer": {
            "imputer_1": "SimpleImputer",
            "SimpleImputer_method": "mean",
        },
        "balancing": {"balancing_1": "no_processing"},
        "scaling": {"scaling_2": "Standardize"},
        "feature_selection": {"feature_selection_1": "no_processing"},
        "model": {
            "model_13": "LinearRegression",
        },
    }

    clf = TabularObjective(
        params,
    )
    clf.setup(
        params,
        _X=data[features],
        _y=data[response],
        encoder=encoder,
        imputer=imputer,
        balancing=balancing,
        scaling=scaling,
        feature_selection=feature_selection,
        models=models,
        model_name="obj_3",
        task_mode="regression",
        objective="MAE",
        validation=True,
        valid_size=0.15,
        full_status=False,
        reset_index=True,
        _iter=1,
        seed=1,
    )
    result = clf.step()
    clf.reset_config(params)

    assert isinstance(result, dict), "Objective function should return a dict."
    assert "loss" in result.keys(), "Objective function should return loss."
    assert (
        "fitted_model" in result.keys()
    ), "Objective function should return fitted model."
    assert (
        "training_status" in result.keys()
    ), "Objective function should return training status."


def test_objective_4():

    from My_AutoML._hpo._utils import TabularObjective
    from My_AutoML._encoding import DataEncoding
    from My_AutoML._imputation import SimpleImputer
    from My_AutoML._base import no_processing
    from My_AutoML._scaling import Standardize
    from My_AutoML._model import LinearRegression

    # test load_data here
    data = load_data().load("example/example_data", "insurance")
    data = data["insurance"]

    features = list(data.columns)
    features.remove("expenses")
    response = ["expenses"]

    encoder = {"DataEncoding": DataEncoding}
    imputer = {"SimpleImputer": SimpleImputer}
    balancing = {"no_processing": no_processing}
    scaling = {"Standardize": Standardize}
    feature_selection = {"no_processing": no_processing}
    models = {"LinearRegression": LinearRegression}

    params = {
        "encoder": {
            "encoder_1": "DataEncoding",
        },
        "imputer": {
            "imputer_1": "SimpleImputer",
            "SimpleImputer_method": "mean",
        },
        "balancing": {"balancing_1": "no_processing"},
        "scaling": {"scaling_2": "Standardize"},
        "feature_selection": {"feature_selection_1": "no_processing"},
        "model": {
            "model_13": "LinearRegression",
        },
    }

    clf = TabularObjective(
        params,
    )
    clf.setup(
        params,
        _X=data[features],
        _y=data[response],
        encoder=encoder,
        imputer=imputer,
        balancing=balancing,
        scaling=scaling,
        feature_selection=feature_selection,
        models=models,
        model_name="obj_4",
        task_mode="regression",
        objective="R2",
        validation=True,
        valid_size=0.15,
        full_status=True,
        reset_index=True,
        _iter=1,
        seed=1,
    )
    result = clf.step()
    clf.reset_config(params)

    assert isinstance(result, dict), "Objective function should return a dict."
    assert "loss" in result.keys(), "Objective function should return loss."
    assert (
        "fitted_model" in result.keys()
    ), "Objective function should return fitted model."
    assert (
        "training_status" in result.keys()
    ), "Objective function should return training status."


def test_objective_5():

    from My_AutoML._hpo._utils import TabularObjective
    from My_AutoML._encoding import DataEncoding
    from My_AutoML._imputation import SimpleImputer
    from My_AutoML._base import no_processing
    from My_AutoML._scaling import Standardize
    from My_AutoML._model import LogisticRegression

    # test load_data here
    data = load_data().load("example/example_data", "heart")
    data = data["heart"]

    features = list(data.columns)
    features.remove("HeartDisease")
    response = ["HeartDisease"]

    encoder = {"DataEncoding": DataEncoding}
    imputer = {"SimpleImputer": SimpleImputer}
    balancing = {"no_processing": no_processing}
    scaling = {"Standardize": Standardize}
    feature_selection = {"no_processing": no_processing}
    models = {"LogisticRegression": LogisticRegression}

    params = {
        "encoder": {
            "encoder_1": "DataEncoding",
        },
        "imputer": {
            "imputer_1": "SimpleImputer",
            "SimpleImputer_method": "mean",
        },
        "balancing": {"balancing_1": "no_processing"},
        "scaling": {"scaling_2": "Standardize"},
        "feature_selection": {"feature_selection_1": "no_processing"},
        "model": {
            "model_17": "LogisticRegression",
            "LogisticRegression_penalty": "l2",
            "LogisticRegression_tol": 1e-4,
            "LogisticRegression_C": 1,
        },
    }

    clf = TabularObjective(
        params,
    )
    clf.setup(
        params,
        _X=data[features],
        _y=data[response],
        encoder=encoder,
        imputer=imputer,
        balancing=balancing,
        scaling=scaling,
        feature_selection=feature_selection,
        models=models,
        model_name="obj_5",
        task_mode="classification",
        objective="precision",
        validation=True,
        valid_size=0.15,
        full_status=False,
        reset_index=True,
        _iter=1,
        seed=1,
    )
    result = clf.step()
    clf.reset_config(params)

    assert isinstance(result, dict), "Objective function should return a dict."
    assert "loss" in result.keys(), "Objective function should return loss."
    assert (
        "fitted_model" in result.keys()
    ), "Objective function should return fitted model."
    assert (
        "training_status" in result.keys()
    ), "Objective function should return training status."


def test_objective_6():

    from My_AutoML._hpo._utils import TabularObjective
    from My_AutoML._encoding import DataEncoding
    from My_AutoML._imputation import SimpleImputer
    from My_AutoML._base import no_processing
    from My_AutoML._scaling import Standardize
    from My_AutoML._model import LogisticRegression

    # test load_data here
    data = load_data().load("example/example_data", "heart")
    data = data["heart"]

    features = list(data.columns)
    features.remove("HeartDisease")
    response = ["HeartDisease"]

    encoder = {"DataEncoding": DataEncoding}
    imputer = {"SimpleImputer": SimpleImputer}
    balancing = {"no_processing": no_processing}
    scaling = {"Standardize": Standardize}
    feature_selection = {"no_processing": no_processing}
    models = {"LogisticRegression": LogisticRegression}

    params = {
        "encoder": {
            "encoder_1": "DataEncoding",
        },
        "imputer": {
            "imputer_1": "SimpleImputer",
            "SimpleImputer_method": "mean",
        },
        "balancing": {"balancing_1": "no_processing"},
        "scaling": {"scaling_2": "Standardize"},
        "feature_selection": {"feature_selection_1": "no_processing"},
        "model": {
            "model_17": "LogisticRegression",
            "LogisticRegression_penalty": "l2",
            "LogisticRegression_tol": 1e-4,
            "LogisticRegression_C": 1,
        },
    }

    clf = TabularObjective(
        params,
    )
    clf.setup(
        params,
        _X=data[features],
        _y=data[response],
        encoder=encoder,
        imputer=imputer,
        balancing=balancing,
        scaling=scaling,
        feature_selection=feature_selection,
        models=models,
        model_name="obj_6",
        task_mode="classification",
        objective="hinge",
        validation=False,
        valid_size=0.15,
        full_status=False,
        reset_index=True,
        _iter=1,
        seed=1,
    )
    result = clf.step()
    clf.reset_config(params)

    assert isinstance(result, dict), "Objective function should return a dict."
    assert "loss" in result.keys(), "Objective function should return loss."
    assert (
        "fitted_model" in result.keys()
    ), "Objective function should return fitted model."
    assert (
        "training_status" in result.keys()
    ), "Objective function should return training status."


def test_objective_7():

    from My_AutoML._hpo._utils import TabularObjective
    from My_AutoML._encoding import DataEncoding
    from My_AutoML._imputation import SimpleImputer
    from My_AutoML._base import no_processing
    from My_AutoML._scaling import Standardize
    from My_AutoML._model import LogisticRegression

    # test load_data here
    data = load_data().load("example/example_data", "heart")
    data = data["heart"]

    features = list(data.columns)
    features.remove("HeartDisease")
    response = ["HeartDisease"]

    encoder = {"DataEncoding": DataEncoding}
    imputer = {"SimpleImputer": SimpleImputer}
    balancing = {"no_processing": no_processing}
    scaling = {"Standardize": Standardize}
    feature_selection = {"no_processing": no_processing}
    models = {"LogisticRegression": LogisticRegression}

    params = {
        "encoder": {
            "encoder_1": "DataEncoding",
        },
        "imputer": {
            "imputer_1": "SimpleImputer",
            "SimpleImputer_method": "mean",
        },
        "balancing": {"balancing_1": "no_processing"},
        "scaling": {"scaling_2": "Standardize"},
        "feature_selection": {"feature_selection_1": "no_processing"},
        "model": {
            "model_17": "LogisticRegression",
            "LogisticRegression_penalty": "l2",
            "LogisticRegression_tol": 1e-4,
            "LogisticRegression_C": 1,
        },
    }

    clf = TabularObjective(
        params,
    )
    clf.setup(
        params,
        _X=data[features],
        _y=data[response],
        encoder=encoder,
        imputer=imputer,
        balancing=balancing,
        scaling=scaling,
        feature_selection=feature_selection,
        models=models,
        model_name="obj_7",
        task_mode="classification",
        objective="f1",
        validation=False,
        valid_size=0.15,
        full_status=False,
        reset_index=True,
        _iter=1,
        seed=1,
    )
    result = clf.step()
    clf.reset_config(params)

    assert isinstance(result, dict), "Objective function should return a dict."
    assert "loss" in result.keys(), "Objective function should return loss."
    assert (
        "fitted_model" in result.keys()
    ), "Objective function should return fitted model."
    assert (
        "training_status" in result.keys()
    ), "Objective function should return training status."


def test_objective_8():

    from My_AutoML._hpo._utils import TabularObjective
    from My_AutoML._encoding import DataEncoding
    from My_AutoML._imputation import SimpleImputer
    from My_AutoML._base import no_processing
    from My_AutoML._scaling import Standardize
    from My_AutoML._model import LinearRegression

    # test load_data here
    data = load_data().load("example/example_data", "insurance")
    data = data["insurance"]

    features = list(data.columns)
    features.remove("expenses")
    response = ["expenses"]

    encoder = {"DataEncoding": DataEncoding}
    imputer = {"SimpleImputer": SimpleImputer}
    balancing = {"no_processing": no_processing}
    scaling = {"Standardize": Standardize}
    feature_selection = {"no_processing": no_processing}
    models = {"LinearRegression": LinearRegression}

    params = {
        "encoder": {
            "encoder_1": "DataEncoding",
        },
        "imputer": {
            "imputer_1": "SimpleImputer",
            "SimpleImputer_method": "mean",
        },
        "balancing": {"balancing_1": "no_processing"},
        "scaling": {"scaling_2": "Standardize"},
        "feature_selection": {"feature_selection_1": "no_processing"},
        "model": {
            "model_13": "LinearRegression",
        },
    }

    clf = TabularObjective(
        params,
    )
    clf.setup(
        params,
        _X=data[features],
        _y=data[response],
        encoder=encoder,
        imputer=imputer,
        balancing=balancing,
        scaling=scaling,
        feature_selection=feature_selection,
        models=models,
        model_name="obj_8",
        task_mode="regression",
        objective="MSE",
        validation=False,
        valid_size=0.15,
        full_status=False,
        reset_index=True,
        _iter=1,
        seed=1,
    )
    result = clf.step()
    clf.reset_config(params)

    assert isinstance(result, dict), "Objective function should return a dict."
    assert "loss" in result.keys(), "Objective function should return loss."
    assert (
        "fitted_model" in result.keys()
    ), "Objective function should return fitted model."
    assert (
        "training_status" in result.keys()
    ), "Objective function should return training status."


def test_objective_9():

    from My_AutoML._hpo._utils import TabularObjective
    from My_AutoML._encoding import DataEncoding
    from My_AutoML._imputation import SimpleImputer
    from My_AutoML._base import no_processing
    from My_AutoML._scaling import Standardize
    from My_AutoML._model import LinearRegression

    # test load_data here
    data = load_data().load("example/example_data", "insurance")
    data = data["insurance"]

    features = list(data.columns)
    features.remove("expenses")
    response = ["expenses"]

    encoder = {"DataEncoding": DataEncoding}
    imputer = {"SimpleImputer": SimpleImputer}
    balancing = {"no_processing": no_processing}
    scaling = {"Standardize": Standardize}
    feature_selection = {"no_processing": no_processing}
    models = {"LinearRegression": LinearRegression}

    params = {
        "encoder": {
            "encoder_1": "DataEncoding",
        },
        "imputer": {
            "imputer_1": "SimpleImputer",
            "SimpleImputer_method": "mean",
        },
        "balancing": {"balancing_1": "no_processing"},
        "scaling": {"scaling_2": "Standardize"},
        "feature_selection": {"feature_selection_1": "no_processing"},
        "model": {
            "model_13": "LinearRegression",
        },
    }

    clf = TabularObjective(
        params,
    )
    clf.setup(
        params,
        _X=data[features],
        _y=data[response],
        encoder=encoder,
        imputer=imputer,
        balancing=balancing,
        scaling=scaling,
        feature_selection=feature_selection,
        models=models,
        model_name="obj_9",
        task_mode="regression",
        objective="MAX",
        validation=True,
        valid_size=0.15,
        full_status=True,
        reset_index=True,
        _iter=1,
        seed=1,
    )
    result = clf.step()
    clf.reset_config(params)

    assert isinstance(result, dict), "Objective function should return a dict."
    assert "loss" in result.keys(), "Objective function should return loss."
    assert (
        "fitted_model" in result.keys()
    ), "Objective function should return fitted model."
    assert (
        "training_status" in result.keys()
    ), "Objective function should return training status."


def test_heart():

    # test load_data here
    data = load_data().load("example/example_data", "heart")
    data = data["heart"]

    features = list(data.columns)
    features.remove("HeartDisease")
    response = ["HeartDisease"]

    mol = My_AutoML.AutoTabular(
        model_name="heart",
        search_algo="GridSearch",
        timeout=60,
    )
    mol.fit(data[features], data[response])

    y_pred = mol.predict(data[features])

    assert (
        os.path.exists("tmp/heart/init.txt") == True
    ), "Classification for Heart data failed to initiated."
    assert mol._fitted == True, "Classification for Heart data failed to fit."
    assert (
        os.path.exists("tmp/heart/optimal_setting.txt") == True
    ), "Classification for Heart data failed to find optimal setting."


def test_insurance():

    # test load_data here
    data = load_data().load("example/example_data", "insurance")
    data = data["insurance"]

    features = list(data.columns)
    features.remove("expenses")
    response = ["expenses"]

    mol = My_AutoML.AutoTabular(
        model_name="insurance",
        objective="MAE",
        timeout=60,
    )
    mol.fit(data[features], data[response])
    y_pred = mol.predict(data[features])

    assert (
        os.path.exists("tmp/insurance/init.txt") == True
    ), "Regression for Insurance data failed to initiated."
    assert mol._fitted == True, "Regression for Insurance data failed to fit."
    assert (
        os.path.exists("tmp/insurance/optimal_setting.txt") == True
    ), "Regression for Insurance data failed to find optimal setting."


def test_insurance_R2():

    from My_AutoML._hpo._base import AutoTabularBase

    # test load_data here
    data = load_data().load("example/example_data", "insurance")
    data = data["insurance"]

    features = list(data.columns)
    features.remove("expenses")
    response = ["expenses"]

    mol = AutoTabularBase(
        model_name="insurance_R2",
        task_mode="regression",
        objective="R2",
        max_evals=8,
        timeout=60,
    )
    mol.fit(data[features], data[response])

    assert (
        os.path.exists("tmp/insurance_R2/init.txt") == True
    ), "Regression for Insurance data failed to initiated."
    assert mol._fitted == True, "Regression for Insurance data failed to fit."
    assert (
        os.path.exists("tmp/insurance_R2/optimal_setting.txt") == True
    ), "Regression for Insurance data failed to find optimal setting."


def test_stroke_import_version():

    # test load_data here
    data = load_data().load("Appendix", "healthcare-dataset-stroke-data")
    data = data["healthcare-dataset-stroke-data"]

    features = list(data.columns)
    features.remove("stroke")
    response = ["stroke"]

    mol = My_AutoML.AutoTabular(
        model_name="stroke",
        objective="auc",
        timeout=60,
    )
    mol.fit(data[features], data[response])

    assert (
        os.path.exists("tmp/stroke/init.txt") == True
    ), "Classification for Stroke data (import_version) failed to initiated."
    assert (
        mol._fitted == True
    ), "Classification for Stroke data (import_version) failed to fit."
    assert (
        os.path.exists("tmp/stroke/optimal_setting.txt") == True
    ), "Classification for Stroke data (import_version) failed to find optimal setting."


def test_stroke_loading():

    # test load_data here
    data = load_data().load("Appendix", "healthcare-dataset-stroke-data")
    data = data["healthcare-dataset-stroke-data"]

    features = list(data.columns)
    features.remove("stroke")
    response = ["stroke"]

    mol = My_AutoML.AutoTabular(
        model_name="stroke",
        timeout=60,
    )
    mol.fit(data[features], data[response])

    assert mol._fitted == True, "AutoTabular with loading failed to fit."


def test_stroke_with_limit():

    # test load_data here
    data = load_data().load("Appendix", "healthcare-dataset-stroke-data")
    data = data["healthcare-dataset-stroke-data"]

    features = list(data.columns)
    features.remove("stroke")
    response = ["stroke"]

    mol = My_AutoML.AutoTabular(
        model_name="no_valid",
        encoder=["DataEncoding"],
        imputer=["SimpleImputer"],
        balancing=["no_processing"],
        scaling=["no_processing"],
        feature_selection=["no_processing"],
        models=["DecisionTree"],
        validation=False,
        search_algo="GridSearch",
        objective="precision",
        timeout=60,
    )
    mol.fit(data[features], data[response])

    assert mol._fitted == True, "AutoTabular with limited space failed to fit."


def test_single():

    # test load_data here
    data = load_data().load("example/example_data", "insurance")
    data = data["insurance"]

    features = list(data.columns)
    features.remove("expenses")
    response = ["expenses"]

    mol = My_AutoML.AutoTabular(
        n_estimators=1,
        model_name="insurance",
        objective="MAE",
        timeout=60,
    )
    mol.fit(data[features], data[response])
    y_pred = mol.predict(data[features])

    assert (
        os.path.exists("tmp/insurance/init.txt") == True
    ), "Regression for Insurance data failed to initiated."
    assert mol._fitted == True, "Regression for Insurance data failed to fit."
    assert (
        os.path.exists("tmp/insurance/optimal_setting.txt") == True
    ), "Regression for Insurance data failed to find optimal setting."


def test_bagging():

    # test load_data here
    data = load_data().load("example/example_data", "insurance")
    data = data["insurance"]

    features = list(data.columns)
    features.remove("expenses")
    response = ["expenses"]

    mol = My_AutoML.AutoTabular(
        ensemble_strategy="bagging",
        model_name="insurance",
        objective="MAE",
        timeout=60,
    )
    mol.fit(data[features], data[response])
    y_pred = mol.predict(data[features])

    assert (
        os.path.exists("tmp/insurance/init.txt") == True
    ), "Regression for Insurance data failed to initiated."
    assert mol._fitted == True, "Regression for Insurance data failed to fit."
    assert (
        os.path.exists("tmp/insurance/optimal_setting.txt") == True
    ), "Regression for Insurance data failed to find optimal setting."


def test_boosting():

    # test load_data here
    data = load_data().load("example/example_data", "insurance")
    data = data["insurance"]

    features = list(data.columns)
    features.remove("expenses")
    response = ["expenses"]

    mol = My_AutoML.AutoTabular(
        ensemble_strategy="boosting",
        model_name="insurance",
        objective="MAE",
        timeout=60,
    )
    mol.fit(data[features], data[response])
    y_pred = mol.predict(data[features])

    assert (
        os.path.exists("tmp/insurance/init.txt") == True
    ), "Regression for Insurance data failed to initiated."
    assert mol._fitted == True, "Regression for Insurance data failed to fit."
    assert (
        os.path.exists("tmp/insurance/optimal_setting.txt") == True
    ), "Regression for Insurance data failed to find optimal setting."
