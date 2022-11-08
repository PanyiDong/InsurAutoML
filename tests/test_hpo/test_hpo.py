"""
File: test_hpo.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /tests/test_hpo/test_hpo.py
File: test_hpo.py
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 7th November 2022 11:43:00 pm
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

from InsurAutoML import load_data

# use command line interaction to run the model
# apparently, same class object called in one test case will not be able
# to run the model correctly after the first time
# detect whether optimal setting exists as method of determining whether
# the model is fitted correctly


def test_objective_1():

    from InsurAutoML._hpo._utils import TabularObjective
    from InsurAutoML._encoding import DataEncoding
    from InsurAutoML._imputation import SimpleImputer
    from InsurAutoML._base import no_processing
    from InsurAutoML._scaling import Standardize
    from InsurAutoML._model import LogisticRegression

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

    from InsurAutoML._hpo._utils import TabularObjective
    from InsurAutoML._encoding import DataEncoding
    from InsurAutoML._imputation import SimpleImputer
    from InsurAutoML._base import no_processing
    from InsurAutoML._scaling import Standardize
    from InsurAutoML._model import LogisticRegression

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

    from InsurAutoML._hpo._utils import TabularObjective
    from InsurAutoML._encoding import DataEncoding
    from InsurAutoML._imputation import SimpleImputer
    from InsurAutoML._base import no_processing
    from InsurAutoML._scaling import Standardize
    from InsurAutoML._model import LinearRegression

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

    from InsurAutoML._hpo._utils import TabularObjective
    from InsurAutoML._encoding import DataEncoding
    from InsurAutoML._imputation import SimpleImputer
    from InsurAutoML._base import no_processing
    from InsurAutoML._scaling import Standardize
    from InsurAutoML._model import LinearRegression

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

    from InsurAutoML._hpo._utils import TabularObjective
    from InsurAutoML._encoding import DataEncoding
    from InsurAutoML._imputation import SimpleImputer
    from InsurAutoML._base import no_processing
    from InsurAutoML._scaling import Standardize
    from InsurAutoML._model import LogisticRegression

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
        full_status=True,
        reset_index=True,
        _iter=1,
        seed=1,
    )
    result = clf.step()
    clf.save_checkpoint("tmp")
    clf.load_checkpoint("tmp")
    clf.reset_config(params)

    assert isinstance(result, dict), "Objective function should return a dict."
    assert "loss" in result.keys(), "Objective function should return loss."
    assert (
        "fitted_model" in result.keys()
    ), "Objective function should return fitted model."
    assert (
        "training_status" in result.keys()
    ), "Objective function should return training status."
