"""
File Name: test_hpo_2.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.5
Relative Path: /tests/test_hpo/test_hpo_2.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Saturday, 16th December 2023 7:14:06 pm
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

from InsurAutoML import load_data


def test_objective_6():
    from InsurAutoML.hpo.utils import TabularObjective
    from InsurAutoML.encoding import DataEncoding
    from InsurAutoML.imputation import SimpleImputer
    from InsurAutoML.base import no_processing
    from InsurAutoML.scaling import Standardize
    from InsurAutoML.model import LogisticRegression

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
        data_split=[
            [(data[features], data[response]), (data[features], data[response])]
        ],
        encoder=encoder,
        imputer=imputer,
        balancing=balancing,
        scaling=scaling,
        feature_selection=feature_selection,
        models=models,
        model_name="obj_6",
        task_mode="classification",
        objective="hinge",
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
    from InsurAutoML.hpo.utils import TabularObjective
    from InsurAutoML.encoding import DataEncoding
    from InsurAutoML.imputation import SimpleImputer
    from InsurAutoML.base import no_processing
    from InsurAutoML.scaling import Standardize
    from InsurAutoML.model import LogisticRegression

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
        data_split=[
            [(data[features], data[response]), (data[features], data[response])]
        ],
        encoder=encoder,
        imputer=imputer,
        balancing=balancing,
        scaling=scaling,
        feature_selection=feature_selection,
        models=models,
        model_name="obj_7",
        task_mode="classification",
        objective="f1",
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
    from InsurAutoML.hpo.utils import TabularObjective
    from InsurAutoML.encoding import DataEncoding
    from InsurAutoML.imputation import SimpleImputer
    from InsurAutoML.base import no_processing
    from InsurAutoML.scaling import Standardize
    from InsurAutoML.model import LinearRegression

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
        data_split=[
            [(data[features], data[response]), (data[features], data[response])]
        ],
        encoder=encoder,
        imputer=imputer,
        balancing=balancing,
        scaling=scaling,
        feature_selection=feature_selection,
        models=models,
        model_name="obj_8",
        task_mode="regression",
        objective="MSE",
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
    from InsurAutoML.hpo.utils import TabularObjective
    from InsurAutoML.encoding import DataEncoding
    from InsurAutoML.imputation import SimpleImputer
    from InsurAutoML.base import no_processing
    from InsurAutoML.scaling import Standardize
    from InsurAutoML.model import LinearRegression

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
        data_split=[
            [(data[features], data[response]), (data[features], data[response])]
        ],
        encoder=encoder,
        imputer=imputer,
        balancing=balancing,
        scaling=scaling,
        feature_selection=feature_selection,
        models=models,
        model_name="obj_9",
        task_mode="regression",
        objective="MAX",
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


def test_best_results_only_fitted_trials():
    # timed-out trials (no saved pipeline) can tie with fitted trials whose loss
    # is infinite (e.g. a diverged model): only fitted trials may be selected
    import numpy as np
    import pandas as pd

    from InsurAutoML.hpo.base import AutoTabularBase

    fit_results = pd.DataFrame(
        {
            "training_status": ["TIMEOUT", "FITTED", "TIMEOUT", "FITTED", None],
            "loss": [np.inf, np.inf, np.inf, -0.5, np.nan],
            "path": ["timeout_1", "fitted_inf", "timeout_2", "fitted", "error"],
        }
    )

    best = AutoTabularBase._best_results(fit_results, 5)

    assert list(best["path"]) == ["fitted", "fitted_inf"]
    assert list(AutoTabularBase._best_results(fit_results, 1)["path"]) == ["fitted"]


def test_objective_timeout_keeps_fitted_result(tmp_path, monkeypatch):
    # a later step of a trial timing out keeps the fitted result of the earlier
    # step (its pipeline is saved) and stops the trial; a new trial on a reused
    # actor does not inherit it
    import func_timeout
    import numpy as np

    from InsurAutoML.base import no_processing
    from InsurAutoML.encoding import DataEncoding
    from InsurAutoML.hpo import utils as hpo_utils
    from InsurAutoML.hpo.utils import TabularObjective
    from InsurAutoML.imputation import SimpleImputer
    from InsurAutoML.model import LinearRegression

    monkeypatch.chdir(tmp_path)
    data = load_data().load(
        os.path.join(os.path.dirname(__file__), "..", "..", "example", "example_data"),
        "insurance",
    )["insurance"]
    features = [col for col in data.columns if col != "expenses"]

    params = {
        "encoder": {"encoder_1": "DataEncoding"},
        "imputer": {"imputer_1": "SimpleImputer", "SimpleImputer_method": "mean"},
        "balancing": {"balancing_1": "no_processing"},
        "scaling": {"scaling_1": "no_processing"},
        "feature_selection": {"feature_selection_1": "no_processing"},
        "model": {"model_1": "LinearRegression"},
    }
    trainable = TabularObjective.__new__(TabularObjective)
    trainable.setup(
        params,
        data_split=[
            [(data[features], data[["expenses"]]), (data[features], data[["expenses"]])]
        ],
        encoder={"DataEncoding": DataEncoding},
        imputer={"SimpleImputer": SimpleImputer},
        balancing={"no_processing": no_processing},
        scaling={"no_processing": no_processing},
        feature_selection={"no_processing": no_processing},
        models={"LinearRegression": LinearRegression},
        model_name="obj_timeout",
        task_mode="regression",
        objective="MAE",
        seed=1,
    )
    fitted = trainable.step()
    assert fitted["training_status"] == "FITTED"

    def _timed_out(timeout, func, *args, **kwargs):
        raise func_timeout.FunctionTimedOut()

    monkeypatch.setattr(hpo_utils.func_timeout, "func_timeout", _timed_out)

    result = trainable.step()
    assert result["training_status"] == "FITTED"
    assert result["loss"] == fitted["loss"]
    assert result["done"] is True
    assert (tmp_path / "obj_timeout").exists()

    # reused actor: the next trial times out without a fitted step of its own
    trainable.reset_config(params)
    result = trainable.step()
    assert result["training_status"] == "TIMEOUT"
    assert result["loss"] == np.inf
    assert "done" not in result
