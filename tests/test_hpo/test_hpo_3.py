"""
File: test_hpo_3.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /tests/test_hpo/test_hpo_3.py
File: test_hpo_3.py
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Wednesday, 16th November 2022 12:56:36 pm
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
import InsurAutoML
from InsurAutoML import load_data

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


def test_heart():

    # test load_data here
    data = load_data().load("example/example_data", "heart")
    data = data["heart"]

    features = list(data.columns)
    features.remove("HeartDisease")
    response = ["HeartDisease"]

    mol = InsurAutoML.AutoTabular(
        model_name="heart",
        search_algo="GridSearch",
        max_evals=5,
        timeout=60,
    )
    mol.fit(data[features], data[response])

    y_pred = mol.predict(data[features])

    assert (
        os.path.exists("tmp/heart/init.txt")
    ), "Classification for Heart data failed to initiated."
    assert mol._fitted, "Classification for Heart data failed to fit."
    assert (
        os.path.exists("tmp/heart/optimal_setting.txt")
    ), "Classification for Heart data failed to find optimal setting."


def test_insurance():

    # test load_data here
    data = load_data().load("example/example_data", "insurance")
    data = data["insurance"]

    features = list(data.columns)
    features.remove("expenses")
    response = ["expenses"]

    mol = InsurAutoML.AutoTabular(
        model_name="insurance",
        objective="MAE",
        max_evals=5,
        timeout=60,
    )
    mol.fit(data[features], data[response])
    y_pred = mol.predict(data[features])

    assert (
        os.path.exists("tmp/insurance/init.txt")
    ), "Regression for Insurance data failed to initiated."
    assert mol._fitted, "Regression for Insurance data failed to fit."
    assert (
        os.path.exists("tmp/insurance/optimal_setting.txt")
    ), "Regression for Insurance data failed to find optimal setting."


def test_insurance_R2():

    from InsurAutoML._hpo._base import AutoTabularBase

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
        os.path.exists("tmp/insurance_R2/init.txt")
    ), "Regression for Insurance data failed to initiated."
    assert mol._fitted, "Regression for Insurance data failed to fit."
    assert (
        os.path.exists("tmp/insurance_R2/optimal_setting.txt")
    ), "Regression for Insurance data failed to find optimal setting."


def test_stroke_import_version():

    # test load_data here
    data = load_data().load("Appendix", "healthcare-dataset-stroke-data")
    data = data["healthcare-dataset-stroke-data"]

    features = list(data.columns)
    features.remove("stroke")
    response = ["stroke"]

    mol = InsurAutoML.AutoTabular(
        model_name="stroke",
        objective="auc",
        max_evals=5,
        timeout=60,
    )
    mol.fit(data[features], data[response])

    assert (
        os.path.exists("tmp/stroke/init.txt")
    ), "Classification for Stroke data (import_version) failed to initiated."
    assert (
        mol._fitted
    ), "Classification for Stroke data (import_version) failed to fit."
    assert (
        os.path.exists("tmp/stroke/optimal_setting.txt")
    ), "Classification for Stroke data (import_version) failed to find optimal setting."

    # test loading
    mol = InsurAutoML.AutoTabular(
        model_name="stroke",
        max_evals=5,
        timeout=60,
    )
    mol.fit(data[features], data[response])

    assert mol._fitted, "AutoTabular with loading failed to fit."


# Update: Nov. 16, 2022
# loading combined with above test
# def test_stroke_loading():

#     # test load_data here
#     data = load_data().load("Appendix", "healthcare-dataset-stroke-data")
#     data = data["healthcare-dataset-stroke-data"]

#     features = list(data.columns)
#     features.remove("stroke")
#     response = ["stroke"]

#     mol = InsurAutoML.AutoTabular(
#         model_name="stroke",
#         timeout=60,
#     )
#     mol.fit(data[features], data[response])

#     assert mol._fitted == True, "AutoTabular with loading failed to fit."


def test_stroke_with_limit():

    # test load_data here
    data = load_data().load("Appendix", "healthcare-dataset-stroke-data")
    data = data["healthcare-dataset-stroke-data"]

    features = list(data.columns)
    features.remove("stroke")
    response = ["stroke"]

    mol = InsurAutoML.AutoTabular(
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
        max_evals=5,
        timeout=60,
    )
    mol.fit(data[features], data[response])

    assert mol._fitted, "AutoTabular with limited space failed to fit."


def test_single():

    # test load_data here
    data = load_data().load("example/example_data", "insurance")
    data = data["insurance"]

    features = list(data.columns)
    features.remove("expenses")
    response = ["expenses"]

    mol = InsurAutoML.AutoTabular(
        n_estimators=1,
        model_name="insurance_single",
        objective="MAE",
        max_evals=5,
        timeout=60,
    )
    mol.fit(data[features], data[response])
    y_pred = mol.predict(data[features])

    assert (
        os.path.exists("tmp/insurance_single/init.txt")
    ), "Regression for Insurance data failed to initiated."
    assert mol._fitted, "Regression for Insurance data failed to fit."
    assert (
        os.path.exists("tmp/insurance_single/optimal_setting.txt")
    ), "Regression for Insurance data failed to find optimal setting."


def test_stacking():

    # test load_data here
    data = load_data().load("example/example_data", "insurance")
    data = data["insurance"]

    features = list(data.columns)
    features.remove("expenses")
    response = ["expenses"]

    mol = InsurAutoML.AutoTabular(
        ensemble_strategy="stacking",
        model_name="insurance_stacking",
        objective="MAE",
        max_evals=5,
        timeout=60,
    )
    mol.fit(data[features], data[response])
    y_pred = mol.predict(data[features])

    assert (
        os.path.exists("tmp/insurance_stacking/init.txt")
    ), "Regression for Insurance data failed to initiated."
    assert mol._fitted, "Regression for Insurance data failed to fit."
    assert (
        os.path.exists("tmp/insurance_stacking/optimal_setting.txt")
    ), "Regression for Insurance data failed to find optimal setting."


def test_bagging():

    # test load_data here
    data = load_data().load("example/example_data", "insurance")
    data = data["insurance"]

    features = list(data.columns)
    features.remove("expenses")
    response = ["expenses"]

    mol = InsurAutoML.AutoTabular(
        ensemble_strategy="bagging",
        model_name="insurance_bagging",
        objective="MAE",
        max_evals=5,
        timeout=60,
    )
    mol.fit(data[features], data[response])
    y_pred = mol.predict(data[features])

    assert (
        os.path.exists("tmp/insurance_bagging/init.txt")
    ), "Regression for Insurance data failed to initiated."
    assert mol._fitted, "Regression for Insurance data failed to fit."
    assert (
        os.path.exists("tmp/insurance_bagging/optimal_setting.txt")
    ), "Regression for Insurance data failed to find optimal setting."


def test_boosting():

    # test load_data here
    data = load_data().load("example/example_data", "insurance")
    data = data["insurance"]

    features = list(data.columns)
    features.remove("expenses")
    response = ["expenses"]

    mol = InsurAutoML.AutoTabular(
        ensemble_strategy="boosting",
        model_name="insurance_boosting",
        objective="MAE",
        max_evals=5,
        timeout=60,
    )
    mol.fit(data[features], data[response])
    y_pred = mol.predict(data[features])

    assert (
        os.path.exists("tmp/insurance_boosting/init.txt")
    ), "Regression for Insurance data failed to initiated."
    assert mol._fitted, "Regression for Insurance data failed to fit."
    assert (
        os.path.exists("tmp/insurance_boosting/optimal_setting.txt")
    ), "Regression for Insurance data failed to find optimal setting."


def test_search_Optuna():

    # test load_data here
    data = load_data().load("example/example_data", "heart")
    data = data["heart"]

    features = list(data.columns)
    features.remove("HeartDisease")
    response = ["HeartDisease"]

    # mol = InsurAutoML.AutoTabular(
    #     model_name="heart_optuna_stacking",
    #     search_algo="Optuna",
    #     ensemble_strategy="stacking",
    #     max_evals = 5,
    #     timeout=60,
    # )
    # mol.fit(data[features], data[response])

    # y_pred = mol.predict(data[features])

    # assert (
    #     os.path.exists("tmp/heart_optuna_stacking/init.txt") == True
    # ), "Classification for Heart data failed to initiated."
    # assert mol._fitted == True, "Classification for Heart data failed to fit."
    # assert (
    #     os.path.exists("tmp/heart_optuna_stacking/optimal_setting.txt") == True
    # ), "Classification for Heart data failed to find optimal setting."

    mol = InsurAutoML.AutoTabular(
        model_name="heart_optuna_bagging",
        search_algo="Optuna",
        ensemble_strategy="bagging",
        max_evals=5,
        timeout=60,
    )
    mol.fit(data[features], data[response])

    y_pred = mol.predict(data[features])

    assert (
        os.path.exists("tmp/heart_optuna_bagging/init.txt")
    ), "Classification for Heart data failed to initiated."
    assert mol._fitted, "Classification for Heart data failed to fit."
    assert (
        os.path.exists("tmp/heart_optuna_bagging/optimal_setting.txt")
    ), "Classification for Heart data failed to find optimal setting."

    mol = InsurAutoML.AutoTabular(
        model_name="heart_optuna_boosting",
        search_algo="Optuna",
        ensemble_strategy="boosting",
        max_evals=5,
        timeout=60,
    )
    mol.fit(data[features], data[response])

    y_pred = mol.predict(data[features])

    assert (
        os.path.exists("tmp/heart_optuna_boosting/init.txt")
    ), "Classification for Heart data failed to initiated."
    assert mol._fitted, "Classification for Heart data failed to fit."
    assert (
        os.path.exists("tmp/heart_optuna_boosting/optimal_setting.txt")
    ), "Classification for Heart data failed to find optimal setting."

# def test_search_BlendSearch():

#     # test load_data here
#     data = load_data().load("example/example_data", "heart")
#     data = data["heart"]

#     features = list(data.columns)
#     features.remove("HeartDisease")
#     response = ["HeartDisease"]

#     mol = InsurAutoML.AutoTabular(
#         model_name="heart_blendsearch",
#         search_algo="BlendSearch",
#         ensemble_strategy="stacking",
#         timeout=60,
#     )
#     mol.fit(data[features], data[response])

#     y_pred = mol.predict(data[features])

#     assert (
#         os.path.exists("tmp/heart_blendsearch/init.txt") == True
#     ), "Classification for Heart data failed to initiated."
#     assert mol._fitted == True, "Classification for Heart data failed to fit."
#     assert (
#         os.path.exists("tmp/heart_blendsearch/optimal_setting.txt") == True
#     ), "Classification for Heart data failed to find optimal setting."


def test_search_CFO():

    # test load_data here
    data = load_data().load("example/example_data", "heart")
    data = data["heart"]

    features = list(data.columns)
    features.remove("HeartDisease")
    response = ["HeartDisease"]

    mol = InsurAutoML.AutoTabular(
        model_name="heart_cfo",
        search_algo="CFO",
        ensemble_strategy="stacking",
        max_evals=5,
        timeout=60,
    )
    mol.fit(data[features], data[response])

    y_pred = mol.predict(data[features])

    assert (
        os.path.exists("tmp/heart_cfo/init.txt")
    ), "Classification for Heart data failed to initiated."
    assert mol._fitted, "Classification for Heart data failed to fit."
    assert (
        os.path.exists("tmp/heart_cfo/optimal_setting.txt")
    ), "Classification for Heart data failed to find optimal setting."


def test_search_HyperOpt():

    # test load_data here
    data = load_data().load("example/example_data", "heart")
    data = data["heart"]

    features = list(data.columns)
    features.remove("HeartDisease")
    response = ["HeartDisease"]

    mol = InsurAutoML.AutoTabular(
        model_name="heart_hyperopt",
        search_algo="HyperOpt",
        ensemble_strategy="stacking",
        max_evals=5,
        timeout=60,
    )
    mol.fit(data[features], data[response])

    y_pred = mol.predict(data[features])

    assert (
        os.path.exists("tmp/heart_hyperopt/init.txt")
    ), "Classification for Heart data failed to initiated."
    assert mol._fitted, "Classification for Heart data failed to fit."
    assert (
        os.path.exists("tmp/heart_hyperopt/optimal_setting.txt")
    ), "Classification for Heart data failed to find optimal setting."


def test_search_Nevergrad():

    # test load_data here
    data = load_data().load("example/example_data", "heart")
    data = data["heart"]

    features = list(data.columns)
    features.remove("HeartDisease")
    response = ["HeartDisease"]

    mol = InsurAutoML.AutoTabular(
        model_name="heart_nevergrad",
        search_algo="Nevergrad",
        ensemble_strategy="stacking",
        max_evals=5,
        timeout=60,
    )
    mol.fit(data[features], data[response])

    y_pred = mol.predict(data[features])

    assert (
        os.path.exists("tmp/heart_nevergrad/init.txt")
    ), "Classification for Heart data failed to initiated."
    assert mol._fitted, "Classification for Heart data failed to fit."
    assert (
        os.path.exists("tmp/heart_nevergrad/optimal_setting.txt")
    ), "Classification for Heart data failed to find optimal setting."
