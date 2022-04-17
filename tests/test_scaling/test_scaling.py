"""
File: test_scaling.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Latest Version: 0.2.0
Relative Path: /tests/test_scaling/test_scaling.py
File Created: Saturday, 9th April 2022 1:56:15 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Saturday, 16th April 2022 11:57:34 pm
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

import unittest
import numpy as np
import pandas as pd

from My_AutoML._scaling import scalings

data_X = pd.DataFrame(
    {
        "col_1": [3, 8, 6, 7, 9, 9, 8, 8, 7, 5],
        "col_2": [9, 7, 2, 1, 6, 8, 8, 9, 3, 6],
    }
)
data_y = pd.DataFrame({"col_3": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]})


class TestScaling(unittest.TestCase):
    def test_Scaling(self):

        self.method_dict = scalings
        self.method_names = list(self.method_dict.keys())
        self.method_objects = list(self.method_dict.values())

        for method_name, method_object in zip(self.method_names, self.method_objects):

            mol = method_object()
            scaled_X = mol.fit_transform(data_X, data_y)

            # check whether the method is fitted
            self.assertEqual(
                mol._fitted,
                True,
                "The fit_transform method {} is not correctly fitted.".format(
                    method_name
                ),
            )

            if method_name != "Winsorization":
                mol.inverse_transform(scaled_X)

                self.assertEqual(
                    mol._fitted,
                    False,
                    "The inverse method {} is not correctly fitted.".format(
                        method_name
                    ),
                )


# test decrepted methods
def test_feature_manipulation():

    from My_AutoML._scaling import Feature_Manipulation

    data = pd.DataFrame(
        np.arange(15).reshape(5, 3), columns=["column_" + str(i + 1) for i in range(3)]
    )
    transformed_data = Feature_Manipulation(
        data,
        columns=["column_1", "column_2", "column_3"],
        manipulation=["* 100", "ln", "+ 1"],
        rename_columns={"column_2": "log_column_2"},
    )

    target_data = data.copy(deep=True)
    target_data["column_1_* 100"] = data["column_1"] * 100
    target_data["log_column_2"] = np.log(data["column_2"])
    target_data["column_3_+ 1"] = data["column_3"] + 1

    assert (
        (transformed_data == target_data).all().all()
    ), "The feature manipulation is not correctly done."


def test_feature_truncation():

    from My_AutoML._scaling import Feature_Truncation

    data = pd.DataFrame(
        np.random.randint(0, 100, size=(100, 10)),
        columns=["column_" + str(i) for i in range(10)],
    )

    transformer = Feature_Truncation(
        quantile=[0.2 * np.random.random() + 0.8 for _ in range(10)],
    )

    transformer.fit(data)
    capped_data = transformer.transform(data)

    assert (
        (capped_data <= transformer.quantile_list).all().all()
    ), "The feature truncation is not correctly done."


def test_get_algo():

    from My_AutoML._utils._optimize import get_algo

    get_algo("GridSearch")
    get_algo("HyperOpt")
    get_algo("Repeater")
    get_algo("ConcurrencyLimtiter")

    assert True, "The get_algo method is not correctly done."


def test_get_scheduler():

    from My_AutoML._utils._optimize import get_scheduler

    get_scheduler("FIFOScheduler")
    get_scheduler("ASHAScheduler")
    get_scheduler("HyperBandScheduler")
    get_scheduler("MeidanStoppingRule")
    get_scheduler("PopulationBasedScheduler")

    assert True, "The get_scheduler method is not correctly done."


def test_get_progress_reporter():

    from My_AutoML._utils._optimize import get_progress_reporter

    get_progress_reporter("CLIReporter", max_evals=64, max_error=4)
    get_progress_reporter("JupyterNotebookReporter", max_evals=64, max_error=4)


def test_get_logger():

    from My_AutoML._utils._optimize import get_logger

    get_logger(["Logger", "TBX", "JSON", "CSV", "MLflow"])
