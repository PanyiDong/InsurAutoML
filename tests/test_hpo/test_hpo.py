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
Last Modified: Sunday, 10th April 2022 1:43:09 pm
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

import pytest
import pandas as pd

from My_AutoML import AutoTabular

# read regression data
insurance_data = pd.read_csv("example/example_data/insurance.csv")

insurance_features = list(insurance_data.columns)
insurance_features.remove("expenses")
insurance_response = ["expenses"]

# read classification data
heart_data = pd.read_csv("example/example_data/heart.csv")

heart_features = list(heart_data.columns)
heart_features.remove("HeartDisease")
heart_response = ["HeartDisease"]

data = [
    (
        "insurance",
        insurance_data[insurance_features],
        insurance_data[insurance_response],
    ),
    ("heart", heart_data[heart_features], heart_data[heart_response]),
]


@pytest.mark.parametrize(
    "model_name, reg_X, reg_y", data, ids=["regression", "classification"]
)
def test_regression(model_name, reg_X, reg_y):

    mol = AutoTabular(
        model_name=model_name,
    )

    mol.fit(reg_X, reg_y)

    assert mol._fitted == True, "Model successfully fitted."
