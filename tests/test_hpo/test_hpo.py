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
Last Modified: Thursday, 14th April 2022 11:14:03 am
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

import pandas as pd

# use command line interaction to run the model
# apparently, same class object called in one test case will not be able
# to run the model correctly after the first time
# detect whether optimal setting exists as method of determining whether
# the model is fitted correctly


def test_heart():

    from My_AutoML import AutoTabular

    data = pd.read_csv("example/example_data/heart.csv")
    response = "HeartDisease"
    features = list(data.columns)
    features.remove(response)
    response = [response]

    mol_heart = AutoTabular(model_name="heart")
    mol_heart.fit(data[features], data[response])

    assert (
        mol_heart._fitted == True
    ), "Classification for Heart data successfully fitted."


def test_stroke():

    from My_AutoML import AutoTabularClassifier

    data = pd.read_csv("Appendix/healthcare-dataset-stroke-data.csv")
    response = "stroke"
    features = list(data.columns)
    features.remove(response)
    response = [response]

    mol_stroke = AutoTabularClassifier(model_name="healthcare-dataset-stroke-data")
    mol_stroke.fit(data[features], data[response])

    assert (
        mol_stroke._fitted == True
    ), "Classification for Stroke data successfully fitted."


def test_insurance():

    from My_AutoML import AutoTabularRegressor

    data = pd.read_csv("example/example_data/insurance.csv")
    response = "expenses"
    features = list(data.columns)
    features.remove(response)
    response = [response]

    mol_insurance = AutoTabularRegressor(model_name="insurance")
    mol_insurance.fit(data[features], data[response])

    assert (
        mol_insurance._fitted == True
    ), "Regression for Insurance data successfully fitted."
