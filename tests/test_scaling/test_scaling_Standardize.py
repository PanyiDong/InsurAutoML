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
Last Modified: Saturday, 9th April 2022 2:15:15 pm
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

from My_AutoML._scaling import (
    Standardize,
)

data = pd.DataFrame(
    {
        "col_1": [3, 8, 6, 7, 9, 9, 8, 8, 7, 5],
        "col_2": [9, 7, 2, 1, 6, 8, 8, 9, 3, 6],
    }
)

expected_data = pd.DataFrame(
    {
        "col_1": [
            -2.12132,
            0.53033,
            -0.53033,
            0.00000,
            1.06066,
            1.06066,
            0.53033,
            0.53033,
            0.00000,
            -1.06066,
        ],
        "col_2": [
            1.060522,
            0.376314,
            -1.334205,
            -1.676309,
            0.034210,
            0.718418,
            0.718418,
            1.060522,
            -0.992101,
            0.034210,
        ],
    }
)
expected_mean = np.array([7.0, 5.9])
expected_std = np.array([1.8856180831641267, 2.9230881691191666])


class TestStandardize(unittest.TestCase):
    def setUp(self):

        self.method = Standardize()

    def test_data(self):

        fit_data = self.method.fit_transform(data)

        # check whether the mean and std are correct
        self.assertEqual(
            (self.method._mean - expected_mean < 1e-5).all(),
            True,
            "Mean should be {}, get {}.".format(self.method._mean, expected_mean),
        )

        self.assertEqual(
            (self.method._std - expected_std < 1e-5).all(),
            True,
            "Mean should be {}, get {}.".format(self.method._std, expected_std),
        )

        # check whether the data is transformed correctly
        self.assertEqual(
            (fit_data - expected_data < 1e-5).all().all(),
            True,
            "Transformed data should be {}, get {}.".format(expected_data, fit_data),
        )
