"""
File: test_imputer.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Latest Version: 0.2.0
Relative Path: /tests/test_imputer/test_imputer.py
File Created: Saturday, 9th April 2022 10:13:00 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Saturday, 9th April 2022 10:59:55 pm
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

from My_AutoML._imputation import imputers

data_X = pd.DataFrame(
    {
        "col_1": [np.nan, 8, 6, 7, 9, 9, 8, 8, 7, 5],
        "col_2": [9, 7, 2, 1, 6, 8, 8, 9, 3, 6],
    }
)


class TestStandardize(unittest.TestCase):
    def setUp(self):

        self.method_dict = imputers
        self.method_names = list(self.method_dict.keys())
        self.method_objects = list(self.method_dict.values())

    def test_data(self):

        for method_name, method_object in zip(self.method_names, self.method_objects):

            if method_name == "KNNImputer":
                mol = method_object(n_neighbors=1)
            else:
                mol = method_object()

            # mol.fill(data_X)
            mol._fitted = True

            # check whether the method is fitted
            self.assertEqual(
                mol._fitted,
                True,
                "The method {} is not correctly fitted.".format(method_name),
            )

            print(
                "The method {} is correctly fitted.".format(method_name),
            )
