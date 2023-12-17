"""
File Name: test_balancing.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.5
Relative Path: /tests/test_balancing/test_balancing.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Saturday, 16th December 2023 5:33:08 pm
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

from InsurAutoML.balancing import balancings

data_X = pd.DataFrame(
    np.random.normal(0, 10, (100, 10)),
    columns=["col_" + str(i) for i in range(10)],
)
data_y = pd.DataFrame(
    [1 for _ in range(90)] + [0 for _ in range(10)], columns=["col_y"]
)


class TestScaling(unittest.TestCase):
    def test_Scaling(self):
        self.method_dict = balancings
        self.method_names = list(self.method_dict.keys())
        self.method_objects = list(self.method_dict.values())

        for method_name, method_object in zip(self.method_names, self.method_objects):
            if method_name != "no_processing":
                mol = method_object(imbalance_threshold=0.8)
                mol.fit_transform(data_X, data_y)

                # check whether the method is fitted
                self.assertEqual(
                    mol._fitted,
                    True,
                    "The method {} is not correctly fitted.".format(method_name),
                )
