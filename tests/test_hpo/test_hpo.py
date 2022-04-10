"""
File: test_hpo.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Latest Version: 0.2.0
Relative Path: /tests/test_hpo/test_hpo.py
File Created: Sunday, 10th April 2022 12:26:37 am
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Sunday, 10th April 2022 12:34:11 am
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

from My_AutoML import AutoTabular

# read classification data
heart_data = pd.DataFrame("example/example_data/heart.csv")

features = list(heart_data.columns)
features.remove("HeartDisease")
response = ["HeartDisease"]

class_X = heart_data[features]
class_y = heart_data[response]

# read regression data
insurance_data = pd.DataFrame("example/example_data/insurance.csv")

features = list(insurance_data.columns)
features.remove("expenses")
response = ["expenses"]

reg_X = insurance_data[features]
reg_y = insurance_data[response]


class TestHPO(unittest.TestCase):
    def setUp(self):
        self.mol = AutoTabular

    def test_upper(self):

        self.reg = self.mol()
        self.reg.fit(class_X, class_y)

        self.assertEqual(self.reg._fitted, True, "Classification successfully fitted.")

        self.clf = self.mol()
        self.clf.fit(reg_X, reg_y)

        self.assertEqual(self.clf._fitted, True, "Regression successfully fitted.")
