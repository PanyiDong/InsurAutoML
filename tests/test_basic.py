"""
File: basic.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Latest Version: 0.2.0
Relative Path: /tests/basic.py
File Created: Saturday, 9th April 2022 11:25:27 am
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Saturday, 9th April 2022 11:51:42 am
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

from My_AutoML._utils import get_missing_matrix

data = pd.DataFrame(
    {
        "col_1": [1, 2, np.nan, 4],
        "col_2": [np.nan, 3, np.nan, 5],
    }
)

expected_result = pd.DataFrame(
    {
        "col_1": [0, 0, 1, 0],
        "col_2": [1, 0, 1, 0],
    }
)


class TestGetMissingMatrix(unittest.TestCase):
    def setUp(self):
        self.die = get_missing_matrix

    def test_upper(self):
        self.assertEqual(
            (self.die(data) == expected_result.values).all(),
            True,
            "Should be {}, get {}.".format(expected_result.values, self.die(data)),
        )
