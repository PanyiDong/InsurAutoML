"""
File: test_utils.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Latest Version: 0.2.0
Relative Path: /tests/test_utils/test_utils.py
File Created: Friday, 15th April 2022 7:42:15 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Friday, 15th April 2022 7:53:53 pm
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

import numpy as np


def test_random_guess():

    from My_AutoML._utils._base import random_guess

    assert random_guess(1) == 1, "random_guess(1) should be 1, get {}".format(
        random_guess(1)
    )
    assert random_guess(0) == 0, "random_guess(0) should be 0, get {}".format(
        random_guess(0)
    )
    assert (
        random_guess(0.5) == 0 or random_guess(0.5) == 1
    ), "random_guess(0.5) should be either 0 or 1, get {}".format(random_guess(0.5))


def test_random_index():

    from My_AutoML._utils._base import random_index

    assert (
        np.sort(random_index(5)) == np.array([0, 1, 2, 3, 4])
    ).all(), "random_index(5) should contain [0, 1, 2, 3, 4], get {}".format(
        random_index(5)
    )


def test_random_list():

    from My_AutoML._utils._base import random_list

    assert (
        np.sort(random_list([0, 1, 2, 3, 4])) == np.array([0, 1, 2, 3, 4])
    ).all(), "random_index(5) should contain [0, 1, 2, 3, 4], get {}".format(
        random_list([0, 1, 2, 3, 4])
    )


def test_minloc():

    from My_AutoML._utils._base import minloc

    assert (
        minloc([4, 2, 6, 2, 1]) == 4
    ), "minloc([4, 2, 6, 2, 1]) should be 5, get {}".format(minloc([4, 2, 6, 2, 1]))


def test_maxloc():

    from My_AutoML._utils._base import maxloc

    assert (
        maxloc([4, 2, 6, 2, 1]) == 2
    ), "maxloc([4, 2, 6, 2, 1]) should be 5, get {}".format(maxloc([4, 2, 6, 2, 1]))


def test_True_index():

    from My_AutoML._utils._base import True_index

    assert True_index([True, False, 1, 0, "hello", 5]) == [
        0,
        2,
    ], "True_index([True, False, 1, 0, 'hello', 5]) should be [0, 2], get {}".format(
        True_index([True, False, 1, 0, "hello", 5])
    )


def test_type_of_script():

    from My_AutoML._utils._base import type_of_script

    assert (
        type_of_script() == "terminal"
    ), "type_of_script() should be 'terminal', get {}".format(type_of_script())
