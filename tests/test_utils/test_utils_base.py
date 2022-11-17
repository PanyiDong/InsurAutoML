"""
File Name: test_utils_base.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /tests/test_utils/test_utils_base.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 14th November 2022 8:23:20 pm
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
import pandas as pd


def test_load_data():

    from InsurAutoML import load_data

    data = load_data().load("Appendix", "insurance")

    assert isinstance(
        data, dict
    ), "load_data should return a dict database, get {}".format(type(data))

    assert isinstance(
        data["insurance"], pd.DataFrame
    ), "load_data should return a dict database containing dataframes, get {}".format(
        type(data["insurance"])
    )


def test_random_guess():

    from InsurAutoML._utils._base import random_guess

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

    from InsurAutoML._utils._base import random_index

    assert (
        np.sort(random_index(5)) == np.array([0, 1, 2, 3, 4])
    ).all(), "random_index(5) should contain [0, 1, 2, 3, 4], get {}".format(
        random_index(5)
    )


def test_random_list():

    from InsurAutoML._utils._base import random_list

    assert (
        np.sort(random_list([0, 1, 2, 3, 4])) == np.array([0, 1, 2, 3, 4])
    ).all(), "random_index(5) should contain [0, 1, 2, 3, 4], get {}".format(
        random_list([0, 1, 2, 3, 4])
    )


def test_is_date():

    from InsurAutoML._utils._base import is_date

    test = pd.DataFrame(
        {
            "col_1": [1, 2, 3, 4, 5],
            "col_2": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-04",
                "2020-01-05",
            ],
        }
    )

    assert is_date(
        test, rule="all"), "The is_date method is not correctly done."


def test_feature_rounding():

    from InsurAutoML._utils._base import feature_rounding

    test = pd.DataFrame(
        {
            "col_1": [1, 2, 3, 4, 5],
            "col_2": [1.2, 2.2, 3.2, 4.2, 5.2],
        }
    )

    target_data = pd.DataFrame(
        {
            "col_1": [1, 2, 3, 4, 5],
            "col_2": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    assert (
        feature_rounding(test) == target_data
    ).all().all(), "The feature_rounding method is not correctly done."


def test_timer():

    from InsurAutoML._utils._base import Timer
    import time

    timer = Timer()
    timer.start()
    time.sleep(4)
    timer.stop()
    timer.start()
    time.sleep(3)
    timer.stop()

    assert timer.sum() / timer.avg() == 2.0, "The timer is not correctly done."
    assert timer.cumsum(
    )[-1] == timer.sum(), "The timer is not correctly done."


def test_minloc():

    from InsurAutoML._utils._base import minloc

    assert (
        minloc([4, 2, 6, 2, 1]) == 4
    ), "minloc([4, 2, 6, 2, 1]) should be 5, get {}".format(minloc([4, 2, 6, 2, 1]))


def test_maxloc():

    from InsurAutoML._utils._base import maxloc

    assert (
        maxloc([4, 2, 6, 2, 1]) == 2
    ), "maxloc([4, 2, 6, 2, 1]) should be 5, get {}".format(maxloc([4, 2, 6, 2, 1]))


def test_True_index():

    from InsurAutoML._utils._base import True_index

    assert True_index([True, False, 1, 0, "hello", 5]) == [
        0,
        2,
    ], "True_index([True, False, 1, 0, 'hello', 5]) should be [0, 2], get {}".format(
        True_index([True, False, 1, 0, "hello", 5])
    )


def test_type_of_script():

    from InsurAutoML._utils._base import type_of_script

    assert (
        type_of_script() == "terminal"
    ), "type_of_script() should be 'terminal', get {}".format(type_of_script())


def test_has_method():

    from InsurAutoML._utils._base import has_method
    from sklearn.linear_model import LogisticRegression

    mol = LogisticRegression()

    assert has_method(mol, "fit"), "The has_method function is not correct."
    assert has_method(
        mol, "__fit") == False, "The has_method function is not correct."


def test_is_none():

    from InsurAutoML._utils._base import is_none

    assert is_none(None), "The is_none function is not correct."
    assert is_none("not none") == False, "The is_none function is not correct."


def test_format_hyper_dict():

    from InsurAutoML._utils._base import format_hyper_dict

    input = {"encoder": "encoder", "n_estimators": 2}
    expected = {"encoder_1": "encoder", "encoder_n_estimators": 2}

    assert (
        format_hyper_dict(input, 1, ref="encoder", search_algo="RandomSearch") == input
    ), "The format_hyper_dict function is not correct."
    assert (format_hyper_dict(input, 1, ref="encoder", search_algo="HyperOpt")
            == expected), "The format_hyper_dict function is not correct."
