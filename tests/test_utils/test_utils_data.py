"""
File Name: test_utils_data.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /tests/test_utils/test_utils_data.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 28th November 2022 11:41:25 pm
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


def test_as_dataframe():
    from InsurAutoML.utils.data import as_dataframe

    converter = as_dataframe()

    _array = converter.to_array(pd.DataFrame([1, 2, 3, 4]))

    _df = converter.to_df(_array)

    assert isinstance(
        _array, np.ndarray
    ), "as_dataframe.to_array should return a np.ndarray, get {}".format(type(_array))

    assert isinstance(
        _df, pd.DataFrame
    ), "as_dataframe.to_df should return a pd.DataFrame, get {}".format(type(_df))


def test_unify_nan():
    from InsurAutoML.utils.data import unify_nan

    data = np.arange(15).reshape(5, 3)
    data = pd.DataFrame(data, columns=["column_1", "column_2", "column_3"])
    data.loc[:, "column_1"] = "novalue"
    data.loc[3, "column_2"] = "None"

    target_data = pd.DataFrame(
        {
            "column_1": ["novalue", "novalue", "novalue", "novalue", "novalue"],
            "column_2": [1, 4, 7, "None", 13],
            "column_3": [2, 5, 8, 11, 14],
            "column_1_useNA": [np.nan, np.nan, np.nan, np.nan, np.nan],
            "column_2_useNA": [1, 4, 7, np.nan, 13],
        }
    )

    assert (
        (unify_nan(data).astype(str) == target_data.astype(str)).all().all()
    ), "unify_nan should return target dataframe {}, get {}".format(
        target_data, unify_nan(data)
    )


def test_remove_index_columns():
    from InsurAutoML.utils.data import remove_index_columns

    data = pd.DataFrame(
        {
            "col_1": [1, 1, 1, 1, 1],
            "col_2": [1, 2, 3, 4, 5],
            "col_3": [1, 2, 3, 4, 5],
            "col_4": [1, 2, 3, 4, 5],
            "col_5": [1, 2, 3, 4, 5],
        }
    )

    remove_data_0 = remove_index_columns(data.values, axis=0, threshold=0.8)
    remove_data_1 = remove_index_columns(data, axis=1, threshold=0.8, save=True)

    assert isinstance(
        remove_data_0, pd.DataFrame
    ), "remove_index_columns should return a pd.DataFrame, get {}".format(
        type(remove_data_0)
    )
    assert isinstance(
        remove_data_1, pd.DataFrame
    ), "remove_index_columns should return a pd.DataFrame, get {}".format(
        type(remove_data_1)
    )

    remove_data_0 = remove_index_columns(
        data, axis=0, threshold=[0.8, 0.8, 0.8, 0.8, 0.8]
    )
    remove_data_1 = remove_index_columns(
        data, axis=1, threshold=[0.8, 0.8, 0.8, 0.8, 0.8], save=True
    )

    assert isinstance(
        remove_data_0, pd.DataFrame
    ), "remove_index_columns should return a pd.DataFrame, get {}".format(
        type(remove_data_0)
    )
    assert isinstance(
        remove_data_1, pd.DataFrame
    ), "remove_index_columns should return a pd.DataFrame, get {}".format(
        type(remove_data_1)
    )


def test_formatting():
    from InsurAutoML.utils.data import formatting

    data = pd.read_csv("Appendix/insurance.csv")
    train = data.iloc[100:, :]
    test = data.iloc[:100, :]

    formatter = formatting()
    formatter.fit(train)
    formatter.refit(train)
    formatter.refit(test)

    assert True, "The formatting is not correctly done."


def test_get_missing_matrix():
    from InsurAutoML.utils.data import get_missing_matrix

    test = pd.DataFrame(
        {
            "col_1": [1, 2, 3, np.nan, 4, "NA"],
            "col_2": [7, "novalue", "none", 10, 11, None],
            "col_3": [
                np.nan,
                "3/12/2000",
                "3/13/2000",
                np.nan,
                "3/12/2000",
                "3/13/2000",
            ],
        }
    )
    test["col_3"] = pd.to_datetime(test["col_3"])

    target_test = pd.DataFrame(
        {
            "col_1": [0, 0, 0, 1, 0, 1],
            "col_2": [0, 1, 1, 0, 0, 1],
            "col_3": [1, 0, 0, 1, 0, 0],
        }
    )

    assert (
        (get_missing_matrix(test) == target_test).all().all()
    ), "The missing matrix is not correct."


def test_extremeclass():
    from InsurAutoML.utils.data import ExtremeClass

    cutter = ExtremeClass(extreme_threshold=0.9)
    test = pd.DataFrame(
        np.random.randint(0, 10, size=(100, 10)),
        columns=["col_" + str(i) for i in range(10)],
    )
    test = cutter.cut(test)

    assert True, "The extreme class is not correctly done."


def test_assign_classes():
    from InsurAutoML.utils.data import assign_classes

    test = [[0.9, 0.1], [0.2, 0.8]]

    assert (
        assign_classes(test) == np.array([0, 1])
    ).all(), "The classes are not correctly assigned."


def test_softmax():
    from InsurAutoML.utils.data import softmax

    a = np.array([0.1, -0.1, 1])

    assert (
        softmax(a).sum(axis=1) == np.ones(3)
    ).all(), "The softmax function is not correct."

    a = np.array([[0.1, 0.2], [0.1, -0.2], [1, 0]])

    assert (
        softmax(a).sum(axis=1) == np.ones(3)
    ).all(), "The softmax function is not correct."
