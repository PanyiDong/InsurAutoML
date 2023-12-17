"""
File Name: test_imputer.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /tests/test_imputer/test_imputer.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 28th November 2022 11:21:10 pm
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

# from My_AutoML._imputation import imputers

# data_X = pd.DataFrame(
#     {
#         "col_1": [np.nan, 8, 6, 7, 9, 9, 8, 8, 7, 5],
#         "col_2": [9, 7, 2, 1, 6, 8, 8, 9, 3, 6],
#     }
# )


# class TestImputer(unittest.TestCase):
#     def test_Imputer(self):

#         self.method_dict = imputers
#         self.method_names = list(self.method_dict.keys())
#         self.method_objects = list(self.method_dict.values())

# for method_name, method_object in zip(self.method_names,
# self.method_objects):

#             if method_name == "KNNImputer":
#                 mol = method_object(n_neighbors=1)
#             else:
#                 mol = method_object()

#             # mol.fill(data_X)
#             mol._fitted = True

#             # check whether the method is fitted
#             self.assertEqual(
#                 mol._fitted,
#                 True,
#                 "The method {} is not correctly fitted.".format(method_name),
#             )

#             print(
#                 "The method {} is correctly fitted.".format(method_name),
#             )


def test_imputer():
    from InsurAutoML.imputation import imputers
    from InsurAutoML.utils import formatting

    for method_name, method_object in zip(imputers.keys(), imputers.values()):
        imputer = method_object()
        if method_name != "KNNImputer":
            data = pd.read_csv("Appendix/healthcare-dataset-stroke-data.csv")

            encoder = formatting()
            encoder.fit(data)

            data = imputer.fill(data)

            assert imputer._fitted, "The method {} is not correctly fitted.".format(
                method_name
            )
            assert (
                data.isnull().any().any() == False
            ), "The imputation method {} fail to impute all missings.".format(
                method_name
            )


def test_DummyImputer():
    from InsurAutoML.imputation import DummyImputer

    data = pd.DataFrame(
        np.random.randint(0, 50, size=(100, 5)),
        columns=["col_1", "col_2", "col_3", "col_4", "col_5"],
    )
    for _index in data.index:
        if np.random.rand() < 0.1:
            data.loc[_index, "col_3"] = np.nan
    data["col_6"] = np.random.randint(0, 10, size=100)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    imputer = DummyImputer(
        method="median",
    )
    filled_data = imputer.fill(X, y)

    assert imputer._fitted, "The method DummyImputer is not correctly fitted."
    assert (
        filled_data.isnull().any().any() == False
    ), "The imputation method DummyImputer fail to impute all missings."


def test_kNNImputer():
    from InsurAutoML.imputation import KNNImputer

    data = pd.DataFrame(
        np.random.randint(0, 50, size=(100, 5)),
        columns=["col_1", "col_2", "col_3", "col_4", "col_5"],
    )
    for _index in data.index:
        if np.random.rand() < 0.1:
            data.loc[_index, "col_3"] = np.nan

    imputer = KNNImputer(
        n_neighbors=3,
        method="median",
    )
    filled_data = imputer.fill(data)

    assert imputer._fitted, "The method KNNImputer is not correctly fitted."
    assert (
        filled_data.isnull().any().any() == False
    ), "The imputation method KNNImputer fail to impute all missings."


# def test_imputer_AAI_kNN():

#     from My_AutoML._imputation._clustering import AAI_kNN

#     # test AAI_kNN

#     # generate missing data
#     data = pd.DataFrame(
#         np.random.randint(0, 100, size=(1000, 5)),
#         columns=["col_" + str(i) for i in range(1, 6)],
#     )
#     for _column in data.columns:
#         for _index in data.index:
#             if np.random.rand() < 0.1:
#                 data.loc[_index, _column] = np.nan

#     imputer = AAI_kNN(similarity="PCC")
#     fill_data = imputer.fill(data)

#     assert imputer._fitted == True, "The method {} is not correctly fitted.".format(
#         "AAI_kNN"
#     )
#     assert (
#         fill_data.isnull().any().any() == False
#     ), "The imputation method {} fail to impute all missings.".format("AAI_kNN")

#     imputer = AAI_kNN(similarity="COS")
#     fill_data = imputer.fill(data)

#     assert imputer._fitted == True, "The method {} is not correctly fitted.".format(
#         "AAI_kNN"
#     )
#     assert (
#         fill_data.isnull().any().any() == False
#     ), "The imputation method {} fail to impute all missings.".format("AAI_kNN")


# def test_imputer_CMI():

#     from My_AutoML._imputation._clustering import CMI

#     # test CMI

#     # generate missing data
#     data = pd.DataFrame(
#         np.random.randint(0, 100, size=(1000, 5)),
#         columns=["col_" + str(i) for i in range(1, 6)],
#     )
#     for _column in data.columns:
#         for _index in data.index:
#             if np.random.rand() < 0.1:
#                 data.loc[_index, _column] = np.nan

#     imputer = CMI()
#     fill_data = imputer.fill(data)

#     assert imputer._fitted == True, "The method {} is not correctly fitted.".format(
#         "CMI"
#     )
#     assert (
#         fill_data.isnull().any().any() == False
#     ), "The imputation method {} fail to impute all missings.".format("CMI")


# def test_imputer_k_Prototype_NN():

#     from My_AutoML._imputation._clustering import k_Prototype_NN

#     # test k_Prototype_NN

#     # generate missing data
#     data = pd.DataFrame(
#         np.random.randint(0, 100, size=(1000, 5)),
#         columns=["col_" + str(i) for i in range(1, 6)],
#     )
#     for _column in data.columns:
#         for _index in data.index:
#             if np.random.rand() < 0.1:
#                 data.loc[_index, _column] = np.nan

#     imputer = k_Prototype_NN()
#     fill_data = imputer.fill(data)

#     assert imputer._fitted == True, "The method {} is not correctly fitted.".format(
#         "k_Prototype_NN"
#     )
#     assert (
#         fill_data.isnull().any().any() == False
#     ), "The imputation method {} fail to impute all missings.".format("k_Prototype_NN")
