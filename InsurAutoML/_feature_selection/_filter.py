"""
File Name: _filter.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_feature_selection/_filter.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 14th November 2022 7:01:24 pm
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

from __future__ import annotations

from typing import Union, List
from itertools import combinations
import warnings
import numpy as np
import pandas as pd

from InsurAutoML._utils import (
    maxloc,
    Pearson_Corr,
    MI,
    ACCC,
)


class FeatureFilter:

    """
    Use certain criteria to score each feature and select most relevent ones

    Parameters
    ----------
    criteria: use what criteria to score features, default = 'Pearson'
    supported {'Pearson', 'MI'}
    'Pearson': Pearson Correlation Coefficient
    'MI': 'Mutual Information'

    n_components: threshold to retain features, default = None
    will be set to n_features

    n_prop: float, default = None
    proprotion of features to select, if None, no limit
    n_components have higher priority than n_prop
    """

    def __init__(
        self,
        criteria: str = "Pearson",
        n_components: int = None,
        n_prop: float = None,
    ) -> None:
        self.criteria = criteria
        self.n_components = n_components
        self.n_prop = n_prop

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> FeatureFilter:

        # check whether y is empty
        if isinstance(y, pd.DataFrame):
            _empty = y.isnull().all().all()
        elif isinstance(y, pd.Series):
            _empty = y.isnull().all()
        elif isinstance(y, np.ndarray):
            _empty = np.all(np.isnan(y))
        else:
            _empty = y == None

        if _empty:
            raise ValueError("Must have response!")

        if self.criteria == "Pearson":
            self._score = Pearson_Corr(X, y)
        elif self.criteria == "MI":
            self._score = MI(X, y)

        self._fitted = True

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        # check if input is a dataframe
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # check whether n_components/n_prop is valid
        if self.n_components is None and self.n_prop is None:
            self.n_components = X.shape[1]
        elif self.n_components is not None:
            self.n_components = min(self.n_components, X.shape[1])
        # make sure selected features is at least 1
        elif self.n_prop is not None:
            self.n_components = max(1, int(self.n_prop * X.shape[1]))

        _columns = np.argsort(self._score)[: self.n_components]

        return X.iloc[:, _columns]


class mRMR:

    """
    mRMR [1] minimal-redundancy-maximal-relevance as criteria for filter-based
    feature selection

    [1] Peng, H., Long, F., & Ding, C. (2005). Feature selection based on mutual
    information criteria of max-dependency, max-relevance, and min-redundancy.
    IEEE Transactions on pattern analysis and machine intelligence, 27(8), 1226-1238.

    Parameters
    ----------
    n_components: int, default = None
    number of components to select, if None, no limit

    n_prop: float, default = None
    proprotion of features to select, if None, no limit
    n_components have higher priority than n_prop
    """

    def __init__(
        self,
        n_components: int = None,
        n_prop: float = None,
    ) -> None:
        self.n_components = n_components
        self.n_prop = n_prop

        self._fitted = False

    def select_feature(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray],
        selected_features: List,
        unselected_features: List,
    ) -> str:

        # select one feature as step, get all possible combinations
        test_item = list(combinations(unselected_features, 1))

        # initialize test results
        results = []
        for _comb in test_item:
            dependency = MI(X.iloc[:, _comb[0]], y)[0]
            if len(selected_features) > 0:
                redundancy = np.mean(
                    [
                        MI(X.iloc[:, item], X.iloc[:, _comb[0]])
                        for item in selected_features
                    ]
                )
                # append test results
                results.append(dependency - redundancy)
            # at initial, no selected feature, so no redundancy
            else:
                results.append(dependency)

        return test_item[maxloc(results)][0]  # use 0 to select item instead of tuple

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> mRMR:

        # check if inputs are dataframes
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y)

        # check whether n_components/n_prop is valid
        if self.n_components is None and self.n_prop is None:
            self.n_components = X.shape[1]
        elif self.n_components is not None:
            self.n_components = min(self.n_components, X.shape[1])
        # make sure selected features is at least 1
        elif self.n_prop is not None:
            self.n_components = max(1, int(self.n_prop * X.shape[1]))

        # initialize selected/unselected features
        selected_features = []
        unselected_features = list(range(X.shape[1]))

        for _ in range(self.n_components):
            # get the current optimal loss and feature
            new_feature = self.select_feature(
                X, y, selected_features, unselected_features
            )
            selected_features.append(new_feature)
            unselected_features.remove(new_feature)

        # record selected features
        self.select_features = selected_features

        self._fitted = True

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        return X.iloc[:, self.select_features]


class FOCI:

    """
    Implementation of Feature Ordering by Conditional Independence (FOCI) introduced in [1].
    Nonparametric feature selection method.

    [1] Azadkia, M., & Chatterjee, S. (2021). A simple measure of conditional dependence.
    The Annals of Statistics, 49(6), 3070-3102.
    """

    def __init__(
        self,
    ):
        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> FOCI:

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y)

        features = list(X.columns)  # list of features

        # initialize the selected/unselected features
        selected_features = []
        unselected_features = (
            features.copy()
        )  # copy of features, do not interfere with original features

        while True:
            tmp_ACCC_list = []  # CC list
            for feature in unselected_features:
                # at start, use unconditional ACCC
                if len(selected_features) == 0:
                    tmp_ACCC_list.append(ACCC(X[[feature]], y))
                # at following steps, use conditional ACCC
                else:
                    tmp_ACCC_list.append(ACCC(X[[feature]], y, X[selected_features]))

            tmp_feature = unselected_features[np.argmax(tmp_ACCC_list)]
            tmp_max_ACCC = tmp_ACCC_list[np.argmax(tmp_ACCC_list)]

            if tmp_max_ACCC > 0:
                selected_features.append(tmp_feature)
                unselected_features.remove(tmp_feature)
            else:
                break

        # record selected features
        self.select_features = selected_features

        self._fitted = True

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        return X[self.select_features]
