"""
File: _filter.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Last Version: 0.2.1
Relative Path: /My_AutoML/_feature_selection/_filter.py
File Created: Monday, 8th August 2022 8:43:53 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 8th August 2022 8:59:43 pm
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

from itertools import combinations
import numpy as np
import pandas as pd

from My_AutoML._utils import (
    maxloc,
    Pearson_Corr,
    MI,
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
        criteria="Pearson",
        n_components=None,
        n_prop=None,
    ):
        self.criteria = criteria
        self.n_components = n_components
        self.n_prop = n_prop

        self._fitted = False

    def fit(self, X, y=None):

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

    def transform(self, X):

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
        n_components=None,
        n_prop=None,
    ):
        self.n_components = n_components
        self.n_prop = n_prop

        self._fitted = False

    def select_feature(self, X, y, selected_features, unselected_features):

        # select one feature as step, get all possible combinations
        test_item = list(combinations(unselected_features, 1))

        # initialize test results
        results = []
        for _comb in test_item:
            dependency = MI(X.iloc[:, _comb[0]], y)
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

    def fit(self, X, y=None):

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

    def transform(self, X):

        return X.iloc[:, self.select_features]
