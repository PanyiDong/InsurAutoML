"""
File Name: filter.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.6
Relative Path: /InsurAutoML/feature_selection/filter.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Wednesday, 3rd September 2025 3:28:49 pm
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
from functools import partial
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor
import warnings
import numpy as np
import pandas as pd

from ..utils import (
    maxloc,
    Pearson_Corr,
    MI,
    CCC,
    ACCC,
)
from .base import BaseFeatureSelection


class FeatureFilter(BaseFeatureSelection):
    """
    Use certain criteria to score each feature and select most relevent ones

    Parameters
    ----------
    criteria: use what criteria to score features, default = 'Pearson'
    supported {'Pearson', 'MI', 'CCC'}
    'Pearson': Pearson Correlation Coefficient
    'MI': 'Mutual Information'

    n_components: threshold to retain features, default = None
    will be set to n_features

    n_prop: float, default = None
    proprotion of features to select, if None, no limit
    n_components have higher priority than n_prop
    """
    
    criteria_mapping = {
        "Pearson": Pearson_Corr,
        "MI": MI,
        "CCC": CCC,
    }

    def __init__(
        self,
        criteria: str = "Pearson",
        n_components: int = None,
        n_prop: float = None,
    ) -> None:
        self.criteria = criteria
        self.n_components = n_components
        self.n_prop = n_prop

        super().__init__()
        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> FeatureFilter:
        # get feature names
        self._check_feature_names(X)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        self.features = list(X.columns)  # list of features

        # check whether y is empty
        if isinstance(y, pd.DataFrame):
            _empty = y.isnull().all().all()
        elif isinstance(y, pd.Series):
            _empty = y.isnull().all()
        elif isinstance(y, np.ndarray):
            _empty = np.all(np.isnan(y))
        else:
            _empty = y is None

        if _empty:
            raise ValueError("Must have response!")
        
        # check X and y have same lengths
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
        
        # check features
        
        # check whether n_components/n_prop is valid
        if self.n_components is None and self.n_prop is None:
            self.n_components = X.shape[1]
        elif self.n_components is not None:
            self.n_components = min(self.n_components, X.shape[1])
        # make sure selected features is at least 1
        elif self.n_prop is not None:
            self.n_components = max(1, int(self.n_prop * X.shape[1]))

        # apply criteria on each column
        if self.criteria == "Pearson":
            self._score = np.abs(Pearson_Corr(X, y))
        elif self.criteria == "MI":
            self._score = MI(X, y)
        elif self.criteria == "CCC":
            self._score = [CCC(X.loc[:, feature], y) for feature in self.features]
        else :
            raise ValueError(f"Unsupported criteria: {self.criteria}")
        
        self.select_features = np.array(self.features)[
            np.argsort(self._score)[-self.n_components:]
        ]

        self._fitted = True

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        # check if input is a dataframe
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.features)

        return X.loc[:, self.select_features]


class mRMR(BaseFeatureSelection):
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

        super().__init__()
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

        # use 0 to select item instead of tuple
        return test_item[maxloc(results)][0]

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> mRMR:
        # get feature names
        self._check_feature_names(X)
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


class FOCI(BaseFeatureSelection):
    """
    Implementation of Feature Ordering by Conditional Independence (FOCI) introduced in [1][2].
    Nonparametric feature selection method.

    [1] https://rdrr.io/cran/FOCI/src/R/foci.R
    [2] Azadkia, M., & Chatterjee, S. (2021). A simple measure of conditional dependence.
    The Annals of Statistics, 49(6), 3070-3102.
    """

    def __init__(
        self,
        num_features: int = None,
        standardize: str = None,
        conditional: bool = False,
    ):
        super().__init__()
        self.num_features = num_features
        self.standardize = standardize
        self.conditional = conditional
        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> FOCI:
        # get feature names
        self._check_feature_names(X)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y)

        self.features = list(X.columns)  # list of features

        self.num_features = (
            len(self.features) if self.num_features is None else self.num_features
        )

        # standardize the data if needed
        if self.standardize == "standardize":
            from sklearn.preprocessing import StandardScaler

            self.scaler = StandardScaler()
            X = pd.DataFrame(self.scaler.fit_transform(X), columns=self.features)
        elif self.standardize == "minmax":
            from sklearn.preprocessing import MinMaxScaler

            self.scaler = MinMaxScaler()
            X = pd.DataFrame(self.scaler.fit_transform(X), columns=self.features)
        elif self.standardize is None:
            from ..base import no_processing

            self.scaler = no_processing()
            self.scaler.fit(X)

        # fit the model
        self._fit(X, y)

        return self
    
    @staticmethod
    def compute_accc(feature, X, y, selected_features):
        if len(selected_features) == 0:
            return ACCC(X[[feature]], y, mode="Q")
        else:
            return ACCC(X[selected_features + [feature]], y, mode="Q")
        
    @staticmethod
    def compute_conditional_accc(feature, X, y, selected_features):
        if len(selected_features) == 0:
            return ACCC(X[[feature]], y, mode="Q")
        else:
            return ACCC(X[[feature]], y, X[selected_features], mode="Q")

    def _fit(self, X, y) -> FOCI:

        # initialize the selected/unselected features
        Q = []
        selected_features = []
        unselected_features = (
            self.features.copy()
        )  # copy of features, do not interfere with original features
        
        # parallel computation of ACCC
        # if conditional, use conditional accc (given selected features)
        if self.conditional :
            func = partial(
                self.compute_conditional_accc, X=X, y=y, selected_features=selected_features
            )
        else:
            func = partial(
                self.compute_accc, X=X, y=y, selected_features=selected_features
            )

        for idx in range(self.num_features):
            with ProcessPoolExecutor() as executor:
                tmp_ACCC_list = list(executor.map(func, unselected_features))

            tmp_feature = unselected_features[np.argmax(tmp_ACCC_list)]
            Q.append(tmp_ACCC_list[np.argmax(tmp_ACCC_list)])
            # check stopping criteria
            # if conditional, check if current feature improves (conditional) ACCC
            if self.conditional:
                condition = Q[idx] > 0
            else:
                condition = Q[0] > 0 if idx == 0 else Q[idx] > Q[idx - 1]

            if condition:
                selected_features.append(tmp_feature)
                unselected_features.remove(tmp_feature)
            else:
                break

        # record selected features
        self.select_features = selected_features

        self._fitted = True

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        # standardize the data if needed
        X = self.scaler.transform(X)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.features)
        return X.loc[:, self.select_features]
