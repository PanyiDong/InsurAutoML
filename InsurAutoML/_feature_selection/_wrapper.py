"""
File Name: _wrapper.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_feature_selection/_wrapper.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 14th November 2022 7:05:52 pm
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

from inspect import isclass
from typing import Union, Callable, List, Tuple
from itertools import combinations
import numpy as np
import pandas as pd
import itertools

from InsurAutoML._utils import (
    minloc,
    maxloc,
)
from InsurAutoML._utils._base import has_method
from InsurAutoML._utils._optimize import (
    get_estimator,
    get_metrics,
)

# FeatureWrapper

# Exhaustive search for optimal feature combination
# Exhaustive search is practically impossible to implement in a reasonable time,
# so it's not included in the package, but it can be used.
class ExhaustiveFS:

    """
    Exhaustive Feature Selection

    Parameters
    ----------
    estimator: str or sklearn estimator, default = "Lasso"
    estimator must have fit/predict methods

    criteria: str or sklearn metric, default = "accuracy"
    """

    def __init__(
        self,
        estimator: str = "Lasso",
        criteria: str = "neg_accuracy",
    ) -> None:
        self.estimator = estimator
        self.criteria = criteria

        self._fitted = False

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]
    ) -> ExhaustiveFS:

        # make sure estimator is recognized
        if self.estimator == "Lasso":
            from sklearn.linear_model import Lasso

            self.estimator = Lasso()
        elif self.estimator == "Ridge":
            from sklearn.linear_model import Ridge

            self.estimator = Ridge()
        elif isclass(type(self.estimator)):
            # if estimator is recognized as a class
            # make sure it has fit/predict methods
            if not has_method(self.estimator, "fit") or not has_method(
                self.estimator, "predict"
            ):
                raise ValueError("Estimator must have fit/predict methods!")
        else:
            raise AttributeError("Unrecognized estimator!")

        # check whether criteria is valid
        if self.criteria == "neg_accuracy":
            from InsurAutoML._utils._stat import neg_accuracy

            self.criteria = neg_accuracy
        elif self.criteria == "MSE":
            from sklearn.metrics import mean_squared_error

            self.criteria = mean_squared_error
        elif isinstance(self.criteria, Callable):
            # if callable, pass
            pass
        else:
            raise ValueError("Unrecognized criteria!")

        # get all combinations of features
        all_comb = []
        for i in range(1, X.shape[1] + 1):
            for item in list(combinations(list(range(X.shape[1])), i)):
                all_comb.append(list(item))
        all_comb = np.array(all_comb).flatten()  # flatten 2D to 1D

        # initialize results
        results = []

        for comb in all_comb:
            self.estimator.fit(X.iloc[:, comb], y)
            results.append(self.criteria(y, self.estimator.predict(X.iloc[:, comb])))

        self.selected_features = all_comb[np.argmin(results)]

        self._fitted = True

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        return X.iloc[:, self.selected_features]


# Sequential Feature Selection (SFS)
class SFS:

    """
    Use Sequential Forward Selection/SFS to select subset of features.

    Parameters
    ----------
    estimators: str or estimator, default = "Lasso"
    string for some pre-defined estimator, or a estimator contains fit/predict methods

    n_components: int, default = None
    limit maximum number of features to select, if None, no limit

    n_prop: float, default = None
    proprotion of features to select, if None, no limit
    n_components have higher priority than n_prop

    criteria: str, default = "accuracy"
    criteria used to select features, can be "accuracy"
    """

    def __init__(
        self,
        estimator: str = "Lasso",
        n_components: int = None,
        n_prop: float = None,
        criteria: str = "neg_accuracy",
    ) -> None:
        self.estimator = estimator
        self.n_components = n_components
        self.n_prop = n_prop
        self.criteria = criteria

        self._fitted = False

    def select_feature(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray],
        estimator: Callable,
        selected_features: List[str],
        unselected_features: List[str],
    ) -> Tuple[float, str]:

        # select one feature as step, get all possible combinations
        test_item = list(combinations(unselected_features, 1))
        # concat new test_comb with selected_features
        test_comb = [list(item) + selected_features for item in test_item]

        # initialize test results
        results = []
        for _comb in test_comb:
            # fit estimator
            estimator.fit(X.iloc[:, _comb], y)
            # get test results
            test_results = self.criteria(y, estimator.predict(X.iloc[:, _comb]))
            # append test results
            results.append(test_results)

        return (
            results[minloc(results)],
            test_item[minloc(results)][0],
        )  # use 0 to select item instead of tuple

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> SFS:

        # check if the input is a dataframe
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # check whether y is empty
        # for SFS, y is required to train a model
        if isinstance(y, pd.DataFrame):
            _empty = y.isnull().all().all()
        elif isinstance(y, pd.Series):
            _empty = y.isnull().all()
        elif isinstance(y, np.ndarray):
            _empty = np.all(np.isnan(y))
        else:
            _empty = y == None

        # if empty, raise error
        if _empty:
            raise ValueError("Must have response!")

        # make sure estimator is recognized
        estimator = get_estimator(self.estimator)

        # check whether n_components/n_prop is valid
        if self.n_components is None and self.n_prop is None:
            self.n_components = X.shape[1]
        elif self.n_components is not None:
            self.n_components = min(self.n_components, X.shape[1])
        # make sure selected features is at least 1
        elif self.n_prop is not None:
            self.n_components = max(1, int(self.n_prop * X.shape[1]))

        # check whether criteria is valid
        self.criteria = get_metrics(self.criteria)

        # initialize selected/unselected features
        selected_features = []
        optimal_loss = np.inf
        unselected_features = list(range(X.shape[1]))

        # iterate until n_components are selected
        for _ in range(self.n_components):
            # get the current optimal loss and feature
            loss, new_feature = self.select_feature(
                X, y, estimator, selected_features, unselected_features
            )
            if loss > optimal_loss:  # if no better combination is found, stop
                break
            else:
                optimal_loss = loss
                selected_features.append(new_feature)
                unselected_features.remove(new_feature)

        # record selected features
        self.selected_features = selected_features

        self._fitted = True

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        return X.iloc[:, self.selected_features]


# Sequential Backward Selection (SBS)
# Sequential Floating Forward Selection (SFFS)
# Adapative Sequential Forward Floating Selection (ASFFS)
class ASFFS:

    """
    Adaptive Sequential Forward Floating Selection (ASFFS)
    Mostly, ASFFS performs the same as Sequential Floating Forward Selection (SFFS),
    where only one feature is considered as a time. But, when the selected features are coming
    close to the predefined maximum, a adaptive generalization limit will be activated, and
    more than one features can be considered at one time. The idea is to consider the correlation
    between features. [1]

    [1] Somol, P., Pudil, P., Novovičová, J. and Paclık, P., 1999. Adaptive floating search
    methods in feature selection. Pattern recognition letters, 20(11-13), pp.1157-1163.

    Parameters
    ----------
    d: maximum features retained, default = None
    will be calculated as max(max(20, n), 0.5 * n)

    Delta: dynamic of maximum number of features, default = 0
    d + Delta features will be retained

    b: range for adaptive generalization limit to activate, default = None
    will be calculated as max(5, 0.05 * n)

    r_max: maximum of generalization limit, default = 5
    maximum features to be considered as one step

    model: the model used to evaluate the objective function, default = 'linear'
    supproted ('Linear', 'Lasso', 'Ridge')

    objective: the objective function of significance of the features, default = 'MSE'
    supported {'MSE', 'MAE'}
    """

    def __init__(
        self,
        n_components: int = None,
        Delta: float = 0.0,
        b: int = None,
        r_max: int = 5,
        model: str = "Linear",
        objective: str = "MSE",
    ) -> None:
        self.n_components = n_components
        self.Delta = Delta
        self.b = b
        self.r_max = r_max
        self.model = model
        self.objective = objective

        self._fitted = False

    def generalization_limit(self, k: int, d: int, b: int) -> int:

        if np.abs(k - d) < b:
            r = self.r_max
        elif np.abs(k - d) < self.r_max + b:
            r = self.r_max + b - np.abs(k - d)
        else:
            r = 1

        return r

    def _Forward_Objective(
        self,
        selected: List[str],
        unselected: List[str],
        o: int,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray],
    ) -> Tuple[str, float]:

        _subset = list(itertools.combinations(unselected, o))
        _comb_subset = [
            selected + list(item) for item in _subset
        ]  # concat selected features with new features

        _objective_list = []
        if self.model == "Linear":
            from sklearn.linear_model import LinearRegression

            _model = LinearRegression()
        elif self.model == "Lasso":
            from sklearn.linear_model import Lasso

            _model = Lasso()
        elif self.model == "Ridge":
            from sklearn.linear_model import Ridge

            _model = Ridge()
        else:
            raise ValueError("Not recognizing model!")

        if self.objective == "MSE":
            from sklearn.metrics import mean_squared_error

            _obj = mean_squared_error
        elif self.objective == "MAE":
            from sklearn.metrics import mean_absolute_error

            _obj = mean_absolute_error

        for _set in _comb_subset:
            _model.fit(X[_set], y)
            _predict = _model.predict(X[_set])
            _objective_list.append(
                1 / _obj(y, _predict)
            )  # the goal is to maximize the objective function

        return (
            _subset[maxloc(_objective_list)],
            _objective_list[maxloc(_objective_list)],
        )

    def _Backward_Objective(
        self,
        selected: List[str],
        o: int,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray],
    ) -> Tuple[str, float]:

        _subset = list(itertools.combinations(selected, o))
        _comb_subset = [
            [_full for _full in selected if _full not in item] for item in _subset
        ]  # remove new features from selected features

        _objective_list = []
        if self.model == "Linear":
            from sklearn.linear_model import LinearRegression

            _model = LinearRegression()
        elif self.model == "Lasso":
            from sklearn.linear_model import Lasso

            _model = Lasso()
        elif self.model == "Ridge":
            from sklearn.linear_model import Ridge

            _model = Ridge()
        else:
            raise ValueError("Not recognizing model!")

        if self.objective == "MSE":
            from sklearn.metrics import mean_squared_error

            _obj = mean_squared_error
        elif self.objective == "MAE":
            from sklearn.metrics import mean_absolute_error

            _obj = mean_absolute_error

        for _set in _comb_subset:
            _model.fit(X[_set], y)
            _predict = _model.predict(X[_set])
            _objective_list.append(
                1 / _obj(y, _predict)
            )  # the goal is to maximize the objective function

        return (
            _subset[maxloc(_objective_list)],
            _objective_list[maxloc(_objective_list)],
        )

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]
    ) -> ASFFS:

        # check if the input is a dataframe
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y)

        n, p = X.shape
        features = list(X.columns)

        if self.n_components == None:
            _n_components = min(max(20, p), int(0.5 * p))
        else:
            _n_components = self.n_components
        if self.b == None:
            _b = min(5, int(0.05 * p))

        _k = 0
        self.J_max = [
            0 for _ in range(p + 1)
        ]  # mark the most significant objective function value
        self._subset_max = [
            [] for _ in range(p + 1)
        ]  # mark the best performing subset features
        _unselected = features.copy()
        # selected  feature stored here, not selected will be stored in features
        _selected = []

        while True:

            # Forward Phase
            _r = self.generalization_limit(_k, _n_components, _b)
            _o = 1

            while (
                _o <= _r and len(_unselected) >= 1
            ):  # not reasonable to add feature when all selected

                _new_feature, _max_obj = self._Forward_Objective(
                    _selected, _unselected, _o, X, y
                )

                if _max_obj > self.J_max[_k + _o]:
                    self.J_max[_k + _o] = _max_obj.copy()
                    _k += _o
                    for (
                        _f
                    ) in (
                        _new_feature
                    ):  # add new features and remove these features from the pool
                        _selected.append(_f)
                    for _f in _new_feature:
                        _unselected.remove(_f)
                    self._subset_max[_k] = _selected.copy()
                    break
                else:
                    if _o < _r:
                        _o += 1
                    else:
                        _k += 1  # the marked in J_max and _subset_max are considered as best for _k features
                        _selected = self._subset_max[
                            _k
                        ].copy()  # read stored best subset
                        _unselected = features.copy()
                        for _f in _selected:
                            _unselected.remove(_f)
                        break

            # Termination Condition
            if _k >= _n_components + self.Delta:
                break

            # Backward Phase
            _r = self.generalization_limit(_k, _n_components, _b)
            _o = 1

            while (
                _o <= _r and _o < _k
            ):  # not reasonable to remove when only _o feature selected

                _new_feature, _max_obj = self._Backward_Objective(_selected, _o, X, y)

                if _max_obj > self.J_max[_k - _o]:
                    self.J_max[_k - _o] = _max_obj.copy()
                    _k -= _o
                    for (
                        _f
                    ) in (
                        _new_feature
                    ):  # add new features and remove these features from the pool
                        _unselected.append(_f)
                    for _f in _new_feature:
                        _selected.remove(_f)
                    self._subset_max[_k] = _selected.copy()

                    _o = 1  # return to the start of backward phase, make sure the best subset is selected
                else:
                    if _o < _r:
                        _o += 1
                    else:
                        break

        self.selected_ = _selected

        self._fitted = True

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        # check if the input is a dataframe
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        return X.loc[:, self.selected_]
