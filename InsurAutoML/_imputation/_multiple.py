"""
File Name: _multiple.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_imputation/_multiple.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 14th November 2022 8:13:40 pm
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
import numpy as np
import pandas as pd
import warnings

from InsurAutoML._constant import UNI_CLASS
from InsurAutoML._utils import random_index, random_list
from ._base import SimpleImputer


class ExpectationMaximization:

    """
    Use Expectation Maximization (EM) to impute missing data[1]

    [1] Impyute.imputation.cs.em

    Parameters
    ----------
    iterations: maximum number of iterations for single imputation, default = 50

    threshold: threshold to early stop iterations, default = 0.01
    only early stop when iterations < self.iterations and change in the imputation < self.threshold

    seed: random seed, default = 1
    """

    def __init__(
        self, iterations: int = 50, threshold: float = 0.01, seed: int = 1
    ) -> None:
        self.iterations = iterations
        self.threshold = threshold
        self.seed = seed

        self._fitted = False  # whether the imputer has been fitted

    def fill(self, X: pd.DataFrame) -> pd.DataFrame:

        self.iterations = int(self.iterations)
        self.threshold = float(self.threshold)

        _X = X.copy(deep=True)
        n = _X.shape[0]

        if _X.isnull().values.any():
            _X = self._fill(_X)

        self._fitted = True

        return _X

    def _fill(self, X: pd.DataFrame) -> pd.DataFrame:

        features = list(X.columns)
        np.random.seed(self.seed)

        _missing_feature = []  # features contains missing values
        _missing_vector = []  # vector with missing values, to mark the missing index
        # create _missing_table with _missing_feature
        # missing index will be 1, existed index will be 0

        for _column in features:
            if X[_column].isnull().values.any():
                _missing_feature.append(_column)
                _missing_vector.append(
                    X[_column].loc[X[_column].isnull()].index.astype(int)
                )

        _missing_vector = np.array(_missing_vector).T
        self._missing_table = pd.DataFrame(_missing_vector, columns=_missing_feature)

        for _column in list(self._missing_table.columns):
            for _index in self._missing_table[_column]:
                X.loc[_index, _column] = self._EM_iter(X, _index, _column)

        return X

    def _EM_iter(self, X: pd.DataFrame, index: Union[int, str], column: str):

        _mark = 1
        for _ in range(self.iterations):
            _mu = np.nanmean(X.loc[:, column])
            _std = np.nanstd(X.loc[:, column])
            _tmp = np.random.normal(loc=_mu, scale=_std)
            _delta = np.abs(_tmp - _mark) / _mark
            if _delta < self.threshold and self.iterations > 10:
                return _tmp
            X.loc[index, column] = _tmp
            _mark = _tmp
        return _tmp


class KNNImputer:

    """
    Use KNN to impute the missing values, further update: use cross validation to select best k [1]

    [1] Stekhoven, D.J. and Bühlmann, P., 2012. MissForest—non-parametric missing value imputation
    for mixed-type data. Bioinformatics, 28(1), pp.112-118.

    Parameters
    ----------
    n_neighbors: list of k, default = None
    default will set to 1:10

    method: method to initaillay impute missing values, default = "mean"

    fold: cross validation number of folds, default = 10

    uni_class: unique class to be considered as categorical columns, default = 31

    seed: random seed, default = 1
    """

    def __init__(
        self,
        n_neighbors: int = None,
        method: str = "mean",
        fold: int = 10,
        uni_class: int = UNI_CLASS,
        seed: int = 1,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.method = method
        self.fold = fold
        self.uni_class = uni_class
        self.seed = seed

        self._fitted = False  # whether the imputer has been fitted

    def fill(self, X: pd.DataFrame) -> pd.DataFrame:

        features = list(X.columns)
        for _column in features:
            if len(X[_column].unique()) <= min(0.1 * len(X), self.uni_class):
                raise ValueError("KNN Imputation not supported for categorical data!")

        _X = X.copy(deep=True)
        if _X.isnull().values.any():
            _X = self._fill(_X)
        else:
            warnings.warn("No nan values found, no change.")

        self._fitted = True

        return _X

    def _fill(self, X: pd.DataFrame) -> pd.DataFrame:

        features = list(X.columns)

        self._missing_feature = []  # features contains missing values
        self._missing_vector = (
            []
        )  # vector with missing values, to mark the missing index
        # create _missing_table with _missing_feature
        # missing index will be 1, existed index will be 0

        for _column in features:
            if X[_column].isnull().values.any():
                self._missing_feature.append(_column)
                self._missing_vector.append(
                    X[_column].loc[X[_column].isnull()].index.astype(int)
                )

        self._missing_vector = np.array(self._missing_vector).T
        self._missing_table = pd.DataFrame(
            self._missing_vector, columns=self._missing_feature
        )

        X = SimpleImputer(method=self.method).fill(
            X
        )  # initial filling for missing values

        random_features = random_list(
            self._missing_feature, self.seed
        )  # the order to regress on missing features
        # _index = random_index(len(X.index))  # random index for cross validation
        _err = []

        # if assigned n_neighbors, use it, otherwise use k-fold cross validation
        if self.n_neighbors is None:
            for i in range(self.fold):
                _test = X.iloc[
                    i * int(len(X.index) / self.fold) : int(len(X.index) / self.fold), :
                ]
                _train = X
                _train.drop(labels=_test.index, axis=0, inplace=True)
                _err.append(self._cross_validation_knn(_train, _test, random_features))

            _err = np.mean(np.array(_err), axis=0)  # mean of cross validation error
            self.optimial_k = np.array(_err).argmin()[0] + 1  # optimal k

            X = self._knn_impute(X, random_features, self.optimial_k)
        else:
            X = self._knn_impute(X, random_features, self.n_neighbors)

        return X

    def _cross_validation_knn(
        self, _train: pd.DataFrame, _test: pd.DataFrame, random_features: List[str]
    ) -> List[Union[float, np.ndarray]]:  # cross validation to return error

        from sklearn.neighbors import KNeighborsRegressor

        if self.n_neighbors == None:
            n_neighbors = [i + 1 for i in range(10)]
        else:
            n_neighbors = (
                self.n_neighbors
                if isinstance(self.n_neighbors, list)
                else [self.n_neighbors]
            )

        _test_mark = _test.copy(deep=True)
        _err = []

        for _k in n_neighbors:
            _test = _test_mark.copy(deep=True)
            for _feature in random_features:
                _subfeatures = list(_train.columns)
                _subfeatures.remove(_feature)

                fit_model = KNeighborsRegressor(n_neighbors=_k)
                fit_model.fit(_train.loc[:, _subfeatures], _train.loc[:, _feature])
                _test.loc[:, _feature] = fit_model.predict(_test.loc[:, _subfeatures])
            _err.append(((_test - _test_mark) ** 2).sum())

        return _err

    def _knn_impute(
        self, X: pd.DataFrame, random_features: List[str], k: int
    ) -> pd.DataFrame:

        from sklearn.neighbors import KNeighborsRegressor

        features = list(X.columns)
        for _column in random_features:
            _subfeature = features.copy()
            _subfeature.remove(_column)
            X.loc[self._missing_table[_column], _column] = np.nan
            fit_model = KNeighborsRegressor(n_neighbors=k)
            fit_model.fit(
                X.loc[~X[_column].isnull(), _subfeature],
                X.loc[~X[_column].isnull(), _column],
            )
            X.loc[X[_column].isnull(), _column] = fit_model.predict(
                X.loc[X[_column].isnull(), _subfeature]
            )

        return X


class MissForestImputer:

    """
    Run Random Forest to impute the missing values [1]

    [1] Stekhoven, D.J. and Bühlmann, P., 2012. MissForest—non-parametric missing
    value imputation for mixed-type data. Bioinformatics, 28(1), pp.112-118.

    Parameters
    ----------
    threshold: threshold to terminate iterations, default = 0
    At default, if difference between iterations increases, the iteration stops

    method: initial imputation method for missing values, default = 'mean'

    uni_class: column with unique classes less than uni_class will be considered as categorical, default = 31
    """

    def __init__(
        self, threshold: float = 0, method: str = "mean", uni_class: int = UNI_CLASS
    ) -> None:
        self.threshold = threshold
        self.method = method
        self.uni_class = uni_class

        self._fitted = False  # whether the imputer has been fitted

    def _RFImputer(self, X: pd.DataFrame) -> pd.DataFrame:

        from sklearn.ensemble import RandomForestRegressor

        _delta = []  # criteria of termination

        while True:
            for _column in list(self._missing_table.columns):
                X_old = X.copy(deep=True)
                _subfeature = list(X_old.columns)
                _subfeature.remove(str(_column))
                _missing_index = self._missing_table[_column].tolist()
                RegModel = RandomForestRegressor()
                RegModel.fit(
                    X.loc[~X.index.astype(int).isin(_missing_index), _subfeature],
                    X.loc[~X.index.astype(int).isin(_missing_index), _column],
                )
                _tmp_column = RegModel.predict(
                    X.loc[X.index.astype(int).isin(_missing_index), _subfeature]
                )
                X.loc[X.index.astype(int).isin(_missing_index), _column] = _tmp_column
                _delta.append(self._delta_cal(X, X_old))
                if len(_delta) >= 2 and _delta[-1] > _delta[-2]:
                    break
            if len(_delta) >= 2 and _delta[-1] > _delta[-2]:
                break

        return X

    # calcualte the difference between data newly imputed and before imputation
    def _delta_cal(self, X_new: pd.DataFrame, X_old: pd.DataFrame) -> float:

        if (X_new.shape[0] != X_old.shape[0]) or (X_new.shape[1] != X_old.shape[1]):
            raise ValueError("New and old data must have same size, get different!")

        _numerical_features = []
        _categorical_features = []
        for _column in list(self._missing_table.columns):
            if len(X_old[_column].unique()) <= self.uni_class:
                _categorical_features.append(_column)
            else:
                _numerical_features.append(_column)

        _N_nume = 0
        _N_deno = 0
        _F_nume = 0
        _F_deno = 0

        if len(_numerical_features) > 0:
            for _column in _numerical_features:
                _N_nume += ((X_new[_column] - X_old[_column]) ** 2).sum()
                _N_deno += (X_new[_column] ** 2).sum()

        if len(_categorical_features) > 0:
            for _column in _categorical_features:
                _F_nume += (X_new[_column] != X_old[_column]).astype(int).sum()
                _F_deno += len(self._missing_table[_column])

        if len(_numerical_features) > 0 and len(_categorical_features) > 0:
            return _N_nume / _N_deno + _F_nume / _F_deno
        elif len(_numerical_features) > 0:
            return _N_nume / _N_deno
        elif len(_categorical_features) > 0:
            return _F_nume / _F_deno

    def fill(self, X: pd.DataFrame) -> pd.DataFrame:

        _X = X.copy(deep=True)
        if _X.isnull().values.any():
            _X = self._fill(_X)
        else:
            warnings.warn("No nan values found, no change.")

        self._fitted = True

        return _X

    def _fill(self, X: pd.DataFrame) -> pd.DataFrame:

        features = list(X.columns)

        for _column in features:
            if (X[_column].dtype == object) or (str(X[_column].dtype) == "category"):
                raise ValueError(
                    "MICE can only handle numerical filling, run encoding first!"
                )

        _missing_feature = []  # features contains missing values
        _missing_vector = []  # vector with missing values, to mark the missing index
        # create _missing_table with _missing_feature
        # missing index will be 1, existed index will be 0
        _missing_count = []  # counts for missing values

        for _column in features:
            if X[_column].isnull().values.any():
                _missing_feature.append(_column)
                _missing_vector.append(X.loc[X[_column].isnull()].index.astype(int))
                _missing_count.append(X[_column].isnull().astype(int).sum())

        # reorder the missing features by missing counts increasing
        _order = np.array(_missing_count).argsort().tolist()
        _missing_count = np.array(_missing_count)[_order].tolist()
        _missing_feature = np.array(_missing_feature)[_order].tolist()
        _missing_vector = np.array(_missing_vector)[_order].T.tolist()

        self._missing_table = pd.DataFrame(_missing_vector, columns=_missing_feature)

        X = SimpleImputer(method=self.method).fill(
            X
        )  # initial filling for missing values
        X = self._RFImputer(X)

        return X


class MICE:

    """
    Multiple Imputation by chained equations (MICE)
    using single imputation to initialize the imputation step, and iteratively build regression/
    classification model to impute features with missing values [1]

    [1] Azur, M.J., Stuart, E.A., Frangakis, C. and Leaf, P.J., 2011. Multiple imputation by
    chained equations: what is it and how does it work?. International journal of methods in
    psychiatric research, 20(1), pp.40-49.

    Parameters
    ----------
    cycle: how many runs of regression/imputation to build the complete data, default = 10

    method: the method to initially fill nan values, default = 'mean'
    supproted methods ['mean', 'zero', 'median', 'most frequent', constant]
    'mean' : fill columns with nan values using mean of non nan values
    'zero': fill columns with nan values using 0
    'median': fill columns with nan values using median of non nan values
    'most frequent': fill columns with nan values using most frequent of non nan values
    constant: fill columns with nan values using predefined values

    seed: random seed, default = 1
    every random draw from the minority class will increase the random seed by 1
    """

    def __init__(self, cycle: int = 10, method: str = "mean", seed: int = 1) -> None:
        self.method = method
        self.cycle = cycle
        self.seed = seed

        self._fitted = False  # whether the imputer has been fitted

    def fill(self, X: pd.DataFrame) -> pd.DataFrame:

        self.cycle = int(self.cycle)

        _X = X.copy(deep=True)

        if _X.isnull().values.any():
            _X = self._fill(_X)
        else:
            warnings.warn("No nan values found, no change.")

        self._fitted = True

        return _X

    def _fill(self, X: pd.DataFrame) -> pd.DataFrame:

        features = list(X.columns)

        for _column in features:
            if (X[_column].dtype == object) or (str(X[_column].dtype) == "category"):
                raise ValueError(
                    "MICE can only handle numerical filling, run encoding first!"
                )

        self._missing_feature = []  # features contains missing values
        self._missing_vector = (
            []
        )  # vector with missing values, to mark the missing index
        # create _missing_table with _missing_feature
        # missing index will be 1, existed index will be 0

        for _column in features:
            if X[_column].isnull().values.any():
                self._missing_feature.append(_column)
                self._missing_vector.append(
                    X.loc[X[_column].isnull()].index.astype(int)
                )

        self._missing_vector = np.array(self._missing_vector).T
        self._missing_table = pd.DataFrame(
            self._missing_vector, columns=self._missing_feature
        )

        X = SimpleImputer(method=self.method).fill(
            X
        )  # initial filling for missing values

        random_features = random_list(
            self._missing_feature, self.seed
        )  # the order to regress on missing features

        for _ in range(self.cycle):
            X = self._cycle_impute(X, random_features)

        return X

    def _cycle_impute(
        self, X: pd.DataFrame, random_features: List[str]
    ) -> pd.DataFrame:

        from sklearn.linear_model import LinearRegression, LogisticRegression, LassoCV

        features = list(X.columns)

        for _column in random_features:
            _subfeature = features
            _subfeature.remove(_column)
            _missing_index = self._missing_table[_column].tolist()
            X.loc[X.index.astype(int).isin(_missing_index), _column] = np.nan
            if len(X[_column].unique()) == 2:
                fit_model = LogisticRegression()
            elif len(features) <= 15:
                fit_model = LinearRegression()
            else:
                fit_model = LassoCV()
            fit_model.fit(
                X.loc[~X[_column].isnull(), _subfeature],
                X.loc[~X[_column].isnull(), _column],
            )
            X.loc[X[_column].isnull(), _column] = fit_model.predict(
                X.loc[X[_column].isnull(), _subfeature]
            )

        return X
