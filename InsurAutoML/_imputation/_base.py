"""
File Name: _base.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_imputation/_base.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 14th November 2022 8:07:35 pm
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

from typing import Union
import numpy as np
import pandas as pd

from InsurAutoML._utils import nan_cov


class SimpleImputer:

    """
    Simple Imputer to fill nan values

    Parameters
    ----------
    method: the method to fill nan values, default = 'mean'
    supproted methods ['mean', 'zero', 'median', 'most frequent', constant]
    'mean' : fill columns with nan values using mean of non nan values
    'zero': fill columns with nan values using 0
    'median': fill columns with nan values using median of non nan values
    'most frequent': fill columns with nan values using most frequent of non nan values
    constant: fill columns with nan values using predefined values
    """

    def __init__(self, method: str = "mean") -> None:
        self.method = method

        self._fitted = False  # whether the imputer has been fitted

    def fill(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        _X = X.copy(deep=True)

        if _X.isnull().values.any():
            features = list(X.columns)
            for _column in features:
                if X[_column].isnull().values.any():
                    _X[_column] = self._fill(_X[_column])

        self._fitted = True

        return _X

    def _fill(self, X: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:

        if self.method == "mean":
            X = X.fillna(np.nanmean(X))
        elif self.method == "zero":
            X = X.fillna(0)
        elif self.method == "median":
            X = X.fillna(np.nanmedian(X))
        elif self.method == "most frequent":
            X = X.fillna(X.value_counts().index[0])
        else:
            X = X.fillna(self.method)

        return X


class DummyImputer:

    """
    Create dummy variable for nan values and fill the original feature with 0
    The idea is that there are possibilities that the nan values are critically related to response, create dummy
    variable to identify the relationship

    Parameters
    ----------
    force: whether to force dummy coding for nan values, default = False
    if force == True, all nan values will create dummy variables, otherwise, only nan values that creates impact
    on response will create dummy variables

    threshold: threshold whether to create dummy variable, default = 0.1
    if mean of nan response and mean of non nan response is different above threshold, threshold will be created

    method: the method to fill nan values for columns not reaching threshold, default = 'mean'
    supproted methods ['mean', 'zero', 'median', 'most frequent', constant]
    'mean' : fill columns with nan values using mean of non nan values
    'zero': fill columns with nan values using 0
    'median': fill columns with nan values using median of non nan values
    'most frequent': fill columns with nan values using most frequent of non nan values
    constant: fill columns with nan values using predefined values
    """

    def __init__(
        self, force: bool = False, threshold: float = 0.1, method: str = "zero"
    ) -> None:
        self.force = force
        self.threshold = threshold
        self.method = method

        self._fitted = False  # whether the imputer has been fitted

    def fill(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        _X = X.copy(deep=True)

        if _X.isnull().values.any():
            _X = self._fill(_X, y)

        self._fitted = True

        return _X

    def _fill(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        features = list(X.columns)

        for _column in features:
            if X[_column].isnull().values.any():
                _mean_nan = y[X[_column].isnull()].mean()
                _mean_non_nan = y[~X[_column].isnull()].mean()
                if abs(_mean_nan / _mean_non_nan - 1) >= self.threshold:
                    X[_column + "_nan"] = X[_column].isnull().astype(int)
                    X[_column] = X[_column].fillna(0)
                else:
                    if self.method == "mean":
                        X[_column] = X[_column].fillna(np.nanmean(X[_column]))
                    elif self.method == "zero":
                        X[_column] = X[_column].fillna(0)
                    elif self.method == "median":
                        X[_column] = X[_column].fillna(np.nanmedian(X[_column]))
                    elif self.method == "most frequent":
                        X[_column] = X[_column].fillna(
                            X[_column].value_counts().index[0]
                        )
                    else:
                        X[_column] = X[_column].fillna(self.method)

        return X


class JointImputer:

    """
    Impute the missing values assume a joint distribution, default as multivariate Gaussian distribution
    """

    def __init__(self, kernel: str = "normal") -> None:
        self.kernel = kernel

        self._fitted = False  # whether the imputer has been fitted

    def fill(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        _X = X.copy(deep=True)

        if _X.isnull().values.any():
            _X = self._fill(_X)

        self._fitted = True

        return _X

    def _fill(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        rows = list(X.index)
        for _row in rows:
            if X.loc[_row, :].isnull().values.any():
                X.loc[_row, :] = self._fill_row(_row, X)

        return X

    def _fill_row(
        self, row_index: Union[int, str], X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        """
        for x = (x_{mis}, x_{obs})^{T} with \mu = (\mu_{mis}, \mu_{obs}).T and \Sigma = ((Sigma_{mis, mis},
        Sigma_{mis, obs}), (Sigma_{obs, Sigma}, Sigma_{obs, obs})),
        Conditional distribution x_{mis}|x_{obs} = a is N(\bar(\mu), \bar(\Sigma))
        where \bar(\mu) = \mu_{mis} + \Sigma_{mis, obs}\Sigma_{obs, obs}^{-1}(a - \mu_{obs})
        and \bar(\Sigma) = \Sigma_{mis, mis} - \Sigma_{mis, obs}\Sigma_{obs, obs}^{-1}\Sigma_{obs, mis}

        in coding, 1 = mis, 2 = obs for simpilicity
        """

        _mis_column = np.argwhere(X.loc[row_index, :].isnull().values).T[0]
        _obs_column = [i for i in range(len(list(X.columns)))]
        for item in _mis_column:
            _obs_column.remove(item)

        _mu_1 = np.nanmean(X.iloc[:, _mis_column], axis=0).T.reshape(
            len(_mis_column), 1
        )
        _mu_2 = np.nanmean(X.iloc[:, _obs_column], axis=0).T.reshape(
            len(_obs_column), 1
        )

        _sigma_11 = nan_cov(X.iloc[:, _mis_column].values)
        _sigma_22 = nan_cov(X.iloc[:, _obs_column].values)
        _sigma_12 = nan_cov(
            X.iloc[:, _mis_column].values, y=X.iloc[:, _obs_column].values
        )
        _sigma_21 = nan_cov(
            X.iloc[:, _obs_column].values, y=X.iloc[:, _mis_column].values
        )

        _a = X.loc[row_index, ~X.loc[row_index, :].isnull()].values.T.reshape(
            len(_obs_column), 1
        )
        _mu = _mu_1 + _sigma_12 @ np.linalg.inv(_sigma_22) @ (_a - _mu_2)
        _mu = _mu[0]  # multivariate_normal only accept 1 dimension mean
        _sigma = _sigma_11 - _sigma_12 @ np.linalg.inv(_sigma_22) @ _sigma_21

        X.loc[row_index, X.loc[row_index, :].isnull()] = np.random.multivariate_normal(
            mean=_mu, cov=_sigma, size=(X.loc[row_index, :].isnull().values.sum(), 1)
        )

        return X.loc[row_index, :]
