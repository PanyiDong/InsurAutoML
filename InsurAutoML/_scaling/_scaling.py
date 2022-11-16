"""
File Name: _scaling.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_scaling/_scaling.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 14th November 2022 8:16:33 pm
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

from typing import Union, Tuple, List, Dict
from logging import warning
from random import random
from re import L
import warnings
import numpy as np
import pandas as pd
import scipy
import scipy.stats

from InsurAutoML._encoding import DataEncoding


class NoScaling:
    def __init__(self) -> None:

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> NoScaling:

        self._fitted = True

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return X

    def inverse_transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        if not self._fitted:
            raise ValueError(
                "Model has not been fitted yet! Can't perform inverse transform."
            )

        self._fitted = False
        return X


class Standardize:

    """
    Standardize the dataset by column (each feature), using _x = (x - mean) / std

    Parameters
    ----------
    with_mean: whether to standardize with mean, default = True

    with_std: whether to standardize with standard variance, default = True
    """

    def __init__(
        self, with_mean: bool = True, with_std: bool = True, deep_copy: bool = True
    ) -> None:
        self.with_mean = with_mean
        self.with_std = with_std
        self.deep_copy = deep_copy

        self._fitted = False  # record whether the model has been fitted

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> Standardize:

        _X = X.copy(deep=self.deep_copy)

        n, p = _X.shape
        if self.with_mean == True:
            self._mean = [0 for _ in range(p)]
        if self.with_std == True:
            self._std = [0 for _ in range(p)]

        for i in range(p):
            _data = _X.iloc[:, i].values
            n_notnan = n - np.isnan(_data).sum()
            _x_sum = 0
            _x_2_sum = 0
            _x_sum += np.nansum(_data)
            _x_2_sum += np.nansum(_data**2)
            if self.with_mean == True:
                self._mean[i] = _x_sum / n_notnan
            if self.with_std == True:
                self._std[i] = np.sqrt(
                    (_x_2_sum - n_notnan * ((_x_sum / n_notnan) ** 2)) / (n_notnan - 1)
                )

        self._fitted = True

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        _X = X.copy(deep=self.deep_copy)
        if self.with_mean:
            _X -= self._mean
        if self.with_std:
            _X /= self._std

        return _X

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> Union[pd.DataFrame, np.ndarray]:

        _X = X.copy(deep=self.deep_copy)

        self.fit(_X, y)

        self._fitted = True

        _X = self.transform(_X)

        return _X

    def inverse_transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        if not self._fitted:
            raise ValueError(
                "Model has not been fitted yet! Can't perform inverse transform."
            )

        if self.with_mean:
            X += self._mean
        if self.with_std:
            X *= self._std

        self._fitted = False

        return X


class Normalize:

    """
    Normalize features with x / x_

    Parameters
    ----------
    norm: how to select x_, default = 'max'
    supported ['l1', 'l2', 'max']
    """

    def __init__(
        self,
        norm: str = "max",
        deep_copy: bool = True,
    ) -> None:
        self.norm = norm
        self.deep_copy = deep_copy

        self._fitted = False  # record whether the model has been fitted

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> Normalize:

        if self.norm not in ["l1", "l2", "max"]:
            raise ValueError("Not recognizing norm method!")

        _X = X.copy(deep=self.deep_copy)
        n, p = _X.shape
        self._scale = [0 for _ in range(p)]

        for i in range(p):
            _data = _X.iloc[:, i].values
            if self.norm == "max":
                self._scale[i] = np.max(np.abs(_data))
            elif self.norm == "l1":
                self._scale[i] = np.abs(_data).sum()
            elif self.norm == "l2":
                self._scale[i] = (_data**2).sum()

        self._fitted = True

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        _X = X.copy(deep=self.deep_copy)
        _X /= self._scale

        return _X

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> Union[pd.DataFrame, np.ndarray]:

        _X = X.copy(deep=self.deep_copy)

        self.fit(_X, y)

        self._fitted = True

        _X = self.transform(_X)

        return _X

    def inverse_transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        if not self._fitted:
            raise ValueError(
                "Model has not been fitted yet! Can't perform inverse transform."
            )

        X *= self._scale

        self._fitted = False

        return X


class RobustScale:

    """
    Use quantile to scale, x / (q_max - q_min)

    Parameters
    ----------
    with_centering: whether to standardize with median, default = True

    with_std: whether to standardize with standard variance, default = True

    quantile: (q_min, q_max), default = (25.0, 75.0)

    uni_variance: whether to set unit variance for scaled data, default = False
    """

    def __init__(
        self,
        with_centering: bool = True,
        with_scale: bool = True,
        quantile: Tuple[float, float] = (25.0, 75.0),
        unit_variance: bool = False,
        deep_copy: bool = True,
    ) -> None:
        self.with_centering = with_centering
        self.with_scale = with_scale
        self.quantile = quantile
        self.unit_variance = unit_variance
        self.deep_copy = deep_copy

        self._fitted = False  # record whether the model has been fitted

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> RobustScale:

        q_min, q_max = self.quantile
        if q_min == None:  # in case no input
            q_min = 25.0
        if q_max == None:
            q_max = 75.0
        if not 0 <= q_min <= q_max <= 100.0:
            raise ValueError(
                "Quantile not in range, get {0:.1f} and {1:.1f}!".format(q_min, q_max)
            )

        _X = X.copy(deep=self.deep_copy)
        n, p = _X.shape
        if self.with_centering == True:
            self._median = [0 for _ in range(p)]
        if self.with_scale == True:
            self._scale = [0 for _ in range(p)]

        for i in range(p):
            _data = _X.iloc[:, i].values
            if self.with_centering == True:
                self._median[i] = np.nanmedian(_data)
                quantile = np.nanquantile(_data, (q_min / 100, q_max / 100))
                quantile = np.transpose(quantile)
                self._scale[i] = quantile[1] - quantile[0]
                if self.unit_variance == True:
                    self._scale[i] = self.scale[i] / (
                        scipy.stats.norm.ppf(q_max / 100.0)
                        - scipy.stats.norm.ppf(q_min / 100.0)
                    )

        # handle 0 in scale
        constant_mask = (
            self._scale < 10 * np.finfo(np.float64).eps
        )  # avoid extremely small values
        for index in [i for i, value in enumerate(constant_mask) if value]:
            self._scale[
                index
            ] = 1.0  # change scale at True index of constant_mask to 1.0

        self._fitted = True

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        _X = X.copy(deep=self.deep_copy)

        if self.with_centering == True:
            _X -= self._median
        if self.with_scale == True:
            _X /= self._scale

        return _X

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> Union[pd.DataFrame, np.ndarray]:

        _X = X.copy(deep=self.deep_copy)

        self.fit(_X, y)

        self._fitted = True

        _X = self.transform(_X)

        return _X

    def inverse_transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        if not self._fitted:
            raise ValueError(
                "Model has not been fitted yet! Can't perform inverse transform."
            )

        if self.with_centering:
            X += self._median
        if self.with_scale:
            X *= self._scale

        self._fitted = False

        return X


class MinMaxScale:

    """
    Use min_max value to scale the feature, x / (x_max - x_min)

    Parameters
    ----------
    feature_range: (feature_min, feature_max) to scale the feature, default = (0, 1)
    """

    def __init__(
        self,
        feature_range: Tuple[float, float] = (0.0, 1.0),
        deep_copy: bool = True,
    ) -> None:
        self.feature_range = feature_range
        self.deep_copy = deep_copy

        self._fitted = False  # record whether the model has been fitted

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> MinMaxScale:

        _X = X.copy(deep=self.deep_copy)
        n, p = _X.shape

        self._min = [0 for _ in range(p)]
        self._max = [0 for _ in range(p)]

        for i in range(p):
            _data = _X.iloc[:, i].values
            self._min[i] = np.nanmin(_data)
            self._max[i] = np.nanmax(_data)

        self._fitted = True

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        f_min, f_max = self.feature_range
        if not f_min < f_max:
            raise ValueError("Minimum of feature range must be smaller than maximum!")

        _X = X.copy(deep=self.deep_copy)
        _X = (_X - self._min) / (np.array(self._max) - np.array(self._min))
        _X = _X * (f_max - f_min) + f_min

        return _X

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> Union[pd.DataFrame, np.ndarray]:

        _X = X.copy(deep=self.deep_copy)

        self.fit(_X, y)

        self._fitted = True

        _X = self.transform(_X)

        return _X

    def inverse_transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        if not self._fitted:
            raise ValueError(
                "Model has not been fitted yet! Can't perform inverse transform."
            )

        f_min, f_max = self.feature_range
        if not f_min < f_max:
            raise ValueError("Minimum of feature range must be smaller than maximum!")

        _X = X.copy(deep=True)
        _X = (_X - f_min) / (f_max - f_min)
        _X = _X * (np.array(self._max) - np.array(self._min)) + self._min

        self._fitted = False

        return _X


class Winsorization:

    """
    Limit feature to certain quantile (remove the effect of extreme values)
    if the response of extreme values are different than non extreme values above threshold, the feature will
    be capped

    No inverse transform available for Winsorization

    Parameters
    ----------
    quantile: quantile to be considered as extreme, default = 0.95

    threshold: threshold to decide whether to cap feature, default = 0.1
    """

    def __init__(
        self,
        quantile: float = 0.95,
        threshold: float = 0.1,
        deep_copy: bool = True,
    ) -> None:
        self.quantile = quantile
        self.threshold = threshold
        self.deep_copy = deep_copy

        self._fitted = False  # record whether the model has been fitted

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]
    ) -> Winsorization:

        if not isinstance(y, pd.DataFrame) and not isinstance(X, pd.Series):
            warnings.warn("Method Winsorization requires response, but not getting it.")

        _X = X.copy(deep=self.deep_copy)

        features = list(_X.columns)
        self._quantile_list = []
        self._list = []

        for _column in features:
            quantile = np.nanquantile(_X[_column], self.quantile, axis=0)
            self._quantile_list.append(quantile)
            _above_quantile = y[_X[_column] > quantile].mean()[0]
            _below_quantile = y[_X[_column] <= quantile].mean()[0]
            # deal with the case where above quantile do not exists
            if not _above_quantile:
                _above_quantile = quantile

            if abs(_above_quantile / _below_quantile - 1) > self.threshold:
                self._list.append(True)
            else:
                self._list.append(False)

        self._fitted = True

        return self

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> Union[pd.DataFrame, np.ndarray]:

        _X = X.copy(deep=self.deep_copy)

        self.fit(_X, y)

        self._fitted = True

        _X = self.transform(_X)

        return _X

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        _X = X.copy(deep=self.deep_copy)
        features = list(_X.columns)
        i = 0

        for _column in features:
            if self._list[i]:
                _X.loc[
                    _X[_column] > self._quantile_list[i], _column
                ] = self._quantile_list[i]
            i += 1

        return _X


class PowerTransformer:

    """
    PowerTransformer, implemented by sklearn, is a transformer that applies
    a power function to each feature.

    [1] I.K. Yeo and R.A. Johnson, "A new family of power transformations to
        improve normality or symmetry." Biometrika, 87(4), pp.954-959,
        (2000).
    [2] G.E.P. Box and D.R. Cox, "An Analysis of Transformations", Journal
        of the Royal Statistical Society B, 26, 211-252 (1964).

    Parameters
    ----------
    method: 'yeo-johnson' or 'box-cox', default = 'yeo-johnson'
        'yeo-johnson' [1]_, works with positive and negative values
        'box-cox' [2]_, only works with strictly positive values

    standardize: boolean, default = True

    deep_copy: whether to use deep copy, default = True
    """

    def __init__(
        self,
        method: str = "yeo-johnson",
        standardize: bool = True,
        deep_copy: bool = True,
    ) -> None:
        self.method = method
        self.standardize = standardize
        self.deep_copy = deep_copy

        self._fitted = False  # record whether the model has been fitted

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> PowerTransformer:

        # check if the inputs are dataframe
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # record features and response
        self.features = list(X.columns)

        from sklearn.preprocessing import PowerTransformer

        self.mol = PowerTransformer(
            method=self.method,
            standardize=self.standardize,
            copy=self.deep_copy,
        )

        self.mol.fit(X, y)

        self._fitted = True

        return self

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> Union[pd.DataFrame, np.ndarray]:

        return pd.DataFrame(self.mol.transform(X), columns=self.features)

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> Union[pd.DataFrame, np.ndarray]:

        self.fit(X, y)

        self._fitted = True

        return self.transform(X)

    def inverse_transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        if not self._fitted:
            raise ValueError(
                "Model has not been fitted yet! Can't perform inverse transform."
            )

        self._fitted = False

        return pd.DataFrame(self.mol.inverse_transform(X), columns=self.features)


class QuantileTransformer:

    """
    QuantileTransformer, implemented by sklearn

    Parameters
    ----------
    n_quantiles: Number of quantiles to be computed, default = 1000

    output_distribution: 'normal' or 'uniform', default = 'normal'

    ignore_implicit_zeros: Only applies to sparse matrices, default = False

    subsample: Maximum number of samples used to estimate the quantiles, default = 100000

    random_state: RandomState instance or None, default = None

    deep_copy: whether to use deep copy, default = True
    """

    def __init__(
        self,
        n_quantiles: int = 1000,
        output_distribution: str = "uniform",
        ignore_implicit_zeros: bool = False,
        subsample: int = int(1e5),
        random_state: int = None,
        deep_copy: bool = True,
    ) -> None:
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.ignore_implicit_zeros = ignore_implicit_zeros
        self.subsample = subsample
        self.random_state = random_state
        self.deep_copy = deep_copy

        self._fitted = False  # record whether the model has been fitted

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> QuantileTransformer:

        # check if the inputs are dataframe
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # record features and response
        self.features = list(X.columns)

        # limit max number of quantiles to entries number
        self.n_quantiles = min(self.n_quantiles, X.shape[0])

        from sklearn.preprocessing import QuantileTransformer

        self.mol = QuantileTransformer(
            n_quantiles=self.n_quantiles,
            output_distribution=self.output_distribution,
            ignore_implicit_zeros=self.ignore_implicit_zeros,
            subsample=self.subsample,
            random_state=self.random_state,
            copy=self.deep_copy,
        )

        self.mol.fit(X, y)

        self._fitted = True

        return self

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> Union[pd.DataFrame, np.ndarray]:

        return pd.DataFrame(self.mol.transform(X), columns=self.features)

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> Union[pd.DataFrame, np.ndarray]:

        self.fit(X, y)

        self._fitted = True

        return self.transform(X)

    def inverse_transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        if not self._fitted:
            raise ValueError(
                "Model has not been fitted yet! Can't perform inverse transform."
            )

        self._fitted = False

        return pd.DataFrame(self.mol.inverse_transform(X), columns=self.features)


####################################################################################################
# Special Case
def Feature_Manipulation(
    X: pd.DataFrame,
    columns: List[str] = [],
    manipulation: List[str] = [],
    rename_columns: Dict[str, str] = {},
    replace: bool = False,
    deep_copy: bool = False,
) -> pd.DataFrame:

    """
    Available methods: +, -, *, /, //, %, ln, log2, log10, exp

    Parameters
    ----------
    columns: columns need manipulation, default = []

    manipulation: list of manipulation, default = []

    rename_columns: specific changing column names, default = {}

    replace: whether to replace the new columns, default = False

    deep_copy: whether need deep copy the input, default = False

    Example
    -------
    >> data = np.arange(15).reshape(5, 3)
    >> data = pd.DataFrame(data, columns = ['column_1', 'column_2', 'column_3'])
    >> data

       column_1  column_2  column_3
    0         0         1         2
    1         3         4         5
    2         6         7         8
    3         9        10        11
    4        12        13        14

    >> data = Feature_Manipulation(
    >>     data, columns= ['column_1', 'column_2', 'column_3'],
    >>     manipulation = ['* 100', 'ln', '+ 1'],
    >>     rename_columns= {'column_2': 'log_column_2'}
    >> )
    >> data

       column_1  column_2  column_3  column_1_* 100  log_column_2  column_3_+ 1
    0         0         1         2               0      0.000000             3
    1         3         4         5             300      1.386294             6
    2         6         7         8             600      1.945910             9
    3         9        10        11             900      2.302585            12
    4        12        13        14            1200      2.564949            15
    """

    # make sure input is dataframe
    if not isinstance(X, pd.DataFrame):
        try:
            X = pd.DataFrame(X)
        except:
            raise ValueError("Expect a dataframe, get {}.".format(type(X)))

    _X = X.copy(deep=deep_copy)

    # if no columns/manipulation specified, raise warning
    if not columns or not manipulation:
        warnings.warn("No manipulation executed.")
        return _X

    # expect one manipulation for one column
    # if not same size, raise Error
    if len(columns) != len(manipulation):
        raise ValueError(
            "Expect same length of columns and manipulation, get {} and {} respectively.".format(
                len(columns), len(manipulation)
            )
        )
    manipulation = dict(zip(columns, manipulation))

    for _column in columns:

        # if observed in rename dict, change column names
        new_column_name = (
            rename_columns[_column] if _column in rename_columns.keys() else _column
        )

        # if not replace, and new column names coincide with old column names
        # new column names = old column names + manipulation
        # for distinguish
        if not replace and new_column_name == _column:
            new_column_name += "_" + manipulation[_column]

        # column manipulation
        if manipulation[_column] == "ln":
            _X[new_column_name] = np.log(_X[_column])
        elif manipulation[_column] == "log2":
            _X[new_column_name] = np.log2(_X[_column])
        elif manipulation[_column] == "log10":
            _X[new_column_name] = np.log10(_X[_column])
        elif manipulation[_column] == "exp":
            _X[new_column_name] = np.exp(_X[_column])
        else:
            exec("_X[new_column_name] = _X[_column]" + manipulation[_column])

    return _X


####################################################################################################
# Feature Truncation
class Feature_Truncation:

    """
    Truncate feature to certain quantile (remove the effect of extreme values)
    No inverse transform available

    Parameters
    ----------
    quantile: quantile to be considered as extreme, default = 0.95
    if quantile less than 0.5, left truncation; else, right truncation

    Example
    -------
    >> scaling = Feature_Truncation(
    >>     columns = ['column_2', 'column_5', 'column_6', 'column_8', 'column_20'],
    >>     quantile = [0.95, 0.95, 0.9, 0.1, 0.8]
    >> )
    >> data = scaling.fit_transform(data)

    (column_2 right truncated at 95 percentile, column_8 left truncated at 10
    percentile, etc.)
    """

    def __init__(
        self,
        columns: List[str] = [],
        quantile: Union[List[float], float] = 0.95,
        deep_copy: bool = False,
    ) -> None:
        self.columns = columns
        self.quantile = quantile
        self.deep_copy = deep_copy

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> Feature_Truncation:

        # make sure input is dataframe
        if not isinstance(X, pd.DataFrame):
            try:
                X = pd.DataFrame(X)
            except:
                raise ValueError("Expect a dataframe, get {}.".format(type(X)))

        _X = X.copy(deep=self.deep_copy)

        self.columns = list(_X.columns) if not self.columns else self.columns

        if isinstance(self.quantile, list):
            if len(self.columns) != len(self.quantile):
                raise ValueError(
                    "Expect same length of columns and quantile, get {} and {} respectively.".format(
                        len(self.columns), len(self.quantile)
                    )
                )
            self.quantile = dict(zip(self.columns, self.quantile))

        self.quantile_list = {}

        for _column in self.columns:
            quantile = np.nanquantile(X[_column], self.quantile[_column], axis=0)
            self.quantile_list[_column] = quantile

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        _X = X.copy(deep=self.deep_copy)

        for _column in self.columns:
            if self.quantile_list[_column] >= 0.5:
                _X.loc[
                    _X[_column] > self.quantile_list[_column], _column
                ] = self.quantile_list[_column]
            else:
                _X.loc[
                    _X[_column] < self.quantile_list[_column], _column
                ] = self.quantile_list[_column]

        return _X

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> Union[pd.DataFrame, np.ndarray]:

        _X = X.copy(deep=self.deep_copy)

        self.fit(X, y)

        _X = self.transform(_X)

        return _X
