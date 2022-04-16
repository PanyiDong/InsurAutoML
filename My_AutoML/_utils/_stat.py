"""
File: _stat.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Latest Version: 0.2.0
Relative Path: /My_AutoML/_utils/_stat.py
File Created: Wednesday, 6th April 2022 12:02:53 am
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Friday, 15th April 2022 12:52:53 pm
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

import warnings
import numpy as np
import pandas as pd
import scipy.stats
import copy

# return non-nan covariance matrix between X and y, (return covariance of X if y = None)
# default calculate at columns (axis = 0), axis = 1 at rows
def nan_cov(X, y=None, axis=0):

    if isinstance(y, pd.DataFrame):
        _empty = y.isnull().all().all()
    elif isinstance(y, pd.Series):
        _empty = y.isnull().all()
    elif isinstance(y, np.ndarray):
        _empty = np.all(np.isnan(y))
    else:
        _empty = y == None

    if _empty:
        y = copy.deepcopy(X)
    else:
        y = y

    if axis == 0:
        if len(X) != len(y):
            raise ValueError("X and y must have same length of rows!")
    elif axis == 1:
        if len(X[0]) != len(y[0]):
            raise ValueError("X and y must have same length of columns!")

    X = np.array(X)
    y = np.array(y)

    # reshape the X/y
    try:
        X.shape[1]
    except IndexError:
        X = X.reshape(len(X), 1)

    try:
        y.shape[1]
    except IndexError:
        y = y.reshape(len(y), 1)

    _x_mean = np.nanmean(X, axis=axis)
    _y_mean = np.nanmean(y, axis=axis)

    _cov = np.array(
        [[0.0 for _i in range(y.shape[1 - axis])] for _j in range(X.shape[1 - axis])]
    )  # initialize covariance matrix

    for i in range(_cov.shape[0]):
        for j in range(_cov.shape[1]):
            if axis == 0:
                _cov[i, j] = np.nansum(
                    (X[:, i] - _x_mean[i]) * (y[:, j] - _y_mean[j])
                ) / (len(X) - 1)
            elif axis == 1:
                _cov[i, j] = np.nansum(
                    (X[i, :] - _x_mean[i]) * (y[j, :] - _y_mean[j])
                ) / (len(X[0]) - 1)

    return _cov


# return class (unique in y) mean of X
def class_means(X, y):

    _class = np.unique(y)
    result = []

    for _cl in _class:
        data = X.loc[y.values == _cl]
        result.append(np.mean(data, axis=0).values)

    return result


# return maximum likelihood estimate for covariance
def empirical_covariance(X, *, assume_centered=False):

    X = np.asarray(X)

    if X.ndim == 1:
        X = np.reshape(X, (1, -1))

    if X.shape[0] == 1:
        warnings.warn("Only one data sample available!")

    if assume_centered:
        covariance = np.dot(X.T, X) / X.shape[0]
    else:
        covariance = np.cov(X.T, bias=1)

    if covariance.ndim == 0:
        covariance = np.array([[covariance]])
    return covariance


# return weighted within-class covariance matrix
def class_cov(X, y, priors):

    _class = np.unique(y)
    cov = np.zeros(shape=(X.shape[1], X.shape[1]))
    for idx, _cl in enumerate(_class):
        _data = X.loc[y.values == _cl, :]
        cov += priors[idx] * empirical_covariance(_data)
    return cov


# return Pearson Correlation Coefficients
def Pearson_Corr(X, y):

    features = list(X.columns)
    result = []
    for _column in features:
        result.append(
            (nan_cov(X[_column], y) / np.sqrt(nan_cov(X[_column]) * nan_cov(y)))[0][0]
        )

    return result


# return Mutual Information
def MI(X, y):

    if len(X) != len(y):
        raise ValueError("X and y not same size!")

    features = list(X.columns)
    _y_column = list(y.columns)
    result = []

    _y_pro = y.groupby(_y_column[0]).size().div(len(X)).values
    _H_y = -sum(item * np.log(item) for item in _y_pro)

    for _column in features:

        _X_y = pd.concat([X[_column], y], axis=1)
        _pro = (
            _X_y.groupby([_column, _y_column[0]]).size().div(len(X))
        )  # combine probability (x, y)
        _pro_val = _pro.values  # take only values
        _X_pro = X[[_column]].groupby(_column).size().div(len(X))  # probability (x)
        _H_y_X = -sum(
            _pro_val[i]
            * np.log(
                _pro_val[i] / _X_pro.loc[_X_pro.index == _pro.index[i][0]].values[0]
            )
            for i in range(len(_pro))
        )
        result.append(_H_y - _H_y_X)

    return result


# return t-statistics of dataset, only two groups dataset are suitable
def t_score(X, y, fvalue=True, pvalue=False):

    if len(X) != len(y):
        raise ValueError("X and y not same size!")

    features = list(X.columns)
    _y_column = list(y.columns)[0]  # only accept one column of response

    _group = y[_y_column].unique()
    if len(_group) != 2:
        raise ValueError(
            "Only 2 group datasets are acceptable, get {}.".format(len(_group))
        )

    _f = []
    _p = []

    for _col in features:
        t_test = scipy.stats.ttest_ind(
            X.loc[y[_y_column] == _group[0], _col],
            X.loc[y[_y_column] == _group[1], _col],
        )
        if fvalue:
            _f.append(t_test[0])
        if pvalue:
            _p.append(t_test[1])

    if fvalue and pvalue:
        return _f, _p
    elif fvalue:
        return _f
    elif pvalue:
        return _p


# return ANOVA of dataset, more than two groups dataset are suitable
def ANOVA(X, y, fvalue=True, pvalue=False):

    if len(X) != len(y):
        raise ValueError("X and y not same size!")

    features = list(X.columns)
    _y_column = list(y.columns)[0]  # only accept one column of response

    _group = y[_y_column].unique()

    _f = []
    _p = []

    for _col in features:
        _group_value = []
        for _g in _group:
            _group_value.append(X.loc[y[_y_column] == _g, _col].flatten())
        _test = scipy.stats.f_oneway(*_group_value)
        if fvalue:
            _f.append(_test[0])
        if pvalue:
            _p.append(_test[1])

    if fvalue and pvalue:
        return _f, _p
    elif fvalue:
        return _f
    elif pvalue:
        return _p
