"""
File Name: _stat.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_utils/_stat.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 14th November 2022 9:44:56 pm
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

from typing import Union, List, Tuple
import warnings
import numpy as np
import pandas as pd
import scipy.stats
import copy

# return non-nan covariance matrix between X and y, (return covariance of X if y = None)
# default calculate at columns (axis = 0), axis = 1 at rows
def nan_cov(
    X: Union[pd.DataFrame, pd.Series, np.ndarray],
    y: Union[pd.DataFrame, pd.Series, np.ndarray] = None,
    axis: int = 0,
) -> np.ndarray:

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
def class_means(
    X: Union[pd.DataFrame, pd.Series, np.ndarray],
    y: Union[pd.DataFrame, pd.Series, np.ndarray],
) -> List[float]:

    _class = np.unique(y)
    result = []

    for _cl in _class:
        data = X.loc[y.values == _cl]
        result.append(np.mean(data, axis=0).values)

    return result


# return maximum likelihood estimate for covariance
def empirical_covariance(
    X: Union[pd.DataFrame, pd.Series, np.ndarray], *, assume_centered: bool = False
) -> np.ndarray:

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
def class_cov(
    X: Union[pd.DataFrame, pd.Series, np.ndarray],
    y: Union[pd.DataFrame, pd.Series, np.ndarray],
    priors: Union[pd.DataFrame, pd.Series, np.ndarray],
) -> np.ndarray:

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    _class = np.unique(y)
    cov = np.zeros(shape=(X.shape[1], X.shape[1]))
    for idx, _cl in enumerate(_class):
        _data = X.loc[y.values == _cl, :]
        cov += priors[idx] * empirical_covariance(_data)

    return cov


# return Pearson Correlation Coefficients
def Pearson_Corr(
    X: Union[pd.DataFrame, pd.Series, np.ndarray],
    y: Union[pd.DataFrame, pd.Series, np.ndarray],
) -> List:

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    features = list(X.columns)

    result = [
        (nan_cov(X[_column], y) / np.sqrt(nan_cov(X[_column]) * nan_cov(y)))[0][0]
        for _column in features
    ]
    # Oct. 10 decrypted:
    # result = []
    # for _column in features:
    #     result.append(
    #         (nan_cov(X[_column], y) / np.sqrt(nan_cov(X[_column]) * nan_cov(y)))[0][0]
    #     )

    return result


# return Mutual Information
def MI(
    X: Union[pd.DataFrame, pd.Series, np.ndarray],
    y: Union[pd.DataFrame, pd.Series, np.ndarray],
) -> List[float]:

    if len(X) != len(y):
        raise ValueError("X and y not same size!")

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(y, pd.DataFrame):
        y = pd.DataFrame(y)

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
def t_score(
    X: Union[pd.DataFrame, pd.Series, np.ndarray],
    y: Union[pd.DataFrame, pd.Series, np.ndarray],
    fvalue: bool = True,
    pvalue: bool = False,
) -> Union[float, Tuple[float, float]]:

    if len(X) != len(y):
        raise ValueError("X and y not same size!")

    features = list(X.columns)
    if len(y.shape) > 1:
        y = y.iloc[:, 0]  # only accept one column of response

    _group = y.unique()
    if len(_group) != 2:
        raise ValueError(
            "Only 2 group datasets are acceptable, get {}.".format(len(_group))
        )

    _f = []
    _p = []

    for _col in features:
        t_test = scipy.stats.ttest_ind(
            X.loc[y == _group[0], _col],
            X.loc[y == _group[1], _col],
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
def ANOVA(
    X: Union[pd.DataFrame, pd.Series, np.ndarray],
    y: Union[pd.DataFrame, pd.Series, np.ndarray],
    fvalue: bool = True,
    pvalue: bool = False,
) -> Union[float, Tuple[float, float]]:

    if len(X) != len(y):
        raise ValueError("X and y not same size!")

    features = list(X.columns)
    if len(y.shape) > 1:
        y = y.iloc[:, 0]  # only accept one column of response

    _group = y.unique()

    _f = []
    _p = []

    for _col in features:
        _group_value = []
        for _g in _group:
            _group_value.append(X.loc[y == _g, _col])
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


# convert metrics to minimize the error (as a loss function)
# add negative sign to make maximization to minimize
def neg_R2(
    y_true: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_pred: Union[pd.DataFrame, pd.Series, np.ndarray],
) -> float:

    from sklearn.metrics import r2_score

    return -r2_score(y_true, y_pred)


def neg_accuracy(
    y_true: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_pred: Union[pd.DataFrame, pd.Series, np.ndarray],
) -> float:

    from sklearn.metrics import accuracy_score

    return -accuracy_score(y_true, y_pred)


def neg_precision(
    y_true: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_pred: Union[pd.DataFrame, pd.Series, np.ndarray],
) -> float:

    from sklearn.metrics import precision_score

    return -precision_score(y_true, y_pred)


def neg_auc(
    y_true: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_pred: Union[pd.DataFrame, pd.Series, np.ndarray],
) -> float:

    from sklearn.metrics import roc_auc_score

    return -roc_auc_score(y_true, y_pred)


def neg_hinge(
    y_true: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_pred: Union[pd.DataFrame, pd.Series, np.ndarray],
) -> float:

    from sklearn.metrics import hinge_loss

    return -hinge_loss(y_true, y_pred)


def neg_f1(
    y_true: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_pred: Union[pd.DataFrame, pd.Series, np.ndarray],
) -> float:

    from sklearn.metrics import f1_score

    return -f1_score(y_true, y_pred)


def ACCC(
    Z: Union[pd.DataFrame, pd.Series, np.ndarray],
    y: Union[pd.DataFrame, pd.Series, np.ndarray],
    X: Union[pd.DataFrame, pd.Series, np.ndarray] = None,
) -> float:

    """
    Empirical implementation of Azadkia-Chatterjee Correlation Coefficient (ACCC) [1]

    if X is None:   correlation between Y (response) and Z (features);

    T_{n}(Y, Z)=\dfrac{\sum_{i=1}^{n}(n\min{R_{i}, R_{M(i)}} - L_{i}^{2})}{\sum_{i=1}^{n}L_{i}(n-L_{i})}

    if X not None:  conditional between Y and Z given X

    T_{n}(Y,Z|X)=\dfrac{\sum_{i=1}^{n}(\min{R_{i}, R_{M(i)}} - \min{R_{i}, R_{N(i)}})}{\sum_{i}^{n}(R_{i}-\min{R_{i}, R_{N(i)}})}

    where R_{i} is the rank of the ith observation in Y, L_{i} is the number of j such that Y_{j}>=Y_{i}, N_{i} be the index of j such that X_{j} is nearest neighbor of X_{i} and M_{i} be the index of j such that (X_{j}, Z_{j}) is the farthest neighbor of (X_{i}, Z_{i}).

    [1] Azadkia, A., & Chatterjee, S. (2015). A new correlation coefficient. Journal of Statistical Theory and Practice, 9(2), 107-118.
    """

    # format data to dataframe
    if not isinstance(Z, pd.DataFrame):
        Z = pd.DataFrame(Z)
    if not isinstance(y, pd.DataFrame):
        y = pd.DataFrame(y)
    if (X is not None) and not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # check shape
    if not len(Z) == len(y):
        raise ValueError(
            "Z and y must have the same length, but Z has length %d and y has length %d"
            % (len(Z), len(y))
        )
    if not (X is None) and not len(y) == len(X):
        raise ValueError(
            "X and y must have the same length, but X has length %d and y has length %d"
            % (len(X), len(y))
        )

    from sklearn.neighbors import KNeighborsRegressor

    # rank of response variable
    y_rank = np.argsort(y.values.reshape(1, -1)[0])

    # shape of features
    n, p = Z.shape

    if X is None:
        # KNN for the ranking
        # only need rank of y corresponding nearest neighbor of Z
        knn_M = KNeighborsRegressor(
            n_neighbors=2, p=2, metric="minkowski"
        )  # standard Euclidean distance
        knn_M.fit(Z, y_rank)
        _, ind = knn_M.kneighbors(Z)
        ind = ind[:, -1]  # get second closest since the closest will be itself
        L = [(y >= item).sum().values for item in y.values]
        L = np.concatenate(L).ravel()

        nume = np.sum(
            [
                n * min(_y_rank, y_rank[_i]) - L[idx] ** 2
                for idx, (_y_rank, _i) in enumerate(zip(y_rank, ind))
            ]
        )
        denom = np.sum([L[i] * (n - L[i]) for i in range(n)])

        return nume / denom
    else:
        # kNN for the ranking
        # need rank of y corresponding nearest neighbor of Z and nearest neighbor of (X, Z)
        knn_M = KNeighborsRegressor(
            n_neighbors=2, p=2, metric="minkowski"
        )  # standard Euclidean distance
        knn_M.fit(pd.concat([X, Z], axis=1, ignore_index=True), y_rank)
        _, ind_M = knn_M.kneighbors(pd.concat([X, Z], axis=1, ignore_index=True))
        ind_M = ind_M[:, -1]  # get second closest since the closest will be itself

        knn_N = KNeighborsRegressor(
            n_neighbors=2, p=2, metric="minkowski"
        )  # standard Euclidean distance
        knn_N.fit(X, y_rank)
        _, ind_N = knn_N.kneighbors(X)
        ind_N = ind_N[:, -1]  # get second closest since the closest will be itself

        nume = np.sum(
            [
                min(_y_rank, y_rank[_ind_M]) - min(_y_rank, y_rank[_ind_N])
                for _y_rank, _ind_M, _ind_N in zip(y_rank, ind_M, ind_N)
            ]
        )
        denom = np.sum(
            [
                _y_rank - min(_y_rank, y_rank[_ind_N])
                for _y_rank, _ind_N in zip(y_rank, ind_N)
            ]
        )

        return nume / denom
