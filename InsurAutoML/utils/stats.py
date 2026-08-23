"""
File Name: stats.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.7
Relative Path: /InsurAutoML/utils/stats.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Tuesday, 16th September 2025 9:06:41 pm
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
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import copy

# merge the mean of two datasets


def merge_mean(old, new):
    (n1, mean1), (n2, mean2) = old, new
    return (mean1 * n1 + mean2 * n2) / (n1 + n2)


# merge the standard deviation of two datasets


def merge_std(old, new):
    (n1, mean1, std1), (n2, mean2, std2) = old, new
    return np.sqrt(
        (std1**2 * n1 + std2**2 * n2) / (n1 + n2)
        + n1 * n2 * (mean1 - mean2) ** 2 / (n1 + n2) ** 2
    )


# merge two dict and sum up the values of common keys


def merge_dict(dict1, dict2):
    return {k: dict1.get(k, 0) + dict2.get(k, 0) for k in set(dict1) | set(dict2)}


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
        _empty = y is None

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

    # check range of R2
    # if out of normal range, return np.inf
    _result = -r2_score(y_true, y_pred)

    if _result < -1 or _result > 1:
        return np.inf
    else:
        return _result


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

    if len(y_pred.shape) == 1:
        return -roc_auc_score(y_true, y_pred)
    if y_pred.shape[1] <= 2:
        return -roc_auc_score(y_true, y_pred[:, -1])
    else:
        return -roc_auc_score(y_true, y_pred, multi_class="ovr")


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


def rankdata_random(values, seed: int = None):
    """
    Return ranks of the data, with random tie breaking.
    """

    # set random seed permutation
    np.random.seed(seed)
    perm = np.random.permutation(len(values))
    # set values to numpy array
    values = np.asarray(values)
    # apply rankdata to permuted values
    ranks = scipy.stats.rankdata(values[perm], method="ordinal")

    out = np.empty_like(ranks)
    out[perm] = ranks
    return out

    # return ranks[np.argsort(perm)]


def _rankdata(values, ties_method: str = "average", seed: int = None) -> np.ndarray:
    """
    Wrap for scipy.stats.rankdata with different ties breaking method

    parameters
    ----------
    values : array-like
        Input data to rank.
    ties_method : str, optional
        Method to handle ties. Default is "average".
        List of options includes "average", "min", "max", "ordinal", "random".
    seed : int, optional
        Random seed for tie breaking.

    Returns
    -------
    ndarray
        Ranks of the input data.
    """

    if ties_method == "random":
        return rankdata_random(values, seed=seed)
    else:
        return scipy.stats.rankdata(values, method=ties_method)


def randomNN(idx):
    # index size
    n = len(np.asarray(idx))
    # sample n indices from [0, n-2]
    x = np.random.randint(0, n - 1, size=n)
    # shift the indices that are >= i by 1 to avoid index equals to position
    x += x >= np.arange(n)

    return idx[x]


def _NN(idx, ties_method: str = "random"):
    """
    A list of methods to deal with identical nearest neighbors.
    """
    if ties_method == "random":
        return randomNN(idx)
    elif ties_method == "first":
        return idx[0] * np.ones(len(idx), dtype=int)
    elif ties_method == "last":
        return idx[-1] * np.ones(len(idx), dtype=int)
    elif ties_method == "ordinal":
        return idx


def _CCC(
    X: Union[pd.DataFrame, pd.Series, np.ndarray],
    y: Union[pd.DataFrame, pd.Series, np.ndarray],
    X_ties_method: str = "random",
    y_ties_method: str = "ordinal",
    seed: int = None,
) -> float:
    """_summary_
    Empirical implementation of Chatterjee Correlation Coefficient (CCC) [1][2]

    [1] https://rdrr.io/cran/XICOR/src/R/calculateXI.R
    [2] Chatterjee, S. (2021). A new coefficient of correlation. Journal of the American Statistical Association, 116(536), 2009-2022.
    """
    # format data to np.ndarray
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    if not isinstance(y, np.ndarray):
        y = np.asarray(y)

    # check shape
    if not len(X) == len(y):
        raise ValueError(
            "X and y must have the same length, but X has length %d and y has length %d"
            % (len(X), len(y))
        )

    n = len(y)

    # rank of X, random tie breaking
    PI = _rankdata(X, ties_method=X_ties_method, seed=seed)
    ord = np.argsort(PI)
    # rank of y and -y (numbers larger than y)
    fr = _rankdata(y, ties_method=y_ties_method) / n
    gr = _rankdata(-y, ties_method=y_ties_method) / n

    # rearrange fr by ord
    fr = fr[ord]
    # calculate the coefficient
    A1 = np.sum(np.abs(fr[:-1] - fr[1:])) / (2 * n)
    CU = np.mean(gr * (1 - gr))

    return 1 - A1 / CU


def _ACCC(
    Z: Union[pd.DataFrame, pd.Series, np.ndarray],
    y: Union[pd.DataFrame, pd.Series, np.ndarray],
    X: Union[pd.DataFrame, pd.Series, np.ndarray] = None,
    Z_ties_method: str = "random",
    y_ties_method: str = "ordinal",
    X_ties_method: str = "random",
    mode: str = "T",
    seed: int = None,
) -> float:
    """
    Empirical implementation of Azadkia-Chatterjee Correlation Coefficient (ACCC) [1][2]

    if X is None:   correlation between Y (response) and Z (features);

    T_{n}(Y, Z)=\dfrac{\sum_{i=1}^{n}(n\min{R_{i}, R_{M(i)}} - L_{i}^{2})}{\sum_{i=1}^{n}L_{i}(n-L_{i})}

    if X not None:  conditional between Y and Z given X

    T_{n}(Y,Z|X)=\dfrac{\sum_{i=1}^{n}(\min{R_{i}, R_{M(i)}} - \min{R_{i}, R_{N(i)}})}{\sum_{i}^{n}(R_{i}-\min{R_{i}, R_{N(i)}})}

    where R_{i} is the rank of the ith observation in Y, L_{i} is the number of j such that Y_{j}>=Y_{i}, N_{i} be the index of j such that X_{j} is nearest neighbor of X_{i} and M_{i} be the index of j such that (X_{j}, Z_{j}) is the farthest neighbor of (X_{i}, Z_{i}).

    [1] https://rdrr.io/cran/FOCI/src/R/codecInternal.R
    [2] Azadkia, A., & Chatterjee, S. (2021). A simple measure of conditional dependence. The Annals of Statistics, 49(6), 3070-3102.

    parameters
    ----------
    Z_ties_method: str = "random"
        Method to handle ties in Z+X. Options are "random", "first", "last", and "ordinal".
    y_ties_method: str = "max"
        Method to handle ties in y. Options are "average", "min", "max", "first", "ordinal", "random".
    X_ties_method: str = "random",
        Method to handle ties in X. Options are "random", "first", "last", and "ordinal".
    """

    def tie_helper(Z, a):
        dists = cdist([Z[a]], np.delete(Z, a, axis=0), metric="euclidean")
        min_dist = np.min(dists)
        id = np.where(dists[0] == min_dist)[0]
        choice = np.random.choice(id)
        return choice + (choice >= a)  # shift index if needed

    # format data to numpy array
    Z = np.asarray(Z)
    y = np.asarray(y)
    X = np.asarray(X) if not (X is None) else None

    # make sure Z is 2D
    if len(Z.shape) == 1:
        Z = Z.reshape(-1, 1)
    # make sure X is 2D
    if X is not None and len(X.shape) == 1:
        X = X.reshape(-1, 1)
    # shape of features
    n, p = Z.shape

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

    if X is None:

        # fit 3-NN on Z to get distance and index
        nn = NearestNeighbors(n_neighbors=3, p=2, metric="minkowski")
        nn.fit(Z)
        distance, ind = nn.kneighbors(Z)

        # nearest neighbor index of Z
        nn_index_Z = ind[:, 1]  # get second closest since the closest will be itself
        # repeated points (distance = 0)
        repeat_data = np.where(distance[:, 1] == 0)[0]

        # handle repeated points by random repeated points
        df_Z = pd.DataFrame(
            {
                "id": repeat_data,
                "group": ind[repeat_data, 0],
            }
        )
        df_Z["rnn"] = df_Z.groupby("group")["id"].transform(
            lambda x: _NN(x.values, ties_method=Z_ties_method)
        )
        nn_index_Z[repeat_data] = df_Z["rnn"].values

        # nearest neighbor with tie
        ties = np.where(distance[:, 1] == distance[:, 2])[0]
        ties = np.setdiff1d(ties, repeat_data)  # remove repeated points
        for idx in ties:
            nn_index_Z[idx] = tie_helper(Z, idx)

        # estimate cc
        R_Y = _rankdata(y, ties_method=y_ties_method, seed=seed)
        L_Y = _rankdata(-y, ties_method=y_ties_method, seed=seed)
        S_n = np.sum(L_Y * (n - L_Y)) / (n**3)
        Q_n = np.sum(np.minimum(R_Y, R_Y[nn_index_Z]) - L_Y**2 / n) / (n**2)

    else:
        W = np.hstack((X, Z))

        # fit 3-NN on X to get distance and index
        nn = NearestNeighbors(n_neighbors=3, p=2, metric="minkowski")
        nn.fit(X)
        distance_X, ind_X = nn.kneighbors(X)

        # nearest neighbor index of X
        nn_index_X = ind_X[:, 1]  # get second closest since the closest will be itself
        # repeated points (distance = 0)
        repeat_data_X = np.where(distance_X[:, 1] == 0)[0]

        # handle repeated points by random repeated points
        df_X = pd.DataFrame(
            {
                "id": repeat_data_X,
                "group": ind_X[repeat_data_X, 0],
            }
        )
        df_X["rnn"] = df_X.groupby("group")["id"].transform(
            lambda x: _NN(x.values, ties_method=X_ties_method)
        )
        nn_index_X[repeat_data_X] = df_X["rnn"].values

        # nearest neighbor with tie
        ties_X = np.where(distance_X[:, 1] == distance_X[:, 2])[0]
        ties_X = np.setdiff1d(ties_X, repeat_data_X)  # remove repeated points
        for idx in ties_X:
            nn_index_X[idx] = tie_helper(X, idx)

        # fit 3-NN on (X, Z) to get distance and index
        nn_W = NearestNeighbors(n_neighbors=3, p=2, metric="minkowski")
        nn_W.fit(W)
        distance_W, ind_W = nn_W.kneighbors(W)

        # handle repeat points
        nn_index_W = ind_W[:, 1]  # get second closest since the closest will be itself
        repeat_data_W = np.where(distance_W[:, 1] == 0)[0]
        df_W = pd.DataFrame(
            {
                "id": repeat_data_W,
                "group": ind_W[repeat_data_W, 0],
            }
        )
        df_W["rnn"] = df_W.groupby("group")["id"].transform(
            lambda x: _NN(x.values, ties_method=X_ties_method)
        )
        nn_index_W[repeat_data_W] = df_W["rnn"].values

        # handle ties
        ties_W = np.where(distance_W[:, 1] == distance_W[:, 2])[0]
        ties_W = np.setdiff1d(ties_W, repeat_data_W)
        for idx in ties_W:
            nn_index_W[idx] = tie_helper(X, idx)

        # estimate ccc
        R_Y = _rankdata(y, ties_method=y_ties_method)
        L_Y = _rankdata(-y, ties_method=y_ties_method)
        S_n = np.sum(R_Y - np.minimum(R_Y, R_Y[nn_index_X])) / (n**2)
        Q_n = np.sum(
            np.minimum(R_Y, R_Y[nn_index_W]) - np.minimum(R_Y, R_Y[nn_index_X])
        ) / (n**2)

    if mode == "T":
        return Q_n / S_n if S_n != 0 else 1
    elif mode == "Q":
        return Q_n


"""
NOTE: Sep. 16, 2025
CCC and ACCC are subject to tie-breaking in identical values. And randomness may be introduced in the results.
By default, only one time tie-breaking is performed. If you want to have more stable results, please set fold > 1 as it repeats the calculation multiple times and takes the average.
"""


def CCC(
    X: Union[pd.DataFrame, pd.Series, np.ndarray],
    y: Union[pd.DataFrame, pd.Series, np.ndarray],
    X_ties_method: str = "random",
    y_ties_method: str = "ordinal",
    fold: int = 1,
    seed: int = None,
) -> float:
    """
    Compute CCC in case of random tie breaking (multiple folds).
    """
    # fix random seed
    np.random.seed(seed)
    if "random" in [X_ties_method, y_ties_method]:
        return np.mean(
            [
                _CCC(X, y, X_ties_method=X_ties_method, y_ties_method=y_ties_method)
                for _ in range(fold)
            ]
        )
    else:
        return _CCC(X, y, X_ties_method=X_ties_method, y_ties_method=y_ties_method)


def ACCC(
    Z: Union[pd.DataFrame, pd.Series, np.ndarray],
    y: Union[pd.DataFrame, pd.Series, np.ndarray],
    X: Union[pd.DataFrame, pd.Series, np.ndarray] = None,
    Z_ties_method: str = "random",
    y_ties_method: str = "ordinal",
    X_ties_method: str = "random",
    mode: str = "T",
    fold: int = 1,
    seed: int = 42,
) -> float:
    """
    Compute the ACCC with multiple folds.
    """
    # fix random seed
    np.random.seed(seed)
    if "random" in [Z_ties_method, y_ties_method, X_ties_method]:
        return np.mean(
            [
                _ACCC(Z, y, X, Z_ties_method, y_ties_method, X_ties_method, mode=mode)
                for _ in range(fold)
            ]
        )
    else:
        return _ACCC(Z, y, X, Z_ties_method, y_ties_method, X_ties_method, mode=mode)
