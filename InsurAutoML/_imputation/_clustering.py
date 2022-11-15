"""
File Name: _clustering.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_imputation/_clustering.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 14th November 2022 8:07:44 pm
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

from typing import Union, Tuple, List
import numpy as np
import pandas as pd
import warnings
from functools import partial
import multiprocessing
from multiprocessing import Pool

from InsurAutoML._utils import formatting
from InsurAutoML._scaling import MinMaxScale


class AAI_kNN(formatting, MinMaxScale):

    """
    kNN Imputation/Neighborhood-based Collaborative Filtering with
    Auto-adaptive Imputation/AutAI [1]
    AutAI's main idea is to distinguish what part of the dataset may
    be important for imputation.

    ----
    [1] Ren, Y., Li, G., Zhang, J. and Zhou, W., 2012, October. The
    efficient imputation method for neighborhood-based collaborative
    filtering. In Proceedings of the 21st ACM international conference
    on Information and knowledge management (pp. 684-693).

    Parameters
    ----------
    k: k nearest neighbors selected, default = 3
    Odd k preferred

    scaling: whether to perform scaling on the features, default = True

    similarity: how to calculate similarity among rows, default = 'PCC'
    support ['PCC', 'COS']

    AutAI: whether to use AutAI, default = True

    AutAI_tmp: whether AutAI is temporary imputation or permanent
    imputation, default = True
    if True (temporary), AutAI imputation will not be preserved, but
    can take much longer

    threads: number of threads to use, default = -1
    if -1, use all threads

    deep_copy: whether to deep_copy dataframe, default = False
    """

    def __init__(
        self,
        k: int = 3,
        scaling: bool = True,
        similarity: str = "PCC",
        AutAI: bool = True,
        AutAI_tmp: bool = True,
        threads: int = -1,
        deep_copy: bool = False,
    ) -> None:
        self.k = k
        self.scaling = scaling
        self.similarity = similarity
        self.AutAI = AutAI
        self.AutAI_tmp = AutAI_tmp
        self.threads = threads
        self.deep_copy = deep_copy

        self._fitted = False  # whether fitted on train set
        self.train = pd.DataFrame()  # store the imputed train set

    # calculate Pearson Correlation Coefficient/PCC
    # PCC = \sum_{i}(x_{i}-\mu_{x})(y_{i}-\mu_{y}) /
    # \sqrt{\sum_{i}(x_{i}-\mu_{x})^{2}\sum_{i}(y_{i}-\mu_{y})^{2}}
    def Pearson_Correlation_Coefficient(self, x: np.ndarray, y: np.ndarray) -> float:

        # convert to numpy array
        x = np.array(x)
        y = np.array(y)

        # get mean of x and y
        u_x = np.nanmean(x)
        u_y = np.nanmean(y)

        # get numerator and denominator
        numerator = np.nansum((x - u_x) * (y - u_y))
        denominator = np.sqrt(np.nansum((x - u_x) ** 2) * np.nansum((y - u_y) ** 2))

        # special case of denominator being 0
        if denominator == 0:
            return 1
        else:
            return numerator / denominator

    # calculate Cosine-based similarity/COS
    # COS = x * y / (|x|*|y|)
    def Cosine_based_similarity(self, x: np.ndarray, y: np.ndarray) -> float:

        # convert to numpy array
        x = np.array(x)
        y = np.array(y)

        # get numerator and denominator
        numerator = np.nansum(x * y)
        denominator = np.sqrt(np.nansum(x**2) * np.nansum(y**2))

        # special case of denominator being 0
        if denominator == 0:
            return 1
        else:
            return numerator / denominator

    # get column values from k nearest neighbors
    def _get_k_neighbors(
        self, test: pd.DataFrame, train: pd.DataFrame, column: List[str]
    ) -> Tuple[List[pd.DataFrame], List[Union[int, str]], List[float]]:

        similarity_list = []

        for index in list(train.index):
            # get similarity between test and all rows in train
            if self.similarity == "PCC":
                similarity_list.append(
                    self.Pearson_Correlation_Coefficient(
                        test.values, train.loc[index].values
                    )
                )
            elif self.similarity == "COS":
                similarity_list.append(
                    self.Cosine_based_similarity(test.values, train.loc[index].values)
                )

        # get index of k largest similarity in list
        k_order = np.argsort(similarity_list)[-self.k :]
        # convert similarity list order to data index
        k_index = [list(train.index)[i] for i in k_order]

        # get k largest similarity
        k_similarity = [similarity_list[_index] for _index in k_order]

        # get the k largest values in the list
        k_values = [train.loc[_index, column] for _index in k_index]

        return k_values, k_index, k_similarity

    # AutAI imputation
    def _AAI_impute(
        self, X: pd.DataFrame, index: List[Union[str, int]], column: List[str]
    ) -> pd.DataFrame:

        _X = X.copy(deep=self.deep_copy)

        # (index, column) gives a location of missing value
        # the goal is to find and impute relatively important data

        # get indexes where column values are not missing
        U_a = list(_X.loc[~_X[column].isnull()].index.astype(int))

        # find union of columns where both non_missing (each from above)
        # and missing rows have data
        T_s = []  # ultimate union of features
        loc_non_missing_column = set(_X.columns[~_X.loc[index, :].isnull()])

        for _index in U_a:
            # get all intersection from U_a rows
            non_missing_columns = set(_X.columns[~_X.loc[_index, :].isnull()])
            intersection_columns = loc_non_missing_column.intersection(
                non_missing_columns
            )

            if not T_s:  # if empty
                T_s = intersection_columns
            else:
                T_s = T_s.union(intersection_columns)

        T_s = list(T_s)  # convert to list

        # range (U_a, T_s) considered important data

        # use kNN with weight of similarity for imputation
        for _column in T_s:
            for _index in list(set(X[X[_column].isnull()].index) & set(U_a)):
                # get kNN column values, index, and similarity
                k_values, k_index, k_similarity = self._get_k_neighbors(
                    _X.loc[_index, :], _X.loc[~_X[_column].isnull()], _column
                )

                # normalize k_similarity
                k_similarity = [item / sum(k_similarity) for item in k_similarity]

                # get kNN row mean
                k_means = [np.nanmean(_X.loc[_index, :]) for _index in k_index]

                # calculate impute value
                _impute = np.nanmean(_X.loc[index, :])
                for i in range(self.k):
                    _impute += k_similarity[i] * (k_values[i] - k_means[i])

                _X.loc[_index, _column] = _impute

        return _X

    # pool tasks on the index chunks
    # every pool task works on part of the chunks
    def Pool_task(
        self, X: pd.DataFrame, index_list: List[List[Union[str, int]]]
    ) -> pd.DataFrame:

        _X = X.copy(deep=self.deep_copy)

        for _column in self.columns:
            # get missing rows
            # select in index_list and get missing rows
            missing = _X.loc[index_list].loc[_X[_column].isnull()]

            # make sure k is at least not larger than rows of non_missing
            self.k = min(self.k, len(_X) - len(missing))

            if missing.empty:  # if no missing found in the column, skip
                pass
            else:
                for _index in list(missing.index):
                    # if need AutAI, perform AutAI imputation first
                    # if fitted, no need for AutAI, directly run kNN imputation
                    if self.AutAI and not self._fitted:
                        if self.AutAI_tmp:
                            _X_tmp = self._AAI_impute(_X, _index, _column)
                            # get non-missing (determined by _column) rows
                            non_missing = _X_tmp.loc[~_X_tmp[_column].isnull()]
                        else:
                            _X = self._AAI_impute(_X, _index, _column)
                            # get non-missing (determined by _column) rows
                            non_missing = _X.loc[~_X[_column].isnull()]
                    elif not self._fitted:
                        # get non-missing (determined by _column) rows
                        non_missing = _X.loc[~_X[_column].isnull()]

                    # use kNN imputation for (_index, _column)
                    # if fitted, use imputed dataset for imputation
                    if not self._fitted:
                        k_values, _, _ = self._get_k_neighbors(
                            _X.loc[_index, :], non_missing, _column
                        )
                        _X.loc[_index, _column] = np.mean(k_values)
                    else:
                        k_values, _, _ = self._get_k_neighbors(
                            _X.loc[_index, :], self.train, _column
                        )
                        _X.loc[_index, _column] = np.mean(k_values)

        # return only the working part
        return _X.loc[index_list, :]

    def fill(self, X: pd.DataFrame) -> pd.DataFrame:

        # make sure input is a dataframe
        if not isinstance(X, pd.DataFrame):
            try:
                X = pd.DataFrame(X)
            except:
                raise TypeError("Expect a dataframe, get {}.".format(X))

        _X = X.copy(deep=self.deep_copy)

        # initialize columns
        self.columns = list(_X.columns)

        # initialize number of working threads
        self.threads = (
            multiprocessing.cpu_count() if self.threads == -1 else int(self.threads)
        )

        if _X[self.columns].isnull().values.any():
            _X = self._fill(_X)
        else:
            warnings.warn("No missing values found, no change.")

        return _X

    def _fill(self, X: pd.DataFrame) -> pd.DataFrame:

        _X = X.copy(deep=self.deep_copy)

        # convert categorical to numerical
        formatter = formatting(columns=self.columns, inplace=True)
        formatter.fit(_X)

        # if scaling, use MinMaxScale to scale the features
        if self.scaling:
            scaling = MinMaxScale()
            _X = scaling.fit_transform(_X)

        # kNN imputation

        # parallelized pool workflow
        pool = Pool(processes=self.threads)
        # divide indexes to evenly sized chunks
        divide_index = np.array_split(list(_X.index), self.threads)

        # parallelized work
        pool_data = pool.map(partial(self.Pool_task, _X), divide_index)
        pool.close()
        pool.join()

        # concat the chunks of the datset
        _X = pd.concat(pool_data).sort_index()

        # convert self._fitted and store self.train
        self._fitted = True
        # only when empty need to store
        # stored train is imputed, formatted, scaled dataset
        self.train = _X.copy() if self.train.empty else self.train

        # if scaling, scale back
        if self.scaling:
            _X = scaling.inverse_transform(_X)

        # convert numerical back to categorical
        formatter.refit(_X)

        return _X


"""
For clustering methods, if get warning of empty mean (all cluster 
have nan for the feature) or reduction in number of clusters (get 
empty cluster), number of cluster is too large compared to the 
data size and should be decreased. However, no error will be 
raised, and all observations will be correctly assigned.
"""


class KMI(formatting, MinMaxScale):

    """
    KMI/K-Means Imputation[1], the idea is to incorporate K-Means
    Clustering with kNN imputation

    ----
    [1] Hruschka, E.R., Hruschka, E.R. and Ebecken, N.F., 2004, December.
    Towards efficient imputation by nearest-neighbors: A clustering-based
    approach. In Australasian Joint Conference on Artificial Intelligence
    (pp. 513-525). Springer, Berlin, Heidelberg.

    Parameters
    ----------
    """

    def __init__(
        self,
        scaling: bool = True,
    ) -> None:
        self.scaling = scaling

        raise NotImplementedError("Not implemented!")


class CMI(formatting, MinMaxScale):

    """
    Clustering-based Missing Value Imputation/CMI[1], introduces the idea
    of clustering observations into groups and use the kernel statistsics
    to impute the missing values for the groups.

    ----
    [1] Zhang, S., Zhang, J., Zhu, X., Qin, Y. and Zhang, C., 2008. Missing
    value imputation based on data clustering. In Transactions on computational
    science I (pp. 128-138). Springer, Berlin, Heidelberg.

    Parameters
    ----------
    k: number of cluster groups, default = 10

    distance: metrics of calculating the distance between two rows, default = 'l2'
    used for selecting clustering groups,
    support ['l1', 'l2']

    delta: threshold to stop k Means Clustering, default = 0
    delta defined as number of group assignment changes for clustering
    0 stands for best k Means Clustering

    scaling: whether to use scaling before imputation, default = True

    seed: random seed, default = 1
    used for k group initialization

    threads: number of threads to use, default = -1
    if -1, use all threads

    deep_copy: whether to use deep copy, default = False
    """

    def __init__(
        self,
        k: int = 10,
        distance: str = "l2",
        delta: float = 0.0,
        scaling: bool = True,
        seed: int = 1,
        threads: int = -1,
        deep_copy: bool = False,
    ) -> None:
        self.k = k
        self.distance = distance
        self.delta = delta
        self.scaling = scaling
        self.seed = seed
        self.threads = threads
        self.deep_copy = deep_copy

        np.random.seed(seed=self.seed)

        self._fitted = False  # whether fitted on train set
        self.train = pd.DataFrame()  # store the imputed train set

    # calculate distance between row and k group mean
    # 'l1' or 'l2' Euclidean distance
    def _distance(self, row: Union[pd.DataFrame, np.ndarray], k: int) -> float:

        if self.distance == "l2":
            return np.sqrt(np.nansum((row - self.k_means[k]) ** 2))
        elif self.distance == "l1":
            return np.nansum(np.abs(row - self.k_means[k]))

    # get the Gaussian kernel values
    def _kernel(
        self,
        row1: Union[pd.DataFrame, np.ndarray],
        row2: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:

        return np.prod(
            np.exp(-(((row1 - row2) / self.bandwidth) ** 2) / 2) / np.sqrt(2 * np.pi)
        )

    # get k_means for group k
    def _get_k_means(self, data: pd.DataFrame, k: int) -> Union[float, np.ndarray]:

        # get observations in _k group
        group_data = data.loc[np.where(self.group_assign == k)[0], :]

        if not group_data.empty:
            return np.nanmean(group_data, axis=0)
        else:
            # return None array
            return np.array([None])

    # get group assign for the chunk of data (by index_list)
    def _get_group_assign(
        self, data: pd.DataFrame, index_list: List[Union[str, int]]
    ) -> List[int]:

        result = []
        for _index in index_list:
            # get distance between row and every k groups
            distance = [self._distance(data.loc[_index, :], _k) for _k in range(self.k)]
            # assign the row to closest range group
            result.append(np.argsort(distance)[0])

        return result

    # assign clustering groups according to Euclidean distance
    # only support numerical features
    def _k_means_clustering(self, X: pd.DataFrame) -> None:

        _X = X.copy(deep=self.deep_copy)

        n, p = _X.shape  # number of observations

        # make sure self.k is smaller than n
        self.k = min(self.k, n)

        # if not fitted (on train dataset), run complete k Means clustering
        # else, use train k_means for clustering
        if not self._fitted:

            # get bandwidth for the kernel function
            self.bandwidth = np.sum(
                np.abs(_X[self.columns].max() - _X[self.columns].min())
            )

            # initialize k groups
            self.group_assign = np.random.randint(0, high=self.k, size=n)
            # initialize k means
            self.k_means = np.empty([self.k, p])

            # parallelized k means calculation
            pool = Pool(processes=min(int(self.k), self.threads))
            self.k_means = pool.map(partial(self._get_k_means, _X), list(range(self.k)))
            pool.close()
            pool.join()

            # if group empty, raise warning and set new k
            self.k_means = [item for item in self.k_means if item.all()]
            if len(self.k_means) < self.k:
                warnings.warn("Empty cluster found and removed.")
                self.k = len(self.k_means)

            while True:
                # store the group assignment
                # need deep copy, or the stored assignment will change accordingly
                previous_group_assign = self.group_assign.copy()

                # assign each observation to new group based on k_means
                pool = Pool(processes=self.threads)
                divide_index = np.array_split(list(_X.index), self.threads)
                self.group_assign = pool.map(
                    partial(self._get_group_assign, _X), divide_index
                )
                # flatten 2d list to 1d
                self.group_assign = np.array(np.concatenate(self.group_assign).flat)
                pool.close()
                pool.join()

                # calculate the new k_means
                # parallelized k means calculation
                pool = Pool(processes=min(int(self.k), self.threads))
                self.k_means = pool.map(
                    partial(self._get_k_means, _X), list(range(self.k))
                )
                pool.close()
                pool.join()

                # if group empty, raise warning and set new k
                self.k_means = [item for item in self.k_means if item.all()]
                if len(self.k_means) < self.k:
                    warnings.warn("Empty cluster found and removed.")
                    self.k = len(self.k_means)

                # if k Means constructed, break the loop
                if (
                    np.sum(np.abs(previous_group_assign - self.group_assign))
                    <= self.delta
                ):
                    break
        else:
            # copy the train group assignment
            self.group_assign_train = self.group_assign.copy()

            self.group_assign = np.zeros(
                n
            )  # re-initialize the group_assign (n may change)
            # assign each observation to new group based on k_means
            pool = Pool(processes=self.threads)
            divide_index = np.array_split(list(_X.index), self.threads)
            self.group_assign = pool.map(
                partial(self._get_group_assign, _X), divide_index
            )
            # flatten 2d list to 1d
            self.group_assign = np.array(np.concatenate(self.group_assign).flat)
            pool.close()
            pool.join()

    # pool tasks on the column chunks
    # every pool task works on part of the chunks
    def Pool_task(
        self,
        X: pd.DataFrame,
        _column: List[str],
        non_missing_index: List[Union[str, int]],
        n: int,
        index_list: List[Union[str, int]],
    ) -> pd.DataFrame:

        _X = X.copy(deep=self.deep_copy)

        # find missing indexes belongs to _k group
        for _index in index_list:
            if not self._fitted:
                # get the kernel values
                kernel = np.array(
                    [
                        self._kernel(
                            _X.loc[_index, _X.columns != _column],
                            _X.loc[__index, _X.columns != _column],
                        )
                        for __index in non_missing_index
                    ]
                )
                # impute the missing_values
                _X.loc[_index, _column] = np.sum(
                    _X.loc[non_missing_index, _column] * kernel
                ) / (np.sum(kernel) + n ** (-2))
            else:
                # get the kernel values
                kernel = np.array(
                    [
                        self._kernel(
                            _X.loc[_index, _X.columns != _column],
                            self.train.loc[__index, self.train.columns != _column],
                        )
                        for __index in non_missing_index
                    ]
                )
                # impute the missing_values
                _X.loc[_index, _column] = np.sum(
                    self.train.loc[non_missing_index, _column] * kernel
                ) / (np.sum(kernel) + n ** (-2))

        # return group data
        return _X.loc[index_list, _column]

    def fill(self, X: pd.DataFrame) -> pd.DataFrame:

        # make sure input is a dataframe
        if not isinstance(X, pd.DataFrame):
            try:
                X = pd.DataFrame(X)
            except:
                raise TypeError("Expect a dataframe, get {}.".format(type(X)))

        _X = X.copy(deep=self.deep_copy)

        # initialize columns
        self.columns = list(_X.columns)

        # initialize number of working threads
        self.threads = (
            multiprocessing.cpu_count() if self.threads == -1 else int(self.threads)
        )

        if _X[self.columns].isnull().values.any():
            _X = self._fill(_X)
        else:
            warnings.warn("No missing values found, no change.")

        return _X

    def _fill(self, X: pd.DataFrame) -> pd.DataFrame:

        _X = X.copy(deep=self.deep_copy)

        # convert categorical to numerical
        formatter = formatting(columns=self.columns, inplace=True)
        formatter.fit(_X)

        # if scaling, use MinMaxScale to scale the features
        if self.scaling:
            scaling = MinMaxScale()
            _X = scaling.fit_transform(_X)

        # imputation on formatted, scaled datasets
        # assign observations to self.k groups
        # get self.group_assign and self.k_means
        # if already fitted (working on test data now), get
        # k_means from train dataset and the group assignment
        # for train dataset
        self._k_means_clustering(_X)

        for _column in self.columns:
            for _k in range(self.k):
                group_index = np.where(self.group_assign == _k)[0]
                n = len(group_index)  # number of observations in the group
                # get the missing/non-missing indexes
                missing_index = list(
                    set(_X[_X[_column].isnull()].index) & set(group_index)
                )
                if not self._fitted:  # if not fitted, use _X
                    non_missing_index = list(
                        set(_X[~_X[_column].isnull()].index) & set(group_index)
                    )
                else:  # if fitted, use self.train
                    # after imputation, there should be no missing in train
                    # so take all of them
                    non_missing_index = np.where(self.group_assign_train == _k)[0]

                # parallelize imputation
                divide_missing_index = np.array_split(missing_index, self.threads)
                pool = Pool(self.threads)
                imputation = pool.map(
                    partial(self.Pool_task, _X, _column, non_missing_index, n),
                    divide_missing_index,
                )
                pool.close()
                pool.join()

                imputation = pd.concat(imputation).sort_index()
                _X.loc[missing_index, _column] = imputation

        # convert self._fitted and store self.train
        self._fitted = True
        # only when empty need to store
        # stored train is imputed, formatted, scaled dataset
        self.train = _X.copy() if self.train.empty else self.train

        # if scaling, scale back
        if self.scaling:
            _X = scaling.inverse_transform(_X)

        # convert numerical back to categorical
        formatter.refit(_X)

        return _X


class k_Prototype_NN(formatting, MinMaxScale):

    """
    Three clustering models are provided: k-Means Paradigm, k-Modes Paradigm,
    k-Prototypes Paradigm [1]

    Parameters
    ----------
    k: number of cluster groups, default = 10

    distance: metrics of calculating the distance between two rows, default = 'l2'
    used for selecting clustering groups,
    support ['l1', 'l2']

    dissimilarity: how to calculate the dissimilarity for categorical columns, default = 'weighted'
    support ['simple', 'weighted']

    scaling: whether to use scaling before imputation, default = True

    numerics: numerical columns

    seed: random seed, default = 1
    used for k group initialization

    threads: number of threads to use, default = -1
    if -1, use all threads

    deep_copy: whether to use deep copy, default = False

    ----
    [1] Madhuri, R., Murty, M.R., Murthy, J.V.R., Reddy, P.P. and Satapathy,
    S.C., 2014. Cluster analysis on different data sets using K-modes and
    K-prototype algorithms. In ICT and Critical Infrastructure: Proceedings
    of the 48th Annual Convention of Computer Society of India-Vol II
    (pp. 137-144). Springer, Cham.
    """

    def __init__(
        self,
        k: int = 10,
        distance: str = "l2",
        dissimilarity: str = "weighted",
        scaling: bool = True,
        numerics: List[str] = [
            "int16",
            "int32",
            "int64",
            "float16",
            "float32",
            "float64",
        ],
        threads: int = -1,
        deep_copy: bool = False,
        seed: int = 1,
    ) -> None:
        self.k = k
        self.distance = distance
        self.dissimilarity = dissimilarity
        self.scaling = scaling
        self.numerics = numerics
        self.threads = threads
        self.deep_copy = deep_copy
        self.seed = seed

        np.random.seed(self.seed)

        self._fitted = False  # check whether fitted on train data
        self.models = {}  # save fit models

    # calculate distance between row and k group mean
    # 'l1' or 'l2' Euclidean distance
    def _distance(
        self,
        row: Union[pd.DataFrame, np.ndarray],
        k_centroids: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:

        # if not defining np.float64, may get float and
        # raise Error not able of iterating
        if self.distance == "l2":
            return np.sqrt(
                np.nansum((row - k_centroids) ** 2, axis=1, dtype=np.float64)
            )
        elif self.distance == "l1":
            return np.nansum(np.abs(row - k_centroids), axis=1, dtype=np.float64)

    # calculate dissimilarity difference between row
    # and k group
    def _dissimilarity(
        self,
        row: Union[pd.DataFrame, np.ndarray],
        k_centroids: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:

        k, p = k_centroids.shape

        # simple dissimilarity, number of different categories
        if self.dissimilarity == "simple":
            return np.sum((row.values != k_centroids.values).astype(int), axis=1)
        # weighted dissimilarity, weighted based on number of unique categories
        elif self.dissimilarity == "weighted":
            # set difference between row and k_centroids
            different_matrix = (row.values != k_centroids.values).astype(int)

            # initialize number counts
            row_count = np.empty(p)
            centroids_count = np.empty([k, p])

            for idx, _column in enumerate(list(k_centroids.columns)):
                # get the category
                cate = row[_column]
                # find the corresponding count
                # if get missing value, count as 0
                row_count[idx] = (
                    self.categorical_table[_column][cate] if not cate else 0
                )
                for _k in range(k):
                    # get the category
                    cate = k_centroids.loc[_k, _column]
                    # find the corresponding count
                    centroids_count[_k, idx] = self.categorical_table[_column][cate]

            # calculate the weights based on number of categories
            weight = np.empty([k, p])
            # in case get denominator of 0
            for _p in range(p):
                for _k in range(k):
                    weight[_k, _p] = (
                        (row_count[_p] + centroids_count[_k, _p])
                        / (row_count[_p] * centroids_count[_k, _p])
                        if row_count[_p] != 0
                        else 0
                    )

            return np.nansum(np.multiply(weight, different_matrix), axis=1)

    # calculate the measurement for given index
    def _get_group_assign(
        self,
        data: pd.DataFrame,
        numerical_columns: List[str],
        categorical_columns: List[str],
        index_list: List[Union[int, str]],
    ) -> List[int]:

        group_assign = []

        for _index in index_list:
            measurement = self._distance(
                data.loc[_index, numerical_columns], self.k_centroids[numerical_columns]
            ) + self._dissimilarity(
                data.loc[_index, categorical_columns],
                self.k_centroids[categorical_columns],
            )

            # assign the observations to closest centroids
            group_assign.append(np.argsort(measurement)[0])

        return group_assign

    # calculate the k_centroids for group k
    def _get_k_centroids(
        self,
        data: pd.DataFrame,
        numerical_columns: List[str],
        categorical_columns: List[str],
        k: int,
    ) -> pd.DataFrame:

        k_centroids = pd.DataFrame(index=[k], columns=data.columns)

        group_data = data.loc[np.where(self.group_assign == k)[0], :]
        # get column means of the group
        # if get empty group_data, no return
        if not group_data.empty:
            # centroids for numerical columns are mean of the features
            k_centroids[numerical_columns] = np.nanmean(
                group_data[numerical_columns], axis=0
            )
            # centroids for categorical columns are modes of the features
            # in case multiple modes occur, get the first one
            k_centroids[categorical_columns] = (
                group_data[categorical_columns].mode(dropna=True).values[0]
            )

        return k_centroids

    # assign clustering groups according to dissimilarity
    # compared to modes
    # best support categorical features
    def _k_modes_clustering(self, X: pd.DataFrame) -> None:

        _X = X.copy(deep=self.deep_copy)

        n, p = _X.shape  # number of observations

        # make sure self.k is smaller than n
        self.k = min(self.k, n)

    # combine k_Means and k_Modes clustering for mixed
    # numerical/categorical datasets
    # numerical columns will use k_means with distance
    # categorical columns will use k_modes with dissimilarity
    def _k_prototypes_clustering(
        self,
        X: pd.DataFrame,
        numerical_columns: List[str],
        categorical_columns: List[str],
    ) -> None:

        _X = X.copy(deep=self.deep_copy)

        n, p = _X.shape  # number of observations

        # make sure self.k is smaller than n
        self.k = min(self.k, n)

        if not self._fitted:
            # create categorical count table
            # unique values descending according to number of observations
            self.categorical_table = {}
            for _column in categorical_columns:
                self.categorical_table[_column] = (
                    _X[_column]
                    .value_counts(sort=True, ascending=False, dropna=True)
                    .to_dict()
                )

            # initialize clustering group assignments
            self.group_assign = np.zeros(n)
            # initialize the corresponding centroids
            # use dataframe to match the column names
            self.k_centroids = pd.DataFrame(index=range(self.k), columns=_X.columns)
            # random initialization
            for _column in list(_X.columns):
                self.k_centroids[_column] = (
                    _X.loc[~_X[_column].isnull(), _column]
                    .sample(n=self.k, replace=True, random_state=self.seed)
                    .values
                )

            while True:
                # calculate sum of Euclidean distance (numerical features) and
                # dissimilarity difference (categorical features) for every
                # observations among all centroids

                # parallelized calculation for group assignment
                pool = Pool(processes=self.threads)
                divide_list = np.array_split(list(_X.index), self.threads)
                self.group_assign = pool.map(
                    partial(
                        self._get_group_assign,
                        _X,
                        numerical_columns,
                        categorical_columns,
                    ),
                    divide_list,
                )
                # flatten 2d list to 1d
                self.group_assign = np.array(np.concatenate(self.group_assign).flat)
                pool.close()
                pool.join()

                # save k_centroids for comparison
                previous_k_centroids = self.k_centroids.copy()

                # recalculate the k_centroids
                # calculate the new k_means
                pool = Pool(processes=min(int(self.k), self.threads))
                self.k_centroids = pool.map(
                    partial(
                        self._get_k_centroids,
                        _X,
                        numerical_columns,
                        categorical_columns,
                    ),
                    list(range(self.k)),
                )
                # concat the k centroids to one dataframe
                self.k_centroids = pd.concat(self.k_centroids).sort_index()
                pool.close()
                pool.join()

                # if get empty cluster, sort index and renew k
                self.k_centroids.dropna(inplace=True)
                if len(self.k_centroids) < self.k:
                    self.k_centroids.reset_index(drop=True, inplace=True)
                    self.k = len(self.k_centroids)

                # stopping criteria
                # check whether same k (in case delete centroids in the process)
                if len(previous_k_centroids) == len(self.k_centroids):
                    if np.all(previous_k_centroids.values == self.k_centroids.values):
                        break
        # if fitted, use the trained k_centroids assigning groups
        else:
            # parallelized calculation for group assignment
            pool = Pool(processes=self.threads)
            divide_list = np.array_split(list(_X.index), self.threads)
            self.group_assign = pool.map(
                partial(
                    self._get_group_assign, _X, numerical_columns, categorical_columns
                ),
                divide_list,
            )
            # flatten 2d list to 1d
            self.group_assign = np.array(np.concatenate(self.group_assign).flat)
            pool.close()
            pool.join()

    # impute on cluster k
    def _kNN_impute(self, data: pd.DataFrame, k: int) -> pd.DataFrame:

        from sklearn.impute import KNNImputer

        # use 1-NN imputer with clustered groups
        # on train dataset, fit the models
        if k not in self.models.keys():
            self.models[k] = KNNImputer(n_neighbors=1)
            self.models[k].fit(data.loc[np.where(self.group_assign == k)[0], :])
        # impute the missing values
        data.loc[np.where(self.group_assign == k)[0], :] = self.models[k].transform(
            data.loc[np.where(self.group_assign == k)[0], :]
        )

        return data.loc[np.where(self.group_assign == k)[0], :]

    def fill(self, X: pd.DataFrame) -> pd.DataFrame:

        # make sure input is a dataframe
        if not isinstance(X, pd.DataFrame):
            try:
                X = pd.DataFrame(X)
            except:
                raise TypeError("Expect a dataframe, get {}.".format(type(X)))

        _X = X.copy(deep=self.deep_copy)

        # initialize columns
        self.columns = list(_X.columns)

        # initialize number of working threads
        self.threads = (
            multiprocessing.cpu_count() if self.threads == -1 else int(self.threads)
        )

        if _X[self.columns].isnull().values.any():
            _X = self._fill(_X)
        else:
            warnings.warn("No missing values found, no change.")

        return _X

    def _fill(self, X: pd.DataFrame) -> pd.DataFrame:

        _X = X.copy(deep=self.deep_copy)

        # all numerical columns
        numeric_columns = list(_X.select_dtypes(include=self.numerics).columns)
        # select numerical columns in self.columns
        numeric_columns = list(set(_X.columns) & set(numeric_columns))
        # select categorical columns in self.columns
        categorical_columns = list(set(_X.columns) - set(numeric_columns))

        # format columns
        # convert categorical to numerical,
        # but no numerical manipulation
        formatter = formatting(columns=list(_X.columns), inplace=True)
        formatter.fit(_X)

        # if scaling, scaling the numerical columns
        if self.scaling:
            scaling = MinMaxScale()
            _X = scaling.fit_transform(_X)

        # imputation procedure
        # assign observations to clustering groups using
        # k_Prototypes clustering
        self._k_prototypes_clustering(_X, numeric_columns, categorical_columns)

        # use the clustered groups to impute
        # parallelized the imputation process according to clusters
        pool = Pool(processes=min(int(self.k), self.threads))
        pool_data = pool.map(partial(self._kNN_impute, _X), list(range(self.k)))
        # concat pool_data and order according to index
        _X = pd.concat(pool_data).sort_index()
        pool.close()
        pool.join()

        # set fitted to true
        self._fitted = True

        # if scaling, scale back
        if self.scaling:
            _X = scaling.inverse_transform(_X)

        # make sure column types retains
        formatter.refit(_X)

        return _X
