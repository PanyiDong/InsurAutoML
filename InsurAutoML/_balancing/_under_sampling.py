"""
File Name: _under_sampling.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_balancing/_under_sampling.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 14th November 2022 6:58:22 pm
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

from typing import Union, Tuple
import numpy as np
import pandas as pd
import warnings
import sklearn
import sklearn.utils

from InsurAutoML._utils._data import is_imbalance, LinkTable

"""
Reference for: Simple Random Over Sampling, Simple Random Under Sampling, Tomek Link, \
    Edited Nearest Neighbor,  Condensed Nearest Neighbor, One Sided Selection, CNN_TomekLink, \
    Smote, Smote_TomekLink, Smote_ENN
    
Batista, G.E., Prati, R.C. and Monard, M.C., 2004. A study of the behavior of several methods for 
balancing machine learning training data. ACM SIGKDD explorations newsletter, 6(1), pp.20-29.
"""


class SimpleRandomUnderSampling:

    """
    Simple Random Under-Sampling
    Randomly eliminate samples from majority class

    Parameters
    ----------
    imbalance_threshold: determine to what extent will the data be considered as imbalanced data, default = 0.9

    all: whether to stop until all features are balanced, default = False

    max_iter: Maximum number of iterations for over-/under-sampling, default = 1000

    seed: random seed, default = 1
    every random draw from the majority class will increase the random seed by 1
    """

    def __init__(
        self,
        imbalance_threshold: float = 0.9,
        all: bool = False,
        max_iter: int = 1000,
        seed: int = 1,
    ) -> None:
        self.imbalance_threshold = imbalance_threshold
        self.all = all
        self.max_iter = max_iter
        self.seed = seed

        self._fitted = False  # whether the model has been fitted

    def fit_transform(
        self, X: pd.DataFrame, y: pd.DataFrame = None
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:

        try:  # if missing y, will be None value; or will be dataframe, use df.empty for judge
            _empty = y.empty
        except AttributeError:
            _empty = y == None

        if (
            not _empty
        ):  # if no y input, only convert X; or, combine X and y to consider balancing
            features = list(X.columns)
            response = list(y.columns)
            data = pd.concat([X, y], axis=1)
        else:
            features = list(X.columns)
            response = None
            data = X

        _data = data.copy(deep=True)
        if not is_imbalance(_data, self.imbalance_threshold):
            warnings.warn("The dataset is balanced, no change.")
        else:
            if self.all == True:
                while is_imbalance(_data, self.imbalance_threshold):
                    _data = self._fit_transform(_data)
            else:
                _data = self._fit_transform(_data)

        self._fitted = True

        if not _empty:
            return _data[features], _data[response]
        else:
            return _data

    def _fit_transform(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:  # using random over-sampling to balance the first imbalanced feature

        features = list(X.columns)
        _imbalanced_feature, _majority = is_imbalance(
            X, self.imbalance_threshold, value=True
        )
        _seed = self.seed
        _iter = 0

        while (
            is_imbalance(X[[_imbalanced_feature]], self.imbalance_threshold)
            and _iter <= self.max_iter
        ):
            _majority_class = X.loc[X[_imbalanced_feature] == _majority]
            sample = _majority_class.sample(n=1, random_state=_seed)
            X = X.drop(sample.index)
            _seed += 1
            _iter += 1
        X = sklearn.utils.shuffle(X.reset_index(drop=True)).reset_index(drop=True)

        return X


class TomekLink:

    """
    Use Tomek Links to remove noisy or border significant majority class sample
    Tomek links define as nearest neighbors with different classification

    Parameters
    ----------
    imbalance_threshold: determine to what extent will the data be considered as imbalanced data, default = 0.9

    norm: how the distance between different samples calculated, default = 'l2'
    all supported norm ['l1', 'l2']

    all: whether to stop until all features are balanced, default = False

    max_iter: Maximum number of iterations for over-/under-sampling, default = 1000

    seed: random seed, default = 1
    every random draw from the majority class will increase the random seed by 1
    """

    def __init__(
        self,
        imbalance_threshold: float = 0.9,
        norm: str = "l2",
        all: bool = False,
        max_iter: int = 1000,
        seed: int = 1,
    ) -> None:
        self.imbalance_threshold = imbalance_threshold
        self.norm = norm
        self.all = all
        self.max_iter = max_iter
        self.seed = seed

        self._fitted = False  # whether the model has been fitted

    def fit_transform(
        self, X: pd.DataFrame, y: pd.DataFrame = None
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:

        try:  # if missing y, will be None value; or will be dataframe, use df.empty for judge
            _empty = y.empty
        except AttributeError:
            _empty = y == None

        if (
            not _empty
        ):  # if no y input, only convert X; or, combine X and y to consider balancing
            features = list(X.columns)
            response = list(y.columns)
            data = pd.concat([X, y], axis=1)
        else:
            features = list(X.columns)
            response = None
            data = X

        _data = data.copy(deep=True)
        if not is_imbalance(_data, self.imbalance_threshold):
            warnings.warn("The dataset is balanced, no change.")
        else:
            if self.all == True:
                while is_imbalance(_data, self.imbalance_threshold):
                    _data = self._fit_transform(_data)
            else:
                _data = self._fit_transform(_data)

        self._fitted = True

        if not _empty:
            return _data[features], _data[response]
        else:
            return _data

    def _fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        _imbalanced_feature, _majority = is_imbalance(
            X, self.imbalance_threshold, value=True
        )
        _seed = self.seed
        _iter = 0

        while (
            is_imbalance(X[[_imbalanced_feature]], self.imbalance_threshold)
            and _iter <= self.max_iter
        ):
            _minority_class = X.loc[X[_imbalanced_feature] != _majority]
            _minority_sample = _minority_class.sample(
                n=max(int(len(_minority_class) / 100), 1), random_state=_seed
            )
            _link_table = LinkTable(_minority_sample, X, self.norm)
            drop_index = []
            for _link_item in _link_table:
                _nearest = _link_item.index(
                    sorted(_link_item)[1]
                )  # since the closest will always be the sample itself
                # if nearest is the majority class, add to drop_index
                if X.loc[_nearest, _imbalanced_feature] == _majority:
                    drop_index.append(_nearest)
                drop_index = list(set(drop_index))  # get unique drop indexes
            X = X.drop(index=drop_index, axis=0).reset_index(drop=True)
            _seed += 1
            _iter += 1

        return X


class EditedNearestNeighbor:

    """
    Edited Nearest Neighbor (ENN)
    Under-sampling method, drop samples where majority of k nearest neighbors belong to different class

    Parameters
    ----------
    imbalance_threshold: determine to what extent will the data be considered as imbalanced data, default = 0.9

    norm: how the distance between different samples calculated, default = 'l2'
    all supported norm ['l1', 'l2']

    all: whether to stop until all features are balanced, default = False

    max_iter: Maximum number of iterations for over-/under-sampling, default = 1000

    seed: random seed, default = 1
    every random draw from the majority class will increase the random seed by 1

    k: nearest neighbors to find, default = 3
    """

    def __init__(
        self,
        imbalance_threshold: float = 0.9,
        norm: str = "l2",
        all: bool = False,
        max_iter: int = 1000,
        seed: int = 1,
        k: int = 3,
    ) -> None:
        self.imbalance_threshold = imbalance_threshold
        self.norm = norm
        self.all = all
        self.max_iter = max_iter
        self.seed = seed
        self.k = k

        self._fitted = False  # whether the model has been fitted

    def fit_transform(
        self, X: pd.DataFrame, y: pd.DataFrame = None
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:

        try:  # if missing y, will be None value; or will be dataframe, use df.empty for judge
            _empty = y.empty
        except AttributeError:
            _empty = y == None

        if (
            not _empty
        ):  # if no y input, only convert X; or, combine X and y to consider balancing
            features = list(X.columns)
            response = list(y.columns)
            data = pd.concat([X, y], axis=1)
        else:
            features = list(X.columns)
            response = None
            data = X

        _data = data.copy(deep=True)

        if (self.k % 2) == 0:
            warnings.warn(
                "Criteria of majority better select odd k nearest neighbors, get {}.".format(
                    self.k
                )
            )

        if not is_imbalance(_data, self.imbalance_threshold):
            warnings.warn("The dataset is balanced, no change.")
        else:
            if self.all == True:
                while is_imbalance(_data, self.imbalance_threshold):
                    _data = self._fit_transform(_data)
            else:
                _data = self._fit_transform(_data)

        self._fitted = True

        if not _empty:
            return _data[features], _data[response]
        else:
            return _data

    def _fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:

        _imbalanced_feature, _majority = is_imbalance(
            X, self.imbalance_threshold, value=True
        )
        _seed = self.seed
        _iter = 0

        while (
            is_imbalance(X[[_imbalanced_feature]], self.imbalance_threshold)
            and _iter <= self.max_iter
        ):
            _majority_class = X.loc[X[_imbalanced_feature] == _majority]
            _majority_index = _majority_class.index
            _sample = X.sample(n=1, random_state=_seed)
            _sample_type = (
                "majority" if (_sample.index[0] in _majority_index) else "minority"
            )
            _link_table = LinkTable(_sample, X, self.norm)
            for _link_item in _link_table:
                _k_nearest = [
                    _link_item.index(item)
                    for item in sorted(_link_item)[1 : (self.k + 1)]
                ]
                count = 0
                for _index in _k_nearest:
                    _class = "majority" if (_index in _majority_index) else "minority"
                    if _class == _sample_type:
                        count += 1
                if count < (self.k + 1) / 2:
                    # if sample belongs to majority, remove the sample; else, remove the nearest neighbor
                    if _sample_type == "majority":
                        X = X.drop(_sample.index).reset_index(drop=True)
                    else:
                        X = X.drop(_link_item.index(sorted(_link_item)[1])).reset_index(
                            drop=True
                        )
            _seed += 1
            _iter += 1

            if len(_majority_class) == len(X):
                warnings.warn("No minority class left!")
                break

            if len(X) < 1:
                warnings.warn("No sample left!")

        return X


class CondensedNearestNeighbor:

    """
    Condensed Nearest Neighbor Rule (CNN)
    get subset of E that can predict the same as E using 1-NN
    algorithm: build the subset with all minority class and one of random majority class,
    build a 1-NN model and predict on all samples, put all misclassified data to the subset

    Parameters
    ----------
    imbalance_threshold: determine to what extent will the data be considered as imbalanced data, default = 0.9

    all: whether to stop until all features are balanced, default = False

    max_iter: Maximum number of iterations for over-/under-sampling, default = 1000

    seed: random seed, default = 1
    every random draw from the majority class will increase the random seed by 1
    """

    def __init__(
        self,
        imbalance_threshold: float = 0.9,
        all: bool = False,
        max_iter: int = 1000,
        seed: int = 1,
    ) -> None:
        self.imbalance_threshold = imbalance_threshold
        self.all = all
        self.max_iter = max_iter
        self.seed = seed

        self._fitted = False  # whether the model has been fitted

    def fit_transform(
        self, X: pd.DataFrame, y: pd.DataFrame = None
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:

        try:  # if missing y, will be None value; or will be dataframe, use df.empty for judge
            _empty = y.empty
        except AttributeError:
            _empty = y == None

        if (
            not _empty
        ):  # if no y input, only convert X; or, combine X and y to consider balancing
            features = list(X.columns)
            response = list(y.columns)
            data = pd.concat([X, y], axis=1)
        else:
            features = list(X.columns)
            response = None
            data = X

        _data = data.copy(deep=True)
        if not is_imbalance(_data, self.imbalance_threshold):
            warnings.warn("The dataset is balanced, no change.")
        else:
            if self.all == True:
                while is_imbalance(_data, self.imbalance_threshold):
                    _data = self._fit_transform(_data)
            else:
                _data = self._fit_transform(_data)

        self._fitted = True

        if not _empty:
            return _data[features], _data[response]
        else:
            return _data

    def _fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:

        from sklearn.neighbors import KNeighborsClassifier

        _imbalanced_feature, _majority = is_imbalance(
            X, self.imbalance_threshold, value=True
        )
        _seed = self.seed
        _iter = 0

        while (
            is_imbalance(X[[_imbalanced_feature]], self.imbalance_threshold)
            and _iter <= self.max_iter
        ):
            _minority_class = X.loc[X[_imbalanced_feature] != _majority]
            _majority_class = X.loc[X[_imbalanced_feature] == _majority]
            _subset = pd.concat(
                [_minority_class, _majority_class.sample(n=1, random_state=_seed)]
            ).reset_index(drop=True)
            neigh = KNeighborsClassifier(n_neighbors=1)
            neigh.fit(
                _subset.loc[:, _subset.columns != _imbalanced_feature],
                _subset[_imbalanced_feature],
            )
            y_predict = neigh.predict(
                X.loc[
                    ~X.index.isin(list(_subset.index)), X.columns != _imbalanced_feature
                ]
            )
            y_true = X.loc[
                ~X.index.isin(list(_subset.index)), X.columns == _imbalanced_feature
            ].values.T[0]
            _not_matching_index = np.where((np.array(y_predict) != np.array(y_true)))[0]
            X = pd.concat([_subset, X.iloc[_not_matching_index, :]]).reset_index(
                drop=True
            )
            _seed += 1
            _iter += 1

        return X


class OneSidedSelection(TomekLink, CondensedNearestNeighbor):

    """
    One Sided Selection (OSS)
    employs Tomek Link to remove noisy and border majority class samples, then use CNN to remove majority
    samples that are distinct to decision boundary

    Parameters
    ----------
    imbalance_threshold: determine to what extent will the data be considered as imbalanced data, default = 0.9

    norm: how the distance between different samples calculated, default = 'l2'
    all supported norm ['l1', 'l2']

    all: whether to stop until all features are balanced, default = False

    max_iter: Maximum number of iterations for over-/under-sampling, default = 1000

    seed: random seed, default = 1
    every random draw from the majority class will increase the random seed by 1
    """

    def __init__(
        self,
        imbalance_threshold: float = 0.9,
        norm: str = "l2",
        all: bool = False,
        max_iter: int = 1000,
        seed: int = 1,
    ) -> None:
        self.imbalance_threshold = imbalance_threshold
        self.norm = norm
        self.all = all
        self.max_iter = max_iter
        self.seed = seed

        self._fitted = False  # whether the model has been fitted

    def fit_transform(
        self, X: pd.DataFrame, y: pd.DataFrame = None
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:

        try:  # if missing y, will be None value; or will be dataframe, use df.empty for judge
            _empty = y.empty
        except AttributeError:
            _empty = y == None

        if (
            not _empty
        ):  # if no y input, only convert X; or, combine X and y to consider balancing
            features = list(X.columns)
            response = list(y.columns)
            data = pd.concat([X, y], axis=1)
        else:
            features = list(X.columns)
            response = None
            data = X

        _data = data.copy(deep=True)
        if not is_imbalance(_data, self.imbalance_threshold):
            warnings.warn("The dataset is balanced, no change.")
        else:
            super().__init__(
                imbalance_threshold=(1.0 + self.imbalance_threshold) / 2,
                norm=self.norm,
                all=self.all,
                max_iter=self.max_iter,
                seed=self.seed,
            )
            _data = super().fit_transform(_data)

            super(TomekLink, self).__init__(
                imbalance_threshold=self.imbalance_threshold,
                all=self.all,
                max_iter=self.max_iter,
                seed=self.seed,
            )
            _data = super(TomekLink, self).fit_transform(_data)

        self._fitted = True

        if not _empty:
            return _data[features], _data[response]
        else:
            return _data


class CNN_TomekLink(CondensedNearestNeighbor, TomekLink):

    """
    CNN_Tomek Link
    employs CNN first and Tomek Link to reduce the calculation for Tomek Link (large calculation for distance
    between each sample points, especially for large sample size)

    Parameters
    ----------
    imbalance_threshold: determine to what extent will the data be considered as imbalanced data, default = 0.9

    norm: how the distance between different samples calculated, default = 'l2'
    all supported norm ['l1', 'l2']

    all: whether to stop until all features are balanced, default = False

    max_iter: Maximum number of iterations for over-/under-sampling, default = 1000

    seed: random seed, default = 1
    every random draw from the majority class will increase the random seed by 1
    """

    def __init__(
        self,
        imbalance_threshold: float = 0.9,
        norm: str = "l2",
        all: bool = False,
        max_iter: int = 1000,
        seed: int = 1,
    ) -> None:
        self.imbalance_threshold = imbalance_threshold
        self.norm = norm
        self.all = all
        self.max_iter = max_iter
        self.seed = seed

        self._fitted = False  # whether the model has been fitted

    def fit_transform(
        self, X: pd.DataFrame, y: pd.DataFrame = None
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:

        try:  # if missing y, will be None value; or will be dataframe, use df.empty for judge
            _empty = y.empty
        except AttributeError:
            _empty = y == None

        if (
            not _empty
        ):  # if no y input, only convert X; or, combine X and y to consider balancing
            features = list(X.columns)
            response = list(y.columns)
            data = pd.concat([X, y], axis=1)
        else:
            features = list(X.columns)
            response = None
            data = X

        _data = data.copy(deep=True)
        if not is_imbalance(_data, self.imbalance_threshold):
            warnings.warn("The dataset is balanced, no change.")
        else:
            super().__init__(
                imbalance_threshold=(1.0 + self.imbalance_threshold) / 2,
                all=self.all,
                max_iter=self.max_iter,
                seed=self.seed,
            )
            _data = super().fit_transform(_data)

            super(CondensedNearestNeighbor, self).__init__(
                imbalance_threshold=self.imbalance_threshold,
                norm=self.norm,
                all=self.all,
                max_iter=self.max_iter,
                seed=self.seed,
            )
            _data = super(CondensedNearestNeighbor, self).fit_transform(_data)

        self._fitted = True

        if not _empty:
            return _data[features], _data[response]
        else:
            return _data
