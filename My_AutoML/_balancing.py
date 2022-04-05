"""
File: _balancing.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Relative Path: /_balancing.py
File Created: Friday, 25th February 2022 6:13:42 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 4th April 2022 7:58:41 pm
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

import numpy as np
import pandas as pd
import warnings
import sklearn
import sklearn.utils

# determine whether the data contains imbalanced data
# if value, returns the column header and majority class from the unbalanced dataset
def is_imbalance(data, threshold, value=False):

    features = list(data.columns)

    for _column in features:
        unique_values = data[_column].unique()

        if len(unique_values) == 1:  # only one class exists
            if value:
                return None, None
            else:
                return False

        for _value in unique_values:
            if (
                len(data.loc[data[_column] == _value, _column]) / len(data[_column])
                > threshold
            ):
                if value:
                    return _column, _value
                else:
                    return True
    if value:
        return None, None
    else:
        return False


# return the distance between sample and the table sample points
# notice: the distance betwen sample and sample itself will be included, be aware to deal with it
# supported norm ['l1', 'l2']
def LinkTable(sample, table, norm="l2"):

    if sample.shape[1] != table.shape[1]:
        raise ValueError("Not same size of columns!")

    _sample = sample.values
    features = list(table.columns)
    _table = table.copy(deep=True)
    _linktable = []

    for sample_point in _sample:
        for i in range(len(features)):
            if norm == "l2":
                # print(sample[_column], sample[_column][0])
                _table.iloc[:, i] = (_table.iloc[:, i] - sample_point[i]) ** 2
            if norm == "l1":
                _table.iloc[:, i] = np.abs(_table.iloc[:, i] - sample_point[i])
        _linktable.append(_table.sum(axis=1).values.tolist())

    return _linktable


"""
Reference for: Simple Random Over Sampling, Simple Random Under Sampling, Tomek Link, \
    Edited Nearest Neighbor,  Condensed Nearest Neighbor, One Sided Selection, CNN_TomekLink, \
    Smote, Smote_TomekLink, Smote_ENN
    
Batista, G.E., Prati, R.C. and Monard, M.C., 2004. A study of the behavior of several methods for 
balancing machine learning training data. ACM SIGKDD explorations newsletter, 6(1), pp.20-29.
"""


class ExtremeClass:

    """
    remove the features where only one unique class exists, since no information provided by the feature

    Parameters
    ----------
    extreme_threshold: default = 1
    the threshold percentage of which the class holds in the feature will drop the feature
    """

    def __init__(self, extreme_threshold=1):
        self.extreme_threshold = extreme_threshold

    def cut(self, X):

        _X = X.copy(deep=True)

        features = list(_X.columns)
        for _column in features:
            unique_values = sorted(_X[_column].dropna().unique())
            for _value in unique_values:
                if (
                    len(X.loc[X[_column] == _value, _column]) / len(X)
                    >= self.extreme_threshold
                ):
                    _X.remove(labels=_column, inplace=True)
                    break
        return _X


"""
Oct. 29, 2021 Update: Since balancing requries input of both X and y (features and response), 
or the shape of X and y will cause problems (over-/under-sampling will increase/decrease the 
observations), add a choice of input for y in all fit_transform methods so the shape will be
consistent. The idea is to combine X and y if y exists, and record the features/response column
names. When balancing completed, divide the combined dataset into balanced X and y.
"""


class SimpleRandomOverSampling:

    """
    Simple Random Over-Sampling
    Randomly draw samples from minority class and replicate the sample

    Parameters
    ----------
    imbalance_threshold: determine to what extent will the data be considered as imbalanced data, default = 0.9

    all: whether to stop until all features are balanced, default = False

    max_iter: Maximum number of iterations for over-/under-sampling, default = 1000

    seed: random seed, default = 1
    every random draw from the minority class will increase the random seed by 1
    """

    def __init__(
        self,
        imbalance_threshold=0.9,
        all=False,
        max_iter=1000,
        seed=1,
    ):
        self.imbalance_threshold = imbalance_threshold
        self.all = all
        self.max_iter = max_iter
        self.seed = seed

    def fit_transform(self, X, y=None):

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

        if not _empty:  # return balanced X and y if y is also inputted
            return _data[features], _data[response]
        else:
            return _data

    def _fit_transform(
        self, X
    ):  # using random over-sampling to balance the first imbalanced feature

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
            _minority_class = X.loc[X[_imbalanced_feature] != _majority]
            X = pd.concat([X, _minority_class.sample(n=1, random_state=_seed)])
            _seed += 1
            _iter += 1
        X = sklearn.utils.shuffle(X.reset_index(drop=True)).reset_index(drop=True)

        return X


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
        imbalance_threshold=0.9,
        all=False,
        max_iter=1000,
        seed=1,
    ):
        self.imbalance_threshold = imbalance_threshold
        self.all = all
        self.max_iter = max_iter
        self.seed = seed

    def fit_transform(self, X, y=None):

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

        if not _empty:
            return _data[features], _data[response]
        else:
            return _data

    def _fit_transform(
        self, X
    ):  # using random over-sampling to balance the first imbalanced feature

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
        self, imbalance_threshold=0.9, norm="l2", all=False, max_iter=1000, seed=1
    ):
        self.imbalance_threshold = imbalance_threshold
        self.norm = norm
        self.all = all
        self.max_iter = max_iter
        self.seed = seed

    def fit_transform(self, X, y=None):

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

        if not _empty:
            return _data[features], _data[response]
        else:
            return _data

    def _fit_transform(self, X):

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
            for _link_item in _link_table:
                _nearest = _link_item.index(
                    sorted(_link_item)[1]
                )  # since the closest will always be the sample itself
                if X.iloc[_nearest, :][_imbalanced_feature] == _majority:
                    X = X.drop(X.index[_nearest]).reset_index(drop=True)
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
        imbalance_threshold=0.9,
        norm="l2",
        all=False,
        max_iter=1000,
        seed=1,
        k=3,
    ):
        self.imbalance_threshold = imbalance_threshold
        self.norm = norm
        self.all = all
        self.max_iter = max_iter
        self.seed = seed
        self.k = k

    def fit_transform(self, X, y=None):

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

        if not _empty:
            return _data[features], _data[response]
        else:
            return _data

    def _fit_transform(self, X):

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

    def __init__(self, imbalance_threshold=0.9, all=False, max_iter=1000, seed=1):
        self.imbalance_threshold = imbalance_threshold
        self.all = all
        self.max_iter = max_iter
        self.seed = seed

    def fit_transform(self, X, y=None):

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

        if not _empty:
            return _data[features], _data[response]
        else:
            return _data

    def _fit_transform(self, X):

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
        self, imbalance_threshold=0.9, norm="l2", all=False, max_iter=1000, seed=1
    ):
        self.imbalance_threshold = imbalance_threshold
        self.norm = norm
        self.all = all
        self.max_iter = max_iter
        self.seed = seed

    def fit_transform(self, X, y=None):

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
        self, imbalance_threshold=0.9, norm="l2", all=False, max_iter=1000, seed=1
    ):
        self.imbalance_threshold = imbalance_threshold
        self.norm = norm
        self.all = all
        self.max_iter = max_iter
        self.seed = seed

    def fit_transform(self, X, y=None):

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

        if not _empty:
            return _data[features], _data[response]
        else:
            return _data


class Smote:

    """
    Synthetic Minority Over-sampling Technique (Smote)
    use over-sampling to generate minority class points using nearest neighbors

    Parameters
    ----------
    imbalance_threshold: determine to what extent will the data be considered as imbalanced data, default = 0.9

    norm: how the distance between different samples calculated, default = 'l2'
    all supported norm ['l1', 'l2']

    all: whether to stop until all features are balanced, default = False

    max_iter: Maximum number of iterations for over-/under-sampling, default = 1000

    seed: random seed, default = 1
    every random draw from the minority class will increase the random seed by 1

    k: number of nearest neighbors to choose from, default = 5
    the link sample will be chosen from these k nearest neighbors

    generation: how to generation new sample, default = 'mean'
    use link sample and random sample to generate the new sample
    """

    def __init__(
        self,
        imbalance_threshold=0.9,
        norm="l2",
        all=False,
        max_iter=1000,
        seed=1,
        k=5,
        generation="mean",
    ):
        self.imbalance_threshold = imbalance_threshold
        self.norm = norm
        self.all = all
        self.max_iter = max_iter
        self.seed = seed
        self.k = k
        self.generation = generation

    def fit_transform(self, X, y=None):

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

        if not _empty:
            return _data[features], _data[response]
        else:
            return _data

    def _fit_transform(self, X):

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
            _sample = _minority_class.sample(n=1, random_state=_seed)
            _link_table = LinkTable(_sample, X, self.norm)
            for _link_item in _link_table:
                _k_nearest = [
                    _link_item.index(item)
                    for item in sorted(_link_item)[1 : (self.k + 1)]
                ]
                _link = _k_nearest[np.random.randint(0, len(_k_nearest))]
                if self.generation == "mean":
                    X.loc[len(X), :] = X.loc[
                        [_sample.index[0], X.index[_link]], :
                    ].mean()
                elif self.generation == "random":
                    X.loc[len(X), :] = X.loc[_sample.index, :] + np.random.rand() * (
                        X.loc[X.index[_link], :] - X.lox[_sample.index, :]
                    )
                else:
                    raise ValueError(
                        'Not recognizing generation method! Should be in \
                        ["mean", "random"], get {}'.format(
                            self.generation
                        )
                    )
            _seed += 1
            _iter += 1

        return X


class Smote_TomekLink(Smote, TomekLink):

    """
    Run Smote then run Tomek Link to balance dataset
    """

    def __init__(
        self,
        imbalance_threshold=0.9,
        norm="l2",
        all=False,
        max_iter=1000,
        seed=1,
        k=5,
        generation="mean",
    ):
        self.imbalance_threshold = imbalance_threshold
        self.norm = norm
        self.all = all
        self.max_iter = max_iter
        self.seed = seed
        self.k = k
        self.generation = generation

    def fit_transform(self, X, y=None):

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
                k=self.k,
                generation=self.generation,
            )
            _data = super().fit_transform(_data)

            super(Smote, self).__init__(
                imbalance_threshold=self.imbalance_threshold,
                norm=self.norm,
                all=self.all,
                max_iter=self.max_iter,
                seed=self.seed,
            )
            _data = super(Smote, self).fit_transform(_data)

        if not _empty:
            return _data[features], _data[response]
        else:
            return _data


class Smote_ENN(Smote, EditedNearestNeighbor):

    """
    Run Smote then run ENN to balance dataset
    """

    def __init__(
        self,
        imbalance_threshold=0.9,
        norm="l2",
        all=False,
        max_iter=1000,
        seed=1,
        k=5,
        generation="mean",
    ):
        self.imbalance_threshold = imbalance_threshold
        self.norm = norm
        self.all = all
        self.max_iter = max_iter
        self.seed = seed
        self.k = k
        self.generation = generation

    def fit_transform(self, X, y=None):

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
                k=self.k,
                generation=self.generation,
            )
            _data = super().fit_transform(_data)

            super(Smote, self).__init__(
                imbalance_threshold=self.imbalance_threshold,
                all=self.all,
                max_iter=self.max_iter,
                seed=self.seed,
                k=self.k,
            )
            _data = super(Smote, self).fit_transform(_data)

        if not _empty:
            return _data[features], _data[response]
        else:
            return _data
