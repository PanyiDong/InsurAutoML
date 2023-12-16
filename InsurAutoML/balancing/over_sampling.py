"""
File Name: over_sampling.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.5
Relative Path: /InsurAutoML/balancing/over_sampling.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Friday, 1st December 2023 10:29:09 pm
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

from ..constant import MAX_ITER
from .base import BaseBalancing
from ..utils.data import is_imbalance, LinkTable

"""
Reference for: Simple Random Over Sampling, Simple Random Under Sampling, Tomek Link, \
    Edited Nearest Neighbor,  Condensed Nearest Neighbor, One Sided Selection, CNN_TomekLink, \
    Smote, Smote_TomekLink, Smote_ENN

Batista, G.E., Prati, R.C. and Monard, M.C., 2004. A study of the behavior of several methods for
balancing machine learning training data. ACM SIGKDD explorations newsletter, 6(1), pp.20-29.
"""


class SimpleRandomOverSampling(BaseBalancing):

    """
    Simple Random Over-Sampling
    Randomly draw samples from minority class and replicate the sample

    Parameters
    ----------
    imbalance_threshold: determine to what extent will the data be considered as imbalanced data, default = 0.9

    all: whether to stop until all features are balanced, default = True

    max_iter: Maximum number of iterations for over-/under-sampling, default = 1024

    seed: random seed, default = 1
    every random draw from the minority class will increase the random seed by 1
    """

    def __init__(
        self,
        imbalance_threshold: float = 0.9,
        all: bool = True,
        max_iter: int = None,
        seed: int = None,
    ) -> None:
        self.imbalance_threshold = imbalance_threshold
        self.all = all
        self.max_iter = max_iter if max_iter is not None else MAX_ITER
        self.seed = seed if seed is not None else 1

        super().__init__()
        self._fitted = False  # whether the model has been fitted

    def fit_transform(
        self, X: pd.DataFrame, y: pd.DataFrame = None
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        try:  # if missing y, will be None value; or will be dataframe, use df.empty for judge
            _empty = y.empty
        except AttributeError:
            _empty = y is None

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
            if self.all:
                while is_imbalance(_data, self.imbalance_threshold):
                    _data = self._fit_transform(_data)
            else:
                _data = self._fit_transform(_data)

        self._fitted = True

        if not _empty:  # return balanced X and y if y is also inputted
            return _data[features], _data[response]
        else:
            return _data

    def _fit_transform(
        self, X: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]
    ) -> (
        pd.DataFrame
    ):  # using random over-sampling to balance the first imbalanced feature
        n, features = len(X), list(X.columns)
        _imbalanced_feature, _majority = is_imbalance(
            X, self.imbalance_threshold, value=True
        )
        _minority_class = X.loc[X[_imbalanced_feature] != _majority]

        # calculate the number of samples needed to be added
        # n_majority = n - len(_minority_class)
        # n_minor_needed = max(1, int(n_majority / self.imbalance_threshold - n))
        n_minor_needed = max(1, int(len(_minority_class) / 20))

        # randomly draw samples from minority class and replicate the sample
        X = pd.concat(
            [
                X,
                _minority_class.sample(
                    n=n_minor_needed, random_state=self.seed, replace=True
                ),
            ]
        )
        # shuffle the data
        X = sklearn.utils.shuffle(X.reset_index(drop=True)).reset_index(drop=True)

        return X


class Smote(BaseBalancing):

    """
    Synthetic Minority Over-sampling Technique (Smote)
    use over-sampling to generate minority class points using nearest neighbors

    Parameters
    ----------
    imbalance_threshold: determine to what extent will the data be considered as imbalanced data, default = 0.9

    norm: how the distance between different samples calculated, default = 'l2'
    all supported norm ['l1', 'l2']

    all: whether to stop until all features are balanced, default = True

    max_iter: Maximum number of iterations for over-/under-sampling, default = 1024

    seed: random seed, default = 1
    every random draw from the minority class will increase the random seed by 1

    k: number of nearest neighbors to choose from, default = 5
    the link sample will be chosen from these k nearest neighbors

    generation: how to generation new sample, default = 'mean'
    use link sample and random sample to generate the new sample
    """

    def __init__(
        self,
        imbalance_threshold: float = 0.9,
        norm: str = "l2",
        all: bool = True,
        max_iter: int = None,
        seed: int = None,
        k: int = 5,
        generation: str = "mean",
    ) -> None:
        self.imbalance_threshold = imbalance_threshold
        self.norm = norm
        self.all = all
        self.max_iter = max_iter if max_iter is not None else MAX_ITER
        self.seed = seed if seed is not None else 1
        self.k = k
        self.generation = generation

        super().__init__()
        self._fitted = False  # whether the model has been fitted

    def fit_transform(
        self, X: pd.DataFrame, y: pd.DataFrame = None
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        try:  # if missing y, will be None value; or will be dataframe, use df.empty for judge
            _empty = y.empty
        except AttributeError:
            _empty = y is None

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
            if self.all:
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
        n, features = len(X), list(X.columns)
        _imbalanced_feature, _majority = is_imbalance(
            X, self.imbalance_threshold, value=True
        )
        _minority_class = X.loc[X[_imbalanced_feature] != _majority]

        # calculate the number of samples needed to be added
        n_majority = n - len(_minority_class)
        n_minor_needed = max(1, int(n_majority / self.imbalance_threshold - n))

        for _ in range(n_minor_needed):
            _sample = _minority_class.sample(n=1, random_state=self.seed, replace=True)
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
                    ].mean(axis=0)
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

        return X
