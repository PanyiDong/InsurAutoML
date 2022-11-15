"""
File Name: _mixed_sampling.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_balancing/_mixed_sampling.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 14th November 2022 6:58:03 pm
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

from InsurAutoML._utils._data import is_imbalance
from ._over_sampling import Smote
from ._under_sampling import TomekLink, EditedNearestNeighbor

"""
Reference for: Simple Random Over Sampling, Simple Random Under Sampling, Tomek Link, \
    Edited Nearest Neighbor,  Condensed Nearest Neighbor, One Sided Selection, CNN_TomekLink, \
    Smote, Smote_TomekLink, Smote_ENN
    
Batista, G.E., Prati, R.C. and Monard, M.C., 2004. A study of the behavior of several methods for 
balancing machine learning training data. ACM SIGKDD explorations newsletter, 6(1), pp.20-29.
"""


class Smote_TomekLink(Smote, TomekLink):

    """
    Run Smote then run Tomek Link to balance dataset
    """

    def __init__(
        self,
        imbalance_threshold: float = 0.9,
        norm: str = "l2",
        all: bool = False,
        max_iter: int = 1000,
        k: int = 5,
        generation: str = "mean",
        seed: int = 1,
    ) -> None:
        self.imbalance_threshold = imbalance_threshold
        self.norm = norm
        self.all = all
        self.max_iter = max_iter
        self.seed = seed
        self.k = k
        self.generation = generation

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

        self._fitted = True

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
        imbalance_threshold: float = 0.9,
        norm: str = "l2",
        all: bool = False,
        max_iter: int = 1000,
        seed: int = 1,
        k: int = 5,
        generation: str = "mean",
    ) -> None:
        self.imbalance_threshold = imbalance_threshold
        self.norm = norm
        self.all = all
        self.max_iter = max_iter
        self.seed = seed
        self.k = k
        self.generation = generation

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

        self._fitted = True

        if not _empty:
            return _data[features], _data[response]
        else:
            return _data
