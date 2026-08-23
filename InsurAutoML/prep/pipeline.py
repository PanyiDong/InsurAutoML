"""
File Name: pipeline.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Actuarial and Risk Management Sciences, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.6
Relative Path: /InsurAutoML/prep/pipeline.py
File Created: Monday, 1st December 2025 1:07:34 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Friday, 26th December 2025 10:37:51 pm
Modified By: Panyi Dong (panyid2@illinois.edu)

-----
MIT License

Copyright (c) 2025, Panyi Dong

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
from missforest import MissForest

from ..ext import SPlit, TwinReduction
from ..imputation import MeanModeImputer, MissForestImputer
from ..feature_selection import CCCFilter, FeatureFilter
from .utils import CompleteSet, MissingSet


def standardization(X: pd.DataFrame) -> pd.DataFrame:
    """Standardize the numerical columns of the DataFrame."""
    X_std = X.copy()
    num_cols = X_std.select_dtypes(include=[np.number]).columns
    X_std[num_cols] = (X_std[num_cols] - X_std[num_cols].mean()) / X_std[num_cols].std()
    return X_std


class PrepPipeline:
    """
    Base Data Preparation Pipeline.

    This is an abstract base class for data preparation pipelines.
    Specific pipelines should implement fit() and transform() methods.

    Parameters
    ----------
    """

    def __init__(self) -> None:
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        raise NotImplementedError(
            "PrepPipeline is an abstract base class. Implement fit() in subclasses."
        )

    def transform(self, X: pd.DataFrame, y: pd.Series = None):
        raise NotImplementedError(
            "PrepPipeline is an abstract base class. Implement transform() in subclasses."
        )

    @staticmethod
    def _ensure_dataframe(X):
        """Ensure X is a DataFrame."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X = X.reset_index(drop=True)
        return X

    @staticmethod
    def _ensure_series(y):
        """Ensure y is a Series."""
        # flatten if 2d array-like
        if len(y.shape) > 1:
            y = np.ravel(y)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        y = y.reset_index(drop=True)
        return y

    def _ensure_xy(self, X, y=None):
        """Ensure X is a DataFrame and y is a Series and reset their indexes.

        - Converts X to `pd.DataFrame` if needed.
        - Converts y to `pd.Series` if needed (when not None).
        - If lengths differ, attempts to align by index intersection; raises if no overlap.
        - Resets indexes (drop=True) for both and returns (X, y).
        """
        X = self._ensure_dataframe(X)
        y = self._ensure_series(y) if y is not None else None

        # if lengths differ, raise error
        if y is not None and len(X) != len(y):
            raise ValueError(
                "Length of X and y differ and no overlapping index labels found to align."
            )

        return X, y


class CompletePrepPipeline(PrepPipeline):
    """
    Complete Data Preparation Pipeline.

    This pipeline performs a series of data preprocessing steps including:
    - Enforcing complete cases
    - CCC feature selection
    - SPlit data splitting

    Parameters
    ----------
    """

    def __init__(
        self,
        missing_threshold: float = 0.5,
        cc_threshold: float = 0.9,
        split_ratio: float = 0.1,
    ) -> None:
        self.missing_threshold = missing_threshold
        self.cc_threshold = cc_threshold
        self.split_ratio = split_ratio

        self._fitted = False
        super().__init__()

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # normalize inputs
        X, y = self._ensure_xy(X, y)
        # X = standardization(X)
        # select complete subsets
        self.selector = CompleteSet(threshold=self.missing_threshold)
        X_prep = self.selector.fit_transform(X, y)
        # ensure y aligns with selected rows from X
        y_prep = y[X_prep.index] if y is not None else None
        # CCC feature selection
        # self.filter = CCCFilter(threshold=self.cc_threshold)
        self.filter = FeatureFilter(criteria="CCC", n_prop=self.cc_threshold)
        X_prep = self.filter.fit_transform(X_prep, y_prep)
        # SPlit data splitting
        test_idx = X_prep.index[np.array(SPlit(X_prep, split_ratio=self.split_ratio))]
        train_idx = np.setdiff1d(X_prep.index, test_idx)

        self._fitted = True

        # subset y according to the selected X rows
        if y_prep is not None:
            return (
                X_prep.loc[train_idx],
                X_prep.loc[test_idx],
                y_prep.loc[train_idx],
                y_prep.loc[test_idx],
            )
        return X_prep.loc[train_idx], X_prep.loc[test_idx]

    def transform(self, X: pd.DataFrame, y: pd.Series = None):
        if not self._fitted:
            raise ValueError("CompletePrepPipeline must be fitted before transform().")
        # only apply selector and filter on test set
        X, y = self._ensure_xy(X, y)
        # X = standardization(X)
        X_prep = self.selector.transform(X)
        # align y with transformed X
        y_prep = y[X_prep.index] if y is not None else None
        X_prep = self.filter.transform(X_prep)

        return X_prep, y_prep if y is not None else None


class MissingPrepPipeline(PrepPipeline):
    """
    Missing Data Preparation Pipeline.

    This pipeline performs a series of data preprocessing steps including:
    - Enforcing missing cases
    - MeanMode imputation
    - TwinReduction data reduction
    - MissForest imputation
    - CCC feature selection
    - SPlit data splitting

    Parameters
    ----------
    """

    def __init__(
        self,
        missing_threshold: float = 0.8,
        twin_r: int = 50,
        imputation_max_iter: int = 10,
        imputation_n_estimators: int = 100,
        split_ratio: float = 0.1,
        cc_threshold: float = 0.9,
    ) -> None:
        self.missing_threshold = missing_threshold
        self.twin_r = twin_r
        self.imputation_max_iter = imputation_max_iter
        self.imputation_n_estimators = imputation_n_estimators
        self.split_ratio = split_ratio
        self.cc_threshold = cc_threshold

        self._fitted = False
        super().__init__()

    # def fit(self, X: pd.DataFrame, y: pd.Series = None):
    #     # normalize inputs
    #     X, y = self._ensure_xy(X, y)
    #     # X = standardization(X)
    #     # X_prep, y_prep = self._ensure_xy(X, y)
    #     # select missing subsets
    #     self.selector = MissingSet(threshold=self.missing_threshold)
    #     X_prep = self.selector.fit_transform(X, y)
    #     # get missing mask of selected data
    #     missing_mask = X_prep.isna().any(axis=1).to_numpy()
    #     # initial imputation with MeanMode
    #     self.imputer1 = MeanModeImputer()
    #     X_pre_imputed = self.imputer1.fill(X_prep)
    #     # TwinReduction data reduction on pre-imputed data
    #     _, idx_reduced = TwinReduction(
    #         X_pre_imputed, missing_mask, r=self.twin_r, u1=100, return_indices=True
    #     )
    #     # select reduced data
    #     X_prep, y_prep = X_prep.iloc[idx_reduced], (
    #         y[idx_reduced] if y is not None else None
    #     )
    #     # advanced imputation with MissForest
    #     # self.imputer2 = MissForestImputer(
    #     #     max_iter=self.imputation_max_iter,
    #     #     n_estimators=self.imputation_n_estimators,
    #     # )
    #     self.imputer2 = MissForest(
    #         max_iter=self.imputation_max_iter,
    #     )
    #     X_prep = self.imputer2.fit_transform(X_prep)
    #     # CCC feature selection
    #     # self.filter = CCCFilter(threshold=self.cc_threshold)
    #     self.filter = FeatureFilter(criteria="CCC", n_prop=self.cc_threshold)
    #     X_prep = self.filter.fit_transform(X_prep, y_prep)
    #     # SPlit data splitting
    #     test_idx = X_prep.index[np.array(SPlit(X_prep, split_ratio=self.split_ratio))]
    #     train_idx = np.setdiff1d(X_prep.index, test_idx)

    #     self._fitted = True

    #     # subset y according to the selected X rows
    #     if y_prep is not None:
    #         return (
    #             X_prep.loc[train_idx],
    #             X_prep.loc[test_idx],
    #             y_prep.loc[train_idx],
    #             y_prep.loc[test_idx],
    #         )
    #     return X_prep.loc[train_idx], X_prep.loc[test_idx]

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # normalize inputs
        X_prep, y_prep = self._ensure_xy(X, y)
        # select missing subsets
        self.selector = MissingSet(threshold=self.missing_threshold)
        X_prep = self.selector.fit_transform(X_prep, y_prep)
        # advanced imputation with MissForest
        # self.imputer2 = MissForestImputer(
        #     max_iter=self.imputation_max_iter,
        #     n_estimators=self.imputation_n_estimators,
        # )
        self.imputer2 = MissForest(
            max_iter=self.imputation_max_iter,
            verbose=0,
        )
        X_prep = self.imputer2.fit_transform(X_prep)
        # CCC feature selection
        # self.filter = CCCFilter(threshold=self.cc_threshold)
        self.filter = FeatureFilter(criteria="CCC", n_prop=self.cc_threshold)
        X_prep = self.filter.fit_transform(X_prep, y_prep)
        # SPlit data splitting
        test_idx = X_prep.index[np.array(SPlit(X_prep, split_ratio=self.split_ratio))]
        train_idx = np.setdiff1d(X_prep.index, test_idx)

        self._fitted = True

        # subset y according to the selected X rows
        if y_prep is not None:
            return (
                X_prep.loc[train_idx],
                X_prep.loc[test_idx],
                y_prep.loc[train_idx],
                y_prep.loc[test_idx],
            )
        return X_prep.loc[train_idx], X_prep.loc[test_idx]

    def transform(self, X: pd.DataFrame, y: pd.Series = None):
        if not self._fitted:
            raise ValueError("MissingPrepPipeline must be fitted before transform().")
        # only apply selector, imputers, and filter on test set
        # X, y = self._ensure_xy(X, y)
        # X = standardization(X)
        X_prep, y = self._ensure_xy(X, y)
        X_prep = self.selector.transform(X_prep)
        X_prep = self.imputer2.transform(X_prep)
        X_prep = self.filter.transform(X_prep)

        return X_prep, y if y is not None else None
