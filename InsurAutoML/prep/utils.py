"""
File Name: utils.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Actuarial and Risk Management Sciences, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.6
Relative Path: /InsurAutoML/prep/utils.py
File Created: Monday, 1st December 2025 11:29:42 am
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Tuesday, 2nd December 2025 3:20:52 pm
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

from __future__ import annotations
from typing import Optional, List
import pandas as pd

class MissingSet:
    """
    Column Dropper based on missing rate threshold.

    This transformer drops columns whose missing rate (fraction of NaNs) 
    exceeds a specified threshold.

    Parameters
    ----------
    threshold : float, default=0.3
        Maximum allowed missing fraction. Columns with missing rate > threshold will be dropped.
        (e.g., 0.30 means drop columns with >30% missing values).
    """

    def __init__(self, threshold: float = 0.3) -> None:
        self.threshold = threshold
        
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series =None) -> MissingSet:
        # Calculate missing rate for each column
        missing_series = X.isna().mean()
        
        # Identify columns to drop (where missing rate > threshold)
        self.dropped_features_ = missing_series[missing_series > self.threshold].index.tolist()
        self.kept_features_ = missing_series[missing_series <= self.threshold].index.tolist()
        
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise ValueError("MissingSet must be fitted before transform().")

        # Drop the columns
        X_clean = X.drop(columns=self.dropped_features_, errors='ignore')
        
        return X_clean

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
    
class CompleteSet:
    """
    Iterative missing Pruner (Complete Case Enforcer).

    This selector iteratively removes the column with the highest missing rate
    until the dataset's complete-case ratio (rows with no NaNs) meets a specified threshold.

    Parameters
    ----------
    threshold : float, default=0.5
        The required fraction of rows that must be complete (non-NaN) relative to 
        the original dataset size.
    protected_cols : list of str, optional
        List of column names that must NEVER be dropped (e.g., target variable, ID).
    verbose : bool, default=False
        Whether to print progress logs.
    """

    def __init__(
        self, 
        threshold: float = 0.5, 
        protected_cols: Optional[List[str]] = None, 
    ):
        self.threshold = threshold
        self.protected_cols = protected_cols if protected_cols is not None else []
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> CompleteSet:
        # Work on a copy of columns metadata to avoid modifying input
        # We only need to simulate the dropping process here.
        X_ = X.copy()
        n_total = len(X)
        removed = []

        while True:
            # 1. Calculate current complete-case ratio
            #    (rows with NO missing values in current subset of columns)
            n_cc = len(X_.dropna(axis=0))
            ratio = n_cc / n_total

            # 2. Check stop condition
            if ratio >= self.threshold:
                break

            # 3. Identify column to drop
            missing_rates = X_.isna().mean().sort_values(ascending=False)
            # Exclude protected columns from being candidates
            candidates = missing_rates.drop(self.protected_cols, errors='ignore')
            # If no removable columns remain (or all remaining have 0 missing), stop
            if candidates.empty or candidates.iloc[0] == 0:
                break

            col_to_drop = candidates.index[0]
            removed.append(col_to_drop)
            
            # Drop from simulation
            X_ = X_.drop(columns=[col_to_drop])

        self.dropped_features_ = removed
        self.kept_features_ = X_.columns.tolist()
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise ValueError("CompleteSet must be fitted before transform().")

        # 1. Drop the identified columns
        X_reduced = X.drop(columns=self.dropped_features_, errors='ignore')
        # 2. Drop rows that are still missing values (Complete Case)
        X_final = X_reduced.dropna(axis=0)
        
        return X_final

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)