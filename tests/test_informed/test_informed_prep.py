"""
File Name: test_informed_prep.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Actuarial and Risk Management Sciences, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.8
Relative Path: /tests/test_informed/test_informed_prep.py
File Created: Monday, 16th November 2026 9:51:37 am
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 16th November 2026 9:57:15 am
Modified By: Panyi Dong (panyid2@illinois.edu)

-----
MIT License

Copyright (c) 2026, Panyi Dong

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
import pytest

from InsurAutoML.prep import CompletePrepPipeline, MissingPrepPipeline
from InsurAutoML.prep.utils import CompleteSet, MissingSet


def test_missing_set_drops_columns_above_threshold():
    X = pd.DataFrame(
        {
            "keep": [1.0, 2.0, 3.0, 4.0],
            "some_missing": [1.0, np.nan, 3.0, 4.0],  # 25% missing
            "mostly_missing": [np.nan, np.nan, np.nan, 4.0],  # 75% missing
        }
    )

    selector = MissingSet(threshold=0.3)
    X_out = selector.fit_transform(X)

    assert selector.dropped_features_ == ["mostly_missing"]
    assert selector.kept_features_ == ["keep", "some_missing"]
    assert list(X_out.columns) == ["keep", "some_missing"]
    # rows are kept, MissingSet only removes columns
    assert len(X_out) == len(X)


def test_missing_set_requires_fit():
    with pytest.raises(ValueError):
        MissingSet().transform(pd.DataFrame({"a": [1.0]}))


def test_complete_set_drops_most_missing_until_threshold():
    X = pd.DataFrame(
        {
            "full": np.arange(10, dtype=float),
            "low": [np.nan] + list(np.arange(9, dtype=float)),  # 10% missing
            "high": [np.nan] * 6 + list(np.arange(4, dtype=float)),  # 60% missing
        }
    )

    selector = CompleteSet(threshold=0.8)
    X_out = selector.fit_transform(X)

    # dropping "high" gives 90% complete rows >= 80%
    assert selector.dropped_features_ == ["high"]
    assert selector.kept_features_ == ["full", "low"]
    # remaining incomplete rows are removed (complete cases only)
    assert not X_out.isna().any().any()
    assert len(X_out) == 9


def test_complete_set_keeps_protected_columns():
    X = pd.DataFrame(
        {
            "protected": [np.nan] * 6 + list(np.arange(4, dtype=float)),
            "other": [np.nan] + list(np.arange(9, dtype=float)),
        }
    )

    selector = CompleteSet(threshold=0.9, protected_cols=["protected"])
    selector.fit(X)

    assert "protected" in selector.kept_features_
    assert "protected" not in selector.dropped_features_


def test_complete_set_requires_fit():
    with pytest.raises(ValueError):
        CompleteSet().transform(pd.DataFrame({"a": [1.0]}))


def test_complete_prep_pipeline(numeric_missing):
    X, y = numeric_missing

    prep = CompletePrepPipeline(
        missing_threshold=0.5, cc_threshold=0.9, split_ratio=0.2
    )
    X_train, X_valid, y_train, y_valid = prep.fit(X, y)

    assert prep._fitted
    # complete cases only, no missing values left
    assert not X_train.isna().any().any() and not X_valid.isna().any().any()
    # train/validation split is disjoint and aligned with the response
    assert set(X_train.index).isdisjoint(X_valid.index)
    assert list(X_train.index) == list(y_train.index)
    assert list(X_valid.index) == list(y_valid.index)
    assert len(X_valid) > 0 and len(X_train) > len(X_valid)

    # transform applies the same column selection and keeps complete rows only
    X_new, y_new = prep.transform(X, y)
    assert y_new is not None
    assert list(X_new.columns) == list(X_train.columns)
    assert not X_new.isna().any().any()
    assert list(X_new.index) == list(y_new.index)


def test_complete_prep_pipeline_requires_fit(numeric_missing):
    X, _ = numeric_missing
    with pytest.raises(ValueError):
        CompletePrepPipeline().transform(X)


def test_missing_prep_pipeline(numeric_missing):
    X, y = numeric_missing

    prep = MissingPrepPipeline(
        missing_threshold=0.8, imputation_max_iter=3, split_ratio=0.2, cc_threshold=0.9
    )
    X_train, X_valid, y_train, y_valid = prep.fit(X, y)

    assert prep._fitted
    # MissForest imputes all values, every row is kept
    assert not X_train.isna().any().any() and not X_valid.isna().any().any()
    assert len(X_train) + len(X_valid) == len(X)
    assert set(X_train.index).isdisjoint(X_valid.index)
    assert list(X_train.index) == list(y_train.index)

    # transform imputes new data with the fitted imputer
    X_new, _ = prep.transform(X)
    assert len(X_new) == len(X)
    assert not X_new.isna().any().any()
    assert list(X_new.columns) == list(X_train.columns)


def test_missing_prep_pipeline_requires_fit(numeric_missing):
    X, _ = numeric_missing
    with pytest.raises(ValueError):
        MissingPrepPipeline().transform(X)


def test_prep_pipeline_length_mismatch(numeric_missing):
    X, y = numeric_missing
    with pytest.raises(ValueError):
        CompletePrepPipeline().fit(X, y.iloc[:-1])
