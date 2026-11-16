"""
File Name: test_informed_pipeline.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Actuarial and Risk Management Sciences, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.8
Relative Path: /tests/test_informed/test_informed_pipeline.py
File Created: Monday, 16th November 2026 9:51:59 am
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 16th November 2026 9:57:11 am
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

from InsurAutoML.hpo.informed.informed_ensemble import InformedPipeline


class _Fitted:
    _fitted = True


class _CompletePrep(_Fitted):
    """Keep the first 4 rows as complete cases."""

    def transform(self, X, y=None):
        return X.iloc[:4], None


class _MissingPrep(_Fitted):
    """Keep all rows (imputed)."""

    def transform(self, X, y=None):
        return X, None


class _ConstantModel(_Fitted):
    def __init__(self, value, dtype: type = int, n_classes: int = 2):
        self.value, self.dtype, self.n_classes = value, dtype, n_classes

    def predict(self, X):
        return np.full(len(X), self.value, dtype=self.dtype)

    def predict_proba(self, X):
        proba = np.zeros((len(X), self.n_classes))
        proba[:, self.value] = 1.0
        return proba


@pytest.fixture
def X():
    return pd.DataFrame(
        {"a": np.arange(6, dtype=float), "b": np.arange(6, dtype=float)}
    )


def _pipeline(model_complete, model_missing):
    return InformedPipeline(
        complete_prep=_CompletePrep(),
        missing_prep=_MissingPrep(),
        model_complete=model_complete,
        model_missing=model_missing,
    )


@pytest.mark.parametrize("dtype", [int, bool])
def test_predict_averages_class_labels(X, dtype):
    # integer/boolean labels: averaging must not fail on in-place float division
    pipeline = _pipeline(_ConstantModel(1, dtype), _ConstantModel(0, dtype)).fit(X)

    y_pred = pipeline.predict(X)

    np.testing.assert_allclose(y_pred, [0.5, 0.5, 0.5, 0.5, 0.0, 0.0])


def test_predict_averages_regression(X):
    pipeline = _pipeline(_ConstantModel(4.0, float), _ConstantModel(2.0, float)).fit(X)

    np.testing.assert_allclose(pipeline.predict(X), [3.0, 3.0, 3.0, 3.0, 2.0, 2.0])


def test_predict_proba_averages(X):
    pipeline = _pipeline(
        _ConstantModel(1, n_classes=2), _ConstantModel(0, n_classes=2)
    ).fit(X)

    proba = pipeline.predict_proba(X)

    assert proba.shape == (6, 2)
    np.testing.assert_allclose(proba[:4], 0.5)
    np.testing.assert_allclose(proba[4:], [[1.0, 0.0], [1.0, 0.0]])
    np.testing.assert_allclose(np.asarray(proba).sum(axis=1), 1.0)


def test_predict_requires_fit(X):
    with pytest.raises(ValueError):
        _pipeline(_ConstantModel(0), _ConstantModel(0)).predict(X)


def test_fit_requires_fitted_components(X):
    unfitted = _ConstantModel(0)
    unfitted._fitted = False

    with pytest.raises(ValueError):
        _pipeline(unfitted, _ConstantModel(0)).fit(X)
    with pytest.raises(ValueError):
        InformedPipeline(
            complete_prep=_CompletePrep(), missing_prep=_MissingPrep()
        ).fit(X)
