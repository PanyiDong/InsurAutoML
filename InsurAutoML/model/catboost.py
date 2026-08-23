"""
File Name: catboost.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Actuarial and Risk Management Sciences, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.6
Relative Path: /InsurAutoML/model/catboost.py
File Created: Wednesday, 17th December 2025 12:42:17 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Wednesday, 17th December 2025 7:30:28 pm
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
from catboost import CatBoostRegressor, CatBoostClassifier
from .base import BaseModel


class CatBoost_Regressor(BaseModel):
    def __init__(
        self,
        loss_function: str = "RMSE",
        iterations: int = 1000,
        learning_rate: float = 0.03,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        bagging_temperature: float = 1.0,
        random_strength: float = 1.0,
    ) -> None:
        self._fitted = False
        super().__init__()
        self._model = CatBoostRegressor(
            loss_function=loss_function,
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            bagging_temperature=bagging_temperature,
            random_strength=random_strength,
        )
        self._fitted = False

    def fit(self, X, y) -> CatBoost_Regressor:
        self._model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X):
        return self._model.predict(X)


class CatBoost_Classifier(BaseModel):
    def __init__(
        self,
        loss_function: str = "Accuracy",
        iterations: int = 1000,
        learning_rate: float = 0.03,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        bagging_temperature: float = 1.0,
        random_strength: float = 1.0,
    ) -> None:
        self._fitted = False
        super().__init__()
        self._model = CatBoostClassifier(
            loss_function=loss_function,
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            bagging_temperature=bagging_temperature,
            random_strength=random_strength,
        )
        self._fitted = False

    def fit(self, X, y) -> CatBoost_Classifier:
        self._model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)
