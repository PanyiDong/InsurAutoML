"""
File Name: _xgboost.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_model/_xgboost.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 14th November 2022 8:15:52 pm
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

from typing import Union
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor

#####################################################################################################################
# XGBoost support


class XGBoost_Base:

    """
    XGBoost Base model

    Parameters
    ----------
    task_type: task type, take either "classification" or regression, default = "classification"

    eta: step size shrinkage used in update to prevents overfitting, default = 0.3
    alias: learning_rate

    gamma: minimum loss reduction required to make a further partition, default = 0

    max_depth: maximum depth of a tree, default = 6

    min_child_weight: minimum sum of instance weight (hessian) needed in a child, default = 1

    max_delta_step: maximum delta step we allow each leaf output to be, default = 0

    reg_lambda: L2 regularization term on weights, default = 1

    reg_alpha: L1 regularization term on weights, default = 0
    """

    def __init__(
        self,
        task_type: str = "classification",
        eta: float = 0.3,
        gamma: float = 0.0,
        max_depth: int = 6,
        min_child_weight: float = 1.0,
        max_delta_step: float = 0.0,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
    ) -> None:
        self.task_type = task_type
        self.eta = eta
        self.gamma = gamma
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha

        self._fitted = False

    def fit(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> XGBoost_Base:

        if self.task_type == "classification":
            self.model = XGBClassifier(
                eta=self.eta,
                gamma=self.gamma,
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                max_delta_step=self.max_delta_step,
                reg_lambda=self.reg_lambda,
                reg_alpha=self.reg_alpha,
            )
        elif self.task_type == "regression":
            self.model = XGBRegressor(
                eta=self.eta,
                gamma=self.gamma,
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                max_delta_step=self.max_delta_step,
                reg_lambda=self.reg_lambda,
                reg_alpha=self.reg_alpha,
            )

        self.model.fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        return self.model.predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        return self.model.predict_proba(X)


class XGBoost_Classifier(XGBoost_Base):

    """
    XGBoost Classification model

    Parameters
    ----------
    eta: step size shrinkage used in update to prevents overfitting, default = 0.3
    alias: learning_rate

    gamma: minimum loss reduction required to make a further partition, default = 0

    max_depth: maximum depth of a tree, default = 6

    min_child_weight: minimum sum of instance weight (hessian) needed in a child, default = 1

    max_delta_step: maximum delta step we allow each leaf output to be, default = 0

    reg_lambda: L2 regularization term on weights, default = 1

    reg_alpha: L1 regularization term on weights, default = 0
    """

    def __init__(
        self,
        eta: float = 0.3,
        gamma: float = 0.0,
        max_depth: int = 6,
        min_child_weight: float = 1.0,
        max_delta_step: float = 0.0,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
    ) -> None:
        self.eta = eta
        self.gamma = gamma
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha

        self._fitted = False

        super().__init__(
            task_type="classification",
            eta=self.eta,
            gamma=self.gamma,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            max_delta_step=self.max_delta_step,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
        )

    def fit(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> XGBoost_Classifier:

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        return super().predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        return super().predict_proba(X)


class XGBoost_Regressor(XGBoost_Base):

    """
    XGBoost Regression model

    Parameters
    ----------
    eta: step size shrinkage used in update to prevents overfitting, default = 0.3
    alias: learning_rate

    gamma: minimum loss reduction required to make a further partition, default = 0

    max_depth: maximum depth of a tree, default = 6

    min_child_weight: minimum sum of instance weight (hessian) needed in a child, default = 1

    max_delta_step: maximum delta step we allow each leaf output to be, default = 0

    reg_lambda: L2 regularization term on weights, default = 1

    reg_alpha: L1 regularization term on weights, default = 0
    """

    def __init__(
        self,
        eta: float = 0.3,
        gamma: float = 0.0,
        max_depth: int = 6,
        min_child_weight: float = 1.0,
        max_delta_step: float = 0.0,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
    ) -> None:
        self.eta = eta
        self.gamma = gamma
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha

        self._fitted = False

        super().__init__(
            task_type="regression",
            eta=self.eta,
            gamma=self.gamma,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            max_delta_step=self.max_delta_step,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
        )

    def fit(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> XGBoost_Regressor:

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        return super().predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        raise NotImplementedError("predict_proba is not implemented for regression.")
