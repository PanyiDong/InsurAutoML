"""
File Name: _gam.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_model/_gam.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 14th November 2022 8:15:04 pm
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


class GAM_Classifier:
    def __init__(
        self,
        type: str = "logistic",
        tol: float = 1e-4,
    ) -> None:
        self.type = type
        self.tol = tol

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> GAM_Classifier:

        if self.type == "logistic":
            from pygam import LogisticGAM

            self.model = LogisticGAM(tol=self.tol)

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


class GAM_Regressor:
    def __init__(
        self,
        type: str = "linear",
        tol: float = 1e-4,
    ) -> None:
        self.type = type
        self.tol = tol

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> GAM_Regressor:

        if self.type == "linear":
            from pygam import LinearGAM

            self.model = LinearGAM(tol=self.tol)
        elif self.type == "gamma":
            from pygam import GammaGAM

            self.model = GammaGAM(tol=self.tol)
        elif self.type == "poisson":
            from pygam import PoissonGAM

            self.model = PoissonGAM(tol=self.tol)
        elif self.type == "inverse_gaussian":
            from pygam import InvGaussGAM

            self.model = InvGaussGAM(tol=self.tol)

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

        raise NotImplementedError("predict_proba is not implemented for regression.")
