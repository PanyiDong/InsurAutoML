"""
File: _gam.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Latest Version: 0.2.0
Relative Path: /My_AutoML/_model/_gam.py
File Created: Friday, 15th April 2022 8:18:07 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Tuesday, 10th May 2022 7:18:39 pm
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


class GAM_Classifier:
    def __init__(
        self,
        type="logistic",
        tol=1e-4,
    ):
        self.type = type
        self.tol = tol

        self._fitted = False

    def fit(self, X, y):

        if self.type == "logistic":
            from pygam import LogisticGAM

            self.model = LogisticGAM(tol=self.tol)

        self.model.fit(X, y)

        self._fitted = True

        return self

    def predict(self, X):

        return self.model.predict(X)

    def predict_proba(self, X):

        return self.model.predict_proba(X)


class GAM_Regressor:
    def __init__(
        self,
        type="linear",
        tol=1e-4,
    ):
        self.type = type
        self.tol = tol

        self._fitted = False

    def fit(self, X, y):

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

    def predict(self, X):

        return self.model.predict(X)

    def predict_proba(self, X):

        raise NotImplementedError("predict_proba is not implemented for regression.")
