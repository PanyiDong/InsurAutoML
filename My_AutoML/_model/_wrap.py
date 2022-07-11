"""
File: _wrap.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Last Version: 0.2.1
Relative Path: /My_AutoML/_model/_wrap.py
File Created: Sunday, 19th June 2022 10:33:15 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Sunday, 19th June 2022 10:38:02 pm
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


class BaseClassificationWrapper:
    def __init__(self, estimator) -> None:
        self.estimator = estimator

        self._fitted = False

    def fit(self, X, y=None):

        self.estimator.fit(X, y)

        self._fitted = True

        return self

    def predict(self, X):

        if not self._fitted:
            raise Exception("Model has not been fitted yet.")

        return self.estimator.predict(X)

    def predict_proba(self, X):

        if not self._fitted:
            raise Exception("Model has not been fitted yet.")

        return self.estimator.predict_proba(X)


class BaseRegressionWrapper:
    def __init__(self, estimator) -> None:
        self.estimator = estimator

        self._fitted = False

    def fit(self, X, y=None):

        self.estimator.fit(X, y)

        self._fitted = True

        return self

    def predict(self, X):

        if not self._fitted:
            raise Exception("Model has not been fitted yet.")

        return self.estimator.predict(X)

    def predict_proba(self, X):

        raise NotImplementedError("predict_proba is not implemented for regression.")
