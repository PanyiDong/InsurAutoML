"""
File: _sklearn.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Latest Version: 0.2.0
Relative Path: /My_AutoML/_model/_sklearn.py
File Created: Monday, 18th April 2022 12:14:53 am
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 18th April 2022 9:41:29 am
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

import sklearn.linear_model
import sklearn.naive_bayes

# need to enable hist gradient boosting features first
# no need for sklearn version >= 1.0.0
# since conflict between sklearn >=1.0.0 and autosklearn,
# still use 0.24.2 sklearn here
from sklearn.experimental import enable_hist_gradient_boosting
import sklearn.ensemble

####################################################################################################################
# models from sklearn
# wrap for some-degree of flexibility (initialization, _fitted, etc.)

####################################################################################################################
# classifiers


class LogisticRegression(sklearn.linear_model.LogisticRegression):
    def __init__(
        self,
        penalty="l2",
        tol=1e-4,
        C=1.0,
    ):
        self.penalty = penalty
        self.tol = tol
        self.C = C

        super().__init__(
            penalty=self.penalty,
            tol=self.tol,
            C=self.C,
        )

        self._fitted = False

    def fit(self, X, y):

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(self, X):

        return super().predict(X)


class ComplementNB(sklearn.naive_bayes.ComplementNB):
    def __init__(
        self,
        alpha=1.0,
        fit_prior=True,
        norm=False,
    ):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.norm = norm

        super().__init__(
            alpha=self.alpha,
            fit_prior=self.fit_prior,
            norm=self.norm,
        )

        self._fitted = False

    def fit(self, X, y):

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(self, X):

        return super().predict(X)


class HistGradientBoostingClassifier(sklearn.ensemble.HistGradientBoostingClassifier):
    def __init__(
        self,
        loss="auto",
        learning_rate=0.1,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=20,
        l2_regularization=0,
        tol=1e-7,
    ):
        self.loss = loss
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.tol = tol

        super().__init__(
            loss=self.loss,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            tol=self.tol,
        )

        self._fitted = False

    def fit(self, X, y):

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(self, X):

        return super().predict(X)


####################################################################################################################
# regressors


class LinearRegression(sklearn.linear_model.LinearRegression):
    def __init__(
        self,
    ):
        super().__init__()

        self._fitted = False

    def fit(self, X, y):

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(self, X):

        return super().predict(X)


class Lasso(sklearn.linear_model.Lasso):
    def __init__(
        self,
        alpha=1.0,
        tol=1e-4,
    ):
        self.alpha = alpha
        self.tol = tol

        super().__init__(
            alpha=self.alpha,
            tol=self.tol,
        )

        self._fitted = False

    def fit(self, X, y):

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(self, X):

        return super().predict(X)


class Ridge(sklearn.linear_model.Ridge):
    def __init__(
        self,
        alpha=1.0,
        tol=1e-3,
        solver="auto",
    ):
        self.alpha = alpha
        self.tol = tol
        self.solver = solver

        super().__init__(
            alpha=self.alpha,
            tol=self.tol,
            solver=self.solver,
        )

        self._fitted = False

    def fit(self, X, y):

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(self, X):

        return super().predict(X)


class ElasticNet(sklearn.linear_model.ElasticNet):
    def __init__(
        self,
        alpha=1.0,
        l1_ratio=0.5,
        tol=1e-4,
        selection="cyclic",
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.tol = tol
        self.selection = selection

        super().__init__(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            tol=self.tol,
            selection=self.selection,
        )

        self._fitted = False

    def fit(self, X, y):

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(self, X):

        return super().predict(X)


class BayesianRidge(sklearn.linear_model.BayesianRidge):
    def __init__(
        self,
        tol=1e-3,
        alpha_1=1e-6,
        alpha_2=1e-6,
        lambda_1=1e-6,
        lambda_2=1e-6,
    ):
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

        super().__init__(
            tol=self.tol,
            alpha_1=self.alpha_1,
            alpha_2=self.alpha_2,
            lambda_1=self.lambda_1,
            lambda_2=self.lambda_2,
        )

        self._fitted = False

    def fit(self, X, y):

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(self, X):

        return super().predict(X)


class HistGradientBoostingRegressor(sklearn.ensemble.HistGradientBoostingRegressor):
    def __init__(
        self,
        loss="least_squares",
        learning_rate=0.1,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=20,
        l2_regularization=0,
        tol=1e-7,
    ):
        self.loss = loss
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.tol = tol

        super().__init__(
            loss=self.loss,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            tol=self.tol,
        )

        self._fitted = False

    def fit(self, X, y):

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(self, X):

        return super().predict(X)
