"""
File Name: _embed.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_feature_selection/_embed.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 14th November 2022 7:01:17 pm
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

from typing import Union, Tuple
import time
import numbers
import numpy as np
import pandas as pd

# import scipy
# import scipy.linalg
from sklearn.utils.extmath import stable_cumsum, svd_flip

# from My_AutoML._utils import class_means, class_cov, empirical_covariance

import warnings


class PCA_FeatureSelection:

    """
    Principal Component Analysis

    Use Singular Value Decomposition (SVD) to project data to a lower dimensional
    space, and thus achieve feature selection.

    Methods used:
    Full SVD: LAPACK, scipy.linalg.svd
    Trucated SVD: ARPACK, scipy.sparse.linalg.svds
    Randomized truncated SVD:

    Parameters
    ----------
    n_components: remaining features after selection, default = None

    solver: the method to perform SVD, default = 'auto'
    all choices ('auto', 'full', 'truncated', 'randomized')

    tol: Tolerance for singular values computed for truncated SVD

    n_iter: Number of iterations for randomized solver, default = 'auto'

    seed: random seed, default = 1
    """

    def __init__(
        self,
        n_components: int = None,
        solver: str = "auto",
        tol: float = 0.0,
        n_iter: str = "auto",
        seed: str = 1,
    ) -> None:
        self.n_components = n_components
        self.solver = solver
        self.tol = tol
        self.n_iter = n_iter
        self.seed = seed

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> PCA_FeatureSelection:

        n, p = X.shape

        # Deal with default n_componets = None
        if self.n_components == None:
            if self.solver != "truncated":
                n_components = min(n, p)
            else:
                n_components = min(n, p) - 1
        else:
            n_components = self.n_components

        if n_components <= 0:
            raise ValueError("Selection components must be larger than 0!")

        # Deal with solver
        self.fit_solver = self.solver
        if self.solver == "auto":
            if max(n, p) < 500:
                self.fit_solver = "full"
            elif n_components >= 1 and n_components < 0.8 * min(n, p):
                self.fit_solver = "randomized"
            else:
                self.fit_solver = "full"
        else:
            self.fit_solver = self.solver

        if self.fit_solver == "full":
            self.U_, self.S_, self.V_ = self._fit_full(X, n_components)
        elif self.fit_solver in ["truncated", "randomized"]:
            self.U_, self.S_, self.V_ = self._fit_truncated(
                X, n_components, self.fit_solver
            )
        else:
            raise ValueError("Not recognizing solver = {}!".format(self.fit_solver))

        self._fitted = True

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        _features = list(X.columns)

        U = self.U_[:, : self.n_components]

        # X_new = X * V = U * S * Vt * V = U * S
        X_new = U * self.S_[: self.n_components]
        # return dataframe format
        # X_new = pd.DataFrame(U, columns = _features[:self.n_components])

        return X_new

    def _fit_full(
        self, X: Union[pd.DataFrame, np.ndarray], n_components: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        n, p = X.shape
        if n_components < 0 or n_components > min(n, p):
            raise ValueError(
                "n_components must between 0 and {0:d}, but get {1:d}".format(
                    min(n, p), n_components
                )
            )
        elif not isinstance(n_components, numbers.Integral):
            raise ValueError(
                "Expect integer n_components, but get {:.6f}".format(n_components)
            )

        # center the data
        self._x_mean = np.mean(X, axis=0)
        X -= self._x_mean

        # solve for svd
        from scipy.linalg import svd

        U, S, V = svd(X, full_matrices=False)

        # make sure the max column values of U are positive, if not flip the column
        # and flip corresponding V rows
        max_abs_col = np.argmax(np.max(U), axis=0)
        signs = np.sign(U[max_abs_col, range(U.shape[1])])
        U *= signs
        V *= signs.reshape(-1, 1)

        _var = (S**2) / (n - 1)
        total_var = _var.sum()
        _var_ratio = _var / total_var

        if 0 < n_components < 1.0:
            # number of components for which the cumulated explained
            # variance percentage is superior to the desired threshold
            # side='right' ensures that number of features selected
            # their variance is always greater than n_components float
            # passed.
            ratio_cumsum = stable_cumsum(_var_ratio)
            n_components = np.searchsorted(ratio_cumsum, n_components, side="right") + 1
        # Compute noise covariance using Probabilistic PCA model
        # The sigma2 maximum likelihood (cf. eq. 12.46)
        if n_components < min(n, p):
            self._noise_variance_ = _var[n_components:].mean()
        else:
            self._noise_variance_ = 0.0

        self.n_samples, self.n_features = n, p
        self.components_ = V[:n_components]
        self.n_components = n_components
        self._var = _var[:n_components]
        self._var_ratio = _var_ratio[:n_components]
        self.singular_values = S[:n_components]

        return U, S, V

    def _fit_truncated(
        self, X: Union[pd.DataFrame, np.ndarray], n_components: int, solver: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        n, p = X.shape

        self._x_mean = np.mean(X, axis=0)
        X -= self._x_mean

        if solver == "truncated":

            from scipy.sparse.linalg import svds

            np.random.seed(self.seed)
            v0 = np.random.uniform(-1, 1, size=min(X.shape))
            U, S, V = svds(X.values, k=n_components, tol=self.tol, v0=v0)
            S = S[::-1]
            U, V = svd_flip(U[:, ::-1], V[::-1])
        elif solver == "randomized":

            from sklearn.utils.extmath import randomized_svd

            U, S, V = randomized_svd(
                np.array(X),
                n_components=n_components,
                n_iter=self.n_iter,
                flip_sign=True,
                random_state=self.seed,
            )

        self.n_samples, self.n_features = n, p
        self.components_ = V
        self.n_components = n_components

        # Get variance explained by singular values
        self._var = (S**2) / (n - 1)
        total_var = np.var(X, ddof=1, axis=0)
        self._var_ratio = self._var / total_var.sum()
        self.singular_values = S.copy()  # Store the singular values.

        if self.n_components < min(n, p):
            self._noise_variance_ = total_var.sum() - self._var.sum()
            self._noise_variance_ /= min(n, p) - n_components
        else:
            self._noise_variance_ = 0.0

        return U, S, V


# class LDASelection:
#     def __init__(
#         self,
#         priors=None,
#         n_components=None,
#     ):
#         self.priors = priors
#         self.n_components = n_components

#         self._fitted = False

#     def _eigen(self, X, y):

#         self.means_ = class_means(X, y)
#         self.covariance_ = class_cov(X, y, self.priors_)

#         Sw = self.covariance_  # within scatter
#         St = empirical_covariance(X)  # total scatter
#         Sb = St - Sw  # between scatter

#         evals, evecs = scipy.linalg.eigh(Sb, Sw)
#         self.explained_variance_ratio_ = np.sort(evals / np.sum(evals))[::-1][
#             : self._max_components
#         ]
#         evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors

#         self.scalings_ = evecs
#         self.coef_ = np.dot(self.means_, evecs).dot(evecs.T)
#         self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(
#             self.priors_
#         )

#     def fit(self, X, y):

#         self.classes_ = np.unique(y)
#         n, p = X.shape

#         if len(self.classes_) == n:
#             raise ValueError("Classes must be smaller than number of samples!")

#         if self.priors is None:  # estimate priors from sample
#             _y_uni = np.unique(y)  # non-negative ints
#             self.priors_ = []
#             for _value in _y_uni:
#                 if isinstance(y, pd.DataFrame):
#                     self.priors_.append(y.loc[y.values == _value].count()[0] / len(y))
#                 elif isinstance(y, pd.Series):
#                     self.priors_.append(y.loc[y.values == _value].count() / len(y))
#             self.priors_ = np.asarray(self.priors_)
#         else:
#             self.priors_ = np.asarray(self.priors)

#         if (self.priors_ < 0).any():
#             raise ValueError("priors must be non-negative")
#         if not np.isclose(self.priors_.sum(), 1.0):
#             warnings.warn("The priors do not sum to 1. Renormalizing", UserWarning)
#             self.priors_ = self.priors_ / self.priors_.sum()

#         max_components = min(
#             len(self.classes_) - 1, X.shape[1]
#         )  # maximum number of components
#         if self.n_components is None:
#             self._max_components = max_components
#         else:
#             if self.n_components > max_components:
#                 raise ValueError(
#                     "n_components cannot be larger than min(n_features, n_classes - 1)."
#                 )
#             self._max_components = self.n_components

#         self._fitted = True

#         return self

#     def transform(self, X):

#         X_new = np.dot(X, self.scalings_)

#         return X_new[:, : self._max_components]


class RBFSampler:

    """
    Implement of Weighted Sums of Random Kitchen Sinks

    Parameters
    ----------
    gamma: use to determine standard variance of random weight table, default = 1
    Parameter of RBF kernel: exp(-gamma * x^2).

    n_components: number of samples per original feature, default = 100

    seed: random generation seed, default = None
    """

    def __init__(
        self, gamma: float = 1.0, n_components: int = 100, seed: int = 1
    ) -> None:
        self.gamma = gamma
        self.n_components = n_components
        self.seed = seed

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> RBFSampler:

        if isinstance(X, list):
            n_features = len(X[0])
        else:
            n_features = X.shape[1]

        if self.n_components > n_features:
            warnings.warn(
                "N_components {} is larger than n_features {}, will set to n_features.".format(
                    self.n_components, n_features
                )
            )
            self.n_components = n_features
        else:
            self.n_components = self.n_components

        if not self.seed:
            self.seed = np.random.seed(int(time.time()))
        elif not isinstance(self.seed, int):
            raise ValueError("Seed must be integer, receive {}".format(self.seed))

        self._random_weights = np.random.normal(
            0, np.sqrt(2 * self.gamma), size=(n_features, self.n_components)
        )
        self._random_offset = np.random.uniform(
            0, 2 * np.pi, size=(1, self.n_components)
        )

        self._fitted = True

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:

        projection = np.dot(X, self._random_weights)
        projection += self._random_offset
        np.cos(projection, projection)
        projection *= np.sqrt(2.0 / self.n_components)

        # return dataframe
        # projection = pd.DataFrame(projection, columns = list(X.columns)[:self.n_components])

        return projection
