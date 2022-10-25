"""
File: _sklearn.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Latest Version: 0.2.0
Relative Path: /My_AutoML/_feature_selection/_sklearn.py
File Created: Friday, 29th April 2022 10:38:02 am
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 24th October 2022 10:53:40 pm
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

import scipy
import numpy as np
import sklearn.feature_selection
import sklearn.decomposition
import sklearn.cluster
import sklearn.kernel_approximation
import sklearn.preprocessing
import sklearn.ensemble
import warnings

from InsurAutoML._utils._base import is_none

###################################################################################################################
# sklearn replacement of feature selection


class densifier:
    def __init__(
        self,
    ):
        self._fitted = False

    def fit(self, X, y=None):
        self._fitted = True
        return self

    def transform(self, X):
        from scipy import sparse

        if sparse.issparse(X):
            return X.todense().getA()
        else:
            return X


class extra_trees_preproc_for_classification:
    def __init__(
        self,
        n_estimators=5,
        criterion="entropy",
        min_samples_leaf=5,
        min_samples_split=5,
        max_features=0.5,
        bootstrap=False,
        max_leaf_nodes=None,
        max_depth=None,
        min_weight_fraction_leaf=0.0,
        min_impurity_decrease=0.0,
    ):
        self.n_estimators = int(n_estimators)
        self.criterion = criterion
        self.min_samples_leaf = int(min_samples_leaf)
        self.min_samples_split = int(min_samples_split)
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_leaf_nodes = None if is_none(max_leaf_nodes) else int(max_leaf_nodes)
        self.max_depth = None if is_none(max_depth) else int(max_depth)
        self.min_weight_fraction_leaf = float(min_weight_fraction_leaf)
        self.min_impurity_decrease = float(min_impurity_decrease)

        self._fitted = False

    def fit(self, X, y=None):

        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.feature_selection import SelectFromModel

        estimator = ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split,
            max_features=int(X.shape[1] * self.max_features),
            bootstrap=self.bootstrap,
            max_leaf_nodes=self.max_leaf_nodes,
            max_depth=self.max_depth,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
        )
        estimator.fit(X, y)
        self.selector = SelectFromModel(
            estimator,
            threshold="mean",
            prefit=True,
        )

        self._fitted = True

        return self

    def transform(self, X):

        if not self._fitted:
            raise NotImplementedError("The model has not been fitted yet!")
        return self.selector.transform(X)


class extra_trees_preproc_for_regression:
    def __init__(
        self,
        n_estimators=5,
        criterion="mse",
        min_samples_leaf=5,
        min_samples_split=5,
        max_features=0.5,
        bootstrap=False,
        max_leaf_nodes=None,
        max_depth=None,
        min_weight_fraction_leaf=0.0,
    ):
        self.n_estimators = int(n_estimators)
        self.criterion = criterion
        self.min_samples_leaf = int(min_samples_leaf)
        self.min_samples_split = int(min_samples_split)
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_leaf_nodes = None if is_none(max_leaf_nodes) else int(max_leaf_nodes)
        self.max_depth = None if is_none(max_depth) else int(max_depth)
        self.min_weight_fraction_leaf = float(min_weight_fraction_leaf)

        self._fitted = False

    def fit(self, X, y=None):

        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.feature_selection import SelectFromModel

        estimator = ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split,
            max_features=int(np.log(X.shape[1] + 1) * self.max_features),
            bootstrap=self.bootstrap,
            max_leaf_nodes=self.max_leaf_nodes,
            max_depth=self.max_depth,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
        )
        estimator.fit(X, y)
        self.selector = SelectFromModel(
            estimator,
            threshold="mean",
            prefit=True,
        )

        self._fitted = True

        return self

    def transform(self, X):

        if not self._fitted:
            raise NotImplementedError("The model has not been fitted yet!")
        return self.selector.transform(X)


class fast_ica(sklearn.decomposition.FastICA):
    def __init__(
        self,
        algorithm="parallel",
        whiten=False,
        fun="logcosh",
        n_components=5,
    ):
        self.algorithm = algorithm
        self.whiten = whiten
        self.fun = fun
        self.n_components = None if is_none(n_components) else int(n_components)

        super().__init__(
            algorithm=self.algorithm,
            whiten=self.whiten,
            fun=self.fun,
            n_components=self.n_components,
        )

        self._fitted = False

    def fit(self, X, y=None):

        super().fit(X)

        self._fitted = True

        return self

    def transform(self, X):

        if not self._fitted:
            raise NotImplementedError("The model has not been fitted yet!")
        return super().transform(X)


class feature_agglomeration(sklearn.cluster.FeatureAgglomeration):
    def __init__(
        self,
        n_clusters=5,
        affinity="euclidean",
        linkage="ward",
        pooling_func="mean",
    ):
        self.n_clusters = int(n_clusters)
        self.affinity = affinity
        self.linkage = linkage
        self.pooling_func = pooling_func

        self.pooling_func_dict = {
            "mean": np.mean,
            "median": np.median,
            "max": np.max,
        }

        self._fitted = False

    def fit(self, X, y=None):

        if not callable(self.pooling_func):
            self.pooling_func = self.pooling_func_dict[self.pooling_func]

        super().__init__(
            n_clusters=min(self.n_clusters, X.shape[1]),
            affinity=self.affinity,
            linkage=self.linkage,
            pooling_func=self.pooling_func,
        )
        super().fit(X)

        self._fitted = True

        return self

    def transform(self, X):

        if not self._fitted:
            raise NotImplementedError("The model has not been fitted yet!")
        return super().transform(X)


class kernel_pca(sklearn.decomposition.KernelPCA):
    def __init__(
        self,
        n_components=5,
        kernel="rbf",
        gamma=0.1,
        degree=3,
        coef0=0.5,
    ):
        self.n_components = None if is_none(n_components) else int(n_components)
        self.kernel = kernel
        self.gamma = None if is_none(gamma) else float(gamma)
        self.degree = int(degree)
        self.coef0 = float(coef0)

        super().__init__(
            n_components=self.n_components,
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
        )

        self._fitted = False

    def fit(self, X, y=None):

        if scipy.sparse.issparse(X):
            X = X.stype(np.float64)

        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            super().fit(X)

        self._fitted = True

        return self

    def transform(self, X):

        if not self._fitted:
            raise NotImplementedError("The model has not been fitted yet!")

        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            return super().transform(X)


class kitchen_sinks(sklearn.kernel_approximation.RBFSampler):
    def __init__(
        self,
        gamma=0.1,
        n_components=50,
    ):
        self.gamma = float(gamma)
        self.n_components = int(n_components)

        super().__init__(
            gamma=self.gamma,
            n_components=self.n_components,
        )

        self._fitted = False

    def fit(self, X, y=None):

        super().fit(X)

        self._fitted = True

        return self

    def transform(self, X):

        if not self._fitted:
            raise NotImplementedError("The model has not been fitted yet!")

        return super().transform(X)


class liblinear_svc_preprocessor:
    def __init__(
        self,
        penalty="l1",
        loss="squared_hinge",
        dual=False,
        tol=0.0001,
        C=1.0,
        multi_class="ovr",
        fit_intercept=True,
        intercept_scaling=1,
    ):
        self.penalty = penalty
        self.loss = loss
        self.dual = dual
        self.tol = float(tol)
        self.C = float(C)
        self.multi_class = multi_class
        self.fit_intercept = fit_intercept
        self.intercept_scaling = float(intercept_scaling)

        self._fitted = False

    def fit(self, X, y=None):

        from sklearn.svm import LinearSVC
        from sklearn.feature_selection import SelectFromModel

        estimator = LinearSVC(
            penalty=self.penalty,
            loss=self.loss,
            dual=self.dual,
            tol=self.tol,
            C=self.C,
            multi_class=self.multi_class,
            fit_intercept=self.fit_intercept,
            intercept_scaling=self.intercept_scaling,
        )
        estimator.fit(X, y)

        self.selector = SelectFromModel(estimator, threshold="mean", prefit=True)

        self._fitted = True

        return self

    def transform(self, X):

        if not self._fitted:
            raise NotImplementedError("The model has not been fitted yet!")

        return self.selector.transform(X)


class nystroem_sampler(sklearn.kernel_approximation.Nystroem):
    def __init__(
        self,
        kernel="rbf",
        n_components=50,
        gamma=0.1,
        degree=3,
        coef0=0.5,
    ):
        self.kernel = kernel
        self.n_components = int(n_components)
        self.gamma = None if is_none(gamma) else float(gamma)
        self.degree = int(degree)
        self.coef0 = float(coef0)

        super().__init__(
            kernel=self.kernel,
            n_components=self.n_components,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
        )

        self._fitted = False

    def fit(self, X, y=None):

        # for kernel = "chi2", make sure non-negative values are passed
        if self.kernel == "chi2":
            if scipy.sparse.issparse(X):
                X.data[X.data < 0] = 0
            else:
                X[X < 0] = 0

        super().fit(X)

        self._fitted = True

        return self

    def transform(self, X):

        if not self._fitted:
            raise NotImplementedError("The model has not been fitted yet!")

        # for kernel = "chi2", make sure non-negative values are passed
        if self.kernel == "chi2":
            if scipy.sparse.issparse(X):
                X.data[X.data < 0] = 0
            else:
                X[X < 0] = 0

        return super().transform(X)


class pca(sklearn.decomposition.PCA):
    def __init__(
        self,
        keep_variance=0.5,
        whiten=True,
    ):
        self.keep_variance = float(keep_variance)
        self.with_whiten = whiten

        super().__init__(
            n_components=self.keep_variance,
            whiten=self.with_whiten,
        )

        self._fitted = False

    def fit(self, X, y=None):

        super().fit(X)

        self._fitted = True

        return self

    def transform(self, X):

        if not self._fitted:
            raise NotImplementedError("The model has not been fitted yet!")

        return super().transform(X)


class polynomial(sklearn.preprocessing.PolynomialFeatures):
    def __init__(
        self,
        degree=3,
        interaction_only=False,
        include_bias=True,
    ):
        self.degree = int(degree)
        self.interaction_only = interaction_only
        self.include_bias = include_bias

        super().__init__(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias,
        )

        self._fitted = False

    def fit(self, X, y=None):

        super().fit(X, y)

        self._fitted = True

        return self

    def transform(self, X):

        if not self._fitted:
            raise NotImplementedError("The model has not been fitted yet!")

        return super().transform(X)


class random_trees_embedding(sklearn.ensemble.RandomTreesEmbedding):
    def __init__(
        self,
        n_estimators=5,
        max_depth=3,
        min_samples_split=5,
        min_samples_leaf=5,
        min_weight_fraction_leaf=0.0,
        max_leaf_nodes=None,
        bootstrap=True,
    ):
        self.n_estimators = int(n_estimators)
        self.max_depth = None if is_none(max_depth) else int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.min_weight_fraction_leaf = float(min_weight_fraction_leaf)
        self.max_leaf_nodes = None if is_none(max_leaf_nodes) else int(max_leaf_nodes)
        self.bootstrap = bootstrap

        super().__init__(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_leaf_nodes=self.max_leaf_nodes,
            # bootstrap=self.bootstrap,
        )

        self._fitted = False

    def fit(self, X, y=None):

        super().fit(X, y)

        self._fitted = True

        return self

    def transform(self, X):

        if not self._fitted:
            raise NotImplementedError("The model has not been fitted yet!")

        return super().transform(X)


class select_percentile_classification(sklearn.feature_selection.SelectPercentile):
    def __init__(
        self,
        percentile=90,
        score_func="chi2",
    ):
        self.percentile = int(percentile)
        if score_func == "chi2":
            self.score_func = sklearn.feature_selection.chi2
        elif score_func == "f_classif":
            self.score_func = sklearn.feature_selection.f_classif
        elif score_func == "mutual_info":
            self.score_func = sklearn.feature_selection.mutual_info_classif
        else:
            raise ValueError(
                "score_func must be one of 'chi2', 'f_classif', 'mutual_info', but got {}".format(
                    score_func
                )
            )

        super().__init__(
            percentile=self.percentile,
            score_func=self.score_func,
        )

        self._fitted = False

    def fit(self, X, y=None):

        # for score_func = "chi2", make sure non-negative values are passed
        if self.score_func == sklearn.feature_selection.chi2:
            if scipy.sparse.issparse(X):
                X.data[X.data < 0] = 0
            else:
                X[X < 0] = 0

        super().fit(X, y)

        self._fitted = True

        return self

    def transform(self, X):

        if not self._fitted:
            raise NotImplementedError("The model has not been fitted yet!")

        # for score_func = "chi2", make sure non-negative values are passed
        if self.score_func == sklearn.feature_selection.chi2:
            if scipy.sparse.issparse(X):
                X.data[X.data < 0] = 0
            else:
                X[X < 0] = 0

        return super().transform(X)


class select_percentile_regression(sklearn.feature_selection.SelectPercentile):
    def __init__(
        self,
        percentile=90,
        score_func="f_regression",
    ):
        self.percentile = int(percentile)
        if score_func == "f_regression":
            self.score_func = sklearn.feature_selection.f_regression
        elif score_func == "mutual_info":
            self.score_func = sklearn.feature_selection.mutual_info_regression
        else:
            raise ValueError(
                "score_func must be one of 'f_regression', 'mutual_info', but got {}".format(
                    score_func
                )
            )

        super().__init__(
            percentile=self.percentile,
            score_func=self.score_func,
        )

        self._fitted = False

    def fit(self, X, y=None):

        super().fit(X, y)

        self._fitted = True

        return self

    def transform(self, X):

        if not self._fitted:
            raise NotImplementedError("The model has not been fitted yet!")

        return super().transform(X)


class select_rates_classification(sklearn.feature_selection.GenericUnivariateSelect):
    def __init__(
        self,
        alpha=0.3,
        score_func="chi2",
        mode="fpr",
    ):
        self.alpha = float(alpha)
        if score_func == "chi2":
            self.score_func = sklearn.feature_selection.chi2
        elif score_func == "f_classif":
            self.score_func = sklearn.feature_selection.f_classif
        elif score_func == "mutual_info":
            self.score_func = sklearn.feature_selection.mutual_info_classif
        else:
            raise ValueError(
                "score_func must be one of 'chi2', 'f_classif', 'mutual_info', but got {}".format(
                    score_func
                )
            )
        self.mode = mode

        super().__init__(
            param=self.alpha,
            score_func=self.score_func,
            mode=self.mode,
        )

        self._fitted = False

    def fit(self, X, y=None):

        # for score_func = "chi2", make sure non-negative values are passed
        if self.score_func == sklearn.feature_selection.chi2:
            if scipy.sparse.issparse(X):
                X.data[X.data < 0] = 0
            else:
                X[X < 0] = 0

        super().fit(X, y)

        self._fitted = True

        return self

    def transform(self, X):

        if not self._fitted:
            raise NotImplementedError("The model has not been fitted yet!")

        # for score_func = "chi2", make sure non-negative values are passed
        if self.score_func == sklearn.feature_selection.chi2:
            if scipy.sparse.issparse(X):
                X.data[X.data < 0] = 0
            else:
                X[X < 0] = 0

        return super().transform(X)


class select_rates_regression(sklearn.feature_selection.GenericUnivariateSelect):
    def __init__(
        self,
        alpha=0.3,
        score_func="f_regression",
        mode="fpr",
    ):
        self.alpha = float(alpha)
        if score_func == "f_regression":
            self.score_func = sklearn.feature_selection.f_regression
        elif score_func == "mutual_info":
            self.score_func = sklearn.feature_selection.mutual_info_regression
        else:
            raise ValueError(
                "score_func must be one of 'f_regression', 'mutual_info', but got {}".format(
                    score_func
                )
            )
        self.mode = mode

        super().__init__(
            param=self.alpha,
            score_func=self.score_func,
            mode=self.mode,
        )

        self._fitted = False

    def fit(self, X, y=None):

        super().fit(X, y)

        self._fitted = True

        return self

    def transform(self, X):

        if not self._fitted:
            raise NotImplementedError("The model has not been fitted yet!")

        return super().transform(X)


class truncatedSVD(sklearn.decomposition.TruncatedSVD):
    def __init__(
        self,
        target_dim=5,
    ):
        self.target_dim = int(target_dim)

        self._fitted = False

    def fit(self, X, y=None):

        super().__init__(
            n_components=min(self.target_dim, X.shape[1] - 1),
            algorithm="randomized",
        )

        super().fit(X)

        self._fitted = True

        return self

    def transform(self, X):

        if not self._fitted:
            raise NotImplementedError("The model has not been fitted yet!")

        return super().transform(X)
