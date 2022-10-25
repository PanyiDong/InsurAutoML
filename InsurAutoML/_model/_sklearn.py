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
Last Modified: Monday, 24th October 2022 10:51:34 pm
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

import numpy as np
import scipy
import sklearn
import sklearn.linear_model
import sklearn.neighbors
import sklearn.tree
import sklearn.svm
import sklearn.naive_bayes
import sklearn.discriminant_analysis
import sklearn.ensemble
import sklearn.gaussian_process
import sklearn.neural_network
import sklearn.calibration

# need to enable hist gradient boosting features first
# no need for sklearn version >= 1.0.0
sklearn_1_0_0 = sklearn.__version__ < "1.0.0"
if sklearn_1_0_0:
    from sklearn.experimental import enable_hist_gradient_boosting

from InsurAutoML._constant import MAX_ITER
from InsurAutoML._utils._base import is_none
from InsurAutoML._utils._data import softmax

####################################################################################################################
# models from sklearn
# wrap for some-degree of flexibility (initialization, _fitted, etc.)

####################################################################################################################
# classifiers


class AdaboostClassifier(sklearn.ensemble.AdaBoostClassifier):
    def __init__(
        self,
        n_estimators=50,
        learning_rate=0.1,
        algorithm="SAMME.R",
        max_depth=1,
    ):
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.algorithm = algorithm
        self.max_depth = int(max_depth)

        from sklearn.tree import DecisionTreeClassifier

        super().__init__(
            base_estimator=DecisionTreeClassifier(max_depth=self.max_depth),
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            algorithm=self.algorithm,
        )

        self._fitted = False  # whether the model is fitted

    def fit(self, X, y):

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(self, X):

        return super().predict(X)

    def predict_proba(self, X):

        return super().predict_proba(X)


class BernoulliNB(sklearn.naive_bayes.BernoulliNB):
    def __init__(
        self,
        alpha=1,
        fit_prior=True,
    ):
        self.alpha = float(alpha)
        self.fit_prior = fit_prior

        super().__init__(
            alpha=self.alpha,
            fit_prior=self.fit_prior,
        )

        self._fitted = False  # whether the model is fitted

    def fit(self, X, y):

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(self, X):

        return super().predict(X)

    def predict_proba(self, X):

        return super().predict_proba(X)


class DecisionTreeClassifier(sklearn.tree.DecisionTreeClassifier):
    def __init__(
        self,
        criterion="gini",
        max_depth_factor=0.5,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes="None",
        min_impurity_decrease=0.0,
    ):
        self.criterion = criterion
        self.max_depth_factor = max_depth_factor
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.min_weight_fraction_leaf = float(min_weight_fraction_leaf)
        self.max_features = float(max_features)
        self.max_leaf_nodes = None if is_none(max_leaf_nodes) else max_leaf_nodes
        self.min_impurity_decrease = float(min_impurity_decrease)

        self._fitted = False  # whether the model is fitted

    def fit(self, X, y):

        super().__init__(
            criterion=self.criterion,
            max_depth=None
            if is_none(self.max_depth_factor)
            else max(int(self.max_depth_factor * X.shape[1]), 1),
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
        )

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(self, X):

        return super().predict(X)

    def predict_proba(self, X):

        return super().predict_proba(X)


class ExtraTreesClassifier:
    def __init__(
        self,
        criterion="gini",
        max_depth="None",
        max_leaf_nodes="None",
        min_samples_leaf=1,
        min_samples_split=2,
        max_features=0.5,
        bootstrap=False,
        min_weight_fraction_leaf=0.0,
        min_impurity_decrease=0.0,
    ):
        self.criterion = criterion
        self.max_depth = None if is_none(max_depth) else int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.min_weight_fraction_leaf = float(min_weight_fraction_leaf)
        self.max_features = float(max_features)
        self.bootstrap = bootstrap
        self.max_leaf_nodes = None if is_none(max_leaf_nodes) else max_leaf_nodes
        self.min_impurity_decrease = float(min_impurity_decrease)

        self.estimator = None  # the fitted estimator
        self.max_iter = self._get_max_iter()  # limit the number of iterations

        self._fitted = False  # whether the model is fitted

    @staticmethod
    def _get_max_iter():  # define global max_iter

        return MAX_ITER

    def _fit_iteration(self, X, y, n_iter=2):

        if self.estimator is None:
            from sklearn.ensemble import ExtraTreesClassifier

            self.estimator = ExtraTreesClassifier(
                n_estimators=n_iter,
                criterion=self.criterion,
                max_depth=self.max_depth,
                max_leaf_nodes=self.max_leaf_nodes,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                max_features=max(1, int(X.shape[1] ** self.max_features)),
                bootstrap=self.bootstrap,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
            )
        else:
            self.estimator.n_estimators += n_iter
            self.estimator.n_estimators = min(
                self.estimator.n_estimators, self.max_iter
            )

        self.estimator.fit(X, y)

        if (
            self.estimator.n_estimators >= self.max_iter
            or self.estimator.n_estimators >= len(self.estimator.estimators_)
        ):
            self._fitted = True

        return self

    def fit(self, X, y):

        self._fit_iteration(X, y, n_iter=2)

        # accelerate iteration process
        iteration = 2
        while not self._fitted:
            n_iter = int(2**iteration / 2)
            self._fit_iteration(X, y, n_iter=n_iter)
            iteration += 1

        return self

    def predict(self, X):

        return self.estimator.predict(X)

    def predict_proba(self, X):

        return self.estimator.predict_proba(X)


class GaussianNB(sklearn.naive_bayes.GaussianNB):
    def __init__(
        self,
    ):
        super().__init__()

        self._fitted = False  # whether the model is fitted

    def fit(self, X, y):

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(self, X):

        return super().predict(X)

    def predict_proba(self, X):

        return super().predict_proba(X)


class HistGradientBoostingClassifier:
    def __init__(
        self,
        loss="auto",
        learning_rate=0.1,
        min_samples_leaf=20,
        max_depth="None",
        max_leaf_nodes=31,
        max_bins=255,
        l2_regularization=1e-10,
        early_stop="off",
        tol=1e-7,
        scoring="loss",
        n_iter_no_change=10,
        validation_fraction=0.1,
    ):
        self.loss = loss
        self.learning_rate = float(learning_rate)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_depth = None if is_none(max_depth) else int(max_depth)
        self.max_leaf_nodes = int(max_leaf_nodes)
        self.max_bins = int(max_bins)
        self.l2_regularization = float(l2_regularization)
        self.early_stop = early_stop
        self.tol = float(tol)
        self.scoring = scoring
        self.n_iter_no_change = int(n_iter_no_change)
        self.validation_fraction = float(validation_fraction)

        self.estimator = None  # the fitted estimator
        self.max_iter = self._get_max_iter()  # limit the number of iterations

        self._fitted = False  # whether the model is fitted

    @staticmethod
    def _get_max_iter():  # define global max_iter

        return MAX_ITER

    def _fit_iteration(self, X, y, n_iter=2, sample_weight=None):

        if self.estimator is None:
            from sklearn.ensemble import HistGradientBoostingClassifier

            # map from autosklearn parameter space to sklearn parameter space
            if self.early_stop == "off":
                self.n_iter_no_change = 0
                self.validation_fraction_ = None
                self.early_stopping_ = False
            elif self.early_stop == "train":
                self.n_iter_no_change = int(self.n_iter_no_change)
                self.validation_fraction_ = None
                self.early_stopping_ = True
            elif self.early_stop == "valid":
                self.n_iter_no_change = int(self.n_iter_no_change)
                self.validation_fraction = float(self.validation_fraction)
                self.early_stopping_ = True
                n_classes = len(np.unique(y))
                if self.validation_fraction * X.shape[0] < n_classes:
                    self.validation_fraction_ = n_classes
                else:
                    self.validation_fraction_ = self.validation_fraction
            else:
                raise ValueError("early_stop should be either off, train or valid")

            self.estimator = HistGradientBoostingClassifier(
                loss=self.loss,
                learning_rate=self.learning_rate,
                min_samples_leaf=self.min_samples_leaf,
                max_depth=self.max_depth,
                max_leaf_nodes=self.max_leaf_nodes,
                max_bins=self.max_bins,
                l2_regularization=self.l2_regularization,
                early_stopping=self.early_stopping_,
                tol=self.tol,
                scoring=self.scoring,
                n_iter_no_change=self.n_iter_no_change,
                validation_fraction=self.validation_fraction,
                max_iter=n_iter,
            )
        else:
            self.estimator.max_iter += n_iter  # add n_iter to each step
            self.estimator.max_iter = min(
                self.estimator.max_iter, self.max_iter
            )  # limit the number of iterations

        self.estimator.fit(X, y, sample_weight=sample_weight)

        # check whether fully fitted or need to add more iterations
        if (
            self.estimator.max_iter >= self.max_iter
            or self.estimator.max_iter > self.estimator.n_iter_
        ):

            self._fitted = True

        return self

    def fit(self, X, y):

        self._fit_iteration(X, y, n_iter=2)

        # accelerate iteration process
        iteration = 2
        while not self._fitted:
            n_iter = int(2**iteration / 2)
            self._fit_iteration(X, y, n_iter=n_iter)
            iteration += 1

        return self

    def predict(self, X):

        return self.estimator.predict(X)

    def predict_proba(self, X):

        return self.estimator.predict_proba(X)


class KNearestNeighborsClassifier(sklearn.neighbors.KNeighborsClassifier):
    def __init__(
        self,
        n_neighbors=5,
        weights="uniform",
        p=2,
    ):
        self.n_neighbors = int(n_neighbors)
        self.weights = weights
        self.p = int(p)

        super().__init__(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            p=self.p,
        )

        self._fitted = False  # whether the model is fitted

    def fit(self, X, y):

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(self, X):

        return super().predict(X)

    def predict_proba(self, X):

        return super().predict_proba(X)


class LDA(sklearn.discriminant_analysis.LinearDiscriminantAnalysis):
    def __init__(
        self,
        shrinkage="auto",
        shrinkage_factor=0.5,
        tol=1e-4,
    ):
        self.shrinkage = shrinkage
        self.shrinkage_factor = float(shrinkage_factor)
        self.tol = float(tol)

        self._fitted = False  # whether the model is fitted

    def fit(self, X, y):

        if self.shrinkage is None or self.shrinkage == "None":
            self.shrinkage = None
            solver = "svd"
        elif self.shrinkage == "auto":
            self.shrinkage = "auto"
            solver = "lsqr"
        elif self.shrinkage == "manual":
            self.shrinkage = float(self.shrinkage_factor)
            solver = "lsqr"
        else:
            raise ValueError(
                "Not a valid shrinkage parameter, should be None, auto or manual"
            )

        super().__init__(
            shrinkage=self.shrinkage,
            solver=solver,
            tol=self.tol,
        )

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(self, X):
        return super().predict(X)

    def predict_proba(self, X):

        return super().predict_proba(X)


class LibLinear_SVC:
    def __init__(
        self,
        penalty="l2",
        loss="squared_hinge",
        dual=False,
        tol=1e-4,
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

        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV

        base_estimator = LinearSVC(
            penalty=self.penalty,
            loss=self.loss,
            dual=self.dual,
            tol=self.tol,
            C=self.C,
            multi_class=self.multi_class,
            fit_intercept=self.fit_intercept,
            intercept_scaling=self.intercept_scaling,
        )

        # wrap the base estimator to make predict_proba available
        self.estimator = CalibratedClassifierCV(base_estimator)

        self._fitted = False  # whether the model is fitted

    def fit(self, X, y):

        self.estimator.fit(X, y)

        self._fitted = True

        return self

    def predict(self, X):

        return self.estimator.predict(X)

    def predict_proba(self, X):

        return self.estimator.predict_proba(X)


class LibSVM_SVC:
    def __init__(
        self,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma=0.1,
        coef0=0,
        tol=1e-3,
        shrinking=True,
        max_iter=-1,
    ):
        self.C = float(C)
        self.kernel = kernel
        self.degree = 3 if is_none(degree) else int(degree)
        self.gamma = 0.0 if is_none(gamma) else float(gamma)
        self.coef0 = 0.0 if is_none(coef0) else float(coef0)
        self.shrinking = shrinking
        self.tol = float(tol)
        self.max_iter = float(max_iter)

        from sklearn.svm import SVC
        from sklearn.calibration import CalibratedClassifierCV

        base_estimator = SVC(
            C=self.C,
            kernel=self.kernel,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            shrinking=self.shrinking,
            tol=self.tol,
            max_iter=self.max_iter,
        )

        # wrap the base estimator to make predict_proba available
        self.estimator = CalibratedClassifierCV(base_estimator)

        self._fitted = False  # whether the model is fitted

    def fit(self, X, y):

        self.estimator.fit(X, y)

        self._fitted = True

        return self

    def predict(self, X):

        return self.estimator.predict(X)

    def predict_proba(self, X):

        return self.estimator.predict_proba(X)


class MLPClassifier:
    def __init__(
        self,
        hidden_layer_depth=1,
        num_nodes_per_layer=32,
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        early_stopping="valid",
        n_iter_no_change=32,
        validation_fraction=0.1,
        tol=1e-4,
        solver="adam",
        batch_size="auto",
        shuffle=True,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
    ):
        self.hidden_layer_depth = int(hidden_layer_depth)
        self.num_nodes_per_layer = int(num_nodes_per_layer)
        self.hidden_layer_sizes = tuple(
            self.num_nodes_per_layer for _ in range(self.hidden_layer_depth)
        )
        self.activation = str(activation)
        self.alpha = float(alpha)
        self.learning_rate_init = float(learning_rate_init)
        self.early_stopping = str(early_stopping)
        self.n_iter_no_change = int(n_iter_no_change)
        self.validation_fraction = float(validation_fraction)
        self.tol = float(tol)
        self.solver = solver
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.beta_1 = float(beta_1)
        self.beta_2 = float(beta_2)
        self.epsilon = float(epsilon)

        self.estimator = None  # the fitted estimator
        self.max_iter = self._get_max_iter()  # limit the number of iterations

        self._fitted = False  # whether the model is fitted

    @staticmethod
    def _get_max_iter():  # define global max_iter

        return MAX_ITER

    def _fit_iteration(self, X, y, n_iter=2):

        if self.estimator is None:
            from sklearn.neural_network import MLPClassifier

            # map from autosklearn parameter space to sklearn parameter space
            if self.early_stopping == "train":
                self.validation_fraction = 0.0
                self.tol = float(self.tol)
                self.n_iter_no_change = int(self.n_iter_no_change)
                self.early_stopping = False
            elif self.early_stopping == "valid":
                self.validation_fraction = float(self.validation_fraction)
                self.tol = float(self.tol)
                self.n_iter_no_change = int(self.n_iter_no_change)
                self.early_stopping = True
            else:
                raise ValueError(
                    "Early stopping only supports 'train' and 'valid'. Got {}".format(
                        self.early_stopping
                    )
                )

            self.estimator = MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                alpha=self.alpha,
                learning_rate_init=self.learning_rate_init,
                early_stopping=self.early_stopping,
                n_iter_no_change=self.n_iter_no_change,
                validation_fraction=self.validation_fraction,
                tol=self.tol,
                solver=self.solver,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                beta_1=self.beta_1,
                beta_2=self.beta_2,
                epsilon=self.epsilon,
                max_iter=n_iter,
            )
        else:
            # MLPClassifier can record previous training
            self.estimator.max_iter = min(
                self.estimator.max_iter - self.estimator.n_iter_, self.max_iter
            )  # limit the number of iterations

        self.estimator.fit(X, y)

        # check whether fully fitted or need to add more iterations
        if (
            self.estimator.n_iter_ >= self.estimator.max_iter
            or self.estimator._no_improvement_count > self.n_iter_no_change
        ):

            self._fitted = True

        return self

    def fit(self, X, y):

        self._fit_iteration(X, y, n_iter=2)

        # accelerate iteration process
        iteration = 2
        while not self._fitted:
            n_iter = int(2**iteration / 2)
            self._fit_iteration(X, y, n_iter=n_iter)
            iteration += 1

        return self

    def predict(self, X):

        return self.estimator.predict(X)

    def predict_proba(self, X):

        return self.estimator.predict_proba(X)


class MultinomialNB(sklearn.naive_bayes.MultinomialNB):
    def __init__(self, alpha=1, fit_prior=True):
        self.alpha = float(alpha)
        self.fit_prior = fit_prior

        super().__init__(
            alpha=self.alpha,
            fit_prior=self.fit_prior,
        )

        self._fitted = False  # whether the model is fitted

    def fit(self, X, y):

        # make sure the data contains only non-negative values
        if scipy.sparse.issparse(X):
            X.data[X.data < 0] = 0.0
        else:
            X[X < 0] = 0.0

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(self, X):

        return super().predict(X)

    def predict_proba(self, X):

        return super().predict_proba(X)


class PassiveAggressive:
    def __init__(
        self,
        C=1.0,
        fit_intercept=True,
        average=False,
        tol=1e-4,
        loss="hinge",
    ):
        self.C = float(C)
        self.fit_intercept = fit_intercept
        self.average = average
        self.tol = float(tol)
        self.loss = loss

        self.estimator = None  # the fitted estimator
        self.max_iter = self._get_max_iter()  # limit the number of iterations

        self._fitted = False  # whether the model is fitted

    @staticmethod
    def _get_max_iter():  # define global max_iter

        return MAX_ITER

    def _fit_iteration(self, X, y, n_iter=2):

        if self.estimator is None:
            from sklearn.linear_model import PassiveAggressiveClassifier

            self.estimator = PassiveAggressiveClassifier(
                C=self.C,
                fit_intercept=self.fit_intercept,
                average=self.average,
                tol=self.tol,
                loss=self.loss,
                max_iter=n_iter,
            )
        else:
            self.estimator.max_iter += n_iter
            self.estimator.max_iter = min(self.estimator.max_iter, self.max_iter)

        self.estimator.fit(X, y)

        # check whether fully fitted or need to add more iterations
        if (
            self.estimator.max_iter >= self.max_iter
            or self.estimator.max_iter > self.estimator.n_iter_
        ):

            self._fitted = True

        return self

    def fit(self, X, y):

        self._fit_iteration(X, y, n_iter=2)

        # accelerate iteration process
        iteration = 2
        while not self._fitted:
            n_iter = int(2**iteration / 2)
            self._fit_iteration(X, y, n_iter=n_iter)
            iteration += 1

        return self

    def predict(self, X):

        return self.estimator.predict(X)

    def predict_proba(self, X):

        return softmax(self.estimator.decision_function(X))


class QDA(sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis):
    def __init__(
        self,
        reg_param=0.0,
    ):
        self.reg_param = float(reg_param)

        super().__init__(
            reg_param=self.reg_param,
        )

        self._fitted = False  # whether the model is fitted

    def fit(self, X, y):

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(self, X):
        return super().predict(X)

    def predict_proba(self, X):

        return super().predict_proba(X)


class RandomForestClassifier:
    def __init__(
        self,
        criterion="gini",
        max_features=0.5,
        max_depth="None",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        bootstrap=True,
        max_leaf_nodes="None",
        min_impurity_decrease=0.0,
    ):
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = None if is_none(max_depth) else int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.min_weight_fraction_leaf = float(min_weight_fraction_leaf)
        self.bootstrap = bootstrap
        self.max_leaf_nodes = None if is_none(max_leaf_nodes) else int(max_leaf_nodes)
        self.min_impurity_decrease = float(min_impurity_decrease)

        self.estimator = None  # the fitted estimator
        self.max_iter = self._get_max_iter()  # limit the number of iterations

        self._fitted = False  # whether the model is fitted

    @staticmethod
    def _get_max_iter():  # define global max_iter

        return MAX_ITER

    def _fit_iteration(self, X, y, n_iter=2):

        if self.estimator is None:
            from sklearn.ensemble import RandomForestClassifier

            self.estimator = RandomForestClassifier(
                n_estimators=n_iter,
                criterion=self.criterion,
                max_features=self.max_features,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                bootstrap=self.bootstrap,
                max_leaf_nodes=self.max_leaf_nodes,
                min_impurity_decrease=self.min_impurity_decrease,
            )
        else:
            self.estimator.n_estimators += n_iter
            self.estimator.n_estimators = min(
                self.estimator.n_estimators, self.max_iter
            )

        self.estimator.fit(X, y)

        if (
            self.estimator.n_estimators >= self.max_iter
            or self.estimator.n_estimators > len(self.estimator.estimators_)
        ):
            self._fitted = True

        return self

    def fit(self, X, y):

        self._fit_iteration(X, y, n_iter=2)

        # accelerate iteration process
        iteration = 2
        while not self._fitted:
            n_iter = int(2**iteration / 2)
            self._fit_iteration(X, y, n_iter=n_iter)
            iteration += 1

        return self

    def predict(self, X):

        return self.estimator.predict(X)

    def predict_proba(self, X):

        return self.estimator.predict_proba(X)


class SGDClassifier:
    def __init__(
        self,
        loss="log",
        penalty="l2",
        alpha=0.0001,
        fit_intercept=True,
        tol=1e-4,
        learning_rate="invscaling",
        l1_ratio=0.15,
        epsilon=1e-4,
        eta0=0.01,
        power_t=0.5,
        average=False,
    ):
        self.loss = loss
        self.penalty = penalty
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept
        self.tol = float(tol)
        self.learning_rate = learning_rate
        self.l1_ratio = 0.15 if is_none(l1_ratio) else float(l1_ratio)
        self.epsilon = 0.1 if is_none(epsilon) else float(epsilon)
        self.eta0 = float(eta0)
        self.power_t = 0.5 if is_none(power_t) else float(power_t)
        self.average = average

        self.estimator = None  # the fitted estimator
        self.max_iter = self._get_max_iter()  # limit the number of iterations

        self._fitted = False  # whether the model is fitted

    @staticmethod
    def _get_max_iter():  # define global max_iter

        return MAX_ITER

    def _fit_iteration(self, X, y, n_iter=2):

        if self.estimator is None:
            from sklearn.linear_model import SGDClassifier

            self.estimator = SGDClassifier(
                loss=self.loss,
                penalty=self.penalty,
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                tol=self.tol,
                learning_rate=self.learning_rate,
                l1_ratio=self.l1_ratio,
                epsilon=self.epsilon,
                eta0=self.eta0,
                power_t=self.power_t,
                average=self.average,
                max_iter=n_iter,
            )
        else:
            self.estimator.max_iter += n_iter
            self.estimator.max_iter = min(self.estimator.max_iter, self.max_iter)

        self.estimator.fit(X, y)

        if (
            self.estimator.max_iter >= self.max_iter
            or self.estimator.max_iter > self.estimator.n_iter_
        ):
            self._fitted = True

        return self

    def fit(self, X, y):

        self._fit_iteration(X, y, n_iter=2)

        # accelerate iteration process
        iteration = 2
        while not self._fitted:
            n_iter = int(2**iteration / 2)
            self._fit_iteration(X, y, n_iter=n_iter)
            iteration += 1

        return self

    def predict(self, X):

        return self.estimator.predict(X)

    def predict_proba(self, X):

        return self.estimator.predict_proba(X)


class LogisticRegression(sklearn.linear_model.LogisticRegression):
    def __init__(
        self,
        penalty="l2",
        tol=1e-4,
        C=1.0,
    ):
        self.penalty = penalty
        self.tol = float(tol)
        self.C = float(C)

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

    def predict_proba(self, X):

        return super().predict_proba(X)


class ComplementNB(sklearn.naive_bayes.ComplementNB):
    def __init__(
        self,
        alpha=1.0,
        fit_prior=True,
        norm=False,
    ):
        self.alpha = float(alpha)
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

    def predict_proba(self, X):

        return super().predict_proba(X)


# combined with autosklearn version, thus deprecated here
# class HistGradientBoostingClassifier(sklearn.ensemble.HistGradientBoostingClassifier):
#     def __init__(
#         self,
#         loss="auto",
#         learning_rate=0.1,
#         max_leaf_nodes=31,
#         max_depth=None,
#         min_samples_leaf=20,
#         l2_regularization=0,
#         tol=1e-7,
#     ):
#         self.loss = loss
#         self.learning_rate = learning_rate
#         self.max_leaf_nodes = max_leaf_nodes
#         self.max_depth = max_depth
#         self.min_samples_leaf = min_samples_leaf
#         self.l2_regularization = l2_regularization
#         self.tol = tol

#         super().__init__(
#             loss=self.loss,
#             learning_rate=self.learning_rate,
#             max_leaf_nodes=self.max_leaf_nodes,
#             max_depth=self.max_depth,
#             min_samples_leaf=self.min_samples_leaf,
#             l2_regularization=self.l2_regularization,
#             tol=self.tol,
#         )

#         self._fitted = False

#     def fit(self, X, y):

#         super().fit(X, y)

#         self._fitted = True

#         return self

#     def predict(self, X):

#         return super().predict(X)


class GradientBoostingClassifier(sklearn.ensemble.GradientBoostingClassifier):
    def __init__(
        self,
        loss="deviance",
        learning_rate=0.1,
        n_estimators=100,
        subsample=1.0,
        criterion="friedman_mse",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        max_features="auto",
        max_leaf_nodes=31,
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-7,
    ):
        self.loss = loss
        self.learning_rate = float(learning_rate)
        self.n_estimators = int(n_estimators)
        self.subsample = float(subsample)
        self.criterion = criterion
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.min_weight_fraction_leaf = float(min_weight_fraction_leaf)
        self.max_depth = None if is_none(max_depth) else int(max_depth)
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.max_features = max_features
        self.max_leaf_nodes = int(max_leaf_nodes)
        self.validation_fraction = float(validation_fraction)
        self.n_iter_no_change = int(n_iter_no_change)
        self.tol = float(tol)

        super().__init__(
            loss=self.loss,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            criterion=self.criterion,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_depth=self.max_depth,
            min_impurity_decrease=self.min_impurity_decrease,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            tol=self.tol,
        )

        self._fitted = False  # whether the model is fitted

    def fit(self, X, y):

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(self, X):

        return super().predict(X)

    def predict_proba(self, X):

        return super().predict_proba(X)


####################################################################################################################
# regressors


class AdaboostRegressor(sklearn.ensemble.AdaBoostRegressor):
    def __init__(
        self,
        n_estimators=50,
        learning_rate=0.1,
        loss="linear",
        max_depth=1,
    ):
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.loss = loss
        self.max_depth = int(max_depth)

        from sklearn.tree import DecisionTreeRegressor

        super().__init__(
            base_estimator=DecisionTreeRegressor(max_depth=self.max_depth),
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            loss=self.loss,
        )

        self._fitted = False  # whether the model is fitted

    def fit(self, X, y):

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(self, X):

        return super().predict(X)

    def predict_proba(self, X):

        raise NotImplementedError("predict_proba is not implemented for regression.")


class ARDRegression(sklearn.linear_model.ARDRegression):
    def __init__(
        self,
        n_iter=300,
        tol=1e-3,
        alpha_1=1e-6,
        alpha_2=1e-6,
        lambda_1=1e-6,
        lambda_2=1e-6,
        threshold_lambda=1e4,
        fit_intercept=True,
    ):
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.alpha_1 = float(alpha_1)
        self.alpha_2 = float(alpha_2)
        self.lambda_1 = float(lambda_1)
        self.lambda_2 = float(lambda_2)
        self.threshold_lambda = float(threshold_lambda)
        self.fit_intercept = fit_intercept

        super().__init__(
            n_iter=self.n_iter,
            tol=self.tol,
            alpha_1=self.alpha_1,
            alpha_2=self.alpha_2,
            lambda_1=self.lambda_1,
            lambda_2=self.lambda_2,
            threshold_lambda=self.threshold_lambda,
            fit_intercept=self.fit_intercept,
        )

        self._fitted = False  # whether the model is fitted

    def fit(self, X, y):

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(self, X):

        return super().predict(X)

    def predict_proba(self, X):

        raise NotImplementedError("predict_proba is not implemented for regression.")


class DecisionTreeRegressor(sklearn.tree.DecisionTreeRegressor):
    def __init__(
        self,
        criterion="mse",
        max_features=1.0,
        max_depth_factor=0.5,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_leaf_nodes="None",
        min_impurity_decrease=0.0,
    ):
        self.criterion = criterion
        self.max_depth_factor = max_depth_factor
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.min_weight_fraction_leaf = float(min_weight_fraction_leaf)
        self.max_features = float(max_features)
        self.max_leaf_nodes = None if is_none(max_leaf_nodes) else int(max_leaf_nodes)
        self.min_impurity_decrease = float(min_impurity_decrease)

        self._fitted = False  # whether the model is fitted

    def fit(self, X, y):

        super().__init__(
            criterion=self.criterion,
            max_depth=max(int(self.max_depth_factor * X.shape[1]), 1),
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
        )

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(self, X):

        return super().predict(X)

    def predict_proba(self, X):

        raise NotImplementedError("predict_proba is not implemented for regression.")


class ExtraTreesRegressor:
    def __init__(
        self,
        criterion="mse",
        max_depth="None",
        max_leaf_nodes="None",
        min_samples_leaf=1,
        min_samples_split=2,
        max_features=1,
        bootstrap=False,
        min_weight_fraction_leaf=0.0,
        min_impurity_decrease=0.0,
    ):
        self.criterion = criterion
        self.max_depth = None if is_none(max_depth) else int(max_depth)
        self.max_leaf_nodes = None if is_none(max_leaf_nodes) else int(max_leaf_nodes)
        self.min_samples_leaf = int(min_samples_leaf)
        self.min_samples_split = int(min_samples_split)
        self.max_features = float(max_features)
        self.bootstrap = bootstrap
        self.min_weight_fraction_leaf = float(min_weight_fraction_leaf)
        self.min_impurity_decrease = float(min_impurity_decrease)

        self.estimator = None  # the fitted estimator
        self.max_iter = self._get_max_iter()  # limit the number of iterations

        self._fitted = False  # whether the model is fitted

    @staticmethod
    def _get_max_iter():  # define global max_iter

        return MAX_ITER

    def _fit_iteration(self, X, y, n_iter=2):

        if self.estimator is None:
            from sklearn.ensemble import ExtraTreesRegressor

            self.estimator = ExtraTreesRegressor(
                n_estimators=n_iter,
                criterion=self.criterion,
                max_depth=self.max_depth,
                max_leaf_nodes=self.max_leaf_nodes,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
            )
        else:
            self.estimator.n_estimators += n_iter
            self.estimator.n_estimators = min(
                self.estimator.n_estimators, self.max_iter
            )

        self.estimator.fit(X, y)

        if (
            self.estimator.n_estimators >= self.max_iter
            or self.estimator.n_estimators >= len(self.estimator.estimators_)
        ):
            self._fitted = True

        return self

    def fit(self, X, y):

        self._fit_iteration(X, y, n_iter=2)

        # accelerate iteration process
        iteration = 2
        while not self._fitted:
            n_iter = int(2**iteration / 2)
            self._fit_iteration(X, y, n_iter=n_iter)
            iteration += 1

        return self

    def predict(self, X):

        return self.estimator.predict(X)

    def predict_proba(self, X):

        raise NotImplementedError("predict_proba is not implemented for regression.")


class GaussianProcess(sklearn.gaussian_process.GaussianProcessRegressor):
    def __init__(
        self,
        alpha=1e-8,
        thetaL=1e-6,
        thetaU=100000.0,
    ):
        self.alpha = float(alpha)
        self.thetaL = float(thetaL)
        self.thetaU = float(thetaU)

        self._fitted = False

    def fit(self, X, y):

        n_features = X.shape[1]
        kernel = sklearn.gaussian_process.kernels.RBF(
            length_scale=[1.0] * n_features,
            length_scale_bounds=[(self.thetaL, self.thetaU)] * n_features,
        )

        super().__init__(
            kernel=kernel,
            n_restarts_optimizer=10,
            optimizer="fmin_l_bfgs_b",
            alpha=self.alpha,
            copy_X_train=True,
            normalize_y=True,
        )

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(self, X):

        return super().predict(X)

    def predict_proba(self, X):

        raise NotImplementedError("predict_proba is not implemented for regression.")


class HistGradientBoostingRegressor:
    def __init__(
        self,
        loss="least_squares",
        learning_rate=0.1,
        min_samples_leaf=20,
        max_depth="None",
        max_leaf_nodes=31,
        max_bins=255,
        l2_regularization=1e-10,
        early_stop="off",
        tol=1e-7,
        scoring="loss",
        n_iter_no_change=10,
        validation_fraction=0.1,
    ):
        self.loss = loss
        self.learning_rate = float(learning_rate)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_depth = None if is_none(max_depth) else int(max_depth)
        self.max_leaf_nodes = int(max_leaf_nodes)
        self.max_bins = int(max_bins)
        self.l2_regularization = float(l2_regularization)
        self.early_stop = early_stop
        self.tol = float(tol)
        self.scoring = scoring
        self.n_iter_no_change = int(n_iter_no_change)
        self.validation_fraction = float(validation_fraction)

        self.estimator = None  # the fitted estimator
        self.max_iter = self._get_max_iter()  # limit the number of iterations

        self._fitted = False  # whether the model is fitted

    @staticmethod
    def _get_max_iter():  # define global max_iter

        return MAX_ITER

    def _fit_iteration(self, X, y, n_iter=2, sample_weight=None):

        if self.estimator is None:
            from sklearn.ensemble import HistGradientBoostingRegressor

            # map from autosklearn parameter space to sklearn parameter space
            if self.early_stop == "off":
                self.n_iter_no_change = 0
                self.validation_fraction_ = None
                self.early_stopping_ = False
            elif self.early_stop == "train":
                self.n_iter_no_change = int(self.n_iter_no_change)
                self.validation_fraction_ = None
                self.early_stopping_ = True
            elif self.early_stop == "valid":
                self.n_iter_no_change = int(self.n_iter_no_change)
                self.validation_fraction = float(self.validation_fraction)
                self.early_stopping_ = True
                n_classes = len(np.unique(y))
                if self.validation_fraction * X.shape[0] < n_classes:
                    self.validation_fraction_ = n_classes
                else:
                    self.validation_fraction_ = self.validation_fraction
            else:
                raise ValueError("early_stop should be either off, train or valid")

            self.estimator = HistGradientBoostingRegressor(
                loss=self.loss,
                learning_rate=self.learning_rate,
                min_samples_leaf=self.min_samples_leaf,
                max_depth=self.max_depth,
                max_leaf_nodes=self.max_leaf_nodes,
                max_bins=self.max_bins,
                l2_regularization=self.l2_regularization,
                early_stopping=self.early_stopping_,
                tol=self.tol,
                scoring=self.scoring,
                n_iter_no_change=self.n_iter_no_change,
                validation_fraction=self.validation_fraction,
                max_iter=n_iter,
            )
        else:
            self.estimator.max_iter += n_iter  # add n_iter to each step
            self.estimator.max_iter = min(
                self.estimator.max_iter, self.max_iter
            )  # limit the number of iterations

        self.estimator.fit(X, y, sample_weight=sample_weight)

        # check whether fully fitted or need to add more iterations
        if (
            self.estimator.max_iter >= self.max_iter
            or self.estimator.max_iter > self.estimator.n_iter_
        ):

            self._fitted = True

        return self

    def fit(self, X, y):

        self._fit_iteration(X, y, n_iter=2)

        # accelerate iteration process
        iteration = 2
        while not self._fitted:
            n_iter = int(2**iteration / 2)
            self._fit_iteration(X, y, n_iter=n_iter)
            iteration += 1

        return self

    def predict(self, X):

        return self.estimator.predict(X)

    def predict_proba(self, X):

        raise NotImplementedError("predict_proba is not implemented for regression.")


class KNearestNeighborsRegressor(sklearn.neighbors.KNeighborsRegressor):
    def __init__(
        self,
        n_neighbors=1,
        weights="uniform",
        p=2,
    ):
        self.n_neighbors = int(n_neighbors)
        self.weights = weights
        self.p = int(p)

        super().__init__(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            p=self.p,
        )

        self._fitted = False  # whether the model is fitted

    def fit(self, X, y):

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(self, X):

        return super().predict(X)

    def predict_proba(self, X):

        raise NotImplementedError("predict_proba is not implemented for regression.")


class LibLinear_SVR(sklearn.svm.LinearSVR):
    def __init__(
        self,
        epsilon=0.1,
        loss="squared_epsilon_insensitive",
        dual=False,
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
    ):
        self.epsilon = float(epsilon)
        self.loss = loss
        self.dual = dual
        self.tol = float(tol)
        self.C = float(C)
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling

        super().__init__(
            epsilon=self.epsilon,
            loss=self.loss,
            dual=self.dual,
            tol=self.tol,
            C=self.C,
            fit_intercept=self.fit_intercept,
            intercept_scaling=self.intercept_scaling,
        )

        self._fitted = False  # whether the model is fitted

    def fit(self, X, y):

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(self, X):

        return super().predict(X)

    def predict_proba(self, X):

        raise NotImplementedError("predict_proba is not implemented for regression.")


class LibSVM_SVR(sklearn.svm.SVR):
    def __init__(
        self,
        kernel="rbf",
        C=1.0,
        epsilon=0.1,
        degree=3,
        gamma=0.1,
        coef0=0,
        tol=1e-3,
        shrinking=True,
        max_iter=-1,
    ):
        self.C = float(C)
        self.kernel = kernel
        self.epsilon = float(epsilon)
        self.degree = int(degree)
        self.gamma = float(gamma)
        self.coef0 = float(coef0)
        self.shrinking = shrinking
        self.tol = float(tol)
        self.max_iter = max_iter

        super().__init__(
            C=self.C,
            kernel=self.kernel,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            shrinking=self.shrinking,
            tol=self.tol,
            max_iter=self.max_iter,
        )

        self._fitted = False  # whether the model is fitted

    def fit(self, X, y):

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(self, X):

        return super().predict(X)

    def predict_proba(self, X):

        raise NotImplementedError("predict_proba is not implemented for regression.")


class MLPRegressor:
    def __init__(
        self,
        hidden_layer_depth=1,
        num_nodes_per_layer=32,
        activation="tanh",
        alpha=1e-4,
        learning_rate_init=1e-3,
        early_stopping="valid",
        n_iter_no_change=32,
        validation_fraction=0.1,
        tol=1e-4,
        solver="adam",
        batch_size="auto",
        shuffle=True,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
    ):
        self.hidden_layer_depth = int(hidden_layer_depth)
        self.num_nodes_per_layer = int(num_nodes_per_layer)
        self.hidden_layer_sizes = tuple(
            self.num_nodes_per_layer for _ in range(self.hidden_layer_depth)
        )
        self.activation = str(activation)
        self.alpha = float(alpha)
        self.learning_rate_init = float(learning_rate_init)
        self.early_stopping = str(early_stopping)
        self.n_iter_no_change = int(n_iter_no_change)
        self.validation_fraction = float(validation_fraction)
        self.tol = float(tol)
        self.solver = solver
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.beta_1 = float(beta_1)
        self.beta_2 = float(beta_2)
        self.epsilon = float(epsilon)

        self.estimator = None  # the fitted estimator
        self.max_iter = self._get_max_iter()  # limit the number of iterations

        self._fitted = False  # whether the model is fitted

    @staticmethod
    def _get_max_iter():  # define global max_iter

        return MAX_ITER

    def _fit_iteration(self, X, y, n_iter=2):

        if self.estimator is None:
            from sklearn.neural_network import MLPRegressor

            # map from autosklearn parameter space to sklearn parameter space
            if self.early_stopping == "train":
                self.validation_fraction = 0.0
                self.tol = float(self.tol)
                self.n_iter_no_change = int(self.n_iter_no_change)
                self.early_stopping = False
            elif self.early_stopping == "valid":
                self.validation_fraction = float(self.validation_fraction)
                self.tol = float(self.tol)
                self.n_iter_no_change = int(self.n_iter_no_change)
                self.early_stopping = True
            else:
                raise ValueError(
                    "Early stopping only supports 'train' and 'valid'. Got {}".format(
                        self.early_stopping
                    )
                )

            self.estimator = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                alpha=self.alpha,
                learning_rate_init=self.learning_rate_init,
                early_stopping=self.early_stopping,
                n_iter_no_change=self.n_iter_no_change,
                validation_fraction=self.validation_fraction,
                tol=self.tol,
                solver=self.solver,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                beta_1=self.beta_1,
                beta_2=self.beta_2,
                epsilon=self.epsilon,
                max_iter=n_iter,
            )
        else:
            # MLPClassifier can record previous training
            self.estimator.max_iter = min(
                self.estimator.max_iter - self.estimator.n_iter_, self.max_iter
            )  # limit the number of iterations

        self.estimator.fit(X, y)

        # check whether fully fitted or need to add more iterations
        if (
            self.estimator.n_iter_ >= self.estimator.max_iter
            or self.estimator._no_improvement_count > self.n_iter_no_change
        ):

            self._fitted = True

        return self

    def fit(self, X, y):

        self._fit_iteration(X, y, n_iter=2)

        # accelerate iteration process
        iteration = 2
        while not self._fitted:
            n_iter = int(2**iteration / 2)
            self._fit_iteration(X, y, n_iter=n_iter)
            iteration += 1

        return self

    def predict(self, X):

        return self.estimator.predict(X)

    def predict_proba(self, X):

        raise NotImplementedError("predict_proba is not implemented for regression.")


class RandomForestRegressor:
    def __init__(
        self,
        criterion="mse",
        max_features=0.5,
        max_depth="None",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        bootstrap=True,
        max_leaf_nodes="None",
        min_impurity_decrease=0.0,
    ):
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = None if is_none(max_depth) else int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.min_weight_fraction_leaf = float(min_weight_fraction_leaf)
        self.bootstrap = bootstrap
        self.max_leaf_nodes = None if is_none(max_leaf_nodes) else int(max_leaf_nodes)
        self.min_impurity_decrease = float(min_impurity_decrease)

        self.estimator = None  # the fitted estimator
        self.max_iter = self._get_max_iter()  # limit the number of iterations

        self._fitted = False  # whether the model is fitted

    @staticmethod
    def _get_max_iter():  # define global max_iter

        return MAX_ITER

    def _fit_iteration(self, X, y, n_iter=2):

        if self.estimator is None:
            from sklearn.ensemble import RandomForestRegressor

            self.estimator = RandomForestRegressor(
                n_estimators=n_iter,
                criterion=self.criterion,
                max_features=self.max_features,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                bootstrap=self.bootstrap,
                max_leaf_nodes=self.max_leaf_nodes,
                min_impurity_decrease=self.min_impurity_decrease,
            )
        else:
            self.estimator.n_estimators += n_iter
            self.estimator.n_estimators = min(
                self.estimator.n_estimators, self.max_iter
            )

        self.estimator.fit(X, y)

        if (
            self.estimator.n_estimators >= self.max_iter
            or self.estimator.n_estimators > len(self.estimator.estimators_)
        ):
            self._fitted = True

        return self

    def fit(self, X, y):

        self._fit_iteration(X, y, n_iter=2)

        # accelerate iteration process
        iteration = 2
        while not self._fitted:
            n_iter = int(2**iteration / 2)
            self._fit_iteration(X, y, n_iter=n_iter)
            iteration += 1

        return self

    def predict(self, X):

        return self.estimator.predict(X)

    def predict_proba(self, X):

        raise NotImplementedError("predict_proba is not implemented for regression.")


class SGDRegressor:
    def __init__(
        self,
        loss="squared_loss",
        penalty="l2",
        alpha=0.0001,
        fit_intercept=True,
        tol=1e-4,
        learning_rate="invscaling",
        l1_ratio=0.15,
        epsilon=1e-4,
        eta0=0.01,
        power_t=0.5,
        average=False,
    ):
        self.loss = loss
        self.penalty = penalty
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept
        self.tol = float(tol)
        self.learning_rate = learning_rate
        self.l1_ratio = 0.15 if is_none(l1_ratio) else float(l1_ratio)
        self.epsilon = 0.1 if is_none(epsilon) else float(epsilon)
        self.eta0 = float(eta0)
        self.power_t = 0.5 if is_none(power_t) else float(power_t)
        self.average = average

        self.estimator = None  # the fitted estimator
        self.max_iter = self._get_max_iter()  # limit the number of iterations

        self._fitted = False  # whether the model is fitted

    @staticmethod
    def _get_max_iter():  # define global max_iter

        return MAX_ITER

    def _fit_iteration(self, X, y, n_iter=2):

        if self.estimator is None:
            from sklearn.linear_model import SGDRegressor

            self.estimator = SGDRegressor(
                loss=self.loss,
                penalty=self.penalty,
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                tol=self.tol,
                learning_rate=self.learning_rate,
                l1_ratio=self.l1_ratio,
                epsilon=self.epsilon,
                eta0=self.eta0,
                power_t=self.power_t,
                average=self.average,
                max_iter=n_iter,
            )
        else:
            self.estimator.max_iter += n_iter
            self.estimator.max_iter = min(self.estimator.max_iter, self.max_iter)

        self.estimator.fit(X, y)

        if (
            self.estimator.max_iter >= self.max_iter
            or self.estimator.max_iter > self.estimator.n_iter_
        ):
            self._fitted = True

        return self

    def fit(self, X, y):

        self._fit_iteration(X, y, n_iter=2)

        # accelerate iteration process
        iteration = 2
        while not self._fitted:
            n_iter = int(2**iteration / 2)
            self._fit_iteration(X, y, n_iter=n_iter)
            iteration += 1

        return self

    def predict(self, X):

        return self.estimator.predict(X)

    def predict_proba(self, X):

        raise NotImplementedError("predict_proba is not implemented for regression.")


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

    def predict_proba(self, X):

        raise NotImplementedError("predict_proba is not implemented for regression.")


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

    def predict_proba(self, X):

        raise NotImplementedError("predict_proba is not implemented for regression.")


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

    def predict_proba(self, X):

        raise NotImplementedError("predict_proba is not implemented for regression.")


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

    def predict_proba(self, X):

        raise NotImplementedError("predict_proba is not implemented for regression.")


class BayesianRidge(sklearn.linear_model.BayesianRidge):
    def __init__(
        self,
        tol=1e-3,
        alpha_1=1e-6,
        alpha_2=1e-6,
        lambda_1=1e-6,
        lambda_2=1e-6,
    ):
        self.tol = float(tol)
        self.alpha_1 = float(alpha_1)
        self.alpha_2 = float(alpha_2)
        self.lambda_1 = float(lambda_1)
        self.lambda_2 = float(lambda_2)

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

    def predict_proba(self, X):

        raise NotImplementedError("predict_proba is not implemented for regression.")


# combined with autosklearn version, thus deprecated here
# class HistGradientBoostingRegressor(sklearn.ensemble.HistGradientBoostingRegressor):
#     def __init__(
#         self,
#         loss="least_squares",
#         learning_rate=0.1,
#         max_leaf_nodes=31,
#         max_depth=None,
#         min_samples_leaf=20,
#         l2_regularization=0,
#         tol=1e-7,
#     ):
#         self.loss = loss
#         self.learning_rate = learning_rate
#         self.max_leaf_nodes = max_leaf_nodes
#         self.max_depth = max_depth
#         self.min_samples_leaf = min_samples_leaf
#         self.l2_regularization = l2_regularization
#         self.tol = tol

#         super().__init__(
#             loss=self.loss,
#             learning_rate=self.learning_rate,
#             max_leaf_nodes=self.max_leaf_nodes,
#             max_depth=self.max_depth,
#             min_samples_leaf=self.min_samples_leaf,
#             l2_regularization=self.l2_regularization,
#             tol=self.tol,
#         )

#         self._fitted = False

#     def fit(self, X, y):

#         super().fit(X, y)

#         self._fitted = True

#         return self

#     def predict(self, X):

#         return super().predict(X)


class GradientBoostingRegressor(sklearn.ensemble.GradientBoostingRegressor):
    def __init__(
        self,
        loss="ls" if sklearn_1_0_0 else "squared_error",  # for default arguments
        learning_rate=0.1,
        n_estimators=100,
        subsample=1.0,
        criterion="friedman_mse",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        max_features="auto",
        max_leaf_nodes=31,
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-7,
    ):
        self.loss = loss
        self.learning_rate = float(learning_rate)
        self.n_estimators = int(n_estimators)
        self.subsample = float(subsample)
        self.criterion = criterion
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.min_weight_fraction_leaf = float(min_weight_fraction_leaf)
        self.max_depth = None if is_none(max_depth) else int(max_depth)
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.max_features = max_features
        self.max_leaf_nodes = int(max_leaf_nodes)
        self.validation_fraction = float(validation_fraction)
        self.n_iter_no_change = int(n_iter_no_change)
        self.tol = float(tol)

        super().__init__(
            loss=self.loss,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            criterion=self.criterion,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_depth=self.max_depth,
            min_impurity_decrease=self.min_impurity_decrease,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            tol=self.tol,
        )

        self._fitted = False  # whether the model is fitted

    def fit(self, X, y):

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(self, X):

        return super().predict(X)

    def predict_proba(self, X):

        raise NotImplementedError("predict_proba is not implemented for regression.")
