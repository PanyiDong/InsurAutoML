"""
File: _sklearn.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.5
Relative Path: /InsurAutoML/model/sklearn_classifiers.py
File: sklearn_classifier.py
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Thursday, 1st June 2023 9:41:48 am
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

# Update: Nov. 15, 2022
# sklearn > 1.0.0 is required for this module.
# need to enable hist gradient boosting features first
# no need for sklearn version >= 1.0.0
# sklearn_1_0_0 = sklearn.__version__ < "1.0.0"
# if sklearn_1_0_0:
#     from sklearn.experimental import enable_hist_gradient_boosting

from ..constant import MAX_ITER
from ..utils.base import is_none
from ..utils.data import softmax
from .base import BaseModel

##########################################################################
# models from sklearn
# wrap for some-degree of flexibility (initialization, _fitted, etc.)

##########################################################################
# classifiers


class AdaboostClassifier(sklearn.ensemble.AdaBoostClassifier, BaseModel):
    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 0.1,
        algorithm: str = "SAMME.R",
        max_depth: int = 1,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.algorithm = algorithm
        self.max_depth = int(max_depth)

        from sklearn.tree import DecisionTreeClassifier

        super().__init__(
            estimator=DecisionTreeClassifier(max_depth=self.max_depth),
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            algorithm=self.algorithm,
        )

        self._fitted = False  # whether the model is fitted

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> AdaboostClassifier:
        super(AdaboostClassifier, self).fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return super(AdaboostClassifier, self).predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return super(AdaboostClassifier, self).predict_proba(X)


class BernoulliNB(sklearn.naive_bayes.BernoulliNB, BaseModel):
    def __init__(
        self,
        alpha: float = 1,
        fit_prior: bool = True,
    ) -> None:
        self.alpha = float(alpha)
        self.fit_prior = fit_prior

        super().__init__(
            alpha=self.alpha,
            fit_prior=self.fit_prior,
        )

        self._fitted = False  # whether the model is fitted

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> BernoulliNB:
        super(BernoulliNB, self).fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return super(BernoulliNB, self).predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return super(BernoulliNB, self).predict_proba(X)


class DecisionTreeClassifier(sklearn.tree.DecisionTreeClassifier, BaseModel):
    def __init__(
        self,
        criterion: str = "gini",
        max_depth_factor: float = 0.5,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: float = 1.0,
        max_leaf_nodes: Union[str, int] = "None",
        min_impurity_decrease: float = 0.0,
    ) -> None:
        self.criterion = criterion
        self.max_depth_factor = max_depth_factor
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.min_weight_fraction_leaf = float(min_weight_fraction_leaf)
        self.max_features = float(max_features)
        self.max_leaf_nodes = None if is_none(max_leaf_nodes) else max_leaf_nodes
        self.min_impurity_decrease = float(min_impurity_decrease)

        super().__init__()
        self._fitted = False  # whether the model is fitted

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> DecisionTreeClassifier:
        super(DecisionTreeClassifier, self).__init__(
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

        super(DecisionTreeClassifier, self).fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return super(DecisionTreeClassifier, self).predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return super(DecisionTreeClassifier, self).predict_proba(X)


class ExtraTreesClassifier(BaseModel):
    def __init__(
        self,
        criterion: str = "gini",
        max_depth: Union[str, int] = "None",
        max_leaf_nodes: Union[str, int] = "None",
        min_samples_leaf: int = 1,
        min_samples_split: int = 2,
        max_features: float = 0.5,
        bootstrap: bool = False,
        min_weight_fraction_leaf: float = 0.0,
        min_impurity_decrease: float = 0.0,
    ) -> None:
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

        super().__init__()
        self._fitted = False  # whether the model is fitted

    @staticmethod
    def _get_max_iter() -> int:  # define global max_iter
        return MAX_ITER

    def _fit_iteration(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
        n_iter: int = 2,
    ) -> ExtraTreesClassifier:
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

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> ExtraTreesClassifier:
        self._fit_iteration(X, y, n_iter=2)

        # accelerate iteration process
        iteration = 2
        while not self._fitted:
            n_iter = int(2**iteration / 2)
            self._fit_iteration(X, y, n_iter=n_iter)
            iteration += 1

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return self.estimator.predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return self.estimator.predict_proba(X)


class GaussianNB(sklearn.naive_bayes.GaussianNB, BaseModel):
    def __init__(
        self,
    ) -> None:
        super().__init__()

        self._fitted = False  # whether the model is fitted

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> GaussianNB:
        super(GaussianNB, self).fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return super(GaussianNB, self).predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return super(GaussianNB, self).predict_proba(X)


class HistGradientBoostingClassifier(BaseModel):
    def __init__(
        self,
        loss: str = "auto",
        learning_rate: float = 0.1,
        min_samples_leaf: int = 20,
        max_depth: Union[str, int] = "None",
        max_leaf_nodes: int = 31,
        max_bins: int = 255,
        l2_regularization: float = 1e-10,
        early_stop: str = "off",
        tol: float = 1e-7,
        scoring: str = "loss",
        n_iter_no_change: int = 10,
        validation_fraction: float = 0.1,
    ) -> None:
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

        super().__init__()
        self._fitted = False  # whether the model is fitted

    @staticmethod
    def _get_max_iter() -> int:  # define global max_iter
        return MAX_ITER

    def _fit_iteration(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
        n_iter: int = 2,
        sample_weight: Union[pd.DataFrame, pd.Series, np.ndarray] = None,
    ) -> HistGradientBoostingClassifier:
        if self.estimator is None:
            from sklearn.ensemble import HistGradientBoostingClassifier

            # map from autosklearn parameter space to sklearn parameter space
            if self.early_stop == "off":
                self.n_iter_no_change = 1
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

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> HistGradientBoostingClassifier:
        self._fit_iteration(X, y, n_iter=2)

        # accelerate iteration process
        iteration = 2
        while not self._fitted:
            n_iter = int(2**iteration / 2)
            self._fit_iteration(X, y, n_iter=n_iter)
            iteration += 1

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return self.estimator.predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return self.estimator.predict_proba(X)


class KNearestNeighborsClassifier(sklearn.neighbors.KNeighborsClassifier, BaseModel):
    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        p: int = 2,
    ) -> None:
        self.n_neighbors = int(n_neighbors)
        self.weights = weights
        self.p = int(p)

        super().__init__(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            p=self.p,
        )

        self._fitted = False  # whether the model is fitted

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> KNearestNeighborsClassifier:
        super(KNearestNeighborsClassifier, self).fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return super(KNearestNeighborsClassifier, self).predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return super(KNearestNeighborsClassifier, self).predict_proba(X)


class LDA(sklearn.discriminant_analysis.LinearDiscriminantAnalysis, BaseModel):
    def __init__(
        self,
        shrinkage_type: str = "auto",
        tol: float = 1e-4,
        shrinkage_factor: float = 0.5,
    ) -> None:
        self.shrinkage_type = shrinkage_type
        self.tol = float(tol)
        self.shrinkage_factor = float(shrinkage_factor)

        super().__init__()
        self._fitted = False  # whether the model is fitted

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> LDA:
        if self.shrinkage_type is None or self.shrinkage_type == "None":
            _shrinkage = None
            solver = "svd"
        elif self.shrinkage_type == "auto":
            _shrinkage = "auto"
            solver = "lsqr"
        elif self.shrinkage_type == "manual":
            _shrinkage = float(self.shrinkage_factor)
            solver = "lsqr"
        else:
            raise ValueError(
                "Not a valid shrinkage parameter, should be None, auto or manual. Got {}".format(
                    self.shrinkage
                )
            )

        super(LDA, self).__init__(
            shrinkage=_shrinkage,
            solver=solver,
            tol=self.tol,
        )

        super(LDA, self).fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return super(LDA, self).predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return super(LDA, self).predict_proba(X)


class LibLinear_SVC(BaseModel):
    def __init__(
        self,
        penalty: str = "l2",
        loss: str = "squared_hinge",
        dual: bool = False,
        tol: float = 1e-4,
        C: float = 1.0,
        multi_class: str = "ovr",
        fit_intercept: bool = True,
        intercept_scaling: float = 1,
    ) -> None:
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

        super().__init__()
        self._fitted = False  # whether the model is fitted

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> LibLinear_SVC:
        self.estimator.fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return self.estimator.predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return self.estimator.predict_proba(X)


class LibSVM_SVC(BaseModel):
    def __init__(
        self,
        C: float = 1.0,
        kernel: float = "rbf",
        degree: int = 3,
        gamma: float = 0.1,
        coef0: float = 0,
        tol: float = 1e-3,
        shrinking: bool = True,
        max_iter: int = -1,
    ) -> None:
        self.C = float(C)
        self.kernel = kernel
        self.degree = 3 if is_none(degree) else int(degree)
        self.gamma = 0.0 if is_none(gamma) else float(gamma)
        self.coef0 = 0.0 if is_none(coef0) else float(coef0)
        self.shrinking = shrinking
        self.tol = float(tol)
        self.max_iter = int(max_iter)

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

        super().__init__()
        self._fitted = False  # whether the model is fitted

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> LibSVM_SVC:
        self.estimator.fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return self.estimator.predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return self.estimator.predict_proba(X)


class MLPClassifier(BaseModel):
    def __init__(
        self,
        hidden_layer_depth: int = 1,
        num_nodes_per_layer: int = 32,
        activation: str = "relu",
        alpha: float = 1e-4,
        learning_rate_init: float = 1e-3,
        early_stopping: str = "valid",
        n_iter_no_change: int = 32,
        validation_fraction: float = 0.1,
        tol: float = 1e-4,
        solver: str = "adam",
        batch_size: str = "auto",
        shuffle: bool = True,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
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

        super().__init__()
        self._fitted = False  # whether the model is fitted

    @staticmethod
    def _get_max_iter() -> int:  # define global max_iter
        return MAX_ITER

    def _fit_iteration(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
        n_iter: int = 2,
    ) -> MLPClassifier:
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
                self._get_max_iter() - self.estimator.n_iter_, n_iter
            )  # limit the number of iterations

        self.estimator.fit(X, y)

        # check whether fully fitted or need to add more iterations
        if (
            self.estimator.n_iter_ >= self.estimator.max_iter
            or self.estimator._no_improvement_count > self.n_iter_no_change
        ):
            self._fitted = True

        return self

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> MLPClassifier:
        self._fit_iteration(X, y, n_iter=2)

        # accelerate iteration process
        iteration = 2
        while not self._fitted:
            n_iter = int(2**iteration / 2)
            self._fit_iteration(X, y, n_iter=n_iter)
            iteration += 1

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return self.estimator.predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return self.estimator.predict_proba(X)


class MultinomialNB(sklearn.naive_bayes.MultinomialNB, BaseModel):
    def __init__(self, alpha: float = 1.0, fit_prior: bool = True) -> None:
        self.alpha = float(alpha)
        self.fit_prior = fit_prior

        super().__init__(
            alpha=self.alpha,
            fit_prior=self.fit_prior,
        )

        self._fitted = False  # whether the model is fitted

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> MultinomialNB:
        # make sure the data contains only non-negative values
        if scipy.sparse.issparse(X):
            X.data[X.data < 0] = 0.0
        else:
            X[X < 0] = 0.0

        super(MultinomialNB, self).fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return super(MultinomialNB, self).predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return super(MultinomialNB, self).predict_proba(X)


class PassiveAggressive(BaseModel):
    def __init__(
        self,
        C: float = 1.0,
        fit_intercept: bool = True,
        average: bool = False,
        tol: float = 1e-4,
        loss: str = "hinge",
    ) -> None:
        self.C = float(C)
        self.fit_intercept = fit_intercept
        self.average = average
        self.tol = float(tol)
        self.loss = loss

        self.estimator = None  # the fitted estimator
        self.max_iter = self._get_max_iter()  # limit the number of iterations

        super().__init__()
        self._fitted = False  # whether the model is fitted

    @staticmethod
    def _get_max_iter() -> int:  # define global max_iter
        return MAX_ITER

    def _fit_iteration(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
        n_iter: int = 2,
    ) -> PassiveAggressive:
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

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> PassiveAggressive:
        self._fit_iteration(X, y, n_iter=2)

        # accelerate iteration process
        iteration = 2
        while not self._fitted:
            n_iter = int(2**iteration / 2)
            self._fit_iteration(X, y, n_iter=n_iter)
            iteration += 1

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return self.estimator.predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return softmax(self.estimator.decision_function(X))


class QDA(sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis, BaseModel):
    def __init__(
        self,
        reg_param: float = 0.0,
    ) -> None:
        self.reg_param = float(reg_param)

        super().__init__(
            reg_param=self.reg_param,
        )

        self._fitted = False  # whether the model is fitted

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> QDA:
        super(QDA, self).fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return super(QDA, self).predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return super(QDA, self).predict_proba(X)


class RandomForestClassifier(BaseModel):
    def __init__(
        self,
        criterion: str = "gini",
        max_features: float = 0.5,
        max_depth: Union[str, int] = "None",
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_weight_fraction_leaf: float = 0.0,
        bootstrap: bool = True,
        max_leaf_nodes: Union[str, int] = "None",
        min_impurity_decrease: float = 0.0,
    ) -> None:
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

        super().__init__()
        self._fitted = False  # whether the model is fitted

    @staticmethod
    def _get_max_iter() -> int:  # define global max_iter
        return MAX_ITER

    def _fit_iteration(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
        n_iter: int = 2,
    ) -> RandomForestClassifier:
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

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> RandomForestClassifier:
        self._fit_iteration(X, y, n_iter=2)

        # accelerate iteration process
        iteration = 2
        while not self._fitted:
            n_iter = int(2**iteration / 2)
            self._fit_iteration(X, y, n_iter=n_iter)
            iteration += 1

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return self.estimator.predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return self.estimator.predict_proba(X)


class SGDClassifier(BaseModel):
    def __init__(
        self,
        loss: str = "log",
        penalty: str = "l2",
        alpha: float = 0.0001,
        fit_intercept: bool = True,
        tol: float = 1e-4,
        learning_rate: str = "invscaling",
        l1_ratio: float = 0.15,
        epsilon: float = 1e-4,
        eta0: float = 0.01,
        power_t: float = 0.5,
        average: bool = False,
    ) -> None:
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

        super().__init__()
        self._fitted = False  # whether the model is fitted

    @staticmethod
    def _get_max_iter() -> int:  # define global max_iter
        return MAX_ITER

    def _fit_iteration(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
        n_iter: int = 2,
    ) -> SGDClassifier:
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

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> SGDClassifier:
        self._fit_iteration(X, y, n_iter=2)

        # accelerate iteration process
        iteration = 2
        while not self._fitted:
            n_iter = int(2**iteration / 2)
            self._fit_iteration(X, y, n_iter=n_iter)
            iteration += 1

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return self.estimator.predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return self.estimator.predict_proba(X)


class LogisticRegression(sklearn.linear_model.LogisticRegression, BaseModel):
    def __init__(
        self,
        penalty: str = "l2",
        tol: float = 1e-4,
        C: float = 1.0,
    ) -> None:
        self.penalty = penalty
        self.tol = float(tol)
        self.C = float(C)

        super().__init__(
            penalty=self.penalty,
            tol=self.tol,
            C=self.C,
        )

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> LogisticRegression:
        super(LogisticRegression, self).fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return super(LogisticRegression, self).predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return super(LogisticRegression, self).predict_proba(X)


class ComplementNB(sklearn.naive_bayes.ComplementNB, BaseModel):
    def __init__(
        self,
        alpha: float = 1.0,
        fit_prior: bool = True,
        norm: bool = False,
    ) -> None:
        self.alpha = float(alpha)
        self.fit_prior = fit_prior
        self.norm = norm

        super().__init__(
            alpha=self.alpha,
            fit_prior=self.fit_prior,
            norm=self.norm,
        )

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> ComplementNB:
        super(ComplementNB, self).fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return super(ComplementNB, self).predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return super(ComplementNB, self).predict_proba(X)


class GradientBoostingClassifier(
    sklearn.ensemble.GradientBoostingClassifier, BaseModel
):
    def __init__(
        self,
        loss: str = "deviance",
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 1.0,
        criterion: str = "friedman_mse",
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_depth: int = 3,
        min_impurity_decrease: float = 0.0,
        max_features: float = 1.0,
        max_leaf_nodes: int = 31,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 10,
        tol: float = 1e-7,
    ) -> None:
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

        super().__init__()
        self._fitted = False  # whether the model is fitted

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> GradientBoostingClassifier:
        super(GradientBoostingClassifier, self).__init__(
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
            max_features=max(1, int(X.shape[1] ** self.max_features)),
            max_leaf_nodes=self.max_leaf_nodes,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            tol=self.tol,
        )

        super(GradientBoostingClassifier, self).fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return super(GradientBoostingClassifier, self).predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return super(GradientBoostingClassifier, self).predict_proba(X)
