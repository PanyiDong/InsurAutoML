"""
File Name: sklearn_regressors.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.5
Relative Path: /InsurAutoML/model/sklearn_regressors.py
File Created: Monday, 29th May 2023 3:35:28 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Thursday, 1st June 2023 9:42:10 am
Modified By: Panyi Dong (panyid2@illinois.edu)

-----
MIT License

Copyright (c) 2023 - 2023, Panyi Dong

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
from .base import BaseModel

##########################################################################
# models from sklearn
# wrap for some-degree of flexibility (initialization, _fitted, etc.)

##########################################################################
# regressors


class AdaboostRegressor(sklearn.ensemble.AdaBoostRegressor, BaseModel):
    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 0.1,
        loss: str = "linear",
        max_depth: int = 1,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.loss = loss
        self.max_depth = int(max_depth)

        from sklearn.tree import DecisionTreeRegressor

        super().__init__(
            estimator=DecisionTreeRegressor(max_depth=self.max_depth),
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            loss=self.loss,
        )

        self._fitted = False  # whether the model is fitted

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> AdaboostRegressor:
        super(AdaboostRegressor, self).fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return super(AdaboostRegressor, self).predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        raise NotImplementedError("predict_proba is not implemented for regression.")


class ARDRegression(sklearn.linear_model.ARDRegression, BaseModel):
    def __init__(
        self,
        n_iter: int = 300,
        tol: float = 1e-3,
        alpha_1: float = 1e-6,
        alpha_2: float = 1e-6,
        lambda_1: float = 1e-6,
        lambda_2: float = 1e-6,
        threshold_lambda: float = 1e4,
        fit_intercept: bool = True,
    ) -> None:
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

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> ARDRegression:
        super(ARDRegression, self).fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return super(ARDRegression, self).predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        raise NotImplementedError("predict_proba is not implemented for regression.")


class DecisionTreeRegressor(sklearn.tree.DecisionTreeRegressor, BaseModel):
    def __init__(
        self,
        criterion: str = "squared_error",
        max_features: float = 1.0,
        max_depth_factor: float = 0.5,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_leaf_nodes: Union[str, int] = "None",
        min_impurity_decrease: float = 0.0,
    ) -> None:
        self.criterion = criterion
        self.max_depth_factor = max_depth_factor
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.min_weight_fraction_leaf = float(min_weight_fraction_leaf)
        self.max_features = float(max_features)
        self.max_leaf_nodes = None if is_none(max_leaf_nodes) else int(max_leaf_nodes)
        self.min_impurity_decrease = float(min_impurity_decrease)

        super().__init__()
        self._fitted = False  # whether the model is fitted

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> DecisionTreeRegressor:
        super(DecisionTreeRegressor, self).__init__(
            criterion=self.criterion,
            max_depth=max(int(self.max_depth_factor * X.shape[1]), 1),
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
        )

        super(DecisionTreeRegressor, self).fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return super(DecisionTreeRegressor, self).predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        raise NotImplementedError("predict_proba is not implemented for regression.")


class ExtraTreesRegressor(BaseModel):
    def __init__(
        self,
        criterion: str = "squared_error",
        max_depth: Union[str, int] = "None",
        max_leaf_nodes: Union[str, int] = "None",
        min_samples_leaf: int = 1,
        min_samples_split: int = 2,
        max_features: float = 1,
        bootstrap: bool = False,
        min_weight_fraction_leaf: float = 0.0,
        min_impurity_decrease: float = 0.0,
    ) -> None:
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
    ) -> ExtraTreesRegressor:
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

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> ExtraTreesRegressor:
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
        raise NotImplementedError("predict_proba is not implemented for regression.")


class GaussianProcess(sklearn.gaussian_process.GaussianProcessRegressor, BaseModel):
    def __init__(
        self,
        alpha: float = 1e-8,
        thetaL: float = 1e-6,
        thetaU: float = 100000.0,
    ) -> None:
        self.alpha = float(alpha)
        self.thetaL = float(thetaL)
        self.thetaU = float(thetaU)

        super().__init__()
        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> GaussianProcess:
        n_features = X.shape[1]
        kernel = sklearn.gaussian_process.kernels.RBF(
            length_scale=[1.0] * n_features,
            length_scale_bounds=[(self.thetaL, self.thetaU)] * n_features,
        )

        super(GaussianProcess, self).__init__(
            kernel=kernel,
            n_restarts_optimizer=10,
            optimizer="fmin_l_bfgs_b",
            alpha=self.alpha,
            copy_X_train=True,
            normalize_y=True,
        )

        super(GaussianProcess, self).fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return super(GaussianProcess, self).predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        raise NotImplementedError("predict_proba is not implemented for regression.")


class HistGradientBoostingRegressor(BaseModel):
    def __init__(
        self,
        loss: str = "squared_error",
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
    ) -> HistGradientBoostingRegressor:
        if self.estimator is None:
            from sklearn.ensemble import HistGradientBoostingRegressor

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

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> HistGradientBoostingRegressor:
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
        raise NotImplementedError("predict_proba is not implemented for regression.")


class KNearestNeighborsRegressor(sklearn.neighbors.KNeighborsRegressor, BaseModel):
    def __init__(
        self,
        n_neighbors: int = 1,
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
    ) -> KNearestNeighborsRegressor:
        super(KNearestNeighborsRegressor, self).fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return super(KNearestNeighborsRegressor, self).predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        raise NotImplementedError("predict_proba is not implemented for regression.")


class LibLinear_SVR(sklearn.svm.LinearSVR, BaseModel):
    def __init__(
        self,
        epsilon: float = 0.1,
        loss: str = "squared_epsilon_insensitive",
        dual: bool = False,
        tol: float = 1e-4,
        C: float = 1.0,
        fit_intercept: bool = True,
        intercept_scaling: float = 1,
    ) -> None:
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

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> LibLinear_SVR:
        super(LibLinear_SVR, self).fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return super(LibLinear_SVR, self).predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        raise NotImplementedError("predict_proba is not implemented for regression.")


class LibSVM_SVR(sklearn.svm.SVR, BaseModel):
    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        epsilon: float = 0.1,
        degree: int = 3,
        gamma: float = 0.1,
        coef0: float = 0.0,
        tol: float = 1e-3,
        shrinking: bool = True,
        max_iter: int = -1,
    ) -> None:
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

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> LibSVM_SVR:
        super(LibSVM_SVR, self).fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return super(LibSVM_SVR, self).predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        raise NotImplementedError("predict_proba is not implemented for regression.")


class MLPRegressor(BaseModel):
    def __init__(
        self,
        hidden_layer_depth: int = 1,
        num_nodes_per_layer: int = 32,
        activation: str = "tanh",
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
    ) -> MLPRegressor:
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
    ) -> MLPRegressor:
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
        raise NotImplementedError("predict_proba is not implemented for regression.")


class RandomForestRegressor(BaseModel):
    def __init__(
        self,
        criterion: str = "squared_error",
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
    ) -> RandomForestRegressor:
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

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> RandomForestRegressor:
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
        raise NotImplementedError("predict_proba is not implemented for regression.")


class SGDRegressor(BaseModel):
    def __init__(
        self,
        loss: str = "squared_error",
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
    ) -> SGDRegressor:
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

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> SGDRegressor:
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
        raise NotImplementedError("predict_proba is not implemented for regression.")


class LinearRegression(sklearn.linear_model.LinearRegression, BaseModel):
    def __init__(
        self,
    ) -> None:
        super().__init__()

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> LinearRegression:
        super(LinearRegression, self).fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return super(LinearRegression, self).predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        raise NotImplementedError("predict_proba is not implemented for regression.")


class Lasso(sklearn.linear_model.Lasso, BaseModel):
    def __init__(
        self,
        alpha: float = 1.0,
        tol: float = 1e-4,
    ) -> None:
        self.alpha = alpha
        self.tol = tol

        super().__init__(
            alpha=self.alpha,
            tol=self.tol,
        )

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> Lasso:
        super(Lasso, self).fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return super(Lasso, self).predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        raise NotImplementedError("predict_proba is not implemented for regression.")


class Ridge(sklearn.linear_model.Ridge, BaseModel):
    def __init__(
        self,
        alpha: float = 1.0,
        tol: float = 1e-3,
        solver: str = "auto",
    ) -> None:
        self.alpha = alpha
        self.tol = tol
        self.solver = solver

        super().__init__(
            alpha=self.alpha,
            tol=self.tol,
            solver=self.solver,
        )

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> Ridge:
        super(Ridge, self).fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return super(Ridge, self).predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        raise NotImplementedError("predict_proba is not implemented for regression.")


class ElasticNet(sklearn.linear_model.ElasticNet, BaseModel):
    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        tol: float = 1e-4,
        selection: str = "cyclic",
    ) -> None:
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

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> ElasticNet:
        super(ElasticNet, self).fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return super(ElasticNet, self).predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        raise NotImplementedError("predict_proba is not implemented for regression.")


class BayesianRidge(sklearn.linear_model.BayesianRidge, BaseModel):
    def __init__(
        self,
        tol: float = 1e-3,
        alpha_1: float = 1e-6,
        alpha_2: float = 1e-6,
        lambda_1: float = 1e-6,
        lambda_2: float = 1e-6,
    ) -> None:
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

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> BayesianRidge:
        super(BayesianRidge, self).fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return super(BayesianRidge, self).predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        raise NotImplementedError("predict_proba is not implemented for regression.")


class GradientBoostingRegressor(sklearn.ensemble.GradientBoostingRegressor, BaseModel):
    def __init__(
        self,
        loss: str = "squared_error",  # for default arguments
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 1.0,
        criterion: str = "friedman_mse",
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_depth: int = 3,
        min_impurity_decrease: float = 0.0,
        max_features: str = "auto",
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

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> GradientBoostingRegressor:
        super(GradientBoostingRegressor, self).fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return super(GradientBoostingRegressor, self).predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        raise NotImplementedError("predict_proba is not implemented for regression.")
