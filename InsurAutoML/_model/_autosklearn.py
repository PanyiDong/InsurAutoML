"""
File Name: _autosklearn.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_model/_autosklearn.py
File Created: Sunday, 17th April 2022 10:50:47 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 14th November 2022 9:01:06 pm
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
import autosklearn.pipeline.components.classification as apcc
import autosklearn.pipeline.components.regression as apcr

####################################################################################################################
# models from autosklearn
# wrap for some-degree of flexibility (initialization, _fitted, etc.)

####################################################################################################################
# classifiers


class AdaboostClassifier(apcc.adaboost.AdaboostClassifier):
    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 0.1,
        algorithm: str = "SAMME.R",
        max_depth: int = 1,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.max_depth = max_depth

        super().__init__(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            algorithm=self.algorithm,
            max_depth=self.max_depth,
        )

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> AdaboostClassifier:

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict_proba(X)


class BernoulliNB(apcc.bernoulli_nb.BernoulliNB):
    def __init__(
        self,
        alpha: float = 1,
        fit_prior: bool = True,
    ) -> None:
        self.alpha = alpha
        self.fit_prior = fit_prior

        super().__init__(alpha=self.alpha, fit_prior=self.fit_prior)

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> BernoulliNB:

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict_proba(X)


class DecisionTreeClassifier(apcc.decision_tree.DecisionTree):
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
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease

        super().__init__(
            criterion=self.criterion,
            max_depth_factor=self.max_depth_factor,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
        )

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> DecisionTreeClassifier:

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict_proba(X)


class ExtraTreesClassifier(apcc.extra_trees.ExtraTreesClassifier):
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
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.min_impurity_decrease = min_impurity_decrease

        super().__init__(
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

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> ExtraTreesClassifier:

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict_proba(X)


class GaussianNB(apcc.gaussian_nb.GaussianNB):
    def __init__(
        self,
    ) -> None:

        super().__init__()

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> GaussianNB:

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict_proba(X)


class HistGradientBoostingClassifier(apcc.gradient_boosting.GradientBoostingClassifier):
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
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.max_bins = max_bins
        self.l2_regularization = l2_regularization
        self.early_stop = early_stop
        self.tol = tol
        self.scoring = scoring
        self.n_iter_no_change = n_iter_no_change
        self.validation_fraction = validation_fraction

        super().__init__(
            loss=self.loss,
            learning_rate=self.learning_rate,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            max_leaf_nodes=self.max_leaf_nodes,
            max_bins=self.max_bins,
            l2_regularization=self.l2_regularization,
            early_stop=self.early_stop,
            tol=self.tol,
            scoring=self.scoring,
            n_iter_no_change=self.n_iter_no_change,
            validation_fraction=self.validation_fraction,
        )

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> HistGradientBoostingClassifier:

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict_proba(X)


class KNearestNeighborsClassifier(apcc.k_nearest_neighbors.KNearestNeighborsClassifier):
    def __init__(
        self,
        n_neighbors: int = 1,
        weights: str = "uniform",
        p: int = 2,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p

        super().__init__(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            p=self.p,
        )

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> KNearestNeighborsClassifier:

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict_proba(X)


class LDA(apcc.lda.LDA):
    def __init__(
        self,
        shrinkage: str = "None",
        tol: float = 1e-4,
        shrinkage_factor: float = 0.5,
    ) -> None:
        self.shrinkage = shrinkage
        self.tol = tol
        self.shrinkage_factor = shrinkage_factor

        super().__init__(
            shrinkage=self.shrinkage,
            tol=self.tol,
            shrinkage_factor=self.shrinkage_factor,
        )

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> LDA:

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict_proba(X)


class LibLinear_SVC(apcc.liblinear_svc.LibLinear_SVC):
    def __init__(
        self,
        penalty: str = "l2",
        loss: str = "squared_hinge",
        dual: bool = False,
        tol: float = 1e-4,
        C: float = 1.0,
        multi_class: str = "ovr",
        fit_intercept: bool = True,
        intercept_scaling: float = 1.0,
    ) -> None:
        self.penalty = penalty
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.C = C
        self.multi_class = multi_class
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling

        super().__init__(
            penalty=self.penalty,
            loss=self.loss,
            dual=self.dual,
            tol=self.tol,
            C=self.C,
            multi_class=self.multi_class,
            fit_intercept=self.fit_intercept,
            intercept_scaling=self.intercept_scaling,
        )

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> LibLinear_SVC:

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict_proba(X)


class LibSVM_SVC(apcc.libsvm_svc.LibSVM_SVC):
    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        degree: int = 3,
        gamma: float = 0.1,
        coef0: float = 0.0,
        tol: float = 1e-3,
        shrinking: bool = True,
        max_iter: int = -1,
    ) -> None:
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.tol = tol
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

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> LibSVM_SVC:

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict_proba(X)


class MLPClassifier(apcc.mlp.MLPClassifier):
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
        self.hidden_layer_depth = hidden_layer_depth
        self.num_nodes_per_layer = num_nodes_per_layer
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.validation_fraction = validation_fraction
        self.tol = tol
        self.solver = solver
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        super().__init__(
            hidden_layer_depth=self.hidden_layer_depth,
            num_nodes_per_layer=self.num_nodes_per_layer,
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
        )

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> MLPClassifier:

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict_proba(X)


class MultinomialNB(apcc.multinomial_nb.MultinomialNB):
    def __init__(
        self,
        alpha: float = 1.0,
        fit_prior: bool = True,
    ) -> None:
        self.alpha = alpha
        self.fit_prior = fit_prior

        super().__init__(
            alpha=self.alpha,
            fit_prior=self.fit_prior,
        )

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> MultinomialNB:

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict_proba(X)


class PassiveAggressive(apcc.passive_aggressive.PassiveAggressive):
    def __init__(
        self,
        C: float = 1.0,
        fit_intercept: bool = True,
        average: bool = False,
        tol: float = 1e-4,
        loss: str = "hinge",
    ) -> None:
        self.C = C
        self.fit_intercept = fit_intercept
        self.average = average
        self.tol = tol
        self.loss = loss

        super().__init__(
            C=self.C,
            fit_intercept=self.fit_intercept,
            average=self.average,
            tol=self.tol,
            loss=self.loss,
        )

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> PassiveAggressive:

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict_proba(X)


class QDA(apcc.qda.QDA):
    def __init__(
        self,
        reg_param: float = 0.0,
    ) -> None:
        self.reg_param = reg_param

        super().__init__(
            reg_param=self.reg_param,
        )

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> QDA:

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict_proba(X)


class RandomForestClassifier(apcc.random_forest.RandomForest):
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
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.bootstrap = bootstrap
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease

        super().__init__(
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

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> RandomForestClassifier:

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict_proba(X)


class SGDClassifier(apcc.sgd.SGD):
    def __init__(
        self,
        loss: str = "log",
        penalty: str = "l2",
        alpha: float = 0.0001,
        fit_intercept: bool = True,
        tol: bool = 1e-4,
        learning_rate: str = "invscaling",
        l1_ratio: float = 0.15,
        epsilon: float = 1e-4,
        eta0: float = 0.01,
        power_t: float = 0.5,
        average: bool = False,
    ) -> None:
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.learning_rate = learning_rate
        self.l1_ratio = l1_ratio
        self.epsilon = epsilon
        self.eta0 = eta0
        self.power_t = power_t
        self.average = average

        super().__init__(
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
        )

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> SGDClassifier:

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict_proba(X)


####################################################################################################################
# regressors


class AdaboostRegressor(apcr.adaboost.AdaboostRegressor):
    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 0.1,
        loss: str = "linear",
        max_depth: int = 1,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.max_depth = max_depth

        super().__init__(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            loss=self.loss,
            max_depth=self.max_depth,
        )

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> AdaboostRegressor:

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        raise NotImplementedError("predict_proba is not implemented for regression.")


class ARDRegression(apcr.ard_regression.ARDRegression):
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
        self.n_iter = n_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.threshold_lambda = threshold_lambda
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

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> ARDRegression:

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        raise NotImplementedError("predict_proba is not implemented for regression.")


class DecisionTreeRegressor(apcr.decision_tree.DecisionTree):
    def __init__(
        self,
        criterion: str = "mse",
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
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease

        super().__init__(
            criterion=self.criterion,
            max_depth_factor=self.max_depth_factor,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
        )

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> DecisionTreeRegressor:

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        raise NotImplementedError("predict_proba is not implemented for regression.")


class ExtraTreesRegressor(apcr.extra_trees.ExtraTreesRegressor):
    def __init__(
        self,
        criterion: str = "mse",
        max_depth: Union[str, int] = "None",
        max_leaf_nodes: Union[str, int] = "None",
        min_samples_leaf: int = 1,
        min_samples_split: int = 2,
        max_features: Union[int, float] = 1,
        bootstrap: bool = False,
        min_weight_fraction_leaf: float = 0.0,
        min_impurity_decrease: float = 0.0,
    ) -> None:
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.min_impurity_decrease = min_impurity_decrease

        super().__init__(
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

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> ExtraTreesRegressor:

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        raise NotImplementedError("predict_proba is not implemented for regression.")


class GaussianProcess(apcr.gaussian_process.GaussianProcess):
    def __init__(
        self,
        alpha: float = 1e-8,
        thetaL: float = 1e-6,
        thetaU: float = 100000.0,
    ) -> None:
        self.alpha = alpha
        self.thetaL = thetaL
        self.thetaU = thetaU

        super().__init__(
            alpha=self.alpha,
            thetaL=self.thetaL,
            thetaU=self.thetaU,
        )

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> GaussianProcess:

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        raise NotImplementedError("predict_proba is not implemented for regression.")


class HistGradientBoostingRegressor(apcr.gradient_boosting.GradientBoosting):
    def __init__(
        self,
        loss: str = "least_squares",
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
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.max_bins = max_bins
        self.l2_regularization = l2_regularization
        self.early_stop = early_stop
        self.tol = tol
        self.scoring = scoring
        self.n_iter_no_change = n_iter_no_change
        self.validation_fraction = validation_fraction

        super().__init__(
            loss=self.loss,
            learning_rate=self.learning_rate,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            max_leaf_nodes=self.max_leaf_nodes,
            max_bins=self.max_bins,
            l2_regularization=self.l2_regularization,
            early_stop=self.early_stop,
            tol=self.tol,
            scoring=self.scoring,
            n_iter_no_change=self.n_iter_no_change,
            validation_fraction=self.validation_fraction,
        )

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> HistGradientBoostingRegressor:

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        raise NotImplementedError("predict_proba is not implemented for regression.")


class KNearestNeighborsRegressor(apcr.k_nearest_neighbors.KNearestNeighborsRegressor):
    def __init__(
        self,
        n_neighbors: int = 1,
        weights: str = "uniform",
        p: int = 2,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p

        super().__init__(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            p=self.p,
        )

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> KNearestNeighborsRegressor:

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        raise NotImplementedError("predict_proba is not implemented for regression.")


class LibLinear_SVR(apcr.liblinear_svr.LibLinear_SVR):
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
        self.epsilon = epsilon
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.C = C
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

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> LibLinear_SVR:

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        raise NotImplementedError("predict_proba is not implemented for regression.")


class LibSVM_SVR(apcr.libsvm_svr.LibSVM_SVR):
    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        epsilon: float = 0.1,
        degree: int = 3,
        gamma: float = 0.1,
        coef0: float = 0,
        tol: float = 1e-3,
        shrinking: bool = True,
        max_iter: int = -1,
    ) -> None:
        self.C = C
        self.kernel = kernel
        self.epsilon = epsilon
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.tol = tol
        self.max_iter = max_iter

        super().__init__(
            C=self.C,
            kernel=self.kernel,
            epsilon=self.epsilon,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            shrinking=self.shrinking,
            tol=self.tol,
            max_iter=self.max_iter,
        )

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> LibSVM_SVR:

        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y)

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        raise NotImplementedError("predict_proba is not implemented for regression.")


class MLPRegressor(apcr.mlp.MLPRegressor):
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
        self.hidden_layer_depth = hidden_layer_depth
        self.num_nodes_per_layer = num_nodes_per_layer
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.validation_fraction = validation_fraction
        self.tol = tol
        self.solver = solver
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        super().__init__(
            hidden_layer_depth=self.hidden_layer_depth,
            num_nodes_per_layer=self.num_nodes_per_layer,
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
        )

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> MLPRegressor:

        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y)

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        raise NotImplementedError("predict_proba is not implemented for regression.")


class RandomForestRegressor(apcr.random_forest.RandomForest):
    def __init__(
        self,
        criterion: str = "mse",
        max_features: float = 1.0,
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
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.bootstrap = bootstrap
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease

        super().__init__(
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

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> RandomForestRegressor:

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        raise NotImplementedError("predict_proba is not implemented for regression.")


class SGDRegressor(apcr.sgd.SGD):
    def __init__(
        self,
        loss: str = "squared_loss",
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
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.learning_rate = learning_rate
        self.l1_ratio = l1_ratio
        self.epsilon = epsilon
        self.eta0 = eta0
        self.power_t = power_t
        self.average = average

        super().__init__(
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
        )

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> SGDRegressor:

        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y)

        super().fit(X, y)

        self._fitted = True

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        return super().predict(X)

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        raise NotImplementedError("predict_proba is not implemented for regression.")
