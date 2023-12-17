"""
File Name: ensemble.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.5
Relative Path: /InsurAutoML/hpo/ensemble.py
File Created: Friday, 1st December 2023 6:43:44 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Saturday, 16th December 2023 7:42:26 pm
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
from typing import Callable, Union, List, Tuple
import json
import os
import time
import scipy
import logging
import pandas as pd
import numpy as np

from ..utils.data import formatting

logger = logging.getLogger(__name__)


class Pipeline:

    """ "
    A pipeline of entire AutoML process.
    """

    def __init__(
        self,
        encoder: Callable = None,
        imputer: Callable = None,
        balancing: Callable = None,
        scaling: Callable = None,
        feature_selection: Callable = None,
        model: Callable = None,
    ) -> None:
        self.encoder = encoder
        self.imputer = imputer
        self.balancing = balancing
        self.scaling = scaling
        self.feature_selection = feature_selection
        self.model = model

        self._fitted = False  # whether the pipeline is fitted

    def fit(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series] = None
    ) -> Pipeline:
        # loop all components, make sure they are fitted
        # if they are not fitted, fit them
        if self.encoder is not None:
            if self.encoder._fitted:
                pass
            else:
                X = self.encoder.fit(X)
        if self.imputer is not None:
            if self.imputer._fitted:
                pass
            else:
                X = self.imputer.fill(X)
        if self.balancing is not None:
            if self.balancing._fitted:
                pass
            else:
                X, y = self.balancing.fit_transform(X, y)
        if self.scaling is not None:
            if self.scaling._fitted:
                pass
            else:
                self.scaling.fit(X, y)
                X = self.scaling.transform(X)
        if self.feature_selection is not None:
            if self.feature_selection._fitted:
                pass
            else:
                self.feature_selection.fit(X, y)
                X = self.feature_selection.transform(X)

        if scipy.sparse.issparse(X):  # check if returns sparse matrix
            X = X.toarray()

        if self.model is None:
            raise ValueError("model is not defined!")
        if self.model._fitted:
            pass
        else:
            self.model.fit(X, y)

        self._fitted = True

        return self

    def predict(self, X: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        if not self._fitted:
            raise ValueError("Pipeline is not fitted!")

        if self.encoder is not None:
            X = self.encoder.refit(X)
        if self.imputer is not None:
            X = self.imputer.fill(X)
        # no need for balancing
        if self.scaling is not None:
            X = self.scaling.transform(X)

        # unify the features order
        if hasattr(self.feature_selection, "feature_names_in_"):
            if not (self.feature_selection.feature_names_in_ == X.columns).all():
                X = X[self.feature_selection.feature_names_in_]

        if self.feature_selection is not None:
            X = self.feature_selection.transform(X)

        return self.model.predict(X)

    def predict_proba(
        self, X: pd.DataFrame
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        if not self._fitted:
            raise ValueError("Pipeline is not fitted!")

        if self.encoder is not None:
            X = self.encoder.refit(X)
        if self.imputer is not None:
            X = self.imputer.fill(X)
        # no need for balancing
        if self.scaling is not None:
            X = self.scaling.transform(X)

        # unify the features order
        if hasattr(self.feature_selection, "feature_names_in_"):
            if not (self.feature_selection.feature_names_in_ == X.columns).all():
                X = X[self.feature_selection.feature_names_in_]

        if self.feature_selection is not None:
            X = self.feature_selection.transform(X)

        if not hasattr(self.model, "predict_proba"):
            logger.error("model does not have predict_proba method!")

        return self.model.predict_proba(X)


# ensemble methods:
# 1. Stacking Ensemble
# 2. Boosting Ensemble
# 3. Bagging Ensemble


class ClassifierEnsemble(formatting):

    """
    Ensemble of classifiers for classification.
    """

    def __init__(
        self,
        estimators: List[Tuple[str, Pipeline]],
        voting: str = "hard",
        weights: List[float] = None,
        features: List[str] = [],
        strategy: str = "stacking",
    ) -> None:
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.features = features
        self.strategy = strategy

        # initialize the formatting
        super(ClassifierEnsemble, self).__init__(
            inplace=False,
        )

        self._fitted = False

    def fit(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> ClassifierEnsemble:
        # check for voting type
        if self.voting not in ["hard", "soft"]:
            raise ValueError("voting must be either 'hard' or 'soft'")

        # format the weights
        self.weights = (
            [w for est, w in zip(self.estimators, self.weights)]
            if self.weights is not None
            else None
        )

        # if bagging, features much be provided
        if self.strategy == "bagging" and len(self.features) == 0:
            raise ValueError("features must be provided for bagging ensemble")

        # initialize the feature list if not given
        # by full feature list
        if len(self.features) == 0:
            self.features = [X.columns for _ in range(len(self.estimators))]

        # remember the name of response
        if isinstance(y, pd.Series):
            self._response = [y.name]
        elif isinstance(y, pd.DataFrame):
            self._response = list(y.columns)
        elif isinstance(y, np.ndarray):
            y = pd.DataFrame(y, columns=["response"])
            self._response = ["response"]

        # remember all unique labels
        super(ClassifierEnsemble, self).fit(y)

        # check for estimators type
        if not isinstance(self.estimators, list):
            raise TypeError("estimators must be a list")
        for item, feature_subset in zip(self.estimators, self.features):
            if not isinstance(item, tuple):
                raise TypeError("estimators must be a list of tuples.")
            if not isinstance(item[1], Pipeline):
                raise TypeError(
                    "estimators must be a list of tuples of (name, Pipeline)."
                )

            # make sure all estimators are fitted
            if not item[1]._fitted:
                item[1].fit(X[feature_subset], y)

        self._fitted = True

        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise ValueError("Ensemble is not fitted!")

        if self.voting == "hard":
            # calculate predictions for all pipelines
            pred_list = np.asarray(
                [
                    pipeline.predict(X[feature_subset])
                    for (name, pipeline), feature_subset in zip(
                        self.estimators, self.features
                    )
                ]
            ).T
            # if larger than 2d, take until get 2d array
            while True:
                if len(pred_list.shape) > 2:
                    pred_list = pred_list[0]
                else:
                    break

            if self.strategy == "stacking" or self.strategy == "bagging":
                pred = np.apply_along_axis(
                    lambda x: np.argmax(np.bincount(x, weights=self.weights)),
                    axis=1,
                    arr=pred_list,
                )
            elif self.strategy == "boosting":
                pred = np.apply_along_axis(
                    lambda x: np.sum(np.bincount(x, weights=self.weights)),
                    axis=1,
                    arr=pred_list,
                )
        elif self.voting == "soft":
            # calculate probabilities for all pipelines
            prob_list = []
            for (name, pipeline), feature_subset in zip(self.estimators, self.features):
                try:
                    prob_list.append(pipeline.predict_proba(X[feature_subset]))
                except Exception as e:
                    logger.warning(
                        "Pipeline {} has problem {}. Ignoring.".format(e, name)
                    )
            prob_list = np.asarray(prob_list)
            if self.strategy == "stacking" or self.strategy == "bagging":
                pred = np.argmax(
                    np.average(prob_list, axis=0, weights=self.weights), axis=1
                )
            elif self.strategy == "boosting":
                pred = np.sum(np.average(prob_list, axis=0), axis=1)

        # make sure all predictions are seen
        if isinstance(pred, pd.DataFrame):
            return super(ClassifierEnsemble, self).refit(pred)
        # if not dataframe, convert to dataframe for formatting
        else:
            return super(ClassifierEnsemble, self).refit(
                pd.DataFrame(pred, columns=self._response)
            )

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise ValueError("Ensemble is not fitted!")

        # certain hyperparameters are not supported for predict_proba
        # ignore those pipelines
        prob_list = []
        for (name, pipeline), feature_subset in zip(self.estimators, self.features):
            try:
                prob_list.append(pipeline.predict_proba(X[feature_subset]))
            except:
                logger.warning(
                    "Pipeline {} does not support predict_proba. Ignoring.".format(name)
                )

        # if no pipeline supports predict_proba, raise error
        if len(prob_list) == 0:
            raise ValueError("No pipeline supports predict_proba. Aborting.")

        # calculate probabilities for all pipelines
        prob_list = np.asarray(prob_list)

        pred = np.average(prob_list, axis=0, weights=self.weights)

        # ignore formatting for probabilities
        if isinstance(pred, pd.DataFrame):
            return pred
        # if not dataframe, convert to dataframe
        else:
            return pd.DataFrame(
                pred, columns=["class_{}".format(i) for i in range(pred.shape[1])]
            )


class RegressorEnsemble(formatting):

    """
    Ensemble of regressors for regression.
    """

    def __init__(
        self,
        estimators: List[Tuple[str, Pipeline]],
        voting: str = "mean",
        weights: List[float] = None,
        features: List[str] = [],
        strategy: str = "stacking",
    ) -> None:
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.features = features
        self.strategy = strategy

        # initialize the formatting
        super(RegressorEnsemble, self).__init__(
            inplace=False,
        )

        self._fitted = False

        self._voting_methods = {
            "mean": np.average,
            "median": np.median,
            "max": np.max,
            "min": np.min,
        }

    def fit(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> RegressorEnsemble:
        # check for voting type
        if self.voting in ["mean", "median", "max", "min"]:
            self.voting = self._voting_methods[self.voting]
        elif isinstance(self.voting, Callable):
            self.voting = self.voting
        else:
            raise ValueError(
                "voting must be either 'mean', 'median', 'max', 'min' or a callable"
            )

        # format the weights
        self.weights = (
            [w for est, w in zip(self.estimators, self.weights)]
            if self.weights is not None
            else None
        )

        # remember the name of response
        if isinstance(y, pd.Series):
            self._response = [y.name]
        elif isinstance(y, pd.DataFrame):
            self._response = list(y.columns)
        elif isinstance(y, np.ndarray):
            y = pd.DataFrame(y, columns=["response"])
            self._response = ["response"]

        # if bagging, features much be provided
        if self.strategy == "bagging" and len(self.features) == 0:
            raise ValueError("features must be provided for bagging ensemble")

        # initialize the feature list if not given
        # by full feature list
        if len(self.features) == 0:
            self.features = [X.columns for _ in range(len(self.estimators))]

        # check for estimators type
        if not isinstance(self.estimators, list):
            raise TypeError("estimators must be a list")
        for item, feature_subset in zip(self.estimators, self.features):
            if not isinstance(item, tuple):
                raise TypeError("estimators must be a list of tuples.")
            if not isinstance(item[1], Pipeline):
                raise TypeError(
                    "estimators must be a list of tuples of (name, Pipeline)."
                )

            # make sure all estimators are fitted
            if not item[1]._fitted:
                item[1].fit(X[feature_subset], y)

        self._fitted = True

        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise ValueError("Ensemble is not fitted!")

        # calculate predictions for all pipelines
        pred_list = np.asarray(
            [
                pipeline.predict(X[feature_subset]).flatten()
                for (name, pipeline), feature_subset in zip(
                    self.estimators, self.features
                )
            ]
        ).T
        # if larger than 2d, take until get 2d array
        while True:
            if len(pred_list.shape) > 2:
                pred_list = pred_list[0]
            else:
                break

        if self.strategy == "stacking" or self.strategy == "bagging":
            # if weights not included, not use weights
            try:
                pred = self.voting(pred_list, axis=1, weights=self.weights)
            except BaseException:
                # if weights included, but not available in voting function,
                # warn users
                if self.weights is not None:
                    logger.warning("weights are not used in voting method")
                    # warnings.warn("weights are not used in voting method")
                pred = self.voting(pred_list, axis=1)
        elif self.strategy == "boosting":
            pred = np.sum(pred_list, axis=1)

        return (
            pred
            if isinstance(pred, pd.DataFrame)
            else pd.DataFrame(pred, columns=self._response)
        )

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError(
            "predict_proba is not implemented for RegressorEnsemble"
        )
