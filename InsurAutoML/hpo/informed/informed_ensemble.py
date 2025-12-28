"""
File Name: informed_ensemble.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Actuarial and Risk Management Sciences, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.6
Relative Path: /InsurAutoML/hpo/informed/informed_ensemble.py
File Created: Thursday, 4th December 2025 10:09:32 am
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Friday, 19th December 2025 3:12:07 pm
Modified By: Panyi Dong (panyid2@illinois.edu)

-----
MIT License

Copyright (c) 2025, Panyi Dong

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
import scipy
import logging
import pandas as pd
import numpy as np

from ...utils.data import formatting

logger = logging.getLogger(__name__)


class InformedPipeline:
    """ "
    A pipeline of entire AutoML process.
    """

    def __init__(
        self,
        encoder: Callable = None,
        complete_prep: Callable = None,
        missing_prep: Callable = None,
        model_complete: Callable = None,
        model_missing: Callable = None,
    ) -> None:
        self.encoder = encoder
        self.complete_prep = complete_prep
        self.missing_prep = missing_prep
        self.model_complete = model_complete
        self.model_missing = model_missing

        self._fitted = False  # whether the pipeline is fitted

    def fit(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series] = None
    ) -> InformedPipeline:
        # loop all components, make sure they are fitted
        if self.encoder is not None and not self.encoder._fitted:
            raise ValueError("encoder is not fitted!")

        if self.complete_prep is not None and not self.complete_prep._fitted:
            raise ValueError("complete_prep is not fitted!")

        if self.missing_prep is not None and not self.missing_prep._fitted:
            raise ValueError("missing_prep is not fitted!")

        if scipy.sparse.issparse(X):  # check if returns sparse matrix
            X = X.toarray()

        if self.model_complete is None or self.model_missing is None:
            raise ValueError("model_complete or model_missing is not defined!")
        if not self.model_complete._fitted or not self.model_missing._fitted:
            raise ValueError("model_complete or model_missing is not fitted!")

        self._fitted = True

        return self

    def predict(self, X: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        if not self._fitted:
            raise ValueError("Pipeline is not fitted!")

        if self.encoder is not None:
            X = self.encoder.refit(X)

        # transform the data
        X_complete, _ = self.complete_prep.transform(X, None)
        X_missing, _ = self.missing_prep.transform(X, None)
        # get the predictions
        y_pred_complete = self.model_complete.predict(X_complete)
        y_pred_missing = self.model_missing.predict(X_missing)
        # prediction resets index, thus set to original index
        complete_index = [
            i for i, idx in enumerate(X_missing.index) if idx in X_complete.index
        ]
        # use y_pred_missing as base predictions and use y_pred_complete to update the complete cases
        y_pred = y_pred_missing.copy()
        y_pred[complete_index] += y_pred_complete
        # average the two predictions for complete cases
        y_pred[complete_index] /= 2

        return y_pred

    def predict_proba(
        self, X: pd.DataFrame
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        if not self._fitted:
            raise ValueError("Pipeline is not fitted!")

        if self.encoder is not None:
            X = self.encoder.refit(X)

        # transform the data
        X_complete, _ = self.complete_prep.transform(X, None)
        X_missing, _ = self.missing_prep.transform(X, None)

        if not hasattr(self.model_complete, "predict_proba") or not hasattr(
            self.model_missing, "predict_proba"
        ):
            logger.error("model does not have predict_proba method!")

        # get the probabilities
        y_pred_complete = self.model_complete.predict_proba(X_complete)
        y_pred_missing = self.model_missing.predict_proba(X_missing)
        # prediction resets index, thus set to original index
        complete_index = [
            i for i, idx in enumerate(X_missing.index) if idx in X_complete.index
        ]
        # use y_pred_missing as base predictions and use y_pred_complete to update the complete cases
        y_pred = y_pred_missing.copy()
        y_pred[complete_index] += y_pred_complete
        # average the two predictions for complete cases
        y_pred[complete_index] /= 2

        return y_pred


# ensemble methods:
# 1. Stacking Ensemble
# 2. Boosting Ensemble
# 3. Bagging Ensemble


class InformedClassifierEnsemble(formatting):
    """
    Ensemble of classifiers for classification.
    """

    def __init__(
        self,
        estimators: List[Tuple[str, InformedPipeline]],
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
        super(InformedClassifierEnsemble, self).__init__(
            inplace=False,
        )

        self._fitted = False

    def fit(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> InformedClassifierEnsemble:
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
        super(InformedClassifierEnsemble, self).fit(y)

        # check for estimators type
        if not isinstance(self.estimators, list):
            raise TypeError("estimators must be a list")
        for item, feature_subset in zip(self.estimators, self.features):
            if not isinstance(item, tuple):
                raise TypeError("estimators must be a list of tuples.")
            if not isinstance(item[1], InformedPipeline):
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
            # round predictions to nearest integers
            pred_list = np.asarray(
                [
                    pipeline.predict(X[feature_subset]).round().astype(int)
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
            return super(InformedClassifierEnsemble, self).refit(pred)
        # if not dataframe, convert to dataframe for formatting
        else:
            return super(InformedClassifierEnsemble, self).refit(
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


class InformedRegressorEnsemble(formatting):
    """
    Ensemble of regressors for regression.
    """

    def __init__(
        self,
        estimators: List[Tuple[str, InformedPipeline]],
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
        super(InformedRegressorEnsemble, self).__init__(
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
    ) -> InformedRegressorEnsemble:
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
            if not isinstance(item[1], InformedPipeline):
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
