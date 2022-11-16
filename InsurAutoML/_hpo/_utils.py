"""
File: _utils.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_hpo/_utils.py
File: _utils.py
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Wednesday, 16th November 2022 12:55:53 pm
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
import logging

logger = logging.getLogger(__name__)

import warnings
from typing import Callable, Union, List, Tuple, Dict, Any
import scipy
import numpy as np
import pandas as pd

# from logging import warning
import os
import json
from ray import tune
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from InsurAutoML._utils._data import formatting
from InsurAutoML._utils._file import save_methods
from InsurAutoML._utils._data import train_test_split


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
        if self.feature_selection is not None:
            X = self.feature_selection.transform(X)

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
        if not self.voting in ["hard", "soft"]:
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
            prob_list = np.asarray(
                [
                    pipeline.predict_proba(X[feature_subset])
                    for (name, pipeline), feature_subset in zip(
                        self.estimators, self.features
                    )
                ]
            )
            if self.strategy == "stacking" or self.strategy == "bagging":
                pred = np.argmax(
                    np.average(prob_list, axis=0, weights=self.weights), axis=1
                )
            elif self.strategy == "boosting":
                pred = np.sum(
                    np.average(prob_list, axis=0, weights=self.weights), axis=1
                )

        # make sure all predictions are seen
        if isinstance(pred, pd.DataFrame):
            return super(ClassifierEnsemble, self).refit(pred)
        # if not dataframe, convert to dataframe for formatting
        else:
            return super(ClassifierEnsemble, self).refit(
                pd.DataFrame(pred, columns=self._response)
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

        # remember all unique labels
        super(RegressorEnsemble, self).fit(y)

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
            except:
                # if weights included, but not available in voting function, warn users
                if self.weights is not None:
                    logger.warn("weights are not used in voting method")
                    # warnings.warn("weights are not used in voting method")
                pred = self.voting(pred_list, axis=1)
        elif self.strategy == "boosting":
            pred = np.sum(pred_list, axis=1)

        # make sure all predictions are seen
        if isinstance(pred, pd.DataFrame):
            return super(RegressorEnsemble, self).refit(pred)
        # if not dataframe, convert to dataframe for formatting
        else:
            return super(RegressorEnsemble, self).refit(
                pd.DataFrame(pred, columns=self._response)
            )


class TabularObjective(tune.Trainable):
    def setup(
        self,
        config: Dict,
        _X: pd.DataFrame = None,
        _y: pd.DataFrame = None,
        encoder: Dict[str, Callable] = None,
        imputer: Dict[str, Callable] = None,
        balancing: Dict[str, Callable] = None,
        scaling: Dict[str, Callable] = None,
        feature_selection: Dict[str, Callable] = None,
        models: Dict[str, Callable] = None,
        model_name: str = "model",
        task_mode: str = "classification",
        objective: str = "accuracy",
        validation: bool = True,
        valid_size: float = 0.15,
        full_status: bool = False,
        reset_index: bool = True,
        timeout: int = 36,
        _iter: int = 1,
        seed: int = 1,
    ) -> None:
        # assign hyperparameter arguments
        self.encoder = encoder
        self.imputer = imputer
        self.balancing = balancing
        self.scaling = scaling
        self.feature_selection = feature_selection
        self.models = models

        # assign objective parameters
        self._X = _X
        self._y = _y
        self.model_name = model_name
        self.task_mode = task_mode
        self.objective = objective
        self.validation = validation
        self.valid_size = valid_size
        self.full_status = full_status
        self.reset_index = reset_index
        self.timeout = timeout
        self._iter = _iter
        self.seed = seed

        if isinstance(self._X, pd.DataFrame):
            self.dict2config(config)

    def step(self) -> Dict[str, Any]:

        # try:
        #     self.status_dict = self._objective()
        # except:
        #     warnings.warn("Objective not finished due to timeout.")
        #     if self.full_status:
        #         self.status_dict = {
        #             "encoder": self._encoder,
        #             "encoder_hyperparameter": self._encoder_hyper,
        #             "imputer": self._imputer,
        #             "imputer_hyperparameter": self._imputer_hyper,
        #             "balancing": self._balancing,
        #             "balancing_hyperparameter": self._balancing_hyper,
        #             "scaling": self._scaling,
        #             "scaling_hyperparameter": self._scaling_hyper,
        #             "feature_selection": self._feature_selection,
        #             "feature_selection_hyperparameter": self._feature_selection_hyper,
        #             "model": self._model,
        #             "model_hyperparameter": self._model_hyper,
        #             "training_status": "not fitted",
        #             "status": "TIMEOUT",
        #         }
        #     else:
        #         self.status_dict = {
        #             "training_status": "not fitted",
        #             "status": "TIMEOUT",
        #         }
        self.status_dict = self._objective()

        return self.status_dict

    def reset_config(self, new_config: Dict) -> bool:

        self.dict2config(new_config)

        return True

    # convert dict hyperparameter to actual classes
    def dict2config(self, params: Dict) -> None:

        # pipeline of objective, [encoder, imputer, balancing, scaling, feature_selection, model]
        # select encoder and set hyperparameters
        # make sure only those keys are used

        for key in list(params.keys()):
            if key not in [
                "encoder",
                "imputer",
                "balancing",
                "scaling",
                "feature_selection",
                "model",
                "task_type",
            ]:
                params.pop(key, None)

        # issue 1: https://github.com/PanyiDong/My_AutoML/issues/1
        # HyperOpt hyperparameter space conflicts with ray.tune

        # while setting hyperparameters space,
        # the method name is injected into the hyperparameter space
        # so, before fitting, these indications are remove

        # must have encoder
        self._encoder_hyper = params["encoder"].copy()
        # find corresponding encoder key
        try:
            for key in self._encoder_hyper.keys():
                if "encoder_" in key:
                    _encoder_key = key
                    break
            self._encoder = self._encoder_hyper[_encoder_key]
            del self._encoder_hyper[_encoder_key]
            # remove indications
            self._encoder_hyper = {
                k.replace(self._encoder + "_", ""): self._encoder_hyper[k]
                for k in self._encoder_hyper
            }
        # if not get above format, use default format
        except:
            self._encoder = self._encoder_hyper["encoder"]
            del self._encoder_hyper["encoder"]
        self.enc = self.encoder[self._encoder](**self._encoder_hyper)

        # select imputer and set hyperparameters
        self._imputer_hyper = params["imputer"].copy()
        # find corresponding imputer key
        try:
            for key in self._imputer_hyper.keys():
                if "imputer_" in key:
                    _imputer_key = key
                    break
            self._imputer = self._imputer_hyper[_imputer_key]
            del self._imputer_hyper[_imputer_key]
            # remove indications
            self._imputer_hyper = {
                k.replace(self._imputer + "_", ""): self._imputer_hyper[k]
                for k in self._imputer_hyper
            }
        # if not get above format, use default format
        except:
            self._imputer = self._imputer_hyper["imputer"]
            del self._imputer_hyper["imputer"]
        self.imp = self.imputer[self._imputer](**self._imputer_hyper)

        # select balancing and set hyperparameters
        # must have balancing, since no_preprocessing is included
        self._balancing_hyper = params["balancing"].copy()
        # find corresponding balancing key
        try:
            for key in self._balancing_hyper.keys():
                if "balancing_" in key:
                    _balancing_key = key
                    break
            self._balancing = self._balancing_hyper[_balancing_key]
            del self._balancing_hyper[_balancing_key]
            # remove indications
            self._balancing_hyper = {
                k.replace(self._balancing + "_", ""): self._balancing_hyper[k]
                for k in self._balancing_hyper
            }
        # if not get above format, use default format
        except:
            self._balancing = self._balancing_hyper["balancing"]
            del self._balancing_hyper["balancing"]
        self.blc = self.balancing[self._balancing](**self._balancing_hyper)

        # select scaling and set hyperparameters
        # must have scaling, since no_preprocessing is included
        self._scaling_hyper = params["scaling"].copy()
        # find corresponding scaling key
        try:
            for key in self._scaling_hyper.keys():
                if "scaling_" in key:
                    _scaling_key = key
                    break
            self._scaling = self._scaling_hyper[_scaling_key]
            del self._scaling_hyper[_scaling_key]
            # remove indications
            self._scaling_hyper = {
                k.replace(self._scaling + "_", ""): self._scaling_hyper[k]
                for k in self._scaling_hyper
            }
        # if not get above format, use default format
        except:
            self._scaling = self._scaling_hyper["scaling"]
            del self._scaling_hyper["scaling"]
        self.scl = self.scaling[self._scaling](**self._scaling_hyper)

        # select feature selection and set hyperparameters
        # must have feature selection, since no_preprocessing is included
        self._feature_selection_hyper = params["feature_selection"].copy()
        # find corresponding feature_selection key
        try:
            for key in self._feature_selection_hyper.keys():
                if "feature_selection_" in key:
                    _feature_selection_key = key
                    break
            self._feature_selection = self._feature_selection_hyper[
                _feature_selection_key
            ]
            del self._feature_selection_hyper[_feature_selection_key]
            # remove indications
            self._feature_selection_hyper = {
                k.replace(
                    self._feature_selection + "_", ""
                ): self._feature_selection_hyper[k]
                for k in self._feature_selection_hyper
            }
        # if not get above format, use default format
        except:
            self._feature_selection = self._feature_selection_hyper["feature_selection"]
            del self._feature_selection_hyper["feature_selection"]
        self.fts = self.feature_selection[self._feature_selection](
            **self._feature_selection_hyper
        )

        # select model model and set hyperparameters
        # must have a model
        self._model_hyper = params["model"].copy()
        # find corresponding model key
        try:
            for key in self._model_hyper.keys():
                if "model_" in key:
                    _model_key = key
                    break
            self._model = self._model_hyper[_model_key]
            del self._model_hyper[_model_key]
            # remove indications
            self._model_hyper = {
                k.replace(self._model + "_", ""): self._model_hyper[k]
                for k in self._model_hyper
            }
        # if not get above format, use default format
        except:
            self._model = self._model_hyper["model"]
            del self._model_hyper["model"]
        self.mol = self.models[self._model](
            **self._model_hyper
        )  # call the model using passed parameters

        # obj_tmp_directory = self.temp_directory  # + "/iter_" + str(self._iter + 1)
        # if not os.path.isdir(obj_tmp_directory):
        #     os.makedirs(obj_tmp_directory)

        # with open(obj_tmp_directory + "/hyperparameter_settings.txt", "w") as f:
        # if already exists, use append mode
        # else, write mode
        if not os.path.exists("hyperparameter_settings.txt"):
            write_type = "w"
        else:
            write_type = "a"

        with open("hyperparameter_settings.txt", write_type) as f:
            f.write("Encoding method: {}\n".format(self._encoder))
            f.write("Encoding Hyperparameters:")
            print(self._encoder_hyper, file=f, end="\n\n")
            f.write("Imputation method: {}\n".format(self._imputer))
            f.write("Imputation Hyperparameters:")
            print(self._imputer_hyper, file=f, end="\n\n")
            f.write("Balancing method: {}\n".format(self._balancing))
            f.write("Balancing Hyperparameters:")
            print(self._balancing_hyper, file=f, end="\n\n")
            f.write("Scaling method: {}\n".format(self._scaling))
            f.write("Scaling Hyperparameters:")
            print(self._scaling_hyper, file=f, end="\n\n")
            f.write("Feature Selection method: {}\n".format(self._feature_selection))
            f.write("Feature Selection Hyperparameters:")
            print(self._feature_selection_hyper, file=f, end="\n\n")
            f.write("Model: {}\n".format(self._model))
            f.write("Model Hyperparameters:")
            print(self._model_hyper, file=f, end="\n\n")

    def save_checkpoint(self, tmp_checkpoint_dir: str) -> str:
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "status.json")

        with open(checkpoint_path, "w") as out_f:
            json.dump(self.status_dict, out_f)

        # need to return the path of checkpoints to be further processed
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir: str) -> None:
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "status.json")

        with open(checkpoint_path, "r") as inp_f:
            self.status_dict = json.load(inp_f)

    # # wrapped timeout decorator
    # def wrap_timeout(f):
    #     def wrapper(*args):
    #         timeout(args[0].timeout)
    #         return f(*args)

    #     return wrapper

    # # actual objective function
    # @wrap_timeout
    @ignore_warnings(category=ConvergenceWarning)
    def _objective(
        self,
    ) -> Dict[str, Any]:

        # different evaluation metrics for classification and regression
        # notice: if add metrics that is larger the better, need to add - sign
        # at actual fitting process below (since try to minimize the loss)
        if self.task_mode == "regression":
            # evaluation for predictions
            if self.objective == "MSE":
                from sklearn.metrics import mean_squared_error

                _obj = mean_squared_error
            elif self.objective == "MAE":
                from sklearn.metrics import mean_absolute_error

                _obj = mean_absolute_error
            elif self.objective == "MSLE":
                from sklearn.metrics import mean_squared_log_error

                _obj = mean_squared_log_error
            elif self.objective == "R2":
                from sklearn.metrics import r2_score

                _obj = r2_score
            elif self.objective == "MAX":
                from sklearn.metrics import (
                    max_error,
                )  # focus on reducing extreme losses

                _obj = max_error
            elif isinstance(self.objective, Callable):

                # if callable, use the callable
                _obj = self.objective
            else:
                raise ValueError(
                    'Mode {} only support ["MSE", "MAE", "MSLE", "R2", "MAX", callable], get{}'.format(
                        self.task_mode, self.objective
                    )
                )
        elif self.task_mode == "classification":
            # evaluation for predictions
            if self.objective == "accuracy":
                from sklearn.metrics import accuracy_score

                _obj = accuracy_score
            elif self.objective == "precision":
                from sklearn.metrics import precision_score

                _obj = precision_score
            elif self.objective == "auc":
                from sklearn.metrics import roc_auc_score

                _obj = roc_auc_score
            elif self.objective == "hinge":
                from sklearn.metrics import hinge_loss

                _obj = hinge_loss
            elif self.objective == "f1":
                from sklearn.metrics import f1_score

                _obj = f1_score
            elif isinstance(self.objective, Callable):

                # if callable, use the callable
                _obj = self.objective
            else:
                raise ValueError(
                    'Mode {} only support ["accuracy", "precision", "auc", "hinge", "f1", callable], get{}'.format(
                        self.task_mode, self.objective
                    )
                )

        if self.validation:
            # only perform train_test_split when validation
            # train test split so the performance of model selection and
            # hyperparameter optimization can be evaluated
            X_train, X_test, y_train, y_test = train_test_split(
                self._X, self._y, test_perc=self.valid_size, seed=self.seed
            )

            if self.reset_index:
                # reset index to avoid indexing order error
                X_train.reset_index(drop=True, inplace=True)
                X_test.reset_index(drop=True, inplace=True)
                y_train.reset_index(drop=True, inplace=True)
                y_test.reset_index(drop=True, inplace=True)

            _X_train_obj, _X_test_obj = X_train.copy(), X_test.copy()
            _y_train_obj, _y_test_obj = y_train.copy(), y_test.copy()

            # encoding
            _X_train_obj = self.enc.fit(_X_train_obj)
            _X_test_obj = self.enc.refit(_X_test_obj)
            # with open(obj_tmp_directory + "/objective_process.txt", "w") as f:
            with open("objective_process.txt", "w") as f:
                f.write("Encoding finished, in imputation process.")

            # imputer
            _X_train_obj = self.imp.fill(_X_train_obj)
            _X_test_obj = self.imp.fill(_X_test_obj)
            # with open(obj_tmp_directory + "/objective_process.txt", "w") as f:
            with open("objective_process.txt", "w") as f:
                f.write("Imputation finished, in scaling process.")

            # balancing
            _X_train_obj, _y_train_obj = self.blc.fit_transform(
                _X_train_obj, _y_train_obj
            )
            # with open(obj_tmp_directory + "/objective_process.txt", "w") as f:
            with open("objective_process.txt", "w") as f:
                f.write("Balancing finished, in scaling process.")
            # make sure the classes are integers (belongs to certain classes)
            if self.task_mode == "classification":
                _y_train_obj = _y_train_obj.astype(int)
                _y_test_obj = _y_test_obj.astype(int)

            # scaling
            self.scl.fit(_X_train_obj, _y_train_obj)
            _X_train_obj = self.scl.transform(_X_train_obj)
            _X_test_obj = self.scl.transform(_X_test_obj)
            # with open(obj_tmp_directory + "/objective_process.txt", "w") as f:
            with open("objective_process.txt", "w") as f:
                f.write("Scaling finished, in feature selection process.")

            # feature selection
            self.fts.fit(_X_train_obj, _y_train_obj)
            _X_train_obj = self.fts.transform(_X_train_obj)
            _X_test_obj = self.fts.transform(_X_test_obj)
            # with open(obj_tmp_directory + "/objective_process.txt", "w") as f:
            with open("objective_process.txt", "w") as f:
                f.write(
                    "Feature selection finished, in {} model.".format(self.task_mode)
                )

            # fit model
            if scipy.sparse.issparse(_X_train_obj):  # check if returns sparse matrix
                _X_train_obj = _X_train_obj.toarray()
            if scipy.sparse.issparse(_X_test_obj):
                _X_test_obj = _X_test_obj.toarray()

            # store the preprocessed train/test datasets
            if isinstance(_X_train_obj, np.ndarray):  # in case numpy array is returned
                pd.concat(
                    [pd.DataFrame(_X_train_obj), _y_train_obj],
                    axis=1,
                    ignore_index=True,
                ).to_csv("train_preprocessed.csv", index=False)
            elif isinstance(_X_train_obj, pd.DataFrame):
                pd.concat([_X_train_obj, _y_train_obj], axis=1).to_csv(
                    "train_preprocessed.csv", index=False
                )
            else:
                raise TypeError("Only accept numpy array or pandas dataframe!")

            if isinstance(_X_test_obj, np.ndarray):
                pd.concat(
                    [pd.DataFrame(_X_test_obj), _y_test_obj],
                    axis=1,
                    ignore_index=True,
                ).to_csv("test_preprocessed.csv", index=False)
            elif isinstance(_X_test_obj, pd.DataFrame):
                pd.concat([_X_test_obj, _y_test_obj], axis=1).to_csv(
                    "test_preprocessed.csv", index=False
                )
            else:
                raise TypeError("Only accept numpy array or pandas dataframe!")

            self.mol.fit(_X_train_obj, _y_train_obj.values.ravel())
            os.remove("objective_process.txt")

            y_pred = self.mol.predict(_X_test_obj)
            if self.objective in [
                "R2",
                "accuracy",
                "precision",
                "auc",
                "hinge",
                "f1",
            ]:
                # special treatment for ["R2", "accuracy", "precision", "auc", "hinge", "f1"]
                # larger the better, since to minimize, add negative sign
                _loss = -_obj(_y_test_obj.values, y_pred)
            else:
                _loss = _obj(_y_test_obj.values, y_pred)

            # save the fitted model objects
            save_methods(
                self.model_name,
                [self.enc, self.imp, self.blc, self.scl, self.fts, self.mol],
            )

            with open("testing_objective.txt", "w") as f:
                f.write("Loss from objective function is: {:.6f}\n".format(_loss))
                f.write("Loss is calculate using {}.".format(self.objective))
            self._iter += 1

            # since we tries to minimize the objective function, take negative accuracy here
            if self.full_status:
                # tune.report(
                #     encoder=_encoder,
                #     encoder_hyperparameter=_encoder_hyper,
                #     imputer=_imputer,
                #     imputer_hyperparameter=_imputer_hyper,
                #     balancing=_balancing,
                #     balancing_hyperparameter=_balancing_hyper,
                #     scaling=_scaling,
                #     scaling_hyperparameter=_scaling_hyper,
                #     feature_selection=_feature_selection,
                #     feature_selection_hyperparameter=_feature_selection_hyper,
                #     model=_model,
                #     model_hyperparameter=_model_hyper,
                #     fitted_model=_model,
                #     training_status="fitted",
                #     loss=_loss,
                # )
                # only for possible checks
                return {
                    "encoder": self._encoder,
                    "encoder_hyperparameter": self._encoder_hyper,
                    "imputer": self._imputer,
                    "imputer_hyperparameter": self._imputer_hyper,
                    "balancing": self._balancing,
                    "balancing_hyperparameter": self._balancing_hyper,
                    "scaling": self._scaling,
                    "scaling_hyperparameter": self._scaling_hyper,
                    "feature_selection": self._feature_selection,
                    "feature_selection_hyperparameter": self._feature_selection_hyper,
                    "model": self._model,
                    "model_hyperparameter": self._model_hyper,
                    "fitted_model": self._model,
                    "training_status": "fitted",
                    "loss": _loss,
                }
            else:
                # tune.report(
                #     fitted_model=_model,
                #     training_status="fitted",
                #     loss=_loss,
                # )
                # only for possible checks
                return {
                    "fitted_model": self._model,
                    "training_status": "fitted",
                    "loss": _loss,
                }
        else:
            _X_obj = self._X.copy()
            _y_obj = self._y.copy()

            # encoding
            _X_obj = self.enc.fit(_X_obj)
            # with open(obj_tmp_directory + "/objective_process.txt", "w") as f:
            with open("objective_process.txt", "w") as f:
                f.write("Encoding finished, in imputation process.")

            # imputer
            _X_obj = self.imp.fill(_X_obj)
            # with open(obj_tmp_directory + "/objective_process.txt", "w") as f:
            with open("objective_process.txt", "w") as f:
                f.write("Imputation finished, in scaling process.")

            # balancing
            _X_obj, _y_obj = self.blc.fit_transform(_X_obj, _y_obj)
            # with open(obj_tmp_directory + "/objective_process.txt", "w") as f:
            with open("objective_process.txt", "w") as f:
                f.write("Balancing finished, in feature selection process.")

            # scaling
            self.scl.fit(_X_obj, _y_obj)
            _X_obj = self.scl.transform(_X_obj)
            # with open(obj_tmp_directory + "/objective_process.txt", "w") as f:
            with open("objective_process.txt", "w") as f:
                f.write("Scaling finished, in balancing process.")

            # feature selection
            self.fts.fit(_X_obj, _y_obj)
            _X_obj = self.fts.transform(_X_obj)
            # with open(obj_tmp_directory + "/objective_process.txt", "w") as f:
            with open("objective_process.txt", "w") as f:
                f.write(
                    "Feature selection finished, in {} model.".format(self.task_mode)
                )

            # fit model
            if scipy.sparse.issparse(_X_obj):  # check if returns sparse matrix
                _X_obj = _X_obj.toarray()

            # store the preprocessed train/test datasets
            if isinstance(_X_obj, np.ndarray):  # in case numpy array is returned
                pd.concat(
                    [pd.DataFrame(_X_obj), _y_obj],
                    axis=1,
                    ignore_index=True,
                ).to_csv("train_preprocessed.csv", index=False)
            elif isinstance(_X_obj, pd.DataFrame):
                pd.concat([_X_obj, _y_obj], axis=1).to_csv(
                    "train_preprocessed.csv", index=False
                )
            else:
                raise TypeError("Only accept numpy array or pandas dataframe!")

            self.mol.fit(_X_obj, _y_obj.values.ravel())
            os.remove("objective_process.txt")

            y_pred = self.mol.predict(_X_obj)

            if self.objective in [
                "R2",
                "accuracy",
                "precision",
                "auc",
                "hinge",
                "f1",
            ]:
                # special treatment for ["R2", "accuracy", "precision", "auc", "hinge", "f1"]
                # larger the better, since to minimize, add negative sign
                _loss = -_obj(_y_obj.values, y_pred)
            else:
                _loss = _obj(_y_obj.values, y_pred)

            # save the fitted model objects
            save_methods(
                self.model_name,
                [self.enc, self.imp, self.blc, self.scl, self.fts, self.mol],
            )

            # with open(obj_tmp_directory + "/testing_objective.txt", "w") as f:
            with open("testing_objective.txt", "w") as f:
                f.write("Loss from objective function is: {:.6f}\n".format(_loss))
                f.write("Loss is calculate using {}.".format(self.objective))
            self._iter += 1

            if self.full_status:
                # tune.report(
                #     encoder=_encoder,
                #     encoder_hyperparameter=_encoder_hyper,
                #     imputer=_imputer,
                #     imputer_hyperparameter=_imputer_hyper,
                #     balancing=_balancing,
                #     balancing_hyperparameter=_balancing_hyper,
                #     scaling=_scaling,
                #     scaling_hyperparameter=_scaling_hyper,
                #     feature_selection=_feature_selection,
                #     feature_selection_hyperparameter=_feature_selection_hyper,
                #     model=_model,
                #     model_hyperparameter=_model_hyper,
                #     fitted_model=_model,
                #     training_status="fitted",
                #     loss=_loss,
                # )
                # only for possible checks
                return {
                    "encoder": self._encoder,
                    "encoder_hyperparameter": self._encoder_hyper,
                    "imputer": self._imputer,
                    "imputer_hyperparameter": self._imputer_hyper,
                    "balancing": self._balancing,
                    "balancing_hyperparameter": self._balancing_hyper,
                    "scaling": self._scaling,
                    "scaling_hyperparameter": self._scaling_hyper,
                    "feature_selection": self._feature_selection,
                    "feature_selection_hyperparameter": self._feature_selection_hyper,
                    "model": self._model,
                    "model_hyperparameter": self._model_hyper,
                    "fitted_model": self._model,
                    "training_status": "fitted",
                    "loss": _loss,
                }
            else:
                # tune.report(
                #     fitted_model=_model,
                #     training_status="fitted",
                #     loss=_loss,
                # )
                # only for possible checks
                return {
                    "fitted_model": self._model,
                    "training_status": "fitted",
                    "loss": _loss,
                }
