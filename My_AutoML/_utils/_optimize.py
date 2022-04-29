"""
File: _optimize.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Latest Version: 0.2.0
Relative Path: /My_AutoML/_utils/_optimize.py
File Created: Friday, 8th April 2022 11:55:13 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Friday, 29th April 2022 10:27:29 am
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

from logging import warning
import os
import warnings
from inspect import isclass
import json
import copy
import time
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from ray import tune
from ray.tune import Stopper
import importlib
from typing import Callable

import scipy
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

# from wrapt_timeout_decorator import *

from My_AutoML._utils._base import (
    has_method,
)
from My_AutoML._utils._file import (
    save_methods,
)
from My_AutoML._utils._data import (
    train_test_split,
)


class TabularObjective(tune.Trainable):
    def setup(
        self,
        config,
        _X=None,
        _y=None,
        encoder=None,
        imputer=None,
        balancing=None,
        scaling=None,
        feature_selection=None,
        models=None,
        model_name="model",
        task_mode="classification",
        objective="accuracy",
        validation=True,
        valid_size=0.15,
        full_status=False,
        reset_index=True,
        timeout=36,
        _iter=1,
        seed=1,
    ):
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

    def step(self):

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

    def reset_config(self, new_config):

        self.dict2config(new_config)

        return True

    # convert dict hyperparameter to actual classes
    def dict2config(self, params):

        # pipeline of objective, [encoder, imputer, balancing, scaling, feature_selection, model]
        # select encoder and set hyperparameters

        # issue 1: https://github.com/PanyiDong/My_AutoML/issues/1
        # HyperOpt hyperparameter space conflicts with ray.tune

        # while setting hyperparameters space,
        # the method name is injected into the hyperparameter space
        # so, before fitting, these indications are removed

        # must have encoder
        self._encoder_hyper = params["encoder"].copy()
        # find corresponding encoder key
        for key in self._encoder_hyper.keys():
            if "encoder_" in key:
                _encoder_key = key
                break
        self._encoder = self._encoder_hyper[_encoder_key]
        del self._encoder_hyper[_encoder_key]
        # remvoe indcations
        self._encoder_hyper = {
            k.replace(self._encoder + "_", ""): self._encoder_hyper[k]
            for k in self._encoder_hyper
        }
        self.enc = self.encoder[self._encoder](**self._encoder_hyper)

        # select imputer and set hyperparameters
        self._imputer_hyper = params["imputer"].copy()
        # find corresponding imputer key
        for key in self._imputer_hyper.keys():
            if "imputer_" in key:
                _imputer_key = key
                break
        self._imputer = self._imputer_hyper[_imputer_key]
        del self._imputer_hyper[_imputer_key]
        # remvoe indcations
        self._imputer_hyper = {
            k.replace(self._imputer + "_", ""): self._imputer_hyper[k]
            for k in self._imputer_hyper
        }
        self.imp = self.imputer[self._imputer](**self._imputer_hyper)

        # select balancing and set hyperparameters
        # must have balancing, since no_preprocessing is included
        self._balancing_hyper = params["balancing"].copy()
        # find corresponding balancing key
        for key in self._balancing_hyper.keys():
            if "balancing_" in key:
                _balancing_key = key
                break
        self._balancing = self._balancing_hyper[_balancing_key]
        del self._balancing_hyper[_balancing_key]
        # remvoe indcations
        self._balancing_hyper = {
            k.replace(self._balancing + "_", ""): self._balancing_hyper[k]
            for k in self._balancing_hyper
        }
        self.blc = self.balancing[self._balancing](**self._balancing_hyper)

        # select scaling and set hyperparameters
        # must have scaling, since no_preprocessing is included
        self._scaling_hyper = params["scaling"].copy()
        # find corresponding scaling key
        for key in self._scaling_hyper.keys():
            if "scaling_" in key:
                _scaling_key = key
                break
        self._scaling = self._scaling_hyper[_scaling_key]
        del self._scaling_hyper[_scaling_key]
        # remvoe indcations
        self._scaling_hyper = {
            k.replace(self._scaling + "_", ""): self._scaling_hyper[k]
            for k in self._scaling_hyper
        }
        self.scl = self.scaling[self._scaling](**self._scaling_hyper)

        # select feature selection and set hyperparameters
        # must have feature selection, since no_preprocessing is included
        self._feature_selection_hyper = params["feature_selection"].copy()
        # find corresponding feature_selection key
        for key in self._feature_selection_hyper.keys():
            if "feature_selection_" in key:
                _feature_selection_key = key
                break
        self._feature_selection = self._feature_selection_hyper[_feature_selection_key]
        del self._feature_selection_hyper[_feature_selection_key]
        # remvoe indcations
        self._feature_selection_hyper = {
            k.replace(self._feature_selection + "_", ""): self._feature_selection_hyper[
                k
            ]
            for k in self._feature_selection_hyper
        }
        self.fts = self.feature_selection[self._feature_selection](
            **self._feature_selection_hyper
        )

        # select model model and set hyperparameters
        # must have a model
        self._model_hyper = params["model"].copy()
        # find corresponding model key
        for key in self._model_hyper.keys():
            if "model_" in key:
                _model_key = key
                break
        self._model = self._model_hyper[_model_key]
        del self._model_hyper[_model_key]
        # remvoe indcations
        self._model_hyper = {
            k.replace(self._model + "_", ""): self._model_hyper[k]
            for k in self._model_hyper
        }
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

    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "status.json")

        with open(checkpoint_path, "w") as out_f:
            json.dump(self.status_dict, out_f)

        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
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
    ):

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
            _X_obj = self.blc.fit_transform(_X_obj)
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


# create hyperparameter space using ray.tune.choice
# the pipeline of AutoClassifier is [encoder, imputer, scaling, balancing, feature_selection, model]
# only chosen ones will be added to hyperparameter space
def _get_hyperparameter_space(
    X,
    encoders_hyperparameters,
    encoder,
    imputers_hyperparameters,
    imputer,
    balancings_hyperparameters,
    balancing,
    scalings_hyperparameters,
    scaling,
    feature_selection_hyperparameters,
    feature_selection,
    models_hyperparameters,
    models,
    task_mode,
):

    # encoding space
    _encoding_hyperparameter = []
    for _encoder in [*encoder]:
        for item in encoders_hyperparameters:  # search the encoders' hyperparameters
            # find encoder key
            for _key in item.keys():
                if "encoder_" in _key:
                    _encoder_key = _key
                    break
            if item[_encoder_key] == _encoder:
                # create a copy of hyperparameters, avoid changes on original
                _item = copy.deepcopy(item)
                # convert string to tune.choice
                _item[_encoder_key] = tune.choice([_item[_encoder_key]])
                _encoding_hyperparameter.append(_item)
                break

    # raise error if no encoding hyperparameters are found
    if len(_encoding_hyperparameter) == 0:
        raise ValueError(
            "No encoding hyperparameters are found. Please check your encoders."
        )

    _encoding_hyperparameter = tune.choice(_encoding_hyperparameter)

    # imputation space
    _imputer_hyperparameter = []
    if not X.isnull().values.any():  # if no missing, no need for imputation
        _imputer_hyperparameter = tune.choice(
            [{"imputer_0": tune.choice(["no_processing"])}]
        )
    else:
        for _imputer in [*imputer]:
            for item in imputers_hyperparameters:  # search the imputer' hyperparameters
                # find imputer key
                for _key in item.keys():
                    if "imputer_" in _key:
                        _imputer_key = _key
                        break
                if item[_imputer_key] == _imputer:
                    # create a copy of hyperparameters, avoid changes on original
                    _item = copy.deepcopy(item)
                    # convert string to tune.choice
                    _item[_imputer_key] = tune.choice([_item[_imputer_key]])
                    _imputer_hyperparameter.append(_item)
                    break

        # raise error if no imputation hyperparameters are found
        if len(_imputer_hyperparameter) == 0:
            raise ValueError(
                "No imputation hyperparameters are found. Please check your imputers."
            )

        _imputer_hyperparameter = tune.choice(_imputer_hyperparameter)

    # balancing space
    _balancing_hyperparameter = []
    for _balancing in [*balancing]:
        for (
            item
        ) in balancings_hyperparameters:  # search the balancings' hyperparameters
            # find balancing key
            for _key in item.keys():
                if "balancing_" in _key:
                    _balancing_key = _key
                    break
            if item[_balancing_key] == _balancing:
                # create a copy of hyperparameters, avoid changes on original
                _item = copy.deepcopy(item)
                # convert string to tune.choice
                _item[_balancing_key] = tune.choice([_item[_balancing_key]])
                _balancing_hyperparameter.append(_item)
                break

    # raise error if no balancing hyperparameters are found
    if len(_balancing_hyperparameter) == 0:
        raise ValueError(
            "No balancing hyperparameters are found. Please check your balancings."
        )

    _balancing_hyperparameter = tune.choice(_balancing_hyperparameter)

    # scaling space
    _scaling_hyperparameter = []
    for _scaling in [*scaling]:
        for item in scalings_hyperparameters:  # search the scalings' hyperparameters
            # find scaling key
            for _key in item.keys():
                if "scaling_" in _key:
                    _scaling_key = _key
                    break
            if item[_scaling_key] == _scaling:
                # create a copy of hyperparameters, avoid changes on original
                _item = copy.deepcopy(item)
                # convert string to tune.choice
                _item[_scaling_key] = tune.choice([_item[_scaling_key]])
                _scaling_hyperparameter.append(_item)
                break

    # raise error if no scaling hyperparameters are found
    if len(_scaling_hyperparameter) == 0:
        raise ValueError(
            "No scaling hyperparameters are found. Please check your scalings."
        )

    _scaling_hyperparameter = tune.choice(_scaling_hyperparameter)

    # feature selection space
    _feature_selection_hyperparameter = []
    for _feature_selection in [*feature_selection]:
        for (
            item
        ) in (
            feature_selection_hyperparameters
        ):  # search the feature selections' hyperparameters
            # find feature_selection key
            for _key in item.keys():
                if "feature_selection_" in _key:
                    _feature_selection_key = _key
                    break
            if item[_feature_selection_key] == _feature_selection:
                # create a copy of hyperparameters, avoid changes on original
                _item = copy.deepcopy(item)
                # convert string to tune.choice
                _item[_feature_selection_key] = tune.choice(
                    [_item[_feature_selection_key]]
                )
                _feature_selection_hyperparameter.append(_item)
                break

    # raise error if no feature selection hyperparameters are found
    if len(_feature_selection_hyperparameter) == 0:
        raise ValueError(
            "No feature selection hyperparameters are found. Please check your feature selections."
        )

    _feature_selection_hyperparameter = tune.choice(_feature_selection_hyperparameter)

    # model selection and hyperparameter optimization space
    _model_hyperparameter = []
    for _model in [*models]:
        # checked before at models that all models are in default space
        for item in models_hyperparameters:  # search the models' hyperparameters
            # find model key
            for _key in item.keys():
                if "model_" in _key:
                    _model_key = _key
                    break
            if item[_model_key] == _model:
                # create a copy of hyperparameters, avoid changes on original
                _item = copy.deepcopy(item)
                # convert string to tune.choice
                _item[_model_key] = tune.choice([_item[_model_key]])
                _model_hyperparameter.append(_item)
                break

    # raise error if no model hyperparameters are found
    if len(_model_hyperparameter) == 0:
        raise ValueError(
            "No model hyperparameters are found. Please check your models."
        )

    _model_hyperparameter = tune.choice(_model_hyperparameter)

    # the pipeline search space
    # select one of the method/hyperparameter setting from each part
    return {
        "task_type": "tabular_" + task_mode,
        "encoder": _encoding_hyperparameter,
        "imputer": _imputer_hyperparameter,
        "balancing": _balancing_hyperparameter,
        "scaling": _scaling_hyperparameter,
        "feature_selection": _feature_selection_hyperparameter,
        "model": _model_hyperparameter,
    }


# get the hyperparameter optimization algorithm based on string input
def get_algo(search_algo):

    if search_algo == "RandomSearch" or search_algo == "GridSearch":

        # Random Search and Grid Search
        from ray.tune.suggest.basic_variant import BasicVariantGenerator

        algo = BasicVariantGenerator
    # elif search_algo == "BayesOptSearch":

    #     # check whether bayes_opt is installed
    #     bayes_opt_spec = importlib.util.find_spec("bayes_opt")
    #     if bayes_opt_spec is None:
    #         raise ImportError(
    #             "BayesOpt is not installed. Please install it first to use BayesOptSearch. \
    #                 Command to install: pip install bayesian-optimization"
    #         )

    #     # Bayesian Search
    #     from ray.tune.suggest.bayesopt import BayesOptSearch

    #     algo = BayesOptSearch
    elif search_algo == "AxSearch":

        # check whether Ax and sqlalchemy are installed
        Ax_spec = importlib.util.find_spec("ax")
        sqlalchemy_spec = importlib.util.find_spec("sqlalchemy")
        if Ax_spec is None or sqlalchemy_spec is None:
            raise ImportError(
                "Ax or sqlalchemy not installed. Please install these packages to use AxSearch. \
                    Command to install: pip install ax-platform sqlalchemy"
            )

        # Ax Search
        from ray.tune.suggest.ax import AxSearch

        algo = AxSearch
    # elif search_algo == "BOHB":

    #     # check whether HpBandSter and ConfigSpace are installed
    #     hpbandster_spec = importlib.util.find_spec("hpbandster")
    #     ConfigSpace_spec = importlib.util.find_spec("ConfigSpace")
    #     if hpbandster_spec is None or ConfigSpace_spec is None:
    #         raise ImportError(
    #             "HpBandSter or ConfigSpace not installed. Please install these packages to use BOHB. \
    #             Command to install: pip install hpbandster ConfigSpace"
    #         )

    #     # Bayesian Optimization HyperBand/BOHB
    #     from ray.tune.suggest.bohb import TuneBOHB

    #     algo = TuneBOHB
    elif search_algo == "BlendSearch":

        # check whether flaml is installed
        flaml_spec = importlib.util.find_spec("flaml")
        if flaml_spec is None:
            raise ImportError(
                "flaml not installed. Please install it first to use BlendSearch. \
                Command to install: pip install 'flaml[blendsearch]'"
            )

        # Blend Search
        from ray.tune.suggest.flaml import BlendSearch

        algo = BlendSearch
    elif search_algo == "CFO":

        # check whether flaml is installed
        flaml_spec = importlib.util.find_spec("flaml")
        if flaml_spec is None:
            raise ImportError(
                "flaml not installed. Please install it first to use BlendSearch. \
                Command to install: pip install 'flaml[blendsearch]'"
            )

        # Blend Search
        from ray.tune.suggest.flaml import CFO

        algo = CFO
    # elif search_algo == "DragonflySearch":

    #     # check whether dragonfly-opt is installed
    #     dragonfly_spec = importlib.util.find_spec("dragonfly")
    #     if dragonfly_spec is None:
    #         raise ImportError(
    #             "dragonfly-opt not installed. Please install it first to use DragonflySearch. \
    #             Command to install: pip install dragonfly-opt"
    #         )

    #     # Dragonfly Search
    #     from ray.tune.suggest.dragonfly import DragonflySearch

    #     algo = DragonflySearch
    elif search_algo == "HEBO":

        # check whether HEBO is installed
        HEBO_spec = importlib.util.find_spec("HEBO")
        if HEBO_spec is None:
            raise ImportError(
                "HEBO not installed. Please install it first to use HEBO. \
                Command to install: pip install 'HEBO>=0.2.0'"
            )

        # Heteroscedastic Evolutionary Bayesian Optimization/HEBO
        from ray.tune.suggest.hebo import HEBOSearch

        algo = HEBOSearch
    elif search_algo == "HyperOpt":

        # check whether hyperopt is installed
        hyperopt_spec = importlib.util.find_spec("hyperopt")
        if hyperopt_spec is None:
            raise ImportError(
                "hyperopt not installed. Please install it first to use HyperOpt. \
                Command to install: pip install -U hyperopt"
            )

        # HyperOpt Search
        from ray.tune.suggest.hyperopt import HyperOptSearch

        algo = HyperOptSearch
    elif search_algo == "Nevergrad":

        # check whether nevergrad is installed
        nevergrad_spec = importlib.util.find_spec("nevergrad")
        if nevergrad_spec is None:
            raise ImportError(
                "nevergrad not installed. Please install it first to use Nevergrad. \
                Command to install: pip install nevergrad"
            )

        # Nevergrad Search
        from ray.tune.suggest.nevergrad import NevergradSearch

        algo = NevergradSearch
    # elif search_algo == "Optuna":

    #     # check whether optuna is installed
    #     optuna_spec = importlib.util.find_spec("optuna")
    #     if optuna_spec is None:
    #         raise ImportError(
    #             "optuna not installed. Please install it first to use Optuna. \
    #             Command to install: pip install optuna"
    #         )

    #     # Optuna Search
    #     from ray.tune.suggest.optuna import OptunaSearch

    #     algo = OptunaSearch
    # elif search_algo == "SigOpt":

    #     # check whether sigopt is installed
    #     sigopt_spec = importlib.util.find_spec("sigopt")
    #     if sigopt_spec is None:
    #         raise ImportError(
    #             "sigopt not installed. Please install it first to use SigOpt. \
    #             Command to install: pip install sigopt \
    #             Set SigOpt API: export SIGOPT_KEY= ..."
    #         )

    #     # SigOpt Search
    #     from ray.tune.suggest.sigopt import SigOptSearch

    #     algo = SigOptSearch
    # elif search_algo == "Scikit-Optimize":

    #     # check whether scikit-optimize is installed
    #     skopt_spec = importlib.util.find_spec("skopt")
    #     if skopt_spec is None:
    #         raise ImportError(
    #             "scikit-optimize not installed. Please install it first to use Scikit-Optimize. \
    #             Command to install: pip install scikit-optimize"
    #         )

    #     # Scikit-Optimize Search
    #     from ray.tune.suggest.skopt import SkOptSearch

    #     algo = SkOptSearch
    # elif search_algo == "ZOOpt":

    #     # check whether zoopt is installed
    #     zoopt_spec = importlib.util.find_spec("zoopt")
    #     if zoopt_spec is None:
    #         raise ImportError(
    #             "zoopt not installed. Please install it first to use ZOOpt. \
    #             Command to install: pip install zoopt"
    #         )

    #     # ZOOpt Search
    #     from ray.tune.suggest.zoopt import ZOOptSearch

    #     algo = ZOOptSearch
    elif search_algo == "Repeater":

        # Repeated Evaluations
        from ray.tune.suggest import Repeater

        algo = Repeater
    elif search_algo == "ConcurrencyLimiter":

        # ConcurrencyLimiter
        from ray.tune.suggest import ConcurrencyLimiter

        algo = ConcurrencyLimiter
    else:

        # if none above, assume is a callable custom algorithm
        if isinstance(search_algo, Callable):
            algo = search_algo
        # if not callable, raise error
        else:
            raise TypeError(
                "Algorithm {} is not supported. Please use one of the supported algorithms.".format(
                    search_algo
                )
            )

    return algo


# get search scheduler based on string input
def get_scheduler(search_scheduler):

    if search_scheduler == "FIFOScheduler":

        from ray.tune.schedulers import FIFOScheduler

        scheduler = FIFOScheduler
    elif search_scheduler == "ASHAScheduler":

        from ray.tune.schedulers import ASHAScheduler

        scheduler = ASHAScheduler
    elif search_scheduler == "HyperBandScheduler":

        from ray.tune.schedulers import HyperBandScheduler

        scheduler = HyperBandScheduler
    elif search_scheduler == "MedianStoppingRule":

        from ray.tune.schedulers import MedianStoppingRule

        scheduler = MedianStoppingRule
    elif search_scheduler == "PopulationBasedTraining":

        from ray.tune.schedulers import PopulationBasedTraining

        scheduler = PopulationBasedTraining
    elif search_scheduler == "PopulationBasedTrainingReplay":

        from ray.tune.schedulers import PopulationBasedTrainingReplay

        scheduler = PopulationBasedTrainingReplay
    elif search_scheduler == "PB2":

        # check whether GPy2 is installed
        Gpy_spec = importlib.util.find_spec("GPy")
        if Gpy_spec is None:
            raise ImportError(
                "GPy2 not installed. Please install it first to use PB2. \
                Command to install: pip install GPy"
            )

        from ray.tune.schedulers.pb2 import PB2

        scheduler = PB2
    elif search_scheduler == "HyperBandForBOHB":

        from ray.tune.schedulers import HyperBandForBOHB

        scheduler = HyperBandForBOHB
    else:

        # if callable, use it as scheduler
        if isinstance(search_scheduler, Callable):
            scheduler = search_scheduler
        else:
            raise TypeError(
                "Scheduler {} is not supported. Please use one of the supported schedulers.".format(
                    search_scheduler
                )
            )

    return scheduler


# get progress reporter based on string input
def get_progress_reporter(
    progress_reporter,
    max_evals,
    max_error,
):

    if progress_reporter == "CLIReporter":

        from ray.tune.progress_reporter import CLIReporter

        progress_reporter = CLIReporter(
            # metric_columns=[
            #     "fitted_model",
            #     "training_status",
            #     "total time (s)",
            #     "iter",
            #     "loss",
            # ],
            parameter_columns=["task_type"],
            max_progress_rows=max_evals,
            max_error_rows=max_error,
            sort_by_metric=True,
        )
    elif progress_reporter == "JupyterNotebookReporter":

        from ray.tune.progress_reporter import JupyterNotebookReporter

        progress_reporter = JupyterNotebookReporter(
            overwrite=True,
            # metric_columns=[
            #     "fitted_model",
            #     "training_status",
            #     "total time (s)",
            #     "iter",
            #     "loss",
            # ],
            parameter_columns=["task_type"],
            max_progress_rows=max_evals,
            max_error_rows=max_error,
            sort_by_metric=True,
        )

    # add metrics for visualization
    progress_reporter.add_metric_column("fitted_model")
    progress_reporter.add_metric_column("training_status")
    progress_reporter.add_metric_column("loss")

    return progress_reporter


def get_logger(logger):

    if not isinstance(logger, list) and logger is not None:
        raise TypeError("Expect a list of string or None, get {}.".format(logger))

    loggers = []

    if logger is None:

        from ray.tune.logger import LoggerCallback

        return [LoggerCallback()]

    for log in logger:
        if "Logger" == log:
            from ray.tune.logger import LoggerCallback

            loggers.append(LoggerCallback())
        elif "TBX" == log:

            # check whether TensorBoardX is installed
            TensorBoardX_spec = importlib.util.find_spec("tensorboardX")
            if TensorBoardX_spec is None:
                raise ImportError(
                    "TensorBoardX not installed. Please install it first to use TensorBoardX. \
                Command to install: pip install tensorboardX"
                )

            from ray.tune.logger import TBXLoggerCallback

            loggers.append(TBXLoggerCallback())
        elif "JSON" == log:

            from ray.tune.logger import JsonLoggerCallback

            loggers.append(JsonLoggerCallback())
        elif "CSV" == log:

            from ray.tune.logger import CSVLoggerCallback

            loggers.append(CSVLoggerCallback())
        elif "MLflow" == log:

            # checkwhether mlflow is installed
            mlflow_spec = importlib.util.find_spec("mlflow")
            if mlflow_spec is None:
                raise ImportError(
                    "mlflow not installed. Please install it first to use mlflow. \
                    Command to install: pip install mlflow"
                )

            from ray.tune.integration.mlflow import MLflowLoggerCallback

            loggers.append(MLflowLoggerCallback())
        elif "Wandb" == log:

            # check whether wandb is installed
            wandb_spec = importlib.util.find_spec("wandb")
            if wandb_spec is None:
                raise ImportError(
                    "wandb not installed. Please install it first to use wandb. \
                    Command to install: pip install wandb"
                )

            from ray.tune.integration.wandb import WandbLoggerCallback

            loggers.append(WandbLoggerCallback())

    return loggers


class TimePlateauStopper(Stopper):

    """
    Combination of TimeoutStopper and TrialPlateauStopper
    """

    def __init__(
        self,
        timeout=360,
        metric="loss",
        std=0.01,
        num_results=4,
        grace_period=4,
        metric_threshold=None,
        mode="min",
    ):
        self._start = time.time()
        self._deadline = timeout

        self._metric = metric
        self._mode = mode

        self._std = std
        self._num_results = num_results
        self._grace_period = grace_period
        self._metric_threshold = metric_threshold

        self._iter = defaultdict(lambda: 0)
        self._trial_results = defaultdict(lambda: deque(maxlen=self._num_results))

    def __call__(self, trial_id, result):

        metric_result = result.get(self._metric)  # get metric from result
        self._trial_results[trial_id].append(metric_result)
        self._iter[trial_id] += 1  # record trial results and iteration

        # If still in grace period, do not stop yet
        if self._iter[trial_id] < self._grace_period:
            return False

        # If not enough results yet, do not stop yet
        if len(self._trial_results[trial_id]) < self._num_results:
            return False

        # if threshold specified, use threshold to stop
        # If metric threshold value not reached, do not stop yet
        if self._metric_threshold is not None:
            if self._mode == "min" and metric_result > self._metric_threshold:
                return False
            elif self._mode == "max" and metric_result < self._metric_threshold:
                return False

        # if threshold not specified, use std to stop
        # Calculate stdev of last `num_results` results
        try:
            current_std = np.std(self._trial_results[trial_id])
        except Exception:
            current_std = float("inf")

        # If stdev is lower than threshold, stop early.
        return current_std < self._std

    def stop_all(self):

        return time.time() - self._start > self._deadline


# get estimator based on string or class
def get_estimator(estimator_str):

    if estimator_str == "Lasso":
        from sklearn.linear_model import Lasso

        return Lasso()
    elif estimator_str == "Ridge":
        from sklearn.linear_model import Ridge

        return Ridge()
    elif estimator_str == "ExtraTreeRegressor":
        from sklearn.tree import ExtraTreeRegressor

        return ExtraTreeRegressor()
    elif estimator_str == "RandomForestRegressor":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor()
    elif estimator_str == "LogisticRegression":

        from sklearn.linear_model import LogisticRegression

        return LogisticRegression()
    elif estimator_str == "ExtraTreeClassifier":

        from sklearn.tree import ExtraTreeClassifier

        return ExtraTreeClassifier()
    elif estimator_str == "RandomForestClassifier":

        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier()
    elif isclass(type(estimator_str)):
        # if estimator is recognized as a class
        # make sure it has fit/predict methods
        if not has_method(estimator_str, "fit") or not has_method(
            estimator_str, "predict"
        ):
            raise ValueError("Estimator must have fit/predict methods!")

        return estimator_str()
    else:
        raise AttributeError("Unrecognized estimator!")


# get metrics based on string or class
# if not in min mode, call negative of the metric
def get_metrics(metric_str):

    if metric_str == "neg_accuracy":
        from My_AutoML._utils._stat import neg_accuracy

        return neg_accuracy
    elif metric_str == "accuracy":
        from sklearn.metrics import accuracy_score

        warnings.warn(
            "accuracy_score is not for min mode, please use neg_accuracy instead."
        )
        return accuracy_score
    elif metric_str == "neg_precision":
        from My_AutoML._utils._stat import neg_precision

        return neg_precision
    elif metric_str == "precision":
        from sklearn.metrics import precision_score

        warnings.warn(
            "precision_score is not for min mode, please use neg_precision instead."
        )
        return precision_score
    elif metric_str == "neg_auc":
        from My_AutoML._utils._stat import neg_auc

        return neg_auc
    elif metric_str == "auc":
        from sklearn.metrics import roc_auc_score

        warnings.warn("roc_auc_score is not for min mode, please use neg_auc instead.")
        return roc_auc_score
    elif metric_str == "neg_hinge":
        from My_AutoML._utils._stat import neg_hinge

        return neg_hinge
    elif metric_str == "hinge":
        from sklearn.metrics import hinge_loss

        warnings.warn("hinge_loss is not for min mode, please use neg_hinge instead.")
        return hinge_loss
    elif metric_str == "neg_f1":
        from My_AutoML._utils._stat import neg_f1

        return neg_f1
    elif metric_str == "f1":
        from sklearn.metrics import f1_score

        warnings.warn("f1_score is not for min mode, please use neg_f1 instead.")
        return f1_score
    elif metric_str == "MSE":
        from sklearn.metrics import mean_squared_error

        return mean_squared_error
    elif metric_str == "MAE":
        from sklearn.metrics import mean_absolute_error

        return mean_absolute_error
    elif metric_str == "MSLE":
        from sklearn.metrics import mean_squared_log_error

        return mean_squared_log_error
    elif metric_str == "neg_R2":
        from My_AutoML._utils._stat import neg_R2

        return neg_R2
    elif metric_str == "R2":
        from sklearn.metrics import r2_score

        warnings.warn("r2_score is not for min mode, please use neg_R2 instead.")
        return r2_score
    elif metric_str == "MAX":
        from sklearn.metrics import max_error

        return max_error
    elif isinstance(metric_str, Callable):
        # if callable, pass
        return metric_str
    else:
        raise ValueError("Unrecognized criteria!")
