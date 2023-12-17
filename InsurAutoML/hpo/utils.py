"""
File: _utils.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.5
Relative Path: /InsurAutoML/hpo/utils.py
File: _utils.py
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Saturday, 16th December 2023 7:19:48 pm
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
from typing import Callable, Union, Tuple, Dict, Any, List
import json
import os
import time
import scipy
import logging
import func_timeout
from ray import tune
import pandas as pd
import numpy as np

from ..base import set_seed
from ..utils.file import save_methods
from ..utils.optimize import setup_logger, get_metrics

logger = logging.getLogger(__name__)


class TabularObjective(tune.Trainable):
    def setup(
        self,
        config: Dict,
        data_split: List[Tuple[pd.DataFrame, pd.DataFrame]] = None,
        encoder: Dict[str, Callable] = None,
        imputer: Dict[str, Callable] = None,
        balancing: Dict[str, Callable] = None,
        scaling: Dict[str, Callable] = None,
        feature_selection: Dict[str, Callable] = None,
        models: Dict[str, Callable] = None,
        model_name: str = "model",
        task_mode: str = "classification",
        objective: str = "accuracy",
        full_status: bool = False,
        reset_index: bool = True,
        timeout: int = 36,
        _iter: int = 1,
        seed: int = None,
    ) -> None:
        # assign hyperparameter arguments
        self.encoder = encoder
        self.imputer = imputer
        self.balancing = balancing
        self.scaling = scaling
        self.feature_selection = feature_selection
        self.models = models

        # assign objective parameters
        self.data_split = data_split
        self.model_name = model_name
        self.task_mode = task_mode
        self.objective = objective
        self.full_status = full_status
        self.reset_index = reset_index
        self.timeout = timeout
        self._iter = _iter
        self.seed = seed

        if data_split is None:
            pass
        elif isinstance(data_split[0][0][0], pd.DataFrame):
            self.dict2config(config)

        self._logger = setup_logger(__name__, "stdout.log")

    def step(self) -> Dict[str, Any]:
        try:
            # Update: Nov. 24, 2023
            # Signal/Multiprocessing timeout methods can be ignored by ctypes callback
            # Use func_timeout instead
            # self.status_dict = time_limit(self.timeout)(self._objective)()
            self.status_dict = func_timeout.func_timeout(self.timeout, self._objective)
        # except TimeoutError:
        except func_timeout.FunctionTimedOut:
            self._logger.error(
                "Objective not finished due to timeout after {} seconds.".format(
                    self.timeout
                )
            )
            self.status_dict = {
                "training_status": "TIMEOUT",
                "loss": np.inf,
            }
            # return full status if full_status is True
            if self.full_status:
                self.status_dict.update(
                    {
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
                    }
                )

        return self.status_dict

    def reset_config(self, new_config: Dict) -> bool:
        self.dict2config(new_config)

        return True

    @staticmethod
    def _extract_hyper(params: Dict, comp: str) -> Tuple[str, Dict]:
        # issue 1: https://github.com/PanyiDong/InsurAutoML/issues/1
        # HyperOpt hyperparameter space conflicts with ray.tune

        # while setting hyperparameters space,
        # the method name is injected into the hyperparameter space
        # so, before fitting, these indications are remove

        # get hyperparameter of component
        _hyper = params[comp].copy()
        # find corresponding encoder key
        try:
            for key in _hyper.keys():
                if "{}_".format(comp) in key:
                    _key = key
                    break
            _comp = _hyper[_key]
            del _hyper[_key]
            # remove indications
            _hyper = {k.replace(_comp + "_", ""): _hyper[k] for k in _hyper}
        # if not get above format, use default format
        except BaseException:
            _comp = _hyper[comp]
            del _hyper[comp]

        return _comp, _hyper

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

        # get method & hyperparameter of encoder
        self._encoder, self._encoder_hyper = self._extract_hyper(params, "encoder")
        self.enc = self.encoder[self._encoder](**self._encoder_hyper)

        # get method & hyperparameter of imputer
        self._imputer, self._imputer_hyper = self._extract_hyper(params, "imputer")
        self.imp = self.imputer[self._imputer](**self._imputer_hyper)

        # get method & hyperparameter of balancing
        self._balancing, self._balancing_hyper = self._extract_hyper(
            params, "balancing"
        )
        self.blc = self.balancing[self._balancing](**self._balancing_hyper)

        # get method & hyperparameter of scaling
        self._scaling, self._scaling_hyper = self._extract_hyper(params, "scaling")
        self.scl = self.scaling[self._scaling](**self._scaling_hyper)

        # get method & hyperparameter of feature selection
        self._feature_selection, self._feature_selection_hyper = self._extract_hyper(
            params, "feature_selection"
        )
        self.fts = self.feature_selection[self._feature_selection](
            **self._feature_selection_hyper
        )

        # get method & hyperparameter of model
        self._model, self._model_hyper = self._extract_hyper(params, "model")
        self.mol = self.models[self._model](
            **self._model_hyper
        )  # call the model using passed parameters

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

    def _get_objective(self) -> Callable:
        # different evaluation metrics for classification and regression
        # notice: if add metrics that is larger the better, need to add - sign
        # at actual fitting process below (since try to minimize the loss)
        objective_str = (
            self.objective.__name__
            if hasattr(self.objective, "__name__")
            else self.objective
        )
        if self.task_mode == "regression":
            # evaluation for predictions
            if objective_str in ["R2"]:
                _objective = "neg_" + objective_str
            else:
                _objective = self.objective
            try:
                _obj = get_metrics(_objective)
            except:
                self._logger.error(
                    'Mode {} only support ["MSE", "MAE", "MSLE", "R2", "MAX", callable], get{}'.format(
                        self.task_mode, self.objective
                    )
                )

            self._logger.info("Objective: {} by {}".format(_obj, _objective))
        elif self.task_mode == "classification":
            # evaluation for predictions
            if objective_str.lower() in [
                "accuracy",
                "precision",
                "auc",
                "hinge",
                "f1",
            ]:
                _objective = "neg_" + objective_str
            else:
                _objective = self.objective
            try:
                _obj = get_metrics(_objective)
            except:
                self._logger.error(
                    'Mode {} only support ["accuracy", "precision", "auc", "hinge", "f1", callable], get{}'.format(
                        self.task_mode, self.objective
                    )
                )

        return _obj

    def _fit(
        self,
        _X_train_obj: pd.DataFrame,
        _y_train_obj: pd.DataFrame,
        _X_test_obj: pd.DataFrame = None,
        _y_test_obj: pd.DataFrame = None,
        refit: bool = False,
    ) -> Union[
        Tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.DataFrame]], None
    ]:
        # encoding
        start_time = time.time()
        _X_train_obj = self.enc.fit(_X_train_obj)
        if not refit:
            _X_test_obj = self.enc.refit(_X_test_obj)
        end_time = time.time()
        self._logger.info(
            "[INFO] Encoding takes: {:24.4f}s".format(end_time - start_time)
        )
        self._logger.info("[INFO] Encoding finished, in imputation process.")

        # imputer
        start_time = time.time()
        _X_train_obj = self.imp.fill(_X_train_obj)
        if not refit:
            _X_test_obj = self.imp.fill(_X_test_obj)
        end_time = time.time()
        self._logger.info(
            "[INFO] Imputation takes: {:22.4f}s".format(end_time - start_time)
        )
        self._logger.info("[INFO] Imputation finished, in balancing process.")

        # balancing
        start_time = time.time()
        _X_train_obj, _y_train_obj = self.blc.fit_transform(_X_train_obj, _y_train_obj)
        end_time = time.time()
        self._logger.info(
            "[INFO] Balancing takes: {:23.4f}s".format(end_time - start_time)
        )
        self._logger.info("[INFO] Balancing finished, in scaling process.")

        # make sure the classes are integers (belongs to certain classes)
        if self.task_mode == "classification":
            _y_train_obj = _y_train_obj.astype(int)
            if not refit:
                _y_test_obj = _y_test_obj.astype(int)

        # scaling
        start_time = time.time()
        self.scl.fit(_X_train_obj, _y_train_obj)
        end_time = time.time()
        _X_train_obj = self.scl.transform(_X_train_obj)
        if not refit:
            _X_test_obj = self.scl.transform(_X_test_obj)
        self._logger.info(
            "[INFO] Scaling takes: {:25.4f}s".format(end_time - start_time)
        )
        self._logger.info("[INFO] Scaling finished, in feature selection process.")

        # feature selection
        start_time = time.time()
        self.fts.fit(_X_train_obj, _y_train_obj)
        end_time = time.time()
        _X_train_obj = self.fts.transform(_X_train_obj)
        if not refit:
            _X_test_obj = self.fts.transform(_X_test_obj)
        self._logger.info(
            "[INFO] Feature selection takes: {:15.4f}s".format(end_time - start_time)
        )
        self._logger.info(
            "[INFO] Feature selection finished, in {} model.".format(self.task_mode)
        )

        # fit model
        if scipy.sparse.issparse(_X_train_obj):  # check if returns sparse matrix
            self._logger.info(
                "[INFO] Sparse train matrix detected, convert to dense array."
            )
            _X_train_obj = _X_train_obj.toarray()
        if not refit:
            if scipy.sparse.issparse(_X_test_obj):
                self._logger.info(
                    "[INFO] Sparse test matrix detected, convert to dense array."
                )
                _X_test_obj = _X_test_obj.toarray()

        # store the preprocessed train/test datasets when refit
        if refit:
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
                self._logger.error("Only accept numpy array or pandas dataframe!")

        start_time = time.time()
        self.mol.fit(_X_train_obj, _y_train_obj.values.ravel())
        end_time = time.time()
        self._logger.info(
            "[INFO] Model fitting takes: {:19.4f}s".format(end_time - start_time)
        )
        self._logger.info("[INFO] Model fitting finished.")

        if not refit:
            return (_X_test_obj, _y_test_obj)
        else:
            return (_X_train_obj, _y_train_obj)

    def _predict(
        self,
        _X_test_obj: Union[np.ndarray, pd.DataFrame],
        _y_test_obj: Union[np.ndarray, pd.DataFrame],
    ) -> Union[int, float]:
        # get objective function by task mode and input objective
        _obj = self._get_objective()

        # UPDATE: Jul. 20, 2023
        # special case for auc and hinge
        objective_str = (
            self.objective.__name__
            if hasattr(self.objective, "__name__")
            else self.objective
        )
        if objective_str.lower() in ["auc"]:
            y_pred = self.mol.predict_proba(_X_test_obj)
        else:
            y_pred = self.mol.predict(_X_test_obj)

        # UPDATE: Jul. 10, 2023
        # negative losses are handled by get_metrics earlier
        _loss = _obj(_y_test_obj.values, y_pred)
        # register failed losses as np.inf
        _loss = _loss if isinstance(_loss, (int, float)) else np.inf

        return _loss

    # # actual objective function
    # @ignore_warnings(category=ConvergenceWarning)
    def _objective(
        self,
    ) -> Dict[str, Any]:
        # set random seed
        set_seed(self.seed)

        self._logger.info("[INFO] Objective starting...")

        # Update: Dec. 2, 2023
        # split data has been moved to AutoTabular methods for consistent splitting
        # if self.validation == "KFold":
        #     kf = KFold(
        #         n_splits=int(1 / self.valid_size), shuffle=True, random_state=self.seed
        #     )
        #     _data_split = [
        #         [
        #             (self._X.loc[_train_idx, :], self._y.loc[_train_idx, :]),
        #             (self._X.loc[_test_idx, :], self._y.loc[_test_idx, :]),
        #         ]
        #         for idx, (_train_idx, _test_idx) in enumerate(
        #             kf.split(self._X, self._y)
        #         )
        #     ]
        # elif self.validation:
        #     # only perform train_test_split when validation
        #     # train test split so the performance of model selection and
        #     # hyperparameter optimization can be evaluated
        #     X_train, X_test, y_train, y_test = train_test_split(
        #         self._X, self._y, test_perc=self.valid_size, seed=self.seed
        #     )
        #     _data_split = [[(X_train, y_train), (X_test, y_test)]]
        # else:
        #     # if no validation, use all data for training
        #     _data_split = [[(self._X, self._y), (self._X, self._y)]]

        # if self.reset_index:
        #     # reset index to avoid indexing order error
        #     _data_split = [
        #         [
        #             (X_train.reset_index(drop=True), y_train.reset_index(drop=True)),
        #             (X_test.reset_index(drop=True), y_test.reset_index(drop=True)),
        #         ]
        #         for (X_train, y_train), (X_test, y_test) in _data_split
        #     ]

        # fit & predict
        _loss_list = []
        for idx, (_train, _test) in enumerate(self.data_split):
            if len(self.data_split) > 1:
                self._logger.info("[INFO] Fold: {}".format(idx + 1))

            # fit the pipeline and return the preprocessed test datasets
            _test = self._fit(*_train, *_test, refit=False)
            _loss = self._predict(*_test)
            _loss_list.append(_loss)

        # calculate mean loss
        # if KFold, calculate mean of all folds
        self._logger.info("[INFO] Loss from all folds: {}".format(_loss_list))
        _loss = np.average(_loss_list)

        # concat train/test splits if using validation, refit the model with all data
        if self.data_split[0][0][0].shape != self.data_split[0][1][0].shape:
            _X = pd.concat(
                [self.data_split[0][0][0], self.data_split[0][1][0]],
                axis=0,
                ignore_index=True,
            )
            _y = pd.concat(
                [self.data_split[0][0][1], self.data_split[0][1][1]],
                axis=0,
                ignore_index=True,
            )
            self._logger.info("[INFO] Refit the model with all data...")
            _full_processed = self._fit(_X, _y, refit=True)
            self._logger.info(
                "[INFO] Refit loss with all data is {:.6f}".format(
                    self._predict(*_full_processed)
                )
            )

        # save the fitted model objects
        save_methods(
            self.model_name,
            [self.enc, self.imp, self.blc, self.scl, self.fts, self.mol],
        )

        self._logger.info(
            "[INFO] Loss from objective function is: {:.6f} calculated by (negative) {}.".format(
                _loss,
                self.objective,
            )
        )
        self._iter += 1

        # return dictionary of objective
        result = {
            "fitted_model": self._model,
            "training_status": "FITTED",
            "loss": _loss,
        }
        # return full status if full_status is True
        if self.full_status:
            result.update(
                {
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
                }
            )

        return result
