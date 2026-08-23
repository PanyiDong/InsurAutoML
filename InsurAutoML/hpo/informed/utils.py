"""
File Name: utils.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Actuarial and Risk Management Sciences, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.6
Relative Path: /InsurAutoML/hpo/informed/utils.py
File Created: Thursday, 4th December 2025 10:14:34 am
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Friday, 26th December 2025 10:15:05 pm
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
from typing import Any, List, Callable, Dict, Tuple
import os
import json
import time
import func_timeout
import copy
import pandas as pd
import numpy as np
from ray import tune

from ..base import set_seed
from ...utils.file import save_methods
from ...utils.optimize import setup_logger, get_metrics


class InformedTabularObjective(tune.Trainable):

    def setup(
        self,
        config: Dict,
        data_split: List[Tuple[pd.DataFrame, pd.DataFrame]] = None,
        encoder: Dict[str, Callable] = None,
        complete_prep: Dict[str, Callable] = None,
        missing_prep: Dict[str, Callable] = None,
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
        self.complete_prep = complete_prep
        self.missing_prep = missing_prep
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

        self.dict2config(config)

        self._logger = setup_logger(__name__, "stdout.log")

    def step(self) -> Dict[str, Any]:
        try:
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
                        "complete_prep": self._complete_prep,
                        "complete_prep_hyperparameter": self._complete_prep_hyper,
                        "missing_prep": self._missing_prep,
                        "missing_prep_hyperparameter": self._missing_prep_hyper,
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
        for key in list(params.keys()):
            if key not in [
                "encoder",
                "complete_prep",
                "missing_prep",
                "model",
                "task_type",
            ]:
                params.pop(key, None)

        # get method & hyperparameter of encoder
        self._encoder, self._encoder_hyper = self._extract_hyper(params, "encoder")
        self.enc = self.encoder[self._encoder](**self._encoder_hyper)

        # get method & hyperparameter of complete_prep
        self._complete_prep, self._complete_prep_hyper = self._extract_hyper(
            params, "complete_prep"
        )
        self.comp_prep = self.complete_prep[self._complete_prep](
            **self._complete_prep_hyper
        )

        # get method & hyperparameter of missing_prep
        self._missing_prep, self._missing_prep_hyper = self._extract_hyper(
            params, "missing_prep"
        )
        self.miss_prep = self.missing_prep[self._missing_prep](
            **self._missing_prep_hyper
        )

        # get method & hyperparameter of model
        self._model, self._model_hyper = self._extract_hyper(params, "model")
        mol = self.models[self._model](
            **self._model_hyper
        )  # call the model using passed parameters
        self.mol_complete = copy.deepcopy(mol)
        self.mol_missing = copy.deepcopy(mol)

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

    # # actual objective function
    # @ignore_warnings(category=ConvergenceWarning)
    def _objective(
        self,
    ) -> Dict[str, Any]:
        # set random seed
        set_seed(self.seed)

        self._logger.info("[INFO] Objective starting...")

        # fit & predict
        X, y = self.data_split
        # data encoding
        start_time = time.time()
        X_enc = self.enc.fit(X)
        end_time = time.time()
        self._logger.info(
            "[INFO] Encoding takes: {:24.4f}s".format(end_time - start_time)
        )
        self._logger.info("[INFO] Encoding finished, in preprocessing process.")

        # complete data preparation
        start_time = time.time()
        X_train_complete, X_valid_complete, y_train_complete, y_valid_complete = (
            self.comp_prep.fit(X_enc, y)
        )
        end_time = time.time()
        self._logger.info(
            "[INFO] Complete data preparation takes: {:24.4f}s".format(
                end_time - start_time
            )
        )
        self._logger.info(
            "[INFO] Complete data preparation finished, in missing data preparation process."
        )

        # missing data preparation
        start_time = time.time()
        X_train_missing, X_valid_missing, y_train_missing, y_valid_missing = (
            self.miss_prep.fit(X_enc, y)
        )
        end_time = time.time()
        self._logger.info(
            "[INFO] Missing data preparation takes: {:24.4f}s".format(
                end_time - start_time
            )
        )
        self._logger.info(
            "[INFO] Missing data preparation finished, in model training process."
        )

        # train on complete set
        start_time = time.time()
        self.mol_complete.fit(X_train_complete, y_train_complete)
        end_time = time.time()
        self._logger.info(
            "[INFO] Model training on complete set takes: {:24.4f}s".format(
                end_time - start_time
            )
        )
        self._logger.info(
            "[INFO] Model training on complete set finished, in model on missing process."
        )

        # train on missing set
        start_time = time.time()
        self.mol_missing.fit(X_train_missing, y_train_missing)
        end_time = time.time()
        self._logger.info(
            "[INFO] Model training on missing set takes: {:24.4f}s".format(
                end_time - start_time
            )
        )
        self._logger.info(
            "[INFO] Model training on missing set finished, in evaluation process."
        )

        # evaluate on validation set
        # get objective function by task mode and input objective
        _obj = self._get_objective()

        # get predictions for complete & missing set
        objective_str = (
            self.objective.__name__
            if hasattr(self.objective, "__name__")
            else self.objective
        )
        if objective_str.lower() in ["auc"]:
            y_pred_complete = self.mol_complete.predict_proba(X_valid_complete)
            y_pred_missing = self.mol_missing.predict_proba(X_valid_missing)
        else:
            y_pred_complete = self.mol_complete.predict(X_valid_complete)
            y_pred_missing = self.mol_missing.predict(X_valid_missing)

        # # set predictions as dataframe and use index of predictions on complete to
        # # adjust predictions on missing
        # y_pred = pd.Series(y_pred_missing, index=y_valid_missing.index)
        # y_pred[y_pred_complete.index] += y_pred_complete
        # y_pred.loc[y_pred_complete.index] /= 2

        # _loss = _obj(y_valid_missing.values, y_pred.values)
        # # register failed losses as np.inf
        # _loss = _loss if isinstance(_loss, (int, float)) else np.inf

        _loss_complete = _obj(y_valid_complete.values, y_pred_complete)
        _loss_missing = _obj(y_valid_missing.values, y_pred_missing)
        # register failed losses as np.inf
        _loss_complete = (
            _loss_complete if isinstance(_loss_complete, (int, float)) else np.inf
        )
        _loss_missing = (
            _loss_missing if isinstance(_loss_missing, (int, float)) else np.inf
        )

        # calculate mean loss
        # the loss is weighted by the size of validation set
        _loss = (
            len(y_valid_complete) * _loss_complete
            + len(y_valid_missing) * _loss_missing
        )
        _loss /= len(y_valid_complete) + len(y_valid_missing)

        self._logger.info(
            "[INFO] Loss from objective function is: {:.6f} calculated by (negative) {}.".format(
                _loss,
                self.objective,
            )
        )

        # refit the model with all data
        # refit on complete data
        _X_complete = pd.concat(
            [X_train_complete, X_valid_complete], axis=0, ignore_index=True
        )
        _y_complete = pd.concat(
            [y_train_complete, y_valid_complete],
            axis=0,
            ignore_index=True,
        )
        self._logger.info("[INFO] Refit the complete model with all data...")
        self.mol_complete.fit(_X_complete, _y_complete)
        # refit on missing data
        _X_missing = pd.concat(
            [X_train_missing, X_valid_missing],
            axis=0,
            ignore_index=True,
        )
        _y_missing = pd.concat(
            [y_train_missing, y_valid_missing],
            axis=0,
            ignore_index=True,
        )
        self._logger.info("[INFO] Refit the missing model with all data...")
        self.mol_missing.fit(_X_missing, _y_missing)
        # refit loss
        if objective_str.lower() in ["auc"]:
            _y_pred_complete = self.mol_complete.predict_proba(_X_complete)
            _y_pred_missing = self.mol_missing.predict_proba(_X_missing)
        else:
            _y_pred_complete = self.mol_complete.predict(_X_complete)
            _y_pred_missing = self.mol_missing.predict(_X_missing)

        # # set predictions as dataframe and use index of predictions on complete to
        # # adjust predictions on missing
        # _y_pred = pd.Series(_y_pred_missing, index=_y_missing.index)
        # _y_pred.loc[_y_pred_complete.index] += _y_pred_complete
        # _y_pred.loc[_y_pred_complete.index] /= 2
        # _loss_refit = _obj(_y_missing.values, _y_pred.values)

        _loss_complete_refit = _obj(_y_complete.values, _y_pred_complete)
        _loss_missing_refit = _obj(_y_missing.values, _y_pred_missing)
        _loss_refit = (
            len(_y_complete) * _loss_complete_refit
            + len(_y_missing) * _loss_missing_refit
        )
        _loss_refit /= len(_y_complete) + len(_y_missing)
        self._logger.info(
            "[INFO] Refit loss with all data is {:.6f}".format(_loss_refit)
        )

        # save the fitted model objects
        save_methods(
            self.model_name,
            [
                self.enc,
                self.comp_prep,
                self.miss_prep,
                self.mol_complete,
                self.mol_missing,
            ],
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
                    "complete_prep": self._complete_prep,
                    "complete_prep_hyperparameter": self._complete_prep_hyper,
                    "missing_prep": self._missing_prep,
                    "missing_prep_hyperparameter": self._missing_prep_hyper,
                    "model": self._model,
                    "model_hyperparameter": self._model_hyper,
                }
            )

        return result
