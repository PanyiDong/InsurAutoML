"""
File Name: ML.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Actuarial and Risk Management Sciences, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.6
Relative Path: /InsurAutoML/hpo/informed/ML.py
File Created: Thursday, 4th December 2025 1:05:41 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Thursday, 4th December 2025 2:42:24 pm
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
from typing import Union, List, Callable, Dict
import numpy as np
import pandas as pd

from .base import InformedAutoTabularBase
from ...utils.base import type_of_task


class InformedAutoTabularRegressor(InformedAutoTabularBase):

    def __init__(
        self,
        n_estimators: int = 5,
        ensemble_strategy: str = "stacking",
        voting: str = "mean",
        timeout: int = 360,
        max_evals: int = 64,
        timeout_per_trial: int = None,
        allow_error: Union[float, int] = 0.1,
        temp_directory: str = "tmp",
        delete_temp_after_terminate: bool = False,
        save: bool = True,
        resume: Union[bool, str] = "AUTO",
        model_name: str = "model",
        ignore_warning: bool = True,
        models: Union[str, List[str]] = "auto",
        exclude: Dict = {},
        valid_size: float = 0.2,
        objective: Union[str, Callable] = "MSE",
        search_algo: str = "RandomSearch",
        search_algo_settings: Dict = {},
        search_scheduler: str = "FIFOScheduler",
        search_scheduler_settings: Dict = {},
        logger: Union[str, List[str]] = ["Logger"],
        progress_reporter: str = None,
        full_status: bool = False,
        verbose: int = 1,
        cpu_threads: int = None,
        use_gpu: bool = None,
        reset_index: bool = True,
        seed: int = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.ensemble_strategy = ensemble_strategy
        self.voting = voting
        self.timeout = timeout
        self.max_evals = max_evals
        self.timeout_per_trial = timeout_per_trial
        self.allow_error = allow_error
        self.temp_directory = temp_directory
        self.delete_temp_after_terminate = delete_temp_after_terminate
        self.save = save
        self.resume = resume
        self.model_name = model_name
        self.ignore_warning = ignore_warning
        self.models = models
        self.exclude = exclude
        self.valid_size = valid_size
        self.objective = objective
        self.search_algo = search_algo
        self.search_algo_settings = search_algo_settings
        self.search_scheduler = search_scheduler
        self.search_scheduler_settings = search_scheduler_settings
        self.logger = logger
        self.progress_reporter = progress_reporter
        self.full_status = full_status
        self.verbose = verbose
        self.cpu_threads = cpu_threads
        self.use_gpu = use_gpu
        self.reset_index = reset_index
        self.seed = seed

        self._fitted = False  # whether the model has been fitted

        super(InformedAutoTabularRegressor, self).__init__(
            task_mode="regression",
            n_estimators=self.n_estimators,
            ensemble_strategy=self.ensemble_strategy,
            voting=self.voting,
            timeout=self.timeout,
            timeout_per_trial=self.timeout_per_trial,
            max_evals=self.max_evals,
            allow_error=self.allow_error,
            temp_directory=self.temp_directory,
            delete_temp_after_terminate=self.delete_temp_after_terminate,
            save=self.save,
            resume=self.resume,
            model_name=self.model_name,
            ignore_warning=self.ignore_warning,
            models=self.models,
            exclude=self.exclude,
            valid_size=self.valid_size,
            objective=self.objective,
            search_algo=self.search_algo,
            search_algo_settings=self.search_algo_settings,
            search_scheduler=self.search_scheduler,
            search_scheduler_settings=self.search_scheduler_settings,
            logger=self.logger,
            progress_reporter=self.progress_reporter,
            full_status=self.full_status,
            verbose=self.verbose,
            cpu_threads=self.cpu_threads,
            use_gpu=self.use_gpu,
            reset_index=self.reset_index,
            seed=self.seed,
        )


class InformedAutoTabularClassifier(InformedAutoTabularBase):

    def __init__(
        self,
        n_estimators: int = 5,
        ensemble_strategy: str = "stacking",
        voting="hard",
        timeout: int = 360,
        max_evals: int = 64,
        timeout_per_trial: int = None,
        allow_error: Union[float, int] = 0.1,
        temp_directory: str = "tmp",
        delete_temp_after_terminate: bool = False,
        save: bool = True,
        resume: Union[bool, str] = "AUTO",
        model_name: str = "model",
        ignore_warning: bool = True,
        models: Union[str, List[str]] = "auto",
        exclude: Dict = {},
        valid_size: float = 0.2,
        objective: Union[str, Callable] = "accuracy",
        search_algo: str = "RandomSearch",
        search_algo_settings: Dict = {},
        search_scheduler: str = "FIFOScheduler",
        search_scheduler_settings: Dict = {},
        logger: Union[str, List[str]] = ["Logger"],
        progress_reporter: str = None,
        full_status: bool = False,
        verbose: int = 1,
        cpu_threads: int = None,
        use_gpu: bool = None,
        reset_index: bool = True,
        seed: int = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.ensemble_strategy = ensemble_strategy
        self.voting = voting
        self.timeout = timeout
        self.max_evals = max_evals
        self.timeout_per_trial = timeout_per_trial
        self.allow_error = allow_error
        self.temp_directory = temp_directory
        self.delete_temp_after_terminate = delete_temp_after_terminate
        self.save = save
        self.resume = resume
        self.model_name = model_name
        self.ignore_warning = ignore_warning
        self.models = models
        self.exclude = exclude
        self.valid_size = valid_size
        self.objective = objective
        self.search_algo = search_algo
        self.search_algo_settings = search_algo_settings
        self.search_scheduler = search_scheduler
        self.search_scheduler_settings = search_scheduler_settings
        self.logger = logger
        self.progress_reporter = progress_reporter
        self.full_status = full_status
        self.verbose = verbose
        self.cpu_threads = cpu_threads
        self.use_gpu = use_gpu
        self.reset_index = reset_index
        self.seed = seed

        self._fitted = False  # whether the model has been fitted

        super(InformedAutoTabularClassifier, self).__init__(
            task_mode="classification",
            n_estimators=self.n_estimators,
            ensemble_strategy=self.ensemble_strategy,
            voting=self.voting,
            timeout=self.timeout,
            max_evals=self.max_evals,
            timeout_per_trial=self.timeout_per_trial,
            allow_error=self.allow_error,
            temp_directory=self.temp_directory,
            delete_temp_after_terminate=self.delete_temp_after_terminate,
            save=self.save,
            resume=self.resume,
            model_name=self.model_name,
            ignore_warning=self.ignore_warning,
            models=self.models,
            exclude=self.exclude,
            valid_size=self.valid_size,
            objective=self.objective,
            search_algo=self.search_algo,
            search_algo_settings=self.search_algo_settings,
            search_scheduler=self.search_scheduler,
            search_scheduler_settings=self.search_scheduler_settings,
            logger=self.logger,
            progress_reporter=self.progress_reporter,
            full_status=self.full_status,
            verbose=self.verbose,
            cpu_threads=self.cpu_threads,
            use_gpu=self.use_gpu,
            reset_index=self.reset_index,
            seed=self.seed,
        )


class InformedAutoTabular(InformedAutoTabularBase):

    def __init__(
        self,
        n_estimators: int = 5,
        ensemble_strategy: str = "stacking",
        voting: str = None,
        timeout: int = 360,
        max_evals: int = 64,
        timeout_per_trial: int = None,
        allow_error: Union[float, int] = 0.1,
        temp_directory: str = "tmp",
        delete_temp_after_terminate: bool = False,
        save: bool = True,
        resume: Union[bool, str] = "AUTO",
        model_name: str = "model",
        ignore_warning: bool = True,
        models: Union[str, List[str]] = "auto",
        exclude: Dict = {},
        valid_size: float = 0.2,
        objective: Union[str, Callable] = None,
        search_algo: str = "RandomSearch",
        search_algo_settings: Dict = {},
        search_scheduler: str = "FIFOScheduler",
        search_scheduler_settings: Dict = {},
        logger: Union[str, List[str]] = ["Logger"],
        progress_reporter: str = None,
        full_status: bool = False,
        verbose: int = 1,
        cpu_threads: int = None,
        use_gpu: bool = None,
        reset_index: bool = True,
        seed: int = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.ensemble_strategy = ensemble_strategy
        self.voting = voting
        self.timeout = timeout
        self.max_evals = max_evals
        self.timeout_per_trial = timeout_per_trial
        self.allow_error = allow_error
        self.temp_directory = temp_directory
        self.delete_temp_after_terminate = delete_temp_after_terminate
        self.save = save
        self.resume = resume
        self.model_name = model_name
        self.ignore_warning = ignore_warning
        self.models = models
        self.exclude = exclude
        self.valid_size = valid_size
        self.objective = objective
        self.search_algo = search_algo
        self.search_algo_settings = search_algo_settings
        self.search_scheduler = search_scheduler
        self.search_scheduler_settings = search_scheduler_settings
        self.logger = logger
        self.progress_reporter = progress_reporter
        self.full_status = full_status
        self.verbose = verbose
        self.cpu_threads = cpu_threads
        self.use_gpu = use_gpu
        self.reset_index = reset_index
        self.seed = seed

        self._fitted = False  # whether the model has been fitted

    @staticmethod
    def _get_task_mode(type: str) -> str:
        if type in ["binary", "multiclass"]:
            return "classification"
        elif type in ["integer", "continuous"]:
            return "regression"
        else:
            raise ValueError(
                'Not recognizing type, only ["binary", "multiclass", "integer", "continuous"] accepted, get {}!'.format(
                    type
                )
            )

    @staticmethod
    def _get_default_objective(type: str, objective) -> Union[str, Callable]:
        if type in ["binary", "multiclass"]:
            return "accuracy" if not objective else objective
        elif type in ["integer", "continuous"]:
            return "MSE" if not objective else objective
        else:
            raise ValueError(
                'Not recognizing type, only ["binary", "multiclass", "integer", "continuous"] accepted, get {}!'.format(
                    type
                )
            )

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series, np.ndarray] = None,
    ) -> InformedAutoTabular:
        if isinstance(y, (pd.DataFrame, pd.Series, np.ndarray)):
            self._type = type_of_task(y)
        elif not y:
            self._type = "unsupervised"

        super(InformedAutoTabular, self).__init__(
            task_mode=self._get_task_mode(self._type),
            n_estimators=self.n_estimators,
            ensemble_strategy=self.ensemble_strategy,
            voting=self.voting,
            timeout=self.timeout,
            max_evals=self.max_evals,
            timeout_per_trial=self.timeout_per_trial,
            allow_error=self.allow_error,
            temp_directory=self.temp_directory,
            delete_temp_after_terminate=self.delete_temp_after_terminate,
            save=self.save,
            resume=self.resume,
            model_name=self.model_name,
            ignore_warning=self.ignore_warning,
            models=self.models,
            exclude=self.exclude,
            valid_size=self.valid_size,
            objective=self._get_default_objective(self._type, self.objective),
            search_algo=self.search_algo,
            search_algo_settings=self.search_algo_settings,
            search_scheduler=self.search_scheduler,
            search_scheduler_settings=self.search_scheduler_settings,
            logger=self.logger,
            progress_reporter=self.progress_reporter,
            full_status=self.full_status,
            verbose=self.verbose,
            cpu_threads=self.cpu_threads,
            use_gpu=self.use_gpu,
            reset_index=self.reset_index,
            seed=self.seed,
        )

        super(InformedAutoTabular, self).fit(X, y)

        return self

    def predict(self, X: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        # check if the model has been fitted
        if not self._fitted:
            raise ValueError("No tasks found! Need to fit first.")

        return super(InformedAutoTabular, self).predict(X)

    def predict_proba(
        self, X: pd.DataFrame
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        # check if the model has been fitted
        if not self._fitted:
            raise ValueError("No tasks found! Need to fit first.")

        return super(InformedAutoTabular, self).predict_proba(X)
