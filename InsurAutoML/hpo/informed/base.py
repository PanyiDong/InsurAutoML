"""
File Name: base.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Actuarial and Risk Management Sciences, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.6
Relative Path: /InsurAutoML/hpo/informed/base.py
File Created: Tuesday, 2nd December 2025 8:12:56 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 8th December 2025 9:18:16 am
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
from typing import Any, Union, List, Callable, Dict, Tuple
import os
import warnings
import importlib
import shutil
import copy
import datetime
import pandas as pd
import numpy as np
from ray import tune

from .informed_ensemble import (
    InformedPipeline,
    InformedClassifierEnsemble,
    InformedRegressorEnsemble,
)
from .utils import InformedTabularObjective
from ..base import set_seed, AutoTabularBase
from ...utils.base import format_hyper_dict
from ...utils.data import str2list
from ...utils.file import (
    save_methods,
    load_methods,
    find_exact_path,
)
from ...utils.metadata import MetaData
from ...utils.optimize import (
    get_algo,
    set_algo_seed,
    get_scheduler,
    get_logger,
    get_progress_reporter,
    InformedStopper,
    ray_status,
    check_status,
)

warnings.filterwarnings("ignore", category=UserWarning)

# check whether gpu device available
torch_spec = importlib.util.find_spec("torch")
if torch_spec is not None:
    import torch

    device_count = torch.cuda.device_count()
else:
    device_count = 0


class InformedAutoTabularBase(AutoTabularBase):
    """ "
    Base class module for Informed AutoTabular (for classification and regression tasks)

    Parameters
    ----------
    task_mode: Mode of tasks, default: "classification"
    when called by AutoTabularClassification/AutoTabularRegression,
    task mode will be determined without reading data
    support ("classification", "regression")

    n_estimators: top k pipelines used to create the ensemble, default: 5

    voting: voting method used for ensemble, default: None
    if None, use "hard" for classification, "mean" for regression

    ensemble_strategy: strategy of ensemble, default: "stacking"
    support ("stacking", "bagging", "boosting")

    timeout: Total time limit for the job in seconds, default = 360

    max_evals: Maximum number of function evaluations allowed, default = 32

    timeout_per_trial: Time limit for each trial in seconds, default = None
    default by (timeout / max_evals * 5)

    allow_error: proportion of tasks allows failure when float and number by int, default = 0.1
    allowed number of failures is int(max_evals * allow_error) or int(allow_error)

    temp_directory: folder path to store temporary model, default = 'tmp'

    delete_temp_after_terminate: whether to delete temporary information, default = False

    save: whether to save model after training, default = True

    resume: whether to resume training from last checkpoint, default = "AUTO"
    support ("AUTO", bool)

    model_name: saved model name, default = 'model'

    ignore_warning: whether to ignore warning, default = True

    models: Models selected for the job, default = 'auto'
    support classifiers ('AdaboostClassifier', 'BernoulliNB', 'DecisionTree',
            'ExtraTreesClassifier', 'GaussianNB', 'GradientBoostingClassifier',
            'KNearestNeighborsClassifier', 'LDA', 'LibLinear_SVC', 'LibSVM_SVC',
            'MLPClassifier', 'MultinomialNB','PassiveAggressive', 'QDA',
            'RandomForest',  'SGD')
    support regressors ("AdaboostRegressor", "ARDRegression", "DecisionTree",
            "ExtraTreesRegressor", "GaussianProcess", "GradientBoosting",
            "KNearestNeighborsRegressor", "LibLinear_SVR", "LibSVM_SVR",
            "MLPRegressor", "RandomForest", "SGD")
    'auto' will select all default models, or use a list to select

    exclude: components to exclude, default = {}
    keys are components, values are lists of components to exclude
    example: {'encoder': ['DataEncoding'], 'imputer': ['SimpleImputer', 'JointImputer']}

    valid_size: Test percentage used to evaluate the performance, default = 0.2
    only effective when validation = True or "KFold"

    objective: Objective function to test performance, default = 'accuracy'
    support metrics for regression ("MSE", "MAE", "MSLE", "R2", "MAX")
    support metrics for classification ("accuracy", "precision", "auc", "hinge", "f1")

    search_algo: search algorithm used for hyperparameter optimization, default = "RandomSearch"
    support ("RandomSearch", "GridSearch", "BayesOptSearch", "AxSearch", "BOHB",
            "BlendSearch", "CFO", "DragonflySearch", "HEBO", "HyperOpt", "Nevergrad",
            "Optuna", "SigOpt", "Scikit-Optimize", "ZOOpt", "Reapter",
            "ConcurrencyLimiter", callable)

    search_algo_settings: search algorithm settings, default = {}
    need manual configuration for each search algorithm

    search_scheduler: search scheduler used, default = "FIFOScheduler"
    support ("FIFOScheduler", "ASHAScheduler", "HyperBandScheduler", "MedianStoppingRule"
            "PopulationBasedTraining", "PopulationBasedTrainingReplay", "PB2",
            "HyperBandForBOHB", callable)

    search_scheduler_settings: search scheduler settings, default = {}
    need manual configuration for each search scheduler

    logger: callback logger, default = ["Logger"]
    list of supported callbacks, support ("Logger", "TBX", "JSON", "CSV", "MLflow", "Wandb")

    progress_reporter: progress reporter, default = None
    automatically decide what progressbar to use
    support ("CLIReporter", "JupyterNotebookReporter")

    full_status: whether to print full status, default = False

    verbose: display for output, default = 1
    support (0, 1, 2, 3)

    cpu_threads: number of cpu threads to use, default = None
    if None, get all available cpu threads

    use_gpu: whether to use gpu, default = None
    if None, will use gpu if available, otherwise False (not to use gpu)

    reset_index: whether to reset index during training, default = True
    there are methods that are index independent (ignore index, reset, e.g. GAIN)
    if you wish to use these methods and set reset_index = False, please make sure
    all input index are ordered and starting from 0

    seed: random seed, default = 1
    """

    def __init__(
        self,
        task_mode: str = "classification",
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
        objective: Union[str, Callable] = "accuracy",
        search_algo: str = "RandomSearch",
        search_algo_settings: Dict = {},
        search_scheduler: str = "FIFOScheduler",
        search_scheduler_settings: Dict = {},
        logger: List[str] = ["Logger"],
        progress_reporter: str = None,
        full_status: bool = False,
        verbose: int = 1,
        cpu_threads: int = None,
        use_gpu: bool = None,
        reset_index: bool = True,
        seed: int = None,
    ) -> None:
        self.task_mode = task_mode
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
        self.seed = seed if seed else 42

        self._iter = 0  # record iteration number
        self._fitted = False  # record whether the model has been fitted

        # set random seed
        set_seed(self.seed)
        super().__init__(
            task_mode=task_mode,
            n_estimators=n_estimators,
            ensemble_strategy=ensemble_strategy,
            voting=voting,
            timeout=timeout,
            max_evals=max_evals,
            timeout_per_trial=timeout_per_trial,
            allow_error=allow_error,
            temp_directory=temp_directory,
            delete_temp_after_terminate=delete_temp_after_terminate,
            save=save,
            resume=resume,
            model_name=model_name,
            ignore_warning=ignore_warning,
            models=models,
            exclude=exclude,
            valid_size=valid_size,
            objective=objective,
            search_algo=search_algo,
            search_algo_settings=search_algo_settings,
            search_scheduler=search_scheduler,
            search_scheduler_settings=search_scheduler_settings,
            logger=logger,
            progress_reporter=progress_reporter,
            full_status=full_status,
            verbose=verbose,
            cpu_threads=cpu_threads,
            use_gpu=use_gpu,
            reset_index=reset_index,
            seed=seed,
        )

    def get_hyperparameter_space(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[Dict]:
        # initialize default search options
        # and select the search options based on the input restrictions
        # use copy to allows multiple manipulation

        # Encoding: convert string types to numerical type
        # all encoders available
        from ...encoding import encoders

        # if additional exists, import, otherwise set to default
        try:
            from additional import add_encoders
        except:
            add_encoders = {}

        # include original encoders
        _all_encoders = copy.deepcopy(encoders)
        # include additional encoders
        _all_encoders.update(add_encoders)

        # get encoder methods space
        encoder = copy.deepcopy(_all_encoders)

        # exclude unwanted encoders if specified
        if "encoder" in self.exclude.keys():
            for _encoder in self.exclude["encoder"]:
                encoder.pop(_encoder, None)

        # Complete and Missing Data Preparation
        from ...prep import CompletePrepPipeline, MissingPrepPipeline

        # get complete prep pipeline space
        complete_prep = {"CompletePrepPipeline": CompletePrepPipeline}
        # get missing prep pipeline space
        missing_prep = {"MissingPrepPipeline": MissingPrepPipeline}

        # Model selection/Hyperparameter optimization
        # using Bayesian Optimization
        # all models available
        # if mode is classification, use classification models
        # if mode is regression, use regression models
        if self.task_mode == "classification":
            from ...model import classifiers

            # if additional exists, import, otherwise set to default
            try:
                from additional import add_classifiers
            except:
                add_classifiers = {}

            # include original classifiers
            _all_models = copy.deepcopy(classifiers)
            # include additional classifiers
            _all_models.update(add_classifiers)
        elif self.task_mode == "regression":
            from ...model import regressors

            # if additional exists, import, otherwise set to default
            try:
                from additional import add_regressors
            except:
                add_regressors = {}

            # include original regressors
            _all_models = copy.deepcopy(regressors)
            # include additional regressors
            _all_models.update(add_regressors)

        # special treatment, remove SVM methods when observations are large
        # SVM suffers from the complexity o(n_samples^2 * n_features),
        # which is time-consuming for large datasets
        if X.shape[0] * X.shape[1] > 10000:
            # in case the methods are not included, will check before delete
            if self.task_mode == "classification":
                del _all_models["LibLinear_SVC"]
                del _all_models["LibSVM_SVC"]
            elif self.task_mode == "regression":
                del _all_models["LibLinear_SVR"]
                del _all_models["LibSVM_SVR"]
        # Remove GAM methods if multi-class classification
        if (
            self.task_mode == "classification"
            and len(pd.unique(y.to_numpy().flatten())) > 2
        ):
            del _all_models["GAM_Classifier"]

        # model space, only select chosen models to space
        if self.models == "auto":  # if auto, model pool will be all default models
            models = copy.deepcopy(_all_models)
        else:
            self.models = str2list(self.models)  # string list to list
            models = {}  # if specified, check if models in default models
            for _model in self.models:
                if _model not in [*_all_models]:
                    self._logger.error(
                        "Only supported models are {}, get {}.".format(
                            [*_all_models], _model
                        )
                    )
                models[_model] = _all_models[_model]

        # exclude unwanted models if specified
        if "model" in self.exclude.keys():
            for _model in self.exclude["model"]:
                models.pop(_model, None)

        # initialize default search space
        from ...utils.optimize import _get_informed_hyperparameter_space

        from ...hyperparameters import (
            encoder_hyperparameter,
            complete_prep_hyperparameter,
            missing_prep_hyperparameter,
            classifier_hyperparameter,
            regressor_hyperparameter,
        )

        # if additional exists, import, otherwise set to default
        try:
            from additional import (
                add_encoder_hyperparameter,
                add_classifier_hyperparameter,
                add_regressor_hyperparameter,
            )
        except:
            add_encoder_hyperparameter = {}
            add_classifier_hyperparameter = {}
            add_regressor_hyperparameter = {}

        # if needed, modify default hyperparameter space
        # like model hyperparameter space below
        # all hyperparameters for encoders
        _all_encoders_hyperparameters = copy.deepcopy(encoder_hyperparameter)
        # include additional hyperparameters
        _all_encoders_hyperparameters += add_encoder_hyperparameter

        # all hyperparameters for complete prep pipeline
        _all_complete_prep_hyperparameters = copy.deepcopy(complete_prep_hyperparameter)
        # all hyperparameters for missing prep pipeline
        _all_missing_prep_hyperparameters = copy.deepcopy(missing_prep_hyperparameter)

        # all hyperparameters for the models by mode
        if self.task_mode == "classification":
            _all_models_hyperparameters = copy.deepcopy(classifier_hyperparameter)
            # include additional hyperparameters
            _all_models_hyperparameters += add_classifier_hyperparameter
        elif self.task_mode == "regression":
            _all_models_hyperparameters = copy.deepcopy(regressor_hyperparameter)
            # include additional hyperparameters
            _all_models_hyperparameters += add_regressor_hyperparameter

        # special treatment, for LightGBM_Classifier
        # if binary classification, use LIGHTGBM_BINARY_CLASSIFICATION
        # if multiclass, use LIGHTGBM_MULTICLASS_CLASSIFICATION
        if self.task_mode == "classification":
            # get LightGBM_Regressor key
            for item in _all_models_hyperparameters:
                if "LightGBM_Classifier" in item.values():
                    # flatten to 1d
                    if len(pd.unique(y.to_numpy().flatten())) == 2:
                        from ...constant import LIGHTGBM_BINARY_CLASSIFICATION

                        item["objective"] = tune.choice(LIGHTGBM_BINARY_CLASSIFICATION)
                    else:
                        from ...constant import (
                            LIGHTGBM_MULTICLASS_CLASSIFICATION,
                        )

                        item["objective"] = tune.choice(
                            LIGHTGBM_MULTICLASS_CLASSIFICATION
                        )

        # check status of hyperparameter space
        check_status(encoder, _all_encoders_hyperparameters, ref="encoder")
        check_status(
            complete_prep, _all_complete_prep_hyperparameters, ref="complete_prep"
        )
        check_status(
            missing_prep, _all_missing_prep_hyperparameters, ref="missing_prep"
        )
        check_status(models, _all_models_hyperparameters, ref="model")

        # format default search space
        _all_encoders_hyperparameters = [
            format_hyper_dict(
                dict, order + 1, ref="encoder", search_algo=self.search_algo
            )
            for order, dict in enumerate(_all_encoders_hyperparameters)
        ]
        _all_complete_prep_hyperparameters = [
            format_hyper_dict(
                dict, order + 1, ref="complete_prep", search_algo=self.search_algo
            )
            for order, dict in enumerate(_all_complete_prep_hyperparameters)
        ]
        _all_missing_prep_hyperparameters = [
            format_hyper_dict(
                dict, order + 1, ref="missing_prep", search_algo=self.search_algo
            )
            for order, dict in enumerate(_all_missing_prep_hyperparameters)
        ]
        _all_models_hyperparameters = [
            format_hyper_dict(
                dict, order + 1, ref="model", search_algo=self.search_algo
            )
            for order, dict in enumerate(_all_models_hyperparameters)
        ]

        # generate the hyperparameter space
        hyperparameter_space = _get_informed_hyperparameter_space(
            X,
            _all_encoders_hyperparameters,
            encoder,
            _all_complete_prep_hyperparameters,
            complete_prep,
            _all_missing_prep_hyperparameters,
            missing_prep,
            _all_models_hyperparameters,
            models,
            self.task_mode,
            self.search_algo,
        )  # _X to choose whether include imputer
        # others are the combinations of default hyperparameter space & methods
        # selected

        return (
            encoder,
            complete_prep,
            missing_prep,
            models,
            hyperparameter_space,
        )

    # select optimal settings and fit on optimal hyperparameters
    def _fit_optimal(
        self, idx: int, optimal_point: Dict, best_path: str
    ) -> Tuple[str, InformedPipeline]:
        # get optimal encoder & hyperparameters
        optimal_encoder, optimal_encoder_hyperparameters = self._get_optimal_hyper(
            optimal_point, "encoder"
        )
        # get optimal complete prep & hyperparameters
        optimal_complete_prep, optimal_complete_prep_hyperparameters = (
            self._get_optimal_hyper(optimal_point, "complete_prep")
        )
        # get optimal missing prep & hyperparameters
        optimal_missing_prep, optimal_missing_prep_hyperparameters = (
            self._get_optimal_hyper(optimal_point, "missing_prep")
        )
        # get optimal model & hyperparameters
        optimal_model, optimal_model_hyperparameters = self._get_optimal_hyper(
            optimal_point, "model"
        )

        # if already exists, use append mode
        # else, write mode
        if not os.path.exists(
            os.path.join(self.temp_directory, self.model_name, "optimal_setting.txt")
        ) or self.ensemble_strategy in ["boosting"]:
            write_type = "w"
        else:
            write_type = "a"

        # record optimal settings
        with open(
            os.path.join(self.temp_directory, self.model_name, "optimal_setting.txt"),
            write_type,
        ) as f:
            f.write("For pipeline {}:\n".format(idx + 1))
            f.write("Optimal encoding method is: {}\n".format(optimal_encoder))
            f.write("Optimal encoding hyperparameters:")
            print(optimal_encoder_hyperparameters, file=f, end="\n\n")
            f.write(
                "Optimal complete prep method is: {}\n".format(optimal_complete_prep)
            )
            f.write("Optimal complete prep hyperparameters:")
            print(optimal_complete_prep_hyperparameters, file=f, end="\n\n")
            f.write("Optimal missing prep method is: {}\n".format(optimal_missing_prep))
            f.write("Optimal missing prep hyperparameters:")
            print(optimal_missing_prep_hyperparameters, file=f, end="\n\n")
            f.write("Optimal {} model is: {}\n".format(self.task_mode, optimal_model))
            f.write("Optimal {} hyperparameters:".format(self.task_mode))
            print(optimal_model_hyperparameters, file=f, end="\n\n")

        (
            _fit_encoder,
            _fit_complete_prep,
            _fit_missing_prep,
            _fit_model_complete,
            _fit_model_missing,
        ) = load_methods(best_path)

        # create a pipeline using loaded methods
        pip_setting = {
            "encoder": _fit_encoder,
            "complete_prep": _fit_complete_prep,
            "missing_prep": _fit_missing_prep,
            "model_complete": _fit_model_complete,
            "model_missing": _fit_model_missing,
        }

        return ("pipe_" + str(idx + 1), InformedPipeline(**pip_setting))

    def _fit_ensemble(
        self, trial_id: str, config: Dict, iter: int = None, features: List[str] = None
    ) -> None:
        # initialize ensemble list
        # if ensemble list exists, append to it
        if hasattr(self, "ensemble_list") or hasattr(self, "feature_list"):
            pass
        else:
            # else, initialize the list
            self.ensemble_list = []
            self.feature_list = []

        # if only one optimal input, need to convert to iterable
        if not isinstance(trial_id, pd.Series) or not isinstance(config, pd.Series):
            trial_id = [trial_id]
            config = [config]

        # loop through all configs, trial_id, get model ensemble
        for idx, (trial_id, config) in enumerate(zip(trial_id, config)):
            # find the exact path
            if iter is None:
                _path = find_exact_path(
                    os.path.join(self.sub_directory, self.model_name),
                    "id_" + trial_id,
                )
                _path = os.path.join(_path, self.model_name)

                self.ensemble_list.append(self._fit_optimal(idx, config, _path))
            else:
                _path = find_exact_path(
                    os.path.join(
                        self.sub_directory,
                        self.model_name
                        + "_"
                        + self.ensemble_strategy
                        + "_"
                        + str(iter + 1),
                    ),
                    "id_" + trial_id,
                )
                _path = os.path.join(_path, self.model_name)

                self.ensemble_list.append(self._fit_optimal(iter, config, _path))
            if (
                features is not None
            ):  # if feature subset is provided, save the feature subsets
                self.feature_list.append(features)

        # wrap pipelines into ensemble
        if self.task_mode == "classification":
            self.ensemble = InformedClassifierEnsemble(
                estimators=self.ensemble_list,
                voting=self.voting,
                features=self.feature_list,
                strategy=self.ensemble_strategy,
            )
        elif self.task_mode == "regression":
            self.ensemble = InformedRegressorEnsemble(
                estimators=self.ensemble_list,
                voting=self.voting,
                features=self.feature_list,
                strategy=self.ensemble_strategy,
            )

    def _check_prep_hyper(self):
        # check whether prep_hyper contains all necessary keys
        required_keys = [
            "missing_threshold",
            "cc_threshold",
            "split_ratio",
            "twin_r",
            "imputation_max_iter",
            "imputation_n_estimators",
        ]
        for key in required_keys:
            if key not in self.prep_hyper.keys():
                self._logger.error(
                    "prep_hyper must contain keys: {}, missing key: {}".format(
                        required_keys, key
                    )
                )

    def fit(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> AutoTabularBase:
        # initialize settings
        self._init_fit()

        # convert to dataframe
        if not isinstance(X, pd.DataFrame):
            try:
                X = pd.DataFrame(X)
                self._logger.info(
                    "[INFO] {} Experiment: {}. Status: X is not a dataframe, converted to dataframe.".format(
                        datetime.datetime.now().strftime("%H:%M:%S %Y-%m-%d"),
                        self.model_name,
                    )
                )
            except BaseException:
                self._logger.error(
                    "Cannot convert X to dataframe, get {}".format(type(X))
                )
        if not isinstance(y, pd.DataFrame):
            try:
                y = pd.DataFrame(y)
                self._logger.info(
                    "[INFO] {} Experiment: {}. Status: y is not a dataframe, converted to dataframe.".format(
                        datetime.datetime.now().strftime("%H:%M:%S %Y-%m-%d"),
                        self.model_name,
                    )
                )
            except BaseException:
                self._logger.error(
                    "Cannot convert y to dataframe, get {}".format(type(y))
                )

        # get features and response names
        if isinstance(X, pd.DataFrame):  # expect multiple features
            self.features = list(X.columns)

        if isinstance(y, pd.DataFrame):  # for the case of dataframe
            self.response = list(y.columns)
        elif isinstance(y, pd.Series):  # for the case of series
            self.response = list(y.name)

        _X = X.copy()
        _y = y.copy()

        if self.reset_index:
            # reset index to avoid indexing error
            _X.reset_index(drop=True, inplace=True)
            _y.reset_index(drop=True, inplace=True)

        # get data metadata
        if not hasattr(self, "metadata"):
            self.metadata = MetaData(_X).metadata
        # check if there's unsupported data type
        # if datetime ,recommend to remove
        if ("Datetime", "") in self.metadata.keys():
            self._logger.warning(
                "Found datetime data type columns {}, it's better to remove those columns".format(
                    *self.metadata[("Datetime", "")]
                )
            )
        # TODO: when NLP and Image supported, redirect to corresponding model
        if ("Object", "Text") in self.metadata.keys():
            self._logger.error("Text data type is not supported yet.")
        if ("Path", "") in self.metadata.keys():
            self._logger.error("Image data type is not supported yet.")

        (
            encoder,
            complete_prep,
            missing_prep,
            models,
            hyperparameter_space,
        ) = self.get_hyperparameter_space(_X, _y)

        self._logger.info(
            "[INFO] {} Experiment: {}. Status: Initialized AutoTabular Hyperparameter space.".format(
                datetime.datetime.now().strftime("%H:%M:%S %Y-%m-%d"), self.model_name
            )
        )

        # if the model is already trained, read the setting
        if os.path.exists(self.model_name):
            self._logger.info(
                "[INFO] {} Experiment: {}. Status: Stored model found, load previous model.".format(
                    datetime.datetime.now().strftime("%H:%M:%S %Y-%m-%d"),
                    self.model_name,
                )
            )
            [self.ensemble] = load_methods(self.model_name)

            self._fitted = True  # successfully fitted the model

            return self

        # write basic information to init.txt
        with open(
            os.path.join(self.temp_directory, self.model_name, "init.txt"), "w"
        ) as f:
            f.write("Features of the dataset: {}\n".format(list(_X.columns)))
            f.write(
                "Shape of the design matrix: {} * {}\n".format(_X.shape[0], _X.shape[1])
            )
            f.write("Response of the dataset: {}\n".format(list(_y.columns)))
            f.write(
                "Shape of the response vector: {} * {}\n".format(
                    _y.shape[0], _y.shape[1]
                )
            )
            f.write("Type of the task: {}.\n".format(self.task_mode))

        # use ray for Model Selection and Hyperparameter Selection
        # get search algorithm
        algo = get_algo(self.search_algo)
        # set random seed of search algorithm
        self.search_algo_settings.update(set_algo_seed(self.search_algo, self.seed))

        # get search scheduler
        scheduler = get_scheduler(self.search_scheduler)

        # get callback logger
        logger = get_logger(self.logger)

        # get progress reporter
        progress_reporter = get_progress_reporter(
            self.progress_reporter,
            self.max_evals,
            self.max_error,
        )

        # initialize stopper
        stopper = InformedStopper(
            timeout=self.timeout,
            metric="loss",
            std_ratio=0.1,
            num_results=4,
            grace_period=4,
            mode="min",
        )

        # trial directory name
        def trial_str_creator(trial):
            trialname = "iter_{}_id_{}".format(self._iter + 1, trial.trial_id)
            self._iter += 1
            return trialname

        # log starting of the experiment
        self._logger.info(
            "[INFO] {}  Experiment: {}. Status: Start AutoTabular training.".format(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                self.model_name,
            )
        )

        # set ray status
        rayStatus = ray_status(
            cpu_threads=self.cpu_threads,
            gpu_count=self.gpu_count,
        )

        # ensemble settings
        if self.n_estimators == 1:
            self._logger.warning("Set n_estimators to 1, no ensemble will be used.")

            # get progress reporter
            progress_reporter = get_progress_reporter(
                self.progress_reporter,
                self.max_evals,
                self.max_error,
            )

            # set trainable
            trainer = tune.with_parameters(
                InformedTabularObjective,
                data_split=(_X, _y),
                encoder=encoder,
                complete_prep=complete_prep,
                missing_prep=missing_prep,
                models=models,
                model_name=self.model_name,
                task_mode=self.task_mode,
                objective=self.objective,
                full_status=self.full_status,
                reset_index=self.reset_index,
                timeout=self.timeout_per_trial,
                _iter=self._iter,
                seed=self.seed,
            )

            # initialize ray
            rayStatus.ray_init()

            # subtrial directory
            self.sub_directory = self.temp_directory

            # optimization process
            # partially activated objective function
            # special treatment for optuna, embed search space in search
            # algorithm
            if self.search_algo in ["Optuna"]:
                fit_analysis = tune.run(
                    trainer,
                    # config=hyperparameter_space,
                    name=self.model_name,  # name of the tuning process, use model_name
                    resume=self.resume,
                    checkpoint_freq=8,  # disable checkpoint
                    checkpoint_at_end=True,
                    keep_checkpoints_num=4,
                    checkpoint_score_attr="loss",
                    mode="min",  # always call a minimization process
                    search_alg=algo(
                        space=hyperparameter_space,
                        mode="min",  # always call a minimization process
                        metric="loss",
                        **self.search_algo_settings,
                    ),
                    scheduler=scheduler(**self.search_scheduler_settings),
                    reuse_actors=True,
                    raise_on_failed_trial=False,
                    metric="loss",
                    num_samples=self.max_evals,
                    max_failures=self.max_error,
                    stop=stopper,  # use stopper
                    callbacks=logger,
                    # time_budget_s=self.timeout,  # included in stopper
                    progress_reporter=progress_reporter,
                    verbose=self.verbose,
                    trial_dirname_creator=trial_str_creator,
                    local_dir=self.sub_directory,
                    log_to_file=("stdout.log", "stderr.log"),
                )
            else:
                fit_analysis = tune.run(
                    trainer,
                    config=hyperparameter_space,
                    name=self.model_name,  # name of the tuning process, use model_name
                    resume=self.resume,
                    checkpoint_freq=8,  # disable checkpoint
                    checkpoint_at_end=True,
                    keep_checkpoints_num=4,
                    checkpoint_score_attr="loss",
                    mode="min",  # always call a minimization process
                    search_alg=algo(**self.search_algo_settings),
                    scheduler=scheduler(**self.search_scheduler_settings),
                    reuse_actors=True,
                    raise_on_failed_trial=False,
                    metric="loss",
                    num_samples=self.max_evals,
                    max_failures=self.max_error,
                    stop=stopper,  # use stopper
                    callbacks=logger,
                    # time_budget_s=self.timeout,  # included in stopper
                    progress_reporter=progress_reporter,
                    verbose=self.verbose,
                    trial_dirname_creator=trial_str_creator,
                    local_dir=self.sub_directory,
                    log_to_file=("stdout.log", "stderr.log"),
                )

            # shut down ray
            rayStatus.ray_shutdown()

            self._logger.info(
                "[INFO] {}  Experiment: {}. Status: AutoTabular training finished. Start postprocessing...".format(
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    self.model_name,
                )
            )

            # check status of the trial analysis
            self.check_analysis(fit_analysis)
            # get the best config settings
            best_trial_id = str(
                fit_analysis.get_best_trial(
                    metric="loss", mode="min", scope="all"
                ).trial_id
            )

            # select optimal settings and fit optimal pipeline
            self._fit_ensemble(best_trial_id, fit_analysis.best_config)
        # Stacking ensemble
        elif self.ensemble_strategy == "stacking":
            # get progress reporter
            progress_reporter = get_progress_reporter(
                self.progress_reporter,
                self.max_evals,
                self.max_error,
            )

            # set trainable
            trainer = tune.with_parameters(
                InformedTabularObjective,
                data_split=(_X, _y),
                encoder=encoder,
                complete_prep=complete_prep,
                missing_prep=missing_prep,
                models=models,
                model_name=self.model_name,
                task_mode=self.task_mode,
                objective=self.objective,
                full_status=self.full_status,
                reset_index=self.reset_index,
                timeout=self.timeout_per_trial,
                _iter=self._iter,
                seed=self.seed,
            )

            # initialize ray
            rayStatus.ray_init()

            # subtrial directory
            self.sub_directory = self.temp_directory

            # optimization process
            # partially activated objective function
            # special treatment for optuna, embed search space in search
            # algorithm
            if self.search_algo in ["Optuna"]:
                fit_analysis = tune.run(
                    trainer,
                    # config=hyperparameter_space,
                    name=self.model_name,  # name of the tuning process, use model_name
                    resume=self.resume,
                    checkpoint_freq=8,  # disable checkpoint
                    checkpoint_at_end=True,
                    keep_checkpoints_num=4,
                    checkpoint_score_attr="loss",
                    mode="min",  # always call a minimization process
                    search_alg=algo(
                        space=hyperparameter_space,
                        mode="min",  # always call a minimization process
                        metric="loss",
                        **self.search_algo_settings,
                    ),
                    scheduler=scheduler(**self.search_scheduler_settings),
                    reuse_actors=True,
                    raise_on_failed_trial=False,
                    metric="loss",
                    num_samples=self.max_evals,
                    max_failures=self.max_error,
                    stop=stopper,  # use stopper
                    callbacks=logger,
                    # time_budget_s=self.timeout,  # included in stopper
                    progress_reporter=progress_reporter,
                    verbose=self.verbose,
                    trial_dirname_creator=trial_str_creator,
                    local_dir=self.sub_directory,
                    log_to_file=("stdout.log", "stderr.log"),
                )
            else:
                fit_analysis = tune.run(
                    trainer,
                    config=hyperparameter_space,
                    name=self.model_name,  # name of the tuning process, use model_name
                    resume=self.resume,
                    checkpoint_freq=8,  # disable checkpoint
                    checkpoint_at_end=True,
                    keep_checkpoints_num=4,
                    checkpoint_score_attr="loss",
                    mode="min",  # always call a minimization process
                    search_alg=algo(**self.search_algo_settings),
                    scheduler=scheduler(**self.search_scheduler_settings),
                    reuse_actors=True,
                    raise_on_failed_trial=False,
                    metric="loss",
                    num_samples=self.max_evals,
                    max_failures=self.max_error,
                    stop=stopper,  # use stopper
                    callbacks=logger,
                    # time_budget_s=self.timeout,  # included in stopper
                    progress_reporter=progress_reporter,
                    verbose=self.verbose,
                    trial_dirname_creator=trial_str_creator,
                    local_dir=self.sub_directory,
                    log_to_file=("stdout.log", "stderr.log"),
                )

            # shut down ray
            rayStatus.ray_shutdown()

            self._logger.info(
                "[INFO] {}  Experiment: {}. Status: AutoTabular training finished. Start postprocessing...".format(
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    self.model_name,
                )
            )

            # check status of the trial analysis
            self.check_analysis(fit_analysis)
            # get all configs, trial_id
            analysis_df = fit_analysis.dataframe(metric="loss", mode="min")

            # reformat config to dict
            analysis_df["config"] = analysis_df.apply(
                lambda x: {
                    "encoder": x["config/encoder"],
                    "complete_prep": x["config/complete_prep"],
                    "missing_prep": x["config/missing_prep"],
                    "model": x["config/model"],
                },
                axis=1,
            )
            # if not enough valid trials, raise warning
            if (analysis_df.training_status == "FITTED").sum() < self.n_estimators:
                self._logger.warning(
                    "[WARNING] {}  Experiment: {}. Ask for total {} estimators, but no enough valid trials exists. Use all {} pipelines instead.".format(
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        self.model_name,
                        self.n_estimators,
                        (analysis_df.training_status == "FITTED").sum(),
                    )
                )

            # sort by loss and get top configs
            analysis_df = analysis_df.sort_values(by=["loss"], ascending=True).head(
                min(
                    self.n_estimators,
                    (analysis_df["training_status"] == "FITTED").sum(),
                )
            )

            # select optimal settings and create the ensemble of pipeline
            self._fit_ensemble(analysis_df.trial_id, analysis_df.config)
        # Bagging ensemble
        elif self.ensemble_strategy == "bagging":
            # create a list of feature subsets
            feature_list = [
                np.random.choice(
                    _X.columns,
                    size=2 * len(_X.columns) // self.n_estimators,
                    replace=False,
                )
                for _ in range(self.n_estimators)
            ]

            # loop through feature_list
            for _n, feature_subset in enumerate(feature_list):
                # get n_trials for the subsets
                sub_n_trials = (
                    (self.max_evals // self.n_estimators + 1)
                    if _n < self.max_evals % self.n_estimators
                    else (self.max_evals // self.n_estimators)
                )

                # get progress reporter
                progress_reporter = get_progress_reporter(
                    self.progress_reporter,
                    self.max_evals,
                    self.max_error,
                )

                # set trainable
                trainer = tune.with_parameters(
                    InformedTabularObjective,
                    data_split=(_X.loc[:, feature_subset], _y),
                    encoder=encoder,
                    complete_prep=complete_prep,
                    missing_prep=missing_prep,
                    models=models,
                    model_name=self.model_name,
                    task_mode=self.task_mode,
                    objective=self.objective,
                    full_status=self.full_status,
                    reset_index=self.reset_index,
                    timeout=self.timeout_per_trial,
                    _iter=self._iter,
                    seed=self.seed,
                )

                # initialize ray
                rayStatus.ray_init()

                # subtrial directory
                self.sub_directory = os.path.join(self.temp_directory, self.model_name)

                # optimization process
                # partially activated objective function
                # special treatment for optuna, embed search space in search
                # algorithm
                if self.search_algo in ["Optuna"]:
                    fit_analysis = tune.run(
                        trainer,
                        # config=hyperparameter_space,
                        name=self.model_name
                        + "_"
                        + self.ensemble_strategy
                        + "_"
                        + str(_n + 1),
                        # name of the tuning process, use model_name
                        resume=self.resume,
                        checkpoint_freq=8,  # disable checkpoint
                        checkpoint_at_end=True,
                        keep_checkpoints_num=4,
                        checkpoint_score_attr="loss",
                        mode="min",  # always call a minimization process
                        search_alg=algo(
                            space=hyperparameter_space,
                            metric="loss",
                            mode="min",  # always call a minimization process
                            **self.search_algo_settings,
                        ),
                        scheduler=scheduler(**self.search_scheduler_settings),
                        reuse_actors=True,
                        raise_on_failed_trial=False,
                        metric="loss",
                        num_samples=sub_n_trials,  # only use sub_n_trials for each of n_estimators
                        max_failures=self.max_error,
                        stop=stopper,  # use stopper
                        callbacks=logger,
                        # time_budget_s=self.timeout,  # included in stopper
                        progress_reporter=progress_reporter,
                        verbose=self.verbose,
                        trial_dirname_creator=trial_str_creator,
                        local_dir=self.sub_directory,
                        log_to_file=("stdout.log", "stderr.log"),
                    )
                else:
                    fit_analysis = tune.run(
                        trainer,
                        config=hyperparameter_space,
                        name=self.model_name
                        + "_"
                        + self.ensemble_strategy
                        + "_"
                        + str(_n + 1),
                        # name of the tuning process, use model_name
                        resume=self.resume,
                        checkpoint_freq=8,  # disable checkpoint
                        checkpoint_at_end=True,
                        keep_checkpoints_num=4,
                        checkpoint_score_attr="loss",
                        mode="min",  # always call a minimization process
                        search_alg=algo(**self.search_algo_settings),
                        scheduler=scheduler(**self.search_scheduler_settings),
                        reuse_actors=True,
                        raise_on_failed_trial=False,
                        metric="loss",
                        num_samples=sub_n_trials,  # only use sub_n_trials for each of n_estimators
                        max_failures=self.max_error,
                        stop=stopper,  # use stopper
                        callbacks=logger,
                        # time_budget_s=self.timeout,  # included in stopper
                        progress_reporter=progress_reporter,
                        verbose=self.verbose,
                        trial_dirname_creator=trial_str_creator,
                        local_dir=self.sub_directory,
                        log_to_file=("stdout.log", "stderr.log"),
                    )

                # shut down ray
                rayStatus.ray_shutdown()

                self._logger.info(
                    "[INFO] {}  Experiment: {}. Status: AutoTabular training finished. Start postprocessing...".format(
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        self.model_name,
                    )
                )

                # check status of the trial analysis
                self.check_analysis(fit_analysis)
                # get the best config settings
                best_trial_id = str(
                    fit_analysis.get_best_trial(
                        metric="loss", mode="min", scope="all"
                    ).trial_id
                )

                # select optimal settings and fit optimal pipeline
                self._fit_ensemble(
                    best_trial_id,
                    fit_analysis.get_best_config(
                        metric="loss", mode="min", scope="all"
                    ),
                    iter=_n,
                    features=feature_subset,
                )

        # Boosting ensemble
        elif self.ensemble_strategy == "boosting":
            # loop through n_estimators
            for _n in range(self.n_estimators):
                sub_n_trials = (
                    (self.max_evals // self.n_estimators + 1)
                    if _n < self.max_evals % self.n_estimators
                    else (self.max_evals // self.n_estimators)
                )

                try:
                    # if fitted before, use pred for residuals
                    data_split = (data_split[0], data_split[1] - _y_pred)
                except:
                    # if not, use y as residuals
                    data_split = (_X, _y)

                # get progress reporter
                progress_reporter = get_progress_reporter(
                    self.progress_reporter,
                    self.max_evals,
                    self.max_error,
                )

                # set trainable
                trainer = tune.with_parameters(
                    InformedTabularObjective,
                    data_split=data_split,
                    encoder=encoder,
                    complete_prep=complete_prep,
                    missing_prep=missing_prep,
                    models=models,
                    model_name=self.model_name,
                    task_mode=self.task_mode,
                    objective=self.objective,
                    full_status=self.full_status,
                    reset_index=self.reset_index,
                    timeout=self.timeout_per_trial,
                    _iter=self._iter,
                    seed=self.seed,
                )

                # initialize ray
                rayStatus.ray_init()

                # subtrial directory
                self.sub_directory = os.path.join(self.temp_directory, self.model_name)

                # optimization process
                # partially activated objective function
                if self.search_algo in ["Optuna"]:
                    fit_analysis = tune.run(
                        trainer,
                        # config=hyperparameter_space,
                        name=self.model_name
                        + "_"
                        + self.ensemble_strategy
                        + "_"
                        + str(_n + 1),
                        # name of the tuning process, use model_name
                        resume=self.resume,
                        checkpoint_freq=8,  # disable checkpoint
                        checkpoint_at_end=True,
                        keep_checkpoints_num=4,
                        checkpoint_score_attr="loss",
                        # mode="min",  # always call a minimization process
                        search_alg=algo(
                            hyperparameter_space,
                            metric="loss",
                            mode="min",  # always call a minimization process
                            **self.search_algo_settings,
                        ),
                        scheduler=scheduler(**self.search_scheduler_settings),
                        reuse_actors=True,
                        raise_on_failed_trial=False,
                        # metric="loss",
                        num_samples=sub_n_trials,  # only use sub_n_trials for each of n_estimators
                        max_failures=self.max_error,
                        stop=stopper,  # use stopper
                        callbacks=logger,
                        # time_budget_s=self.timeout,  # included in stopper
                        progress_reporter=progress_reporter,
                        verbose=self.verbose,
                        trial_dirname_creator=trial_str_creator,
                        local_dir=self.sub_directory,
                        log_to_file=("stdout.log", "stderr.log"),
                    )
                else:
                    fit_analysis = tune.run(
                        trainer,
                        config=hyperparameter_space,
                        name=self.model_name
                        + "_"
                        + self.ensemble_strategy
                        + "_"
                        + str(_n + 1),
                        # name of the tuning process, use model_name
                        resume=self.resume,
                        checkpoint_freq=8,  # disable checkpoint
                        checkpoint_at_end=True,
                        keep_checkpoints_num=4,
                        checkpoint_score_attr="loss",
                        mode="min",  # always call a minimization process
                        search_alg=algo(**self.search_algo_settings),
                        scheduler=scheduler(**self.search_scheduler_settings),
                        reuse_actors=True,
                        raise_on_failed_trial=False,
                        metric="loss",
                        num_samples=sub_n_trials,  # only use sub_n_trials for each of n_estimators
                        max_failures=self.max_error,
                        stop=stopper,  # use stopper
                        callbacks=logger,
                        # time_budget_s=self.timeout,  # included in stopper
                        progress_reporter=progress_reporter,
                        verbose=self.verbose,
                        trial_dirname_creator=trial_str_creator,
                        local_dir=self.sub_directory,
                        log_to_file=("stdout.log", "stderr.log"),
                    )

                # shut down ray
                rayStatus.ray_shutdown()

                self._logger.info(
                    "[INFO] {}  Experiment: {}. Status: AutoTabular training finished. Start postprocessing...".format(
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        self.model_name,
                    )
                )

                # check status of the trial analysis
                self.check_analysis(fit_analysis)
                # get the best config settings
                best_trial_id = str(
                    fit_analysis.get_best_trial(
                        metric="loss", mode="min", scope="all"
                    ).trial_id
                )

                # select optimal settings and fit optimal pipeline
                self._fit_ensemble(
                    best_trial_id,
                    fit_analysis.get_best_config(
                        metric="loss", mode="min", scope="all"
                    ),
                    iter=_n,
                )

                # make sure the ensemble is fitted
                # usually, most of the methods are already fitted
                self.ensemble.fit(_X, _y)

                # get predictions on the residuals
                # only use the last/latest pipeline
                _best_estimator = self.ensemble.estimators[-1][1]
                _y_pred = _best_estimator.predict(_X)

        # make sure the ensemble is fitted
        # usually, every method is already fitted
        # but all pipelines need to be checked and set to fitted
        self.ensemble.fit(_X, _y)

        # if need to save the ensemble
        if self.save:
            save_methods(self.model_name, [self.ensemble])

        # whether to retain temp files
        if self.delete_temp_after_terminate:
            shutil.rmtree(self.temp_directory)

        self._logger.info(
            "[INFO] {}  Experiment: {}. Status: AutoTabular fitting finished.".format(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), self.model_name
            )
        )

        self._fitted = True

        return self
