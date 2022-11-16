"""
File: _base.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_hpo/_base.py
File: _base.py
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Tuesday, 15th November 2022 9:09:06 pm
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

formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

from typing import Union, List, Callable, Dict, Tuple
import os
import datetime
import copy
import shutil
import importlib
import warnings
import random
import numpy as np
import pandas as pd

# import ray
from ray import tune

from InsurAutoML._hpo._utils import (
    TabularObjective,
    Pipeline,
    ClassifierEnsemble,
    RegressorEnsemble,
)
from InsurAutoML._constant import UNI_CLASS, MAX_TIME, LOGGINGLEVEL
from InsurAutoML._base import no_processing
from InsurAutoML._utils._base import type_of_script, format_hyper_dict
from InsurAutoML._utils._file import (
    save_methods,
    load_methods,
    find_exact_path,
)
from InsurAutoML._utils._data import (
    str2list,
    str2dict,
)
from InsurAutoML._utils._metadata import MetaData
from InsurAutoML._utils._optimize import (
    get_algo,
    get_scheduler,
    get_logger,
    get_progress_reporter,
    TimePlateauStopper,
    ray_status,
    check_status,
)

# filter certain warnings
warnings.filterwarnings("ignore", message="The dataset is balanced, no change.")
warnings.filterwarnings("ignore", message="Variables are collinear")
# warnings.filterwarnings("ignore", message="Function checkpointing is disabled")
# warnings.filterwarnings(
#     "ignore", message="The TensorboardX logger cannot be instantiated"
# )
# Update: Nov 15, 2022
# autosklearn decrypted, sklearn new versions supported
# I wish to use sklearn v1.0 for new features
# but there's conflicts between autosklearn models and sklearn models
# mae <-> absolute_error, mse <-> squared_error inconsistency
warnings.filterwarnings("ignore", message="Criterion 'mse' was deprecated in v1.0")
warnings.filterwarnings("ignore", message="Criterion 'mae' was deprecated in v1.0")
warnings.filterwarnings("ignore", message="'normalize' was deprecated in version 1.0")
warnings.filterwarnings("ignore", category=UserWarning)

# check whether gpu device available
import importlib

torch_spec = importlib.util.find_spec("torch")
if torch_spec is not None:
    import torch

    device_count = torch.cuda.device_count()
else:
    device_count = 0


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


class AutoTabularBase(MetaData):

    """ "
    Base class module for AutoTabular (for classification and regression tasks)

    Parameters
    ----------
    task_mode: Mode of tasks, default: "classification"
    when called by AutoTabularClassification/AutoTabularRegression,
    task mode will be determined without reading data
    support ("classification", "regression")

    n_estimators: top k pipelines used to create the ensemble, default: 5

    ensemble_strategy: strategy of ensemble, default: "stacking"
    support ("stacking", "bagging", "boosting")

    timeout: Total time limit for the job in seconds, default = 360

    max_evals: Maximum number of function evaluations allowed, default = 32

    allow_error_prop: proportion of tasks allows failure, default = 0.1
    allowed number of failures is int(max_evals * allow_error_prop)

    temp_directory: folder path to store temporary model, default = 'tmp'

    delete_temp_after_terminate: whether to delete temporary information, default = False

    save: whether to save model after training, default = True

    model_name: saved model name, default = 'model'

    ignore_warning: whether to ignore warning, default = True

    encoder: Encoders selected for the job, default = 'auto'
    support ('DataEncoding')
    'auto' will select all default encoders, or use a list to select

    imputer: Imputers selected for the job, default = 'auto'
    support ('SimpleImputer', 'JointImputer', 'ExpectationMaximization', 'KNNImputer',
    'MissForestImputer', 'MICE', 'GAIN')
    'auto' will select all default imputers, or use a list to select

    balancing: Balancings selected for the job, default = 'auto'
    support ('no_processing', 'SimpleRandomOverSampling', 'SimpleRandomUnderSampling',
    'TomekLink', 'EditedNearestNeighbor', 'CondensedNearestNeighbor', 'OneSidedSelection',
    'CNN_TomekLink', 'Smote', 'Smote_TomekLink', 'Smote_ENN')
    'auto' will select all default balancings, or use a list to select

    scaling: Scalings selected for the job, default = 'auto'
    support ('no_processing', 'MinMaxScale', 'Standardize', 'Normalize', 'RobustScale',
    'PowerTransformer', 'QuantileTransformer', 'Winsorization')
    'auto' will select all default scalings, or use a list to select

    feature_selection: Feature selections selected for the job, default = 'auto'
    support ('no_processing', 'LDASelection', 'PCA_FeatureSelection', 'RBFSampler',
    'FeatureFilter', 'ASFFS', 'GeneticAlgorithm', 'extra_trees_preproc_for_classification',
    'fast_ica', 'feature_agglomeration', 'kernel_pca', 'kitchen_sinks',
    'liblinear_svc_preprocessor', 'nystroem_sampler', 'pca', 'polynomial',
    'random_trees_embedding', 'select_percentile_classification','select_rates_classification',
    'truncatedSVD')
    'auto' will select all default feature selections, or use a list to select

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

    validation: Whether to use train_test_split to test performance on test set, default = True

    valid_size: Test percentage used to evaluate the performance, default = 0.15
    only effective when validation = True

    objective: Objective function to test performance, default = 'accuracy'
    support metrics for regression ("MSE", "MAE", "MSLE", "R2", "MAX")
    support metrics for classification ("accuracy", "precision", "auc", "hinge", "f1")

    search_algo: search algorithm used for hyperparameter optimization, deafult = "HyperOpt"
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

    reset_index: whether to reset index during traning, default = True
    there are methods that are index independent (ignore index, resetted, e.g. GAIN)
    if you wish to use these methods and set reset_index = False, please make sure
    all input index are ordered and starting from 0

    seed: random seed, default = 1
    """

    def __init__(
        self,
        task_mode: str = "classification",
        n_estimators: int = 5,
        ensemble_strategy: str = "stacking",
        timeout: int = 360,
        max_evals: int = 64,
        allow_error_prop: float = 0.1,
        temp_directory: str = "tmp",
        delete_temp_after_terminate: bool = False,
        save: bool = True,
        model_name: str = "model",
        ignore_warning: bool = True,
        encoder: Union[str, List[str]] = "auto",
        imputer: Union[str, List[str]] = "auto",
        balancing: Union[str, List[str]] = "auto",
        scaling: Union[str, List[str]] = "auto",
        feature_selection: Union[str, List[str]] = "auto",
        models: Union[str, List[str]] = "auto",
        validation: bool = True,
        valid_size: float = 0.15,
        objective: Union[str, Callable] = "accuracy",
        search_algo: str = "HyperOpt",
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
        seed: int = 1,
    ) -> None:
        self.task_mode = task_mode
        self.n_estimators = n_estimators
        self.ensemble_strategy = ensemble_strategy
        self.timeout = timeout
        self.max_evals = max_evals
        self.allow_error_prop = allow_error_prop
        self.temp_directory = temp_directory
        self.delete_temp_after_terminate = delete_temp_after_terminate
        self.save = save
        self.model_name = model_name
        self.ignore_warning = ignore_warning
        self.encoder = encoder
        self.imputer = imputer
        self.balancing = balancing
        self.scaling = scaling
        self.feature_selection = feature_selection
        self.models = models
        self.validation = validation
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

        self._iter = 0  # record iteration number
        self._fitted = False  # record whether the model has been fitted

    def get_hyperparameter_space(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[Dict]:

        # initialize default search options
        # and select the search options based on the input restrictions
        # use copy to allows multiple manipulation

        # Encoding: convert string types to numerical type
        # all encoders available
        from InsurAutoML._encoding import encoders
        from additional import add_encoders

        # include original encoders
        self._all_encoders = copy.deepcopy(encoders)
        # include additional encoders
        self._all_encoders.update(add_encoders)

        # get default encoder methods space
        if self.encoder == "auto":
            encoder = copy.deepcopy(self._all_encoders)
        else:
            self.encoder = str2list(self.encoder)  # string list to list
            encoder = {}  # if specified, check if encoders in default encoders
            for _encoder in self.encoder:
                if _encoder not in [*self._all_encoders]:
                    self._logger.error(
                        "Only supported encoders are {}, get {}.".format(
                            [*self._all_encoders], _encoder
                        ),
                        ValueError,
                    )
                    # raise ValueError(
                    #     "Only supported encoders are {}, get {}.".format(
                    #         [*self._all_encoders], _encoder
                    #     )
                    # )
                encoder[_encoder] = self._all_encoders[_encoder]

        # Imputer: fill missing values
        # all imputers available
        from InsurAutoML._imputation import imputers
        from additional import add_imputers

        # include original imputers
        self._all_imputers = copy.deepcopy(imputers)
        # include additional imputers
        self._all_imputers.update(add_imputers)

        # special case: kNN imputer can not handle categorical data
        # remove kNN imputer from all imputers
        for _column in list(X.columns):
            if len(X[_column].unique()) <= min(0.1 * len(X), UNI_CLASS):
                del self._all_imputers["KNNImputer"]
                break

        # get default imputer methods space
        if self.imputer == "auto":
            if not X.isnull().values.any():  # if no missing values
                imputer = {"no_processing": no_processing}
                self._all_imputers = imputer  # limit default imputer space
            else:
                imputer = copy.deepcopy(self._all_imputers)
        else:
            self.imputer = str2list(self.imputer)  # string list to list
            if not X.isnull().values.any():  # if no missing values
                imputer = {"no_processing": no_processing}
                self._all_imputers = imputer
            else:
                imputer = {}  # if specified, check if imputers in default imputers
                for _imputer in self.imputer:
                    if _imputer not in [*self._all_imputers]:
                        raise ValueError(
                            "Only supported imputers are {}, get {}.".format(
                                [*self._all_imputers], _imputer
                            )
                        )
                    imputer[_imputer] = self._all_imputers[_imputer]

        # Balancing: deal with imbalanced dataset, using over-/under-sampling methods
        # all balancings available
        from InsurAutoML._balancing import balancings
        from additional import add_balancings

        # include original balancings
        self._all_balancings = copy.deepcopy(balancings)
        # include additional balancings
        self._all_balancings.update(add_balancings)

        # get default balancing methods space
        if self.balancing == "auto":
            balancing = copy.deepcopy(self._all_balancings)
        else:
            self.balancing = str2list(self.balancing)  # string list to list
            balancing = {}  # if specified, check if balancings in default balancings
            for _balancing in self.balancing:
                if _balancing not in [*self._all_balancings]:
                    raise ValueError(
                        "Only supported balancings are {}, get {}.".format(
                            [*self._all_balancings], _balancing
                        )
                    )
                balancing[_balancing] = self._all_balancings[_balancing]

        # Scaling
        # all scalings available
        from InsurAutoML._scaling import scalings
        from additional import add_scalings

        # include original scalings
        self._all_scalings = copy.deepcopy(scalings)
        # include additional scalings
        self._all_scalings.update(add_scalings)

        # get default scaling methods space
        if self.scaling == "auto":
            scaling = copy.deepcopy(self._all_scalings)
        else:
            self.scaling = str2list(self.scaling)  # string list to list
            scaling = {}  # if specified, check if scalings in default scalings
            for _scaling in self.scaling:
                if _scaling not in [*self._all_scalings]:
                    raise ValueError(
                        "Only supported scalings are {}, get {}.".format(
                            [*self._all_scalings], _scaling
                        )
                    )
                scaling[_scaling] = self._all_scalings[_scaling]

        # Feature selection: Remove redundant features, reduce dimensionality
        # all feature selections available
        from InsurAutoML._feature_selection import feature_selections
        from additional import add_feature_selections

        # include original feature selections
        self._all_feature_selection = copy.deepcopy(feature_selections)
        # include additional feature selections
        self._all_feature_selection.update(add_feature_selections)

        if self.task_mode == "classification":
            # special treatment, if classification
            # remove some feature selection for regression
            del self._all_feature_selection["extra_trees_preproc_for_regression"]
            del self._all_feature_selection["select_percentile_regression"]
            del self._all_feature_selection["select_rates_regression"]
        elif self.task_mode == "regression":
            # special treatment, if regression
            # remove some feature selection for classification
            del self._all_feature_selection["extra_trees_preproc_for_classification"]
            del self._all_feature_selection["select_percentile_classification"]
            del self._all_feature_selection["select_rates_classification"]

        if X.shape[0] * X.shape[1] > 10000 or self.task_mode == "regression":
            del self._all_feature_selection["liblinear_svc_preprocessor"]

        # get default feature selection methods space
        if self.feature_selection == "auto":
            feature_selection = copy.deepcopy(self._all_feature_selection)
        else:
            self.feature_selection = str2list(
                self.feature_selection
            )  # string list to list
            feature_selection = (
                {}
            )  # if specified, check if balancings in default balancings
            for _feature_selection in self.feature_selection:
                if _feature_selection not in [*self._all_feature_selection]:
                    raise ValueError(
                        "Only supported feature selections are {}, get {}.".format(
                            [*self._all_feature_selection], _feature_selection
                        )
                    )
                feature_selection[_feature_selection] = self._all_feature_selection[
                    _feature_selection
                ]

        # Model selection/Hyperparameter optimization
        # using Bayesian Optimization
        # all models available
        # if mode is classification, use classification models
        # if mode is regression, use regression models
        if self.task_mode == "classification":
            from InsurAutoML._model import classifiers
            from additional import add_classifiers

            # include original classifiers
            self._all_models = copy.deepcopy(classifiers)
            # include additional classifiers
            self._all_models.update(add_classifiers)
        elif self.task_mode == "regression":
            from InsurAutoML._model import regressors
            from additional import add_regressors

            # include original regressors
            self._all_models = copy.deepcopy(regressors)
            # include additional regressors
            self._all_models.update(add_regressors)

        # special treatment, remove SVM methods when observations are large
        # SVM suffers from the complexity o(n_samples^2 * n_features),
        # which is time-consuming for large datasets
        if X.shape[0] * X.shape[1] > 10000:
            # in case the methods are not included, will check before delete
            if self.task_mode == "classification":
                del self._all_models["LibLinear_SVC"]
                del self._all_models["LibSVM_SVC"]
            elif self.task_mode == "regression":
                del self._all_models["LibLinear_SVR"]
                del self._all_models["LibSVM_SVR"]

        # model space, only select chosen models to space
        if self.models == "auto":  # if auto, model pool will be all default models
            models = copy.deepcopy(self._all_models)
        else:
            self.models = str2list(self.models)  # string list to list
            models = {}  # if specified, check if models in default models
            for _model in self.models:
                if _model not in [*self._all_models]:
                    raise ValueError(
                        "Only supported models are {}, get {}.".format(
                            [*self._all_models], _model
                        )
                    )
                models[_model] = self._all_models[_model]

        # # initialize model hyperparameter space
        # _all_models_hyperparameters = copy.deepcopy(self._all_models_hyperparameters)

        # initialize default search space
        from InsurAutoML._utils._optimize import _get_hyperparameter_space

        from InsurAutoML._hyperparameters import (
            encoder_hyperparameter,
            imputer_hyperparameter,
            scaling_hyperparameter,
            balancing_hyperparameter,
            feature_selection_hyperparameter,
            classifier_hyperparameter,
            regressor_hyperparameter,
        )

        from additional import (
            add_encoder_hyperparameter,
            add_imputer_hyperparameter,
            add_scaling_hyperparameter,
            add_balancing_hyperparameter,
            add_feature_selection_hyperparameter,
            add_classifier_hyperparameter,
            add_regressor_hyperparameter,
        )

        # if needed, modify default hyperparameter space
        # like model hyperparameter space below
        # all hyperparameters for encoders
        _all_encoders_hyperparameters = copy.deepcopy(encoder_hyperparameter)
        # include additional hyperparameters
        _all_encoders_hyperparameters += add_encoder_hyperparameter

        # # initialize encoders hyperparameter space
        # _all_encoders_hyperparameters = copy.deepcopy(self._all_encoders_hyperparameters)

        # all hyperparameters for imputers
        if "no_processing" in imputer.keys():
            _all_imputers_hyperparameters = [{"imputer": "no_processing"}]
        else:
            _all_imputers_hyperparameters = copy.deepcopy(imputer_hyperparameter)
        # include additional hyperparameters
        _all_imputers_hyperparameters += add_imputer_hyperparameter

        # # initialize imputers hyperparameter space
        # _all_imputers_hyperparameters = copy.deepcopy(self._all_imputers_hyperparameters)

        # all hyperparameters for balancing methods
        _all_balancings_hyperparameters = copy.deepcopy(balancing_hyperparameter)
        # include additional hyperparameters
        _all_balancings_hyperparameters += add_balancing_hyperparameter

        # # initialize balancing hyperparameter space
        # _all_balancings_hyperparameters = copy.deepcopy(self._all_balancings_hyperparameters)

        # all hyperparameters for scalings
        _all_scalings_hyperparameters = copy.deepcopy(scaling_hyperparameter)
        # include additional hyperparameters
        _all_scalings_hyperparameters += add_scaling_hyperparameter

        # # initialize scaling hyperparameter space
        # _all_scalings_hyperparameters = copy.deepcopy(self._all_scalings_hyperparameters)

        # all hyperparameters for feature selections
        _all_feature_selection_hyperparameters = copy.deepcopy(
            feature_selection_hyperparameter
        )
        # include additional hyperparameters
        _all_feature_selection_hyperparameters += add_feature_selection_hyperparameter

        # special treatment, for SFS hyperparameter space
        if self.task_mode == "classification":
            for item in _all_feature_selection_hyperparameters:
                if "SFS" in item.values():
                    from InsurAutoML._constant import (
                        CLASSIFICATION_ESTIMATORS,
                        CLASSIFICATION_CRITERIA,
                    )

                    item["estimator"] = tune.choice(CLASSIFICATION_ESTIMATORS)
                    item["criteria"] = tune.choice(CLASSIFICATION_CRITERIA)
                    break
        elif self.task_mode == "regression":
            for item in _all_feature_selection_hyperparameters:
                if "SFS" in item.values():
                    from InsurAutoML._constant import (
                        REGRESSION_ESTIMATORS,
                        REGRESSION_CRITERIA,
                    )

                    item["estimator"] = tune.choice(REGRESSION_ESTIMATORS)
                    item["criteria"] = tune.choice(REGRESSION_CRITERIA)
                    break

        # # initialize feature selection hyperparameter space
        # _all_feature_selection_hyperparameters = copy.deepcopy(
        #     self._all_feature_selection_hyperparameters
        # )

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
                        from InsurAutoML._constant import LIGHTGBM_BINARY_CLASSIFICATION

                        item["objective"] = tune.choice(LIGHTGBM_BINARY_CLASSIFICATION)
                    else:
                        from InsurAutoML._constant import (
                            LIGHTGBM_MULTICLASS_CLASSIFICATION,
                        )

                        item["objective"] = tune.choice(
                            LIGHTGBM_MULTICLASS_CLASSIFICATION
                        )

        # check status of hyperparameter space
        check_status(encoder, _all_encoders_hyperparameters, ref="encoder")
        check_status(imputer, _all_imputers_hyperparameters, ref="imputer")
        check_status(balancing, _all_balancings_hyperparameters, ref="balancing")
        check_status(scaling, _all_scalings_hyperparameters, ref="scaling")
        check_status(
            feature_selection,
            _all_feature_selection_hyperparameters,
            ref="feature_selection",
        )
        check_status(models, _all_models_hyperparameters, ref="model")

        # format default search space
        _all_encoders_hyperparameters = [
            format_hyper_dict(
                dict, order + 1, ref="encoder", search_algo=self.search_algo
            )
            for order, dict in enumerate(_all_encoders_hyperparameters)
        ]
        _all_imputers_hyperparameters = [
            format_hyper_dict(
                dict, order + 1, ref="imputer", search_algo=self.search_algo
            )
            for order, dict in enumerate(_all_imputers_hyperparameters)
        ]
        _all_balancings_hyperparameters = [
            format_hyper_dict(
                dict, order + 1, ref="balancing", search_algo=self.search_algo
            )
            for order, dict in enumerate(_all_balancings_hyperparameters)
        ]
        _all_scalings_hyperparameters = [
            format_hyper_dict(
                dict, order + 1, ref="scaling", search_algo=self.search_algo
            )
            for order, dict in enumerate(_all_scalings_hyperparameters)
        ]
        _all_feature_selection_hyperparameters = [
            format_hyper_dict(
                dict, order + 1, ref="feature_selection", search_algo=self.search_algo
            )
            for order, dict in enumerate(_all_feature_selection_hyperparameters)
        ]
        _all_models_hyperparameters = [
            format_hyper_dict(
                dict, order + 1, ref="model", search_algo=self.search_algo
            )
            for order, dict in enumerate(_all_models_hyperparameters)
        ]

        # generate the hyperparameter space
        hyperparameter_space = _get_hyperparameter_space(
            X,
            _all_encoders_hyperparameters,
            encoder,
            _all_imputers_hyperparameters,
            imputer,
            _all_balancings_hyperparameters,
            balancing,
            _all_scalings_hyperparameters,
            scaling,
            _all_feature_selection_hyperparameters,
            feature_selection,
            _all_models_hyperparameters,
            models,
            self.task_mode,
            self.search_algo,
        )  # _X to choose whether include imputer
        # others are the combinations of default hyperparameter space & methods selected

        return (
            encoder,
            imputer,
            balancing,
            scaling,
            feature_selection,
            models,
            hyperparameter_space,
        )

    # method is deprecated as pickle save/load methods are now supported
    # load hyperparameter settings and train on the data
    # def load_model(self, _X, _y):

    #     # load hyperparameter settings
    #     with open(self.model_name) as f:
    #         optimal_setting = f.readlines()

    #     # remove change line signs
    #     optimal_setting = [item.replace("\n", "") for item in optimal_setting]
    #     # remove blank spaces
    #     while "" in optimal_setting:
    #         optimal_setting.remove("")

    #     # convert strings to readable dictionaries
    #     self.optimal_encoder = optimal_setting[0]
    #     self.optimal_encoder_hyperparameters = ast.literal_eval(optimal_setting[1])
    #     self.optimal_imputer = optimal_setting[2]
    #     self.optimal_imputer_hyperparameters = ast.literal_eval(optimal_setting[3])
    #     self.optimal_balancing = optimal_setting[4]
    #     self.optimal_balancing_hyperparameters = ast.literal_eval(optimal_setting[5])
    #     self.optimal_scaling = optimal_setting[6]
    #     self.optimal_scaling_hyperparameters = ast.literal_eval(optimal_setting[7])
    #     self.optimal_feature_selection = optimal_setting[8]
    #     self.optimal_feature_selection_hyperparameters = ast.literal_eval(
    #         optimal_setting[9]
    #     )
    #     self.optimal_model = optimal_setting[10]
    #     self.optimal_model_hyperparameters = ast.literal_eval(optimal_setting[11])

    #     # map the methods and hyperparameters
    #     # fit the methods
    #     # encoding
    #     self._fit_encoder = self._all_encoders[self.optimal_encoder](
    #         **self.optimal_encoder_hyperparameters
    #     )
    #     _X = self._fit_encoder.fit(_X)
    #     # imputer
    #     self._fit_imputer = self._all_imputers[self.optimal_imputer](
    #         **self.optimal_imputer_hyperparameters
    #     )
    #     _X = self._fit_imputer.fill(_X)
    #     # balancing
    #     self._fit_balancing = self._all_balancings[self.optimal_balancing](
    #         **self.optimal_balancing_hyperparameters
    #     )
    #     _X, _y = self._fit_balancing.fit_transform(_X, _y)

    #     # make sure the classes are integers (belongs to certain classes)
    #     if self.task_mode == "classification":
    #         _y = _y.astype(int)
    #     # scaling
    #     self._fit_scaling = self._all_scalings[self.optimal_scaling](
    #         **self.optimal_scaling_hyperparameters
    #     )
    #     self._fit_scaling.fit(_X, _y)
    #     _X = self._fit_scaling.transform(_X)
    #     # feature selection
    #     self._fit_feature_selection = self._all_feature_selection[
    #         self.optimal_feature_selection
    #     ](**self.optimal_feature_selection_hyperparameters)
    #     self._fit_feature_selection.fit(_X, _y)
    #     _X = self._fit_feature_selection.transform(_X)
    #     # model
    #     self._fit_model = self._all_models[self.optimal_model](
    #         **self.optimal_model_hyperparameters
    #     )
    #     self._fit_model.fit(_X, _y.values.ravel())

    #     return self

    # select optimal settings and fit on optimal hyperparameters
    def _fit_optimal(
        self, idx: int, optimal_point: Dict, best_path: str
    ) -> Tuple(str, Pipeline):

        # optimal encoder
        optimal_encoder_hyperparameters = optimal_point["encoder"]
        # find optimal encoder key
        for _key in optimal_encoder_hyperparameters.keys():
            if "encoder" in _key:
                _encoder_key = _key
                break
        optimal_encoder = optimal_encoder_hyperparameters[_encoder_key]
        del optimal_encoder_hyperparameters[_encoder_key]
        # remove indications
        optimal_encoder_hyperparameters = {
            k.replace(optimal_encoder + "_", ""): optimal_encoder_hyperparameters[k]
            for k in optimal_encoder_hyperparameters
        }

        # optimal imputer
        optimal_imputer_hyperparameters = optimal_point["imputer"]
        # find optimal imputer key
        for _key in optimal_imputer_hyperparameters.keys():
            if "imputer" in _key:
                _imputer_key = _key
                break
        optimal_imputer = optimal_imputer_hyperparameters[_imputer_key]
        del optimal_imputer_hyperparameters[_imputer_key]
        # remove indications
        optimal_imputer_hyperparameters = {
            k.replace(optimal_imputer + "_", ""): optimal_imputer_hyperparameters[k]
            for k in optimal_imputer_hyperparameters
        }

        # optimal balancing
        optimal_balancing_hyperparameters = optimal_point["balancing"]
        # find optimal balancing key
        for _key in optimal_balancing_hyperparameters.keys():
            if "balancing" in _key:
                _balancing_key = _key
                break
        optimal_balancing = optimal_balancing_hyperparameters[_balancing_key]
        del optimal_balancing_hyperparameters[_balancing_key]
        # remove indications
        optimal_balancing_hyperparameters = {
            k.replace(optimal_balancing + "_", ""): optimal_balancing_hyperparameters[k]
            for k in optimal_balancing_hyperparameters
        }

        # optimal scaling
        optimal_scaling_hyperparameters = optimal_point["scaling"]
        # find optimal scaling key
        for _key in optimal_scaling_hyperparameters.keys():
            if "scaling" in _key:
                _scaling_key = _key
                break
        optimal_scaling = optimal_scaling_hyperparameters[_scaling_key]
        del optimal_scaling_hyperparameters[_scaling_key]
        # remove indications
        optimal_scaling_hyperparameters = {
            k.replace(optimal_scaling + "_", ""): optimal_scaling_hyperparameters[k]
            for k in optimal_scaling_hyperparameters
        }

        # optimal feature selection
        optimal_feature_selection_hyperparameters = optimal_point["feature_selection"]
        # find optimal feature_selection key
        for _key in optimal_feature_selection_hyperparameters.keys():
            if "feature_selection" in _key:
                _feature_selection_key = _key
                break
        optimal_feature_selection = optimal_feature_selection_hyperparameters[
            _feature_selection_key
        ]
        del optimal_feature_selection_hyperparameters[_feature_selection_key]
        # remove indications
        optimal_feature_selection_hyperparameters = {
            k.replace(
                optimal_feature_selection + "_", ""
            ): optimal_feature_selection_hyperparameters[k]
            for k in optimal_feature_selection_hyperparameters
        }

        # optimal classifier
        optimal_model_hyperparameters = optimal_point["model"]  # optimal model selected
        # find optimal model key
        for _key in optimal_model_hyperparameters.keys():
            if "model" in _key:
                _model_key = _key
                break
        optimal_model = optimal_model_hyperparameters[
            _model_key
        ]  # optimal hyperparameter settings selected
        del optimal_model_hyperparameters[_model_key]
        # remove indications
        optimal_model_hyperparameters = {
            k.replace(optimal_model + "_", ""): optimal_model_hyperparameters[k]
            for k in optimal_model_hyperparameters
        }

        # if already exists, use append mode
        # else, write mode
        if not os.path.exists(
            os.path.join(self.temp_directory, self.model_name, "optimal_setting.txt")
        ):
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
            f.write("Optimal imputation method is: {}\n".format(optimal_imputer))
            f.write("Optimal imputation hyperparameters:")
            print(optimal_imputer_hyperparameters, file=f, end="\n\n")
            f.write("Optimal balancing method is: {}\n".format(optimal_balancing))
            f.write("Optimal balancing hyperparameters:")
            print(optimal_balancing_hyperparameters, file=f, end="\n\n")
            f.write("Optimal scaling method is: {}\n".format(optimal_scaling))
            f.write("Optimal scaling hyperparameters:")
            print(optimal_scaling_hyperparameters, file=f, end="\n\n")
            f.write(
                "Optimal feature selection method is: {}\n".format(
                    optimal_feature_selection
                )
            )
            f.write("Optimal feature selection hyperparameters:")
            print(optimal_feature_selection_hyperparameters, file=f, end="\n\n")
            f.write("Optimal {} model is: {}\n".format(self.task_mode, optimal_model))
            f.write("Optimal {} hyperparameters:".format(self.task_mode))
            print(optimal_model_hyperparameters, file=f, end="\n\n")

        # method is deprecated as pickle save/load methods are now supported
        # # encoding
        # self._fit_encoder = self._all_encoders[self.optimal_encoder](
        #     **self.optimal_encoder_hyperparameters
        # )
        # _X = self._fit_encoder.fit(_X)

        # # imputer
        # self._fit_imputer = self._all_imputers[self.optimal_imputer](
        #     **self.optimal_imputer_hyperparameters
        # )
        # _X = self._fit_imputer.fill(_X)

        # # balancing
        # self._fit_balancing = self._all_balancings[self.optimal_balancing](
        #     **self.optimal_balancing_hyperparameters
        # )
        # _X, _y = self._fit_balancing.fit_transform(_X, _y)
        # # make sure the classes are integers (belongs to certain classes)
        # _y = _y.astype(int)
        # _y = _y.astype(int)

        # # scaling
        # self._fit_scaling = self._all_scalings[self.optimal_scaling](
        #     **self.optimal_scaling_hyperparameters
        # )
        # self._fit_scaling.fit(_X, _y)
        # _X = self._fit_scaling.transform(_X)

        # # feature selection
        # self._fit_feature_selection = self._all_feature_selection[
        #     self.optimal_feature_selection
        # ](**self.optimal_feature_selection_hyperparameters)
        # self._fit_feature_selection.fit(_X, _y)
        # _X = self._fit_feature_selection.transform(_X)

        # # model fitting
        # self._fit_model = self._all_models[self.optimal_model](
        #     **self.optimal_model_hyperparameters
        # )
        # self._fit_model.fit(_X, _y.values.ravel())

        (
            _fit_encoder,
            _fit_imputer,
            _fit_balancing,
            _fit_scaling,
            _fit_feature_selection,
            _fit_model,
        ) = load_methods(best_path)

        # # save the model
        # if self.save:
        #     # save_model(
        #     #     self.optimal_encoder,
        #     #     self.optimal_encoder_hyperparameters,
        #     #     self.optimal_imputer,
        #     #     self.optimal_imputer_hyperparameters,
        #     #     self.optimal_balancing,
        #     #     self.optimal_balancing_hyperparameters,
        #     #     self.optimal_scaling,
        #     #     self.optimal_scaling_hyperparameters,
        #     #     self.optimal_feature_selection,
        #     #     self.optimal_feature_selection_hyperparameters,
        #     #     self.optimal_model,
        #     #     self.optimal_model_hyperparameters,
        #     #     self.model_name,
        #     # )
        #     save_methods(
        #         self.model_name,
        #         [
        #             self._fit_encoder,
        #             self._fit_imputer,
        #             self._fit_balancing,
        #             self._fit_scaling,
        #             self._fit_feature_selection,
        #             self._fit_model,
        #         ],
        #     )

        # create a pipeline using loaded methods
        pip_setting = {
            "encoder": _fit_encoder,
            "imputer": _fit_imputer,
            "balancing": _fit_balancing,
            "scaling": _fit_scaling,
            "feature_selection": _fit_feature_selection,
            "model": _fit_model,
        }

        return ("pipe_" + str(idx + 1), Pipeline(**pip_setting))

    def _fit_ensemble(
        self, trial_id: str, config: Dict, iter: int = None, features: List[str] = None
    ) -> None:

        # initialize ensemble list
        try:
            # if ensemble list exists, append to it
            if len(ensemble_list) > 0 or len(feature_list) > 0:
                pass
        except:
            # else, initialize the list
            ensemble_list = []
            feature_list = []

        # if only one optimal input, need to convert to iterable
        if not isinstance(trial_id, pd.Series) or not isinstance(config, pd.Series):
            trial_id = [trial_id]
            config = [config]

        # loop through all configs, trial_id, get model ensemble
        for idx, (trial_id, config) in enumerate(zip(trial_id, config)):
            # find the exact path
            if iter is None:
                _path = find_exact_path(
                    os.path.join(
                        self.sub_directory,
                        self.model_name,
                    ),
                    "id_" + trial_id,
                )
                _path = os.path.join(_path, self.model_name)

                ensemble_list.append(self._fit_optimal(idx, config, _path))
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

                ensemble_list.append(self._fit_optimal(iter, config, _path))
            if (
                features is not None
            ):  # if feature subset is provided, save the feature subsets
                feature_list.append(features)

        # wrap pipelines into ensemble
        if self.task_mode == "classification":
            self._ensemble = ClassifierEnsemble(
                estimators=ensemble_list,
                features=feature_list,
                strategy=self.ensemble_strategy,
            )
        elif self.task_mode == "regression":
            self._ensemble = RegressorEnsemble(
                estimators=ensemble_list,
                features=feature_list,
                strategy=self.ensemble_strategy,
            )

    def fit(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> AutoTabularBase:

        # initialize temp directory
        # check if temp directory exists, if exists, empty it
        if os.path.isdir(os.path.join(self.temp_directory, self.model_name)):
            shutil.rmtree(os.path.join(self.temp_directory, self.model_name))
        if not os.path.isdir(self.temp_directory):
            os.makedirs(self.temp_directory)
        os.makedirs(os.path.join(self.temp_directory, self.model_name))

        # # set up logging file
        # if not os.path.exists(
        #     os.path.join(self.temp_directory, self.model_name, "logging.conf")
        # ):
        #     with open(
        #         os.path.join(self.temp_directory, self.model_name, "logging.conf"), "w"
        #     ) as fp:
        #         pass
        # logging.config.fileConfig(
        #     os.path.join(self.temp_directory, self.model_name, "logging.conf"),
        #     disable_existing_loggers=False,
        # )
        # # set up verbosity of logging
        self._logger = setup_logger(
            __name__,
            os.path.join(self.temp_directory, self.model_name, "logging.conf"),
            level=logging.INFO,
        )
        # logging.basicConfig(level=LOGGINGLEVEL[self.verbose])
        # self._logger = logging.getLogger(__name__)

        self._logger.info(
            "[INFO] {} Experiment: {}. Status: Start preparing AutoTabular...".format(
                datetime.datetime.now().strftime("%H:%M:%S %Y-%m-%d"), self.model_name
            )
        )

        if self.ignore_warning:  # ignore all warnings to generate clearer outputs
            warnings.filterwarnings("ignore")

        # convert to dataframe
        if not isinstance(X, pd.DataFrame):
            try:
                X = pd.DataFrame(X)
            except:
                raise TypeError("Cannot convert X to dataframe, get {}".format(type(X)))
        if not isinstance(y, pd.DataFrame):
            try:
                y = pd.DataFrame(y)
            except:
                raise TypeError("Cannot convert y to dataframe, get {}".format(type(y)))

        # get data metadata
        super(AutoTabularBase, self).__init__(X)
        # check if there's unsupported data type
        # if datetime ,recommend to remove
        if ("Datetime", "") in self.metadata.keys():
            warnings.warn(
                "Found datatime data type columns {}, it's better to remove those columns".format(
                    *self.metadata[("Datetime", "")]
                )
            )
        # TODO: when NLP and Image supported, redirect to corresponding model
        if ("Object", "Text") in self.metadata.keys():
            raise NotImplementedError("Text data type is not supported yet.")
        if ("Path", "") in self.metadata.keys():
            raise NotImplementedError("Image data type is not supported yet.")

        # get features and response names
        if isinstance(X, pd.DataFrame):  # expect multiple features
            self.features = list(X.columns)

        if isinstance(y, pd.DataFrame):  # for the case of dataframe
            self.response = list(y.columns)
        elif isinstance(y, pd.Series):  # for the case of series
            self.response = list(y.name)

        # get device info
        self.cpu_threads = (
            os.cpu_count() if self.cpu_threads is None else self.cpu_threads
        )
        # auto use_gpu selection if gpu is available
        self.use_gpu = (device_count > 0) if self.use_gpu is None else self.use_gpu
        # count gpu available
        self.gpu_count = device_count if self.use_gpu else 0

        # print warning if gpu available but not used
        if device_count > 0 and not self.use_gpu:
            warnings.warn(
                "You have {} GPU(s) available, but you have not set use_gpu to True, \
                which may drastically increase time to train neural networks.".format(
                    device_count
                )
            )
        # raise error if gpu not available but used
        if device_count == 0 and self.use_gpu:
            raise SystemError(
                "You have no GPU available, but you have set use_gpu to True. \
                Please check your GPU availability."
            )

        # make sure n_estimators is a integer smaller than max_evals
        # Oct. 11, 2022 updates:
        # if do not limit max_evals (=-1), then set n_estimators to pre-defined one
        if self.max_evals > 0:
            self.n_estimators = int(self.n_estimators)
        else:
            # raise warnings if n_estimators set larger than max_evals
            if self.max_evals < self.n_estimators:
                warnings.warn(
                    "n_estimators {} larger than max_evals {}, will be set to {}.".format(
                        self.n_estimators, self.max_evals, self.max_evals
                    )
                )
            self.n_estimators = int(min(self.n_estimators, self.max_evals))

        # at least one constraint of time/evaluations should be provided
        if self.timeout == -1 and self.max_evals == -1:
            warnings.warn(
                "None of time or evaluation constraint is provided, will set time limit to 1 hour."
            )
            self.timeout = 3600

        # make sure time budgets are controlled
        if self.timeout == -1:
            self.timeout = MAX_TIME
        else:
            if self.timeout > MAX_TIME:
                warnings.warn(
                    "Time budget is too long, will set time limit to {} seconds.".format(
                        MAX_TIME
                    )
                )
            self.timeout = min(self.timeout, MAX_TIME)

        # get progress report from environment
        # if specified, use specified progress report
        self.progress_reporter = (
            (
                "CLIReporter"
                if type_of_script() == "terminal"
                else "JupyterNotebookReporter"
            )
            if self.progress_reporter is None
            else self.progress_reporter
        )

        if self.progress_reporter not in ["CLIReporter", "JupyterNotebookReporter"]:
            raise TypeError(
                "Progress reporter must be either CLIReporter or JupyterNotebookReporter, get {}.".format(
                    self.progress_reporter
                )
            )

        # make sure _X is a dataframe
        if isinstance(X, pd.DataFrame):
            pass
        else:
            try:
                X = pd.DataFrame(X)
                self._logger.info(
                    "[INFO] {} Experiment: {}. Status: X is not a dataframe, converted to dataframe.".format(
                        datetime.datetime.now().strftime("%H:%M:%S %Y-%m-%d"),
                        self.model_name,
                    )
                )
            except:
                raise TypeError(
                    "X must be a dataframe! Can't convert {} to dataframe.".format(
                        type(X)
                    )
                )

        _X = X.copy()
        _y = y.copy()

        if self.reset_index:
            # reset index to avoid indexing error
            _X.reset_index(drop=True, inplace=True)
            _y.reset_index(drop=True, inplace=True)

        (
            encoder,
            imputer,
            balancing,
            scaling,
            feature_selection,
            models,
            hyperparameter_space,
        ) = self.get_hyperparameter_space(_X, _y)

        self._logger.info(
            "[INFO] {} Experiment: {}. Status: Initialized AutoTabular Hyperparameter space.".format(
                datetime.datetime.now().strftime("%H:%M:%S %Y-%m-%d"), self.model_name
            )
        )

        # print([item.sample() for key, item in hyperparameter_space.items() if key != "task_type"])

        # if the model is already trained, read the setting
        if os.path.exists(self.model_name):
            self._logger.info(
                "[INFO] {} Experiment: {}. Status: Stored model found, load previous model.".format(
                    datetime.datetime.now().strftime("%H:%M:%S %Y-%m-%d"),
                    self.model_name,
                )
            )
            # print(
            #     "[INFO] {} Stored model found, load previous model.".format(
            #         datetime.datetime.now().strftime("%H:%M:%S %Y-%m-%d")
            #     )
            # )
            # self.load_model(_X, _y)
            # (
            #     self._fit_encoder,
            #     self._fit_imputer,
            #     self._fit_balancing,
            #     self._fit_scaling,
            #     self._fit_feature_selection,
            #     self._fit_model,
            # ) = load_methods(self.model_name)
            [self._ensemble] = load_methods(self.model_name)

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

        # set random seed
        np.random.seed(self.seed)
        random.seed(self.seed)

        # get maximum allowed errors
        self.max_error = int(self.max_evals * self.allow_error_prop)

        # load dict settings for search_algo and search_scheduler
        self.search_algo_settings = str2dict(self.search_algo_settings)
        self.search_scheduler_settings = str2dict(self.search_scheduler_settings)

        # use ray for Model Selection and Hyperparameter Selection
        # get search algorithm
        algo = get_algo(self.search_algo)

        # special requirement for Nevergrad, need a algorithm setting
        if self.search_algo == "Nevergrad" and len(self.search_algo_settings) == 0:
            self._logger.warn(
                "No algorithm setting for Nevergrad find, use OnePlusOne."
            )
            # warnings.warn("No algorithm setting for Nevergrad find, use OnePlusOne.")
            import nevergrad as ng

            self.search_algo_settings = {"optimizer": ng.optimizers.OnePlusOne}

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
        stopper = TimePlateauStopper(
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

        # set ray status
        rayStatus = ray_status(
            cpu_threads=self.cpu_threads,
            gpu_count=self.gpu_count,
        )

        # ensemble settings
        if self.n_estimators == 1:
            self._logger.warn("Set n_estimators to 1, no ensemble will be used.")
            # warnings.warn("Set n_estimators to 1, no ensemble will be used.")

            # get progress reporter
            progress_reporter = get_progress_reporter(
                self.progress_reporter,
                self.max_evals,
                self.max_error,
            )

            # set trainable
            trainer = tune.with_parameters(
                TabularObjective,
                _X=_X,
                _y=_y,
                encoder=encoder,
                imputer=imputer,
                balancing=balancing,
                scaling=scaling,
                feature_selection=feature_selection,
                models=models,
                model_name=self.model_name,
                task_mode=self.task_mode,
                objective=self.objective,
                validation=self.validation,
                valid_size=self.valid_size,
                full_status=self.full_status,
                reset_index=self.reset_index,
                # timeout=self.timeout, # specified in stopper
                _iter=self._iter,
                seed=self.seed,
            )

            # initialize ray
            # # if already initialized, do nothing
            # if not ray.is_initialized():
            #     ray.init(
            #         # local_mode=True,
            #         num_cpus=self.cpu_threads,
            #         num_gpus=self.gpu_count,
            #     )
            # # check if ray is initialized
            # assert ray.is_initialized() == True, "Ray is not initialized."
            rayStatus.ray_init()

            # subtrial directory
            self.sub_directory = self.temp_directory

            self._logger.info(
                "[INFO] {}  Experiment: {}. Status: Start AutoTabular training.".format(
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    self.model_name,
                )
            )

            # optimization process
            # partially activated objective function
            # special treatment for optuna, embed search space in search algorithm
            if self.search_algo in ["Optuna"]:
                fit_analysis = tune.run(
                    trainer,
                    # config=hyperparameter_space,
                    name=self.model_name,  # name of the tuning process, use model_name
                    resume="AUTO",
                    checkpoint_freq=8,  # disable checkpoint
                    checkpoint_at_end=True,
                    keep_checkpoints_num=4,
                    checkpoint_score_attr="loss",
                    # mode="min",  # always call a minimization process
                    search_alg=algo(
                        space=hyperparameter_space,
                        mode="min",  # always call a minimization process
                        metric="loss",
                        **self.search_algo_settings,
                    ),
                    scheduler=scheduler(**self.search_scheduler_settings),
                    reuse_actors=True,
                    raise_on_failed_trial=False,
                    # metric="loss",
                    num_samples=self.max_evals,
                    max_failures=self.max_error,
                    stop=stopper,  # use stopper
                    callbacks=logger,
                    # time_budget_s=self.timeout,  # included in stopper
                    progress_reporter=progress_reporter,
                    verbose=self.verbose,
                    trial_dirname_creator=trial_str_creator,
                    local_dir=self.sub_directory,
                    log_to_file=True,
                )
            else:
                fit_analysis = tune.run(
                    trainer,
                    config=hyperparameter_space,
                    name=self.model_name,  # name of the tuning process, use model_name
                    resume="AUTO",
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
                    log_to_file=True,
                )

            # shut down ray
            # ray.shutdown()
            # # check if ray is shutdown
            # assert ray.is_initialized() == False, "Ray is not shutdown."
            rayStatus.ray_shutdown()

            self._logger.info(
                "[INFO] {}  Experiment: {}. Status: AutoTabular training finished. Start postprocessing...".format(
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    self.model_name,
                )
            )

            # get the best config settings
            best_trial_id = str(
                fit_analysis.get_best_trial(
                    metric="loss", mode="min", scope="all"
                ).trial_id
            )
            # # find the exact path
            # best_path = find_exact_path(
            #     os.path.join(self.temp_directory, self.model_name), "id_" + best_trial_id
            # )
            # best_path = os.path.join(best_path, self.model_name)

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
                TabularObjective,
                _X=_X,
                _y=_y,
                encoder=encoder,
                imputer=imputer,
                balancing=balancing,
                scaling=scaling,
                feature_selection=feature_selection,
                models=models,
                model_name=self.model_name,
                task_mode=self.task_mode,
                objective=self.objective,
                validation=self.validation,
                valid_size=self.valid_size,
                full_status=self.full_status,
                reset_index=self.reset_index,
                # timeout=self.timeout, # specified in stopper
                _iter=self._iter,
                seed=self.seed,
            )

            # initialize ray
            rayStatus.ray_init()

            # subtrial directory
            self.sub_directory = self.temp_directory

            # optimization process
            # partially activated objective function
            # special treatment for optuna, embed search space in search algorithm
            if self.search_algo in ["Optuna"]:
                fit_analysis = tune.run(
                    trainer,
                    # config=hyperparameter_space,
                    name=self.model_name,  # name of the tuning process, use model_name
                    resume="AUTO",
                    checkpoint_freq=8,  # disable checkpoint
                    checkpoint_at_end=True,
                    keep_checkpoints_num=4,
                    checkpoint_score_attr="loss",
                    # mode="min",  # always call a minimization process
                    search_alg=algo(
                        space=hyperparameter_space,
                        mode="min",  # always call a minimization process
                        metric="loss",
                        **self.search_algo_settings,
                    ),
                    scheduler=scheduler(**self.search_scheduler_settings),
                    reuse_actors=True,
                    raise_on_failed_trial=False,
                    # metric="loss",
                    num_samples=self.max_evals,
                    max_failures=self.max_error,
                    stop=stopper,  # use stopper
                    callbacks=logger,
                    # time_budget_s=self.timeout,  # included in stopper
                    progress_reporter=progress_reporter,
                    verbose=self.verbose,
                    trial_dirname_creator=trial_str_creator,
                    local_dir=self.sub_directory,
                    log_to_file=True,
                )
            else:
                fit_analysis = tune.run(
                    trainer,
                    config=hyperparameter_space,
                    name=self.model_name,  # name of the tuning process, use model_name
                    resume="AUTO",
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
                    log_to_file=True,
                )

            # shut down ray
            rayStatus.ray_shutdown()

            self._logger.info(
                "[INFO] {}  Experiment: {}. Status: AutoTabular training finished. Start postprocessing...".format(
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    self.model_name,
                )
            )

            # get all configs, trial_id
            analysis_df = fit_analysis.dataframe(metric="loss", mode="min")

            # reformat config to dict
            analysis_df["config"] = analysis_df.apply(
                lambda x: {
                    "encoder": x["config/encoder"],
                    "imputer": x["config/imputer"],
                    "balancing": x["config/balancing"],
                    "scaling": x["config/scaling"],
                    "feature_selection": x["config/feature_selection"],
                    "model": x["config/model"],
                },
                axis=1,
            )
            # sort by loss and get top configs
            analysis_df = analysis_df.sort_values(by=["loss"], ascending=True).head(
                self.n_estimators
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
                    TabularObjective,
                    _X=_X[feature_subset],
                    _y=_y,
                    encoder=encoder,
                    imputer=imputer,
                    balancing=balancing,
                    scaling=scaling,
                    feature_selection=feature_selection,
                    models=models,
                    model_name=self.model_name,
                    task_mode=self.task_mode,
                    objective=self.objective,
                    validation=self.validation,
                    valid_size=self.valid_size,
                    full_status=self.full_status,
                    reset_index=self.reset_index,
                    # timeout=self.timeout, # specified in stopper
                    _iter=self._iter,
                    seed=self.seed,
                )

                # initialize ray
                rayStatus.ray_init()

                # subtrial directory
                self.sub_directory = os.path.join(self.temp_directory, self.model_name)

                # optimization process
                # partially activated objective function
                # special treatment for optuna, embed search space in search algorithm
                if self.search_algo in ["Optuna"]:
                    fit_analysis = tune.run(
                        trainer,
                        # config=hyperparameter_space,
                        name=self.model_name
                        + "_"
                        + self.ensemble_strategy
                        + "_"
                        + str(_n + 1),  # name of the tuning process, use model_name
                        resume="AUTO",
                        checkpoint_freq=8,  # disable checkpoint
                        checkpoint_at_end=True,
                        keep_checkpoints_num=4,
                        checkpoint_score_attr="loss",
                        # mode="min",  # always call a minimization process
                        search_alg=algo(
                            space=hyperparameter_space,
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
                        log_to_file=True,
                    )
                else:
                    fit_analysis = tune.run(
                        trainer,
                        config=hyperparameter_space,
                        name=self.model_name
                        + "_"
                        + self.ensemble_strategy
                        + "_"
                        + str(_n + 1),  # name of the tuning process, use model_name
                        resume="AUTO",
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
                        log_to_file=True,
                    )

                # shut down ray
                rayStatus.ray_shutdown()

                self._logger.info(
                    "[INFO] {}  Experiment: {}. Status: AutoTabular training finished. Start postprocessing...".format(
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        self.model_name,
                    )
                )

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
                    _y_resid -= _y_pred
                except:
                    # if not, use y as residuals
                    _y_resid = _y

                # get progress reporter
                progress_reporter = get_progress_reporter(
                    self.progress_reporter,
                    self.max_evals,
                    self.max_error,
                )

                # set trainable
                trainer = tune.with_parameters(
                    TabularObjective,
                    _X=_X,
                    _y=_y_resid,
                    encoder=encoder,
                    imputer=imputer,
                    balancing=balancing,
                    scaling=scaling,
                    feature_selection=feature_selection,
                    models=models,
                    model_name=self.model_name,
                    task_mode=self.task_mode,
                    objective=self.objective,
                    validation=self.validation,
                    valid_size=self.valid_size,
                    full_status=self.full_status,
                    reset_index=self.reset_index,
                    # timeout=self.timeout, # specified in stopper
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
                        + str(_n + 1),  # name of the tuning process, use model_name
                        resume="AUTO",
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
                        log_to_file=True,
                    )
                else:
                    fit_analysis = tune.run(
                        trainer,
                        config=hyperparameter_space,
                        name=self.model_name
                        + "_"
                        + self.ensemble_strategy
                        + "_"
                        + str(_n + 1),  # name of the tuning process, use model_name
                        resume="AUTO",
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
                        log_to_file=True,
                    )

                # shut down ray
                rayStatus.ray_shutdown()

                self._logger.info(
                    "[INFO] {}  Experiment: {}. Status: AutoTabular training finished. Start postprocessing...".format(
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        self.model_name,
                    )
                )

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
                self._ensemble.fit(_X, _y)

                # get predictions on the residuals
                # only use the last/latest pipeline
                _y_pred = self._ensemble.estimators[-1][1].predict(_X)

        # make sure the ensemble is fitted
        # usually, every method is already fitted
        # but all pipelines need to be checked and set to fitted
        self._ensemble.fit(_X, _y)

        # if need to save the ensemble
        if self.save:
            save_methods(self.model_name, [self._ensemble])

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

    def predict(self, X: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

        if self.reset_index:
            # reset index to avoid indexing error
            X.reset_index(drop=True, inplace=True)

        # check if fitted
        if not self._fitted:
            raise ValueError("Pipeline not fitted yet! Call fit() first.")

        _X = X.copy()

        # since pipeline is converted to ensemble, no need to predict on each component
        # may need preprocessing for test data, the preprocessing should be the same as in fit part
        # Encoding
        # # convert string types to numerical type
        # _X = self._fit_encoder.refit(_X)

        # # Imputer
        # # fill missing values
        # _X = self._fit_imputer.fill(_X)

        # # Balancing
        # # deal with imbalanced dataset, using over-/under-sampling methods
        # # No need to balance on test data

        # # Scaling
        # _X = self._fit_scaling.transform(_X)

        # # Feature selection
        # # Remove redundant features, reduce dimensionality
        # _X = self._fit_feature_selection.transform(_X)

        # # use model to predict
        # return self._fit_model.predict(_X)

        return self._ensemble.predict(_X)
