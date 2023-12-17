"""
File Name: ML.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.5
Relative Path: /InsurAutoML/hpo/ML.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Friday, 1st December 2023 6:43:25 pm
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
from typing import Union, List, Callable, Dict
import numpy as np
import pandas as pd

from .base import AutoTabularBase
from ..utils.base import type_of_task


class AutoTabularRegressor(AutoTabularBase):

    """ "
    AutoTabular for regression tasks build on top of AutoTabularBase.

    Parameters
    ----------
    n_estimators: top k pipelines used to create the ensemble, default: 5

    ensemble_strategy: strategy of ensemble, default: "stacking"
    support ("stacking", "bagging", "boosting")

    voting: voting method used for ensemble, default: "mean"

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
    support regressors ("AdaboostRegressor", "ARDRegression", "DecisionTree",
            "ExtraTreesRegressor", "GaussianProcess", "GradientBoosting",
            "KNearestNeighborsRegressor", "LibLinear_SVR", "LibSVM_SVR",
            "MLPRegressor", "RandomForest", "SGD")
    'auto' will select all default models, or use a list to select

    exclude: components to exclude, default = {}
    keys are components, values are lists of components to exclude
    example: {'encoder': ['DataEncoding'], 'imputer': ['SimpleImputer', 'JointImputer']}

    validation: Whether to use train_test_split to test performance on test set, default = True
    optional KFold, K is inverse of valid_size (K = int(1 / valid_size)))

    valid_size: Test percentage used to evaluate the performance, default = 0.2
    only effective when validation = True or "KFold"

    objective: Objective function to test performance, default = 'accuracy'
    support metrics for regression ("MSE", "MAE", "MSLE", "R2", "MAX")

    search_algo: search algorithm used for hyperparameter optimization, deafult = "RandomSearch"
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
        encoder: Union[str, List[str]] = "auto",
        imputer: Union[str, List[str]] = "auto",
        balancing: Union[str, List[str]] = "auto",
        scaling: Union[str, List[str]] = "auto",
        feature_selection: Union[str, List[str]] = "auto",
        models: Union[str, List[str]] = "auto",
        exclude: Dict = {},
        validation: Union[bool, str] = True,
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
        self.encoder = encoder
        self.imputer = imputer
        self.balancing = balancing
        self.scaling = scaling
        self.feature_selection = feature_selection
        self.models = models
        self.exclude = exclude
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

        self._fitted = False  # whether the model has been fitted

        super(AutoTabularRegressor, self).__init__(
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
            encoder=self.encoder,
            imputer=self.imputer,
            balancing=self.balancing,
            scaling=self.scaling,
            feature_selection=self.feature_selection,
            models=self.models,
            exclude=self.exclude,
            validation=self.validation,
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


class AutoTabularClassifier(AutoTabularBase):

    """ "
    AutoTabular for classification tasks build on top of AutoTabularBase

    Parameters
    ----------
    n_estimators: top k pipelines used to create the ensemble, default: 5

    ensemble_strategy: strategy of ensemble, default: "stacking"
    support ("stacking", "bagging", "boosting")

    voting: voting method used for ensemble, default: "hard"

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
    'auto' will select all default models, or use a list to select

    exclude: components to exclude, default = {}
    keys are components, values are lists of components to exclude
    example: {'encoder': ['DataEncoding'], 'imputer': ['SimpleImputer', 'JointImputer']}

    validation: Whether to use train_test_split to test performance on test set, default = True
    optional KFold, K is inverse of valid_size (K = int(1 / valid_size)))

    valid_size: Test percentage used to evaluate the performance, default = 0.2
    only effective when validation = True or "KFold"

    objective: Objective function to test performance, default = 'accuracy'
    support metrics for classification ("accuracy", "precision", "auc", "hinge", "f1")

    search_algo: search algorithm used for hyperparameter optimization, deafult = "RandomSearch"
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
        encoder: Union[str, List[str]] = "auto",
        imputer: Union[str, List[str]] = "auto",
        balancing: Union[str, List[str]] = "auto",
        scaling: Union[str, List[str]] = "auto",
        feature_selection: Union[str, List[str]] = "auto",
        models: Union[str, List[str]] = "auto",
        exclude: Dict = {},
        validation: Union[bool, str] = True,
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
        self.encoder = encoder
        self.imputer = imputer
        self.balancing = balancing
        self.scaling = scaling
        self.feature_selection = feature_selection
        self.models = models
        self.exclude = exclude
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

        self._fitted = False  # whether the model has been fitted

        super(AutoTabularClassifier, self).__init__(
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
            encoder=self.encoder,
            imputer=self.imputer,
            balancing=self.balancing,
            scaling=self.scaling,
            feature_selection=self.feature_selection,
            models=self.models,
            exclude=self.exclude,
            validation=self.validation,
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


class AutoTabular(AutoTabularBase):

    """
    AutoTabular that automatically assign to AutoTabularClassifier or AutoTabularRegressor

    Parameters
    ----------
    n_estimators: top k pipelines used to create the ensemble, default: 5

    ensemble_strategy: strategy of ensemble, default: "stacking"
    support ("stacking", "bagging", "boosting")

    voting: voting method used for ensemble, default: None
    if None, use "soft" for classification, "mean" for regression

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

    exclude: components to exclude, default = {}
    keys are components, values are lists of components to exclude
    example: {'encoder': ['DataEncoding'], 'imputer': ['SimpleImputer', 'JointImputer']}

    validation: Whether to use train_test_split to test performance on test set, default = True
    optional KFold, K is inverse of valid_size (K = int(1 / valid_size)))

    valid_size: Test percentage used to evaluate the performance, default = 0.2
    only effective when validation = True or "KFold"

    objective: Objective function to test performance, default = 'accuracy'
    support metrics for regression ("MSE", "MAE", "MSLE", "R2", "MAX")
    support metrics for classification ("accuracy", "precision", "auc", "hinge", "f1")

    search_algo: search algorithm used for hyperparameter optimization, deafult = "RandomSearch"
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
        encoder: Union[str, List[str]] = "auto",
        imputer: Union[str, List[str]] = "auto",
        balancing: Union[str, List[str]] = "auto",
        scaling: Union[str, List[str]] = "auto",
        feature_selection: Union[str, List[str]] = "auto",
        models: Union[str, List[str]] = "auto",
        exclude: Dict = {},
        validation: Union[bool, str] = True,
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
        self.encoder = encoder
        self.imputer = imputer
        self.balancing = balancing
        self.scaling = scaling
        self.feature_selection = feature_selection
        self.models = models
        self.exclude = exclude
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
        self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series, np.ndarray] = None
    ) -> AutoTabular:
        if isinstance(y, (pd.DataFrame, pd.Series, np.ndarray)):
            self._type = type_of_task(y)
        elif not y:
            self._type = "unsupervised"

        super(AutoTabular, self).__init__(
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
            encoder=self.encoder,
            imputer=self.imputer,
            balancing=self.balancing,
            scaling=self.scaling,
            feature_selection=self.feature_selection,
            models=self.models,
            exclude=self.exclude,
            validation=self.validation,
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

        super(AutoTabular, self).fit(X, y)

        return self

    def predict(self, X: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        # check if the model has been fitted
        if not self._fitted:
            raise ValueError("No tasks found! Need to fit first.")

        return super(AutoTabular, self).predict(X)

    def predict_proba(
        self, X: pd.DataFrame
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        # check if the model has been fitted
        if not self._fitted:
            raise ValueError("No tasks found! Need to fit first.")

        return super(AutoTabular, self).predict_proba(X)
