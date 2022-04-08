"""
File: _ML.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Relative Path: /My_AutoML/_model_selection/_ML.py
File Created: Tuesday, 5th April 2022 10:50:27 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Thursday, 7th April 2022 11:28:46 pm
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

import numpy as np
import pandas as pd

from ._base import AutoTabularBase

from My_AutoML._utils._base import type_of_task


class AutoTabularRegressor(AutoTabularBase):

    """ "
    AutoTabular for regression tasks build on top of AutoTabularBase.

    Parameters
    ----------
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

    search_algo: search algorithm used for hyperparameter optimization, deafult = "HyperOpt"
    support ("RandomSearch", "GridSearch", "BayesOptSearch", "AxSearch", "BOHB",
            "BlendSearch", "CFO", "DragonflySearch", "HEBO", "HyperOpt", "Nevergrad",
            "Optuna", "SigOpt", "Scikit-Optimize", "ZOOpt", "Reapter",
            "ConcurrencyLimiter", callable)

    search_algo_setttings: search algorithm settings, default = {}
    need manual configuration for each search algorithm

    search_scheduler: search scheduler used, default = "FIFOScheduler"
    support ("FIFOScheduler", "ASHAScheduler", "HyperBandScheduler", "MedianStoppingRule"
            "PopulationBasedTraining", "PopulationBasedTrainingReplay", "PB2",
            "HyperBandForBOHB", callable)

    search_scheduler_settings: search scheduler settings, default = {}
    need manual configuration for each search scheduler

    progress_reporter: progress reporter, default = "CLIReporter"
    support ("CLIReporter", "JupyterNotebookReporter")

    full_status: whether to print full status, default = False

    verbose: display for output, default = 1
    support (0, 1, 2, 3)

    cpu_threads: number of cpu threads to use, default = None
    if None, get all available cpu threads

    use_gpu: whether to use gpu, default = False

    seed: random seed, default = 1
    """

    def __init__(
        self,
        timeout=360,
        max_evals=64,
        allow_error_prop=0.1,
        temp_directory="tmp",
        delete_temp_after_terminate=False,
        save=True,
        model_name="model",
        ignore_warning=True,
        encoder="auto",
        imputer="auto",
        balancing="auto",
        scaling="auto",
        feature_selection="auto",
        models="auto",
        validation=True,
        valid_size=0.15,
        objective="MSE",
        search_algo="HyperOpt",
        search_algo_setttings={},
        search_scheduler="FIFOScheduler",
        search_scheduler_settings={},
        progress_reporter="CLIReporter",
        full_status=False,
        verbose=1,
        cpu_threads=None,
        use_gpu=False,
        seed=1,
    ):
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
        self.search_algo_setttings = search_algo_setttings
        self.search_scheduler = search_scheduler
        self.search_scheduler_settings = search_scheduler_settings
        self.progress_reporter = progress_reporter
        self.full_status = full_status
        self.verbose = verbose
        self.cpu_threads = cpu_threads
        self.use_gpu = use_gpu
        self.seed = seed

        super().__init__(
            mode="regression",
            timeout=self.timeout,
            max_evals=self.max_evals,
            allow_error_prop=self.allow_error_prop,
            temp_directory=self.temp_directory,
            delete_temp_after_terminate=self.delete_temp_after_terminate,
            save=self.save,
            model_name=self.model_name,
            ignore_warning=self.ignore_warning,
            encoder=self.encoder,
            imputer=self.imputer,
            balancing=self.balancing,
            scaling=self.scaling,
            feature_selection=self.feature_selection,
            models=self.models,
            validation=self.validation,
            valid_size=self.valid_size,
            objective=self.objective,
            search_algo=self.search_algo,
            search_algo_setttings=self.search_algo_setttings,
            search_scheduler=self.search_scheduler,
            search_scheduler_settings=self.search_scheduler_settings,
            progress_reporter=self.progress_reporter,
            full_status=self.full_status,
            verbose=self.verbose,
            cpu_threads=self.cpu_threads,
            use_gpu=self.use_gpu,
            seed=self.seed,
        )

    def fit(self, X, y):

        return super().fit(X, y)

    def predict(self, X):

        return super().predict(X)


class AutoTabularClassifier(AutoTabularBase):

    """ "
    AutoTabular for classification tasks build on top of AutoTabularBase

    Parameters
    ----------
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
    'auto' will select all default models, or use a list to select

    validation: Whether to use train_test_split to test performance on test set, default = True

    valid_size: Test percentage used to evaluate the performance, default = 0.15
    only effective when validation = True

    objective: Objective function to test performance, default = 'accuracy'
    support metrics for classification ("accuracy", "precision", "auc", "hinge", "f1")

    search_algo: search algorithm used for hyperparameter optimization, deafult = "HyperOpt"
    support ("RandomSearch", "GridSearch", "BayesOptSearch", "AxSearch", "BOHB",
            "BlendSearch", "CFO", "DragonflySearch", "HEBO", "HyperOpt", "Nevergrad",
            "Optuna", "SigOpt", "Scikit-Optimize", "ZOOpt", "Reapter",
            "ConcurrencyLimiter", callable)

    search_algo_setttings: search algorithm settings, default = {}
    need manual configuration for each search algorithm

    search_scheduler: search scheduler used, default = "FIFOScheduler"
    support ("FIFOScheduler", "ASHAScheduler", "HyperBandScheduler", "MedianStoppingRule"
            "PopulationBasedTraining", "PopulationBasedTrainingReplay", "PB2",
            "HyperBandForBOHB", callable)

    search_scheduler_settings: search scheduler settings, default = {}
    need manual configuration for each search scheduler

    progress_reporter: progress reporter, default = "CLIReporter"
    support ("CLIReporter", "JupyterNotebookReporter")

    full_status: whether to print full status, default = False

    verbose: display for output, default = 1
    support (0, 1, 2, 3)

    cpu_threads: number of cpu threads to use, default = None
    if None, get all available cpu threads

    use_gpu: whether to use gpu, default = False

    seed: random seed, default = 1
    """

    def __init__(
        self,
        timeout=360,
        max_evals=64,
        allow_error_prop=0.1,
        temp_directory="tmp",
        delete_temp_after_terminate=False,
        save=True,
        model_name="model",
        ignore_warning=True,
        encoder="auto",
        imputer="auto",
        balancing="auto",
        scaling="auto",
        feature_selection="auto",
        models="auto",
        validation=True,
        valid_size=0.15,
        objective="accuracy",
        search_algo="HyperOpt",
        search_algo_setttings={},
        search_scheduler="FIFOScheduler",
        search_scheduler_settings={},
        progress_reporter="CLIReporter",
        full_status=False,
        verbose=1,
        cpu_threads=None,
        use_gpu=False,
        seed=1,
    ):
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
        self.search_algo_setttings = search_algo_setttings
        self.search_scheduler = search_scheduler
        self.search_scheduler_settings = search_scheduler_settings
        self.progress_reporter = progress_reporter
        self.full_status = full_status
        self.verbose = verbose
        self.cpu_threads = cpu_threads
        self.use_gpu = use_gpu
        self.seed = seed

        super().__init__(
            mode="classification",
            timeout=self.timeout,
            max_evals=self.max_evals,
            allow_error_prop=self.allow_error_prop,
            temp_directory=self.temp_directory,
            delete_temp_after_terminate=self.delete_temp_after_terminate,
            save=self.save,
            model_name=self.model_name,
            ignore_warning=self.ignore_warning,
            encoder=self.encoder,
            imputer=self.imputer,
            balancing=self.balancing,
            scaling=self.scaling,
            feature_selection=self.feature_selection,
            models=self.models,
            validation=self.validation,
            valid_size=self.valid_size,
            objective=self.objective,
            search_algo=self.search_algo,
            search_algo_setttings=self.search_algo_setttings,
            search_scheduler=self.search_scheduler,
            search_scheduler_settings=self.search_scheduler_settings,
            progress_reporter=self.progress_reporter,
            full_status=self.full_status,
            verbose=self.verbose,
            cpu_threads=self.cpu_threads,
            use_gpu=self.use_gpu,
            seed=self.seed,
        )

    def fit(self, X, y):

        return super().fit(X, y)

    def predict(self, X):

        return super().predict(X)


class AutoTabular(AutoTabularClassifier, AutoTabularRegressor):

    """
    AutoTabular that automatically assign to AutoTabularClassifier or AutoTabularRegressor

    Parameters
    ----------
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

    search_algo_setttings: search algorithm settings, default = {}
    need manual configuration for each search algorithm

    search_scheduler: search scheduler used, default = "FIFOScheduler"
    support ("FIFOScheduler", "ASHAScheduler", "HyperBandScheduler", "MedianStoppingRule"
            "PopulationBasedTraining", "PopulationBasedTrainingReplay", "PB2",
            "HyperBandForBOHB", callable)

    search_scheduler_settings: search scheduler settings, default = {}
    need manual configuration for each search scheduler

    progress_reporter: progress reporter, default = "CLIReporter"
    support ("CLIReporter", "JupyterNotebookReporter")

    full_status: whether to print full status, default = False

    verbose: display for output, default = 1
    support (0, 1, 2, 3)

    cpu_threads: number of cpu threads to use, default = None
    if None, get all available cpu threads

    use_gpu: whether to use gpu, default = False

    seed: random seed, default = 1
    """

    def __init__(
        self,
        timeout=360,
        max_evals=64,
        allow_error_prop=0.1,
        temp_directory="tmp",
        delete_temp_after_terminate=False,
        save=True,
        model_name="model",
        ignore_warning=True,
        encoder="auto",
        imputer="auto",
        balancing="auto",
        scaling="auto",
        feature_selection="auto",
        models="auto",
        validation=True,
        valid_size=0.15,
        objective=None,
        search_algo="HyperOpt",
        search_algo_setttings={},
        search_scheduler="FIFOScheduler",
        search_scheduler_settings={},
        progress_reporter="CLIReporter",
        full_status=False,
        verbose=1,
        cpu_threads=None,
        use_gpu=False,
        seed=1,
    ):
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
        self.search_algo_setttings = search_algo_setttings
        self.search_scheduler = search_scheduler
        self.search_scheduler_settings = search_scheduler_settings
        self.progress_reporter = progress_reporter
        self.full_status = full_status
        self.verbose = verbose
        self.cpu_threads = cpu_threads
        self.use_gpu = use_gpu
        self.seed = seed

    def fit(self, X, y=None):

        if isinstance(y, pd.DataFrame) or isinstance(y, np.ndarray):
            self._type = type_of_task(y)
        elif y == None:
            self._type = "Unsupervised"

        if self._type in ["binary", "multiclass"]:  # assign classification tasks
            self.model = AutoTabularClassifier(
                timeout=self.timeout,
                max_evals=self.max_evals,
                allow_error_prop=self.allow_error_prop,
                temp_directory=self.temp_directory,
                delete_temp_after_terminate=self.delete_temp_after_terminate,
                save=self.save,
                model_name=self.model_name,
                ignore_warning=self.ignore_warning,
                encoder=self.encoder,
                imputer=self.imputer,
                balancing=self.balancing,
                scaling=self.scaling,
                feature_selection=self.feature_selection,
                models=self.models,
                validation=self.validation,
                valid_size=self.valid_size,
                objective="accuracy" if not self.objective else self.objective,
                search_algo=self.search_algo,
                search_algo_setttings=self.search_algo_setttings,
                search_scheduler=self.search_scheduler,
                search_scheduler_settings=self.search_scheduler_settings,
                progress_reporter=self.progress_reporter,
                full_status=self.full_status,
                verbose=self.verbose,
                cpu_threads=self.cpu_threads,
                use_gpu=self.use_gpu,
                seed=self.seed,
            )
        elif self._type in ["integer", "continuous"]:  # assign regression tasks
            self.model = AutoTabularRegressor(
                timeout=self.timeout,
                max_evals=self.max_evals,
                allow_error_prop=self.allow_error_prop,
                temp_directory=self.temp_directory,
                delete_temp_after_terminate=self.delete_temp_after_terminate,
                save=self.save,
                model_name=self.model_name,
                ignore_warning=self.ignore_warning,
                encoder=self.encoder,
                imputer=self.imputer,
                balancing=self.balancing,
                scaling=self.scaling,
                feature_selection=self.feature_selection,
                models=self.models,
                validation=self.validation,
                valid_size=self.valid_size,
                objective="MSE" if not self.objective else self.objective,
                search_algo=self.search_algo,
                search_algo_setttings=self.search_algo_setttings,
                search_scheduler=self.search_scheduler,
                search_scheduler_settings=self.search_scheduler_settings,
                progress_reporter=self.progress_reporter,
                full_status=self.full_status,
                verbose=self.verbose,
                cpu_threads=self.cpu_threads,
                use_gpu=self.use_gpu,
                seed=self.seed,
            )
        else:
            raise ValueError(
                'Not recognizing type, only ["binary", "multiclass", "integer", "continuous"] accepted, get {}!'.format(
                    self._type
                )
            )

        self.model.fit(X, y)
        return self

    def predict(self, X):

        if self.model:
            return self.model.predict(X)
        else:
            raise ValueError("No tasks found! Need to fit first.")


# import numpy as np
# import pandas as pd

# from My_AutoML._utils import type_of_task
# from ._base import AutoTabularClassifier, AutoTabularRegressor


# class AutoTabular(AutoTabularClassifier, AutoTabularRegressor):

#     """
#     Automatically assign to AutoTabularClassifier or AutoTabularRegressor
#     """

#     def __init__(
#         self,
#         timeout=360,
#         max_evals=64,
#         temp_directory="tmp",
#         delete_temp_after_terminate=False,
#         save=True,
#         model_name="model",
#         ignore_warning=True,
#         encoder="auto",
#         imputer="auto",
#         balancing="auto",
#         scaling="auto",
#         feature_selection="auto",
#         models="auto",
#         validation=True,
#         valid_size=0.15,
#         objective=None,
#         method="Bayesian",
#         algo="tpe",
#         spark_trials=False,
#         progressbar=True,
#         seed=1,
#     ):
#         self.timeout = timeout
#         self.max_evals = max_evals
#         self.temp_directory = temp_directory
#         self.delete_temp_after_terminate = delete_temp_after_terminate
#         self.save = save
#         self.model_name = model_name
#         self.ignore_warning = ignore_warning
#         self.encoder = encoder
#         self.imputer = imputer
#         self.balancing = balancing
#         self.scaling = scaling
#         self.feature_selection = feature_selection
#         self.models = models
#         self.validation = validation
#         self.valid_size = valid_size
#         self.objective = objective
#         self.method = method
#         self.algo = algo
#         self.spark_trials = spark_trials
#         self.progressbar = progressbar
#         self.seed = seed

#     def fit(self, X, y=None):

#         if isinstance(y, pd.DataFrame) or isinstance(y, np.ndarray):
#             self._type = type_of_task(y)
#         elif y == None:
#             self._type = "Unsupervised"

#         if self._type in ["binary", "multiclass"]:  # assign classification tasks
#             self.model = AutoTabularClassifier(
#                 timeout=self.timeout,
#                 max_evals=self.max_evals,
#                 temp_directory=self.temp_directory,
#                 delete_temp_after_terminate=self.delete_temp_after_terminate,
#                 save=self.save,
#                 model_name=self.model_name,
#                 ignore_warning=self.ignore_warning,
#                 encoder=self.encoder,
#                 imputer=self.imputer,
#                 balancing=self.balancing,
#                 scaling=self.scaling,
#                 feature_selection=self.feature_selection,
#                 models=self.models,
#                 validation=self.validation,
#                 valid_size=self.valid_size,
#                 objective="accuracy" if not self.objective else self.objective,
#                 method=self.method,
#                 algo=self.algo,
#                 spark_trials=self.spark_trials,
#                 progressbar=self.progressbar,
#                 seed=self.seed,
#             )
#         elif self._type in ["integer", "continuous"]:  # assign regression tasks
#             self.model = AutoTabularRegressor(
#                 timeout=self.timeout,
#                 max_evals=self.max_evals,
#                 temp_directory=self.temp_directory,
#                 delete_temp_after_terminate=self.delete_temp_after_terminate,
#                 save=self.save,
#                 model_name=self.model_name,
#                 ignore_warning=self.ignore_warning,
#                 encoder=self.encoder,
#                 imputer=self.imputer,
#                 balancing=self.balancing,
#                 scaling=self.scaling,
#                 feature_selection=self.feature_selection,
#                 models=self.models,
#                 validation=self.validation,
#                 valid_size=self.valid_size,
#                 objective="MSE" if not self.objective else self.objective,
#                 method=self.method,
#                 algo=self.algo,
#                 spark_trials=self.spark_trials,
#                 progressbar=self.progressbar,
#                 seed=self.seed,
#             )
#         else:
#             raise ValueError(
#                 'Not recognizing type, only ["binary", "multiclass", "integer", "continuous"] accepted, get {}!'.format(
#                     self._type
#                 )
#             )

#         self.model.fit(X, y)
#         return self

#     def predict(self, X):

#         if self.model:
#             return self.model.predict(X)
#         else:
#             raise ValueError("No tasks found! Need to fit first.")
