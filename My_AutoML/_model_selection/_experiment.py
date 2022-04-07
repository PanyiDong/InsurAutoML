"""
File: _experiment.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Relative Path: /My_AutoML/_model_selection/_experiment.py
File Created: Wednesday, 6th April 2022 3:46:21 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Thursday, 7th April 2022 3:47:43 pm
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

from ray import tune

import os
import shutil
import importlib
import warnings
import ast
import numpy as np
import pandas as pd
import scipy
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from My_AutoML._encoding import encoders
from My_AutoML._imputation import imputers
from My_AutoML._balancing import balancings
from My_AutoML._scaling import scalings
from My_AutoML._feature_selection import feature_selections
from My_AutoML._model import (
    classifiers,
    regressors,
)
from My_AutoML._hyperparameters import (
    encoder_hyperparameter,
    imputer_hyperparameter,
    scaling_hyperparameter,
    balancing_hyperparameter,
    feature_selection_hyperparameter,
    classifier_hyperparameter,
    regressor_hyperparameter,
)

from My_AutoML._base import no_processing
from My_AutoML._utils._base import type_of_task
from My_AutoML._utils._file import save_model

# filter certain warnings
warnings.filterwarnings("ignore", message="The dataset is balanced, no change.")
warnings.filterwarnings("ignore", message="Variables are collinear")
warnings.filterwarnings("ignore", category=UserWarning)


class AutoTabularBase:
    def __init__(
        self,
        mode="classification",
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
        search_algo="BayesOptSearch",
        search_scheduler="FIFOScheduler",
        progress_reporter="CLIReporter",
        verbose=3,
        seed=1,
    ):
        self.mode = mode
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
        self.search_scheduler = search_scheduler
        self.progress_reporter = progress_reporter
        self.verbose = verbose
        self.seed = seed

        self._iter = 0  # record iteration number

    # create hyperparameter space using ray.tune.choice
    # the pipeline of AutoClassifier is [encoder, imputer, scaling, balancing, feature_selection, model]
    # only chosen ones will be added to hyperparameter space
    def _get_hyperparameter_space(
        self,
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
    ):

        # encoding space
        _encoding_hyperparameter = []
        for _encoder in [*encoder]:
            for (
                item
            ) in encoders_hyperparameters:  # search the encoders' hyperparameters
                if item["encoder"] == _encoder:
                    _encoding_hyperparameter.append(item)

        _encoding_hyperparameter = tune.choice(_encoding_hyperparameter)

        # imputation space
        _imputer_hyperparameter = []
        if not X.isnull().values.any():  # if no missing, no need for imputation
            _imputer_hyperparameter = tune.choice([{"imputer": "no_processing"}])
        else:
            for _imputer in [*imputer]:
                for (
                    item
                ) in imputers_hyperparameters:  # search the imputer' hyperparameters
                    if item["imputer"] == _imputer:
                        _imputer_hyperparameter.append(item)

            _imputer_hyperparameter = tune.choice(_imputer_hyperparameter)

        # balancing space
        _balancing_hyperparameter = []
        for _balancing in [*balancing]:
            for (
                item
            ) in balancings_hyperparameters:  # search the balancings' hyperparameters
                if item["balancing"] == _balancing:
                    _balancing_hyperparameter.append(item)

        _balancing_hyperparameter = tune.choice(_balancing_hyperparameter)

        # scaling space
        _scaling_hyperparameter = []
        for _scaling in [*scaling]:
            for (
                item
            ) in scalings_hyperparameters:  # search the scalings' hyperparameters
                if item["scaling"] == _scaling:
                    _scaling_hyperparameter.append(item)

        _scaling_hyperparameter = tune.choice(_scaling_hyperparameter)

        # feature selection space
        _feature_selection_hyperparameter = []
        for _feature_selection in [*feature_selection]:
            for (
                item
            ) in (
                feature_selection_hyperparameters
            ):  # search the feature selections' hyperparameters
                if item["feature_selection"] == _feature_selection:
                    _feature_selection_hyperparameter.append(item)

        _feature_selection_hyperparameter = tune.choice(
            _feature_selection_hyperparameter
        )

        # model selection and hyperparameter optimization space
        _model_hyperparameter = []
        for _model in [*models]:
            # checked before at models that all models are in default space
            for item in models_hyperparameters:  # search the models' hyperparameters
                if item["model"] == _model:
                    _model_hyperparameter.append(item)

        _model_hyperparameter = tune.choice(_model_hyperparameter)

        # the pipeline search space
        # select one of the method/hyperparameter setting from each part
        return {
            "encoder": _encoding_hyperparameter,
            "imputer": _imputer_hyperparameter,
            "balancing": _balancing_hyperparameter,
            "scaling": _scaling_hyperparameter,
            "feature_selection": _feature_selection_hyperparameter,
            "model": _model_hyperparameter,
        }

    def get_hyperparameter_space(self, X, y):

        # initialize default search options
        # and select the search options based on the input restrictions
        # use copy to allows multiple manipulation

        # Encoding: convert string types to numerical type
        # all encoders available
        self._all_encoders = encoders.copy()

        # get default encoder methods space
        if self.encoder == "auto":
            encoder = self._all_encoders.copy()
        else:
            encoder = {}  # if specified, check if encoders in default encoders
            for _encoder in self.encoder:
                if _encoder not in [*self._all_encoders]:
                    raise ValueError(
                        "Only supported encoders are {}, get {}.".format(
                            [*self._all_encoders], _encoder
                        )
                    )
                encoder[_encoder] = self._all_encoders[_encoder]

        # all hyperparameters for encoders
        self._all_encoders_hyperparameters = encoder_hyperparameter.copy()

        # initialize encoders hyperparameter space
        _all_encoders_hyperparameters = self._all_encoders_hyperparameters.copy()

        # Imputer: fill missing values
        # all imputers available
        self._all_imputers = imputers.copy()

        # get default imputer methods space
        if self.imputer == "auto":
            if not X.isnull().values.any():  # if no missing values
                imputer = {"no_processing": no_processing}
                self._all_imputers = imputer  # limit default imputer space
            else:
                imputer = self._all_imputers.copy()
        else:
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

        # all hyperparemeters for imputers
        self._all_imputers_hyperparameters = imputer_hyperparameter.copy()

        # initialize imputers hyperparameter space
        _all_imputers_hyperparameters = self._all_imputers_hyperparameters.copy()

        # Balancing: deal with imbalanced dataset, using over-/under-sampling methods
        # all balancings available
        self._all_balancings = balancings.copy()

        # get default balancing methods space
        if self.balancing == "auto":
            balancing = self._all_balancings.copy()
        else:
            balancing = {}  # if specified, check if balancings in default balancings
            for _balancing in self.balancing:
                if _balancing not in [*self._all_balancings]:
                    raise ValueError(
                        "Only supported balancings are {}, get {}.".format(
                            [*self._all_balancings], _balancing
                        )
                    )
                balancing[_balancing] = self._all_balancings[_balancing]

        # all hyperparameters for balancing methods
        self._all_balancings_hyperparameters = balancing_hyperparameter.copy()

        # initialize balancing hyperparameter space
        _all_balancings_hyperparameters = self._all_balancings_hyperparameters.copy()

        # Scaling
        # all scalings available
        self._all_scalings = scalings.copy()

        # get default scaling methods space
        if self.scaling == "auto":
            scaling = self._all_scalings.copy()
        else:
            scaling = {}  # if specified, check if scalings in default scalings
            for _scaling in self.scaling:
                if _scaling not in [*self._all_scalings]:
                    raise ValueError(
                        "Only supported scalings are {}, get {}.".format(
                            [*self._all_scalings], _scaling
                        )
                    )
                scaling[_scaling] = self._all_scalings[_scaling]

        # all hyperparameters for scalings
        self._all_scalings_hyperparameters = scaling_hyperparameter.copy()

        # initialize scaling hyperparameter space
        _all_scalings_hyperparameters = self._all_scalings_hyperparameters.copy()

        # Feature selection: Remove redundant features, reduce dimensionality
        # all feature selections available
        self._all_feature_selection = feature_selections.copy()
        # special treatment, remove some feature selection for regression
        del self._all_feature_selection["extra_trees_preproc_for_regression"]
        del self._all_feature_selection["select_percentile_regression"]
        del self._all_feature_selection["select_rates_regression"]
        if X.shape[0] * X.shape[1] > 10000:
            del self._all_feature_selection["liblinear_svc_preprocessor"]

        # get default feature selection methods space
        if self.feature_selection == "auto":
            feature_selection = self._all_feature_selection.copy()
        else:
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

        # all hyperparameters for feature selections
        self._all_feature_selection_hyperparameters = (
            feature_selection_hyperparameter.copy()
        )

        # initialize feature selection hyperparameter space
        _all_feature_selection_hyperparameters = (
            self._all_feature_selection_hyperparameters.copy()
        )

        # Model selection/Hyperparameter optimization
        # using Bayesian Optimization
        # all models available
        # if mode is classification, use classification models
        # if mode is regression, use regression models
        if self.mode == "classification":
            self._all_models = classifiers.copy()
        elif self.mode == "regression":
            self._all_models = regressors.copy()

        # special treatment, remove SVM methods when observations are large
        # SVM suffers from the complexity o(n_samples^2 * n_features),
        # which is time-consuming for large datasets
        if X.shape[0] * X.shape[1] > 10000:
            # in case the methods are not included, will check before delete
            if "LibLinear_SVC" in self._all_models.keys():
                del self._all_models["LibLinear_SVC"]
            if "LibSVM_SVC" in self._all_models.keys():
                del self._all_models["LibSVM_SVC"]

        # model space, only select chosen models to space
        if self.models == "auto":  # if auto, model pool will be all default models
            models = self._all_models.copy()
        else:
            models = {}  # if specified, check if models in default models
            for _model in self.models:
                if _model not in [*self._all_models]:
                    raise ValueError(
                        "Only supported models are {}, get {}.".format(
                            [*self._all_models], _model
                        )
                    )
                models[_model] = self._all_models[_model]

        # all hyperparameters for the models by mode
        if self.mode == "classification":
            self._all_models_hyperparameters = classifier_hyperparameter.copy()
        elif self.mode == "regression":
            self._all_models_hyperparameters = regressor_hyperparameter.copy()

        # initialize model hyperparameter space
        _all_models_hyperparameters = self._all_models_hyperparameters.copy()

        # initialize default search space
        self.hyperparameter_space = None

        # generate the hyperparameter space
        if self.hyperparameter_space is None:
            self.hyperparameter_space = self._get_hyperparameter_space(
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
            )  # _X to choose whether include imputer
            # others are the combinations of default hyperparameter space & methods selected

        return encoder, imputer, balancing, scaling, feature_selection, models

    # get the hyperparameter optimization algorithm
    # set random seed for search algorithms
    def get_algo(self):

        if self.search_algo == "RandomSearch" or self.search_algo == "GridSearch":

            # Random Search and Grid Search
            from ray.tune.suggest.basic_variant import BasicVariantGenerator

            algo = BasicVariantGenerator(
                random_state=np.random.RandomState(seed=self.seed)
            )
        elif self.search_algo == "BayesOptSearch":

            # check whether bayes_opt is installed
            bayes_opt_spec = importlib.util.find_spec("bayes_opt")
            if bayes_opt_spec is None:
                raise ImportError(
                    "BayesOpt is not installed. Please install it first to use BayesOptSearch. \
                    Command to install: pip install bayesian-optimization"
                )

            # Bayesian Search
            from ray.tune.suggest.bayesopt import BayesOptSearch

            algo = BayesOptSearch(random_state=self.seed)
        elif self.search_algo == "AxSearch":

            # check whether Ax and sqlalchemy are installed
            Ax_spec = importlib.util.find_spec("ax")
            sqlalchemy_spec = importlib.util.find_spec("sqlalchemy")
            if Ax_spec is None or sqlalchemy_spec is None:
                raise ImportError(
                    "Ax or sqlalchemy not installed. Please install these packages to use AxSearch. \
                    Command to install: pip install ax-platform sqlalchemy"
                )

            # Ax Search
            from ax.service.ax_client import AxClient
            from ray.tune.suggest.ax import AxSearch

            client = AxClient(random_seed=self.seed)
            algo = AxSearch(client)
        elif self.search_algo == "BOHB":

            # check whether HpBandSter and ConfigSpace are installed
            hpbandster_spec = importlib.util.find_spec("hpbandster")
            ConfigSpace_spec = importlib.util.find_spec("ConfigSpace")
            if hpbandster_spec is None or ConfigSpace_spec is None:
                raise ImportError(
                    "HpBandSter or ConfigSpace not installed. Please install these packages to use BOHB. \
                    Command to install: pip install hpbandster ConfigSpace"
                )

            # Bayesian Optimization HyperBand/BOHB
            from ray.tune.suggest.bohb import TuneBOHB

            algo = TuneBOHB(seed=self.seed)
        elif self.search_algo == "BlendSearch":

            # check whether flaml is installed
            flaml_spec = importlib.util.find_spec("flaml")
            if flaml_spec is None:
                raise ImportError(
                    "flaml not installed. Please install it first to use BlendSearch. \
                    Command to install: pip install 'flaml[blendsearch]'"
                )

            # Blend Search
            from ray.tune.suggest.flaml import BlendSearch

            algo = BlendSearch()
        elif self.search_algo == "CFO":

            # check whether flaml is installed
            flaml_spec = importlib.util.find_spec("flaml")
            if flaml_spec is None:
                raise ImportError(
                    "flaml not installed. Please install it first to use BlendSearch. \
                    Command to install: pip install 'flaml[blendsearch]'"
                )

            # Blend Search
            from ray.tune.suggest.flaml import CFO

            algo = CFO()
        elif self.search_algo == "DragonflySearch":

            # check whether dragonfly-opt is installed
            dragonfly_spec = importlib.util.find_spec("dragonfly")
            if dragonfly_spec is None:
                raise ImportError(
                    "dragonfly-opt not installed. Please install it first to use DragonflySearch. \
                    Command to install: pip install dragonfly-opt"
                )

            # Dragonfly Search
            from ray.tune.suggest.dragonfly import DragonflySearch

            algo = DragonflySearch(random_state_seed=self.seed)
        elif self.search_algo == "HEBO":

            # Heteroscedastic Evolutionary Bayesian Optimization/HEBO
            from ray.tune.suggest.hebo import HEBOSearch

            algo = HEBOSearch(random_state_seed=self.seed)
        elif self.search_algo == "HyperOpt":

            # check whether hyperopt is installed
            hyperopt_spec = importlib.util.find_spec("hyperopt")
            if hyperopt_spec is None:
                raise ImportError(
                    "hyperopt not installed. Please install it first to use HyperOpt. \
                    Command to install: pip install -U hyperopt"
                )

            # HyperOpt Search
            from ray.tune.suggest.hyperopt import HyperOptSearch

            algo = HyperOptSearch(random_state_seed=self.seed)
        elif self.search_algo == "Nevergrad":

            # check whether nevergrad is installed
            nevergrad_spec = importlib.util.find_spec("nevergrad")
            if nevergrad_spec is None:
                raise ImportError(
                    "nevergrad not installed. Please install it first to use Nevergrad. \
                    Command to install: pip install nevergrad"
                )

            # Nevergrad Search
            from ray.tune.suggest.nevergrad import NevergradSearch

            algo = NevergradSearch()
        elif self.search_algo == "Optuna":

            # check whether optuna is installed
            optuna_spec = importlib.util.find_spec("optuna")
            if optuna_spec is None:
                raise ImportError(
                    "optuna not installed. Please install it first to use Optuna. \
                    Command to install: pip install optuna"
                )

            # Optuna Search
            from ray.tune.suggest.optuna import OptunaSearch

            algo = OptunaSearch()
        elif self.search_algo == "SigOpt":

            # check whether sigopt is installed
            sigopt_spec = importlib.util.find_spec("sigopt")
            if sigopt_spec is None:
                raise ImportError(
                    "sigopt not installed. Please install it first to use SigOpt. \
                    Command to install: pip install sigopt \
                    Set SigOpt API: export SIGOPT_KEY= ..."
                )

            # SigOpt Search
            from ray.tune.suggest.sigopt import SigOptSearch

            algo = SigOptSearch()
        elif self.search_algo == "Scikit-Optimize":

            # check whether scikit-optimize is installed
            skopt_spec = importlib.util.find_spec("skopt")
            if skopt_spec is None:
                raise ImportError(
                    "scikit-optimize not installed. Please install it first to use Scikit-Optimize. \
                    Command to install: pip install scikit-optimize"
                )

            # Scikit-Optimize Search
            from ray.tune.suggest.skopt import SkOptSearch

            algo = SkOptSearch()
        elif self.search_algo == "ZOOpt":

            # check whether zoopt is installed
            zoopt_spec = importlib.util.find_spec("zoopt")
            if zoopt_spec is None:
                raise ImportError(
                    "zoopt not installed. Please install it first to use ZOOpt. \
                    Command to install: pip install zoopt"
                )

            # ZOOpt Search
            from ray.tune.suggest.zoopt import ZOOptSearch

            algo = ZOOptSearch()
        elif self.search_algo == "Repeater":

            # Repeated Evaluations
            from ray.tune.suggest import Repeater

            algo = Repeater()
        elif self.search_algo == "ConcurrencyLimiter":

            # ConcurrencyLimiter
            from ray.tune.suggest import ConcurrencyLimiter

            algo = ConcurrencyLimiter()
        else:

            # if none above, assume is a callable custom algorithm
            if isinstance(self.search_algo, callable):
                algo = self.search_algo()
            # if not callable, raise error
            else:
                raise TypeError(
                    "Algorithm {} is not supported. Please use one of the supported algorithms.".format(
                        self.search_algo
                    )
                )

        return algo

    # get search scheduler
    def get_scheduler(self):

        if self.search_scheduler == "FIFOScheduler":

            from ray.tune.schedulers import FIFOScheduler

            scheduler = FIFOScheduler()

        return scheduler

    # get progress reporter
    def get_progress_reporter(self):

        if self.progress_reporter == "CLIReporter":

            from ray.tune.progress_reporter import CLIReporter

            progress_reporter = CLIReporter(
                max_progress_rows=self.max_evals, max_error_rows=self.max_error
            )
        elif self.progress_reporter == "JupyterNotebookReporter":

            from ray.tune.progress_reporter import JupyterNotebookReporter

            progress_reporter = JupyterNotebookReporter(
                max_progress_rows=self.max_evals, max_error_rows=self.max_error
            )

        return progress_reporter

    # load hyperparameter settings and train on the data
    def load_model(self, _X, _y):

        # load hyperparameter settings
        with open(self.model_name) as f:
            optimal_setting = f.readlines()

        # remove change line signs
        optimal_setting = [item.replace("\n", "") for item in optimal_setting]
        # remove blank spaces
        while "" in optimal_setting:
            optimal_setting.remove("")

        # convert strings to readable dictionaries
        self.optimal_encoder = optimal_setting[0]
        self.optimal_encoder_hyperparameters = ast.literal_eval(optimal_setting[1])
        self.optimal_imputer = optimal_setting[2]
        self.optimal_imputer_hyperparameters = ast.literal_eval(optimal_setting[3])
        self.optimal_balancing = optimal_setting[4]
        self.optimal_balancing_hyperparameters = ast.literal_eval(optimal_setting[5])
        self.optimal_scaling = optimal_setting[6]
        self.optimal_scaling_hyperparameters = ast.literal_eval(optimal_setting[7])
        self.optimal_feature_selection = optimal_setting[8]
        self.optimal_feature_selection_hyperparameters = ast.literal_eval(
            optimal_setting[9]
        )
        self.optimal_model = optimal_setting[10]
        self.optimal_model_hyperparameters = ast.literal_eval(optimal_setting[11])

        # map the methods and hyperparameters
        # fit the methods
        # encoding
        self._fit_encoder = self._all_encoders[self.optimal_encoder](
            **self.optimal_encoder_hyperparameters
        )
        _X = self._fit_encoder.fit(_X)
        # imputer
        self._fit_imputer = self._all_imputers[self.optimal_imputer](
            **self.optimal_imputer_hyperparameters
        )
        _X = self._fit_imputer.fill(_X)
        # balancing
        self._fit_balancing = self._all_balancings[self.optimal_balancing](
            **self.optimal_balancing_hyperparameters
        )
        _X, _y = self._fit_balancing.fit_transform(_X, _y)

        # make sure the classes are integers (belongs to certain classes)
        if self.mode == "classification":
            _y = _y.astype(int)
        # scaling
        self._fit_scaling = self._all_scalings[self.optimal_scaling](
            **self.optimal_scaling_hyperparameters
        )
        self._fit_scaling.fit(_X, _y)
        _X = self._fit_scaling.transform(_X)
        # feature selection
        self._fit_feature_selection = self._all_feature_selection[
            self.optimal_feature_selection
        ](**self.optimal_feature_selection_hyperparameters)
        self._fit_feature_selection.fit(_X, _y)
        _X = self._fit_feature_selection.transform(_X)
        # model
        self._fit_model = self._all_models[self.optimal_model](
            **self.optimal_model_hyperparameters
        )
        self._fit_model.fit(_X, _y.values.ravel())

        return self

    # select optimal settings and fit on optimal hyperparameters
    def _fit_optimal(self, optimal_point, _X, _y):

        # optimal encoder
        self.optimal_encoder_hyperparameters = optimal_point["encoder"]
        self.optimal_encoder = self.optimal_encoder_hyperparameters["encoder"]
        del self.optimal_encoder_hyperparameters["encoder"]
        # optimal imputer
        self.optimal_imputer_hyperparameters = optimal_point["imputer"]
        self.optimal_imputer = self.optimal_imputer_hyperparameters["imputer"]
        del self.optimal_imputer_hyperparameters["imputer"]
        # optimal balancing
        self.optimal_balancing_hyperparameters = optimal_point["balancing"]
        self.optimal_balancing = self.optimal_balancing_hyperparameters["balancing"]
        del self.optimal_balancing_hyperparameters["balancing"]
        # optimal scaling
        self.optimal_scaling_hyperparameters = optimal_point["scaling"]
        self.optimal_scaling = self.optimal_scaling_hyperparameters["scaling"]
        del self.optimal_scaling_hyperparameters["scaling"]
        # optimal feature selection
        self.optimal_feature_selection_hyperparameters = optimal_point[
            "feature_selection"
        ]
        self.optimal_feature_selection = self.optimal_feature_selection_hyperparameters[
            "feature_selection"
        ]
        del self.optimal_feature_selection_hyperparameters["feature_selection"]
        # optimal classifier
        self.optimal_model_hyperparameters = optimal_point[
            "model"
        ]  # optimal model selected
        self.optimal_model = self.optimal_model_hyperparameters[
            "model"
        ]  # optimal hyperparameter settings selected
        del self.optimal_model_hyperparameters["model"]

        # record optimal settings
        with open(self.temp_directory + "/optimal_setting.txt", "w") as f:
            f.write("Optimal encoding method is: {}\n".format(self.optimal_encoder))
            f.write("Optimal encoding hyperparameters:")
            print(self.optimal_encoder_hyperparameters, file=f, end="\n\n")
            f.write("Optimal imputation method is: {}\n".format(self.optimal_imputer))
            f.write("Optimal imputation hyperparameters:")
            print(self.optimal_imputer_hyperparameters, file=f, end="\n\n")
            f.write("Optimal balancing method is: {}\n".format(self.optimal_balancing))
            f.write("Optimal balancing hyperparamters:")
            print(self.optimal_balancing_hyperparameters, file=f, end="\n\n")
            f.write("Optimal scaling method is: {}\n".format(self.optimal_scaling))
            f.write("Optimal scaling hyperparameters:")
            print(self.optimal_scaling_hyperparameters, file=f, end="\n\n")
            f.write(
                "Optimal feature selection method is: {}\n".format(
                    self.optimal_feature_selection
                )
            )
            f.write("Optimal feature selection hyperparameters:")
            print(self.optimal_feature_selection_hyperparameters, file=f, end="\n\n")
            f.write("Optimal {} model is: {}\n".format(self.mode, self.optimal_model))
            f.write("Optimal {} hyperparameters:".format(self.mode))
            print(self.optimal_model_hyperparameters, file=f, end="\n\n")

        # encoding
        self._fit_encoder = self._all_encoders[self.optimal_encoder](
            **self.optimal_encoder_hyperparameters
        )
        _X = self._fit_encoder.fit(_X)
        # imputer
        self._fit_imputer = self._all_imputers[self.optimal_imputer](
            **self.optimal_imputer_hyperparameters
        )
        _X = self._fit_imputer.fill(_X)
        # balancing
        self._fit_balancing = self._all_balancings[self.optimal_balancing](
            **self.optimal_balancing_hyperparameters
        )
        _X, _y = self._fit_balancing.fit_transform(_X, _y)

        # make sure the classes are integers (belongs to certain classes)
        _y = _y.astype(int)
        _y = _y.astype(int)
        # scaling
        self._fit_scaling = self._all_scalings[self.optimal_scaling](
            **self.optimal_scaling_hyperparameters
        )
        self._fit_scaling.fit(_X, _y)
        _X = self._fit_scaling.transform(_X)
        # feature selection
        self._fit_feature_selection = self._all_feature_selection[
            self.optimal_feature_selection
        ](**self.optimal_feature_selection_hyperparameters)
        self._fit_feature_selection.fit(_X, _y)
        _X = self._fit_feature_selection.transform(_X)
        # classification
        self._fit_model = self._all_models[self.optimal_model](
            **self.optimal_model_hyperparameters
        )
        self._fit_model.fit(_X, _y.values.ravel())

        # save the model
        if self.save:
            save_model(
                self.optimal_encoder,
                self.optimal_encoder_hyperparameters,
                self.optimal_imputer,
                self.optimal_imputer_hyperparameters,
                self.optimal_balancing,
                self.optimal_balancing_hyperparameters,
                self.optimal_scaling,
                self.optimal_scaling_hyperparameters,
                self.optimal_feature_selection,
                self.optimal_feature_selection_hyperparameters,
                self._fit_model,
                self.optimal_model_hyperparameters,
                self.model_name,
            )

        return self

    def fit(self, X, y):

        if self.ignore_warning:  # ignore all warnings to generate clearer outputs
            warnings.filterwarnings("ignore")

        _X = X.copy()
        _y = y.copy()

        (
            encoder,
            imputer,
            balancing,
            scaling,
            feature_selection,
            models,
        ) = self.get_hyperparameter_space(_X, _y)

        # if the model is already trained, read the setting
        if os.path.exists(self.model_name):

            print("Stored model found, load previous model.")
            self.load_model(_X, _y)

            return self

        # initialize temp directory
        # check if temp directory exists, if exists, empty it
        if os.path.isdir(self.temp_directory):
            shutil.rmtree(self.temp_directory)
        os.makedirs(self.temp_directory)

        # write basic information to init.txt
        with open(self.temp_directory + "/init.txt", "w") as f:
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
            f.write("Type of the task: {}.\n".format(self.mode))

        # set random seed
        np.random.seed(self.seed)

        # get maximum allowed errors
        self.max_error = int(self.max_evals * self.allow_error_prop)

        if self.validation:  # only perform train_test_split when validation
            # train test split so the performance of model selection and
            # hyperparameter optimization can be evaluated
            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(
                _X, _y, test_size=self.valid_size, random_state=self.seed
            )

        # the objective function of Bayesian Optimization tries to minimize
        # use accuracy score
        @ignore_warnings(category=ConvergenceWarning)
        def _objective(params):
            # different evaluation metrics for classification and regression
            if self.mode == "regression":
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
                else:
                    raise ValueError(
                        'Mode {} only support ["MSE", "MAE", "MSLE", "R2", "MAX"], get{}'.format(
                            self.mode, self.objective
                        )
                    )
            elif self.mode == "classification":
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
                else:
                    raise ValueError(
                        'Mode {} only support ["accuracy", "precision", "auc", "hinge", "f1"], get{}'.format(
                            self.mode, self.objective
                        )
                    )

            # pipeline of objective, [encoder, imputer, balancing, scaling, feature_selection, model]
            # select encoder and set hyperparameters
            # must have encoder
            _encoder_hyper = params["encoder"]
            _encoder = _encoder_hyper["encoder"]
            del _encoder_hyper["encoder"]
            enc = encoder[_encoder](**_encoder_hyper)

            # select imputer and set hyperparameters
            _imputer_hyper = params["imputer"]
            _imputer = _imputer_hyper["imputer"]
            del _imputer_hyper["imputer"]
            imp = imputer[_imputer](**_imputer_hyper)

            # select balancing and set hyperparameters
            # must have balancing, since no_preprocessing is included
            _balancing_hyper = params["balancing"]
            _balancing = _balancing_hyper["balancing"]
            del _balancing_hyper["balancing"]
            blc = balancing[_balancing](**_balancing_hyper)

            # select scaling and set hyperparameters
            # must have scaling, since no_preprocessing is included
            _scaling_hyper = params["scaling"]
            _scaling = _scaling_hyper["scaling"]
            del _scaling_hyper["scaling"]
            scl = scaling[_scaling](**_scaling_hyper)

            # select feature selection and set hyperparameters
            # must have feature selection, since no_preprocessing is included
            _feature_selection_hyper = params["feature_selection"]
            _feature_selection = _feature_selection_hyper["feature_selection"]
            del _feature_selection_hyper["feature_selection"]
            fts = feature_selection[_feature_selection](**_feature_selection_hyper)

            # select model model and set hyperparameters
            # must have a model
            _model_hyper = params["model"]
            _model = _model_hyper["model"]
            del _model_hyper["model"]
            reg = models[_model](
                **_model_hyper
            )  # call the model using passed parameters

            obj_tmp_directory = self.temp_directory + "/iter_" + str(self._iter + 1)
            if not os.path.isdir(obj_tmp_directory):
                os.makedirs(obj_tmp_directory)

            with open(obj_tmp_directory + "/hyperparameter_settings.txt", "w") as f:
                f.write("Encoding method: {}\n".format(_encoder))
                f.write("Encoding Hyperparameters:")
                print(_encoder_hyper, file=f, end="\n\n")
                f.write("Imputation method: {}\n".format(_imputer))
                f.write("Imputation Hyperparameters:")
                print(_imputer_hyper, file=f, end="\n\n")
                f.write("Balancing method: {}\n".format(_balancing))
                f.write("Balancing Hyperparameters:")
                print(_balancing_hyper, file=f, end="\n\n")
                f.write("Scaling method: {}\n".format(_scaling))
                f.write("Scaling Hyperparameters:")
                print(_scaling_hyper, file=f, end="\n\n")
                f.write("Feature Selection method: {}\n".format(_feature_selection))
                f.write("Feature Selection Hyperparameters:")
                print(_feature_selection_hyper, file=f, end="\n\n")
                f.write("Model: {}\n".format(_model))
                f.write("Model Hyperparameters:")
                print(_model_hyper, file=f, end="\n\n")

            if self.validation:
                _X_train_obj, _X_test_obj = X_train.copy(), X_test.copy()
                _y_train_obj, _y_test_obj = y_train.copy(), y_test.copy()

                # encoding
                _X_train_obj = enc.fit(_X_train_obj)
                _X_test_obj = enc.refit(_X_test_obj)
                with open(obj_tmp_directory + "/objective_process.txt", "w") as f:
                    f.write("Encoding finished, in imputation process.")
                # imputer
                _X_train_obj = imp.fill(_X_train_obj)
                _X_test_obj = imp.fill(_X_test_obj)
                with open(obj_tmp_directory + "/objective_process.txt", "w") as f:
                    f.write("Imputation finished, in scaling process.")
                # balancing
                _X_train_obj, _y_train_obj = blc.fit_transform(
                    _X_train_obj, _y_train_obj
                )
                with open(obj_tmp_directory + "/objective_process.txt", "w") as f:
                    f.write("Balancing finished, in scaling process.")

                # make sure the classes are integers (belongs to certain classes)
                _y_train_obj = _y_train_obj.astype(int)
                _y_test_obj = _y_test_obj.astype(int)
                # scaling
                scl.fit(_X_train_obj, _y_train_obj)
                _X_train_obj = scl.transform(_X_train_obj)
                _X_test_obj = scl.transform(_X_test_obj)
                with open(obj_tmp_directory + "/objective_process.txt", "w") as f:
                    f.write("Scaling finished, in feature selection process.")
                # feature selection
                fts.fit(_X_train_obj, _y_train_obj)
                _X_train_obj = fts.transform(_X_train_obj)
                _X_test_obj = fts.transform(_X_test_obj)
                with open(obj_tmp_directory + "/objective_process.txt", "w") as f:
                    f.write(
                        "Feature selection finished, in {} model.".format(self.mode)
                    )

                # fit model
                if scipy.sparse.issparse(
                    _X_train_obj
                ):  # check if returns sparse matrix
                    _X_train_obj = _X_train_obj.toarray()
                if scipy.sparse.issparse(_X_test_obj):
                    _X_test_obj = _X_test_obj.toarray()

                # store the preprocessed train/test datasets
                if isinstance(
                    _X_train_obj, np.ndarray
                ):  # in case numpy array is returned
                    pd.concat(
                        [pd.DataFrame(_X_train_obj), _y_train_obj],
                        axis=1,
                        ignore_index=True,
                    ).to_csv(obj_tmp_directory + "/train_preprocessed.csv", index=False)
                elif isinstance(_X_train_obj, pd.DataFrame):
                    pd.concat([_X_train_obj, _y_train_obj], axis=1).to_csv(
                        obj_tmp_directory + "/train_preprocessed.csv", index=False
                    )
                else:
                    raise TypeError("Only accept numpy array or pandas dataframe!")

                if isinstance(_X_test_obj, np.ndarray):
                    pd.concat(
                        [pd.DataFrame(_X_test_obj), _y_test_obj],
                        axis=1,
                        ignore_index=True,
                    ).to_csv(obj_tmp_directory + "/test_preprocessed.csv", index=False)
                elif isinstance(_X_test_obj, pd.DataFrame):
                    pd.concat([_X_test_obj, _y_test_obj], axis=1).to_csv(
                        obj_tmp_directory + "/test_preprocessed.csv", index=False
                    )
                else:
                    raise TypeError("Only accept numpy array or pandas dataframe!")

                reg.fit(_X_train_obj, _y_train_obj.values.ravel())
                os.remove(obj_tmp_directory + "/objective_process.txt")

                y_pred = reg.predict(_X_test_obj)
                if self.objective == "R2":  # special treatment for r2_score
                    _loss = -_obj(y_pred, _y_test_obj.values)
                else:
                    _loss = _obj(y_pred, _y_test_obj.values)

                with open(obj_tmp_directory + "/testing_objective.txt", "w") as f:
                    f.write("Loss from objective function is: {:.6f}\n".format(_loss))
                    f.write("Loss is calculate using {}.".format(self.objective))
                self._iter += 1

                # since we tries to minimize the objective function, take negative accuracy here
                tune.report(loss=_loss, status="fitted")
            else:
                _X_obj = _X.copy()
                _y_obj = _y.copy()

                # encoding
                _X_obj = enc.fit(_X_obj)
                with open(obj_tmp_directory + "/objective_process.txt", "w") as f:
                    f.write("Encoding finished, in imputation process.")
                # imputer
                _X_obj = imp.fill(_X_obj)
                with open(obj_tmp_directory + "/objective_process.txt", "w") as f:
                    f.write("Imputation finished, in scaling process.")
                # balancing
                _X_obj = blc.fit_transform(_X_obj)
                with open(obj_tmp_directory + "/objective_process.txt", "w") as f:
                    f.write("Balancing finished, in feature selection process.")
                # scaling
                scl.fit(_X_obj, _y_obj)
                _X_obj = scl.transform(_X_obj)
                with open(obj_tmp_directory + "/objective_process.txt", "w") as f:
                    f.write("Scaling finished, in balancing process.")
                # feature selection
                fts.fit(_X_obj, _y_obj)
                _X_obj = fts.transform(_X_obj)
                with open(obj_tmp_directory + "/objective_process.txt", "w") as f:
                    f.write(
                        "Feature selection finished, in {} model.".format(self.mode)
                    )

                # fit model
                reg.fit(_X_obj.values, _y_obj.values.ravel())
                pd.concat([_X_obj, _y_obj], axis=1).to_csv(
                    obj_tmp_directory + "/data_preprocessed.csv", index=False
                )
                os.remove(obj_tmp_directory + "/objective_process.txt")

                y_pred = reg.predict(_X_obj.values)

                if self.objective == "R2":  # special treatment for r2_score
                    _loss = -_obj(y_pred, _y_obj.values)
                else:
                    _loss = _obj(y_pred, _y_obj.values)

                with open(obj_tmp_directory + "/testing_objective.txt", "w") as f:
                    f.write("Loss from objective function is: {.6f}\n".format(_loss))
                    f.write("Loss is calculate using {}.".format(self.objective))
                self._iter += 1

                tune.report(loss=_loss, status="fitted")

        # use ray for Model Selection and Hyperparameter Selection
        # get search algorithm
        algo = self.get_algo()

        # get search scheduler
        scheduler = self.get_scheduler()

        # get progress reporter
        progress_reporter = self.get_progress_reporter()

        # optimization process
        fit_analysis = tune.run(
            _objective,
            config=self.hyperparameter_space,
            name=self.model_name,  # name of the tuning process, use model_name
            mode="min",  # always call a minization process
            search_alg=algo,
            scheduler=scheduler,
            metric="loss",
            num_samples=self.max_evals,
            max_failures=self.max_error,
            time_budget_s=self.timeout,
            progress_reporter=progress_reporter,
            verbose=self.verbose,
            local_dir=self.temp_directory,
            log_to_file=self.temp_directory + "/log",
        )

        # select optimal settings and fit optimal pipeline
        self._fit_optimal(fit_analysis.best_config, _X, _y)

        # whether to retain temp files
        if self.delete_temp_after_terminate:
            shutil.rmtree(self.temp_directory)

        return self

    def predict(self, X):

        _X = X.copy()

        # may need preprocessing for test data, the preprocessing should be the same as in fit part
        # Encoding
        # convert string types to numerical type
        _X = self._fit_encoder.refit(_X)

        # Imputer
        # fill missing values
        _X = self._fit_imputer.fill(_X)

        # Balancing
        # deal with imbalanced dataset, using over-/under-sampling methods
        # No need to balance on test data

        # Scaling
        _X = self._fit_scaling.transform(_X)

        # Feature selection
        # Remove redundant features, reduce dimensionality
        _X = self._fit_feature_selection.transform(_X)

        # use model to predict
        return self._fit_model.predict(_X)


class AutoTabularRegressor(AutoTabularBase):
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
        search_algo="BayesOptSearch",
        search_scheduler="FIFOScheduler",
        progress_reporter="CLIReporter",
        verbose=3,
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
        self.search_scheduler = search_scheduler
        self.progress_reporter = progress_reporter
        self.verbose = verbose
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
            search_scheduler=self.search_scheduler,
            progress_reporter=self.progress_reporter,
            verbose=self.verbose,
            seed=self.seed,
        )

    def fit(self, X, y):

        return super().fit(X, y)

    def predict(self, X):

        return super().predict(X)


class AutoTabularClassifier(AutoTabularBase):
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
        search_algo="BayesOptSearch",
        search_scheduler="FIFOScheduler",
        progress_reporter="CLIReporter",
        verbose=3,
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
        self.search_scheduler = search_scheduler
        self.progress_reporter = progress_reporter
        self.verbose = verbose
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
            search_scheduler=self.search_scheduler,
            progress_reporter=self.progress_reporter,
            verbose=self.verbose,
            seed=self.seed,
        )

    def fit(self, X, y):

        return super().fit(X, y)

    def predict(self, X):

        return super().predict(X)


class AutoTabular(AutoTabularClassifier, AutoTabularRegressor):

    """
    Automatically assign to AutoTabularClassifier or AutoTabularRegressor
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
        search_algo="BayesOptSearch",
        search_scheduler="FIFOScheduler",
        progress_reporter="CLIReporter",
        verbose=3,
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
        self.search_scheduler = search_scheduler
        self.progress_reporter = progress_reporter
        self.verbose = verbose
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
                search_scheduler=self.search_scheduler,
                progress_reporter=self.progress_reporter,
                verbose=self.verbose,
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
                search_scheduler=self.search_scheduler,
                progress_reporter=self.progress_reporter,
                verbose=self.verbose,
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
