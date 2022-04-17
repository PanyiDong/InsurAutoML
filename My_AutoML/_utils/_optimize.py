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
Last Modified: Sunday, 17th April 2022 5:03:56 pm
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

import copy
from ray import tune
import importlib
from typing import Callable

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
        elif "JSON" in logger:

            from ray.tune.logger import JsonLoggerCallback

            loggers.append(JsonLoggerCallback())
        elif "CSV" in logger:

            from ray.tune.logger import CSVLogger

            loggers.append(CSVLogger())
        elif "MLflow" in logger:

            # checkwhether mlflow is installed
            mlflow_spec = importlib.util.find_spec("mlflow")
            if mlflow_spec is None:
                raise ImportError(
                    "mlflow not installed. Please install it first to use mlflow. \
                    Command to install: pip install mlflow"
                )

            from ray.tune.integration.mlflow import MLflowLoggerCallback

            loggers.append(MLflowLoggerCallback())
        elif "Wandb" in logger:

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
