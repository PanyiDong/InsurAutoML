"""
File: _optimize.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_utils/_optimize.py
File: _optimize.py
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Tuesday, 15th November 2022 11:03:27 pm
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

import logging
logger = logging.getLogger(__name__)

from typing import Dict, Callable, Union, List, Any
import warnings
from inspect import isclass
import copy
import time
from collections import defaultdict, deque
from functools import partial
import numpy as np
import pandas as pd
import ray
from ray import tune
from ray.tune import Stopper
import importlib

# from wrapt_timeout_decorator import *

from InsurAutoML._utils._base import (
    has_method,
    distribution_to_suggest,
    DisableLogger,
)
from InsurAutoML._constant import (
    METHOD_MAPPING
)

# create hyperparameter space using ray.tune.choice
# the pipeline of AutoClassifier is [encoder, imputer, scaling, balancing, feature_selection, model]
# only chosen ones will be added to hyperparameter space
def _base_get_hyperparameter_space(
    X: pd.DataFrame,
    encoders_hyperparameters: Dict,
    encoder: Dict,
    imputers_hyperparameters: Dict,
    imputer: Dict,
    balancings_hyperparameters: Dict,
    balancing: Dict,
    scalings_hyperparameters: Dict,
    scaling: Dict,
    feature_selection_hyperparameters: Dict,
    feature_selection: Dict,
    models_hyperparameters: Dict,
    models: Dict,
    task_mode: str,
) -> Dict:
    # encoding space
    # get all method names in hyperparameter space
    all_encoders = [item["encoder"] for item in encoders_hyperparameters]
    _encoding_hyperparameter = []
    for _encoder in [*encoder]:
        _encoding_hyperparameter.append(encoders_hyperparameters[all_encoders.index(_encoder)])
        
    # raise error if no encoding hyperparameters are found
    if len(_encoding_hyperparameter) == 0:
        raise ValueError(
            "No encoding hyperparameters are found. Please check your encoders."
        )

    _encoding_hyperparameter = tune.choice(_encoding_hyperparameter)
    
    # imputation space
    # get all method names in hyperparameter space
    all_imputers = [item["imputer"] for item in imputers_hyperparameters]
    _imputer_hyperparameter = []
    for _imputer in [*imputer]:
        _imputer_hyperparameter.append(imputers_hyperparameters[all_imputers.index(_imputer)])
        
    # raise error if no encoding hyperparameters are found
    if len(_imputer_hyperparameter) == 0:
        raise ValueError(
            "No imputer hyperparameters are found. Please check your encoders."
        )

    _imputer_hyperparameter = tune.choice(_imputer_hyperparameter)
    
    # balancing space
    # get all method names in hyperparameter space
    all_balancings = [item["balancing"] for item in balancings_hyperparameters]
    _balancing_hyperparameter = []
    for _balancing in [*balancing]:
        _balancing_hyperparameter.append(balancings_hyperparameters[all_balancings.index(_balancing)])
        
    # raise error if no encoding hyperparameters are found
    if len(_balancing_hyperparameter) == 0:
        raise ValueError(
            "No balancing hyperparameters are found. Please check your encoders."
        )

    _balancing_hyperparameter = tune.choice(_balancing_hyperparameter)
    
    # scaling space
    # get all method names in hyperparameter space
    all_scalings = [item["scaling"] for item in scalings_hyperparameters]
    _scaling_hyperparameter = []
    for _scaling in [*scaling]:
        _scaling_hyperparameter.append(scalings_hyperparameters[all_scalings.index(_scaling)])
        
    # raise error if no encoding hyperparameters are found
    if len(_scaling_hyperparameter) == 0:
        raise ValueError(
            "No scaling hyperparameters are found. Please check your encoders."
        )

    _scaling_hyperparameter = tune.choice(_scaling_hyperparameter)
    
    # feature selection space
    # get all method names in hyperparameter space
    all_feature_selections = [item["feature_selection"] for item in feature_selection_hyperparameters]
    _feature_selection_hyperparameter = []
    for _feature_selection in [*feature_selection]:
        _feature_selection_hyperparameter.append(
            feature_selection_hyperparameters[all_feature_selections.index(_feature_selection)]
        )
        
    # raise error if no encoding hyperparameters are found
    if len(_feature_selection_hyperparameter) == 0:
        raise ValueError(
            "No feature selection hyperparameters are found. Please check your encoders."
        )

    _feature_selection_hyperparameter = tune.choice(_feature_selection_hyperparameter)
    
    # model space
    # get all method names in hyperparameter space
    all_models = [item["model"] for item in models_hyperparameters]
    _model_hyperparameter = []
    for _model in [*models]:
        _model_hyperparameter.append(
            models_hyperparameters[all_models.index(_model)]
        )
        
    # raise error if no encoding hyperparameters are found
    if len(_model_hyperparameter) == 0:
        raise ValueError(
            "No model hyperparameters are found. Please check your encoders."
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

def _hyerpot_get_hyperparameter_space(
    X: pd.DataFrame,
    encoders_hyperparameters: Dict,
    encoder: Dict,
    imputers_hyperparameters: Dict,
    imputer: Dict,
    balancings_hyperparameters: Dict,
    balancing: Dict,
    scalings_hyperparameters: Dict,
    scaling: Dict,
    feature_selection_hyperparameters: Dict,
    feature_selection: Dict,
    models_hyperparameters: Dict,
    models: Dict,
    task_mode: str,
) -> Dict:

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
    
def _optuna_get_hyperparameter_space(
    X: pd.DataFrame,
    encoders_hyperparameters: Dict,
    encoder: Dict,
    imputers_hyperparameters: Dict,
    imputer: Dict,
    balancings_hyperparameters: Dict,
    balancing: Dict,
    scalings_hyperparameters: Dict,
    scaling: Dict,
    feature_selection_hyperparameters: Dict,
    feature_selection: Dict,
    models_hyperparameters: Dict,
    models: Dict,
    task_mode: str,
) -> Callable:
    # check installation of optuna
    optuna_spec = importlib.util.find_spec("optuna")
    if optuna_spec is None:
        raise ImportError(
            "optuna not installed. Please install it first to use Optuna. \
            Command to install: pip install optuna"
        )
        
    import optuna
    from ray.tune.suggest.optuna import OptunaSearch
        
    # config space
    # encoding space
    # get all method names in hyperparameter space
    all_encoders = [item["encoder"] for item in encoders_hyperparameters]
    _encoding_hyperparameter = []
    for _encoder in [*encoder]:
        _encoding_hyperparameter.append(
            OptunaSearch.convert_search_space(encoders_hyperparameters[all_encoders.index(_encoder)])
        )
        
    # raise error if no encoding hyperparameters are found
    if len(_encoding_hyperparameter) == 0:
        raise ValueError(
            "No encoding hyperparameters are found. Please check your encoders."
        )
    
    # imputation space
    # get all method names in hyperparameter space
    all_imputers = [item["imputer"] for item in imputers_hyperparameters]
    _imputer_hyperparameter = []
    for _imputer in [*imputer]:
        _imputer_hyperparameter.append(
            OptunaSearch.convert_search_space(imputers_hyperparameters[all_imputers.index(_imputer)])
        )
        
    # raise error if no encoding hyperparameters are found
    if len(_imputer_hyperparameter) == 0:
        raise ValueError(
            "No imputer hyperparameters are found. Please check your encoders."
        )
    
    # balancing space
    # get all method names in hyperparameter space
    all_balancings = [item["balancing"] for item in balancings_hyperparameters]
    _balancing_hyperparameter = []
    for _balancing in [*balancing]:
        _balancing_hyperparameter.append(
            OptunaSearch.convert_search_space(balancings_hyperparameters[all_balancings.index(_balancing)])
        )
        
    # raise error if no encoding hyperparameters are found
    if len(_balancing_hyperparameter) == 0:
        raise ValueError(
            "No balancing hyperparameters are found. Please check your encoders."
        )
    
    # scaling space
    # get all method names in hyperparameter space
    all_scalings = [item["scaling"] for item in scalings_hyperparameters]
    _scaling_hyperparameter = []
    for _scaling in [*scaling]:
        _scaling_hyperparameter.append(
            OptunaSearch.convert_search_space(scalings_hyperparameters[all_scalings.index(_scaling)])
        )
        
    # raise error if no encoding hyperparameters are found
    if len(_scaling_hyperparameter) == 0:
        raise ValueError(
            "No scaling hyperparameters are found. Please check your encoders."
        )
    
    # feature selection space
    # get all method names in hyperparameter space
    all_feature_selections = [item["feature_selection"] for item in feature_selection_hyperparameters]
    _feature_selection_hyperparameter = []
    for _feature_selection in [*feature_selection]:
        _feature_selection_hyperparameter.append(
            OptunaSearch.convert_search_space(
                feature_selection_hyperparameters[all_feature_selections.index(_feature_selection)]
            )
        )
        
    # raise error if no encoding hyperparameters are found
    if len(_feature_selection_hyperparameter) == 0:
        raise ValueError(
            "No feature selection hyperparameters are found. Please check your encoders."
        )
    
    # model space
    # get all method names in hyperparameter space
    all_models = [item["model"] for item in models_hyperparameters]
    _model_hyperparameter = []
    for _model in [*models]:
        _model_hyperparameter.append(
            OptunaSearch.convert_search_space(models_hyperparameters[all_models.index(_model)])
        )
        
    # raise error if no encoding hyperparameters are found
    if len(_model_hyperparameter) == 0:
        raise ValueError(
            "No model hyperparameters are found. Please check your encoders."
        )
        
    def search_space(
        trial: optuna.Trial,
        _encoder,
        _imputer,
        _balancing,
        _scaling,
        _feature_selection,
        _models,
        _encoding_hyperparameter,
        _imputer_hyperparameter,
        _balancing_hyperparameter,
        _scaling_hyperparameter,
        _feature_selection_hyperparameter,
        _model_hyperparameter,
        task_mode,
    ) :
        encoder = trial.suggest_categorical("encoder", _encoder)
        imputer = trial.suggest_categorical("imputer", _imputer)
        balancing = trial.suggest_categorical("balancing", _balancing)
        scaling = trial.suggest_categorical("scaling", _scaling)
        feature_selection = trial.suggest_categorical("feature_selection", _feature_selection)
        model = trial.suggest_categorical("model", _models)
        
        # the pipeline search space
        # select one of the method/hyperparameter setting from each part
        encoder_hyper = {}
        encoder_hyper["encoder"] = encoder
        for key, value in _encoding_hyperparameter[[*_encoder].index(encoder)].items():
            encoder_hyper[key] = distribution_to_suggest(trial, encoder, key, value, algo = "Optuna")
            
        imputer_hyper = {}
        imputer_hyper["imputer"] = imputer
        for key, value in _imputer_hyperparameter[[*_imputer].index(imputer)].items():
            imputer_hyper[key] = distribution_to_suggest(trial, imputer, key, value, algo = "Optuna")
            
        balancing_hyper = {}
        balancing_hyper["balancing"] = balancing
        for key, value in _balancing_hyperparameter[[*_balancing].index(balancing)].items():
            balancing_hyper[key] = distribution_to_suggest(trial, balancing, key, value, algo = "Optuna")
            
        scaling_hyper = {}
        scaling_hyper["scaling"] = scaling
        for key, value in _scaling_hyperparameter[[*_scaling].index(scaling)].items():
            scaling_hyper[key] = distribution_to_suggest(trial, scaling, key, value, algo = "Optuna")
            
        feature_selection_hyper = {}
        feature_selection_hyper["feature_selection"] = feature_selection
        for key, value in _feature_selection_hyperparameter[[*_feature_selection].index(feature_selection)].items():
            feature_selection_hyper[key] = distribution_to_suggest(trial, feature_selection, key, value, algo = "Optuna")
            
        model_hyper = {}
        model_hyper["model"] = model
        for key, value in _model_hyperparameter[[*_models].index(model)].items():
            model_hyper[key] = distribution_to_suggest(trial, model, key, value, algo = "Optuna")
            
        return {
            "task_type": "tabular_" + task_mode,
            "encoder": encoder_hyper,
            "imputer": imputer_hyper,
            "balancing": balancing_hyper,
            "scaling": scaling_hyper,
            "feature_selection": feature_selection_hyper,
            "model": model_hyper,
        }
    
    return partial(search_space,
        _encoder = encoder,
        _imputer = imputer,
        _balancing = balancing,
        _scaling = scaling,
        _feature_selection = feature_selection,
        _models = models,
        _encoding_hyperparameter = _encoding_hyperparameter,
        _imputer_hyperparameter = _imputer_hyperparameter,
        _balancing_hyperparameter = _balancing_hyperparameter,
        _scaling_hyperparameter = _scaling_hyperparameter,
        _feature_selection_hyperparameter = _feature_selection_hyperparameter,
        _model_hyperparameter = _model_hyperparameter,
        task_mode = task_mode,
    )
    
def _converted_get_hyperparameter_space(
    X: pd.DataFrame,
    encoders_hyperparameters: Dict,
    encoder: Dict,
    imputers_hyperparameters: Dict,
    imputer: Dict,
    balancings_hyperparameters: Dict,
    balancing: Dict,
    scalings_hyperparameters: Dict,
    scaling: Dict,
    feature_selection_hyperparameters: Dict,
    feature_selection: Dict,
    models_hyperparameters: Dict,
    models: Dict,
    task_mode: str,
    search_algo: str = "Nevergrad",
) -> Dict:
    # Note: Nov. 15, 2022
    # this will be checked in get_algo, no need to check here
    # # check installation of optuna
    # nevergrad_spec = importlib.util.find_spec("nevergrad")
    # if nevergrad_spec is None:
    #     raise ImportError(
    #         "nevergrad not installed. Please install it first to use Nevergrad. \
    #         Command to install: pip install nevergrad"
    #     )
        
    if search_algo == "Nevergrad":
        from ray.tune.suggest.nevergrad import NevergradSearch
        searcher = NevergradSearch
    elif search_algo == "BlendSearch":
        from ray.tune.suggest.flaml import BlendSearch
        searcher = BlendSearch
    
    # encoding space
    # get all method names in hyperparameter space
    all_encoders = [item["encoder"] for item in encoders_hyperparameters]
    _encoding_hyperparameter = []
    for _encoder in [*encoder]:
        # need a layer of hyperparameter space to retain it
        encoders_hyperparameters[all_encoders.index(_encoder)]["encoder"] = tune.choice(
            [encoders_hyperparameters[all_encoders.index(_encoder)]["encoder"]]
        )
        _encoding_hyperparameter.append(
            searcher.convert_search_space(encoders_hyperparameters[all_encoders.index(_encoder)])
        )
        
    # raise error if no encoding hyperparameters are found
    if len(_encoding_hyperparameter) == 0:
        raise ValueError(
            "No encoding hyperparameters are found. Please check your encoders."
        )
    
    # imputation space
    # get all method names in hyperparameter space
    all_imputers = [item["imputer"] for item in imputers_hyperparameters]
    _imputer_hyperparameter = []
    for _imputer in [*imputer]:
        imputers_hyperparameters[all_imputers.index(_imputer)]["imputer"] = tune.choice(
            [imputers_hyperparameters[all_imputers.index(_imputer)]["imputer"]]
        )
        _imputer_hyperparameter.append(
            searcher.convert_search_space(imputers_hyperparameters[all_imputers.index(_imputer)])
        )
        
    # raise error if no encoding hyperparameters are found
    if len(_imputer_hyperparameter) == 0:
        raise ValueError(
            "No imputer hyperparameters are found. Please check your encoders."
        )
    
    # balancing space
    # get all method names in hyperparameter space
    all_balancings = [item["balancing"] for item in balancings_hyperparameters]
    _balancing_hyperparameter = []
    for _balancing in [*balancing]:
        balancings_hyperparameters[all_balancings.index(_balancing)]["balancing"] = tune.choice(
            [balancings_hyperparameters[all_balancings.index(_balancing)]["balancing"]]
        )
        _balancing_hyperparameter.append(
            searcher.convert_search_space(balancings_hyperparameters[all_balancings.index(_balancing)])
        )
        
    # raise error if no encoding hyperparameters are found
    if len(_balancing_hyperparameter) == 0:
        raise ValueError(
            "No balancing hyperparameters are found. Please check your encoders."
        )
    
    # scaling space
    # get all method names in hyperparameter space
    all_scalings = [item["scaling"] for item in scalings_hyperparameters]
    _scaling_hyperparameter = []
    for _scaling in [*scaling]:
        scalings_hyperparameters[all_scalings.index(_scaling)]["scaling"] = tune.choice(
            [scalings_hyperparameters[all_scalings.index(_scaling)]["scaling"]]
        )
        _scaling_hyperparameter.append(
            searcher.convert_search_space(scalings_hyperparameters[all_scalings.index(_scaling)])
        )
        
    # raise error if no encoding hyperparameters are found
    if len(_scaling_hyperparameter) == 0:
        raise ValueError(
            "No scaling hyperparameters are found. Please check your encoders."
        )
    
    # feature selection space
    # get all method names in hyperparameter space
    all_feature_selections = [item["feature_selection"] for item in feature_selection_hyperparameters]
    _feature_selection_hyperparameter = []
    for _feature_selection in [*feature_selection]:
        feature_selection_hyperparameters[
            all_feature_selections.index(_feature_selection)
        ]["feature_selection"] = tune.choice(
            [feature_selection_hyperparameters[
                all_feature_selections.index(_feature_selection)
            ]["feature_selection"]]
        )
        _feature_selection_hyperparameter.append(
            searcher.convert_search_space(
                feature_selection_hyperparameters[all_feature_selections.index(_feature_selection)]
            )
        )
        
    # raise error if no encoding hyperparameters are found
    if len(_feature_selection_hyperparameter) == 0:
        raise ValueError(
            "No feature selection hyperparameters are found. Please check your encoders."
        )
    
    # model space
    # get all method names in hyperparameter space
    all_models = [item["model"] for item in models_hyperparameters]
    _model_hyperparameter = []
    for _model in [*models]:
        models_hyperparameters[all_models.index(_model)]["model"] = tune.choice(
            [models_hyperparameters[all_models.index(_model)]["model"]]
        )
        _model_hyperparameter.append(
            searcher.convert_search_space(models_hyperparameters[all_models.index(_model)])
        )
        
    # raise error if no encoding hyperparameters are found
    if len(_model_hyperparameter) == 0:
        raise ValueError(
            "No model hyperparameters are found. Please check your encoders."
        )
    
    # the pipeline search space
    # select one of the method/hyperparameter setting from each part
    return {
        "task_type": "tabular_" + task_mode,
        "encoder": tune.choice(_encoding_hyperparameter),
        "imputer": tune.choice(_imputer_hyperparameter),
        "balancing": tune.choice(_balancing_hyperparameter),
        "scaling": tune.choice(_scaling_hyperparameter),
        "feature_selection": tune.choice(_feature_selection_hyperparameter),
        "model": tune.choice(_model_hyperparameter),
    }    
    
# Update: Nov. 9, 2022
# skopt have quite different idea of hyperparameter space, so it's not used here.
# def _skopt_get_hyperparameter_space(
#     X,
#     encoders_hyperparameters,
#     encoder,
#     imputers_hyperparameters,
#     imputer,
#     balancings_hyperparameters,
#     balancing,
#     scalings_hyperparameters,
#     scaling,
#     feature_selection_hyperparameters,
#     feature_selection,
#     models_hyperparameters,
#     models,
#     task_mode,
# ):
#     # check installation of optuna
#     skopt_spec = importlib.util.find_spec("skopt")
#     if skopt_spec is None:
#         raise ImportError(
#             "skopt not installed. Please install it first to use Scikit-Optimize. \
#             Command to install: pip install scikit-optimize"
#         )
        
#     from ray.tune.suggest.skopt import SkOptSearch
    
#     # encoding space
#     # get all method names in hyperparameter space
#     all_encoders = [item["encoder"] for item in encoders_hyperparameters]
#     _encoding_hyperparameter = []
#     for _encoder in [*encoder]:
#         _encoding_hyperparameter.append(
#             SkOptSearch.convert_search_space(encoders_hyperparameters[all_encoders.index(_encoder)])
#         )
        
#     # raise error if no encoding hyperparameters are found
#     if len(_encoding_hyperparameter) == 0:
#         raise ValueError(
#             "No encoding hyperparameters are found. Please check your encoders."
#         )

#     _encoding_hyperparameter = tune.choice(_encoding_hyperparameter)
    
#     # imputation space
#     # get all method names in hyperparameter space
#     all_imputers = [item["imputer"] for item in imputers_hyperparameters]
#     _imputer_hyperparameter = []
#     for _imputer in [*imputer]:
#         _imputer_hyperparameter.append(
#             SkOptSearch.convert_search_space(imputers_hyperparameters[all_imputers.index(_imputer)])
#         )
        
#     # raise error if no encoding hyperparameters are found
#     if len(_imputer_hyperparameter) == 0:
#         raise ValueError(
#             "No imputer hyperparameters are found. Please check your encoders."
#         )

#     _imputer_hyperparameter = tune.choice(_imputer_hyperparameter)
    
#     # balancing space
#     # get all method names in hyperparameter space
#     all_balancings = [item["balancing"] for item in balancings_hyperparameters]
#     _balancing_hyperparameter = []
#     for _balancing in [*balancing]:
#         _balancing_hyperparameter.append(
#             SkOptSearch.convert_search_space(balancings_hyperparameters[all_balancings.index(_balancing)])
#         )
        
#     # raise error if no encoding hyperparameters are found
#     if len(_balancing_hyperparameter) == 0:
#         raise ValueError(
#             "No balancing hyperparameters are found. Please check your encoders."
#         )

#     _balancing_hyperparameter = tune.choice(_balancing_hyperparameter)
    
#     # scaling space
#     # get all method names in hyperparameter space
#     all_scalings = [item["scaling"] for item in scalings_hyperparameters]
#     _scaling_hyperparameter = []
#     for _scaling in [*scaling]:
#         _scaling_hyperparameter.append(
#             SkOptSearch.convert_search_space(scalings_hyperparameters[all_scalings.index(_scaling)])
#         )
        
#     # raise error if no encoding hyperparameters are found
#     if len(_scaling_hyperparameter) == 0:
#         raise ValueError(
#             "No scaling hyperparameters are found. Please check your encoders."
#         )

#     _scaling_hyperparameter = tune.choice(_scaling_hyperparameter)
    
#     # feature selection space
#     # get all method names in hyperparameter space
#     all_feature_selections = [item["feature_selection"] for item in feature_selection_hyperparameters]
#     _feature_selection_hyperparameter = []
#     for _feature_selection in [*feature_selection]:
#         _feature_selection_hyperparameter.append(
#             SkOptSearch.convert_search_space(
#                 feature_selection_hyperparameters[all_feature_selections.index(_feature_selection)]
#             )
#         )
        
#     # raise error if no encoding hyperparameters are found
#     if len(_feature_selection_hyperparameter) == 0:
#         raise ValueError(
#             "No feature selection hyperparameters are found. Please check your encoders."
#         )

#     _feature_selection_hyperparameter = tune.choice(_feature_selection_hyperparameter)
    
#     # model space
#     # get all method names in hyperparameter space
#     all_models = [item["model"] for item in models_hyperparameters]
#     _model_hyperparameter = []
#     for _model in [*models]:
#         _model_hyperparameter.append(
#             SkOptSearch.convert_search_space(models_hyperparameters[all_models.index(_model)])
#         )
        
#     # raise error if no encoding hyperparameters are found
#     if len(_model_hyperparameter) == 0:
#         raise ValueError(
#             "No model hyperparameters are found. Please check your encoders."
#         )

#     _model_hyperparameter = tune.choice(_model_hyperparameter)
    
#     # the pipeline search space
#     # select one of the method/hyperparameter setting from each part
#     return {
#         "task_type": "tabular_" + task_mode,
#         "encoder": _encoding_hyperparameter,
#         "imputer": _imputer_hyperparameter,
#         "balancing": _balancing_hyperparameter,
#         "scaling": _scaling_hyperparameter,
#         "feature_selection": _feature_selection_hyperparameter,
#         "model": _model_hyperparameter,
#     }    
        

def _get_hyperparameter_space(
    X: pd.DataFrame,
    encoders_hyperparameters: Dict,
    encoder: Dict,
    imputers_hyperparameters: Dict,
    imputer: Dict,
    balancings_hyperparameters: Dict,
    balancing: Dict,
    scalings_hyperparameters: Dict,
    scaling: Dict,
    feature_selection_hyperparameters: Dict,
    feature_selection: Dict,
    models_hyperparameters: Dict,
    models: Dict,
    task_mode: str,
    search_aglo: str,
) -> Union[Dict, Callable]:
    if search_aglo in ["RandomSearch", "GridSearch", "CFO"]:
        # BlendSearch can start from iter #2
        return _base_get_hyperparameter_space(
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
        )
    elif search_aglo in ["HyperOpt"] :
        return _hyerpot_get_hyperparameter_space(
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
        )
    elif search_aglo in ["Optuna"] :
        with DisableLogger() :
            return _optuna_get_hyperparameter_space(
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
            )
    elif search_aglo in ["Nevergrad"] :
        with DisableLogger() :
            return _converted_get_hyperparameter_space(
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
                search_algo = search_aglo,
            )
    # elif search_aglo in ["Scikit-Optimize"] :
    #     return _skopt_get_hyperparameter_space(
    #         X,
    #         encoders_hyperparameters,
    #         encoder,
    #         imputers_hyperparameters,
    #         imputer,
    #         balancings_hyperparameters,
    #         balancing,
    #         scalings_hyperparameters,
    #         scaling,
    #         feature_selection_hyperparameters,
    #         feature_selection,
    #         models_hyperparameters,
    #         models,
    #         task_mode,
    #     )


# get the hyperparameter optimization algorithm based on string input
def get_algo(search_algo: str) -> Callable:

    if search_algo == "RandomSearch" or search_algo == "GridSearch":

        # Random Search and Grid Search
        from ray.tune.suggest.basic_variant import BasicVariantGenerator

        algo = BasicVariantGenerator
    # Update: Nov. 9, 2022
    # The method BayesOptSearch requires non-discrete hyperparameter space, which is not supported 
    # by the our version of space.
    # The work around is to generate a continuous range for discrete hyperparameters, and then 
    # convert the continuous range to discrete range in the objective function.
    # e.g. for binary categorical hyperparameter, we can generate a continuous range from 0 to 1,
    # and then convert the continuous range to 0 or 1 in the objective function.
    # This is not within the scope of the current version.
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
    # elif search_algo == "AxSearch":

    #     # check whether Ax and sqlalchemy are installed
    #     Ax_spec = importlib.util.find_spec("ax")
    #     sqlalchemy_spec = importlib.util.find_spec("sqlalchemy")
    #     if Ax_spec is None or sqlalchemy_spec is None:
    #         raise ImportError(
    #             "Ax or sqlalchemy not installed. Please install these packages to use AxSearch. \
    #                 Command to install: pip install ax-platform sqlalchemy"
    #         )

    #     # Ax Search
    #     from ray.tune.suggest.ax import AxSearch

    #     algo = AxSearch
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
    # elif search_algo == "BlendSearch":

    #     # check whether flaml is installed
    #     flaml_spec = importlib.util.find_spec("flaml")
    #     if flaml_spec is None:
    #         raise ImportError(
    #             "flaml not installed. Please install it first to use BlendSearch. \nCommand to install: pip install 'flaml[blendsearch]'"
    #         )

    #     # Blend Search
    #     from ray.tune.suggest.flaml import BlendSearch

    #     algo = BlendSearch
    elif search_algo == "CFO":

        # check whether flaml is installed
        flaml_spec = importlib.util.find_spec("flaml")
        if flaml_spec is None:
            raise ImportError(
                "flaml not installed. Please install it first to use CFO. \nCommand to install: pip install flaml"
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
    # elif search_algo == "HEBO":

    #     # check whether HEBO is installed
    #     HEBO_spec = importlib.util.find_spec("HEBO")
    #     if HEBO_spec is None:
    #         raise ImportError(
    #             "HEBO not installed. Please install it first to use HEBO. \
    #             Command to install: pip install 'HEBO>=0.2.0'"
    #         )

    #     # Heteroscedastic Evolutionary Bayesian Optimization/HEBO
    #     from ray.tune.suggest.hebo import HEBOSearch

    #     algo = HEBOSearch
    elif search_algo == "HyperOpt":

        # check whether hyperopt is installed
        hyperopt_spec = importlib.util.find_spec("hyperopt")
        if hyperopt_spec is None:
            raise ImportError(
                "hyperopt not installed. Please install it first to use HyperOpt. \nCommand to install: pip install -U hyperopt"
            )

        # HyperOpt Search
        from ray.tune.suggest.hyperopt import HyperOptSearch

        algo = HyperOptSearch
    elif search_algo == "Nevergrad":

        # check whether nevergrad is installed
        nevergrad_spec = importlib.util.find_spec("nevergrad")
        if nevergrad_spec is None:
            raise ImportError(
                "nevergrad not installed. Please install it first to use Nevergrad. \nCommand to install: pip install nevergrad"
            )

        # Nevergrad Search
        from ray.tune.suggest.nevergrad import NevergradSearch

        algo = NevergradSearch
    # default hyeprparameter space can not be easily converted
    elif search_algo == "Optuna":

        # check whether optuna is installed
        optuna_spec = importlib.util.find_spec("optuna")
        if optuna_spec is None:
            raise ImportError(
                "optuna not installed. Please install it first to use Optuna. \nCommand to install: pip install optuna"
            )

        # Optuna Search
        from ray.tune.suggest.optuna import OptunaSearch

        algo = OptunaSearch
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
    # Update: Nov. 9, 2022
    # skopt have quite different idea of hyperparameter space, so it's not used here.
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
def get_scheduler(search_scheduler: str) -> Callable:

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
    progress_reporter: str,
    max_evals: int,
    max_error: int,
) -> Callable:

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


def get_logger(logger: List) -> List[Callable]:

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
        timeout: int=360,
        metric: str="loss",
        std_ratio: float=0.01,
        num_results: int=4,
        grace_period: int=4,
        metric_threshold: float=None,
        mode: str="min",
    ) -> None:
        self._start = time.time()
        self._deadline = timeout

        self._metric = metric
        self._mode = mode

        self._std_ratio = std_ratio
        self._num_results = num_results
        self._grace_period = grace_period
        self._metric_threshold = metric_threshold

        self._iter = defaultdict(lambda: 0)
        self._trial_results = defaultdict(lambda: deque(maxlen=self._num_results))

    def __call__(self, trial_id: str, result: Any) -> bool:

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

        # if threshold not specified, use std/mean to stop
        # Calculate mean/stdev of last `num_results` results
        try:
            current_std = np.std(self._trial_results[trial_id])
        except Exception:
            current_std = float("inf")
            
        try: 
            current_mean = np.mean(self._trial_results[trial_id])
        except Exception:
            current_mean = float("inf")

        # If stdev is lower than threshold, stop early.
        return current_std / current_mean < self._std_ratio

    def stop_all(self) -> bool:

        return time.time() - self._start > self._deadline


# get estimator based on string or class
def get_estimator(estimator_str: str) -> Callable:

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
def get_metrics(metric_str: str) -> Callable:

    if metric_str == "neg_accuracy":
        from InsurAutoML._utils._stat import neg_accuracy

        return neg_accuracy
    elif metric_str == "accuracy":
        from sklearn.metrics import accuracy_score
        
        logger.warn("accuracy_score is not for min mode, please use neg_accuracy instead.")
        # warnings.warn(
        #     "accuracy_score is not for min mode, please use neg_accuracy instead."
        # )
        return accuracy_score
    elif metric_str == "neg_precision":
        from InsurAutoML._utils._stat import neg_precision

        return neg_precision
    elif metric_str == "precision":
        from sklearn.metrics import precision_score

        logger.warn("precision_score is not for min mode, please use neg_precision instead.")
        # warnings.warn(
        #     "precision_score is not for min mode, please use neg_precision instead."
        # )
        return precision_score
    elif metric_str == "neg_auc":
        from InsurAutoML._utils._stat import neg_auc

        return neg_auc
    elif metric_str == "auc":
        from sklearn.metrics import roc_auc_score

        logger.warn("roc_auc_score is not for min mode, please use neg_auc instead.")
        # warnings.warn("roc_auc_score is not for min mode, please use neg_auc instead.")
        return roc_auc_score
    elif metric_str == "neg_hinge":
        from InsurAutoML._utils._stat import neg_hinge

        return neg_hinge
    elif metric_str == "hinge":
        from sklearn.metrics import hinge_loss
        
        logger.warn("hinge_loss is not for min mode, please use neg_hinge instead.")
        # warnings.warn("hinge_loss is not for min mode, please use neg_hinge instead.")
        return hinge_loss
    elif metric_str == "neg_f1":
        from InsurAutoML._utils._stat import neg_f1

        return neg_f1
    elif metric_str == "f1":
        from sklearn.metrics import f1_score

        logger.warn("f1_score is not for min mode, please use neg_f1 instead.")
        # warnings.warn("f1_score is not for min mode, please use neg_f1 instead.")
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
        from InsurAutoML._utils._stat import neg_R2

        return neg_R2
    elif metric_str == "R2":
        from sklearn.metrics import r2_score
        
        logger.warn("r2_score is not for min mode, please use neg_R2 instead.")
        # warnings.warn("r2_score is not for min mode, please use neg_R2 instead.")
        return r2_score
    elif metric_str == "MAX":
        from sklearn.metrics import max_error

        return max_error
    elif isinstance(metric_str, Callable):
        # if callable, pass
        return metric_str
    else:
        raise ValueError("Unrecognized criteria!")


# ray initialization and shutdown
class ray_status:
    def __init__(
        self,
        cpu_threads: int,
        gpu_count: int,
    ) -> None:
        self.cpu_threads = cpu_threads
        self.gpu_count = gpu_count

    def ray_init(self) -> None:

        # initialize ray
        # if already initialized, do nothing
        if not ray.is_initialized():
            ray.init(
                # local_mode=True,
                num_cpus=self.cpu_threads,
                num_gpus=self.gpu_count,
            )
        # check if ray is initialized
        assert ray.is_initialized() == True, "Ray is not initialized."

    def ray_shutdown(self) -> None:

        # shut down ray
        ray.shutdown()
        # check if ray is shutdown
        assert ray.is_initialized() == False, "Ray is not shutdown."
        
# check if an object has a method
def check_func(obj: Callable, ref: str) -> None:
    
    for _func in METHOD_MAPPING[ref]:
        if not callable(getattr(obj, _func, None)):
            raise AttributeError(f"{ref} {obj} does not have {_func} method!")
        
    return None

# check hyperparameter space
def check_status(
    methods: Dict,
    hyperparameter_space: List[Dict],
    ref: str = "encoder",
) -> None:
    # get all method names in hyperparameter space
    all_methods = []
    for _dict in hyperparameter_space :
        # get the method name by reference key
        for key in _dict.keys() :
            if ref in key :
                all_methods.append(_dict[key])
                break
    # check status of encoder hyperparameter space
    for method_name, method_object in methods.items():
        # check whether the method exists
        if method_name not in all_methods :
            raise ValueError(
                "Method {} is not in {} hyperparameter space!".format(method_name, ref)
            )
        # check whether the method has corresponding functions
        check_func(method_object, ref)
        
        # check whether the hyperparameter space is valid
        # get corresponding hyperparameter space
        for key in hyperparameter_space[all_methods.index(method_name)].keys() :
            if ref in key :
                continue
            else :
                if not key in method_object.__init__.__code__.co_varnames :
                    raise ValueError(
                        "Hyperparameter {} is not in {}!".format(key, method_name)
                    )
        
    return None
        
if __name__ == "__main__" :
    pass
        