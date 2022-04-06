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
Last Modified: Tuesday, 5th April 2022 11:00:53 pm
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

from My_AutoML import (
    no_processing,  # base
    type_of_task,  # utils
    formatting,
)
from ._base import AutoTabularClassifier, AutoTabularRegressor


class AutoTabular(AutoTabularClassifier, AutoTabularRegressor):

    """
    Automatically assign to AutoTabularClassifier or AutoTabularRegressor
    """

    def __init__(
        self,
        timeout=360,
        max_evals=64,
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
        method="Bayesian",
        algo="tpe",
        spark_trials=False,
        progressbar=True,
        seed=1,
    ):
        self.timeout = timeout
        self.max_evals = max_evals
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
        self.method = method
        self.algo = algo
        self.spark_trials = spark_trials
        self.progressbar = progressbar
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
                method=self.method,
                algo=self.algo,
                spark_trials=self.spark_trials,
                progressbar=self.progressbar,
                seed=self.seed,
            )
        elif self._type in ["integer", "continuous"]:  # assign regression tasks
            self.model = AutoTabularRegressor(
                timeout=self.timeout,
                max_evals=self.max_evals,
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
                method=self.method,
                algo=self.algo,
                spark_trials=self.spark_trials,
                progressbar=self.progressbar,
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
