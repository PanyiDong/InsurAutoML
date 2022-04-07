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
Last Modified: Thursday, 7th April 2022 9:13:10 am
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
import warnings
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from My_AutoML import encoders
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
from My_AutoML._utils._file import save_model

# filter certain warnings
warnings.filterwarnings("ignore", message="The dataset is balanced, no change.")
warnings.filterwarnings("ignore", message="Variables are collinear")
warnings.filterwarnings("ignore", category=UserWarning)


class AutoTabularBase:
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
        objective="accuracy",
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

        self._iter = 0  # record iteration number

    def get_hyperparameter_space(self, X, y):

        # initialize default search options
        # use copy to allows multiple manipulation
        # all encoders available
        self._all_encoders = encoders.copy()

        # all hyperparameters for encoders
        self._all_encoders_hyperparameters = encoder_hyperparameter.copy()

        # all imputers available
        self._all_imputers = imputers.copy()

        # all hyperparemeters for imputers
        self._all_imputers_hyperparameters = imputer_hyperparameter.copy()

        # all scalings available
        self._all_scalings = scalings.copy()

        # all balancings available
        self._all_balancings = balancings.copy()

        # all hyperparameters for balancing methods
        self._all_balancings_hyperparameters = balancing_hyperparameter.copy()

        # all hyperparameters for scalings
        self._all_scalings_hyperparameters = scaling_hyperparameter.copy()

        # all feature selections available
        self._all_feature_selection = feature_selections.copy()
        # special treatment, remove some feature selection for regression
        del self._all_feature_selection["extra_trees_preproc_for_regression"]
        del self._all_feature_selection["select_percentile_regression"]
        del self._all_feature_selection["select_rates_regression"]
        if X.shape[0] * X.shape[1] > 10000:
            del self._all_feature_selection["liblinear_svc_preprocessor"]

        # all hyperparameters for feature selections
        self._all_feature_selection_hyperparameters = (
            feature_selection_hyperparameter.copy()
        )

        # all classification models available
        self._all_models = classifiers.copy()
        # special treatment, remove SVM methods when observations are large
        # SVM suffers from the complexity o(n_samples^2 * n_features),
        # which is time-consuming for large datasets
        if X.shape[0] * X.shape[1] > 10000:
            del self._all_models["LibLinear_SVC"]
            del self._all_models["LibSVM_SVC"]

        # all hyperparameters for the classification models
        self._all_models_hyperparameters = classifier_hyperparameter.copy()

        # initialize default search space
        self.hyperparameter_space = None

        # select from setttings

        return encoder, imputer, balancing, scaling, feature_selection, models

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

        raise NotImplementedError("This method is not implemented in the base class.")

    def predict(self, X):

        raise NotImplementedError("This method is not implemented in the base class.")


scaling_hyperparameter = tune.choice(
    [
        {"scaling": "NoScaling", "length": tune.qrandint(1, 10, 1)},
        {"scaling": "Standardize", "length": tune.qrandint(3, 5, 1)},
    ]
)


def objective(config):
    loss = config["length"]
    return {"loss": loss, "status": "fitted"}


analysis = tune.run(
    objective,
    config=scaling_hyperparameter,
    num_samples=10,
    mode="min",
    metric="loss",
    stop={"training_iteration": 100},
)

print(analysis.best_config)
