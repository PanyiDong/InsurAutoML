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
Last Modified: Wednesday, 6th April 2022 10:17:59 pm
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

    def fit(self, X, y):

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
