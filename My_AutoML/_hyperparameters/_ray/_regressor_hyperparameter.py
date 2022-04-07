"""
File: _regressor_hyperparameter.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Relative Path: /My_AutoML/_hyperparameters/_regressor_hyperparameter.py
File Created: Tuesday, 5th April 2022 11:06:33 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Thursday, 7th April 2022 12:27:00 am
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

# regressor hyperparameters
regressor_hyperparameter = [
    # extract from autosklearn
    {
        "model": "AdaboostRegressor",
        "n_estimators": tune.qrandint(50, 500, 1),
        "learning_rate": tune.loguniform(0.01, 2),
        "loss": tune.choice(["linear", "square", "exponential"]),
        # for base_estimator of Decision Tree
        "max_depth": tune.qrandint(1, 10, 1),
    },
    {
        "model": "ARDRegression",
        "n_iter": tune.choice([300]),
        "tol": tune.loguniform(1e-5, 1e-1),
        "alpha_1": tune.loguniform(1e-10, 1e-3),
        "alpha_2": tune.loguniform(1e-10, 1e-3),
        "lambda_1": tune.loguniform(1e-10, 1e-3),
        "lambda_2": tune.loguniform(1e-10, 1e-3),
        "threshold_lambda": tune.loguniform(1e3, 1e5),
        "fit_intercept": tune.choice([True]),
    },
    {
        "model": "DecisionTree",
        "criterion": tune.choice(["mse", "friedman_mse", "mae"]),
        "max_features": tune.choice([1.0]),
        "max_depth_factor": tune.uniform(0.0, 2.0),
        "min_samples_split": tune.qrandint(2, 20, 1),
        "min_samples_leaf": tune.qrandint(1, 20, 1),
        "min_weight_fraction_leaf": tune.choice([0.0]),
        "max_leaf_nodes": tune.choice([None]),
        "min_impurity_decrease": tune.choice([0.0]),
    },
    {
        "model": "ExtraTreesRegressor",
        "criterion": tune.choice(["mse", "friedman_mse", "mae"]),
        "min_samples_leaf": tune.qrandint(1, 20, 1),
        "min_samples_split": tune.qrandint(2, 20, 1),
        "max_features": tune.uniform(0.1, 1.0),
        "bootstrap": tune.choice([True, False]),
        "max_leaf_nodes": tune.choice([None]),
        "max_depth": tune.choice([None]),
        "min_weight_fraction_leaf": tune.choice([0.0]),
        "min_impurity_decrease": tune.choice([0.0]),
    },
    {
        "model": "GaussianProcess",
        "alpha": tune.loguniform(1e-14, 1),
        "thetaL": tune.loguniform(1e-10, 1e-3),
        "thetaU": tune.loguniform(1, 1e5),
    },
    {
        "model": "GradientBoosting",
        # n_iter_no_change only selected for early_stop in ['valid', 'train']
        # validation_fraction only selected for early_stop = 'valid'
        "loss": tune.choice(["least_squares"]),
        "learning_rate": tune.loguniform(0.01, 1),
        "min_samples_leaf": tune.qlograndint(1, 200, 1),
        "max_depth": tune.choice([None]),
        "max_leaf_nodes": tune.qlograndint(3, 2047, 1),
        "max_bins": tune.choice([255]),
        "l2_regularization": tune.loguniform(1e-10, 1),
        "early_stop": tune.choice(["off", "train", "valid"]),
        "tol": tune.choice([1e-7]),
        "scoring": tune.choice(["loss"]),
        "n_iter_no_change": tune.qrandint(1, 20, 1),
        "validation_fraction": tune.uniform(0.01, 0.4),
    },
    {
        "model": "KNearestNeighborsRegressor",
        "n_neighbors": tune.qrandint(1, 100, 1),
        "weights": tune.choice(["uniform", "distance"]),
        "p": tune.choice([1, 2]),
    },
    {
        "model": "LibLinear_SVR",
        # forbid loss = 'epsilon_insensitive' and dual = False
        "epsilon": tune.loguniform(0.001, 1),
        "loss": tune.choice(
            ["squared_epsilon_insensitive"],
        ),
        "dual": tune.choice([False]),
        "tol": tune.loguniform(1e-5, 1e-1),
        "C": tune.loguniform(0.03125, 32768),
        "fit_intercept": tune.choice([True]),
        "intercept_scaling": tune.choice([1]),
    },
    {
        "model": "LibSVM_SVR",
        # degree only selected for kernel in ['poly', 'rbf', 'sigmoid']
        # gamma only selected for kernel in ['poly', 'rbf']
        # coef0 only selected for kernel in ['poly', 'sigmoid']
        "kernel": tune.choice(["linear", "poly", "rbf", "sigmoid"]),
        "C": tune.loguniform(0.03125, 32768),
        "epsilon": tune.uniform(1e-5, 1),
        "tol": tune.loguniform(1e-5, 1e-1),
        "shrinking": tune.choice([True, False]),
        "degree": tune.qrandint(2, 5, 1),
        "gamma": tune.loguniform(3.0517578125e-5, 8),
        "coef0": tune.uniform(-1, 1),
        "max_iter": tune.choice([-1]),
    },
    {
        "model": "MLPRegressor",
        # validation_fraction only selected for early_stopping = 'valid'
        "hidden_layer_depth": tune.qrandint(1, 3, 1),
        "num_nodes_per_layer": tune.qlograndint(16, 264, 1),
        "activation": tune.choice(["tanh", "relu"]),
        "alpha": tune.loguniform(1e-7, 1e-1),
        "learning_rate_init": tune.loguniform(1e-4, 0.5),
        "early_stopping": tune.choice(["valid", "train"]),
        "solver": tune.choice(["adam"]),
        "batch_size": tune.choice(["auto"]),
        "n_iter_no_change": tune.choice([32]),
        "tol": tune.choice([1e-4]),
        "shuffle": tune.choice([True]),
        "beta_1": tune.choice([0.9]),
        "beta_2": tune.choice([0.999]),
        "epsilon": tune.choice([1e-8]),
        "validation_fraction": tune.choice([0.1]),
    },
    {
        "model": "RandomForest",
        "criterion": tune.choice(["mse", "friedman_mse", "mae"]),
        "max_features": tune.uniform(0.1, 1.0),
        "max_depth": tune.choice([None]),
        "min_samples_split": tune.qrandint(2, 20, 1),
        "min_samples_leaf": tune.qrandint(1, 20, 1),
        "min_weight_fraction_leaf": tune.choice([0.0]),
        "bootstrap": tune.choice([True, False]),
        "max_leaf_nodes": tune.choice([None]),
        "min_impurity_decrease": tune.choice([0.0]),
    },
    {
        "model": "SGD",
        # l1_ratio only selected for penalty = 'elasticnet'
        # epsilon only selected for loss in ['huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
        # eta0 only selected for learning_rate in ['constant', 'invscaling']
        # power_t only selected for learning_rate = 'invscaling'
        "loss": tune.choice(
            [
                "squared_loss",
                "huber",
                "epsilon_insensitive",
                "squared_epsilon_insensitive",
            ],
        ),
        "penalty": tune.choice(["l1", "l2", "elasticnet"]),
        "alpha": tune.loguniform(1e-7, 1e-1),
        "fit_intercept": tune.choice([True]),
        "tol": tune.loguniform(1e-5, 1e-1),
        "learning_rate": tune.choice(["constant", "optimal", "invscaling"]),
        "l1_ratio": tune.loguniform(1e-9, 1.0),
        "epsilon": tune.loguniform(1e-5, 1e-1),
        "power_t": tune.uniform(1e-5, 1),
        "average": tune.choice([True, False]),
    },
    # self-defined models
    {
        "model": "MLP_Regressor",
        "hidden_layer": tune.qrandint(1, 5, 1),
        "hidden_size": tune.qrandint(1, 20, 1),
        "activation": tune.choice(["ReLU", "Tanh", "Sigmoid"]),
        "learning_rate": tune.uniform(1e-5, 1),
        "optimizer": tune.choice(["Adam", "SGD"]),
        "criteria": tune.choice(["MSE", "MAE"]),
        "batch_size": tune.choice([16, 32, 64]),
        "num_epochs": tune.qrandint(5, 50, 1),
    },
]
