"""
File: _regressor_hyperparameter.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_hyperparameters/_hyperopt/_regressor_hyperparameter.py
File: _regressor_hyperparameter.py
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Tuesday, 15th November 2022 4:10:31 pm
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
from hyperopt import hp
from hyperopt.pyll import scope

# regressor hyperparameters
regressor_hyperparameter = [
    # extract from sklearn
    {
        "model": "AdaboostRegressor",
        "n_estimators": scope.int(
            hp.quniform("AdaboostRegressor_n_estimators", 50, 500, 1)
        ),
        "learning_rate": hp.loguniform(
            "AdaboostRegressor_learning_rate", np.log(0.01), np.log(2)
        ),
        "loss": hp.choice(
            "AdaboostRegressor_algorithm", ["linear", "square", "exponential"]
        ),
        # for base_estimator of Decision Tree
        "max_depth": scope.int(hp.quniform("AdaboostRegressor_max_depth", 1, 10, 1)),
    },
    {
        "model": "ARDRegression",
        "n_iter": hp.choice("ARDRegression_n_iter", [300]),
        "tol": hp.loguniform("ARDRegression_tol", np.log(1e-5), np.log(1e-1)),
        "alpha_1": hp.loguniform("ARDRegression_alpha_1", np.log(1e-10), np.log(1e-3)),
        "alpha_2": hp.loguniform("ARDRegression_alpha_2", np.log(1e-10), np.log(1e-3)),
        "lambda_1": hp.loguniform(
            "ARDRegression_lambda_1", np.log(1e-10), np.log(1e-3)
        ),
        "lambda_2": hp.loguniform(
            "ARDRegression_lambda_2", np.log(1e-10), np.log(1e-3)
        ),
        "threshold_lambda": hp.loguniform(
            "ARDRegression_threshold_lambda", np.log(1e3), np.log(1e5)
        ),
        "fit_intercept": hp.choice("ARDRegression_fit_intercept", [True]),
    },
    {
        "model": "DecisionTree",
        "criterion": hp.choice(
            "DecisionTree_criterion", ["mse", "friedman_mse", "mae"]
        ),
        "max_features": hp.choice("DecisionTree_max_features", [1.0]),
        "max_depth_factor": hp.uniform("DecisionTree_max_depth_factor", 0.0, 2.0),
        "min_samples_split": scope.int(
            hp.quniform("DecisionTree_min_samples_split", 2, 20, 1)
        ),
        "min_samples_leaf": scope.int(
            hp.quniform("DecisionTree_min_samples_leaf", 1, 20, 1)
        ),
        "min_weight_fraction_leaf": hp.choice(
            "DecisionTree_min_weight_fraction_leaf", [0.0]
        ),
        "max_leaf_nodes": hp.choice("DecisionTree_max_leaf_nodes", [None]),
        "min_impurity_decrease": hp.choice("DecisionTree_min_impurity_decrease", [0.0]),
    },
    {
        "model": "ExtraTreesRegressor",
        "criterion": hp.choice(
            "ExtraTreesRegressor_criterion", ["mse", "friedman_mse", "mae"]
        ),
        "min_samples_leaf": scope.int(
            hp.quniform("ExtraTreesRegressor_min_samples_leaf", 1, 20, 1)
        ),
        "min_samples_split": scope.int(
            hp.quniform("ExtraTreesRegressor_min_samples_split", 2, 20, 1)
        ),
        "max_features": hp.uniform("ExtraTreesRegressor_max_features", 0.1, 1.0),
        "bootstrap": hp.choice("ExtraTreesRegressor_bootstrap", [True, False]),
        "max_leaf_nodes": hp.choice("ExtraTreesRegressor_max_leaf_nodes", [None]),
        "max_depth": hp.choice("ExtraTreesRegressor_max_depth", [None]),
        "min_weight_fraction_leaf": hp.choice(
            "ExtraTreesRegressor_min_weight_fraction_leaf", [0.0]
        ),
        "min_impurity_decrease": hp.choice(
            "ExtraTreesRegressor_min_impurity_decrease", [0.0]
        ),
    },
    {
        "model": "GaussianProcess",
        "alpha": hp.loguniform("GaussianProcess_alpha", np.log(1e-14), np.log(1)),
        "thetaL": hp.loguniform("GaussianProcess_thetaL", np.log(1e-10), np.log(1e-3)),
        "thetaU": hp.loguniform("GaussianProcess_thetaU", np.log(1), np.log(1e5)),
    },
    {
        "model": "GradientBoosting",
        # n_iter_no_change only selected for early_stop in ['valid', 'train']
        # validation_fraction only selected for early_stop = 'valid'
        "loss": hp.choice("GradientBoosting_loss", ["least_squares"]),
        "learning_rate": hp.loguniform(
            "GradientBoosting_learning_rate", np.log(0.01), np.log(1)
        ),
        "min_samples_leaf": scope.int(
            hp.loguniform("GradientBoosting_min_samples_leaf", np.log(1), np.log(200))
        ),
        "max_depth": hp.choice("GradientBoosting_max_depth", [None]),
        "max_leaf_nodes": scope.int(
            hp.loguniform("GradientBoosting_max_leaf_nodes", np.log(3), np.log(2047))
        ),
        "max_bins": hp.choice("GradientBoosting_max_bins", [255]),
        "l2_regularization": hp.loguniform(
            "GradientBoosting_l2_regularization", np.log(1e-10), np.log(1)
        ),
        "early_stop": hp.choice(
            "GradientBoosting_early_stop", ["off", "train", "valid"]
        ),
        "tol": hp.choice("GradientBoosting_tol", [1e-7]),
        "scoring": hp.choice("GradientBoosting_scoring", ["loss"]),
        "n_iter_no_change": scope.int(
            hp.quniform("GradientBoosting_n_iter_no_change", 1, 20, 1)
        ),
        "validation_fraction": hp.uniform(
            "GradientBoosting_validation_fraction", 0.01, 0.4
        ),
    },
    {
        "model": "KNearestNeighborsRegressor",
        "n_neighbors": scope.int(
            hp.quniform("KNearestNeighborsRegressor_n_neighbors", 1, 100, 1)
        ),
        "weights": hp.choice(
            "KNearestNeighborsRegressor_weights", ["uniform", "distance"]
        ),
        "p": hp.choice("KNearestNeighborsRegressor_p", [1, 2]),
    },
    {
        "model": "LibLinear_SVR",
        # forbid loss = 'epsilon_insensitive' and dual = False
        "epsilon": hp.loguniform("LibLinear_SVR_tol", np.log(0.001), np.log(1)),
        "loss": hp.choice(
            "LibLinear_SVR__loss",
            ["squared_epsilon_insensitive"],
        ),
        "dual": hp.choice("LibLinear_SVR__dual", [False]),
        "tol": hp.loguniform("LibLinear_SVR__tol", np.log(1e-5), np.log(1e-1)),
        "C": hp.loguniform("LibLinear_SVR__C", np.log(0.03125), np.log(32768)),
        "fit_intercept": hp.choice("LibLinear_SVR__fit_intercept", [True]),
        "intercept_scaling": hp.choice("LibLinear_SVR__intercept_scaling", [1]),
    },
    {
        "model": "LibSVM_SVR",
        # degree only selected for kernel in ['poly', 'rbf', 'sigmoid']
        # gamma only selected for kernel in ['poly', 'rbf']
        # coef0 only selected for kernel in ['poly', 'sigmoid']
        "kernel": hp.choice("LibSVM_SVR_kernel", ["linear", "poly", "rbf", "sigmoid"]),
        "C": hp.loguniform("LibSVM_SVR__C", np.log(0.03125), np.log(32768)),
        "epsilon": hp.uniform("LibSVM_SVR_epsilon", 1e-5, 1),
        "tol": hp.loguniform("LibSVM_SVR__tol", np.log(1e-5), np.log(1e-1)),
        "shrinking": hp.choice("LibSVM_SVR_shrinking", [True, False]),
        "degree": scope.int(hp.quniform("LibSVM_SVR_degree", 2, 5, 1)),
        "gamma": hp.loguniform("LibSVM_SVR_gamma", np.log(3.0517578125e-5), np.log(8)),
        "coef0": hp.uniform("LibSVM_SVR_coef0", -1, 1),
        "max_iter": hp.choice("LibSVM_SVR_max_iter", [-1]),
    },
    {
        "model": "MLPRegressor",
        # validation_fraction only selected for early_stopping = 'valid'
        "hidden_layer_depth": scope.int(
            hp.quniform("MLPRegressor_hidden_layer_depth", 1, 3, 1)
        ),
        "num_nodes_per_layer": scope.int(
            hp.loguniform("MLPRegressor_num_nodes_per_layer", np.log(16), np.log(264))
        ),
        "activation": hp.choice("MLPRegressor_activation", ["tanh", "relu"]),
        "alpha": hp.loguniform("MLPRegressor_alpha", np.log(1e-7), np.log(1e-1)),
        "learning_rate_init": hp.loguniform(
            "MLPRegressor_learning_rate_init", np.log(1e-4), np.log(0.5)
        ),
        "early_stopping": hp.choice("MLPRegressor_early_stopping", ["valid", "train"]),
        "solver": hp.choice("MLPRegressor_solver", ["adam"]),
        "batch_size": hp.choice("MLPRegressor_batch_size", ["auto"]),
        "n_iter_no_change": hp.choice("MLPRegressor_n_iter_no_change", [32]),
        "tol": hp.choice("MLPRegressor_tol", [1e-4]),
        "shuffle": hp.choice("MLPRegressor_shuffle", [True]),
        "beta_1": hp.choice("MLPRegressor_beta_1", [0.9]),
        "beta_2": hp.choice("MLPRegressor_beta_2", [0.999]),
        "epsilon": hp.choice("MLPRegressor_epsilon", [1e-8]),
        "validation_fraction": hp.choice("MLPRegressor_validation_fraction", [0.1]),
    },
    {
        "model": "RandomForest",
        "criterion": hp.choice(
            "RandomForest_criterion", ["mse", "friedman_mse", "mae"]
        ),
        "max_features": hp.uniform("RandomForest_max_features", 0.1, 1.0),
        "max_depth": hp.choice("RandomForest_max_depth", [None]),
        "min_samples_split": scope.int(
            hp.quniform("RandomForest_min_samples_split", 2, 20, 1)
        ),
        "min_samples_leaf": scope.int(
            hp.quniform("RandomForest_min_samples_leaf", 1, 20, 1)
        ),
        "min_weight_fraction_leaf": hp.choice(
            "RandomForest_min_weight_fraction_leaf", [0.0]
        ),
        "bootstrap": hp.choice("RandomForest_bootstrap", [True, False]),
        "max_leaf_nodes": hp.choice("RandomForest_max_leaf_nodes", [None]),
        "min_impurity_decrease": hp.choice("RandomForest_min_impurity_decrease", [0.0]),
    },
    {
        "model": "SGD",
        # l1_ratio only selected for penalty = 'elasticnet'
        # epsilon only selected for loss in ['huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
        # eta0 only selected for learning_rate in ['constant', 'invscaling']
        # power_t only selected for learning_rate = 'invscaling'
        "loss": hp.choice(
            "SGD_loss",
            [
                "squared_loss",
                "huber",
                "epsilon_insensitive",
                "squared_epsilon_insensitive",
            ],
        ),
        "penalty": hp.choice("SGD_penalty", ["l1", "l2", "elasticnet"]),
        "alpha": hp.loguniform("SGD_alpha", np.log(1e-7), np.log(1e-1)),
        "fit_intercept": hp.choice("SGD_fit_intercept", [True]),
        "tol": hp.loguniform("SGD_tol", np.log(1e-5), np.log(1e-1)),
        "learning_rate": hp.choice(
            "SGD_learning_rate", ["constant", "optimal", "invscaling"]
        ),
        "l1_ratio": hp.loguniform("SGD_l1_ratio", np.log(1e-9), np.log(1.0)),
        "epsilon": hp.loguniform("SGD_epsilon", np.log(1e-5), np.log(1e-1)),
        "power_t": hp.uniform("SGD_power_t", 1e-5, 1),
        "average": hp.choice("SGD_average", [True, False]),
    },
    # self-defined models
    {
        "model": "MLP_Regressor",
        "hidden_layer": scope.int(hp.quniform("MLP_Regressor_hidden_layer", 1, 5, 1)),
        "hidden_size": scope.int(hp.quniform("MLP_Regressor_hidden_size", 1, 20, 1)),
        "activation": hp.choice(
            "MLP_Regressor_activation", ["ReLU", "Tanh", "Sigmoid"]
        ),
        "learning_rate": hp.uniform("MLP_Regressor_learning_rate", 1e-5, 1),
        "optimizer": hp.choice("MLP_Regressor_optimizer", ["Adam", "SGD"]),
        "criteria": hp.choice("MLP_Regressor_criteria", ["MSE", "MAE"]),
        "batch_size": hp.choice("MLP_Regressor_batch_size", [16, 32, 64]),
        "num_epochs": scope.int(hp.quniform("MLP_Regressor_num_epochs", 5, 50, 1)),
    },
]
