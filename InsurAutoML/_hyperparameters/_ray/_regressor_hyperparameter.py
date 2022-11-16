"""
File: _regressor_hyperparameter.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_hyperparameters/_ray/_regressor_hyperparameter.py
File: _regressor_hyperparameter.py
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Tuesday, 15th November 2022 8:57:57 pm
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

# Update: Nov. 15, 2022
# sklearn > 1.0.0 is required for this module.
# NOTE:
# As sklearn enters version 1.0, some of the losses have changed its name,
# hyperparameters will change accordingly
# import sklearn

# sklearn_1_0_0 = sklearn.__version__ <= "1.0.0"

from ray import tune

from InsurAutoML._constant import (
    LIGHTGBM_REGRESSION,
    LIGHTGBM_BOOSTING,
    LIGHTGBM_TREE_LEARNER,
)
from InsurAutoML._utils._base import format_hyper_dict

ADABOOSTREGRESSOR = {
    "model": "AdaboostRegressor",
    "n_estimators": tune.qrandint(50, 500, 1),
    "learning_rate": tune.loguniform(0.01, 2),
    "loss": tune.choice(["linear", "square", "exponential"]),
    # for base_estimator of Decision Tree
    "max_depth": tune.qrandint(1, 10, 1),
}
ARDREGRESSION = {
    "model": "ARDRegression",
    "n_iter": tune.choice([300]),
    "tol": tune.loguniform(1e-5, 1e-1),
    "alpha_1": tune.loguniform(1e-10, 1e-3),
    "alpha_2": tune.loguniform(1e-10, 1e-3),
    "lambda_1": tune.loguniform(1e-10, 1e-3),
    "lambda_2": tune.loguniform(1e-10, 1e-3),
    "threshold_lambda": tune.loguniform(1e3, 1e5),
    "fit_intercept": tune.choice([True]),
}
DECISIONTREE = {
    "model": "DecisionTree",
    "criterion": tune.choice(
        ["squared_error", "friedman_mse", "absolute_error", "poisson"]
    ),
    "max_features": tune.choice([1.0]),
    "max_depth_factor": tune.uniform(0.0, 2.0),
    "min_samples_split": tune.qrandint(2, 20, 1),
    "min_samples_leaf": tune.qrandint(1, 20, 1),
    "min_weight_fraction_leaf": tune.choice([0.0]),
    "max_leaf_nodes": tune.choice([None]),
    "min_impurity_decrease": tune.choice([0.0]),
}
EXTRATREESREGRESSOR = {
    "model": "ExtraTreesRegressor",
    "criterion": tune.choice(["squared_error", "absolute_error"]),
    "min_samples_leaf": tune.qrandint(1, 20, 1),
    "min_samples_split": tune.qrandint(2, 20, 1),
    "max_features": tune.uniform(0.0, 1.0),
    "bootstrap": tune.choice([True, False]),
    "max_leaf_nodes": tune.choice([None]),
    "max_depth": tune.choice([None]),
    "min_weight_fraction_leaf": tune.choice([0.0]),
    "min_impurity_decrease": tune.choice([0.0]),
}
GAUSSIANPROCESS = {
    "model": "GaussianProcess",
    "alpha": tune.loguniform(1e-14, 1),
    "thetaL": tune.loguniform(1e-10, 1e-3),
    "thetaU": tune.loguniform(1, 1e5),
}
HISTGRADIENTBOOSTINGREGRESSOR = {
    "model": "HistGradientBoostingRegressor",
    # n_iter_no_change only selected for early_stop in ['valid', 'train']
    # validation_fraction only selected for early_stop = 'valid'
    "loss": tune.choice(["squared_error"]),
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
}
KNEARESTNEIGHBORSREGRESSOR = {
    "model": "KNearestNeighborsRegressor",
    "n_neighbors": tune.qrandint(1, 100, 1),
    "weights": tune.choice(["uniform", "distance"]),
    "p": tune.choice([1, 2]),
}
LIBLINEARSVR = {
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
}
LIBSVMSVR = {
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
}
MLPREGRESSOR = {
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
}
RANDOMFOREST = {
    "model": "RandomForest",
    "criterion": tune.choice(["squared_error", "absolute_error", "poisson"]),
    "max_features": tune.uniform(0.1, 1.0),
    "max_depth": tune.choice([None]),
    "min_samples_split": tune.qrandint(2, 20, 1),
    "min_samples_leaf": tune.qrandint(1, 20, 1),
    "min_weight_fraction_leaf": tune.choice([0.0]),
    "bootstrap": tune.choice([True, False]),
    "max_leaf_nodes": tune.choice([None]),
    "min_impurity_decrease": tune.choice([0.0]),
}
SGD = {
    "model": "SGD",
    # l1_ratio only selected for penalty = 'elasticnet'
    # epsilon only selected for loss in ['huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
    # eta0 only selected for learning_rate in ['constant', 'invscaling']
    # power_t only selected for learning_rate = 'invscaling'
    "loss": tune.choice(
        [
            "squared_error",
            "huber",
            "epsilon_insensitive",
            "squared_epsilon_insensitive",
        ]
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
}
LINEARREGRESSION = {
    "model": "LinearRegression",
}
LASSO = {
    "model": "Lasso",
    "alpha": tune.loguniform(1e-7, 1e3),
    "tol": tune.loguniform(1e-5, 1e-1),
}
RIDGEREGRESSION = {
    "model": "RidgeRegression",
    "alpha": tune.loguniform(1e-7, 1e3),
    "tol": tune.loguniform(1e-5, 1e-1),
    "solver": tune.choice(
        ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
    ),
}
ELASTICNET = {
    "model": "ElasticNet",
    "alpha": tune.loguniform(1e-7, 1e3),
    "l1_ratio": tune.uniform(0, 1.0),
    "tol": tune.loguniform(1e-5, 1e-1),
    "selection": tune.choice(["cyclic", "random"]),
}
BAYESIANRIDGE = {
    "model": "BayesianRidge",
    "tol": tune.loguniform(1e-5, 1e-1),
    "alpha_1": tune.loguniform(1e-7, 1e-1),
    "alpha_2": tune.loguniform(1e-7, 1e-1),
    "lambda_1": tune.loguniform(1e-7, 1e-1),
    "lambda_2": tune.loguniform(1e-7, 1e-1),
}
# HISTGRADIENTBOOSTINGREGRESSOR = {
#     "model": "HistGradientBoostingRegressor",
#     # "loss": tune.choice(
#     #     ["squared_error", "absolute_error", "poisson"]
#     # ),
#     "loss": tune.choice(
#         ["least_squares", "least_absolute_deviation", "poisson"]
#     ),
#     "learning_rate": tune.loguniform(1e-7, 1e-1),
#     "max_leaf_nodes": tune.choice([None]),
#     "max_depth": tune.choice([None]),
#     "min_samples_leaf": tune.qrandint(1, 20, 1),
#     "l2_regularization": tune.uniform(0, 1),
#     "tol": tune.loguniform(1e-5, 1e-1),
# }
GRADIENTBOOSTINGREGRESSOR = {
    "model": "GradientBoostingRegressor",
    "loss": tune.choice(["squared_error", "absolute_error", "huber", "quantile"]),
    "learning_rate": tune.loguniform(0.01, 1),
    "n_estimators": tune.qlograndint(10, 500, 1),
    "subsample": tune.uniform(0.1, 1),
    "criterion": tune.choice(["friedman_mse", "squared_error"]),
    "min_samples_split": tune.qrandint(2, 20, 1),
    "min_samples_leaf": tune.qlograndint(1, 200, 1),
    "min_weight_fraction_leaf": tune.uniform(0.0, 0.5),
    "max_depth": tune.randint(1, 31),
    "min_impurity_decrease": tune.uniform(0.0, 1.0),
    # "max_features": tune.choice(["sqrt", "log2", tune.uniform(0.0, 1.0)]),
    "max_features": tune.uniform(0.0, 1.0),
    "max_leaf_nodes": tune.qlograndint(3, 2047, 1),
    "validation_fraction": tune.uniform(0.01, 0.4),
    "n_iter_no_change": tune.qrandint(1, 20, 1),
    "tol": tune.choice([1e-7]),
}
MLP_REGRESSOR = {
    "model": "MLP_Regressor",
    "hidden_layer": tune.qrandint(1, 5, 1),
    "hidden_size": tune.qrandint(1, 10, 1),
    "activation": tune.choice(["ReLU"]),
    "learning_rate": tune.uniform(1e-5, 1),
    "optimizer": tune.choice(["Adam", "SGD"]),
    "criteria": tune.choice(["MSE", "MAE"]),
    "batch_size": tune.choice([16, 32, 64]),
    "num_epochs": tune.qrandint(5, 30, 1),
}
RNN_REGRESSOR = {
    "model": "RNN_Regressor",
    "hidden_size": tune.choice([16, 32, 64, 128, 256]),
    "n_layers": tune.qrandint(1, 5, 1),
    "RNN_unit": tune.choice(["RNN", "LSTM", "GRU"]),
    "activation": tune.choice(["ReLU"]),
    "dropout": tune.loguniform(1e-7, 0.8),
    "learning_rate": tune.loguniform(1e-7, 1),
    "optimizer": tune.choice(["Adam", "SGD"]),
    "criteria": tune.choice(["MSE", "MAE"]),
    "batch_size": tune.choice([16, 32, 64]),
    "num_epochs": tune.qrandint(5, 30, 1),
}
LIGHTGBMREGRESSOR = {
    "model": "LightGBM_Regressor",
    "objective": tune.choice(LIGHTGBM_REGRESSION),
    "boosting": tune.choice(LIGHTGBM_BOOSTING),
    "n_estimators": tune.qlograndint(50, 500, 1),
    # max_depth == -1 for no limit
    "max_depth": tune.randint(-1, 31),
    "num_leaves": tune.qlograndint(3, 2047, 1),
    "min_data_in_leaf": tune.qrandint(1, 20, 1),
    "learning_rate": tune.loguniform(1e-7, 1),
    "tree_learner": tune.choice(LIGHTGBM_TREE_LEARNER),
    "num_iterations": tune.qlograndint(50, 500, 1),
}
XGBOOSTREGRESSOR = {
    "model": "XGBoost_Regressor",
    "eta": tune.uniform(0, 1),
    "gamma": tune.loguniform(1e-4, 1e3),
    "max_depth": tune.randint(1, 12),
    "min_child_weight": tune.loguniform(1e-4, 1e3),
    "max_delta_step": tune.loguniform(1e-3, 1e1),
    "reg_lambda": tune.uniform(0, 1),
    "reg_alpha": tune.uniform(0, 1),
}
GAMREGRESSOR = {
    "model": "GAM_Regressor",
    "type": tune.choice(["linear", "gamma", "poisson", "inverse_gaussian"]),
    "tol": tune.loguniform(1e-4, 1),
}

# regressor hyperparameters
regressor_hyperparameter = [
    # extract from sklearn
    ADABOOSTREGRESSOR,
    ARDREGRESSION,
    DECISIONTREE,
    EXTRATREESREGRESSOR,
    GAUSSIANPROCESS,
    HISTGRADIENTBOOSTINGREGRESSOR,
    KNEARESTNEIGHBORSREGRESSOR,
    LIBLINEARSVR,
    LIBSVMSVR,
    MLPREGRESSOR,
    RANDOMFOREST,
    SGD,
    LINEARREGRESSION,
    LASSO,
    RIDGEREGRESSION,
    ELASTICNET,
    BAYESIANRIDGE,
    # HISTGRADIENTBOOSTINGREGRESSOR,
    GRADIENTBOOSTINGREGRESSOR,
    # self-defined models
    MLP_REGRESSOR,
    RNN_REGRESSOR,
    LIGHTGBMREGRESSOR,
    XGBOOSTREGRESSOR,
    GAMREGRESSOR,
]

# deprecated, add custom hyperparameter construction by search algorithm in AutoTabularBase class
# regressor_hyperparameter = [
#     format_hyper_dict(dict, order + 1, ref = "model")
#     for order, dict in enumerate(regressor_hyperparameter)
# ]

if __name__ == "__main__":
    pass
