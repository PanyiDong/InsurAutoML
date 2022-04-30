"""
File: _regressor_hyperparameter.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Latest Version: 0.2.0
Relative Path: /My_AutoML/_hyperparameters/_ray/_regressor_hyperparameter.py
File Created: Friday, 8th April 2022 9:04:05 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Friday, 29th April 2022 10:38:01 pm
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

# NOTE:
# As sklearn enters version 1.0, some of the losses have changed its name,
# hyperparameters will change accordingly
import sklearn

sklearn_1_0_0 = sklearn.__version__ <= "1.0.0"

from ray import tune

from My_AutoML._constant import (
    LIGHTGBM_REGRESSION,
    LIGHTGBM_BOOSTING,
    LIGHTGBM_TREE_LEARNER,
)

# regressor hyperparameters
regressor_hyperparameter = [
    # extract from autosklearn
    {
        "model_1": "AdaboostRegressor",
        "AdaboostRegressor_n_estimators": tune.qrandint(50, 500, 1),
        "AdaboostRegressor_learning_rate": tune.loguniform(0.01, 2),
        "AdaboostRegressor_loss": tune.choice(["linear", "square", "exponential"]),
        # for base_estimator of Decision Tree
        "AdaboostRegressor_max_depth": tune.qrandint(1, 10, 1),
    },
    {
        "model_2": "ARDRegression",
        "ARDRegression_n_iter": tune.choice([300]),
        "ARDRegression_tol": tune.loguniform(1e-5, 1e-1),
        "ARDRegression_alpha_1": tune.loguniform(1e-10, 1e-3),
        "ARDRegression_alpha_2": tune.loguniform(1e-10, 1e-3),
        "ARDRegression_lambda_1": tune.loguniform(1e-10, 1e-3),
        "ARDRegression_lambda_2": tune.loguniform(1e-10, 1e-3),
        "ARDRegression_threshold_lambda": tune.loguniform(1e3, 1e5),
        "ARDRegression_fit_intercept": tune.choice([True]),
    },
    {
        "model_3": "DecisionTree",
        "DecisionTree_criterion": tune.choice(["mse", "friedman_mse", "mae"])
        if sklearn_1_0_0
        else tune.choice(
            ["squared_error", "friedman_mse", "absolute_error", "poisson"]
        ),
        "DecisionTree_max_features": tune.choice([1.0]),
        "DecisionTree_max_depth_factor": tune.uniform(0.0, 2.0),
        "DecisionTree_min_samples_split": tune.qrandint(2, 20, 1),
        "DecisionTree_min_samples_leaf": tune.qrandint(1, 20, 1),
        "DecisionTree_min_weight_fraction_leaf": tune.choice([0.0]),
        "DecisionTree_max_leaf_nodes": tune.choice([None]),
        "DecisionTree_min_impurity_decrease": tune.choice([0.0]),
    },
    {
        "model_4": "ExtraTreesRegressor",
        "ExtraTreesRegressor_criterion": tune.choice(["mse", "friedman_mse", "mae"])
        if sklearn_1_0_0
        else tune.choice(["squared_error", "absolute_error"]),
        "ExtraTreesRegressor_min_samples_leaf": tune.qrandint(1, 20, 1),
        "ExtraTreesRegressor_min_samples_split": tune.qrandint(2, 20, 1),
        "ExtraTreesRegressor_max_features": tune.uniform(0.1, 1.0),
        "ExtraTreesRegressor_bootstrap": tune.choice([True, False]),
        "ExtraTreesRegressor_max_leaf_nodes": tune.choice([None]),
        "ExtraTreesRegressor_max_depth": tune.choice([None]),
        "ExtraTreesRegressor_min_weight_fraction_leaf": tune.choice([0.0]),
        "ExtraTreesRegressor_min_impurity_decrease": tune.choice([0.0]),
    },
    {
        "model_5": "GaussianProcess",
        "GaussianProcess_alpha": tune.loguniform(1e-14, 1),
        "GaussianProcess_thetaL": tune.loguniform(1e-10, 1e-3),
        "GaussianProcess_thetaU": tune.loguniform(1, 1e5),
    },
    {
        "model_6": "HistGradientBoostingRegressor",
        # n_iter_no_change only selected for early_stop in ['valid', 'train']
        # validation_fraction only selected for early_stop = 'valid'
        # "GradientBoosting_loss": tune.choice(["squared_error"]),
        "HistGradientBoostingRegressor_loss": tune.choice(["least_squares"]),
        "HistGradientBoostingRegressor_learning_rate": tune.loguniform(0.01, 1),
        "HistGradientBoostingRegressor_min_samples_leaf": tune.qlograndint(1, 200, 1),
        "HistGradientBoostingRegressor_max_depth": tune.choice([None]),
        "HistGradientBoostingRegressor_max_leaf_nodes": tune.qlograndint(3, 2047, 1),
        "HistGradientBoostingRegressor_max_bins": tune.choice([255]),
        "HistGradientBoostingRegressor_l2_regularization": tune.loguniform(1e-10, 1),
        "HistGradientBoostingRegressor_early_stop": tune.choice(
            ["off", "train", "valid"]
        ),
        "HistGradientBoostingRegressor_tol": tune.choice([1e-7]),
        "HistGradientBoostingRegressor_scoring": tune.choice(["loss"]),
        "HistGradientBoostingRegressor_n_iter_no_change": tune.qrandint(1, 20, 1),
        "HistGradientBoostingRegressor_validation_fraction": tune.uniform(0.01, 0.4),
    },
    {
        "model_7": "KNearestNeighborsRegressor",
        "KNearestNeighborsRegressor_n_neighbors": tune.qrandint(1, 100, 1),
        "KNearestNeighborsRegressor_weights": tune.choice(["uniform", "distance"]),
        "KNearestNeighborsRegressor_p": tune.choice([1, 2]),
    },
    {
        "model_8": "LibLinear_SVR",
        # forbid loss = 'epsilon_insensitive' and dual = False
        "LibLinear_SVR_epsilon": tune.loguniform(0.001, 1),
        "LibLinear_SVR_loss": tune.choice(
            ["squared_epsilon_insensitive"],
        ),
        "LibLinear_SVR_dual": tune.choice([False]),
        "LibLinear_SVR_tol": tune.loguniform(1e-5, 1e-1),
        "LibLinear_SVR_C": tune.loguniform(0.03125, 32768),
        "LibLinear_SVR_fit_intercept": tune.choice([True]),
        "LibLinear_SVR_intercept_scaling": tune.choice([1]),
    },
    {
        "model_9": "LibSVM_SVR",
        # degree only selected for kernel in ['poly', 'rbf', 'sigmoid']
        # gamma only selected for kernel in ['poly', 'rbf']
        # coef0 only selected for kernel in ['poly', 'sigmoid']
        "LibSVM_SVR_kernel": tune.choice(["linear", "poly", "rbf", "sigmoid"]),
        "LibSVM_SVR_C": tune.loguniform(0.03125, 32768),
        "LibSVM_SVR_epsilon": tune.uniform(1e-5, 1),
        "LibSVM_SVR_tol": tune.loguniform(1e-5, 1e-1),
        "LibSVM_SVR_shrinking": tune.choice([True, False]),
        "LibSVM_SVR_degree": tune.qrandint(2, 5, 1),
        "LibSVM_SVR_gamma": tune.loguniform(3.0517578125e-5, 8),
        "LibSVM_SVR_coef0": tune.uniform(-1, 1),
        "LibSVM_SVR_max_iter": tune.choice([-1]),
    },
    {
        "model_10": "MLPRegressor",
        # validation_fraction only selected for early_stopping = 'valid'
        "MLPRegressor_hidden_layer_depth": tune.qrandint(1, 3, 1),
        "MLPRegressor_num_nodes_per_layer": tune.qlograndint(16, 264, 1),
        "MLPRegressor_activation": tune.choice(["tanh", "relu"]),
        "MLPRegressor_alpha": tune.loguniform(1e-7, 1e-1),
        "MLPRegressor_learning_rate_init": tune.loguniform(1e-4, 0.5),
        "MLPRegressor_early_stopping": tune.choice(["valid", "train"]),
        "MLPRegressor_solver": tune.choice(["adam"]),
        "MLPRegressor_batch_size": tune.choice(["auto"]),
        "MLPRegressor_n_iter_no_change": tune.choice([32]),
        "MLPRegressor_tol": tune.choice([1e-4]),
        "MLPRegressor_shuffle": tune.choice([True]),
        "MLPRegressor_beta_1": tune.choice([0.9]),
        "MLPRegressor_beta_2": tune.choice([0.999]),
        "MLPRegressor_epsilon": tune.choice([1e-8]),
        "MLPRegressor_validation_fraction": tune.choice([0.1]),
    },
    {
        "model_11": "RandomForest",
        "RandomForest_criterion": tune.choice(["mse", "friedman_mse", "mae"])
        if sklearn_1_0_0
        else tune.choice(["squared_error", "absolute_error", "poisson"]),
        "RandomForest_max_features": tune.uniform(0.1, 1.0),
        "RandomForest_max_depth": tune.choice([None]),
        "RandomForest_min_samples_split": tune.qrandint(2, 20, 1),
        "RandomForest_min_samples_leaf": tune.qrandint(1, 20, 1),
        "RandomForest_min_weight_fraction_leaf": tune.choice([0.0]),
        "RandomForest_bootstrap": tune.choice([True, False]),
        "RandomForest_max_leaf_nodes": tune.choice([None]),
        "RandomForest_min_impurity_decrease": tune.choice([0.0]),
    },
    {
        "model_12": "SGD",
        # l1_ratio only selected for penalty = 'elasticnet'
        # epsilon only selected for loss in ['huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
        # eta0 only selected for learning_rate in ['constant', 'invscaling']
        # power_t only selected for learning_rate = 'invscaling'
        "SGD_loss": tune.choice(
            [
                "squared_loss",
                "huber",
                "epsilon_insensitive",
                "squared_epsilon_insensitive",
            ],
        )
        if sklearn_1_0_0
        else tune.choice(
            [
                "squared_error",
                "huber",
                "epsilon_insensitive",
                "squared_epsilon_insensitive",
            ]
        ),
        "SGD_penalty": tune.choice(["l1", "l2", "elasticnet"]),
        "SGD_alpha": tune.loguniform(1e-7, 1e-1),
        "SGD_fit_intercept": tune.choice([True]),
        "SGD_tol": tune.loguniform(1e-5, 1e-1),
        "SGD_learning_rate": tune.choice(["constant", "optimal", "invscaling"]),
        "SGD_l1_ratio": tune.loguniform(1e-9, 1.0),
        "SGD_epsilon": tune.loguniform(1e-5, 1e-1),
        "SGD_power_t": tune.uniform(1e-5, 1),
        "SGD_average": tune.choice([True, False]),
    },
    {
        "model_13": "LinearRegression",
    },
    {
        "model_14": "Lasso",
        "Lasso_alpha": tune.loguniform(1e-7, 1e3),
        "Lasso_tol": tune.loguniform(1e-5, 1e-1),
    },
    {
        "model_15": "RidgeRegression",
        "RidgeRegression_alpha": tune.loguniform(1e-7, 1e3),
        "RidgeRegression_tol": tune.loguniform(1e-5, 1e-1),
        "RidgeRegression_solver": tune.choice(
            ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
        ),
    },
    {
        "model_16": "ElasticNet",
        "ElasticNet_alpha": tune.loguniform(1e-7, 1e3),
        "ElasticNet_l1_ratio": tune.uniform(0, 1.0),
        "ElasticNet_tol": tune.loguniform(1e-5, 1e-1),
        "ElasticNet_selection": tune.choice(["cyclic", "random"]),
    },
    {
        "model_17": "BayesianRidge",
        "BayesianRidge_tol": tune.loguniform(1e-5, 1e-1),
        "BayesianRidge_alpha_1": tune.loguniform(1e-7, 1e-1),
        "BayesianRidge_alpha_2": tune.loguniform(1e-7, 1e-1),
        "BayesianRidge_lambda_1": tune.loguniform(1e-7, 1e-1),
        "BayesianRidge_lambda_2": tune.loguniform(1e-7, 1e-1),
    },
    # {
    #     "model_18": "HistGradientBoostingRegressor",
    #     # "HistGradientBoostingRegressor_loss": tune.choice(
    #     #     ["squared_error", "absolute_error", "poisson"]
    #     # ),
    #     "HistGradientBoostingRegressor_loss": tune.choice(
    #         ["least_squares", "least_absolute_deviation", "poisson"]
    #     ),
    #     "HistGradientBoostingRegressor_learning_rate": tune.loguniform(1e-7, 1e-1),
    #     "HistGradientBoostingRegressor_max_leaf_nodes": tune.choice([None]),
    #     "HistGradientBoostingRegressor_max_depth": tune.choice([None]),
    #     "HistGradientBoostingRegressor_min_samples_leaf": tune.qrandint(1, 20, 1),
    #     "HistGradientBoostingRegressor_l2_regularization": tune.uniform(0, 1),
    #     "HistGradientBoostingRegressor_tol": tune.loguniform(1e-5, 1e-1),
    # },
    # self-defined models
    {
        "model_19": "MLP_Regressor",
        "MLP_Regressor_hidden_layer": tune.qrandint(1, 5, 1),
        "MLP_Regressor_hidden_size": tune.qrandint(1, 10, 1),
        "MLP_Regressor_activation": tune.choice(["ReLU"]),
        "MLP_Regressor_learning_rate": tune.uniform(1e-5, 1),
        "MLP_Regressor_optimizer": tune.choice(["Adam", "SGD"]),
        "MLP_Regressor_criteria": tune.choice(["MSE", "MAE"]),
        "MLP_Regressor_batch_size": tune.choice([16, 32, 64]),
        "MLP_Regressor_num_epochs": tune.qrandint(5, 30, 1),
    },
    {
        "model_20": "RNN_Regressor",
        "RNN_Regressor_hidden_size": tune.choice([16, 32, 64, 128, 256]),
        "RNN_Regressor_n_layers": tune.qrandint(1, 5, 1),
        "RNN_Regressor_RNN_unit": tune.choice(["RNN", "LSTM", "GRU"]),
        "RNN_Regressor_activation": tune.choice(["ReLU"]),
        "RNN_Regressor_dropout": tune.loguniform(1e-7, 0.8),
        "RNN_Regressor_learning_rate": tune.loguniform(1e-7, 1),
        "RNN_Regressor_optimizer": tune.choice(["Adam", "SGD"]),
        "RNN_Regressor_criteria": tune.choice(["MSE", "MAE"]),
        "RNN_Regressor_batch_size": tune.choice([16, 32, 64]),
        "RNN_Regressor_num_epochs": tune.qrandint(5, 30, 1),
    },
    {
        "model_21": "LightGBM_Regressor",
        "LightGBM_Regressor_objective": tune.choice(LIGHTGBM_REGRESSION),
        "LightGBM_Regressor_boosting": tune.choice(LIGHTGBM_BOOSTING),
        "LightGBM_Regressor_n_estimators": tune.qlograndint(50, 500, 1),
        # max_depth == -1 for no limit
        "LightGBM_Regressor_max_depth": tune.randint(-1, 31),
        "LightGBM_Regressor_num_leaves": tune.qlograndint(3, 2047, 1),
        "LightGBM_Regressor_min_data_in_leaf": tune.qrandint(1, 20, 1),
        "LightGBM_Regressor_learning_rate": tune.loguniform(1e-7, 1),
        "LightGBM_Regressor_tree_learner": tune.choice(LIGHTGBM_TREE_LEARNER),
        "LightGBM_Regressor_num_iterations": tune.qlograndint(50, 500, 1),
    },
    {
        "model_22": "XGBoost_Regressor",
        "XGBoost_Regressor_eta": tune.uniform(0, 1),
        "XGBoost_Regressor_gamma": tune.loguniform(1e-4, 1e3),
        "XGBoost_Regressor_max_depth": tune.randint(1, 12),
        "XGBoost_Regressor_min_child_weight": tune.loguniform(1e-4, 1e3),
        "XGBoost_Regressor_max_delta_step": tune.loguniform(1e-3, 1e1),
        "XGBoost_Regressor_reg_lambda": tune.uniform(0, 1),
        "XGBoost_Regressor_reg_alpha": tune.uniform(0, 1),
    },
    {
        "model_23": "GAM_Regressor",
        "GAM_Regressor_type": tune.choice(
            ["linear", "gamma", "poisson", "inverse_gaussian"]
        ),
        "GAM_Regressor_tol": tune.loguniform(1e-4, 1),
    },
]
