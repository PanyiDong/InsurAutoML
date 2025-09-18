"""
File: _classifier_hyperparameter.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.6
Relative Path: /InsurAutoML/hyperparameters/ray/classifier_hyperparameter.py
File: _classifier_hyperparameter.py
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Thursday, 18th September 2025 4:12:06 pm
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

from ...constant import LIGHTGBM_BOOSTING, LIGHTGBM_TREE_LEARNER
from ...utils.base import format_hyper_dict

ADABOOSTCLASSIFIER = {
    "model": "AdaboostClassifier",
    "n_estimators": tune.qlograndint(10, 500, 1),
    "learning_rate": tune.uniform(0.01, 2),
    "algorithm": tune.choice(["SAMME"]),
    # for base_estimator of Decision Tree
    "max_depth": tune.qrandint(1, 10, 1),
}
BERNOULLINB = {
    "model": "BernoulliNB",
    "alpha": tune.loguniform(1e-2, 100),
    "fit_prior": tune.choice([True, False]),
}
DECISIONTREE = {
    "model": "DecisionTree",
    "criterion": tune.choice(["gini", "entropy"]),
    "max_features": tune.choice([1.0]),
    "max_depth_factor": tune.uniform(0.0, 2.0),
    "min_samples_split": tune.qrandint(2, 20, 1),
    "min_samples_leaf": tune.qrandint(1, 20, 1),
    "min_weight_fraction_leaf": tune.choice([0.0]),
    "max_leaf_nodes": tune.choice(["None"]),
    "min_impurity_decrease": tune.choice([0.0]),
}
EXTRATREESCLASSIFIER = {
    "model": "ExtraTreesClassifier",
    "criterion": tune.choice(["gini", "entropy"]),
    "min_samples_leaf": tune.qrandint(1, 20, 1),
    "min_samples_split": tune.qrandint(2, 20, 1),
    "max_features": tune.uniform(0.0, 1.0),
    "bootstrap": tune.choice([True, False]),
    "max_leaf_nodes": tune.choice(["None"]),
    "max_depth": tune.choice(["None"]),
    "min_weight_fraction_leaf": tune.choice([0.0]),
    "min_impurity_decrease": tune.choice([0.0]),
}
GAUSSIANNB = {
    "model": "GaussianNB",
}
HISTGRADIENTBOOSTINGCLASSIFIER = {
    "model": "HistGradientBoostingClassifier",
    "loss": tune.choice(["log_loss"]),
    "learning_rate": tune.loguniform(0.01, 1),
    "min_samples_leaf": tune.qlograndint(1, 200, 1),
    "max_depth": tune.choice(["None"]),
    "max_leaf_nodes": tune.qlograndint(3, 2047, 1),
    "max_bins": tune.choice([255]),
    "l2_regularization": tune.loguniform(1e-10, 1),
    "early_stop": tune.choice(["off", "train", "valid"]),
    "tol": tune.choice([1e-7]),
    "scoring": tune.choice(["loss"]),
    "n_iter_no_change": tune.qrandint(1, 20, 1),
    "validation_fraction": tune.uniform(0.01, 0.4),
}
KNEARESTNEIGHBORSCLASSIFIER = {
    "model": "KNearestNeighborsClassifier",
    "n_neighbors": tune.qrandint(1, 100, 1),
    "weights": tune.choice(["uniform", "distance"]),
    "p": tune.choice([1, 2]),
}
LDA = {
    "model": "LDA",
    "shrinkage_type": tune.choice(["None", "auto", "manual"]),
    "tol": tune.loguniform(1e-5, 1e-1),
    "shrinkage_factor": tune.uniform(0.0, 1.0),
}
LIBLINEARSVC = {
    "model": "LibLinear_SVC",
    # forbid penalty = 'l1' and loss = 'hinge'
    # forbid penalty = 'l2', loss = 'hinge' and dual = False
    # forbid penalty = 'l1' and dual = False
    "penalty": tune.choice(["l2"]),
    "loss": tune.choice(["squared_hinge"]),
    "dual": tune.choice([False]),
    "tol": tune.loguniform(1e-5, 1e-1),
    "C": tune.loguniform(0.03125, 32768),
    "multi_class": tune.choice(["ovr"]),
    "fit_intercept": tune.choice([True]),
    "intercept_scaling": tune.choice([1]),
}
LIBSVMSVC = {
    "model": "LibSVM_SVC",
    # degree only selected when kernel = 'poly'
    # coef0 only selected when kernel = ['poly', 'sigmoid']
    "C": tune.loguniform(0.03125, 32768),
    "kernel": tune.choice(["poly", "rbf", "sigmoid"]),
    "gamma": tune.loguniform(3.0517578125e-05, 8),
    "shrinking": tune.choice([True, False]),
    "tol": tune.loguniform(1e-5, 1e-1),
    "max_iter": tune.choice([-1]),
    "degree": tune.qrandint(2, 5, 1),
    "coef0": tune.uniform(-1, 1),
}
MLPCLASSIFIER = {
    "model": "MLPClassifier",
    "hidden_layer_depth": tune.qrandint(1, 3, 1),
    "num_nodes_per_layer": tune.qlograndint(16, 264, 1),
    "activation": tune.choice(["tanh", "relu"]),
    "alpha": tune.loguniform(1e-7, 1e-1),
    "learning_rate_init": tune.loguniform(1e-4, 0.5),
    "early_stopping": tune.choice(["train", "valid"]),
    # 'solver' : tune.choice('solver', ['lbfgs', 'sgd', 'adam']),
    # autosklearn must include _no_improvement_count, where only supported by
    # 'sgd' and 'adam'
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
MULTINOMIALNB = {
    "model": "MultinomialNB",
    "alpha": tune.loguniform(1e-2, 100),
    "fit_prior": tune.choice([True, False]),
}
PASSIVEAGGRESSIVE = {
    "model": "PassiveAggressive",
    "C": tune.loguniform(1e-5, 10),
    "fit_intercept": tune.choice([True]),
    "tol": tune.loguniform(1e-5, 1e-1),
    "loss": tune.choice(["hinge", "squared_hinge"]),
    "average": tune.choice([True, False]),
}
QDA = {"model": "QDA", "reg_param": tune.uniform(0.0, 1.0)}
RANDOMFOREST = {
    "model": "RandomForest",
    "criterion": tune.choice(["gini", "entropy"]),
    "max_features": tune.uniform(0.0, 1.0),
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
    # epsilon only selected for loss = 'modified_huber'
    # power_t only selected for learning_rate = 'invscaling'
    # eta0 only selected for learning_rate in ['constant', 'invscaling']
    "loss": tune.choice(
        ["hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron"],
    ),
    "penalty": tune.choice(["l1", "l2", "elasticnet"]),
    "alpha": tune.loguniform(1e-7, 1e-1),
    "fit_intercept": tune.choice([True]),
    "tol": tune.loguniform(1e-5, 1e-1),
    "learning_rate": tune.choice(["constant", "optimal", "invscaling"]),
    "l1_ratio": tune.loguniform(1e-9, 1),
    "epsilon": tune.loguniform(1e-5, 1e-1),
    "eta0": tune.loguniform(1e-7, 1e-1),
    "power_t": tune.uniform(1e-5, 1),
    "average": tune.choice([True, False]),
}
LOGISTICREGRESSION = {
    "model": "LogisticRegression",
    "penalty": tune.choice(["l2", None]),
    "tol": tune.loguniform(1e-5, 1e-1),
    "C": tune.loguniform(1e-5, 10),
}
COMPLEMENTNB = {
    "model": "ComplementNB",
    "alpha": tune.uniform(0, 1),
    "fit_prior": tune.choice([True, False]),
    "norm": tune.choice([True, False]),
}
# HISTGRADIENTBOOSTINGCLASSIFIER = {
#     "model_19": "HistGradientBoostingClassifier",
#     "loss": tune.choice(["auto"]),
#     "learning_rate": tune.uniform(1e-7, 1),
#     "max_leaf_nodes": tune.choice([None]),
#     "max_depth": tune.choice([None]),
#     "min_samples_leaf": tune.qrandint(1, 20, 1),
#     "l2_regularization": tune.uniform(0, 1),
#     "tol": tune.loguniform(1e-5, 1e-1),
# }
GRADIENTBOOSTINGCLASSIFIER = {
    "model": "GradientBoostingClassifier",
    "loss": tune.choice(["log_loss", "exponential"]),
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
MLP_CLASSIFIER = {
    "model": "MLP_Classifier",
    "hidden_layer": tune.qrandint(1, 5, 1),
    "hidden_size": tune.qrandint(1, 10, 1),
    "activation": tune.choice(["Tanh", "Sigmoid"]),
    "learning_rate": tune.uniform(1e-5, 1),
    "optimizer": tune.choice(["Adam", "SGD"]),
    "criteria": tune.choice(["CrossEntropy", "NegativeLogLikelihood"]),
    "batch_size": tune.choice([16, 32, 64]),
    "num_epochs": tune.qrandint(5, 30, 1),
}
RNN_CLASSIFIER = {
    "model": "RNN_Classifier",
    "hidden_size": tune.choice([16, 32, 64, 128, 256]),
    "n_layers": tune.qrandint(1, 5, 1),
    "RNN_unit": tune.choice(["RNN", "LSTM", "GRU"]),
    "activation": tune.choice(["ReLU", "Tanh", "Sigmoid"]),
    "dropout": tune.loguniform(1e-7, 0.8),
    "learning_rate": tune.loguniform(1e-7, 1),
    "optimizer": tune.choice(["Adam", "SGD"]),
    "criteria": tune.choice(["CrossEntropy", "NegativeLogLikelihood"]),
    "batch_size": tune.choice([16, 32, 64]),
    "num_epochs": tune.qrandint(5, 30, 1),
}
LIGHTGBMCLASSIFIER = {
    "model": "LightGBM_Classifier",
    "objective": tune.choice(["Need to specify in HPO by response"]),
    "boosting": tune.choice(LIGHTGBM_BOOSTING),
    # same as num_iterations
    # "n_estimators": tune.qlograndint(50, 500, 1),
    # max_depth == -1 for no limit
    "max_depth": tune.randint(-1, 31),
    "num_leaves": tune.qlograndint(3, 2047, 1),
    "min_data_in_leaf": tune.qrandint(1, 20, 1),
    "learning_rate": tune.loguniform(1e-4, 1),
    "tree_learner": tune.choice(LIGHTGBM_TREE_LEARNER),
    "num_iterations": tune.qlograndint(50, 500, 1),
    "min_gain_to_split": tune.loguniform(1e-8, 1),
    "early_stopping_round": tune.randint(1, 200),
    "max_bin": tune.qlograndint(3, 1024, 1),
    "feature_fraction": tune.uniform(0.1, 1),
}
XGBOOSTCLASSIFIER = {
    "model": "XGBoost_Classifier",
    "eta": tune.uniform(0, 1),
    "gamma": tune.loguniform(1e-4, 1e3),
    "max_depth": tune.randint(1, 12),
    "min_child_weight": tune.loguniform(1e-4, 1e3),
    "max_delta_step": tune.loguniform(1e-3, 1e1),
    "reg_lambda": tune.uniform(0, 1),
    "reg_alpha": tune.uniform(0, 1),
}
GAMCLASSIFIER = {
    "model": "GAM_Classifier",
    "type": tune.choice(["logistic"]),
    "tol": tune.loguniform(1e-4, 1),
}

# classifier hyperparameters
classifier_hyperparameter = [
    # classification models from sklearn
    ADABOOSTCLASSIFIER,
    BERNOULLINB,
    DECISIONTREE,
    EXTRATREESCLASSIFIER,
    GAUSSIANNB,
    HISTGRADIENTBOOSTINGCLASSIFIER,
    KNEARESTNEIGHBORSCLASSIFIER,
    LDA,
    LIBLINEARSVC,
    LIBSVMSVC,
    MLPCLASSIFIER,
    MULTINOMIALNB,
    PASSIVEAGGRESSIVE,
    QDA,
    RANDOMFOREST,
    SGD,
    # classification models from sklearn
    LOGISTICREGRESSION,
    COMPLEMENTNB,
    # HISTGRADIENTBOOSTINGCLASSIFIER,
    GRADIENTBOOSTINGCLASSIFIER,
    # self-defined models
    MLP_CLASSIFIER,
    RNN_CLASSIFIER,
    LIGHTGBMCLASSIFIER,
    XGBOOSTCLASSIFIER,
    GAMCLASSIFIER,
]

# deprecated, add custom hyperparameter construction by search algorithm in AutoTabularBase class
# classifier_hyperparameter = [
#     format_hyper_dict(dict, order + 1, ref = "model")
#     for order, dict in enumerate(classifier_hyperparameter)
# ]

if __name__ == "__main__":
    pass
