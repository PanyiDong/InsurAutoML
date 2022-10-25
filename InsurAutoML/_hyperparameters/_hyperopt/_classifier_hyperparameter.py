"""
File: _classifier_hyperparameter.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Latest Version: 0.2.0
Relative Path: /My_AutoML/_hyperparameters/_hyperopt/_classifier_hyperparameter.py
File Created: Tuesday, 5th April 2022 11:05:31 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Friday, 8th April 2022 10:22:42 pm
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

# classifier hyperparameters
classifier_hyperparameter = [
    # extract from autosklearn
    {
        "model": "AdaboostClassifier",
        "n_estimators": scope.int(
            hp.quniform("AdaboostClassifier_n_estimators", 10, 500, 1)
        ),
        "learning_rate": hp.uniform("AdaboostClassifier_learning_rate", 0.01, 2),
        "algorithm": hp.choice("AdaboostClassifier_algorithm", ["SAMME", "SAMME.R"]),
        # for base_estimator of Decision Tree
        "max_depth": scope.int(hp.quniform("AdaboostClassifier_max_depth", 1, 10, 1)),
    },
    {
        "model": "BernoulliNB",
        "alpha": hp.loguniform("BernoulliNB_alpha", np.log(1e-2), np.log(100)),
        "fit_prior": hp.choice("BernoulliNB_fit_prior", [True, False]),
    },
    {
        "model": "DecisionTree",
        "criterion": hp.choice("DecisionTree_criterion", ["gini", "entropy"]),
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
        "max_leaf_nodes": hp.choice("DecisionTree_max_leaf_nodes", ["None"]),
        "min_impurity_decrease": hp.choice("DecisionTree_min_impurity_decrease", [0.0]),
    },
    {
        "model": "ExtraTreesClassifier",
        "criterion": hp.choice("ExtraTreesClassifier_criterion", ["gini", "entropy"]),
        "min_samples_leaf": scope.int(
            hp.quniform("ExtraTreesClassifier_min_samples_leaf", 1, 20, 1)
        ),
        "min_samples_split": scope.int(
            hp.quniform("ExtraTreesClassifier_min_samples_split", 2, 20, 1)
        ),
        "max_features": hp.uniform("ExtraTreesClassifier_max_features", 0.0, 1.0),
        "bootstrap": hp.choice("ExtraTreesClassifier_bootstrap", [True, False]),
        "max_leaf_nodes": hp.choice("ExtraTreesClassifier_max_leaf_nodes", ["None"]),
        "max_depth": hp.choice("ExtraTreesClassifier_max_depth", ["None"]),
        "min_weight_fraction_leaf": hp.choice(
            "ExtraTreesClassifier_min_weight_fraction_leaf", [0.0]
        ),
        "min_impurity_decrease": hp.choice(
            "ExtraTreesClassifier_min_impurity_decrease", [0.0]
        ),
    },
    {"model": "GaussianNB"},
    {
        "model": "GradientBoostingClassifier",
        "loss": hp.choice("GradientBoostingClassifier_loss", ["auto"]),
        "learning_rate": hp.loguniform(
            "GradientBoostingClassifier_learning_rate", np.log(0.01), np.log(1)
        ),
        "min_samples_leaf": scope.int(
            hp.loguniform(
                "GradientBoostingClassifier_min_samples_leaf", np.log(1), np.log(200)
            )
        ),
        "max_depth": hp.choice("GradientBoostingClassifier_max_depth", ["None"]),
        "max_leaf_nodes": scope.int(
            hp.loguniform(
                "GradientBoostingClassifier_max_leaf_nodes", np.log(3), np.log(2047)
            )
        ),
        "max_bins": hp.choice("GradientBoostingClassifier_max_bins", [255]),
        "l2_regularization": hp.loguniform(
            "GradientBoostingClassifier_l2_regularization", np.log(1e-10), np.log(1)
        ),
        "early_stop": hp.choice(
            "GradientBoostingClassifier_early_stop", ["off", "train", "valid"]
        ),
        "tol": hp.choice("GradientBoostingClassifier_tol", [1e-7]),
        "scoring": hp.choice("GradientBoostingClassifier_scoring", ["loss"]),
        "validation_fraction": hp.uniform(
            "GradientBoostingClassifier_validation_fraction", 0.01, 0.4
        ),
    },
    {
        "model": "KNearestNeighborsClassifier",
        "n_neighbors": scope.int(
            hp.quniform("KNearestNeighborsClassifier_n_neighbors", 1, 100, 1)
        ),
        "weights": hp.choice(
            "KNearestNeighborsClassifier_weights", ["uniform", "distance"]
        ),
        "p": hp.choice("KNearestNeighborsClassifier_p", [1, 2]),
    },
    {
        "model": "LDA",
        "shrinkage": hp.choice("LDA_shrinkage", [None, "auto", "manual"]),
        "shrinkage_factor": hp.uniform("LDA", 0.0, 1.0),
        "tol": hp.loguniform("LDA_tol", np.log(1e-5), np.log(1e-1)),
    },
    {
        "model": "LibLinear_SVC",
        # forbid penalty = 'l1' and loss = 'hinge'
        # forbid penalty = 'l2', loss = 'hinge' and dual = False
        # forbid penalty = 'l1' and dual = False
        "penalty": hp.choice("LibLinear_SVC_penalty", ["l2"]),
        "loss": hp.choice("LibLinear_SVC_loss", ["squared_hinge"]),
        "dual": hp.choice("LibLinear_SVC_dual", [False]),
        "tol": hp.loguniform("LibLinear_SVC_tol", np.log(1e-5), np.log(1e-1)),
        "C": hp.loguniform("LibLinear_SVC_C", np.log(0.03125), np.log(32768)),
        "multi_class": hp.choice("LibLinear_SVC_multi_class", ["ovr"]),
        "fit_intercept": hp.choice("LibLinear_SVC_fit_intercept", [True]),
        "intercept_scaling": hp.choice("LibLinear_SVC_intercept_scaling", [1]),
    },
    {
        "model": "LibSVM_SVC",
        # degree only selected when kernel = 'poly'
        # coef0 only selected when kernel = ['poly', 'sigmoid']
        "C": hp.loguniform("LibSVM_SVC_C", np.log(0.03125), np.log(32768)),
        "kernel": hp.choice("LibSVM_SVC_kernel", ["poly", "rbf", "sigmoid"]),
        "gamma": hp.loguniform("LibSVM_SVC_gamma", np.log(3.0517578125e-05), np.log(8)),
        "shrinking": hp.choice("LibSVM_SVC_shrinking", [True, False]),
        "tol": hp.loguniform("LibSVM_SVC_tol", np.log(1e-5), np.log(1e-1)),
        "max_iter": hp.choice("LibSVM_SVC_max_iter", [-1]),
        "degree": scope.int(hp.quniform("LibSVM_SVC_degree", 2, 5, 1)),
        "coef0": hp.uniform("LibSVM_SVC_coef0", -1, 1),
    },
    {
        "model": "MLPClassifier",
        "hidden_layer_depth": scope.int(
            hp.quniform("MLPClassifier_hidden_layer_depth", 1, 3, 1)
        ),
        "num_nodes_per_layer": scope.int(
            hp.loguniform("MLPClassifier_num_nodes_per_layer", np.log(16), np.log(264))
        ),
        "activation": hp.choice("MLPClassifier_activation", ["tanh", "relu"]),
        "alpha": hp.loguniform("MLPClassifier_alpha", np.log(1e-7), np.log(1e-1)),
        "learning_rate_init": hp.loguniform(
            "MLPClassifier_learning_rate_init", np.log(1e-4), np.log(0.5)
        ),
        "early_stopping": hp.choice("MLPClassifier_early_stopping", ["train", "valid"]),
        #'solver' : hp.choice('MLPClassifier_solver', ['lbfgs', 'sgd', 'adam']),
        # autosklearn must include _no_improvement_count, where only supported by 'sgd' and 'adam'
        "solver": hp.choice("MLPClassifier_solver", ["adam"]),
        "batch_size": hp.choice("MLPClassifier_batch_size", ["auto"]),
        "n_iter_no_change": hp.choice("MLPClassifier_n_iter_no_change", [32]),
        "tol": hp.choice("MLPClassifier_tol", [1e-4]),
        "shuffle": hp.choice("MLPClassifier_shuffle", [True]),
        "beta_1": hp.choice("MLPClassifier_beta_1", [0.9]),
        "beta_2": hp.choice("MLPClassifier_beta_2", [0.999]),
        "epsilon": hp.choice("MLPClassifier_epsilon", [1e-8]),
        "validation_fraction": hp.choice("MLPClassifier_validation_fraction", [0.1]),
    },
    {
        "model": "MultinomialNB",
        "alpha": hp.loguniform("MultinomialNB_alpha", np.log(1e-2), np.log(100)),
        "fit_prior": hp.choice("MultinomialNB_fit_prior", [True, False]),
    },
    {
        "model": "PassiveAggressive",
        "C": hp.loguniform("PassiveAggressive_C", np.log(1e-5), np.log(10)),
        "fit_intercept": hp.choice("PassiveAggressive_fit_intercept", [True]),
        "tol": hp.loguniform("PassiveAggressive_tol", np.log(1e-5), np.log(1e-1)),
        "loss": hp.choice("PassiveAggressive_loss", ["hinge", "squared_hinge"]),
        "average": hp.choice("PassiveAggressive_average", [True, False]),
    },
    {"model": "QDA", "reg_param": hp.uniform("QDA_reg_param", 0.0, 1.0)},
    {
        "model": "RandomForest",
        "criterion": hp.choice("RandomForest_criterion", ["gini", "entropy"]),
        "max_features": hp.uniform("RandomForest_max_features", 0.0, 1.0),
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
        # epsilon only selected for loss = 'modified_huber'
        # power_t only selected for learning_rate = 'invscaling'
        # eta0 only selected for learning_rate in ['constant', 'invscaling']
        "loss": hp.choice(
            "SGD_loss",
            ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
        ),
        "penalty": hp.choice("SGD_penalty", ["l1", "l2", "elasticnet"]),
        "alpha": hp.loguniform("SGD_alpha", np.log(1e-7), np.log(1e-1)),
        "fit_intercept": hp.choice("SGD_fit_intercept", [True]),
        "tol": hp.loguniform("SGD_tol", np.log(1e-5), np.log(1e-1)),
        "learning_rate": hp.choice(
            "SGD_learning_rate", ["constant", "optimal", "invscaling"]
        ),
        "l1_ratio": hp.loguniform("SGD_l1_ratio", np.log(1e-9), np.log(1)),
        "epsilon": hp.loguniform("SGD_epsilon", np.log(1e-5), np.log(1e-1)),
        "eta0": hp.loguniform("SGD_eta0", np.log(1e-7), np.log(1e-1)),
        "power_t": hp.uniform("SGD_power_t", 1e-5, 1),
        "average": hp.choice("SGD_average", [True, False]),
    },
    # self-defined models
    {
        "model": "MLP_Classifier",
        "hidden_layer": scope.int(hp.quniform("MLP_Classifier_hidden_layer", 1, 5, 1)),
        "hidden_size": scope.int(hp.quniform("MLP_Classifier_hidden_size", 1, 20, 1)),
        "activation": hp.choice(
            "MLP_Classifier_activation", ["ReLU", "Tanh", "Sigmoid"]
        ),
        "learning_rate": hp.uniform("MLP_Classifier_learning_rate", 1e-5, 1),
        "optimizer": hp.choice("MLP_Classifier_optimizer", ["Adam", "SGD"]),
        "criteria": hp.choice("MLP_Classifier_criteria", ["CrossEntropy"]),
        "batch_size": hp.choice("MLP_Classifier_batch_size", [16, 32, 64]),
        "num_epochs": scope.int(hp.quniform("MLP_Classifier_num_epochs", 5, 50, 1)),
    },
]
