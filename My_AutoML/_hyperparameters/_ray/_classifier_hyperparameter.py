"""
File: _classifier_hyperparameter.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Relative Path: /My_AutoML/_hyperparameters/_classifier_hyperparameter.py
File Created: Tuesday, 5th April 2022 11:05:31 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Thursday, 7th April 2022 11:17:49 am
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

# classifier hyperparameters
classifier_hyperparameter = [
    # extract from autosklearn
    {
        "model": "AdaboostClassifier",
        "n_estimators": tune.qrandint(10, 500, 1),
        "learning_rate": tune.uniform(0.01, 2),
        "algorithm": tune.choice(["SAMME", "SAMME.R"]),
        # for base_estimator of Decision Tree
        "max_depth": tune.qrandint(1, 10, 1),
    },
    {
        "model": "BernoulliNB",
        "alpha": tune.loguniform(1e-2, 100),
        "fit_prior": tune.choice([True, False]),
    },
    {
        "model": "DecisionTree",
        "criterion": tune.choice(["gini", "entropy"]),
        "max_features": tune.choice([1.0]),
        "max_depth_factor": tune.uniform(0.0, 2.0),
        "min_samples_split": tune.qrandint(2, 20, 1),
        "min_samples_leaf": tune.qrandint(1, 20, 1),
        "min_weight_fraction_leaf": tune.choice([0.0]),
        "max_leaf_nodes": tune.choice(["None"]),
        "min_impurity_decrease": tune.choice([0.0]),
    },
    {
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
    },
    {"model": "GaussianNB"},
    {
        "model": "GradientBoostingClassifier",
        "loss": tune.choice(["auto"]),
        "learning_rate": tune.loguniform(0.01, 1),
        "min_samples_leaf": tune.qlograndint(1, 200, 1),
        "max_depth": tune.choice(["None"]),
        "max_leaf_nodes": tune.qlograndint(3, 2047, 1),
        "max_bins": tune.choice([255]),
        "l2_regularization": tune.loguniform(1e-10, 1),
        "early_stop": tune.choice(["off", "train", "valid"]),
        "tol": tune.choice([1e-7]),
        "scoring": tune.choice(["loss"]),
        "validation_fraction": tune.uniform(0.01, 0.4),
    },
    {
        "model": "KNearestNeighborsClassifier",
        "n_neighbors": tune.qrandint(1, 100, 1),
        "weights": tune.choice(["uniform", "distance"]),
        "p": tune.choice([1, 2]),
    },
    {
        "model": "LDA",
        "shrinkage": tune.choice([None, "auto", "manual"]),
        "shrinkage_factor": tune.uniform(0.0, 1.0),
        "tol": tune.loguniform(1e-5, 1e-1),
    },
    {
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
    },
    {
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
    },
    {
        "model": "MLPClassifier",
        "hidden_layer_depth": tune.qrandint(1, 3, 1),
        "num_nodes_per_layer": tune.qlograndint(16, 264, 1),
        "activation": tune.choice(["tanh", "relu"]),
        "alpha": tune.loguniform(1e-7, 1e-1),
        "learning_rate_init": tune.loguniform(1e-4, 0.5),
        "early_stopping": tune.choice(["train", "valid"]),
        #'solver' : tune.choice('MLPClassifier_solver', ['lbfgs', 'sgd', 'adam']),
        # autosklearn must include _no_improvement_count, where only supported by 'sgd' and 'adam'
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
        "model": "MultinomialNB",
        "alpha": tune.loguniform(1e-2, 100),
        "fit_prior": tune.choice([True, False]),
    },
    {
        "model": "PassiveAggressive",
        "C": tune.loguniform(1e-5, 10),
        "fit_intercept": tune.choice([True]),
        "tol": tune.loguniform(1e-5, 1e-1),
        "loss": tune.choice(["hinge", "squared_hinge"]),
        "average": tune.choice([True, False]),
    },
    {"model": "QDA", "reg_param": tune.uniform(0.0, 1.0)},
    {
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
    },
    {
        "model": "SGD",
        # l1_ratio only selected for penalty = 'elasticnet'
        # epsilon only selected for loss = 'modified_huber'
        # power_t only selected for learning_rate = 'invscaling'
        # eta0 only selected for learning_rate in ['constant', 'invscaling']
        "loss": tune.choice(
            ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
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
    },
    # self-defined models
    {
        "model": "MLP_Classifier",
        "hidden_layer": tune.qrandint(1, 5, 1),
        "hidden_size": tune.qrandint(1, 20, 1),
        "activation": tune.choice(["ReLU", "Tanh", "Sigmoid"]),
        "learning_rate": tune.uniform(1e-5, 1),
        "optimizer": tune.choice(["Adam", "SGD"]),
        "criteria": tune.choice(["CrossEntropy"]),
        "batch_size": tune.choice([16, 32, 64]),
        "num_epochs": tune.qrandint(5, 50, 1),
    },
]
