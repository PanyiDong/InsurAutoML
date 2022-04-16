"""
File: _classifier_hyperparameter.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Latest Version: 0.2.0
Relative Path: /My_AutoML/_hyperparameters/_ray/_classifier_hyperparameter.py
File Created: Friday, 8th April 2022 9:04:05 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Friday, 15th April 2022 8:41:38 pm
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

from My_AutoML._constant import LIGHTGBM_BOOSTING, LIGHTGBM_TREE_LEARNER

# classifier hyperparameters
classifier_hyperparameter = [
    # classification models from autosklearn
    {
        "model_1": "AdaboostClassifier",
        "AdaboostClassifier_n_estimators": tune.qlograndint(10, 500, 1),
        "AdaboostClassifier_learning_rate": tune.uniform(0.01, 2),
        "AdaboostClassifier_algorithm": tune.choice(["SAMME", "SAMME.R"]),
        # for base_estimator of Decision Tree
        "AdaboostClassifier_max_depth": tune.qrandint(1, 10, 1),
    },
    {
        "model_2": "BernoulliNB",
        "BernoulliNB_alpha": tune.loguniform(1e-2, 100),
        "BernoulliNB_fit_prior": tune.choice([True, False]),
    },
    {
        "model_3": "DecisionTree",
        "DecisionTree_criterion": tune.choice(["gini", "entropy"]),
        "DecisionTree_max_features": tune.choice([1.0]),
        "DecisionTree_max_depth_factor": tune.uniform(0.0, 2.0),
        "DecisionTree_min_samples_split": tune.qrandint(2, 20, 1),
        "DecisionTree_min_samples_leaf": tune.qrandint(1, 20, 1),
        "DecisionTree_min_weight_fraction_leaf": tune.choice([0.0]),
        "DecisionTree_max_leaf_nodes": tune.choice(["None"]),
        "DecisionTree_min_impurity_decrease": tune.choice([0.0]),
    },
    {
        "model_4": "ExtraTreesClassifier",
        "ExtraTreesClassifier_criterion": tune.choice(["gini", "entropy"]),
        "ExtraTreesClassifier_min_samples_leaf": tune.qrandint(1, 20, 1),
        "ExtraTreesClassifier_min_samples_split": tune.qrandint(2, 20, 1),
        "ExtraTreesClassifier_max_features": tune.uniform(0.0, 1.0),
        "ExtraTreesClassifier_bootstrap": tune.choice([True, False]),
        "ExtraTreesClassifier_max_leaf_nodes": tune.choice(["None"]),
        "ExtraTreesClassifier_max_depth": tune.choice(["None"]),
        "ExtraTreesClassifier_min_weight_fraction_leaf": tune.choice([0.0]),
        "ExtraTreesClassifier_min_impurity_decrease": tune.choice([0.0]),
    },
    {"model_5": "GaussianNB"},
    {
        "model_6": "GradientBoostingClassifier",
        "GradientBoostingClassifier_loss": tune.choice(["auto"]),
        "GradientBoostingClassifier_learning_rate": tune.loguniform(0.01, 1),
        "GradientBoostingClassifier_min_samples_leaf": tune.qlograndint(1, 200, 1),
        "GradientBoostingClassifier_max_depth": tune.choice(["None"]),
        "GradientBoostingClassifier_max_leaf_nodes": tune.qlograndint(3, 2047, 1),
        "GradientBoostingClassifier_max_bins": tune.choice([255]),
        "GradientBoostingClassifier_l2_regularization": tune.loguniform(1e-10, 1),
        "GradientBoostingClassifier_early_stop": tune.choice(["off", "train", "valid"]),
        "GradientBoostingClassifier_tol": tune.choice([1e-7]),
        "GradientBoostingClassifier_scoring": tune.choice(["loss"]),
        "GradientBoostingClassifier_validation_fraction": tune.uniform(0.01, 0.4),
    },
    {
        "model_7": "KNearestNeighborsClassifier",
        "KNearestNeighborsClassifier_n_neighbors": tune.qrandint(1, 100, 1),
        "KNearestNeighborsClassifier_weights": tune.choice(["uniform", "distance"]),
        "KNearestNeighborsClassifier_p": tune.choice([1, 2]),
    },
    {
        "model_8": "LDA",
        "LDA_shrinkage": tune.choice([None, "auto", "manual"]),
        "LDA_shrinkage_factor": tune.uniform(0.0, 1.0),
        "LDA_tol": tune.loguniform(1e-5, 1e-1),
    },
    {
        "model_9": "LibLinear_SVC",
        # forbid penalty = 'l1' and loss = 'hinge'
        # forbid penalty = 'l2', loss = 'hinge' and dual = False
        # forbid penalty = 'l1' and dual = False
        "LibLinear_SVC_penalty": tune.choice(["l2"]),
        "LibLinear_SVC_loss": tune.choice(["squared_hinge"]),
        "LibLinear_SVC_dual": tune.choice([False]),
        "LibLinear_SVC_tol": tune.loguniform(1e-5, 1e-1),
        "LibLinear_SVC_C": tune.loguniform(0.03125, 32768),
        "LibLinear_SVC_multi_class": tune.choice(["ovr"]),
        "LibLinear_SVC_fit_intercept": tune.choice([True]),
        "LibLinear_SVC_intercept_scaling": tune.choice([1]),
    },
    {
        "model_10": "LibSVM_SVC",
        # degree only selected when kernel = 'poly'
        # coef0 only selected when kernel = ['poly', 'sigmoid']
        "LibSVM_SVC_C": tune.loguniform(0.03125, 32768),
        "LibSVM_SVC_kernel": tune.choice(["poly", "rbf", "sigmoid"]),
        "LibSVM_SVC_gamma": tune.loguniform(3.0517578125e-05, 8),
        "LibSVM_SVC_shrinking": tune.choice([True, False]),
        "LibSVM_SVC_tol": tune.loguniform(1e-5, 1e-1),
        "LibSVM_SVC_max_iter": tune.choice([-1]),
        "LibSVM_SVC_degree": tune.qrandint(2, 5, 1),
        "LibSVM_SVC_coef0": tune.uniform(-1, 1),
    },
    {
        "model_11": "MLPClassifier",
        "MLPClassifier_hidden_layer_depth": tune.qrandint(1, 3, 1),
        "MLPClassifier_num_nodes_per_layer": tune.qlograndint(16, 264, 1),
        "MLPClassifier_activation": tune.choice(["tanh", "relu"]),
        "MLPClassifier_alpha": tune.loguniform(1e-7, 1e-1),
        "MLPClassifier_learning_rate_init": tune.loguniform(1e-4, 0.5),
        "MLPClassifier_early_stopping": tune.choice(["train", "valid"]),
        #'solver' : tune.choice('MLPClassifier_solver', ['lbfgs', 'sgd', 'adam']),
        # autosklearn must include _no_improvement_count, where only supported by 'sgd' and 'adam'
        "MLPClassifier_solver": tune.choice(["adam"]),
        "MLPClassifier_batch_size": tune.choice(["auto"]),
        "MLPClassifier_n_iter_no_change": tune.choice([32]),
        "MLPClassifier_tol": tune.choice([1e-4]),
        "MLPClassifier_shuffle": tune.choice([True]),
        "MLPClassifier_beta_1": tune.choice([0.9]),
        "MLPClassifier_beta_2": tune.choice([0.999]),
        "MLPClassifier_epsilon": tune.choice([1e-8]),
        "MLPClassifier_validation_fraction": tune.choice([0.1]),
    },
    {
        "model_12": "MultinomialNB",
        "MultinomialNB_alpha": tune.loguniform(1e-2, 100),
        "MultinomialNB_fit_prior": tune.choice([True, False]),
    },
    {
        "model_13": "PassiveAggressive",
        "PassiveAggressive_C": tune.loguniform(1e-5, 10),
        "PassiveAggressive_fit_intercept": tune.choice([True]),
        "PassiveAggressive_tol": tune.loguniform(1e-5, 1e-1),
        "PassiveAggressive_loss": tune.choice(["hinge", "squared_hinge"]),
        "PassiveAggressive_average": tune.choice([True, False]),
    },
    {"model_14": "QDA", "QDA_reg_param": tune.uniform(0.0, 1.0)},
    {
        "model_15": "RandomForest",
        "RandomForest_criterion": tune.choice(["gini", "entropy"]),
        "RandomForest_max_features": tune.uniform(0.0, 1.0),
        "RandomForest_max_depth": tune.choice([None]),
        "RandomForest_min_samples_split": tune.qrandint(2, 20, 1),
        "RandomForest_min_samples_leaf": tune.qrandint(1, 20, 1),
        "RandomForest_min_weight_fraction_leaf": tune.choice([0.0]),
        "RandomForest_bootstrap": tune.choice([True, False]),
        "RandomForest_max_leaf_nodes": tune.choice([None]),
        "RandomForest_min_impurity_decrease": tune.choice([0.0]),
    },
    {
        "model_16": "SGD",
        # l1_ratio only selected for penalty = 'elasticnet'
        # epsilon only selected for loss = 'modified_huber'
        # power_t only selected for learning_rate = 'invscaling'
        # eta0 only selected for learning_rate in ['constant', 'invscaling']
        "SGD_loss": tune.choice(
            ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
        ),
        "SGD_penalty": tune.choice(["l1", "l2", "elasticnet"]),
        "SGD_alpha": tune.loguniform(1e-7, 1e-1),
        "SGD_fit_intercept": tune.choice([True]),
        "SGD_tol": tune.loguniform(1e-5, 1e-1),
        "SGD_learning_rate": tune.choice(["constant", "optimal", "invscaling"]),
        "SGD_l1_ratio": tune.loguniform(1e-9, 1),
        "SGD_epsilon": tune.loguniform(1e-5, 1e-1),
        "SGD_eta0": tune.loguniform(1e-7, 1e-1),
        "SGD_power_t": tune.uniform(1e-5, 1),
        "SGD_average": tune.choice([True, False]),
    },
    # classification models from sklearn
    {
        "model_17": "LogisticRegression",
        "LogisticRegression_penalty": tune.choice(["l2", "none"]),
        "LogisticRegression_tol": tune.loguniform(1e-5, 1e-1),
        "LogisticRegression_C": tune.loguniform(1e-5, 10),
    },
    {
        "model_18": "ComplementNB",
        "ComplementNB_alpha": tune.uniform(0, 1),
        "ComplementNB_fit_prior": tune.choice([True, False]),
        "ComplementNB_norm": tune.choice([True, False]),
    },
    {
        "model_19": "HistGradientBoostingClassifier",
        "HistGradientBoostingClassifier_loss": tune.choice(["auto"]),
        "HistGradientBoostingClassifier_learning_rate": tune.uniform(1e-7, 1),
        "HistGradientBoostingClassifier_max_leaf_nodes": tune.choice([None]),
        "HistGradientBoostingClassifier_max_depth": tune.choice([None]),
        "HistGradientBoostingClassifier_min_samples_leaf": tune.qrandint(1, 20, 1),
        "HistGradientBoostingClassifier_l2_regularization": tune.uniform(0, 1),
        "HistGradientBoostingClassifier_tol": tune.loguniform(1e-5, 1e-1),
    },
    # self-defined models
    {
        "model_20": "MLP_Classifier",
        "MLP_Classifier_hidden_layer": tune.qrandint(1, 5, 1),
        "MLP_Classifier_hidden_size": tune.qrandint(1, 10, 1),
        "MLP_Classifier_activation": tune.choice(["Tanh", "Sigmoid"]),
        "MLP_Classifier_learning_rate": tune.uniform(1e-5, 1),
        "MLP_Classifier_optimizer": tune.choice(["Adam", "SGD"]),
        "MLP_Classifier_criteria": tune.choice(
            ["CrossEntropy", "NegativeLogLikelihood"]
        ),
        "MLP_Classifier_batch_size": tune.choice([16, 32, 64]),
        "MLP_Classifier_num_epochs": tune.qrandint(5, 30, 1),
    },
    {
        "model_21": "RNN_Classifier",
        "RNN_Classifier_hidden_size": tune.choice([16, 32, 64, 128, 256]),
        "RNN_Classifier_n_layers": tune.qrandint(1, 5, 1),
        "RNN_Classifier_RNN_unit": tune.choice(["RNN", "LSTM", "GRU"]),
        "RNN_Classifier_activation": tune.choice(["ReLU", "Tanh", "Sigmoid"]),
        "RNN_Classifier_dropout": tune.loguniform(1e-7, 0.8),
        "RNN_Classifier_learning_rate": tune.loguniform(1e-7, 1),
        "RNN_Classifier_optimizer": tune.choice(["Adam", "SGD"]),
        "RNN_Classifier_criteria": tune.choice(
            ["CrossEntropy", "NegativeLogLikelihood"]
        ),
        "RNN_Classifier_batch_size": tune.choice([16, 32, 64]),
        "RNN_Classifier_num_epochs": tune.qrandint(5, 30, 1),
    },
    {
        "model_22": "LightGBM_Classifier",
        "LightGBM_Classifier_objective": tune.choice(
            ["Need to specify in HPO by response"]
        ),
        "LightGBM_Classifier_boosting": tune.choice(LIGHTGBM_BOOSTING),
        "LightGBM_Classifier_n_estimators": tune.qlograndint(50, 500, 1),
        # max_depth == -1 for no limit
        "LightGBM_Classifier_max_depth": tune.randint(-1, 31),
        "LightGBM_Classifier_num_leaves": tune.qlograndint(3, 2047, 1),
        "LightGBM_Classifier_min_data_in_leaf": tune.qrandint(1, 20, 1),
        "LightGBM_Classifier_learning_rate": tune.loguniform(1e-7, 1),
        "LightGBM_Classifier_tree_learner": tune.choice(LIGHTGBM_TREE_LEARNER),
        "LightGBM_Classifier_num_iterations": tune.qlograndint(50, 500, 1),
    },
    {
        "model_23": "XGBoost_Classifier",
        "XGBoost_Classifier_eta": tune.uniform(0, 1),
        "XGBoost_Classifier_gamma": tune.loguniform(1e-10, 1e3),
        "XGBoost_Classifier_max_depth": tune.randint(1, 31),
        "XGBoost_Classifier_min_child_weight": tune.loguniform(1e-10, 1e3),
        "XGBoost_Classifier_max_delta_step": tune.loguniform(1e-7, 1e1),
        "XGBoost_Classifier_reg_lambda": tune.uniform(0, 1),
        "XGBoost_Classifier_reg_alpha": tune.uniform(0, 1),
    },
    {
        "model_24": "GAM_Classifier",
        "GAM_Classifier_type": tune.choice(["logistic"]),
        "GAM_Classifier_tol": tune.loguniform(1e-7, 1),
    },
]
