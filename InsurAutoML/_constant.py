"""
File: _constant.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_constant.py
File: _constant.py
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Tuesday, 15th November 2022 4:03:08 pm
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

import logging

LOGGINGLEVEL = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]

FULLTYPE_MAPPING = {
    ("Int", "Numerical"): "con",
    ("Int", "Categorical"): "cat",
    ("Float", ""): "con",
    ("Datetime", ""): "",
    ("Object", "Categorical"): "cat",
    ("Object", "Text"): "txt",
    ("Path", ""): "img",
}

UNIQUE_FULLTYPE = list(FULLTYPE_MAPPING.keys())

# encoders
ENCODERS = ["DataEncoding"]

# imputers
IMPUTERS = [
    "SimpleImputer",
    "DummyImputer",
    "JointImputer",
    "ExpectationMaximization",
    "KNNImputer",
    "MissForestImputer",
    "MICE",
    "AAI_kNN",
    "KMI",
    "CMI",
    "k_Prototype_NN",
]

# balancings
BALANCINGS = [
    "no_processing",
    "SimpleRandomOverSampling",
    "SimpleRandomUnderSampling",
    "TomekLink",
    "EditedNearestNeighbor",
    "CondensedNearestNeighbor",
    "OneSidedSelection",
    "CNN_TomekLink",
    "Smote",
    "Smote_TomekLink",
    "Smote_ENN",
]

# scalings
SCALINGS = [
    "no_processing",
    "MinMaxScale",
    "Standardize",
    "Normalize",
    "RobustScale",
    "PowerTransformer",
    "QuantileTransformer",
    "Winsorization",
    "Feature_Manipulation",
    "Feature_Truncation",
]

# feature_selection
FEATURE_SELECTION = [
    "no_processing",
    "LDASelection",
    "PCA_FeatureSelection",
    "RBFSampler",
    "FeatureFilter",
    "ASFFS",
    "GeneticAlgorithm",
    "extra_trees_preproc_for_classification",
    "extra_trees_preproc_for_regression",
    "liblinear_svc_preprocessor",
    "polynomial",
    "select_percentile_classification",
    "select_percentile_regression",
    "select_rates_classification",
    "select_rates_regression",
    "truncatedSVD",
]

# classifiers
CLASSIFIERS = [
    "AdaboostClassifier",
    "BernoulliNB",
    "DecisionTree",
    "ExtraTreesClassifier",
    "GaussianNB",
    "GradientBoostingClassifier",
    "KNearestNeighborsClassifier",
    "LDA",
    "LibLinear_SVC",
    "LibSVM_SVC",
    "MLPClassifier",
    "MultinomialNB",
    "PassiveAggressive",
    "QDA",
    "RandomForest",
    "SGD",
    "LogisticRegression",
    "ComplementNB",
    "HistGradientBoostingClassifier",
    "LightGBM_Classifier",
    "XGBoost_Classifier",
    "GAM_Classifier",
    "MLP_Classifier",
    "RNN_Classifier",
]

# regressors
REGRESSORS = [
    "AdaboostRegressor",
    "ARDRegression",
    "DecisionTree",
    "ExtraTreesRegressor",
    "GaussianProcess",
    "GradientBoosting",
    "KNearestNeighborsRegressor",
    "LibLinear_SVR",
    "LibSVM_SVR",
    "MLPRegressor",
    "RandomForest",
    "SGD",
    "LinearRegression",
    "Lasso",
    "RidgeRegression",
    "ElasticNet",
    "BayesianRidge",
    "HistGradientBoostingRegressor",
    "LightGBM_Regressor",
    "XGBoost_Regressor",
    "GAM_Regressor",
    "MLP_Regressor",
    "RNN_Regressor",
]

# maximum unique classes determined as categorical variable
# 31 is capped by days in a month
UNI_CLASS = 31

# maximum iteration allowed for the algorithm
MAX_ITER = 1024

# maximum time budge allowed per run (in seconds)
# set at 3 days
MAX_TIME = 259200

# LightGBM default object (metric/loss)
# binary classification
LIGHTGBM_BINARY_CLASSIFICATION = ["binary", "cross_entropy"]
# multiclass classification
LIGHTGBM_MULTICLASS_CLASSIFICATION = ["multiclass", "multiclassova", "num_class"]
# regression
LIGHTGBM_REGRESSION = [
    "regression",
    "regression_l1",
    "huber",
    "fair",
    "poisson",
    "quantile",
    "mape",
    "gamma",
    "tweedie",
]

# LightGBM boosting methods
LIGHTGBM_BOOSTING = ["gbdt", "dart", "goss"]  # suppress "rf"


# LightGBM tree learner
LIGHTGBM_TREE_LEARNER = ["serial", "feature", "data", "voting"]


# Classification estimators
CLASSIFICATION_ESTIMATORS = [
    "LogisticRegression",
    "ExtraTreeClassifier",
    "RandomForestClassifier",
]

# Classification metrics
CLASSIFICATION_CRITERIA = [
    "neg_accuracy",
    "neg_precision",
    "neg_auc",
    "neg_hinge",
    "neg_f1",
]

# Regression estimators
REGRESSION_ESTIMATORS = [
    "Lasso",
    "Ridge",
    "ExtraTreeRegressor",
    "RandomForestRegressor",
]

# Regression metrics
REGRESSION_CRITERIA = [
    "MSE",
    "MAE",
    # "MSLE", # not general since it needs non-negative values
    "neg_R2",
    "MAX",
]

# methods corresponding process must have
METHOD_MAPPING = {
    "encoder": ["fit", "refit"],
    "imputer": ["fill"],
    "balancing": ["fit_transform"],
    "scaling": ["fit", "transform", "fit_transform"],
    "feature_selection": ["fit", "transform"],
    "model": ["fit", "predict"],
}
