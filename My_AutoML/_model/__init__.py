"""
File: __init__.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Latest Version: 0.2.0
Relative Path: /My_AutoML/_model/__init__.py
File Created: Friday, 8th April 2022 9:04:05 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 18th April 2022 12:20:17 am
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

from ._autosklearn import (
    # autosklearn classifiers
    AdaboostClassifier,
    BernoulliNB,
    DecisionTreeClassifier,
    ExtraTreesClassifier,
    GaussianNB,
    GradientBoostingClassifier,
    KNearestNeighborsClassifier,
    LDA,
    LibLinear_SVC,
    LibSVM_SVC,
    MLPClassifier,
    MultinomialNB,
    PassiveAggressive,
    QDA,
    RandomForestClassifier,
    SGDClassifier,
    # autosklearn regressors
    AdaboostRegressor,
    ARDRegression,
    DecisionTreeRegressor,
    ExtraTreesRegressor,
    GaussianProcess,
    GradientBoosting,
    KNearestNeighborsRegressor,
    LibLinear_SVR,
    # LibSVM_SVR,
    # MLPRegressor,
    RandomForestRegressor,
    # SGDRegressor
)

from ._sklearn import (
    # sklearn classifiers
    LogisticRegression,
    # ComplementNB,
    HistGradientBoostingClassifier,
    # sklearn regressors
    LinearRegression,
    Lasso,
    Ridge,
    ElasticNet,
    BayesianRidge,
    # HistGradientBoostingRegressor,
)

####################################################################################################
# classifiers

classifiers = {
    # classification models from autosklearn
    "AdaboostClassifier": AdaboostClassifier,
    "BernoulliNB": BernoulliNB,
    "DecisionTree": DecisionTreeClassifier,
    "ExtraTreesClassifier": ExtraTreesClassifier,
    "GaussianNB": GaussianNB,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "KNearestNeighborsClassifier": KNearestNeighborsClassifier,
    "LDA": LDA,
    "LibLinear_SVC": LibLinear_SVC,
    "LibSVM_SVC": LibSVM_SVC,
    "MLPClassifier": MLPClassifier,
    "MultinomialNB": MultinomialNB,
    "PassiveAggressive": PassiveAggressive,
    "QDA": QDA,
    "RandomForest": RandomForestClassifier,
    "SGD": SGDClassifier,
    # classification models from sklearn
    "LogisticRegression": LogisticRegression,
    # "ComplementNB": ComplementNB,
    "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
    # self-defined models
}

# regressors
regressors = {
    # regression models from autosklearn
    "AdaboostRegressor": AdaboostRegressor,
    "ARDRegression": ARDRegression,
    "DecisionTree": DecisionTreeRegressor,
    "ExtraTreesRegressor": ExtraTreesRegressor,
    "GaussianProcess": GaussianProcess,
    "GradientBoosting": GradientBoosting,
    "KNearestNeighborsRegressor": KNearestNeighborsRegressor,
    "LibLinear_SVR": LibLinear_SVR,
    # "LibSVM_SVR": LibSVM_SVR,
    # "MLPRegressor": MLPRegressor,
    "RandomForest": RandomForestRegressor,
    # "SGD": SGDSGDRegressor,
    # regression models from sklearn
    "LinearRegression": LinearRegression,
    "Lasso": Lasso,
    "RidgeRegression": Ridge,
    "ElasticNet": ElasticNet,
    "BayesianRidge": BayesianRidge,
    # "HistGradientBoostingRegressor": HistGradientBoostingRegressor, # not well-supported by package conflicts
    # self-defined models
}


"""
LibSVM_SVR, MLP and SGD have problems of requiring inverse_transform 
of StandardScaler while having 1D array
https://github.com/automl/auto-sklearn/issues/1297
problem solved
"""

import importlib

# check whether lightgbm installed
# if installed, add lightgbm depended classifiers/regressors
lightgbm_spec = importlib.util.find_spec("lightgbm")
if lightgbm_spec is not None:
    from ._lightgbm import LightGBM_Classifier, LightGBM_Regressor

    classifiers["LightGBM_Classifier"] = LightGBM_Classifier

    regressors["LightGBM_Regressor"] = LightGBM_Regressor

# check weather xgboost installed
xgboost_spec = importlib.util.find_spec("xgboost")
if xgboost_spec is not None:
    from ._xgboost import XGBoost_Classifier, XGBoost_Regressor

    classifiers["XGBoost_Classifier"] = XGBoost_Classifier

    regressors["XGBoost_Regressor"] = XGBoost_Regressor

# check whether pygam installed
pygam_spec = importlib.util.find_spec("pygam")
if pygam_spec is not None:
    from ._gam import GAM_Classifier, GAM_Regressor

    classifiers["GAM_Classifier"] = GAM_Classifier

    regressors["GAM_Regressor"] = GAM_Regressor

# check whether torch installed
# if installed, add torch depended classifiers/regressors
torch_spec = importlib.util.find_spec("torch")
if torch_spec is not None:
    from ._FNN import MLP_Classifier, MLP_Regressor
    from ._RNN import RNN_Classifier, RNN_Regressor

    classifiers["MLP_Classifier"] = MLP_Classifier
    classifiers["RNN_Classifier"] = RNN_Classifier

    regressors["MLP_Regressor"] = MLP_Regressor
    regressors["RNN_Regressor"] = RNN_Regressor
