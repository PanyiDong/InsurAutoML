"""
File: __init__.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_model/__init__.py
File: __init__.py
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Tuesday, 15th November 2022 4:18:37 pm
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

####################################################################################################
# classifiers

classifiers = {
    # self-defined models
}

# regressors
regressors = {
    # self-defined models
}

"""
autosklearn models
LibSVM_SVR, MLP and SGD have problems of requiring inverse_transform 
of StandardScaler while having 1D array
https://github.com/automl/auto-sklearn/issues/1297
problem solved
"""

import importlib

# Update: Nov 15, 2022
# autosklearn decrypted, no need to consider it
# check whether auto-sklearn has been installed
# if not, import the compatible models from sklearn
# autosklearn_spec = importlib.util.find_spec("autosklearn")
# sklearn_spec = importlib.util.find_spec("sklearn")
# if autosklearn_spec is not None:
#     from ._autosklearn import (
#         # autosklearn classifiers
#         AdaboostClassifier,
#         BernoulliNB,
#         DecisionTreeClassifier,
#         ExtraTreesClassifier,
#         GaussianNB,
#         HistGradientBoostingClassifier,
#         KNearestNeighborsClassifier,
#         LDA,
#         LibLinear_SVC,
#         LibSVM_SVC,
#         MLPClassifier,
#         MultinomialNB,
#         PassiveAggressive,
#         QDA,
#         RandomForestClassifier,
#         SGDClassifier,
#         # autosklearn regressors
#         AdaboostRegressor,
#         ARDRegression,
#         DecisionTreeRegressor,
#         ExtraTreesRegressor,
#         GaussianProcess,
#         HistGradientBoostingRegressor,
#         KNearestNeighborsRegressor,
#         LibLinear_SVR,
#         LibSVM_SVR,
#         MLPRegressor,
#         RandomForestRegressor,
#         SGDRegressor,
#     )

#     # classification models from autosklearn
#     classifiers["AdaboostClassifier"] = AdaboostClassifier
#     classifiers["BernoulliNB"] = BernoulliNB
#     classifiers["DecisionTree"] = DecisionTreeClassifier
#     classifiers["ExtraTreesClassifier"] = ExtraTreesClassifier
#     classifiers["GaussianNB"] = GaussianNB
#     classifiers["HistGradientBoostingClassifier"] = HistGradientBoostingClassifier
#     classifiers["KNearestNeighborsClassifier"] = KNearestNeighborsClassifier
#     classifiers["LDA"] = LDA
#     classifiers["LibLinear_SVC"] = LibLinear_SVC
#     classifiers["LibSVM_SVC"] = LibSVM_SVC
#     classifiers["MLPClassifier"] = MLPClassifier
#     classifiers["MultinomialNB"] = MultinomialNB
#     classifiers["PassiveAggressive"] = PassiveAggressive
#     classifiers["QDA"] = QDA
#     classifiers["RandomForest"] = RandomForestClassifier
#     classifiers["SGD"] = SGDClassifier

#     # regression models from autosklearn
#     regressors["AdaboostRegressor"] = AdaboostRegressor
#     regressors["ARDRegression"] = ARDRegression
#     regressors["DecisionTree"] = DecisionTreeRegressor
#     regressors["ExtraTreesRegressor"] = ExtraTreesRegressor
#     regressors["GaussianProcess"] = GaussianProcess
#     regressors["HistGradientBoostingRegressor"] = HistGradientBoostingRegressor
#     regressors["KNearestNeighborsRegressor"] = KNearestNeighborsRegressor
#     regressors["LibLinear_SVR"] = LibLinear_SVR
#     regressors["LibSVM_SVR"] = LibSVM_SVR
#     regressors["MLPRegressor"] = MLPRegressor
#     regressors["RandomForest"] = RandomForestRegressor
#     regressors["SGD"] = SGDRegressor

# # if sklearn is also not installed, raise an error
# elif sklearn_spec is None:
#     raise ImportError("Neither auto-sklearn nor sklearn is installed.")
# # else import the compatible models from sklearn
# else:
from ._sklearn import (
    # sklearn classifiers
    AdaboostClassifier,
    BernoulliNB,
    DecisionTreeClassifier,
    ExtraTreesClassifier,
    GaussianNB,
    HistGradientBoostingClassifier,
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
    # sklearn regressors
    AdaboostRegressor,
    ARDRegression,
    DecisionTreeRegressor,
    ExtraTreesRegressor,
    GaussianProcess,
    HistGradientBoostingRegressor,
    KNearestNeighborsRegressor,
    LibLinear_SVR,
    LibSVM_SVR,
    MLPRegressor,
    RandomForestRegressor,
    SGDRegressor,
    # sklearn classifiers
    LogisticRegression,
    # ComplementNB,
    # HistGradientBoostingClassifier,
    GradientBoostingClassifier,
    # sklearn regressors
    LinearRegression,
    Lasso,
    Ridge,
    ElasticNet,
    BayesianRidge,
    # HistGradientBoostingRegressor,
    GradientBoostingRegressor,
)

# classification models from sklearn
classifiers["AdaboostClassifier"] = AdaboostClassifier
classifiers["BernoulliNB"] = BernoulliNB
classifiers["DecisionTree"] = DecisionTreeClassifier
classifiers["ExtraTreesClassifier"] = ExtraTreesClassifier
classifiers["GaussianNB"] = GaussianNB
classifiers["HistGradientBoostingClassifier"] = HistGradientBoostingClassifier
classifiers["KNearestNeighborsClassifier"] = KNearestNeighborsClassifier
classifiers["LDA"] = LDA
classifiers["LibLinear_SVC"] = LibLinear_SVC
classifiers["LibSVM_SVC"] = LibSVM_SVC
classifiers["MLPClassifier"] = MLPClassifier
classifiers["MultinomialNB"] = MultinomialNB
classifiers["PassiveAggressive"] = PassiveAggressive
classifiers["QDA"] = QDA
classifiers["RandomForest"] = RandomForestClassifier
classifiers["SGD"] = SGDClassifier
classifiers["LogisticRegression"] = LogisticRegression
# classifiers["ComplementNB"] = ComplementNB
# classifiers["HistGradientBoostingClassifier"] = HistGradientBoostingClassifier
classifiers["GradientBoostingClassifier"] = GradientBoostingClassifier

# regression models from sklearn
regressors["AdaboostRegressor"] = AdaboostRegressor
regressors["ARDRegression"] = ARDRegression
regressors["DecisionTree"] = DecisionTreeRegressor
regressors["ExtraTreesRegressor"] = ExtraTreesRegressor
regressors["GaussianProcess"] = GaussianProcess
regressors["HistGradientBoostingRegressor"] = HistGradientBoostingRegressor
regressors["KNearestNeighborsRegressor"] = KNearestNeighborsRegressor
regressors["LibLinear_SVR"] = LibLinear_SVR
regressors["LibSVM_SVR"] = LibSVM_SVR
regressors["MLPRegressor"] = MLPRegressor
regressors["RandomForest"] = RandomForestRegressor
regressors["SGD"] = SGDRegressor
regressors["LinearRegression"] = LinearRegression
regressors["Lasso"] = Lasso
regressors["RidgeRegression"] = Ridge
regressors["ElasticNet"] = ElasticNet
regressors["BayesianRidge"] = BayesianRidge
# regressors["HistGradientBoostingRegressor"] = HistGradientBoostingRegressor # not well-supported by package conflicts
regressors["GradientBoostingRegressor"] = GradientBoostingRegressor

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
