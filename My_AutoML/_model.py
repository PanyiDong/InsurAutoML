'''
File: _model.py
Author: Panyi Dong
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Relative Path: /_model.py
File Created: Friday, 25th February 2022 6:13:42 pm
Author: Panyi Dong (panyid2@illinois.edu)
-----
Last Modified: Tuesday, 1st March 2022 12:09:12 am
Modified By: Panyi Dong (panyid2@illinois.edu>)
-----
Copyright (c) 2022 - 2022, Panyi Dong
All rights reserved.
'''

import autosklearn.pipeline.components.classification
import autosklearn.pipeline.components.regression

# classifiers
classifiers = {
    "AdaboostClassifier": autosklearn.pipeline.components.classification.adaboost.AdaboostClassifier,  # autosklearn classifiers
    "BernoulliNB": autosklearn.pipeline.components.classification.bernoulli_nb.BernoulliNB,
    "DecisionTree": autosklearn.pipeline.components.classification.decision_tree.DecisionTree,
    "ExtraTreesClassifier": autosklearn.pipeline.components.classification.extra_trees.ExtraTreesClassifier,
    "GaussianNB": autosklearn.pipeline.components.classification.gaussian_nb.GaussianNB,
    "GradientBoostingClassifier": autosklearn.pipeline.components.classification.gradient_boosting.GradientBoostingClassifier,
    "KNearestNeighborsClassifier": autosklearn.pipeline.components.classification.k_nearest_neighbors.KNearestNeighborsClassifier,
    "LDA": autosklearn.pipeline.components.classification.lda.LDA,
    "LibLinear_SVC": autosklearn.pipeline.components.classification.liblinear_svc.LibLinear_SVC,
    "LibSVM_SVC": autosklearn.pipeline.components.classification.libsvm_svc.LibSVM_SVC,
    "MLPClassifier": autosklearn.pipeline.components.classification.mlp.MLPClassifier,
    "MultinomialNB": autosklearn.pipeline.components.classification.multinomial_nb.MultinomialNB,
    "PassiveAggressive": autosklearn.pipeline.components.classification.passive_aggressive.PassiveAggressive,
    "QDA": autosklearn.pipeline.components.classification.qda.QDA,
    "RandomForest": autosklearn.pipeline.components.classification.random_forest.RandomForest,
    "SGD": autosklearn.pipeline.components.classification.sgd.SGD,
}

# regressors
regressors = {
    "AdaboostRegressor": autosklearn.pipeline.components.regression.adaboost.AdaboostRegressor,
    "ARDRegression": autosklearn.pipeline.components.regression.ard_regression.ARDRegression,
    "DecisionTree": autosklearn.pipeline.components.regression.decision_tree.DecisionTree,
    "ExtraTreesRegressor": autosklearn.pipeline.components.regression.extra_trees.ExtraTreesRegressor,
    "GaussianProcess": autosklearn.pipeline.components.regression.gaussian_process.GaussianProcess,
    "GradientBoosting": autosklearn.pipeline.components.regression.gradient_boosting.GradientBoosting,
    "KNearestNeighborsRegressor": autosklearn.pipeline.components.regression.k_nearest_neighbors.KNearestNeighborsRegressor,
    "LibLinear_SVR": autosklearn.pipeline.components.regression.liblinear_svr.LibLinear_SVR,
    "LibSVM_SVR": autosklearn.pipeline.components.regression.libsvm_svr.LibSVM_SVR,
    "MLPRegressor": autosklearn.pipeline.components.regression.mlp.MLPRegressor,
    "RandomForest": autosklearn.pipeline.components.regression.random_forest.RandomForest,
    "SGD": autosklearn.pipeline.components.regression.sgd.SGD,
} # LibSVM_SVR, MLP and SGD have problems of requiring inverse_transform of StandardScaler while having 1D array
  # https://github.com/automl/auto-sklearn/issues/1297
  # problem solved