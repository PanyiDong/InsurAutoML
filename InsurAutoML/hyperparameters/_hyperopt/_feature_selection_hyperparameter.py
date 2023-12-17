"""
File: _feature_selection_hyperparameter.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Latest Version: 0.2.0
Relative Path: /My_AutoML/_hyperparameters/_hyperopt/_feature_selection_hyperparameter.py
File Created: Tuesday, 5th April 2022 11:04:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Friday, 8th April 2022 10:22:54 pm
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

# feature_selection
feature_selection_hyperparameter = [
    {"feature_selection": "no_processing"},
    {"feature_selection": "LDASelection"},
    {"feature_selection": "PCA_FeatureSelection"},
    {"feature_selection": "RBFSampler"},
    {"feature_selection": "FeatureFilter"},
    {"feature_selection": "ASFFS"},
    {"feature_selection": "GeneticAlgorithm"},
    {
        "feature_selection": "extra_trees_preproc_for_classification",
        "n_estimators": hp.choice(
            "extra_trees_preproc_for_classification_n_estimators", [100]
        ),
        "criterion": hp.choice(
            "extra_trees_preproc_for_classification_criterion", ["gini", "entropy"]
        ),
        "min_samples_leaf": scope.int(
            hp.quniform(
                "extra_trees_preproc_for_classification_min_samples_leaf", 1, 20, 1
            )
        ),
        "min_samples_split": scope.int(
            hp.quniform(
                "extra_trees_preproc_for_classification_min_samples_split", 2, 20, 1
            )
        ),
        "max_features": hp.uniform(
            "extra_trees_preproc_for_classification_max_features", 0.1, 1.0
        ),
        "bootstrap": hp.choice(
            "extra_trees_preproc_for_classification_bootstrap", [True, False]
        ),
        "max_leaf_nodes": hp.choice(
            "extra_trees_preproc_for_classification_max_leaf_nodes", [None]
        ),
        "max_depth": hp.choice(
            "extra_trees_preproc_for_classification_max_depth", [None]
        ),
        "min_weight_fraction_leaf": hp.choice(
            "extra_trees_preproc_for_classification_min_weight_fraction_leaf", [0.0]
        ),
        "min_impurity_decrease": hp.choice(
            "extra_trees_preproc_for_classification_min_impurity_decrease", [0.0]
        ),
    },
    {
        "feature_selection": "extra_trees_preproc_for_regression",
        "n_estimators": hp.choice(
            "extra_trees_preproc_for_regression_n_estimators", [100]
        ),
        "criterion": hp.choice(
            "extra_trees_preproc_for_regression_criterion",
            ["mse", "friedman_mse", "mae"],
        ),
        "min_samples_leaf": scope.int(
            hp.quniform("extra_trees_preproc_for_regression_min_samples_leaf", 1, 20, 1)
        ),
        "min_samples_split": scope.int(
            hp.quniform(
                "extra_trees_preproc_for_regression_min_samples_split", 2, 20, 1
            )
        ),
        "max_features": hp.uniform(
            "extra_trees_preproc_for_regression_max_features", 0.1, 1.0
        ),
        "bootstrap": hp.choice(
            "extra_trees_preproc_for_regression_bootstrap", [True, False]
        ),
        "max_leaf_nodes": hp.choice(
            "extra_trees_preproc_for_regression_max_leaf_nodes", [None]
        ),
        "max_depth": hp.choice("extra_trees_preproc_for_regression_max_depth", [None]),
        "min_weight_fraction_leaf": hp.choice(
            "extra_trees_preproc_for_regression_min_weight_fraction_leaf", [0.0]
        ),
    },
    {
        "feature_selection": "fast_ica",
        # n_components only selected when whiten = True
        "algorithm": hp.choice("fast_ica_algorithm", ["parallel", "deflation"]),
        "whiten": hp.choice("fast_ica_whiten", [True, False]),
        "fun": hp.choice("fast_ica_fun", ["logcosh", "exp", "cube"]),
        "n_components": scope.int(hp.quniform("fast_ica_n_components", 10, 2000, 1)),
    },
    {
        "feature_selection": "feature_agglomeration",
        # forbid linkage = 'ward' while affinity in ['manhattan', 'cosine']
        "n_clusters": scope.int(
            hp.quniform("feature_agglomeration_n_clusters", 2, 400, 1)
        ),
        "affinity": hp.choice(
            "feature_agglomeration_affinity", ["euclidean", "manhattan", "cosine"]
        ),
        "linkage": hp.choice(
            "feature_agglomeration_linkage", ["ward", "complete", "average"]
        ),
        "pooling_func": hp.choice(
            "feature_agglomeration_pooling_func", ["mean", "median", "max"]
        ),
    },
    {
        "feature_selection": "kernel_pca",
        # degree only selected when kernel = 'poly'
        # coef0 only selected when kernel in ['poly', 'sigmoid']
        # gamma only selected when kernel in ['poly', 'rbf']
        "n_components": scope.int(hp.quniform("kernel_pca_n_components", 10, 2000, 1)),
        "kernel": hp.choice("kernel_pca_kernel", ["poly", "rbf", "sigmoid", "cosine"]),
        "gamma": hp.loguniform("kernel_pca_gamma", np.log(3.0517578125e-05), np.log(8)),
        "degree": scope.int(hp.quniform("kernel_pca_degree", 2, 5, 1)),
        "coef0": hp.uniform("kernel_pca_coef0", -1, 1),
    },
    {
        "feature_selection": "kitchen_sinks",
        "gamma": hp.loguniform(
            "kitchen_sinks_gamma", np.log(3.0517578125e-05), np.log(8)
        ),
        "n_components": scope.int(
            hp.quniform("kitchen_sinks_n_components", 50, 10000, 1)
        ),
    },
    {
        "feature_selection": "liblinear_svc_preprocessor",
        # forbid penalty = 'l1' while loss = 'hinge'
        "penalty": hp.choice("liblinear_svc_preprocessor_penalty", ["l1"]),
        "loss": hp.choice("liblinear_svc_preprocessor_loss", ["squared_hinge"]),
        "dual": hp.choice("liblinear_svc_preprocessor_dual", [False]),
        "tol": hp.loguniform(
            "liblinear_svc_preprocessor_tol", np.log(1e-5), np.log(1e-1)
        ),
        "C": hp.loguniform(
            "liblinear_svc_preprocessor_C", np.log(0.03125), np.log(32768)
        ),
        "multi_class": hp.choice("liblinear_svc_preprocessor_multi_class", ["ovr"]),
        "fit_intercept": hp.choice("liblinear_svc_preprocessor_fit_intercept", [True]),
        "intercept_scaling": hp.choice(
            "liblinear_svc_preprocessor_intercept_scaling", [1]
        ),
    },
    {
        "feature_selection": "nystroem_sampler",
        # degree only selected when kernel = 'poly'
        # coef0 only selected when kernel in ['poly', 'sigmoid']
        # gamma only selected when kernel in ['poly', 'rbf', 'sigmoid']
        "kernel": hp.choice(
            "nystroem_sampler_kernel", ["poly", "rbf", "sigmoid", "cosine"]
        ),
        "n_components": scope.int(
            hp.quniform("nystroem_sampler_n_components", 50, 10000, 1)
        ),
        "gamma": hp.loguniform(
            "nystroem_sampler_gamma", np.log(3.0517578125e-05), np.log(8)
        ),
        "degree": scope.int(hp.quniform("nystroem_sampler_degree", 2, 5, 1)),
        "coef0": hp.uniform("nystroem_sampler_coef0", -1, 1),
    },
    {
        "feature_selection": "pca",
        "keep_variance": hp.uniform("pca_keep_variance", 0.5, 0.9999),
        "whiten": hp.choice("pca_whiten", [True, False]),
    },
    {
        "feature_selection": "polynomial",
        "degree": scope.int(hp.quniform("polynomial_degree", 2, 3, 1)),
        "interaction_only": hp.choice("polynomial_interaction_only", [True, False]),
        "include_bias": hp.choice("polynomial_include_bias", [True, False]),
    },
    {
        "feature_selection": "random_trees_embedding",
        "n_estimators": scope.int(
            hp.quniform("random_trees_embedding_n_estimators", 10, 100, 1)
        ),
        "max_depth": scope.int(
            hp.quniform("random_trees_embedding_max_depth", 2, 10, 1)
        ),
        "min_samples_split": scope.int(
            hp.quniform("random_trees_embedding_min_samples_split", 2, 20, 1)
        ),
        "min_samples_leaf": scope.int(
            hp.quniform("random_trees_embedding_min_samples_leaf", 1, 20, 1)
        ),
        "min_weight_fraction_leaf": hp.choice(
            "random_trees_embedding_min_weight_fraction_leaf", [1.0]
        ),
        "max_leaf_nodes": hp.choice("random_trees_embedding_max_leaf_nodes", [None]),
        "bootstrap": hp.choice("random_trees_embedding_bootstrap", [True, False]),
    },
    {
        "feature_selection": "select_percentile_classification",
        "percentile": scope.int(
            hp.quniform("select_percentile_classification_percentile", 1, 99, 1)
        ),
        "score_func": hp.choice(
            "select_percentile_classification_score_func",
            ["chi2", "f_classif", "mutual_info"],
        ),
    },
    {
        "feature_selection": "select_percentile_regression",
        "percentile": scope.int(
            hp.quniform("select_percentile_regression_percentile", 1, 99, 1)
        ),
        "score_func": hp.choice(
            "select_percentile_regression_score_func", ["f_regression", "mutual_info"]
        ),
    },
    {
        "feature_selection": "select_rates_classification",
        "alpha": hp.uniform("select_rates_classification_alpha", 0.01, 0.5),
        "score_func": hp.choice(
            "select_rates_classification_score_func",
            ["chi2", "f_classif", "mutual_info_classif"],
        ),
        "mode": hp.choice("select_rates_classification_mode", ["fpr", "fdr", "fwe"]),
    },
    {
        "feature_selection": "select_rates_regression",
        "alpha": hp.uniform("select_rates_regression_alpha", 0.01, 0.5),
        "score_func": hp.choice(
            "select_rates_classification_score_func", ["f_regression"]
        ),
        "mode": hp.choice("select_rates_classification_mode", ["fpr", "fdr", "fwe"]),
    },
    {
        "feature_selection": "truncatedSVD",
        "target_dim": scope.int(hp.quniform("truncatedSVD_target_dim", 10, 256, 1)),
    },
]
