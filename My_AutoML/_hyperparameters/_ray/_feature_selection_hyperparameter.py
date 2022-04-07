"""
File: _feature_selection_hyperparameter.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Relative Path: /My_AutoML/_hyperparameters/_feature_selection_hyperparameter.py
File Created: Tuesday, 5th April 2022 11:04:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Thursday, 7th April 2022 11:20:29 am
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
        "n_estimators": tune.choice([100]),
        "criterion": tune.choice(["gini", "entropy"]),
        "min_samples_leaf": tune.qrandint(1, 20, 1),
        "min_samples_split": tune.qrandint(2, 20, 1),
        "max_features": tune.uniform(0.1, 1.0),
        "bootstrap": tune.choice([True, False]),
        "max_leaf_nodes": tune.choice([None]),
        "max_depth": tune.choice([None]),
        "min_weight_fraction_leaf": tune.choice([0.0]),
        "min_impurity_decrease": tune.choice([0.0]),
    },
    {
        "feature_selection": "extra_trees_preproc_for_regression",
        "n_estimators": tune.choice([100]),
        "criterion": tune.choice(
            ["mse", "friedman_mse", "mae"],
        ),
        "min_samples_leaf": tune.qrandint(1, 20, 1),
        "min_samples_split": tune.qrandint(2, 20, 1),
        "max_features": tune.uniform(0.1, 1.0),
        "bootstrap": tune.choice([True, False]),
        "max_leaf_nodes": tune.choice([None]),
        "max_depth": tune.choice([None]),
        "min_weight_fraction_leaf": tune.choice([0.0]),
    },
    {
        "feature_selection": "fast_ica",
        # n_components only selected when whiten = True
        "algorithm": tune.choice(["parallel", "deflation"]),
        "whiten": tune.choice([True, False]),
        "fun": tune.choice(["logcosh", "exp", "cube"]),
        "n_components": tune.qrandint(10, 2000, 1),
    },
    {
        "feature_selection": "feature_agglomeration",
        # forbid linkage = 'ward' while affinity in ['manhattan', 'cosine']
        "n_clusters": tune.qrandint(2, 400, 1),
        "affinity": tune.choice(["euclidean", "manhattan", "cosine"]),
        "linkage": tune.choice(["ward", "complete", "average"]),
        "pooling_func": tune.choice(["mean", "median", "max"]),
    },
    {
        "feature_selection": "kernel_pca",
        # degree only selected when kernel = 'poly'
        # coef0 only selected when kernel in ['poly', 'sigmoid']
        # gamma only selected when kernel in ['poly', 'rbf']
        "n_components": tune.qrandint(10, 2000, 1),
        "kernel": tune.choice(["poly", "rbf", "sigmoid", "cosine"]),
        "gamma": tune.loguniform(3.0517578125e-05, 8),
        "degree": tune.qrandint(2, 5, 1),
        "coef0": tune.uniform(-1, 1),
    },
    {
        "feature_selection": "kitchen_sinks",
        "gamma": tune.loguniform(3.0517578125e-05, 8),
        "n_components": tune.qrandint(50, 10000, 1),
    },
    {
        "feature_selection": "liblinear_svc_preprocessor",
        # forbid penalty = 'l1' while loss = 'hinge'
        "penalty": tune.choice(["l1"]),
        "loss": tune.choice(["squared_hinge"]),
        "dual": tune.choice([False]),
        "tol": tune.loguniform(1e-5, 1e-1),
        "C": tune.loguniform(0.03125, 32768),
        "multi_class": tune.choice(["ovr"]),
        "fit_intercept": tune.choice([True]),
        "intercept_scaling": tune.choice([1]),
    },
    {
        "feature_selection": "nystroem_sampler",
        # degree only selected when kernel = 'poly'
        # coef0 only selected when kernel in ['poly', 'sigmoid']
        # gamma only selected when kernel in ['poly', 'rbf', 'sigmoid']
        "kernel": tune.choice(["poly", "rbf", "sigmoid", "cosine"]),
        "n_components": tune.qrandint(50, 10000, 1),
        "gamma": tune.loguniform(3.0517578125e-05, 8),
        "degree": tune.qrandint(2, 5, 1),
        "coef0": tune.uniform(-1, 1),
    },
    {
        "feature_selection": "pca",
        "keep_variance": tune.uniform(0.5, 0.9999),
        "whiten": tune.choice([True, False]),
    },
    {
        "feature_selection": "polynomial",
        "degree": tune.qrandint(2, 3, 1),
        "interaction_only": tune.choice([True, False]),
        "include_bias": tune.choice([True, False]),
    },
    {
        "feature_selection": "random_trees_embedding",
        "n_estimators": tune.qrandint(10, 100, 1),
        "max_depth": tune.qrandint(2, 10, 1),
        "min_samples_split": tune.qrandint(2, 20, 1),
        "min_samples_leaf": tune.qrandint(1, 20, 1),
        "min_weight_fraction_leaf": tune.choice([1.0]),
        "max_leaf_nodes": tune.choice([None]),
        "bootstrap": tune.choice([True, False]),
    },
    {
        "feature_selection": "select_percentile_classification",
        "percentile": tune.qrandint(1, 99, 1),
        "score_func": tune.choice(
            ["chi2", "f_classif", "mutual_info"],
        ),
    },
    {
        "feature_selection": "select_percentile_regression",
        "percentile": tune.qrandint(1, 99, 1),
        "score_func": tune.choice(["f_regression", "mutual_info"]),
    },
    {
        "feature_selection": "select_rates_classification",
        "alpha": tune.uniform(0.01, 0.5),
        "score_func": tune.choice(
            ["chi2", "f_classif", "mutual_info_classif"],
        ),
        "mode": tune.choice(["fpr", "fdr", "fwe"]),
    },
    {
        "feature_selection": "select_rates_regression",
        "alpha": tune.uniform(0.01, 0.5),
        "score_func": tune.choice(["f_regression"]),
        "mode": tune.choice(["fpr", "fdr", "fwe"]),
    },
    {
        "feature_selection": "truncatedSVD",
        "target_dim": tune.qrandint(10, 256, 1),
    },
]
