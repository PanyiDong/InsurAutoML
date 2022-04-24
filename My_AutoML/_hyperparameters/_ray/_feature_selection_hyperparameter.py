"""
File: _feature_selection_hyperparameter.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Latest Version: 0.2.0
Relative Path: /My_AutoML/_hyperparameters/_ray/_feature_selection_hyperparameter.py
File Created: Wednesday, 6th April 2022 10:06:01 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Sunday, 24th April 2022 5:50:41 pm
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
    {"feature_selection_1": "no_processing"},
    {"feature_selection_2": "LDASelection"},
    {"feature_selection_3": "PCA_FeatureSelection"},
    {"feature_selection_4": "RBFSampler"},
    {
        "feature_selection_5": "FeatureFilter",
        "FeatureFilter_criteria": tune.choice(["Pearson", "MI"]),
        "FeatureFilter_n_prop": tune.uniform(0, 1),
    },
    {
        "feature_selection_6": "ASFFS",
        "ASFFS_model": tune.choice(["Linear", "Lasso", "Ridge"]),
        "ASFFS_objective": tune.choice(["MSE", "MAE"]),
    },
    {
        "feature_selection_7": "GeneticAlgorithm",
        "GeneticAlgorithm_n_prop": tune.uniform(0.6, 1),
        "GeneticAlgorithm_n_generations": tune.qrandint(10, 40),
        "GeneticAlgorithm_feature_selection": tune.choice(
            ["auto", "random", "Entropy", "t_statistics"]
        ),
        "GeneticAlgorithm_n_initial": tune.qrandint(5, 15),
        "GeneticAlgorithm_fitness_fit": tune.choice(
            ["Linear", "Logistic", "Random Forest", "SVM"]
        ),
        "GeneticAlgorithm_p_crossover": tune.uniform(0.8, 1),
        "GeneticAlgorithm_p_mutation": tune.loguniform(1e-5, 1),
    },
    {
        "feature_selection_8": "extra_trees_preproc_for_classification",
        "extra_trees_preproc_for_classification_n_estimators": tune.choice([100]),
        "extra_trees_preproc_for_classification_criterion": tune.choice(
            ["gini", "entropy"]
        ),
        "extra_trees_preproc_for_classification_min_samples_leaf": tune.qrandint(
            1, 20, 1
        ),
        "extra_trees_preproc_for_classification_min_samples_split": tune.qrandint(
            2, 20, 1
        ),
        "extra_trees_preproc_for_classification_max_features": tune.uniform(0.1, 1.0),
        "extra_trees_preproc_for_classification_bootstrap": tune.choice([True, False]),
        "extra_trees_preproc_for_classification_max_leaf_nodes": tune.choice([None]),
        "extra_trees_preproc_for_classification_max_depth": tune.choice([None]),
        "extra_trees_preproc_for_classification_min_weight_fraction_leaf": tune.choice(
            [0.0]
        ),
        "extra_trees_preproc_for_classification_min_impurity_decrease": tune.choice(
            [0.0]
        ),
    },
    {
        "feature_selection_9": "extra_trees_preproc_for_regression",
        "extra_trees_preproc_for_regression_n_estimators": tune.choice([100]),
        "extra_trees_preproc_for_regression_criterion": tune.choice(
            ["mse", "friedman_mse", "mae"],
        ),
        "extra_trees_preproc_for_regression_min_samples_leaf": tune.qrandint(1, 20, 1),
        "extra_trees_preproc_for_regression_min_samples_split": tune.qrandint(2, 20, 1),
        "extra_trees_preproc_for_regression_max_features": tune.uniform(0.1, 1.0),
        "extra_trees_preproc_for_regression_bootstrap": tune.choice([True, False]),
        "extra_trees_preproc_for_regression_max_leaf_nodes": tune.choice([None]),
        "extra_trees_preproc_for_regression_max_depth": tune.choice([None]),
        "extra_trees_preproc_for_regression_min_weight_fraction_leaf": tune.choice(
            [0.0]
        ),
    },
    {
        "feature_selection_10": "fast_ica",
        # n_components only selected when whiten = True
        "fast_ica_algorithm": tune.choice(["parallel", "deflation"]),
        "fast_ica_whiten": tune.choice([True, False]),
        "fast_ica_fun": tune.choice(["logcosh", "exp", "cube"]),
        "fast_ica_n_components": tune.qrandint(10, 2000, 1),
    },
    {
        "feature_selection_11": "feature_agglomeration",
        # forbid linkage = 'ward' while affinity in ['manhattan', 'cosine']
        "feature_agglomeration_n_clusters": tune.qrandint(2, 400, 1),
        "feature_agglomeration_affinity": tune.choice(
            ["euclidean", "manhattan", "cosine"]
        ),
        "feature_agglomeration_linkage": tune.choice(["ward", "complete", "average"]),
        "feature_agglomeration_pooling_func": tune.choice(["mean", "median", "max"]),
    },
    {
        "feature_selection_12": "kernel_pca",
        # degree only selected when kernel = 'poly'
        # coef0 only selected when kernel in ['poly', 'sigmoid']
        # gamma only selected when kernel in ['poly', 'rbf']
        "kernel_pca_n_components": tune.qrandint(10, 2000, 1),
        "kernel_pca_kernel": tune.choice(["poly", "rbf", "sigmoid", "cosine"]),
        "kernel_pca_gamma": tune.loguniform(3.0517578125e-05, 8),
        "kernel_pca_degree": tune.qrandint(2, 5, 1),
        "kernel_pca_coef0": tune.uniform(-1, 1),
    },
    {
        "feature_selection_13": "kitchen_sinks",
        "kitchen_sinks_gamma": tune.loguniform(3.0517578125e-05, 8),
        "kitchen_sinks_n_components": tune.qrandint(50, 10000, 1),
    },
    {
        "feature_selection_14": "liblinear_svc_preprocessor",
        # forbid penalty = 'l1' while loss = 'hinge'
        "liblinear_svc_preprocessor_penalty": tune.choice(["l1"]),
        "liblinear_svc_preprocessor_loss": tune.choice(["squared_hinge"]),
        "liblinear_svc_preprocessor_dual": tune.choice([False]),
        "liblinear_svc_preprocessor_tol": tune.loguniform(1e-5, 1e-1),
        "liblinear_svc_preprocessor_C": tune.loguniform(0.03125, 32768),
        "liblinear_svc_preprocessor_multi_class": tune.choice(["ovr"]),
        "liblinear_svc_preprocessor_fit_intercept": tune.choice([True]),
        "liblinear_svc_preprocessor_intercept_scaling": tune.choice([1]),
    },
    {
        "feature_selection_15": "nystroem_sampler",
        # degree only selected when kernel = 'poly'
        # coef0 only selected when kernel in ['poly', 'sigmoid']
        # gamma only selected when kernel in ['poly', 'rbf', 'sigmoid']
        "nystroem_sampler_kernel": tune.choice(["poly", "rbf", "sigmoid", "cosine"]),
        "nystroem_sampler_n_components": tune.qrandint(50, 10000, 1),
        "nystroem_sampler_gamma": tune.loguniform(3.0517578125e-05, 8),
        "nystroem_sampler_degree": tune.qrandint(2, 5, 1),
        "nystroem_sampler_coef0": tune.uniform(-1, 1),
    },
    {
        "feature_selection_16": "pca",
        "pca_keep_variance": tune.uniform(0.5, 0.9999),
        "pca_whiten": tune.choice([True, False]),
    },
    {
        "feature_selection_17": "polynomial",
        "polynomial_degree": tune.qrandint(2, 4, 1),
        "polynomial_interaction_only": tune.choice([True, False]),
        "polynomial_include_bias": tune.choice([True, False]),
    },
    {
        "feature_selection_18": "random_trees_embedding",
        "random_trees_embedding_n_estimators": tune.qrandint(10, 100, 1),
        "random_trees_embedding_max_depth": tune.qrandint(2, 10, 1),
        "random_trees_embedding_min_samples_split": tune.qrandint(2, 20, 1),
        "random_trees_embedding_min_samples_leaf": tune.qrandint(1, 20, 1),
        "random_trees_embedding_min_weight_fraction_leaf": tune.choice([1.0]),
        "random_trees_embedding_max_leaf_nodes": tune.choice([None]),
        "random_trees_embedding_bootstrap": tune.choice([True, False]),
    },
    {
        "feature_selection_19": "select_percentile_classification",
        "select_percentile_classification_percentile": tune.qrandint(1, 99, 1),
        "select_percentile_classification_score_func": tune.choice(
            ["chi2", "f_classif", "mutual_info"],
        ),
    },
    {
        "feature_selection_20": "select_percentile_regression",
        "select_percentile_regression_percentile": tune.qrandint(1, 99, 1),
        "select_percentile_regression_score_func": tune.choice(
            ["f_regression", "mutual_info"]
        ),
    },
    {
        "feature_selection_21": "select_rates_classification",
        "select_rates_classification_alpha": tune.uniform(0.01, 0.5),
        "select_rates_classification_score_func": tune.choice(
            ["chi2", "f_classif", "mutual_info_classif"],
        ),
        "select_rates_classification_mode": tune.choice(["fpr", "fdr", "fwe"]),
    },
    {
        "feature_selection_22": "select_rates_regression",
        "select_rates_regression_alpha": tune.uniform(0.01, 0.5),
        "select_rates_regression_score_func": tune.choice(["f_regression"]),
        "select_rates_regression_mode": tune.choice(["fpr", "fdr", "fwe"]),
    },
    {
        "feature_selection_23": "truncatedSVD",
        "truncatedSVD_target_dim": tune.qrandint(10, 256, 1),
    },
    {
        "feature_selection_24": "ExhaustiveFS",
        "ExhaustiveFS_estimator": tune.choice(["Lasso"]),
        "ExhaustiveFS_criteria": tune.choice(["accuracy"]),
    },
    {
        "feature_selection_25": "SFS",
        # will be specified by task_type when initializing the HPO model
        "SFS_estimator": tune.choice(["Need to specify by task_type"]),
        "SFS_n_prop": tune.uniform(0, 1),
        "SFS_criteria": tune.choice(["Need to specify by task_type"]),
    },
    {
        "feature_selection_26": "mRMR",
        "mRMR_n_prop": tune.uniform(0, 1),
    },
    {
        "feature_selection_27": "CBFS",
        "CBFS_copula": tune.choice(["empirical"]),
        "CBFS_n_prop": tune.uniform(0, 1),
    },
]
