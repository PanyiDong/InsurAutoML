"""
File Name: feature_selection_hyperparameter.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: ray
Latest Version: <<projectversion>>
Relative Path: /feature_selection_hyperparameter.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Wednesday, 17th May 2023 3:24:48 pm
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
from InsurAutoML.utils.base import format_hyper_dict

NOPROCESSING = {"feature_selection": "no_processing"}
LDASELECTION = {"feature_selection": "LDASelection"}
PCAFEATURESELECTION = {"feature_selection": "PCA_FeatureSelection"}
RBFSAMPLER = {"feature_selection": "RBFSampler"}
FEATUREFILTER = {
    "feature_selection": "FeatureFilter",
    "criteria": tune.choice(["Pearson", "MI"]),
    "n_prop": tune.uniform(0, 1),
}
ASFFS = {
    "feature_selection": "ASFFS",
    "model": tune.choice(["Linear", "Lasso", "Ridge"]),
    "objective": tune.choice(["MSE", "MAE"]),
}
GENETICALGORITHM = {
    "feature_selection": "GeneticAlgorithm",
    "n_prop": tune.uniform(0.6, 1),
    "n_generations": tune.qrandint(10, 40),
    "fs_method": tune.choice(["auto", "random", "Entropy", "t_statistics"]),
    "n_initial": tune.qrandint(5, 15),
    "fitness_fit": tune.choice(["Linear", "Decision Tree", "Random Forest", "SVM"]),
    "p_crossover": tune.uniform(0.8, 1),
    "p_mutation": tune.loguniform(1e-5, 1),
}
EXTRATREESPREPROCFORCLASSIFICATION = {
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
}
EXTRATREESPREPROCFORREGRESSION = {
    "feature_selection": "extra_trees_preproc_for_regression",
    "n_estimators": tune.choice([100]),
    "criterion": tune.choice(
        # ["mse", "friedman_mse", "mae"],
        ["squared_error", "absolute_error", "friedman_mse"],
    ),
    "min_samples_leaf": tune.qrandint(1, 20, 1),
    "min_samples_split": tune.qrandint(2, 20, 1),
    "max_features": tune.uniform(0.1, 1.0),
    "bootstrap": tune.choice([True, False]),
    "max_leaf_nodes": tune.choice([None]),
    "max_depth": tune.choice([None]),
    "min_weight_fraction_leaf": tune.choice([0.0]),
}
FASTICA = {
    "feature_selection": "fast_ica",
    # n_components only selected when whiten = True
    "algorithm": tune.choice(["parallel", "deflation"]),
    "whiten": tune.choice([True, False]),
    "fun": tune.choice(["logcosh", "exp", "cube"]),
    "n_components": tune.qrandint(10, 2000, 1),
}
FEATUREAGGLOMERATION = {
    "feature_selection": "feature_agglomeration",
    # forbid linkage = 'ward' while affinity in ['manhattan', 'cosine']
    "n_clusters": tune.qrandint(2, 400, 1),
    "affinity": tune.choice(["euclidean", "manhattan", "cosine"]),
    "linkage": tune.choice(["ward", "complete", "average"]),
    "pooling_func": tune.choice(["mean", "median", "max"]),
}
KERNELPCA = {
    "feature_selection": "kernel_pca",
    # degree only selected when kernel = 'poly'
    # coef0 only selected when kernel in ['poly', 'sigmoid']
    # gamma only selected when kernel in ['poly', 'rbf']
    "n_components": tune.qrandint(10, 2000, 1),
    "kernel": tune.choice(["poly", "rbf", "sigmoid", "cosine"]),
    "gamma": tune.loguniform(3.0517578125e-05, 8),
    "degree": tune.qrandint(2, 5, 1),
    "coef0": tune.uniform(-1, 1),
}
KITCHENSINKS = {
    "feature_selection": "kitchen_sinks",
    "gamma": tune.loguniform(3.0517578125e-05, 8),
    "n_components": tune.qrandint(50, 10000, 1),
}
LIBLINEARSVCPREPROCESSOR = {
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
}
NYSTROEMSAMPLER = {
    "feature_selection": "nystroem_sampler",
    # degree only selected when kernel = 'poly'
    # coef0 only selected when kernel in ['poly', 'sigmoid']
    # gamma only selected when kernel in ['poly', 'rbf', 'sigmoid']
    "kernel": tune.choice(["poly", "rbf", "sigmoid", "cosine"]),
    "n_components": tune.qrandint(50, 10000, 1),
    "gamma": tune.loguniform(3.0517578125e-05, 8),
    "degree": tune.qrandint(2, 5, 1),
    "coef0": tune.uniform(-1, 1),
}
PCA = {
    "feature_selection": "pca",
    "keep_variance": tune.uniform(0.5, 0.9999),
    "whiten": tune.choice([True, False]),
}
POLYNOMIAL = {
    "feature_selection": "polynomial",
    "degree": tune.qrandint(2, 4, 1),
    "interaction_only": tune.choice([True, False]),
    "include_bias": tune.choice([True, False]),
}
RANDOMTREESEMBEDDING = {
    "feature_selection": "random_trees_embedding",
    "n_estimators": tune.qrandint(10, 100, 1),
    "max_depth": tune.qrandint(2, 10, 1),
    "min_samples_split": tune.qrandint(2, 20, 1),
    "min_samples_leaf": tune.qrandint(1, 20, 1),
    "min_weight_fraction_leaf": tune.choice([1.0]),
    "max_leaf_nodes": tune.choice([None]),
    "bootstrap": tune.choice([True, False]),
}
SELECTPERCENTILECLASSIFICATION = {
    "feature_selection": "select_percentile_classification",
    "percentile": tune.qrandint(1, 99, 1),
    "score_func": tune.choice(
        ["chi2", "f_classif", "mutual_info"],
    ),
}
SELECTPERCENTILEREGRESSION = {
    "feature_selection": "select_percentile_regression",
    "percentile": tune.qrandint(1, 99, 1),
    "score_func": tune.choice(["f_regression", "mutual_info"]),
}
SELECTRATESCLASSIFICATION = {
    "feature_selection": "select_rates_classification",
    "alpha": tune.uniform(0.01, 0.5),
    "score_func": tune.choice(
        ["chi2", "f_classif", "mutual_info"],
    ),
    "mode": tune.choice(["fpr", "fdr", "fwe"]),
}
SELECTRATESREGRESSION = {
    "feature_selection": "select_rates_regression",
    "alpha": tune.uniform(0.01, 0.5),
    "score_func": tune.choice(["f_regression"]),
    "mode": tune.choice(["fpr", "fdr", "fwe"]),
}
TRUNCATEDSVD = {
    "feature_selection": "truncatedSVD",
    "target_dim": tune.qrandint(10, 256, 1),
}
EXHAUSTIVEFS = {
    "feature_selection": "ExhaustiveFS",
    "estimator": tune.choice(["Lasso"]),
    "criteria": tune.choice(["accuracy"]),
}
SFS = {
    "feature_selection": "SFS",
    # will be specified by task_type when initializing the HPO model
    "estimator": tune.choice(["Need to specify by task_type"]),
    "n_prop": tune.uniform(0, 1),
    "criteria": tune.choice(["Need to specify by task_type"]),
}
MRMR = {
    "feature_selection": "mRMR",
    "n_prop": tune.uniform(0, 1),
}
CBFS = {
    "feature_selection": "CBFS",
    "copula": tune.choice(["empirical"]),
    "n_prop": tune.uniform(0, 1),
}
FOCI = {
    "feature_selection": "FOCI",
}

# feature_selection
feature_selection_hyperparameter = [
    NOPROCESSING,
    LDASELECTION,
    PCAFEATURESELECTION,
    RBFSAMPLER,
    FEATUREFILTER,
    ASFFS,
    GENETICALGORITHM,
    EXTRATREESPREPROCFORCLASSIFICATION,
    EXTRATREESPREPROCFORREGRESSION,
    FASTICA,
    FEATUREAGGLOMERATION,
    KERNELPCA,
    KITCHENSINKS,
    LIBLINEARSVCPREPROCESSOR,
    NYSTROEMSAMPLER,
    PCA,
    POLYNOMIAL,
    RANDOMTREESEMBEDDING,
    SELECTPERCENTILECLASSIFICATION,
    SELECTPERCENTILEREGRESSION,
    SELECTRATESCLASSIFICATION,
    SELECTRATESREGRESSION,
    TRUNCATEDSVD,
    EXHAUSTIVEFS,
    SFS,
    MRMR,
    CBFS,
    # FOCI, # tend to get no feature selected
]

# deprecated, add custom hyperparameter construction by search algorithm in AutoTabularBase class
# feature_selection_hyperparameter = [
#     format_hyper_dict(dict, order + 1, ref = "feature_selection")
#     for order, dict in enumerate(feature_selection_hyperparameter)
# ]

if __name__ == "__main__":
    pass
