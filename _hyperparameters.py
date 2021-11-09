import numpy as np
import autosklearn
import autosklearn.pipeline.components.classification
import autosklearn.pipeline.components.regression
from hyperopt import hp
from hyperopt.pyll import scope

# hyperparameters for AutoML pipeline

# encoder
encoder_hyperparameter = [{"encoder": "DataEncoding"}]

# imputer
imputer_hyperparameter = [
    {
        "imputer": "SimpleImputer",
        "method": hp.choice(
            "SimpleImputer_method", ["mean", "zero", "median", "most frequent"]
        ),
    },
    {"imputer": "DummyImputer"},
    {"imputer": "JointImputer"},
    {
        "imputer": "ExpectationMaximization",
        "iterations": hp.quniform("ExpectationMaximization_iterations", 10, 100, 1),
        "threshold": hp.uniform(
            "ExpectationMaximization_iterations_threshold", 1e-5, 1
        ),
    },
    {
        "imputer": "KNNImputer",
        "n_neighbors": scope.int(hp.quniform("KNNImputer_n_neighbors", 1, 15, 1)),
        "fold" : scope.int(hp.quniform("KNNImputer_fold", 5, 15, 1))
    },
    {"imputer": "MissForestImputer"},
    {"imputer": "MICE", "cycle": hp.quniform("MICE_cycle", 5, 20, 1)},
    {"imputer": "GAIN"},
]

# scaling
scaling_hyperparameter = [
    {"scaling": "NoScaling"},
    {"scaling": "Standardize"},
    {"scaling": "Normalize"},
    {"scaling": "RobustScale"},
    {"scaling": "MinMaxScale"},
    {"scaling": "Winsorization"},
]

# balancing
# if the imbalance threshold small, TomekLink will take too long
balancing_hyperparameter = [
    {"balancing": "no_processing"},
    {
        "balancing": "SimpleRandomOverSampling",
        "imbalance_threshold": hp.uniform(
            "SimpleRandomOverSampling_imbalance_threshold", 0.8, 1
        ),
    },
    {
        "balancing": "SimpleRandomUnderSampling",
        "imbalance_threshold": hp.uniform(
            "SimpleRandomUnderSampling_imbalance_threshold", 0.8, 1
        ),
    },
    {
        "balancing": "TomekLink",
        "imbalance_threshold": hp.uniform("TomekLink_imbalance_threshold", 0.8, 1),
    },
    {
        "balancing": "EditedNearestNeighbor",
        "imbalance_threshold": hp.uniform(
            "EditedNearestNeighbor_imbalance_threshold", 0.8, 1
        ),
        "k" : scope.int(hp.quniform("EditedNearestNeighbor_k", 1, 7, 1))
    },
    {
        "balancing": "CondensedNearestNeighbor",
        "imbalance_threshold": hp.uniform(
            "CondensedNearestNeighbor_imbalance_threshold", 0.8, 1
        ),
    },
    {
        "balancing": "OneSidedSelection",
        "imbalance_threshold": hp.uniform(
            "OneSidedSelection_imbalance_threshold", 0.8, 1
        ),
    },
    {
        "balancing": "CNN_TomekLink",
        "imbalance_threshold": hp.uniform("CNN_TomekLink_imbalance_threshold", 0.8, 1),
    },
    {
        "balancing": "Smote",
        "imbalance_threshold": hp.uniform("Smote_imbalance_threshold", 0.8, 1),
        "k" : scope.int(hp.quniform("Smote_k", 1, 10, 1))
    },
    {
        "balancing": "Smote_TomekLink",
        "imbalance_threshold": hp.uniform(
            "Smote_TomekLink_imbalance_threshold", 0.8, 1
        ),
        "k" : scope.int(hp.quniform("Smote_TomekLink_k", 1, 10, 1))
    },
    {
        "balancing": "Smote_ENN",
        "imbalance_threshold": hp.uniform("Smote_ENN_imbalance_threshold", 0.8, 1),
        "k" : scope.int(hp.quniform("Smote_ENN_k", 1, 10, 1))
    },
]

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
        "loss": hp.choice(
            "liblinear_svc_preprocessor_loss", ["squared_hinge"]
        ),
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

# classifier hyperparameters
# extract from autosklearn
classifier_hyperparameter = [
    {
        "model": "AdaboostClassifier",
        "n_estimators": scope.int(
            hp.quniform("AdaboostClassifier_n_estimators", 10, 500, 1)
        ),
        "learning_rate": hp.uniform("AdaboostClassifier_learning_rate", 0.01, 2),
        "algorithm": hp.choice("AdaboostClassifier_algorithm", ["SAMME", "SAMME.R"]),
        # for base_estimator of Decision Tree
        "max_depth": scope.int(hp.quniform("AdaboostClassifier_max_depth", 1, 10, 1)),
    },
    {
        "model": "BernoulliNB",
        "alpha": hp.loguniform("BernoulliNB_alpha", np.log(1e-2), np.log(100)),
        "fit_prior": hp.choice("BernoulliNB_fit_prior", [True, False]),
    },
    {
        "model": "DecisionTree",
        "criterion": hp.choice("DecisionTree_criterion", ["gini", "entropy"]),
        "max_features": hp.choice("DecisionTree_max_features", [1.0]),
        "max_depth_factor": hp.uniform("DecisionTree_max_depth_factor", 0.0, 2.0),
        "min_samples_split": scope.int(
            hp.quniform("DecisionTree_min_samples_split", 2, 20, 1)
        ),
        "min_samples_leaf": scope.int(
            hp.quniform("DecisionTree_min_samples_leaf", 1, 20, 1)
        ),
        "min_weight_fraction_leaf": hp.choice(
            "DecisionTree_min_weight_fraction_leaf", [0.0]
        ),
        "max_leaf_nodes": hp.choice("DecisionTree_max_leaf_nodes", ["None"]),
        "min_impurity_decrease": hp.choice("DecisionTree_min_impurity_decrease", [0.0]),
    },
    {
        "model": "ExtraTreesClassifier",
        "criterion": hp.choice("ExtraTreesClassifier_criterion", ["gini", "entropy"]),
        "min_samples_leaf": scope.int(
            hp.quniform("ExtraTreesClassifier_min_samples_leaf", 1, 20, 1)
        ),
        "min_samples_split": scope.int(
            hp.quniform("ExtraTreesClassifier_min_samples_split", 2, 20, 1)
        ),
        "max_features": hp.uniform("ExtraTreesClassifier_max_features", 0.0, 1.0),
        "bootstrap": hp.choice("ExtraTreesClassifier_bootstrap", [True, False]),
        "max_leaf_nodes": hp.choice("ExtraTreesClassifier_max_leaf_nodes", ["None"]),
        "max_depth": hp.choice("ExtraTreesClassifier_max_depth", ["None"]),
        "min_weight_fraction_leaf": hp.choice(
            "ExtraTreesClassifier_min_weight_fraction_leaf", [0.0]
        ),
        "min_impurity_decrease": hp.choice(
            "ExtraTreesClassifier_min_impurity_decrease", [0.0]
        ),
    },
    {"model": "GaussianNB"},
    {
        "model": "GradientBoostingClassifier",
        "loss": hp.choice("GradientBoostingClassifier_loss", ["auto"]),
        "learning_rate": hp.loguniform(
            "GradientBoostingClassifier_learning_rate", np.log(0.01), np.log(1)
        ),
        "min_samples_leaf": scope.int(
            hp.loguniform(
                "GradientBoostingClassifier_min_samples_leaf", np.log(1), np.log(200)
            )
        ),
        "max_depth": hp.choice("GradientBoostingClassifier_max_depth", ["None"]),
        "max_leaf_nodes": scope.int(
            hp.loguniform(
                "GradientBoostingClassifier_max_leaf_nodes", np.log(3), np.log(2047)
            )
        ),
        "max_bins": hp.choice("GradientBoostingClassifier_max_bins", [255]),
        "l2_regularization": hp.loguniform(
            "GradientBoostingClassifier_l2_regularization", np.log(1e-10), np.log(1)
        ),
        "early_stop": hp.choice(
            "GradientBoostingClassifier_early_stop", ["off", "train", "valid"]
        ),
        "tol": hp.choice("GradientBoostingClassifier_tol", [1e-7]),
        "scoring": hp.choice("GradientBoostingClassifier_scoring", ["loss"]),
        "validation_fraction": hp.uniform(
            "GradientBoostingClassifier_validation_fraction", 0.01, 0.4
        ),
    },
    {
        "model": "KNearestNeighborsClassifier",
        "n_neighbors": scope.int(
            hp.quniform("KNearestNeighborsClassifier_n_neighbors", 1, 100, 1)
        ),
        "weights": hp.choice(
            "KNearestNeighborsClassifier_weights", ["uniform", "distance"]
        ),
        "p": hp.choice("KNearestNeighborsClassifier_p", [1, 2]),
    },
    {
        "model": "LDA",
        "shrinkage": hp.choice("LDA_shrinkage", [None, "auto", "manual"]),
        "shrinkage_factor": hp.uniform("LDA", 0.0, 1.0),
        "tol": hp.loguniform("LDA_tol", np.log(1e-5), np.log(1e-1)),
    },
    {
        "model": "LibLinear_SVC",
        # forbid penalty = 'l1' and loss = 'hinge'
        # forbid penalty = 'l2', loss = 'hinge' and dual = False
        # forbid penalty = 'l1' and dual = False
        "penalty": hp.choice("LibLinear_SVC_penalty", ["l2"]),
        "loss": hp.choice("LibLinear_SVC_loss", ["squared_hinge"]),
        "dual": hp.choice("LibLinear_SVC_dual", [False]),
        "tol": hp.loguniform("LibLinear_SVC_tol", np.log(1e-5), np.log(1e-1)),
        "C": hp.loguniform("LibLinear_SVC_C", np.log(0.03125), np.log(32768)),
        "multi_class": hp.choice("LibLinear_SVC_multi_class", ["ovr"]),
        "fit_intercept": hp.choice("LibLinear_SVC_fit_intercept", [True]),
        "intercept_scaling": hp.choice("LibLinear_SVC_intercept_scaling", [1]),
    },
    {
        "model": "LibSVM_SVC",
        # degree only selected when kernel = 'poly'
        # coef0 only selected when kernel = ['poly', 'sigmoid']
        "C": hp.loguniform("LibSVM_SVC_C", np.log(0.03125), np.log(32768)),
        "kernel": hp.choice("LibSVM_SVC_kernel", ["poly", "rbf", "sigmoid"]),
        "gamma": hp.loguniform("LibSVM_SVC_gamma", np.log(3.0517578125e-05), np.log(8)),
        "shrinking": hp.choice("LibSVM_SVC_shrinking", [True, False]),
        "tol": hp.loguniform("LibSVM_SVC_tol", np.log(1e-5), np.log(1e-1)),
        "max_iter": hp.choice("LibSVM_SVC_max_iter", [-1]),
        "degree": scope.int(hp.quniform("LibSVM_SVC_degree", 2, 5, 1)),
        "coef0": hp.uniform("LibSVM_SVC_coef0", -1, 1),
    },
    {
        "model": "MLPClassifier",
        "hidden_layer_depth": scope.int(
            hp.quniform("MLPClassifier_hidden_layer_depth", 1, 3, 1)
        ),
        "num_nodes_per_layer": scope.int(
            hp.loguniform("MLPClassifier_num_nodes_per_layer", np.log(16), np.log(264))
        ),
        "activation": hp.choice("MLPClassifier_activation", ["tanh", "relu"]),
        "alpha": hp.loguniform("MLPClassifier_alpha", np.log(1e-7), np.log(1e-1)),
        "learning_rate_init": hp.loguniform(
            "MLPClassifier_learning_rate_init", np.log(1e-4), np.log(0.5)
        ),
        "early_stopping": hp.choice("MLPClassifier_early_stopping", ["train", "valid"]),
        #'solver' : hp.choice('MLPClassifier_solver', ['lbfgs', 'sgd', 'adam']),
        # autosklearn must include _no_improvement_count, where only supported by 'sgd' and 'adam'
        "solver": hp.choice("MLPClassifier_solver", ["adam"]),
        "batch_size": hp.choice("MLPClassifier_batch_size", ["auto"]),
        "n_iter_no_change": hp.choice("MLPClassifier_n_iter_no_change", [32]),
        "tol": hp.choice("MLPClassifier_tol", [1e-4]),
        "shuffle": hp.choice("MLPClassifier_shuffle", [True]),
        "beta_1": hp.choice("MLPClassifier_beta_1", [0.9]),
        "beta_2": hp.choice("MLPClassifier_beta_2", [0.999]),
        "epsilon": hp.choice("MLPClassifier_epsilon", [1e-8]),
        "validation_fraction": hp.choice("MLPClassifier_validation_fraction", [0.1]),
    },
    {
        "model": "MultinomialNB",
        "alpha": hp.loguniform("MultinomialNB_alpha", np.log(1e-2), np.log(100)),
        "fit_prior": hp.choice("MultinomialNB_fit_prior", [True, False]),
    },
    {
        "model": "PassiveAggressive",
        "C": hp.loguniform("PassiveAggressive_C", np.log(1e-5), np.log(10)),
        "fit_intercept": hp.choice("PassiveAggressive_fit_intercept", [True]),
        "tol": hp.loguniform("PassiveAggressive_tol", np.log(1e-5), np.log(1e-1)),
        "loss": hp.choice("PassiveAggressive_loss", ["hinge", "squared_hinge"]),
        "average": hp.choice("PassiveAggressive_average", [True, False]),
    },
    {"model": "QDA", "reg_param": hp.uniform("QDA_reg_param", 0.0, 1.0)},
    {
        "model": "RandomForest",
        "criterion": hp.choice("RandomForest_criterion", ["gini", "entropy"]),
        "max_features": hp.uniform("RandomForest_max_features", 0.0, 1.0),
        "max_depth": hp.choice("RandomForest_max_depth", [None]),
        "min_samples_split": scope.int(
            hp.quniform("RandomForest_min_samples_split", 2, 20, 1)
        ),
        "min_samples_leaf": scope.int(
            hp.quniform("RandomForest_min_samples_leaf", 1, 20, 1)
        ),
        "min_weight_fraction_leaf": hp.choice(
            "RandomForest_min_weight_fraction_leaf", [0.0]
        ),
        "bootstrap": hp.choice("RandomForest_bootstrap", [True, False]),
        "max_leaf_nodes": hp.choice("RandomForest_max_leaf_nodes", [None]),
        "min_impurity_decrease": hp.choice("RandomForest_min_impurity_decrease", [0.0]),
    },
    {
        "model": "SGD",
        # l1_ratio only selected for penalty = 'elasticnet'
        # epsilon only selected for loss = 'modified_huber'
        # power_t only selected for learning_rate = 'invscaling'
        # eta0 only selected for learning_rate in ['constant', 'invscaling']
        "loss": hp.choice(
            "SGD_loss",
            ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
        ),
        "penalty": hp.choice("SGD_penalty", ["l1", "l2", "elasticnet"]),
        "alpha": hp.loguniform("SGD_alpha", np.log(1e-7), np.log(1e-1)),
        "fit_intercept": hp.choice("SGD_fit_intercept", [True]),
        "tol": hp.loguniform("SGD_tol", np.log(1e-5), np.log(1e-1)),
        "learning_rate": hp.choice(
            "SGD_learning_rate", ["constant", "optimal", "invscaling"]
        ),
        "l1_ratio": hp.loguniform("SGD_l1_ratio", np.log(1e-9), np.log(1)),
        "epsilon": hp.loguniform("SGD_epsilon", np.log(1e-5), np.log(1e-1)),
        "eta0": hp.loguniform("SGD_eta0", np.log(1e-7), np.log(1e-1)),
        "power_t": hp.uniform("SGD_power_t", 1e-5, 1),
        "average": hp.choice("SGD_average", [True, False]),
    },
]

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
    #"LibSVM_SVR": autosklearn.pipeline.components.regression.libsvm_svr.LibSVM_SVR,
    # "MLPRegressor": autosklearn.pipeline.components.regression.mlp.MLPRegressor,
    "RandomForest": autosklearn.pipeline.components.regression.random_forest.RandomForest,
    # "SGD": autosklearn.pipeline.components.regression.sgd.SGD,
} # LibSVM_SVR, MLP and SGD have problems of requiring inverse_transform of StandardScaler while having 1D array
  # https://github.com/automl/auto-sklearn/issues/1297

# regressor hyperparameters
# extract from autosklearn
regressor_hyperparameter = [
    {
        "model": "AdaboostRegressor",
        "n_estimators": scope.int(
            hp.quniform("AdaboostRegressor_n_estimators", 50, 500, 1)
        ),
        "learning_rate": hp.loguniform(
            "AdaboostRegressor_learning_rate", np.log(0.01), np.log(2)
        ),
        "loss": hp.choice(
            "AdaboostRegressor_algorithm", ["linear", "square", "exponential"]
        ),
        # for base_estimator of Decision Tree
        "max_depth": scope.int(hp.quniform("AdaboostRegressor_max_depth", 1, 10, 1)),
    },
    {
        "model": "ARDRegression",
        "n_iter": hp.choice("ARDRegression_n_iter", [300]),
        "tol": hp.loguniform("ARDRegression_tol", np.log(1e-5), np.log(1e-1)),
        "alpha_1": hp.loguniform("ARDRegression_alpha_1", np.log(1e-10), np.log(1e-3)),
        "alpha_2": hp.loguniform("ARDRegression_alpha_2", np.log(1e-10), np.log(1e-3)),
        "lambda_1": hp.loguniform(
            "ARDRegression_lambda_1", np.log(1e-10), np.log(1e-3)
        ),
        "lambda_2": hp.loguniform(
            "ARDRegression_lambda_2", np.log(1e-10), np.log(1e-3)
        ),
        "threshold_lambda": hp.loguniform(
            "ARDRegression_threshold_lambda", np.log(1e3), np.log(1e5)
        ),
        "fit_intercept": hp.choice("ARDRegression_fit_intercept", [True]),
    },
    {
        "model": "DecisionTree",
        "criterion": hp.choice(
            "DecisionTree_criterion", ["mse", "friedman_mse", "mae"]
        ),
        "max_features": hp.choice("DecisionTree_max_features", [1.0]),
        "max_depth_factor": hp.uniform("DecisionTree_max_depth_factor", 0.0, 2.0),
        "min_samples_split": scope.int(
            hp.quniform("DecisionTree_min_samples_split", 2, 20, 1)
        ),
        "min_samples_leaf": scope.int(
            hp.quniform("DecisionTree_min_samples_leaf", 1, 20, 1)
        ),
        "min_weight_fraction_leaf": hp.choice(
            "DecisionTree_min_weight_fraction_leaf", [0.0]
        ),
        "max_leaf_nodes": hp.choice("DecisionTree_max_leaf_nodes", [None]),
        "min_impurity_decrease": hp.choice("DecisionTree_min_impurity_decrease", [0.0]),
    },
    {
        "model": "ExtraTreesRegressor",
        "criterion": hp.choice(
            "ExtraTreesRegressor_criterion", ["mse", "friedman_mse", "mae"]
        ),
        "min_samples_leaf": scope.int(hp.quniform(
            "ExtraTreesRegressor_min_samples_leaf", 1, 20, 1
        )),
        "min_samples_split": scope.int(hp.quniform(
            "ExtraTreesRegressor_min_samples_split", 2, 20, 1
        )),
        "max_features": hp.uniform("ExtraTreesRegressor_max_features", 0.1, 1.0),
        "bootstrap": hp.choice("ExtraTreesRegressor_bootstrap", [True, False]),
        "max_leaf_nodes": hp.choice("ExtraTreesRegressor_max_leaf_nodes", [None]),
        "max_depth": hp.choice("ExtraTreesRegressor_max_depth", [None]),
        "min_weight_fraction_leaf": hp.choice(
            "ExtraTreesRegressor_min_weight_fraction_leaf", [0.0]
        ),
        "min_impurity_decrease": hp.choice(
            "ExtraTreesRegressor_min_impurity_decrease", [0.0]
        ),
    },
    {
        "model": "GaussianProcess",
        "alpha": hp.loguniform("GaussianProcess_alpha", np.log(1e-14), np.log(1)),
        "thetaL": hp.loguniform("GaussianProcess_thetaL", np.log(1e-10), np.log(1e-3)),
        "thetaU": hp.loguniform("GaussianProcess_thetaU", np.log(1), np.log(1e5)),
    },
    {
        "model": "GradientBoosting",
        # n_iter_no_change only selected for early_stop in ['valid', 'train']
        # validation_fraction only selected for early_stop = 'valid'
        "loss": hp.choice("GradientBoosting_loss", ["least_squares"]),
        "learning_rate": hp.loguniform(
            "GradientBoosting_learning_rate", np.log(0.01), np.log(1)
        ),
        "min_samples_leaf": scope.int(
            hp.loguniform(
                "GradientBoosting_min_samples_leaf", np.log(1), np.log(200)
            )
        ),
        "max_depth": hp.choice("GradientBoosting_max_depth", [None]),
        "max_leaf_nodes": scope.int(
            hp.loguniform("GradientBoosting_max_leaf_nodes", np.log(3), np.log(2047))
        ),
        "max_bins": hp.choice("GradientBoosting_max_bins", [255]),
        "l2_regularization": hp.loguniform(
            "GradientBoosting_l2_regularization", np.log(1e-10), np.log(1)
        ),
        "early_stop": hp.choice(
            "GradientBoosting_early_stop", ["off", "train", "valid"]
        ),
        "tol": hp.choice("GradientBoosting_tol", [1e-7]),
        "scoring": hp.choice("GradientBoosting_scoring", ["loss"]),
        "n_iter_no_change": scope.int(
            hp.quniform("GradientBoosting_n_iter_no_change", 1, 20, 1)
        ),
        "validation_fraction": hp.uniform(
            "GradientBoosting_validation_fraction", 0.01, 0.4
        ),
    },
    {
        "model": "KNearestNeighborsRegressor",
        "n_neighbors": scope.int(
            hp.quniform("KNearestNeighborsRegressor_n_neighbors", 1, 100, 1)
        ),
        "weights": hp.choice(
            "KNearestNeighborsRegressor_weights", ["uniform", "distance"]
        ),
        "p": hp.choice("KNearestNeighborsRegressor_p", [1, 2]),
    },
    {
        "model": "LibLinear_SVR",
        # forbid loss = 'epsilon_insensitive' and dual = False
        "epsilon": hp.loguniform("LibLinear_SVR_tol", np.log(0.001), np.log(1)),
        "loss": hp.choice(
            "LibLinear_SVR__loss", ["squared_epsilon_insensitive"],
        ),
        "dual": hp.choice("LibLinear_SVR__dual", [False]),
        "tol": hp.loguniform("LibLinear_SVR__tol", np.log(1e-5), np.log(1e-1)),
        "C": hp.loguniform("LibLinear_SVR__C", np.log(0.03125), np.log(32768)),
        "fit_intercept": hp.choice("LibLinear_SVR__fit_intercept", [True]),
        "intercept_scaling": hp.choice("LibLinear_SVR__intercept_scaling", [1]),
    },
    {
        "model": "LibSVM_SVR",
        # degree only selected for kernel in ['poly', 'rbf', 'sigmoid']
        # gamma only selected for kernel in ['poly', 'rbf']
        # coef0 only selected for kernel in ['poly', 'sigmoid']
        "kernel": hp.choice("LibSVM_SVR_kernel", ["linear", "poly", "rbf", "sigmoid"]),
        "C": hp.loguniform("LibSVM_SVR__C", np.log(0.03125), np.log(32768)),
        "epsilon": hp.uniform("LibSVM_SVR_epsilon", 1e-5, 1),
        "tol": hp.loguniform("LibSVM_SVR__tol", np.log(1e-5), np.log(1e-1)),
        "shrinking": hp.choice("LibSVM_SVR_shrinking", [True, False]),
        "degree": scope.int(hp.quniform("LibSVM_SVR_degree", 2, 5, 1)),
        "gamma": hp.loguniform("LibSVM_SVR_gamma", np.log(3.0517578125e-5), np.log(8)),
        "coef0": hp.uniform("LibSVM_SVR_coef0", -1, 1),
        "max_iter": hp.choice("LibSVM_SVR_max_iter", [-1]),
    },
    {
        "model": "MLPRegressor",
        # validation_fraction only selected for early_stopping = 'valid'
        "hidden_layer_depth": scope.int(
            hp.quniform("MLPRegressor_hidden_layer_depth", 1, 3, 1)
        ),
        "num_nodes_per_layer": scope.int(
            hp.loguniform("MLPRegressor_num_nodes_per_layer", np.log(16), np.log(264))
        ),
        "activation": hp.choice("MLPRegressor_activation", ["tanh", "relu"]),
        "alpha": hp.loguniform("MLPRegressor_alpha", np.log(1e-7), np.log(1e-1)),
        "learning_rate_init": hp.loguniform(
            "MLPRegressor_learning_rate_init", np.log(1e-4), np.log(0.5)
        ),
        "early_stopping": hp.choice("MLPRegressor_early_stopping", ["valid", "train"]),
        "solver": hp.choice("MLPRegressor_solver", ["adam"]),
        "batch_size": hp.choice("MLPRegressor_batch_size", ["auto"]),
        "n_iter_no_change": hp.choice("MLPRegressor_n_iter_no_change", [32]),
        "tol": hp.choice("MLPRegressor_tol", [1e-4]),
        "shuffle": hp.choice("MLPRegressor_shuffle", [True]),
        "beta_1": hp.choice("MLPRegressor_beta_1", [0.9]),
        "beta_2": hp.choice("MLPRegressor_beta_2", [0.999]),
        "epsilon": hp.choice("MLPRegressor_epsilon", [1e-8]),
        "validation_fraction": hp.choice("MLPRegressor_validation_fraction", [0.1]),
    },
    {
        "model": "RandomForest",
        "criterion": hp.choice(
            "RandomForest_criterion", ["mse", "friedman_mse", "mae"]
        ),
        "max_features": hp.uniform("RandomForest_max_features", 0.1, 1.0),
        "max_depth": hp.choice("RandomForest_max_depth", [None]),
        "min_samples_split": scope.int(
            hp.quniform("RandomForest_min_samples_split", 2, 20, 1)
        ),
        "min_samples_leaf": scope.int(
            hp.quniform("RandomForest_min_samples_leaf", 1, 20, 1)
        ),
        "min_weight_fraction_leaf": hp.choice(
            "RandomForest_min_weight_fraction_leaf", [0.0]
        ),
        "bootstrap": hp.choice("RandomForest_bootstrap", [True, False]),
        "max_leaf_nodes": hp.choice("RandomForest_max_leaf_nodes", [None]),
        "min_impurity_decrease": hp.choice("RandomForest_min_impurity_decrease", [0.0]),
    },
    {
        "model": "SGD",
        # l1_ratio only selected for penalty = 'elasticnet'
        # epsilon only selected for loss in ['huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
        # eta0 only selected for learning_rate in ['constant', 'invscaling']
        # power_t only selected for learning_rate = 'invscaling'
        "loss": hp.choice(
            "SGD_loss",
            [
                "squared_loss",
                "huber",
                "epsilon_insensitive",
                "squared_epsilon_insensitive",
            ],
        ),
        "penalty": hp.choice("SGD_penalty", ["l1", "l2", "elasticnet"]),
        "alpha": hp.loguniform("SGD_alpha", np.log(1e-7), np.log(1e-1)),
        "fit_intercept": hp.choice("SGD_fit_intercept", [True]),
        "tol": hp.loguniform("SGD_tol", np.log(1e-5), np.log(1e-1)),
        "learning_rate": hp.choice(
            "SGD_learning_rate", ["constant", "optimal", "invscaling"]
        ),
        "l1_ratio": hp.loguniform("SGD_l1_ratio", np.log(1e-9), np.log(1.0)),
        "epsilon": hp.loguniform("SGD_epsilon", np.log(1e-5), np.log(1e-1)),
        "power_t": hp.uniform("SGD_power_t", 1e-5, 1),
        "average": hp.choice("SGD_average", [True, False]),
    },
]
