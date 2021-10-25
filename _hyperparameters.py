import numpy as np
import autosklearn
import autosklearn.pipeline.components.classification
from hyperopt import hp

# hyperparameters for AutoML pipeline

# encoder
encoder_hyperparameter = [
    {
        'encoder' : 'DataEncoding'
    }
]

# imputer
imputer_hyperparameter = [
    {
        'imputer' : 'SimpleImputer',
        'method' : hp.choice('SimpleImputer_method', ['mean', 'zero', 'median', 'most frequent'])
    },
    {
        'imputer' : 'DummyImputer'
    },
    {
        'imputer' : 'JointImputer'
    },
    {
        'imputer' : 'ExpectationMaximization',
        'iterations' : hp.quniform('ExpectationMaximization_iterations', 10, 100, 1),
        'threshold' : hp.uniform('ExpectationMaximization_iterations_threshold', 1e-5, 1)
    },
    {
        'imputer' : 'KNNImputer'
    },
    {
        'imputer' : 'MissForestImputer'
    },
    {
        'imputer' : 'MICE',
        'cycle' : hp.quniform('MICE_cycle', 5, 20, 1)
    },
    {
        'imputer' : 'GAIN'
    }
]

# scaling
scaling_hyperparameter = [
    {
        'scaling' : 'NoScaling'
    },
    {
        'scaling' : 'Standardize'
    },
    {
        'scaling' : 'Normalize'
    },
    {
        'scaling' : 'RobustScale'
    },
    {
        'scaling' : 'MinMaxScale'
    },
    {
        'scaling' : 'Winsorization'
    }
]

# balancing
balancing_hyperparameter = [
    {
        'balancing' : 'no_processing'
    },
    {
        'balancing' : 'SimpleRandomOverSampling',
        'imbalance_threshold' : hp.uniform('SimpleRandomOverSampling_imbalance_threshold', \
            0.5, 1)
    },
    {
        'balancing' : 'SimpleRandomUnderSampling',
        'imbalance_threshold' : hp.uniform('SimpleRandomUnderSampling_imbalance_threshold', \
            0.5, 1)
    },
    {
        'balancing' : 'TomekLink',
        'imbalance_threshold' : hp.uniform('TomekLink_imbalance_threshold', \
            0.5, 1)
    },
    {
        'balancing' : 'EditedNearestNeighbor',
        'imbalance_threshold' : hp.uniform('EditedNearestNeighbor_imbalance_threshold', \
            0.5, 1)
    },
    {
        'balancing' : 'CondensedNearestNeighbor',
        'imbalance_threshold' : hp.uniform('CondensedNearestNeighbor_imbalance_threshold', \
            0.5, 1)
    },
    {
        'balancing' : 'OneSidedSelection',
        'imbalance_threshold' : hp.uniform('OneSidedSelection_imbalance_threshold', \
            0.5, 1)
    },
    {
        'balancing' : 'CNN_TomekLink',
        'imbalance_threshold' : hp.uniform('CNN_TomekLink_imbalance_threshold', \
            0.5, 1)
    },
    {
        'balancing' : 'Smote',
        'imbalance_threshold' : hp.uniform('Smote_imbalance_threshold', 0.5, 1)
    },
    {
        'balancing' : 'Smote_TomekLink',
        'imbalance_threshold' : hp.uniform('Smote_TomekLink_imbalance_threshold', \
            0.5, 1)
    },
    {
        'balancing' : 'Smote_ENN',
        'imbalance_threshold' : hp.uniform('Smote_ENN_imbalance_threshold', \
            0.5, 1)
    }
]

# feature_selection
feature_selection_hyperparameter = [
    {
        'feature_selection' : 'no_processing'
    },
    {
        'feature_selection' : 'LDASelection'
    },
    {
        'feature_selection' : 'PCA_FeatureSelection'
    },
    {
        'feature_selection' : 'RBFSampler'
    },
    {
        'feature_selection' : 'FeatureFilter'
    },
    {
        'feature_selection' : 'ASFFS'
    },
    {
        'feature_selection' : 'GeneticAlgorithm'
    },
    {
        'feature_selection' : 'extra_trees_preproc_for_classification'
    },
    {
        'feature_selection' : 'fast_ica'
    },
    {
        'feature_selection' : 'feature_agglomeration'
    },
    {
        'feature_selection' : 'kernel_pca'
    },
    {
        'feature_selection' : 'kitchen_sinks'
    },
    {
        'feature_selection' : 'liblinear_svc_preprocessor'
    },
    {
        'feature_selection' : 'nystroem_sampler'
    },
    {
        'feature_selection' : 'pca'
    },
    {
        'feature_selection' : 'polynomial'
    },
    {
        'feature_selection' : 'random_trees_embedding'
    },
    {
        'feature_selection' : 'select_percentile_classification'
    },
    {
        'feature_selection' : 'select_rates_classification'
    },
    {
        'feature_selection' : 'truncatedSVD'
    }
]

# classifiers
classifiers = {
    'AdaboostClassifier' : autosklearn.pipeline.components.classification.adaboost.AdaboostClassifier, # autosklearn classifiers
    'BernoulliNB' : autosklearn.pipeline.components.classification.bernoulli_nb.BernoulliNB,
    'DecisionTree' : autosklearn.pipeline.components.classification.decision_tree.DecisionTree,
    'ExtraTreesClassifier' : autosklearn.pipeline.components.classification.extra_trees.ExtraTreesClassifier,
    'GaussianNB' : autosklearn.pipeline.components.classification.gaussian_nb.GaussianNB,
    'GradientBoostingClassifier' : autosklearn.pipeline.components.classification.gradient_boosting.GradientBoostingClassifier,
    'KNearestNeighborsClassifier' : autosklearn.pipeline.components.classification.k_nearest_neighbors.KNearestNeighborsClassifier,
    'LDA' : autosklearn.pipeline.components.classification.lda.LDA,
    'LibLinear_SVC' : autosklearn.pipeline.components.classification.liblinear_svc.LibLinear_SVC,
    'LibSVM_SVC' : autosklearn.pipeline.components.classification.libsvm_svc.LibSVM_SVC,
    'MLPClassifier' : autosklearn.pipeline.components.classification.mlp.MLPClassifier,
    'MultinomialNB' : autosklearn.pipeline.components.classification.multinomial_nb.MultinomialNB,
    'PassiveAggressive' : autosklearn.pipeline.components.classification.passive_aggressive.PassiveAggressive,
    'QDA' : autosklearn.pipeline.components.classification.qda.QDA,
    'RandomForest' : autosklearn.pipeline.components.classification.random_forest.RandomForest,
    'SGD' : autosklearn.pipeline.components.classification.sgd.SGD
}

# classifier hyperparameters
classifier_hyperparameter = [
    {
        'model' : 'AdaboostClassifier',
        'n_estimators' : hp.quniform('AdaboostClassifier_n_estimators', 10, 100, 1), 
        'learning_rate' : hp.uniform('AdaboostClassifier_learning_rate', 0.00001, 1), 
        'algorithm' : hp.choice('AdaboostClassifier_algorithm', ['SAMME', 'SAMME.R']), 
        # for base_estimator of Decision Tree
        'max_depth' : hp.quniform('AdaboostClassifier_max_depth', 2, 10, 1)
    },
    {
        'model' : 'BernoulliNB',
        'alpha' : hp.loguniform('BernoulliNB_alpha', -3, 1),
        'fit_prior' : hp.choice('BernoulliNB_fit_prior', [True, False])
    },
    {
        'model' : 'DecisionTree',
        'criterion' : hp.choice('DecisionTree_criterion', ['gini', 'entropy']), 
        'max_features' : hp.uniform('DecisionTree_max_features', 0, 1), 
        'max_depth_factor' : hp.uniform('DecisionTree_max_depth_factor', 0, 1),
        'min_samples_split' : hp.quniform('DecisionTree_min_samples_split', 2, 10, 1), 
        'min_samples_leaf' : hp.quniform('DecisionTree_min_samples_leaf', 1, 10, 1), 
        'min_weight_fraction_leaf' : hp.uniform('DecisionTree_min_weight_fraction_leaf', 0, 0.5),
        'max_leaf_nodes' : hp.quniform('DecisionTree_max_leaf_nodes', 1, 100000, 1), 
        'min_impurity_decrease' : hp.uniform('DecisionTree_min_impurity_decrease', 0, 1)
    },
    {
        'model' : 'ExtraTreesClassifier',
        'criterion' : hp.choice('ExtraTreesClassifier_criterion', ['gini', 'entropy']), 
        'min_samples_leaf' : hp.quniform('ExtraTreesClassifier_min_samples_leaf', 1, 10, 1),
        'min_samples_split' : hp.quniform('ExtraTreesClassifier_min_samples_split', 2, 10, 1),  
        'max_features' : hp.choice('ExtraTreesClassifier_max_features', [None, 'auto', 'sqrt', 'log2']), 
        'bootstrap' : hp.choice('ExtraTreesClassifier_bootstrap', [True, False]), 
        'max_leaf_nodes' : hp.quniform('ExtraTreesClassifier_max_leaf_nodes', 1, 100000, 1),
        'max_depth' : hp.quniform('ExtraTreesClassifier_max_depth', 2, 10, 1), 
        'min_weight_fraction_leaf' : hp.uniform('ExtraTreesClassifier_min_weight_fraction_leaf', 0, 1), 
        'min_impurity_decrease' : hp.uniform('ExtraTreesClassifier_min_impurity_decrease', 0, 1)
    },
    {
        'model' : 'GaussianNB'
    },
    {
        'model' : 'GradientBoostingClassifier',
        'loss' : hp.choice('GradientBoostingClassifier_loss', ['auto', 'binary_crossentropy']), 
        # 'categorical_crossentropy' only for multi-class classification
        'learning_rate' : hp.loguniform('GradientBoostingClassifier_learning_rate', -3, 1), 
        'min_samples_leaf' : hp.quniform('GradientBoostingClassifier_min_samples_leaf', 1, 30, 1), 
        'max_depth' : hp.quniform('GradientBoostingClassifier_max_depth', 2, 50, 1),
        'max_leaf_nodes' : hp.quniform('GradientBoostingClassifier_max_leaf_nodes', 1, 100000, 1), 
        'max_bins' : hp.quniform('GradientBoostingClassifier_max_bins', 1, 255, 1), 
        'l2_regularization' : hp.uniform('GradientBoostingClassifier_l2_regularization', 0, 10), 
        'early_stop' : hp.choice('GradientBoostingClassifier_early_stop', ['off', 'train', 'valid']), 
        'tol' : hp.loguniform('GradientBoostingClassifier_tol', -10, 1), 
        'scoring' : hp.choice('GradientBoostingClassifier_scoring', ['loss', 'accuracy', 'roc_auc'])
    },
    {
        'model' : 'KNearestNeighborsClassifier',
        'n_neighbors' : hp.choice('KNearestNeighborsClassifier_n_neighbors', np.arange(1, 21, dtype = int)), 
        'weights' : hp.choice('KNearestNeighborsClassifier_weights', ['uniform', 'distance']), 
        'p' : hp.quniform('KNearestNeighborsClassifier_p', 1, 5, 1)
    },
    {
        'model' : 'LDA',
        'shrinkage' : hp.choice('LDA_shrinkage', [None, 'auto', 'manual']), 
        'tol' : hp.uniform('LDA_tol', 1e-10, 1)
    },
    ###################################################################
    {
        'model' : 'LibLinear_SVC',
        'penalty' : hp.choice('LibLinear_SVC_penalty', ['l1', 'l2']), # conditional 'l1' and 'hinge' not supported
        'loss' : hp.choice('LibLinear_SVC_loss', ['hinge', 'squared_hinge']), 
        'dual' : hp.choice('LibLinear_SVC_dual', [True, False]), 
        'tol' : hp.loguniform('LibLinear_SVC_tol', -10, 1), 
        'C' : hp.loguniform('LibLinear_SVC_C', -5, 1), 
        'multi_class' : hp.choice('LibLinear_SVC_multi_class', ['ovr', 'crammer_singer']),
        'fit_intercept' : hp.choice('LibLinear_SVC_fit_intercept', [True, False]), 
        'intercept_scaling' : hp.loguniform('LibLinear_SVC_intercept_scaling', -5, 1)
    },
    {
        'model' : 'LibSVM_SVC',
        'C' : hp.loguniform('LibSVM_SVC_C', -5, 1), 
        'kernel' : hp.choice('LibSVM_SVC_kernel', ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']), # conditional, precomputed can only be used for n * n kernel matrix
        'gamma' : hp.uniform('LibSVM_SVC_gamma', 0, 1), 
        'shrinking' : hp.choice('LibSVM_SVC_shrinking', [True, False]), 
        'tol' : hp.loguniform('LibSVM_SVC_tol', -10, 1), 
        'max_iter' : hp.quniform('LibSVM_SVC_max_iter', -1, 1e6, 1)
    },
    {
        'model' : 'MLPClassifier',
        'hidden_layer_depth' : hp.quniform('MLPClassifier_hidden_layer_depth', 1, 10, 1), 
        'num_nodes_per_layer' : hp.quniform('MLPClassifier_num_nodes_per_layer', 1, 20, 1), 
        'activation' : hp.choice('MLPClassifier_activation', ['identity', 'logistic', 'tanh', 'relu']), 
        'alpha' : hp.loguniform('MLPClassifier_alpha', -6, 1),
        'learning_rate_init' : hp.loguniform('MLPClassifier_learning_rate_init', -6, 1), 
        'early_stopping' : hp.choice('MLPClassifier_early_stopping', ['train', 'valid']), 
        #'solver' : hp.choice('MLPClassifier_solver', ['lbfgs', 'sgd', 'adam']), 
        # autosklearn must include _no_improvement_count, where only supported by 'sgd' and 'adam'
        'solver' : hp.choice('MLPClassifier_solver', ['sgd', 'adam']), 
        'batch_size' : hp.quniform('MLPClassifier_batch_size', 2, 200, 1),
        'n_iter_no_change' : hp.quniform('MLPClassifier_n_iter_no_change', 1, 20, 1), 
        'tol' : hp.loguniform('MLPClassifier_tol', -10, 1),
        'shuffle' : hp.choice('MLPClassifier_shuffle', [True, False]), 
        'beta_1' : hp.uniform('MLPClassifier_beta_1', 0, 0.999), 
        'beta_2' : hp.uniform('MLPClassifier_beta_2', 0, 0.999), 
        'epsilon' : hp.loguniform('MLPClassifier_epsilon', -10, 10)
    },
    ####################################################################
    {
        'model' : 'MultinomialNB',
        'alpha' : hp.loguniform('MultinomialNB_alpha', -3, 1),
        'fit_prior' : hp.choice('MultinomialNB_fit_prior', [True, False])
    },
    {
        'model' : 'PassiveAggressive',
        'C' : hp.loguniform('PassiveAggressive_C', -3, 1), 
        'fit_intercept' : hp.choice('PassiveAggressive_fit_intercept', [True, False]), 
        'tol' : hp.loguniform('PassiveAggressive_tol', -10, 1), 
        'loss' : hp.choice('PassiveAggressive_loss', ['hinge', 'squared_hinge']), 
        'average' : hp.choice('PassiveAggressive_average', [True, False])
    },
    {
        'model' : 'QDA',
        'reg_param' : hp.uniform('QDA_reg_param', 0, 1)
    },
    {
        'model' : 'RandomForest',
        'criterion' : hp.choice('RandomForest_criterion', ['gini', 'entropy']), 
        'max_features' : hp.choice('RandomForest_max_features', ['auto', 'sqrt', 'log2', \
            hp.uniform('RandomForest_max_features_float', 0, 1)]),
        'max_depth' : hp.quniform('RandomForest_max_depth', 2, 10, 1), 
        'min_samples_split' : hp.quniform('RandomForest_min_samples_split', 2, 10, 1), 
        'min_samples_leaf' : hp.quniform('RandomForest_min_samples_leaf', 1, 30, 1),
        'min_weight_fraction_leaf' : hp.uniform('RandomForest_min_weight_fraction_leaf', 0, 0.5), 
        'bootstrap' : hp.choice('RandomForest_bootstrap', [True, False]), 
        'max_leaf_nodes' : hp.quniform('RandomForest_max_leaf_nodes', 1, 100000, 1),
        'min_impurity_decrease' : hp.uniform('RandomForest_min_impurity_decrease', 0, 1)
    },
    {
        'model' : 'SGD',
        'loss' : hp.choice('SGD_loss', ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']), 
        'penalty' : hp.choice('SGD_penalty', ['l1', 'l2']), 
        'alpha' : hp.loguniform('SGD_alpha', -6, 1), 
        'fit_intercept' : hp.choice('SGD_fit_intercept', [True, False]), 
        'tol' : hp.loguniform('SGD_tol', -10, 1),
        'learning_rate' : hp.choice('SGD_learning_rate', ['constant', 'optimal', 'invscaling', 'adaptive'])
    }
]