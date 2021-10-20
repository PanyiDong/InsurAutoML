from unicodedata import category
import warnings
import numpy as np
import pandas as pd
import scipy
import scipy.stats
import sklearn
import mlflow
import hyperopt
from hyperopt import fmin, hp, rand, tpe, atpe, Trials, SparkTrials, \
    space_eval, STATUS_OK, pyll
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

import My_AutoML

'''
Classifiers/Hyperparameters from sklearn:
1. AdaBoost: n_estimators, learning_rate, algorithm, max_depth
2. Bernoulli naive Bayes: alpha, fit_prior
3. Decision Tree: criterion, max_features, max_depth_factor, min_samples_split, 
            min_samples_leaf, min_weight_fraction_leaf, max_leaf_nodes, min_impurity_decrease
4. Extra Trees: criterion, min_samples_leaf, min_samples_split,  max_features, 
            bootstrap, max_leaf_nodes, max_depth, min_weight_fraction_leaf, min_impurity_decrease
5. Gaussian naive Bayes
6. Gradient boosting: loss, learning_rate, min_samples_leaf, max_depth,
            max_leaf_nodes, max_bins, l2_regularization, early_stop, tol, scoring
7. KNN: n_neighbors, weights, p
8. LDA: shrinkage, tol
9. Linear SVC (LibLinear): penalty, loss, dual, tol, C, multi_class, 
            fit_intercept, intercept_scaling
10. kernel SVC (LibSVM): C, kernel, gamma, shrinking, tol, max_iter
11. MLP (Multilayer Perceptron): hidden_layer_depth, num_nodes_per_layer, activation, alpha,
            learning_rate_init, early_stopping, solver, batch_size, n_iter_no_change, tol,
            shuffle, beta_1, beta_2, epsilon
12. Multinomial naive Bayes: alpha, fit_prior
13. Passive aggressive: C, fit_intercept, tol, loss, average
14. QDA: reg_param
15. Random forest: criterion, max_features, max_depth, min_samples_split, 
            min_samples_leaf, min_weight_fraction_leaf, bootstrap, max_leaf_nodes
16. SGD (Stochastic Gradient Descent): loss, penalty, alpha, fit_intercept, tol,
            learning_rate
'''

# Auto binary classifier
class AutoClassifier() :

    '''
    Perform model selection and hyperparameter optimization for classification tasks
    using sklearn models, predefine hyperparameters

    Parameters
    ----------
    timeout: total time limit for the job in seconds, default = 360
    
    max_evals: Maximum number of function evaluations allowd, default = 32
    
    models: Models selected for the job, default = 'auto'
    support ('AdaboostClassifier', 'BernoulliNB', 'DecisionTree', 'ExtraTreesClassifier',
            'GaussianNB', 'GradientBoostingClassifier', 'KNearestNeighborsClassifier',
            'LDA', 'LibLinear_SVC', 'LibSVM_SVC', 'MLPClassifier', 'MultinomialNB',
            'PassiveAggressive', 'QDA', 'RandomForest',  'SGD')
    'auto' will select all default models, or a list of models used
    
    test_size: Test percentage used to evaluate the perforamance, default = 0.15
    
    method: model selection/hyperparameter optimization methods, default = 'Bayeisan'
    
    algo: Search algorithm, default = 'tpe'
    support (rand, tpe, atpe)
    
    spark_trials: Whether to use SparkTrials, default = True

    progressbar: Whether to show progress bar, default = False
    
    seed: random seed, default = 1
    '''

    def __init__(
        self,
        timeout = 360,
        max_evals = 32,
        models = 'auto',
        test_size = 0.15,
        method = 'Bayeisan',
        algo = 'tpe',
        spark_trials = True,
        progressbar = False,
        seed = 1
    ) : 
        self.timeout = timeout
        self.max_evals = max_evals
        self.models = models
        self.test_size = test_size
        self.method = method
        self.algo = algo
        self.spark_trials = spark_trials
        self.progressbar = progressbar
        self.seed = seed

        # autosklearn classifiers
        from autosklearn.pipeline.components.classification.adaboost import AdaboostClassifier
        from autosklearn.pipeline.components.classification.bernoulli_nb import BernoulliNB
        from autosklearn.pipeline.components.classification.decision_tree import DecisionTree
        from autosklearn.pipeline.components.classification.extra_trees import ExtraTreesClassifier
        from autosklearn.pipeline.components.classification.gaussian_nb import GaussianNB
        from autosklearn.pipeline.components.classification.k_nearest_neighbors import KNearestNeighborsClassifier
        from autosklearn.pipeline.components.classification.lda import LDA
        from autosklearn.pipeline.components.classification.liblinear_svc import LibLinear_SVC
        from autosklearn.pipeline.components.classification.libsvm_svc import LibSVM_SVC
        from autosklearn.pipeline.components.classification.mlp import MLPClassifier
        from autosklearn.pipeline.components.classification.multinomial_nb import MultinomialNB
        from autosklearn.pipeline.components.classification.passive_aggressive import PassiveAggressive
        from autosklearn.pipeline.components.classification.qda import QDA
        from autosklearn.pipeline.components.classification.random_forest import RandomForest
        from autosklearn.pipeline.components.classification.sgd import SGD

        # modifications required
        # autosklearn.GradientBoostingClassifier set validation_fraction = None, 
        # which result in early_stop = 'valid' errors
        # autosklearn.MLPRegressor set validation_fraction = None, 
        # which result in early_stopping = 'valid' errors
        class GradientBoostingClassifier() :

            def __init__(
                self,
                loss, 
                learning_rate, 
                min_samples_leaf, 
                max_depth,
                max_leaf_nodes, 
                max_bins, 
                l2_regularization, 
                early_stop, 
                tol, 
                scoring,
                max_iter = 512,
                n_iter_no_change = 0, 
                validation_fraction = 0.1
            ) :
                self.loss = loss
                self.learning_rate = learning_rate
                self.min_samples_leaf = min_samples_leaf
                self.max_depth = max_depth
                self.max_leaf_nodes = max_leaf_nodes
                self.max_bins = max_bins
                self.l2_regularization = l2_regularization
                self.early_stop = early_stop
                self.tol = tol
                self.scoring = scoring
                self.max_iter = max_iter
                self.n_iter_no_change = n_iter_no_change
                self.validation_fraction = validation_fraction

                from autosklearn.pipeline.components.classification.gradient_boosting \
                    import GradientBoostingClassifier
                self.clf = GradientBoostingClassifier(
                    self.loss,
                    self.learning_rate,
                    self.min_samples_leaf,
                    self.max_depth,
                    self.max_leaf_nodes,
                    self.max_bins,
                    self.l2_regularization,
                    self.early_stop,
                    self.tol,
                    self.scoring,
                    self.n_iter_no_change,
                    validation_fraction = self.validation_fraction
                )
    
            def fit(self, X, y) :

                return self.clf.fit(X, y)
    
            def predict(self, X) :

                return self.clf.predict(X)

            def predict_proba(self, X) :

                return self.clf.predict_proba(X)

        # all encoders avaiable
        self._all_encoders = My_AutoML.encoders

        # all hyperparameters for encoders
        self._all_encoders_hyperparameters = [
            {
                'encoder' : 'DataEncoding'
            }
        ]
        
        # all imputers available
        self._all_imputers = My_AutoML.imputers

        # all hyperparemeters for imputers
        self._all_imputers_hyperparameters = [
            {
                'imputer' : 'SimpleImputer'
            },
            {
                'imputer' : 'DummyImputer'
            },
            {
                'imputer' : 'JointImputer'
            },
            {
                'imputer' : 'ExpectationMaximization'
            },
            {
                'imputer' : 'KNNImputer'
            },
            {
                'imputer' : 'MissForestImputer'
            },
            {
                'imputer' : 'MICE'
            },
            {
                'imputer' : 'GAIN'
            }
        ]

        # all scalings avaiable
        self._all_scalings = My_AutoML.scalings

        # all hyperparameters for scalings
        self._all_scalings_hyperparameters = [
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

        # all balancings available
        self._all_balancings = My_AutoML.imbalance

        # all hyperparameters for balancing methods
        self._all_balancings_hyperparameters = [
            {
                'balancing' : 'no_processing'
            },
            {
                'balancing' : 'SimpleRandomOverSampling'
            },
            {
                'balancing' : 'SimpleRandomUnderSampling'
            },
            {
                'balancing' : 'TomekLink'
            },
            {
                'balancing' : 'EditedNearestNeighbor'
            },
            {
                'balancing' : 'CondensedNearestNeighbor'
            },
            {
                'balancing' : 'OneSidedSelection'
            },
            {
                'balancing' : 'CNN_TomekLink'
            },
            {
                'balancing' : 'Smote'
            },
            {
                'balancing' : 'Smote_TomekLink'
            },
            {
                'balancing' : 'Smote_ENN'
            }
        ]

        # all feature selections available
        self._all_feature_selection = My_AutoML.feature_selection

        # all hyperparameters for feature selections
        self._all_feature_selection_hyperparameters = [
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
        
        # all classfication models available
        self._all_models = {
            'AdaboostClassifier' : AdaboostClassifier,
            'BernoulliNB' : BernoulliNB,
            'DecisionTree' : DecisionTree,
            'ExtraTreesClassifier' : ExtraTreesClassifier,
            'GaussianNB' : GaussianNB,
            'GradientBoostingClassifier' : GradientBoostingClassifier,
            'KNearestNeighborsClassifier' : KNearestNeighborsClassifier,
            'LDA' : LDA,
            'LibLinear_SVC' : LibLinear_SVC,
            'LibSVM_SVC' : LibSVM_SVC,
            'MLPClassifier' : MLPClassifier,
            'MultinomialNB' : MultinomialNB,
            'PassiveAggressive' : PassiveAggressive,
            'QDA' : QDA,
            'RandomForest' : RandomForest,
            'SGD' : SGD
        }
        
        # all hyperparameters for the classification models
        self._all_models_hyperparameters = [
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
                'solver' : hp.choice('MLPClassifier_solver', ['lbfgs', 'sgd', 'adam']), 
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

        self.hyperparameter_space = None

    # create hyperparameter space using Hyperopt.hp.choice
    # only models in models will be added to hyperparameter space
    def get_hyperparameter_space(self, model_hyperparameters, models) :

        # encoding space

        # imputation space

        # scaling space

        # balancing space

        # feature selection space
        
        # model selection and hyperparameter optimization space
        _model_hyperparameter = []
        for _model in [*models] :
            # checked before at models that all models are in default space
            for item in model_hyperparameters : # search the models' hyperparameters
                if item['model'] == _model :
                    _model_hyperparameter.append(item)

        _model_hyperparameter = hp.choice('classification_models', _model_hyperparameter)

        return _model_hyperparameter
        
        # the pipeline search space
        # return pyll.as_apply({
        #     'encoding' : _encoder,
        #     'imputation' : _imputer,
        #     'scaling' : _scaling,
        #     'balancing' : _balancing,
        #     'feature_selection' : _feature_selection,
        #     'classificaiont' : _model_hyperparameter
        # })

    def fit(self, X, y) :

        # Encoding
        # convert string types to numerical type
        from My_AutoML import DataEncoding

        x_encoder = DataEncoding()
        _X = x_encoder.fit(X)

        y_encoder = DataEncoding()
        _y = y_encoder.fit(y)

        # Imputer
        # fill missing values
        from My_AutoML import SimpleImputer
        
        imputer = SimpleImputer(method = 'mean')
        _X = imputer.fill(_X)

        # Scaling
        from My_AutoML import Standardize

        scaling = Standardize()
        scaling.fit(_X)
        _X = scaling.transform(_X)

        # Balancing
        # deal with imbalanced dataset, using over-/under-sampling methods

        # Feature selection
        # Remove redundant features, reduce dimensionality
        
        # train test split so the performance of model selection and 
        # hyperparameter optimization can be evaluated
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            _X, _y, test_size = self.test_size, random_state = self.seed
        )

        if self.models == 'auto' : # if auto, model pool will be all default models
            models = self._all_models.copy()
        else :
            models = {} # if specified, check if models in default models
            for _model in self.models :
                if _model not in [*self._all_models] :
                    raise ValueError(
                        'Only supported models are {}, get {}.'.format([*self._all_models], _model)
                    )
                models[_model] = self._all_models[_model]
        
        # the objective function of Bayesian Optimization tries to minimize
        # use accuracy score
        @ignore_warnings(category = ConvergenceWarning)
        def _objective(params) :

            from sklearn.metrics import accuracy_score

            _model = params['model']
            del params['model']
            clf = models[_model](**params) # call the model using passed parameters
                                           # params must be ordered and for positional arguments
            clf.fit(X_train.values, y_train.values.ravel())        
            y_pred = clf.predict(X_test.values)

            # since fmin of Hyperopt tries to minimize the objective function, take negative accuracy here
            return {'loss' : - accuracy_score(y_pred, y_test.values), 'status' : STATUS_OK}
        
        # initialize the hyperparameter space
        _all_models_hyperparameters = self._all_models_hyperparameters.copy()

        # special treatment for LibSVM_SVC kernel
        # if dataset not in shape of n * n, precomputed should be disabled
        n, p = X_test.shape
        if n != p :
            for item in _all_models_hyperparameters :
                if item['model'] == 'LibSVM_SVC' :
                    item['kernel'] = hp.choice('LibSVM_SVC_kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        
        # generate the hyperparameter space
        if self.hyperparameter_space is None :
            self.hyperparameter_space = self.get_hyperparameter_space(_all_models_hyperparameters, models)
        
        # call hyperopt to use Bayesian Optimization for Model Selection and Hyperparameter Selection

        # search algorithm
        if self.algo == 'rand' :
            algo = rand.suggest
        elif self.algo == 'tpe' :
            algo = tpe.suggest
        elif self.algo == 'atpe' :
            algo = atpe.suggest
        
        # Storage for evaluation points
        if self.spark_trials :
            trials = SparkTrials()
        else :
            trials = Trials()

        with mlflow.start_run() :
            best_results = fmin(
                fn = _objective,
                space = self.hyperparameter_space,
                algo = algo,
                max_evals = self.max_evals,
                timeout = self.timeout,
                trials = trials,
                show_progressbar = self.progressbar,
                rstate = np.random.RandomState(seed = self.seed)
            )
        
        # mapping the optimal model and hyperparameters selected
        # fit the optimal setting
        optimal_point = space_eval(self.hyperparameter_space, best_results)
        self.optimal_model = optimal_point['model'] # optimal model selected
        self.optimal_hyperparameters = optimal_point # optimal hyperparameter settings selected
        del self.optimal_hyperparameters['model']

        self._fit_model = self._all_models[self.optimal_model](**self.optimal_hyperparameters)
        self._fit_model.fit(_X.values, _y.values.ravel())

        return self

    def predict(self, X) :

        _X = X.copy()
        
        # may need preprocessing for test data, the preprocessing shoul be the same as in fit part
        # Encoding
        # convert string types to numerical type

        # Imputer
        # fill missing values

        # Scaling

        # Balancing
        # deal with imbalanced dataset, using over-/under-sampling methods

        # Feature selection
        # Remove redundant features, reduce dimensionality

        return self._fit_model.predict(_X)

'''
Regressors/Hyperparameters from sklearn:
1. AdaBoost: n_estimators, learning_rate, loss, max_depth
2. Ard regression: n_iter, tol, alpha_1, alpha_2, lambda_1, lambda_2,
            threshold_lambda, fit_intercept
3. Decision tree: criterion, max_features, max_depth_factor,
            min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
            max_leaf_nodes, min_impurity_decrease
4. extra trees: criterion, min_samples_leaf, min_samples_split, 
            max_features, bootstrap, max_leaf_nodes, max_depth, 
            min_weight_fraction_leaf, min_impurity_decrease
5. Gaussian Process: alpha, thetaL, thetaU
6. Gradient boosting: loss, learning_rate, min_samples_leaf, max_depth,
            max_leaf_nodes, max_bins, l2_regularization, early_stop, tol, scoring
7. KNN: n_neighbors, weights, p
8. Linear SVR (LibLinear): loss, epsilon, dual, tol, C, fit_intercept,
            intercept_scaling
9. Kernel SVR (LibSVM): kernel, C, epsilon, tol, shrinking
10. Random forest: criterion, max_features, max_depth, min_samples_split, 
            min_samples_leaf, min_weight_fraction_leaf, bootstrap, 
            max_leaf_nodes, min_impurity_decrease
11. SGD (Stochastic Gradient Descent): loss, penalty, alpha, fit_intercept, tol,
            learning_rate
12. MLP (Multilayer Perceptron): hidden_layer_depth, num_nodes_per_layer, 
            activation, alpha, learning_rate_init, early_stopping, solver, 
            batch_size, n_iter_no_change, tol, shuffle, beta_1, beta_2, epsilon
'''

class AutoRegressor() :

    def __init__(
        self,
        timeout = 360,
        max_evals = 32,
        models = 'auto',
        test_size = 0.15,
        method = 'Bayeisan',
        algo = 'tpe',
        spark_trials = True,
        seed = 1
    ) : 
        self.timeout = timeout
        self.max_evals = max_evals
        self.models = models
        self.test_size = test_size
        self.method = method
        self.algo = algo
        self.spark_trials = spark_trials
        self.seed = seed

        # autosklearn regressors
        from autosklearn.pipeline.components.regression.adaboost import AdaboostRegressor
        from autosklearn.pipeline.components.regression.ard_regression import ARDRegression
        from autosklearn.pipeline.components.regression.decision_tree import DecisionTree
        from autosklearn.pipeline.components.regression.extra_trees import ExtraTreesRegressor
        from autosklearn.pipeline.components.regression.gaussian_process import GaussianProcess
        from autosklearn.pipeline.components.regression.gradient_boosting import GradientBoosting
        from autosklearn.pipeline.components.regression.k_nearest_neighbors import KNearestNeighborsRegressor
        from autosklearn.pipeline.components.regression.liblinear_svr import LibLinear_SVR
        from autosklearn.pipeline.components.regression.libsvm_svr import LibSVM_SVR
        from autosklearn.pipeline.components.regression.mlp import MLPRegressor
        from autosklearn.pipeline.components.regression.random_forest import RandomForest
        from autosklearn.pipeline.components.regression.sgd import SGD

        # all regression models available
        self._all_models = {
            'AdaboostRegressor' : AdaboostRegressor,
            'ARDRegression' : ARDRegression,
            'DecisionTree' : DecisionTree,
            'ExtraTreesRegressor' : ExtraTreesRegressor,
            'GaussianProcess' : GaussianProcess,
            'GradientBoosting' : GradientBoosting,
            'KNearestNeighborsRegressor' : KNearestNeighborsRegressor,
            'LibLinear_SVR' : LibLinear_SVR,
            'LibSVM_SVR' : LibSVM_SVR,
            'MLPRegressor' : MLPRegressor,
            'RandomForest' : RandomForest,
            'SGD' : SGD
        }

        # all hyperparameters for the regression models
        self._all_models_hyperparameters = [
            {
                'model' : 'AdaboostRegressor',
                'n_estimators' : hp.quniform('AdaboostRegressor_n_estimators', 10, 100, 1), 
                'learning_rate' : hp.uniform('AdaboostRegressor_learning_rate', 0.00001, 1), 
                'loss' : hp.choice('AdaboostRegressor_algorithm', ['linear', 'square.R', 'exponential']), 
                # for base_estimator of Decision Tree
                'max_depth' : hp.quniform('AdaboostRegressor_max_depth', 2, 10, 1)
            },
            {
                'model' : 'ARDRegression',
                'n_iter' : hp.quniform('ARDRegression_n_iter', 100, 500, 1), 
                'tol' : hp.uniform('ARDRegression_tol', 1e-10, 1), 
                'alpha_1' : hp.uniform('ARDRegression_alpha_1', 1e-10, 1), 
                'alpha_2' : hp.uniform('ARDRegression_alpha_2', 1e-10, 1), 
                'lambda_1' : hp.uniform('ARDRegression_lambda_1', 1e-10, 1), 
                'lambda_2' : hp.uniform('ARDRegression_lambda_2', 1e-10, 1),
                'threshold_lambda' : hp.uniform('ARDRegression_threshold_lambda', 100, 100000), 
                'fit_intercept' : hp.choice('ARDRegression_fit_intercept', [True, False])
            },
            {
                'model' : 'DecisionTree',
                'criterion' : hp.choice('DecisionTree_criterion', ['gini', 'entropy']), 
                'max_features' : hp.choice('DecisionTree_max_features', [None, 'auto', 'sqrt', 'log2']), 
                'max_depth_factor' : hp.uniform('DecisionTree_max_depth_factor', 0, 1),
                'min_samples_split' : hp.quniform('DecisionTree_min_samples_split', 2, 10, 1), 
                'min_samples_leaf' : hp.quniform('DecisionTree_min_samples_leaf', 1, 10, 1), 
                'min_weight_fraction_leaf' : hp.uniform('DecisionTree_min_weight_fraction_leaf', 0, 1),
                'max_leaf_nodes' : hp.quniform('DecisionTree_max_leaf_nodes', 1, 100000, 1), 
                'min_impurity_decrease' : hp.uniform('DecisionTree_min_impurity_decrease', 0, 1)
            },
            {
                'model' : 'ExtraTreesRegressor',
                'criterion' : hp.choice('ExtraTreesRegressor_criterion', ['gini', 'entropy']), 
                'min_samples_leaf' : hp.quniform('ExtraTreesRegressor_min_samples_leaf', 1, 10, 1),
                'min_samples_split' : hp.quniform('ExtraTreesRegressor_min_samples_split', 2, 10, 1),  
                'max_features' : hp.choice('ExtraTreesRegressor_max_features', [None, 'auto', 'sqrt', 'log2']), 
                'bootstrap' : hp.choice('ExtraTreesRegressor_bootstrap', [True, False]), 
                'max_leaf_nodes' : hp.quniform('ExtraTreesRegressor_max_leaf_nodes', 1, 100000, 1),
                'max_depth' : hp.quniform('ExtraTreesRegressor_max_depth', 2, 10, 1), 
                'min_weight_fraction_leaf' : hp.uniform('ExtraTreesRegressor_min_weight_fraction_leaf', 0, 1), 
                'min_impurity_decrease' : hp.uniform('ExtraTreesRegressor_min_impurity_decrease', 0, 1)
            },
            {
                'model' : 'GaussianProcess',
                'alpha' : hp.lognormal('GaussianProcess_alpha', -10, 0), 
                'thetaL' : hp.lognormal('GaussianProcess_alpha', -10, 0), 
                'thetaU' : hp.lognormal('GaussianProcess_alpha', 0, 1)
            },
            {
                'model' : 'GradientBoosting',
                'loss' : hp.choice('GradientBoosting_loss', ['auto', 'binary_crossentropy', 'categorical_crossentropy']), 
                'learning_rate' : hp.loguniform('GradientBoosting_learning_rate', -5, 1), 
                'min_samples_leaf' : hp.quniform('GradientBoosting_min_samples_leaf', 1, 30, 1), 
                'max_depth' : hp.quniform('GradientBoosting_max_depth', 2, 50, 1),
                'max_leaf_nodes' : hp.quniform('GradientBoosting_max_leaf_nodes', 1, 100000, 1), 
                'max_bins' : hp.quniform('GradientBoosting_max_bins', 1, 255, 1), 
                'l2_regularization' : hp.uniform('GradientBoosting_l2_regularization', 0, 10), 
                'early_stop' : hp.choice('GradientBoosting_early_stop', ['auto', True, False]), 
                'tol' : hp.loguniform('GradientBoosting_tol', -10, 1), 
                'scoring' : hp.choice('GradientBoosting_scoring', ['loss', 'accuracy', 'roc_auc'])
            },
            {
                'model' : 'KNearestNeighborsRegressor',
                'n_neighbors' : hp.quniform('KNearestNeighborsRegressor_n_neighbors', 1, 20, 1), 
                'weights' : hp.choice('KNearestNeighborsRegressor_weights', ['uniform', 'distance']), 
                'p' : hp.quniform('KNearestNeighborsRegressor_p', 1, 5, 1)
            },
            {
                'model' : 'LibLinear_SVR',
                'epsilon' : hp.uniform('LibLinear_SVR_tol', 0, 1),
                'loss' : hp.choice('LibLinear_SVR__loss', ['hinge', 'squared_hinge']), 
                'dual' : hp.choice('LibLinear_SVR__dual', [True, False]), 
                'tol' : hp.loguniform('LibLinear_SVR__tol', -10, 1), 
                'C' : hp.loguniform('LibLinear_SVR__C', -5, 10), 
                'fit_intercept' : hp.choice('LibLinear_SVR__fit_intercept', [True, False]), 
                'intercept_scaling' : hp.loguniform('LibLinear_SVR__intercept_scaling', -5, 10)
            },
            {
                'model' : 'LibSVM_SVR',
                'kernel' : hp.choice('LibSVM_SVR_kernel', ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']), 
                'C' : hp.loguniform('LibSVM_SVR', -5, 1),
                'epsilon' : hp.uniform('LibSVM_SVR_epsilon', 1e-5, 1),
                'tol' : hp.loguniform('LibSVM_SVR_tol', -10, 1), 
                'shrinking' : hp.choice('LibSVM_SVR_shrinking', [True, False])
            },
            {
                'model' : 'MLPRegressor',
                'hidden_layer_depth' : hp.quniform('MLPRegressor_hidden_layer_depth', 1, 10, 1), 
                'num_nodes_per_layer' : hp.quniform('MLPRegressor_num_nodes_per_layer', 1, 20, 1), 
                'activation' : hp.choice('MLPRegressor_activation', ['identity', 'logistic', 'tanh', 'relu']), 
                'alpha' : hp.loguniform('MLPRegressor_alpha', -6, 1),
                'learning_rate_init' : hp.loguniform('MLPRegressor_learning_rate_init', -6, 1), 
                'early_stopping' : hp.choice('MLPRegressor_early_stopping', [True, False]), 
                'solver' : hp.choice('MLPRegressor_solver', ['lbfgs', 'sgd', 'adam']), 
                'batch_size' : hp.quniform('MLPRegressor_batch_size', 2, 200, 1),
                'n_iter_no_change' : hp.quniform('MLPRegressor_n_iter_no_change', 1, 20, 1), 
                'tol' : hp.loguniform('MLPRegressor_tol', -10, 1),
                'shuffle' : hp.choice('MLPRegressor_shuffle', [True, False]), 
                'beta_1' : hp.uniform('MLPRegressor_beta_1', 0, 0.999), 
                'beta_2' : hp.uniform('MLPRegressor_beta_2', 0, 0.999), 
                'epsilon' : hp.loguniform('MLPRegressor_epsilon', -10, 1)
            },
            {
                'model' : 'RandomForest',
                'criterion' : hp.choice('RandomForest_criterion', ['gini', 'entropy']), 
                'max_features' : hp.choice('RandomForest_max_features', [None, 'auto', 'sqrt', 'log2']),
                'max_depth' : hp.quniform('RandomForest_max_depth', 2, 10, 1), 
                'min_samples_split' : hp.quniform('RandomForest_min_samples_split', 2, 10, 1), 
                'min_samples_leaf' : hp.quniform('RandomForest_min_samples_leaf', 1, 30, 1),
                'min_weight_fraction_leaf' : hp.uniform('RandomForest_min_weight_fraction_leaf', 0, 1), 
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
        
        self.hyperparameter_space = None

    # create hyperparameter space using Hyperopt.hp.choice
    # only models in models will be added to hyperparameter space
    def get_hyperparameter_space(self, models) :

        _hyperparameter = []
        for _model in [*models] :
            # checked before at models that all models are in default space
            for item in self._all_models_hyperparameters : # search the models' hyperparameters
                if item['model'] == _model :
                    _hyperparameter.append(item)

        return hp.choice('regression_models', _hyperparameter)

    def fit(self, X, y) :

        # Encoding
        # convert string types to numerical type
        from My_AutoML import DataEncoding

        x_encoder = DataEncoding()
        _X = x_encoder.fit(X)

        y_encoder = DataEncoding()
        _y = y_encoder.fit(y)

        # Imputer
        # fill missing values
        from My_AutoML import SimpleImputer
        
        imputer = SimpleImputer(method = 'mean')
        _X = imputer.fill(_X)

        # Scaling

        # Balancing
        # deal with imbalanced dataset, using over-/under-sampling methods

        # Feature selection
        # Remove redundant features, reduce dimensionality
        
        # train test split so the performance of model selection and 
        # hyperparameter optimization can be evaluated
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            _X, _y, test_size = self.test_size, random_state = self.seed
        )

        if self.models == 'auto' : # if auto, model pool will be all default models
            models = self._all_models.copy()
        else :
            models = {} # if specified, check if models in default models
            for _model in self.models :
                if _model not in [*self._all_models] :
                    raise ValueError(
                        'Only supported models are {}, get {}.'.format([*self._all_models], _model)
                    )
                models[_model] = self._all_models[_model]
        
        # initialize the hyperparameter space
        if self.hyperparameter_space is None :
            self.hyperparameter_space = self.get_hyperparameter_space(models)
        
        # call hyperopt to use Bayesian Optimization for Model Selection and Hyperparameter Selection

        # the objective function of Bayesian Optimization tries to minimize
        # use mean_squared_error
        def _objective(params) :

            _model = params['model']
            del params['model']
            clf = models[_model](**params) # call the model using passed parameters
        
            from sklearn.metrics import mean_squared_error

            clf.fit(X_train.values, y_train.values.ravel())
            y_pred = clf.predict(X_test.values)

            # since fmin of Hyperopt tries to minimize the objective function, take negative
            return {'loss' : - mean_squared_error(y_pred, y_test.values), 'status' : STATUS_OK}

        # search algorithm
        if self.algo == 'rand' :
            algo = rand.suggest
        elif self.algo == 'tpe' :
            algo = tpe.suggest
        elif self.algo == 'atpe' :
            algo = atpe.suggest
        
        # Storage for evaluation points
        if self.spark_trials :
            trials = SparkTrials()
        else :
            trials = Trials()

        with mlflow.start_run() :
            best_results = fmin(
                fn = _objective,
                space = self.hyperparameter_space,
                algo = algo,
                max_evals = self.max_evals,
                timeout = self.timeout,
                trials = trials,
                show_progressbar = False
            )
