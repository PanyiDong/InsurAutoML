import imp
import numpy as np
import pandas as pd
import scipy
import scipy.stats
import sklearn
import mlflow
import hyperopt
from hyperopt import fmin, hp, rand, tpe, atpe, Trials, SparkTrials, STATUS_OK

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

class AutoClassifier() :

    '''
    Perform model selection and hyperparameter optimization for classification tasks
    using sklearn models, predefine hyperparameters
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

        from autosklearn.pipeline.components.classification.adaboost import AdaboostClassifier
        from autosklearn.pipeline.components.classification.bernoulli_nb import BernoulliNB
        from autosklearn.pipeline.components.classification.decision_tree import DecisionTree
        from autosklearn.pipeline.components.classification.extra_trees import ExtraTreesClassifier
        from autosklearn.pipeline.components.classification.gaussian_nb import GaussianNB
        from autosklearn.pipeline.components.classification.gradient_boosting import GradientBoostingClassifier
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
        self._all_hyperparameters = [
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
                'alpha' : hp.loguniform('alpha', 0.001, 10),
                'fit_prior' : hp.choice('fit_prior', [True, False])
            },
            {
                'model' : 'DecisionTree',
                'criterion' : hp.choice('criterion', ['gini', 'entropy']), 
                'max_features' : hp.choice('max_features', [None, 'auto', 'sqrt', 'log2']), 
                'max_depth_factor' : hp.uniform('max_depth_factor', 0, 1),
                'min_samples_split' : hp.quniform('min_samples_split', 2, 10, 1), 
                'min_samples_leaf' : hp.quniform('min_samples_leaf', 1, 10, 1), 
                'min_weight_fraction_leaf' : hp.uniform('min_weight_fraction_leaf', 0, 1),
                'max_leaf_nodes' : hp.quniform('max_leaf_nodes', 1, 100000, 1), 
                'min_impurity_decrease' : hp.uniform('min_impurity_decrease', 0, 1)
            },
            {
                'model' : 'ExtraTreesClassifier',
                'criterion' : hp.choice('criterion', ['gini', 'entropy']), 
                'min_samples_leaf' : hp.quniform('min_samples_leaf', 1, 10, 1),
                'min_samples_split' : hp.quniform('min_samples_split', 2, 10, 1),  
                'max_features' : hp.choice('max_features', [None, 'auto', 'sqrt', 'log2']), 
                'bootstrap' : hp.choice('bootstrap', [True, False]), 
                'max_leaf_nodes' : hp.quniform('max_leaf_nodes', 1, 100000, 1),
                'max_depth' : hp.quniform('max_depth', 2, 10, 1), 
                'min_weight_fraction_leaf' : hp.uniform('min_weight_fraction_leaf', 0, 1), 
                'min_impurity_decrease' : hp.uniform('min_impurity_decrease', 0, 1)
            },
            {
                'model' : 'GaussianNB'
            },
            {
                'model' : 'GradientBoostingClassifier',
                'loss' : hp.choice('loss', ['auto', 'binary_crossentropy', 'categorical_crossentropy']), 
                'learning_rate' : hp.loguniform('learning_rate', 0.001, 10), 
                'min_samples_leaf' : hp.quniform('min_samples_leaf', 1, 30, 1), 
                'max_depth' : hp.quniform('max_depth', 2, 50, 1),
                'max_leaf_nodes' : hp.quniform('max_leaf_nodes', 1, 100000, 1), 
                'max_bins' : hp.quniform('max_bins', 1, 255, 1), 
                'l2_regularization' : hp.uniform('l2_regularization', 0, 10), 
                'early_stop' : hp.choice('early_stp', ['auto', True, False]), 
                'tol' : hp.loguniform('tol', 1e-10, 1), 
                'scoring' : hp.choice('scoring', ['loss', 'accuracy', 'roc_auc'])
            },
            {
                'model' : 'KNearestNeighborsClassifier',
                'n_neighbors' : hp.quniform('n_neighbors', 1, 20, 1), 
                'weights' : hp.choice('weights', ['uniform', 'distance']), 
                'p' : hp.quniform('p', 1, 5, 1)
            },
            {
                'model' : 'LDA',
                'shrinkage' : hp.choice('shrinkage', ['svd', 'lsqr', 'eigen']), 
                'tol' : hp.loguniform('tol', 1e-10, 1)
            },
            {
                'model' : 'LibLinear_SVC',
                'penalty' : hp.choice('penalty', ['l1', 'l2']), 
                'loss' : hp.choice('loss', ['hinge', 'squared_hinge']), 
                'dual' : hp.choice('dual', [True, False]), 
                'tol' : hp.loguniform('tol', 1e-10, 1), 
                'C' : hp.loguniform('C', 1e-5, 10), 
                'multi_class' : hp.choice('multi_class', ['ovr', 'crammer_singer']),
                'fit_intercept' : hp.choice('fit_intercept', [True, False]), 
                'intercept_scaling' : hp.loguniform('intercept_scaling', 1e-5, 10)
            },
            {
                'model' : 'LibSVM_SVC',
                'C' : hp.loguniform('C', 1e-5, 10), 
                'kernel' : hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']), 
                'gamma' : hp.choice('gamma', ['auto', 'scale']), 
                'shrinking' : hp.choice('shrinking', [True, False]), 
                'tol' : hp.loguniform('tol', 1e-10, 1), 
                'max_iter' : hp.quniform('max_iter', -1, 50, 1)
            },
            {
                'model' : 'MLPClassifier',
                'hidden_layer_depth' : hp.quniform('hidden_layer_depth', 1, 10, 1), 
                'num_nodes_per_layer' : hp.quniform('num_nodes_per_layer', 1, 20, 1), 
                'activation' : hp.choice('activation', ['identity', 'logistic', 'tanh', 'relu']), 
                'alpha' : hp.loguniform('alpha', 1e-6, 10),
                'learning_rate_init' : hp.loguniform('learning_rate_init', 1e-6, 10), 
                'early_stopping' : hp.choice('early_stopping', [True, False]), 
                'solver' : hp.choice('solver', ['lbfgs', 'sgd', 'adam']), 
                'batch_size' : hp.quniform('batch_size', 2, 200, 1),
                'n_iter_no_change' : hp.quniform('batch_size', 1, 20, 1), 
                'tol' : hp.loguniform('tol', 1e-10, 1),
                'shuffle' : hp.choice('shuffle', [True, False]), 
                'beta_1' : hp.uniform('beta_1', 0, 0.999), 
                'beta_2' : hp.uniform('beta_2', 0, 0.999), 
                'epsilon' : hp.loguniform('epsilon', 1e-10, 10)
            },
            {
                'model' : 'MultinomialNB',
                'alpha' : hp.loguniform('alpha', 0.001, 10),
                'fit_prior' : hp.choice('fit_prior', [True, False])
            },
            {
                'model' : 'PassiveAggressive',
                'C' : hp.loguniform('C', 1e-3, 100), 
                'fit_intercept' : hp.choice('fit_intercept', [True, False]), 
                'tol' : hp.loguniform('tol', 1e-10, 1), 
                'loss' : hp.choice('loss', ['hinge', 'squared_hinge']), 
                'average' : hp.choice('average', [True, False])
            },
            {
                'model' : 'QDA',
                'reg_param' : hp.uniform('reg_param', 0, 1)
            },
            {
                'model' : 'RandomForest',
                'criterion' : hp.choice('criterion', ['gini', 'entropy']), 
                'max_features' : hp.choice('max_features', [None, 'auto', 'sqrt', 'log2']),
                'max_depth' : hp.quniform('max_depth', 2, 10, 1), 
                'min_samples_split' : hp.quniform('min_samples_split', 2, 10, 1), 
                'min_samples_leaf' : hp.quniform('min_samples_leaf', 1, 30, 1),
                'min_weight_fraction_leaf' : hp.uniform('min_weight_fraction_leaf', 0, 1), 
                'bootstrap' : hp.choice('bootstrap', [True, False]), 
                'max_leaf_nodes' : hp.quniform('max_leaf_nodes', 1, 100000, 1),
                'min_impurity_decrease' : hp.uniform('min_impurity_decrease', 0, 1)
            },
            {
                'model' : 'SGD',
                'loss' : hp.choice('loss', ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']), 
                'penalty' : hp.choice('penalty', ['l1', 'l2']), 
                'alpha' : hp.loguniform('alpha', 1e-6, 10), 
                'fit_intercept' : hp.choice('fit_intercept', [True, False]), 
                'tol' : hp.loguniform('tol', 1e-10, 1),
                'learning_rate' : hp.choice('learning_rate', ['constant', 'optimal', 'invscaling', 'adaptive'])
            }
        ]

        self.hyperparameter_space = None

    # create hyperparameter space using Hyperopt.hp.choice
    # only models in models will be added to hyperparameter space
    def get_hyperparameter_space(self, models) :

        _hyperparameter = []
        for _model in [*models] :
            # checked before at models that all models are in default space
            for item in self._all_hyperparameters : # search the models' hyperparameters
                if item['model'] == _model :
                    _hyperparameter.append(item)

        return hp.choice('classification_models', _hyperparameter)

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
        
        # the objective function of Bayesian Optimization tries to minimize
        # use accuracy score
        def _objective(params) :

            _model = params['model']
            del params['model']
            clf = models[_model](**params) # call the model using passed parameters
        
            from sklearn.metrics import accuracy_score

            clf.fit(X_train.values, y_train.values.ravel())
            y_pred = clf.predict(X_test.values)

            # since fmin of Hyperopt tries to minimize the objective function, take negative accuracy here
            return {'loss' : - accuracy_score(y_pred, y_test.values), 'status' : STATUS_OK}
        
        # initialize the hyperparameter space
        if self.hyperparameter_space is None :
            self.hyperparameter_space = self.get_hyperparameter_space(models)
        
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
            trials = SparkTrials
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

        print(best_results)



        