import numpy as np
import pandas as pd
import scipy
import scipy.stats
import sklearn
import hyperopt
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials

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
        models = 'auto',
        test_size = 0.15,
        method = 'Bayeisan',
        seed = 1
    ) : 
        self.models = models
        self.test_size = test_size
        self.method = method
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

        self.hyperparameter_space = None
    
    # the objective function of Bayesian Optimization tries to minimize
    # use accuracy score
    def _objective(self, X_train, y_train, X_test, y_test, params) :

        _model = params['model']
        del params['model']
        clf = self.models[_model](params)
        
        from sklearn.metrics import accuracy_score

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # since fmin of Hyperopt tries to minimize the objective function, take negative accuracy here
        return {'loss' : - accuracy_score(y_pred, y_test), 'status' : STATUS_OK}

    # create hyperparameter space using Hyperopt.hp.choice
    def get_hyperparameter_space(self) :

        return hp.choice('models', [
            {
                'model' : 'AdaBoostClassifier',
                'n_estimators' : hp.quniform('n_estimators', 10, 100, 1), 
                'learning_rate' : hp.loguniform('learning_rate', 0.001, 10), 
                'algorithm' : hp.choice('algorithm', ['SAMME', 'SAMME.R']), 
                'max_depth' : hp.quniform('max_depth', 2, 10, 1) # for base_estimator of Decision Tree
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
            },
            {
                'model' : 'LibSVM_SVC',
            },
            {
                'model' : 'MLPClassifier',
            },
            {
                'model' : 'MultinomialNB',
            },
            {
                'model' : 'PassiveAggressive',
            },
            {
                'model' : 'QDA',
            },
            {
                'model' : 'RandomForest',
            },
            {
                'model' : 'SGD',
            }
        ])

    def fit(self, X, y) :
        
        # train test split so the performance of model selection and 
        # hyperparameter optimization can be evaluated
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = self.test_size, random_state = self.seed
        )

        if self.models == 'auto' : # if auto, model pool will be all default models
            self.models = self._all_models.copy()
        else :
            self.models = self.models # if specified, check if models in default models
            for _model in self.models :
                if _model not in [*self._all_models] :
                    raise ValueError(
                        'Only supported models are {}, get {}.'.format([*self._all_models], _model)
                    )

        if self.hyperparameter_space is None :
            self.hyperparameter_space = self.get_hyperparameter_space()

        