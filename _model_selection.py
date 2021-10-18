import numpy as np
import pandas as pd
import scipy
import scipy.stats
import sklearn

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
11. Multinomial naive Bayes: alpha, fit_prior
12. Passive aggressive: C, fit_intercept, tol, loss, average
13. QDA: reg_param
14. Random forest: criterion, max_features, max_depth, min_samples_split, 
            min_samples_leaf, min_weight_fraction_leaf, bootstrap, max_leaf_nodes
15. SGD (Stochastic Gradient Descent): loss, penalty, alpha, fit_intercept, tol,
            learning_rate
16. MLP (Multilayer Perceptron): hidden_layer_depth, num_nodes_per_layer, activation, alpha,
            learning_rate_init, early_stopping, solver, batch_size, n_iter_no_change, tol,
            shuffle, beta_1, beta_2, epsilon
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
        seed = 1
    ) :
        self.test_size = test_size
        self.seed = seed
        self.models = models

        from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, \
            RandomForestClassifier
        from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
        from sklearn.svm import LinearSVC, SVC
        from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier
        from sklearn.neural_network import MLPClassifier

        self._all_models = {
            'AdaBoostClassifier' : AdaBoostClassifier,
            'BernoulliNB' : BernoulliNB,
            'DecisionTreeClassifier' : DecisionTreeClassifier,
            'ExtraTreesClassifier' : ExtraTreesClassifier,
            'GaussianNB' : GaussianNB,
            'GradientBoostingClassifier' : GradientBoostingClassifier,
            'KNeighborsClassifier' : KNeighborsClassifier,
            'LinearDiscriminantAnalysis' : LinearDiscriminantAnalysis,
            'LinearSVC' : LinearSVC,
            'SVC' : SVC,
            'MultinomialNB' : MultinomialNB,
            'PassiveAggressiveClassifier' : PassiveAggressiveClassifier,
            'QuadraticDiscriminantAnalysis' : QuadraticDiscriminantAnalysis,
            'RandomForestClassifier' : RandomForestClassifier,
            'SGDClassifier' : SGDClassifier,
            'MLPClassifier' : MLPClassifier
        }

        self.hyperparameter_space = None

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

        