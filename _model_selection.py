import numpy as np
import pandas as pd
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