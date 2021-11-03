import random
import warnings
import time
import numbers
import warnings
import numpy as np
import pandas as pd
import scipy
import scipy.linalg
import sklearn
from sklearn.feature_selection._univariate_selection import f_classif
from sklearn.utils.extmath import stable_cumsum, svd_flip
from sympy import solve
import itertools
from functools import partial

from ._utils import nan_cov, maxloc, empirical_covariance, class_means, class_cov, \
    True_index, Pearson_Corr, MI, t_score, ANOVA, random_index

class PCA_FeatureSelection() :

    '''
    Principal Component Analysis
    
    Use Singular Value Decomposition (SVD) to project data to a lower dimensional 
    space, and thus achieve feature selection.

    Methods used:
    Full SVD: LAPACK, scipy.linalg.svd  
    Trucated SVD: ARPACK, scipy.sparse.linalg.svds
    Randomized truncated SVD:

    Parameters
    ----------
    n_components: remaining features after selection, default = None

    solver: the method to perform SVD, default = 'auto'
    all choices ('auto', 'full', 'truncated', 'randomized')

    tol: Tolerance for singular values computed for truncated SVD

    n_iter: Number of iterations for randomized solver, default = 'auto'

    seed: random seed, default = 1
    '''

    def __init__(
        self, 
        n_components = None, 
        solver = 'auto', 
        tol = 0.,
        n_iter = 'auto',
        seed = 1
    ):
        self.n_components = n_components
        self.solver = solver
        self.tol = tol
        self.n_iter = n_iter
        self.seed = seed

    def fit(self, X, y = None) :

        n, p = X.shape

        # Deal with default n_componets = None
        if self.n_components == None :
            if self.solver != 'truncated' :
                n_components = min(n, p)
            else :
                n_components = min(n, p) - 1
        else :
            n_components = self.n_components

        if n_components <= 0 :
            raise ValueError('Selection components must be larger than 0!')

        # Deal with solver
        self.fit_solver = self.solver
        if self.solver == 'auto' :
            if max(n, p) < 500 :
                self.fit_solver = 'full'
            elif n_components >= 1 and n_components < 0.8 * min(n, p) :
                self.fit_solver = 'randomized'
            else :
                self.fit_solver = 'full'
        else :
            self.fit_solver = self.solver

        if self.fit_solver == 'full' :
            self.U_, self.S_, self.V_ = self._fit_full(X, n_components)
        elif self.fit_solver in ['truncated', 'randomized'] :
            self.U_, self.S_, self.V_ = self._fit_truncated(X, n_components, self.fit_solver)
        else :
            raise ValueError('Not recognizing solver = {}!'.format(self.fit_solver))

        return self

    def transform(self, X) :

        _features = list(X.columns)

        U = self.U_[:, : self.n_components]

        # X_new = X * V = U * S * Vt * V = U * S
        X_new = U * self.S_[:self.n_components]
        # return dataframe format
        #X_new = pd.DataFrame(U, columns = _features[:self.n_components])

        return X_new

    def _fit_full(self, X, n_components) :

        n, p = X.shape
        if n_components < 0 or n_components > min(n, p) :
            raise ValueError(
                'n_components must between 0 and {0:d}, but get {1:d}'.format(min(n, p), n_components)
            )
        elif not isinstance(n_components, numbers.Integral) :
            raise ValueError('Expect integer n_components, but get {:.6f}'.format(n_components))

        # center the data
        self._x_mean = np.mean(X, axis = 0)
        X -= self._x_mean

        # solve for svd
        from scipy.linalg import svd
        U, S, V = svd(X, full_matrices = False)

        # make sure the max column values of U are positive, if not flip the column
        # and flip corresponding V rows
        max_abs_col = np.argmax(np.max(U), axis = 0)
        signs = np.sign(U[max_abs_col, range(U.shape[1])])
        U *= signs
        V *= signs.reshape(-1, 1)
        
        _var = (S ** 2) / (n - 1)
        total_var = _var.sum()
        _var_ratio = _var / total_var

        if 0 < n_components < 1.0:
            # number of components for which the cumulated explained
            # variance percentage is superior to the desired threshold
            # side='right' ensures that number of features selected
            # their variance is always greater than n_components float
            # passed.
            ratio_cumsum = stable_cumsum(_var_ratio)
            n_components = np.searchsorted(ratio_cumsum, n_components, side="right") + 1
        # Compute noise covariance using Probabilistic PCA model
        # The sigma2 maximum likelihood (cf. eq. 12.46)
        if n_components < min(n, p):
            self._noise_variance_ = _var[n_components:].mean()
        else:
            self._noise_variance_ = 0.0

        self.n_samples, self.n_features = n, p
        self.components_ = V[:n_components]
        self.n_components = n_components
        self._var = _var[:n_components]
        self._var_ratio = _var_ratio[:n_components]
        self.singular_values = S[:n_components]

        return U, S, V

    def _fit_truncated(self, X, n_components, solver) :

        n, p = X.shape
        
        self._x_mean = np.mean(X, axis = 0)
        X -= self._x_mean

        if solver == 'truncated' :

            from scipy.sparse.linalg import svds

            np.random.seed(self.seed)
            v0 = np.random.uniform(-1, 1)
            U, S, V = svds(X, k = n_components, tol = self.tol, v0 = v0)
            S = S[::-1]
            U, V = svd_flip(U[:, ::-1], V[::-1])
        elif solver == 'randomized' :

            from sklearn.utils.extmath import randomized_svd

            U, S, V = randomized_svd(
                np.array(X),
                n_components = n_components,
                n_iter = self.n_iter,
                flip_sign = True,
                random_state = self.seed
            )
        
        self.n_samples, self.n_features = n, p
        self.components_ = V
        self.n_components = n_components

        # Get variance explained by singular values
        self._var = (S ** 2) / (n - 1)
        total_var = np.var(X, ddof=1, axis=0)
        self._var_ratio = self._var / total_var.sum()
        self.singular_values = S.copy()  # Store the singular values.

        if self.n_components < min(n, p):
            self._noise_variance_ = total_var.sum() - self._var.sum()
            self._noise_variance_ /= min(n, p) - n_components
        else:
            self._noise_variance_ = 0.0

        return U, S, V

class LDASelection() :

    def __init__(
        self,
        priors = None,
        n_components = None,
    ) :
        self.priors = priors
        self.n_components = n_components

    def _eigen(self, X, y) : 

        self.means_ = class_means(X, y)
        self.covariance_ = class_cov(X, y, self.priors_)

        Sw = self.covariance_  # within scatter
        St = empirical_covariance(X)  # total scatter
        Sb = St - Sw  # between scatter

        evals, evecs = scipy.linalg.eigh(Sb, Sw)
        self.explained_variance_ratio_ = np.sort(evals / np.sum(evals))[::-1][:self._max_components]
        evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors

        self.scalings_ = evecs
        self.coef_ = np.dot(self.means_, evecs).dot(evecs.T)
        self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(self.priors_)

    def fit(self, X, y) :

        self.classes_ = np.unique(y)
        n, p = X.shape

        if len(self.classes_) == n :
            raise ValueError('Classes must be smaller than number of samples!')

        if self.priors is None:  # estimate priors from sample
            _y_uni = np.unique(y)  # non-negative ints
            self.priors_ = []
            for _value in _y_uni :
                self.priors_.append(y.loc[y.values == _value].count()[0] / len(y))
            self.priors_ = np.asarray(self.priors_)
        else:
            self.priors_ = np.asarray(self.priors)

        if (self.priors_ < 0).any():
            raise ValueError("priors must be non-negative")
        if not np.isclose(self.priors_.sum(), 1.0):
            warnings.warn("The priors do not sum to 1. Renormalizing", UserWarning)
            self.priors_ = self.priors_ / self.priors_.sum()

        max_components = min(len(self.classes_) - 1, X.shape[1]) # maximum number of components
        if self.n_components is None:
            self._max_components = max_components
        else:
            if self.n_components > max_components:
                raise ValueError(
                    "n_components cannot be larger than min(n_features, n_classes - 1)."
                )
            self._max_components = self.n_components

    def transform(self, X) :

        X_new = np.dot(X, self.scalings_)

        return X_new[:, :self._max_components]
        
class RBFSampler() :

    '''
    Implement of Weighted Sums of Random Kitchen Sinks

    Parameters
    ----------
    gamma: use to determine standard variance of random weight table, default = 1
    Parameter of RBF kernel: exp(-gamma * x^2).

    n_components: number of samples per original feature, default = 100

    seed: random generation seed, default = None
    '''

    def __init__(
        self, 
        gamma = 1., 
        n_components = 100, 
        seed = 1
    ):
        self.gamma = gamma
        self.n_components = n_components
        self.seed = seed

    def fit(self, X, y = None) :
        
        if isinstance(X, list) :
            n_features = len(X[0])
        else :
            n_features = X.shape[1]

        if self.n_components > n_features :
            warnings.warn('N_components {} is larger than n_features {}, will set to n_features.'.format(
                self.n_components, n_features
            ))
            self.n_components = n_features
        else :
            self.n_components = self.n_components

        if not self.seed :
            self.seed = np.random.seed(int(time.time()))
        elif not isinstance(self.seed, int) :
            raise ValueError('Seed must be integer, receive {}'.format(self.seed))

        self._random_weights = np.random.normal(0, np.sqrt(2 * self.gamma), size = (n_features, self.n_components))
        self._random_offset = np.random.uniform(0, 2 * np.pi, size = (1, self.n_components))

        return self

    def transform(self, X) :

        projection = np.dot(X, self._random_weights)
        projection += self._random_offset
        np.cos(projection, projection)
        projection *= np.sqrt(2.0 / self.n_components)

        # return dataframe
        #projection = pd.DataFrame(projection, columns = list(X.columns)[:self.n_components])

        return projection

class FeatureFilter() :

    '''
    Use certain criteria to score each feature and select most relevent ones

    Parameters
    ----------
    criteria: use what criteria to score features, default = 'Pearson'
    supported {'Pearson', 'MI'}
    'Pearson': Pearson Correlation Coefficient
    'MI': 'Mutual Information'

    n_components: threshold to retain features, default = None
    will be set to n_features
    '''

    def __init__(
        self,
        criteria = 'Pearson',
        n_components = None
    ) :
        self.criteria = criteria
        self.n_components = n_components

    def fit(self, X, y = None) :

        try :
            _empty = (y == None).all().values[0]
        except AttributeError :
            _empty = (y == None)
        if _empty :
            raise ValueError('Must have response!')
        
        if self.criteria == 'Pearson' :          
            self._score = Pearson_Corr(X, y)
        elif self.criteria == 'MI' :
            self._score = MI(X, y)

    def transform(self, X) :

        if self.n_components == None :
            n_components = X.shape[1]
        else :
            n_components = self.n_components

        _columns = np.argsort(self._score)[:n_components]

        return X.iloc[:, _columns]

# FeatureWrapper

# Exhaustive search for optimal feature combination

# Sequential Feature Selection (SFS)
# Sequential Backward Selection (SBS)
# Sequential Floating Forward Selection (SFFS)
# Adapative Sequential Forward Floating Selection (ASFFS)
class ASFFS() :

    '''
    Adaptive Sequential Forward Floating Selection (ASFFS)
    Mostly, ASFFS performs the same as Sequential Floating Forward Selection (SFFS),
    where only one feature is considered as a time. But, when the selected features are coming
    close to the predefined maximum, a adaptive generalization limit will be activated, and 
    more than one features can be considered at one time. The idea is to consider the correlation
    between features. [1]

    [1] Somol, P., Pudil, P., Novovičová, J. and Paclık, P., 1999. Adaptive floating search 
    methods in feature selection. Pattern recognition letters, 20(11-13), pp.1157-1163.

    Parameters
    ----------
    d: maximum features retained, default = None
    will be calculated as max(max(20, n), 0.5 * n)
    
    Delta: dynamic of maximum number of features, default = 0
    d + Delta features will be retained

    b: range for adaptive generalization limit to activate, default = None
    will be calculated as max(5, 0.05 * n)

    r_max: maximum of generalization limit, default = 5
    maximum features to be considered as one step

    model: the model used to evaluate the objective function, default = 'linear'
    supproted ('linear', 'lasso', 'ridge')

    objective: the objective function of significance of the features, default = 'MSE'
    supported {'MSE', 'MAE'}
    '''

    def __init__(
        self,
        n_components = None,
        Delta = 0,
        b = None,
        r_max = 5,
        model = 'Linear',
        objective = 'MSE'
    ) :
        self.n_components = n_components
        self.Delta = Delta
        self.b = b
        self.r_max = r_max
        self.model = model
        self.objective = objective

    def generalization_limit(self, k, d, b) :

        if np.abs(k - d) < b :
            r = self.r_max
        elif np.abs(k - d) < self.r_max + b :
            r = self.r_max + b - np.abs(k - d)
        else :
            r = 1
        
        return r

    def _Forward_Objective(self, selected, unselected, o, X, y) :

        _subset = list(itertools.combinations(unselected, o))
        _comb_subset = [selected + list(item) for item in _subset] # concat selected features with new features
        
        _objective_list = []
        if self.model == 'Linear' :
            from sklearn.linear_model import LinearRegression
            _model = LinearRegression()
        elif self.model == 'lasso' :
            from sklearn.linear_model import Lasso
            _model = Lasso()
        elif self.model == 'ridge' :
            from sklearn.linear_model import Ridge
            _model = Ridge()
        else :
            raise ValueError('Not recognizing model!')

        if self.objective == 'MSE' :
            from sklearn.metrics import mean_squared_error
            _obj = mean_squared_error
        elif self.objective == 'MAE' :
            from sklearn.metrics import mean_absolute_error
            _obj = mean_absolute_error

        for _set in _comb_subset :
            _model.fit(X[_set], y)
            _predict = _model.predict(X[_set])
            _objective_list.append(1 / _obj(y, _predict)) # the goal is to maximize the objective function

        return _subset[maxloc(_objective_list)], _objective_list[maxloc(_objective_list)]

    def _Backward_Objective(self, selected, o, X, y) :
        
        _subset = list(itertools.combinations(selected, o))
        _comb_subset = [[_full for _full in selected if _full not in item] for item in _subset] # remove new features from selected features 

        _objective_list = []
        if self.model == 'Linear' :
            from sklearn.linear_model import LinearRegression
            _model = LinearRegression()
        elif self.model == 'lasso' :
            from sklearn.linear_model import Lasso
            _model = Lasso()
        elif self.model == 'ridge' :
            from sklearn.linear_model import Ridge
            _model = Ridge()
        else :
            raise ValueError('Not recognizing model!')

        if self.objective == 'MSE' :
            from sklearn.metrics import mean_squared_error
            _obj = mean_squared_error
        elif self.objective == 'MAE' :
            from sklearn.metrics import mean_absolute_error
            _obj = mean_absolute_error

        for _set in _comb_subset :
            _model.fit(X[_set], y)
            _predict = _model.predict(X[_set])
            _objective_list.append(1 / _obj(y, _predict)) # the goal is to maximize the objective function

        return _subset[maxloc(_objective_list)], _objective_list[maxloc(_objective_list)]

    def fit(self, X, y) :
        
        n, p = X.shape
        features = list(X.columns)

        if self.n_components == None :
            _n_components = min(max(20, p), int(0.5 * p))
        else :
            _n_components = self.n_components
        if self.b == None :
            _b = min(5, int(0.05 * p))

        _k = 0
        self.J_max = [0 for _ in range(p + 1)] # mark the most significant objective function value
        self._subset_max = [[] for _ in range(p + 1)] # mark the best performing subset features
        _unselected = features.copy()
        _selected = [] # selected  feature stored here, not selected will be stored in features       

        while True :
            
            # Forward Phase
            _r = self.generalization_limit(_k, _n_components, _b)
            _o = 1

            while _o <= _r and len(_unselected) >= 1 : # not reasonable to add feature when all selected
             
                _new_feature, _max_obj = self._Forward_Objective(_selected, _unselected, _o, X, y)

                if _max_obj > self.J_max[_k + _o] :
                    self.J_max[_k + _o] = _max_obj.copy()
                    _k += _o
                    for _f in _new_feature : # add new features and remove these features from the pool
                        _selected.append(_f)
                    for _f in _new_feature :
                        _unselected.remove(_f)
                    self._subset_max[_k] = _selected.copy()
                    break
                else :
                    if _o < _r :
                        _o += 1
                    else :
                        _k += 1 # the marked in J_max and _subset_max are considered as best for _k features
                        _selected = self._subset_max[_k].copy() # read stored best subset
                        _unselected = features.copy()
                        for _f in _selected :
                            _unselected.remove(_f)
                        break

            # Termination Condition
            if _k >= _n_components + self.Delta :
                break

            # Backward Phase
            _r = self.generalization_limit(_k, _n_components, _b)
            _o = 1

            while _o <= _r and _o < _k : # not reasonable to remove when only _o feature selected
                
                _new_feature, _max_obj = self._Backward_Objective(_selected, _o, X, y)

                if _max_obj > self.J_max[_k - _o] :
                    self.J_max[_k - _o] = _max_obj.copy()
                    _k -= _o
                    for _f in _new_feature : # add new features and remove these features from the pool
                        _unselected.append(_f)
                    for _f in _new_feature :
                        _selected.remove(_f)
                    self._subset_max[_k] = _selected.copy()
                    
                    _o = 1 # return to the start of backward phase, make sure the best subset is selected
                else :
                    if _o < _r :
                        _o += 1
                    else :
                        break
        
        self.selected_ = _selected
        return self

    def transform(self, X) :

        return X.loc[:, self.selected_]
            
# Genetic Algorithm (GA)
class GeneticAlgorithm() :

    '''
    Use Genetic Algorithm (GA) to select best subset features [1]

    Procedure: (1) Train a feature pool where every individual is trained on predefined methods,
    result is pool of binary lists where 1 for feature selected, 0 for not selected

    (2) Use Genetic Algorithm to generate a new selection binary list
        (a) Selection: Roulette wheel selection, use fitness function to randomly select one individual
        (b) Crossover: Single-point Crossover operator, create child selection list from parents list
        (c) Mutation: Mutate the selection of n bits by certain percentage

    [1] Tan, F., Fu, X., Zhang, Y. and Bourgeois, A.G., 2008. A genetic algorithm-based method for 
    feature subset selection. Soft Computing, 12(2), pp.111-120.

    Parameters
    ----------
    n_components: Number of features to retain, default = 20

    n_generations: Number of looping generation for GA, default = 10

    feature_selection: Feature selection methods to generate a pool of selections, default = 'auto'
    support ('auto', 'random', 'Entropy', 't_statistics', 'SVM_RFE')

    n_initial: Number of random feature selection rules to initialize, default = 10

    fitness_func: Fitness function, default None
    deafult will set as w * Accuracy + (1 - w) / regularization, all functions must be maximization optimization

    fitness_fit: Model to fit selection and calculate accuracy for fitness, default = 'SVM'
    support ('Linear', 'Logistic', 'Random Forest', 'SVM')

    fitness_weight: Default fitness function weight for accuracy, default = 0.9

    n_pair: Number of pairs of new selection rules to generate, default = 5

    ga_selection: How to perform selection in GA, default = 'Roulette Wheel'
    support ('Roulette Wheel', 'Rank', 'Steady State', 'Tournament', 'Elitism', 'Boltzmann')

    p_crossover: Probability to perform crossover, default = 1

    ga_crossover: How to perform crossover in GA, default = 'Single-point'
    support ('Single-point', 'Two-point', 'Uniform')

    crossover_n: Place of crossover points to perform, default = None
    deafult will set to p / 4 for single-point crossover

    p_mutation: Probability to perform mutation (flip bit in selection list), default = 0.001

    mutation_n: Number of mutation points to perform, default = None 
    default will set to p / 10

    seed = 1
    '''

    def __init__(
        self,
        n_components = 20,
        n_generations = 10,
        feature_selection = 'random',
        n_initial = 10,
        fitness_func = None,
        fitness_fit = 'SVM',
        fitness_weight = 0.9,
        n_pair = 5,
        ga_selection = 'Roulette Wheel',
        p_crossover = 1,
        ga_crossover  = 'Single-point',
        crossover_n = None,
        p_mutation = 0.001,
        mutation_n = None,
        seed = 1
    ) :
        self.n_components = n_components
        self.n_generations = n_generations
        self.feature_selection = feature_selection
        self.n_initial = n_initial
        self.fitness_func = fitness_func
        self.fitness_fit = fitness_fit
        self.fitness_weight = fitness_weight
        self.n_pair = n_pair
        self.ga_selection = ga_selection
        self.p_crossover = p_crossover
        self.ga_crossover = ga_crossover
        self.crossover_n = crossover_n
        self.p_mutation = p_mutation
        self.mutation_n = mutation_n
        self.seed = seed

        self._auto_sel = {
                'Entropy' : self._entropy,
                't_statistics' : self._t_statistics,
                'SVM_RFE' : self._SVM_RFE
            }

    def _random(self, X, y, n) :

        # randomly select n features from X
        _, p = X.shape

        if n > p :
            raise ValueError('Selected features can not be larger than dataset limit {}, get {}.'.format(p, n))
        
        _index = random_index(n, p)

        _selected = [0 for _ in range(p)] # default all as 0
        for i in range(n) : # select n_components as selected
            _selected[_index[i]] = 1

        return _selected

    def _entropy(self, X, y, n) :
        
        # call Mutual Information from FeatureFilter
        _score = MI(X, y)

        # select highest scored features
        _score_sort = np.flip(np.argsort(_score))
        
        _selected = [0 for _ in range(len(_score_sort))] # default all as 0
        for i in range(n) : # select n_components as selected
            _selected[_score_sort[i]] = 1

        return _selected

    def _t_statistics(self, X, y, n) :
        
        # for 2 group dataset, use t-statistics; otherwise, use ANOVA
        if len(y[list(y.columns)[0]].unique()) == 2 :
            _score = t_score(X, y)
        elif len(y[list(y.columns)[0]].unique()) > 2 :
            _score = ANOVA(X, y)
        else :
            raise ValueError('Only support for more than 2 groups, get only 1 group!')

        # select lowest scored features
        _score_sort = np.argsort(_score)
        
        _selected = [0 for _ in range(len(_score_sort))] # default all as 0
        for i in range(n) : # select n_components as selected
            _selected[_score_sort[i]] = 1

        return _selected

    def _SVM_RFE(self, X, y, n) :

        from sklearn.feature_selection import RFE
        from sklearn.svm import SVC
        
        # using sklearn RFE to recursively remove one feature using SVR, until n_components left
        _estimator = SVC(kernel="linear")
        _selector = RFE(_estimator, n_features_to_select = n, step = 1)
        _selector = _selector.fit(X.values, y.values.ravel())

        _selected = _selector.support_.tolist() # retunr the mask list of feature selection
        _selected = [int(item) for item in _selected]

        return _selected

    def _cal_fitness(self, X, y, selection) :

        from sklearn.metrics import accuracy_score

        if not self.fitness_func : # fit selected features and calcualte accuracy score
            if self.fitness_fit == 'Linear' :
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
            elif self.fitness_fit == 'Logistic' :
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression()
            elif self.fitness_fit == 'Random Forest' :
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor()
            elif self.fitness_fit == 'SVM' : # select by y using SVC and SVR
                if len(pd.unique(y.values.ravel())) <= 30 :
                    from sklearn.svm import SVC
                    model = SVC()
                else :
                    from sklearn.svm import SVR
                    model = SVR()
            else :
                raise ValueError(
                    'Only support ["Linear", "Logistic", "Random Forest", "SVM"], get {}'.format(self.fitness_fit)
                )
            model.fit(X.iloc[:, True_index(selection)].values, y.values.ravel())
            y_pred = model.predict(X.iloc[:, True_index(selection)])
            _accuracy_score = accuracy_score(y, y_pred)

            return self.fitness_weight * _accuracy_score + (1 - self.fitness_weight) / sum(selection)
        else :
            return self.fitness_func(X, y, selection)

    def _GeneticAlgorithm(self, X, y, selection_pool) :

        n, p = X.shape
        
        # calculate the fitness of all feature selection in the pool
        try : # self._fitness from None value to np.array, type change
            _fitness_empty = not self._fitness
        except ValueError :
            _fitness_empty = not self._fitness.any()

        if _fitness_empty : # first round of calculating fitness for all feature selections
            self._fitness = []
            for _seletion in selection_pool :
                self._fitness.append(self._cal_fitness(X, y, _seletion))
            
            # normalize the fitness
            self._fitness = np.array(self._fitness)
            self._sum_fitness = sum(self._fitness)
            self._fitness /= self._sum_fitness            
        else :
            self._fitness *= self._sum_fitness
            for i in range(2 * self.n_pair) :
                self._fitness = np.append(self._fitness, self._cal_fitness(X, y, selection_pool[-(i + 1)])) # only need to calculate the newly added ones
                self._sum_fitness += self._fitness[-1]
            # normalize the fitness
            self._fitness /= self._sum_fitness

        # Selection
        if self.ga_selection == 'Roulette Wheel' :
            # select two individuals from selection pool based on probability (self._fitness)
            # insert into selection_pool (last two), will be the placeholder for offsprings
            for _ in range(2 * self.n_pair) :
                selection_pool.append(selection_pool[np.random.choice(len(self._fitness), p = self._fitness)])

        # Crossover, generate offsprings
        if np.random.rand() < self.p_crossover : # only certain probability of executing crossover
            if self.ga_crossover == 'single-point' :
                if not self.crossover_n :
                    self.crossover_n = int(p / 4) # default crossover point is first quarter point
                else :
                    if self.crossover_n > p :
                        raise ValueError('Place of cross points must be smaller than p = {}, get {}.'.format(p, self.crossover_n))
                    self.crossover_n == int(self.crossover_n)
                
                for i in range(self.n_pair) :
                    _tmp1 = selection_pool[- (2 * i + 2)]
                    _tmp2 = selection_pool[- (2 * i + 1)]
                    selection_pool[- (2 * i + 2)] = _tmp2[:self.crossover_n] + _tmp1[self.crossover_n:] # exchange first crossover_n bits from parents
                    selection_pool[- (2 * i + 1)] = _tmp1[:self.crossover_n] + _tmp2[self.crossover_n:]

        # Mutation
        for i in range(2 * self.n_pair) : # for two offsprings
            if np.random.rand() < self.p_mutation : # only certain probability of executing mutation
                if not self.mutation_n :
                    self.mutation_n = int(p / 10) # default number of mutation point is first quarter point
                else :
                    if self.mutation_n > p :
                        raise ValueError('Number of mutation points must be smaller than p = {}, get {}.'.format(p, self.mutation_n))
                    self.mutation_n == int(self.mutation_n)

                _mutation_index = random_index(self.mutation_n, p, seed = None) # randomly select mutation points
                selection_pool[-(i + 1)] = [selection_pool[-(i + 1)][j] if j not in _mutation_index else \
                    1 - selection_pool[-(i + 1)][j] for j in range(p)] # flip mutation points (0 to 1, 1 to 0)

        return selection_pool

    def _early_stopping(self) : # only the difference between the best 10 selection rules are smaller than 0.001 will early stop

        if len(self._fitness) < 10 :
            return False
        else :
            _performance_order = np.flip(np.argsort(self._fitness)) # select performance from highest to lowest
            if self._fitness[_performance_order[0]] - self._fitness[_performance_order[9]] < 0.001 :
                return True
            else :
                return False

    def fit(self, X, y) :

        np.random.seed(self.seed) # set random seed
        
        n, p = X.shape
        self.n_components = int(self.n_components)
        self.n_components = min(self.n_components, p) # prevent selected number of features larger than dataset
        if self.n_components == p :
            warnings.warn('All features selected, no selection performed!')
            self.selection_ = [1 for _ in range(self.n_components)]
            return self

        self.n_generations = int(self.n_generations)
        
        # both probability of crossover and mutation must within range [0, 1]
        self.p_crossover = float(self.p_crossover)
        if self.p_crossover > 1 or self.p_crossover < 0 :
            raise ValueError('Probability of crossover must in [0, 1], get {}.'.format(self.p_crossover))

        self.p_mutation = float(self.p_mutation)
        if self.p_mutation > 1 or self.p_mutation < 0 :
            raise ValueError('Probability of mutation must in [0, 1], get {}.'.format(self.p_mutation))
        
        # select feature selection methods
        # if auto, all default methods will be used; if not, use predefined one
        if self.feature_selection == 'auto' :
            self._feature_sel_methods = self._auto_sel
        elif self.feature_selection == 'random' :
            self.n_initial = int(self.n_initial)
            self._feature_sel_methods = {}
            for i in range(self.n_initial) : # get n_initial random feature selection rule
                self._feature_sel_methods['random_' + str(i + 1)] = self._random
        else :
            self._feature_sel_methods = self.feature_selection

            # check if all methods are available
            for _method in self._feature_sel_methods :
                if _method not in [*self._auto_sel] :
                    raise ValueError(
                        'Not recognizing feature selection methods, only support {}, get {}.' \
                            .format([*self._auto_sel], _method)
                    )
        
        self._fit(X, y)

        return self
    
    def _fit(self, X, y) :
        
        # generate the feature selection pool using 
        _sel_methods = [*self._feature_sel_methods]
        _sel_pool = [] # store all selection rules
        self._fitness = None # store the fitness of every individual
        
        # keep diversity for the pool, selection rule can have different number of features retained
        _iter = int(np.log2(self.n_components))
        for i in range(_iter) :
            n = 2 ** (i + 1)
            for _method in _sel_methods :
                _sel_pool.append(self._feature_sel_methods[_method](X, y, n))

        # loop through generations to run Genetic algorithm and Induction algorithm
        for _gen in range(self.n_generations) :
            _sel_pool = self._GeneticAlgorithm(X, y, _sel_pool)

            # if self._early_stopping() :
            #     break

        self.selection_ = _sel_pool[np.flip(np.argsort(self._fitness))[0]] # selected features, {1, 0} list

        return self

    def transform(self, X) :
        
        # check for all/no feature removed cases
        if self.selection_.count(self.selection_[0]) == len(self.selection_) :
            if self.selection_[0] == 0 :
                warnings.warn('All features removed.')
            elif self.selection_[1] == 1 :
                warnings.warn('No feature removed.')
            else :
                raise ValueError('Not recognizing the selection list!')

        return X.iloc[:, True_index(self.selection_)]

# CHCGA

######################################################################################################################
# Modified Feature Selection from autosklearn

class Densifier() :
    
    '''
    from autosklearn.pipeline.components.feature_preprocessing.densifier import Densifier
    
    Parameters
    ----------
    seed: random seed, default = 1
    '''

    def __init__(
        self, 
        seed = 1
    ):
        self.seed = seed
        self.preprocessor = None

    def fit(self, X, y = None):

        return self

    def transform(self, X):

        from scipy import sparse

        if sparse.issparse(X):
            return X.todense().getA()
        else:
            return X

class ExtraTreesPreprocessorClassification() :

    '''
    from autosklearn.pipeline.components.feature_preprocessing.extra_trees_preproc_for_classification import ExtraTreesPreprocessorClassification
    using sklearn.ensemble.ExtraTreesClassifier
    
    Parameters
    ----------
    n_estimators: Number of trees in forest, default = 100
        
    criterion: Function to measure the quality of a split, default = 'gini'
    supported ("gini", "entropy")
        
    min_samples_leaf: Minimum number of samples required to be at a leaf node, default = 1
        
    min_samples_split: Minimum number of samples required to split a node, default = 2
        
    max_features: Number of features to consider, default = 'auto'
    supported ("auto", "sqrt", "log2")
        
    bootstrap: Whether bootstrap samples, default = False
        
    max_leaf_nodes: Maximum number of leaf nodes accepted, default = None
    
    max_depth: Maximum depth of the tree, default = None
       
    min_weight_fraction_leaf: Minimum weighted fraction of the sum total of weights, default = 0.0

    min_impurity_decrease: Threshold to split if this split induces a decrease of the impurity, default = 0.0
        
    oob_score: Whether to use out-of-bag samples, default = False
    
    n_jobs: Parallel jobs to run, default = 1
    
    verbose: Controls the verbosity, default = 0
    
    class weight: Weights associated with classes, default = None
    supported ("balanced", "balanced_subsample"), dict or list of dicts

    seed: random seed, default = 1
    '''
    
    def __init__(
        self, 
        n_estimators = 100, 
        criterion = 'gini', 
        min_samples_leaf = 1,
        min_samples_split = 2, 
        max_features = 'auto', 
        bootstrap = False, 
        max_leaf_nodes = None,
        max_depth = None, 
        min_weight_fraction_leaf = 0.0, 
        min_impurity_decrease = 0.0,
        oob_score=False, 
        n_jobs = 1, 
        verbose = 0,
        class_weight = None,
        seed = 1
    ) :

        self.n_estimators = n_estimators
        self.estimator_increment = 10
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.class_weight = class_weight
        self.seed = seed

        self.preprocessor = None

    def fit(self, X, y, sample_weight = None) :

        import sklearn.ensemble
        import sklearn.feature_selection

        self.n_estimators = int(self.n_estimators)
        self.max_leaf_nodes = None if self.max_leaf_nodes is None else int(self.max_leaf_nodes)
        self.max_depth = None if self.max_depth is None else int(self.max_depth)

        self.bootstrap = True if self.bootstrap is True else False
        self.n_jobs = int(self.n_jobs)
        self.min_impurity_decrease = float(self.min_impurity_decrease)
        self.max_features = self.max_features
        self.min_samples_leaf = int(self.min_samples_leaf)
        self.min_samples_split = int(self.min_samples_split)
        self.verbose = int(self.verbose)
        max_features = int(X.shape[1] ** float(self.max_features))

        estimator = sklearn.ensemble.ExtraTreesClassifier(
            n_estimators  =self.n_estimators, criterion = self.criterion, max_depth = self.max_depth, \
            min_samples_split = self.min_samples_split, min_samples_leaf = self.min_samples_leaf, \
            bootstrap = self.bootstrap, max_features = max_features, max_leaf_nodes = self.max_leaf_nodes, \
            min_impurity_decrease = self.min_impurity_decrease, oob_score = self.oob_score, n_jobs = self.n_jobs, \
            verbose = self.verbose, random_state = self.seed, class_weight = self.class_weight
        )
        estimator.fit(X, y, sample_weight=sample_weight)

        self.preprocessor = sklearn.feature_selection.SelectFromModel(estimator = estimator, threshold = 'mean', prefit = True)
        
        return self

    def transform(self, X):

        if self.preprocessor is None:
            raise NotImplementedError

        return self.preprocessor.transform(X)

class ExtraTreesPreprocessorRegression() :

    '''
    from autosklearn.pipeline.components.feature_preprocessing.extra_trees_preproc_for_regression import ExtraTreesPreprocessorRegression
    using sklearn.ensemble.ExtraTreesRegressor

    Parameters
    ----------
    n_estimators: Number of trees in forest, default = 100
        
    criterion: Function to measure the quality of a split, default = 'squared_error'
    supported ("squared_error", "mse", "absolute_error", "mae")
        
    min_samples_leaf: Minimum number of samples required to be at a leaf node, default = 1
        
    min_samples_split: Minimum number of samples required to split a node, default = 2
        
    max_features: Number of features to consider, default = 'auto'
    supported ("auto", "sqrt", "log2")
        
    bootstrap: Whether bootstrap samples, default = False
        
    max_leaf_nodes: Maximum number of leaf nodes accepted, default = None
    
    max_depth: Maximum depth of the tree, default = None
       
    min_weight_fraction_leaf: Minimum weighted fraction of the sum total of weights, default = 0.0
        
    oob_score: Whether to use out-of-bag samples, default = False
    
    n_jobs: Parallel jobs to run, default = 1
    
    verbose: Controls the verbosity, default = 0
    
    seed: random seed, default = 1
    '''

    def __init__(
        self, 
        n_estimators = 100, 
        criterion = 'squared_error', 
        min_samples_leaf = 1,
        min_samples_split = 2, 
        max_features = 'auto',
        bootstrap = False, 
        max_leaf_nodes = None, 
        max_depth = None,
        min_weight_fraction_leaf = 0.0,
        oob_score = False, 
        n_jobs = 1, 
        verbose = 0, 
        seed = 1
    ):

        self.n_estimators = n_estimators
        self.estimator_increment = 10
        if criterion not in ("mse", "friedman_mse", "mae"):
            raise ValueError("'criterion' is not in ('mse', 'friedman_mse', "
                             "'mae'): %s" % criterion)
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.seed = seed

        self.preprocessor = None

    def fit(self, X, y) :

        import sklearn.ensemble
        import sklearn.feature_selection

        self.n_estimators = int(self.n_estimators)
        self.min_samples_leaf = int(self.min_samples_leaf)
        self.min_samples_split = int(self.min_samples_split)
        self.max_features = float(self.max_features)
        self.bootstrap = True if self.bootstrap is True else False
        self.n_jobs = int(self.n_jobs)
        self.verbose = int(self.verbose)

        self.max_leaf_nodes = None if self.max_leaf_nodes is None else int(self.max_leaf_nodes)
        self.max_depth = None if self.max_depth is None else int(self.max_leaf_nodes)
        self.min_weight_fraction_leaf = float(self.min_weight_fraction_leaf)

        num_features = X.shape[1]
        max_features = int(float(self.max_features) * (np.log(num_features) + 1))

        # Use at most half of the features
        max_features = max(1, min(int(X.shape[1] / 2), max_features))
        estimator = sklearn.ensemble.ExtraTreesRegressor(
            n_estimators = self.n_estimators, criterion = self.criterion, max_depth = self.max_depth, \
            min_samples_split = self.min_samples_split, min_samples_leaf = self.min_samples_leaf, bootstrap = self.bootstrap, \
            max_features = max_features, max_leaf_nodes = self.max_leaf_nodes, oob_score = self.oob_score, n_jobs = self.n_jobs, \
            verbose=self.verbose, min_weight_fraction_leaf = self.min_weight_fraction_leaf, random_state = self.seed
        )

        estimator.fit(X, y)
        self.preprocessor = sklearn.feature_selection.SelectFromModel(estimator = estimator, threshold = 'mean', prefit = True)

        return self
    
    def transform(self, X):

        if self.preprocessor is None:
            raise NotImplementedError

        return self.preprocessor.transform(X)

class FastICA() :

    '''
    from autosklearn.pipeline.components.feature_preprocessing.fast_ica import FastICA
    using 

    Parameters
    ----------
    algorithm: Apply parallel or deflational algorithm, default = 'parallel'
    supported ('parallel', 'deflation')

    whiten: If false, no whitening is performed, default = True

    fun: Functional form of the G function used, default = 'logcosh'
    supported ('logcosh', 'exp', 'cube') or callable

    n_components: Number of components to retain, default = None

    seed: random seed, default = 1
    '''

    def __init__(
        self, 
        algorithm = 'parallel', 
        whiten = True, 
        fun = 'logcosh', 
        n_components = None,
        seed = 1
    ) :
        self.algorithm = algorithm
        self.whiten = whiten
        self.fun = fun
        self.n_components = n_components
        self.seed = seed

        self.preprocessor = None

    def fit(self, X, y=None):

        import sklearn.decomposition

        self.n_components = None if self.n_components is None else int(self.n_components)

        self.preprocessor = sklearn.decomposition.FastICA(
            n_components = self.n_components, algorithm = self.algorithm, fun = self.fun, \
            whiten = self.whiten, random_state = self.seed
        )
        
        # Make the RuntimeWarning an Exception!
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message='array must not contain infs or NaNs')
            try:
                self.preprocessor.fit(X)
            except ValueError as e:
                if 'array must not contain infs or NaNs' in e.args[0]:
                    raise ValueError("Bug in scikit-learn: "
                                     "https://github.com/scikit-learn/scikit-learn/pull/2738")

        return self

    def transform(self, X):

        if self.preprocessor is None:
            raise NotImplementedError()

        return self.preprocessor.transform(X)

class FeatureAgglomeration() :

    '''
    from autosklearn.pipeline.components.feature_preprocessing.feature_agglomeration import FeatureAgglomeration
    using 

    Parameters
    ----------
    n_clusters: Number of clusters, default = 2
        
    affinity: Metric used to compute the linkage, default = 'euclidean'
    supported ("euclidean", "l1", "l2", "manhattan", "cosine", or 'precomputed')
        
    linkage: Linkage criterion, default = 'ward'
    supported ("ward", "complete", "average", "single")

    pooling_func: Combines the values of agglomerated features into a single value, default = np.mean
    
    seed: random seed, default = 1
    '''

    def __init__(
        self, 
        n_clusters = 2, 
        affinity = 'euclidean', 
        linkage = 'ward', 
        pooling_func = np.mean,
        seed = 1
    ) :
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.linkage = linkage
        self.pooling_func = pooling_func
        self.seed = seed

        self.pooling_func_mapping = dict(mean = np.mean, median = np.median, max = np.max)

        self.preprocessor = None

    def fit(self, X, y = None) :

        import sklearn.cluster

        self.n_clusters = int(self.n_clusters)

        n_clusters = min(self.n_clusters, X.shape[1])

        if not callable(self.pooling_func):
            self.pooling_func = self.pooling_func_mapping[self.pooling_func]

        self.preprocessor = sklearn.cluster.FeatureAgglomeration(
            n_clusters = n_clusters, affinity = self.affinity, linkage = self.linkage, pooling_func = self.pooling_func
        )
        self.preprocessor.fit(X)

        return self

    def transform(self, X):

        if self.preprocessor is None:
            raise NotImplementedError()

        return self.preprocessor.transform(X)

class KernelPCA() :

    '''
    from autosklearn.pipeline.components.feature_preprocessing.kernel_pca import KernelPCA
    using sklearn.decomposition.KernelPCA

    Parameters
    ----------
    n_components: number of features to retain, default = None

    kernel: kernel used, default = 'linear'
    supported (linear', 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed')

    degree: Degree for poly kernels, default = 3

    gamma: Kernel coefficient, default = 0.25

    coef0: Independent term in poly and sigmoid kernels, default = 0.0

    seed: random seed, default = 1
    '''

    def __init__(
        self, 
        n_components = None, 
        kernel = 'linear', 
        degree = 3, 
        gamma = 0.25, 
        coef0 = 0.0,
        seed = 1
    ):
        self.n_components = n_components
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.seed = seed

        self.preprocessor = None

    def fit(self, X, y = None) :

        import sklearn.decomposition

        self.n_components = None if self.n_components is None else int(self.n_components)
        self.degree = int(self.degree)
        self.gamma = float(self.gamma)
        self.coef0 = float(self.coef0)

        self.preprocessor = sklearn.decomposition.KernelPCA(
            n_components = self.n_components, kernel = self.kernel, degree = self.degree, gamma = self.gamma, \
            coef0 = self.coef0, remove_zero_eig = True, random_state = self.seed)

        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            self.preprocessor.fit(X)

        if len(self.preprocessor.alphas_ / self.preprocessor.lambdas_) == 0:
            raise ValueError('All features removed.')

        return self

    def transform(self, X) :

        if self.preprocessor is None:
            raise NotImplementedError()

        with warnings.catch_warnings():
            warnings.filterwarnings("error")

            _X = self.preprocessor.transform(X)

            if _X.shape[1] == 0:
                raise ValueError("KernelPCA removed all features!")

            return _X

class RandomKitchenSinks() :

    '''
    from autosklearn.pipeline.components.feature_preprocessing.kitchen_sinks import RandomKitchenSinks
    using sklearn.kernel_approximation.RBFSampler

    Parameters
    ----------
    gamma: use to determine standard variance of random weight table, default = 1
    Parameter of RBF kernel: exp(-gamma * x^2).

    n_components: number of samples per original feature, default = 100

    seed: random seed, default = 1
    '''

    def __init__(
        self,
        gamma = 1.0,
        n_components = 100,
        seed = 1
    ) :
        self.gamma = gamma
        self.n_components = n_components
        self.seed = seed

        self.preprocessor = None

    def fit(self, X, y = None) :

        import sklearn.kernel_approximation

        self.n_components = int(self.n_components)
        self.gamma = float(self.gamma)

        self.preprocessor = sklearn.kernel_approximation.RBFSampler(self.gamma, self.n_components, self.seed)
        self.preprocessor.fit(X)

        return self

    def transform(self, X) :

        if self.preprocessor is None:
            raise NotImplementedError()

        return self.preprocessor.transform(X)

class LibLinear_Preprocessor() :

    '''
    from autosklearn.pipeline.components.feature_preprocessing.liblinear_svc_preprocessor import LibLinear_Preprocessor
    using import sklearn.svm, from sklearn.feature_selection import SelectFromModel

    Parameters
    ----------
    penalty: Norm used in the penalization, default = 'l2'
    supported ('l1', 'l2')

    loss: Loss function, default = 'squared_hinge'
    supported ('hinge', 'squared_hinge')

    dual: Whether to solve the dual or primal, default = True

    tol: Stopping criteria, default = 1e-4

    C: Regularization parameter, default = 1.0

    multi_class: Multi-class strategy, default = 'ovr'
    supported ('ovr', 'crammer_singer')

    fit_intercept: Whether to calculate the intercept, default = True

    intercept_scaling: Intercept scaling rate, default = 1

    class_weight: Class weight, default = None
    supported dict or 'balanced'
 
    seed: random seed, default = 1
    '''

    def __init__(
        self, 
        penalty = 'l2', 
        loss = 'squared_hinge', 
        dual = True, 
        tol = 1e-4, 
        C = 1.0, 
        multi_class = 'ovr',
        fit_intercept = True, 
        intercept_scaling = 1, 
        class_weight = None,
        seed = 1
    ):
        self.penalty = penalty
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.C = C
        self.multi_class = multi_class
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.seed = seed

        self.preprocessor = None

    def fit(self, X, y, sample_weight = None) :

        import sklearn.svm
        from sklearn.feature_selection import SelectFromModel

        self.C = float(self.C)
        self.tol = float(self.tol)
        self.intercept_scaling = float(self.intercept_scaling)

        estimator = sklearn.svm.LinearSVC(
            penalty = self.penalty, loss = self.loss, dual = self.dual, tol=self.tol, C = self.C, \
            class_weight = self.class_weight, fit_intercept = self.fit_intercept, \
            intercept_scaling = self.intercept_scaling, multi_class = self.multi_class, random_state=self.seed)
        estimator.fit(X, y)

        self.preprocessor = SelectFromModel(estimator = estimator, threshold = 'mean', prefit = True)

        return self
    
    def transform(self, X) :

        if self.preprocessor is None:
            raise NotImplementedError()
        
        return self.preprocessor.transform(X)

class Nystroem() :

    '''
    from autosklearn.pipeline.components.feature_preprocessing.nystroem_sampler import Nystroem
    using sklearn.kernel_approximation.Nystroem

    Parameters
    ----------
    kernel: Kernel map to be approximated, default = 'rbf'
    supported: ('additive_chi2', 'chi2', 'linear', 'poly', 'polynomial', 'rbf', 'laplacian', 
    'sigmoid', 'cosine' )

    n_components: Number of features to retain, default = 100

    gamma: Gamma parameter, default = 1.0

    degree: Degree of the polynomial kernel, default = 3

    coef0: Zero coefficient for polynomial and sigmoid kernels, default = 1
    
    seed: random seed, default = 1
    '''
    
    def __init__(
        self,
        kernel = 'rbf', 
        n_components = 100, 
        gamma = 1.0, 
        degree = 3,
        coef0 = 1, 
        seed = 1
    ) :
        self.kernel = kernel
        self.n_components = n_components, 
        self.gamma = gamma, 
        self.degree = degree,
        self.coef0 = coef0, 
        self.seed = seed

        self.preprocessor = None

    def fit(self, X, y = None) :

        import sklearn.kernel_approximation

        self.n_components = int(self.n_components)
        self.gamma = float(self.gamma)
        self.degree = int(self.degree)
        self.coef0 = float(self.coef0)

        if self.kernel == 'chi2' :
            X[X < 0] = 0.

        self.preprocessor = sklearn.kernel_approximation.Nystroem(
            kernel = self.kernel, n_components = self.n_components, gamma = self.gamma, degree = self.degree, \
            coef0 = self.coef0, random_state = self.seed
        )
        self.preprocessor.fit(X)

        return self

    def transform(self, X) :

        if self.preprocessor is None :
            raise NotImplementedError()

        if self.kernel == 'chi2' :
            X[X < 0] = 0.

        return self.preprocessor.transform(X)

class PCA() :

    '''
    from autosklearn.pipeline.components.feature_preprocessing.pca import PCA
    using sklearn.decomposition.PCA

    Parameters
    ----------
    n_components: numer of features to retain, default = None
    all features will be retained

    whiten: default = False
    if True, the `components_` vectors will be modified to ensure uncorrelated outputs

    seed: random seed, default = 1
    '''

    def __init__(
        self,
        n_components = None,
        whiten = False,
        seed = 1
    ) :
        self.n_components = n_components
        self.whiten = whiten
        self.seed = seed

        self.preprocessor = None

    def fit(self, X, y = None) :

        import sklearn.decomposition

        self.n_components = None if self.n_components is None else int(self.n_components)

        self.preprocessor = sklearn.decomposition.PCA(
            n_components = self.n_components, whiten = self.whiten, copy = True
        )
        self.preprocessor.fit(X)

        return self

    def transform(self, X) :

        if self.preprocessor is None :
            raise NotImplementedError()

        return self.preprocessor.transform(X)

class PolynomialFeatures() :

    '''
    from autosklearn.pipeline.components.feature_preprocessing.polynomial import PolynomialFeatures
    using sklearn.preprocessing.PolynomialFeatures

    Parameters
    ----------
    degree: degree of polynomial features, default = 2

    interaction_only: if to only to conclude interaction terms, default = False

    include_bias: if to conclude bias term, default = True

    seed: random seed, default = 1
    '''

    def __init__(
        self,
        degree = 2,
        interaction_only = False, 
        include_bias = True, 
        seed = 1
    ) :
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.seed = seed
        
        self.preprocessor = None
    
    def fit(self, X, y) :

        import sklearn.preprocessing

        self.degree = int(self.degree)

        self.preprocessor = sklearn.preprocessing.PolynomialFeatures(degree = self.degree, \
            interaction_only = self.interaction_only, include_bias = self.include_bias)
        self.preprocessor.fit(X, y)

    def transform(self, X) :

        if self.preprocessor is None :
            raise NotImplementedError()
        
        return self.preprocessor.transform(X)

class RandomTreesEmbedding() :

    '''
    from autosklearn.pipeline.components.feature_preprocessing.random_trees_embedding import RandomTreesEmbedding
    using sklearn.ensemble.RandomTreesEmbedding

    Parameters
    ----------
    n_estimators: Number of trees in the forest to train, deafult = 100
    
    max_depth: Maximum depth of the tree, default = 5
        
    min_samples_split: Minimum number of samples required to split a node, default = 2
    
    min_samples_leaf: Minimum number of samples required to be at a leaf node, default = 1
    
    min_weight_fraction_leaf: Minimum weighted fraction of the sum total of weights, default = 0.
    
    max_leaf_nodes: Maximum number of leaf nodes, deafult = None
    
    bootstrap: Mark if bootstrap, default = False 
    
    sparse_output: If output as sparse format (with False), default = False
    True for dense output

    n_jobs: Number of jobs run in parallel, default = 1
        
    seed: random seed, default = 1
    '''

    def __init__(
        self, 
        n_estimators = 100, 
        max_depth = 5, 
        min_samples_split = 2,
        min_samples_leaf = 1, 
        min_weight_fraction_leaf = 0., 
        max_leaf_nodes = None,
        bootstrap = False, 
        sparse_output = False, 
        n_jobs = 1, 
        seed = 1
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.bootstrap = bootstrap
        self.sparse_output = sparse_output
        self.n_jobs = n_jobs
        self.seed = seed

        self.preprocessor = None

    def fit(self, X, y = None) :

        import sklearn.ensemble

        self.n_estimators = int(self.n_estimators)
        self.max_depth = int(self.max_depth)
        self.min_samples_split = int(self.min_samples_split)
        self.min_samples_leaf = int(self.min_samples_leaf)
        self.max_leaf_nodes = int(self.max_leaf_nodes)
        self.min_weight_fraction_leaf = float(self.min_weight_fraction_leaf)
        self.bootstrap = (True if self.bootstrap is None else False)

        self.preprocessor = sklearn.ensemble.RandomTreesEmbedding(
            n_estimators = self.n_estimators,
            max_depth = self.max_depth,
            min_samples_split = self.min_samples_split,
            min_samples_leaf = self.min_samples_leaf,
            max_leaf_nodes = self.max_leaf_nodes,
            sparse_output = self.sparse_output,
            n_jobs = self.n_jobs,
            random_state=self.seed
        )

        self.preprocessor.fit(X)

        return self

    def transform(self, X) :

        if self.preprocessor == None :
            raise NotImplementedError()

        return self.preprocessor.transform(X)


'''
from autosklearn.pipeline.components.feature_preprocessing.select_percentile import SelectPercentileBase
using sklearn.feature_selection.SelectPercentile
'''

class SelectPercentileClassification() :

    '''
    from autosklearn.pipeline.components.feature_preprocessing.select_percentile_classification import SelectPercentileClassification
    using sklearn.feature_selection.SelectPercentile

    Parameters
    ----------
    percentile: Percent of features to keep, default = 10

    score_func: default = 'chi2'
    supported mode ('chi2', 'f_classif', 'mutual_info_classif')

    seed: random seed, default = 1
    '''

    def __init__(
        self,
        percentile = 10,
        score_func = 'chi2',
        seed = 1
    ) :
        self.percentile = int(float(percentile))
        self.seed = seed
        import sklearn.feature_selection

        if score_func == 'chi2' :
            self.score_func = sklearn.feature_selection.chi2
        elif score_func == 'f_classif' :
            self.score_func = sklearn.feature_selection.f_classif
        elif score_func == 'mutual_info_classif' :
            self.score_func = partial(sklearn.feature_selection.mutual_info_classif, \
                random_state = self.seed)
        else :
            raise ValueError('Not recognizing score_func, supported ("chi2", "f_classif", "mutual_info_classif", \
                get {})'.format(score_func))

        self.preprocessor = None

    def fit(self, X, y) :

        import sklearn.feature_selection

        if self.score_func == sklearn.feature_selection.chi2 :
            X[X < 0] = 0

        self.preprocessor = sklearn.feature_selection.SelectPercentile(score_func = self.score_func, \
            percentile = self.percentile)
        self.preprocessor.fit(X, y)

        return self

    def transform(self, X) :

        import sklearn.feature_selection

        if self.preprocessor is None :
            raise NotImplementedError()

        if self.score_func == sklearn.feature_selection.chi2 :
            X[X < 0] = 0

        _X = self.preprocessor.transform(X)

        if _X.shape[1] == 0 :
            raise ValueError('All features removed.')

        return _X

class SelectPercentileRegression() :
    
    '''
    from autosklearn.pipeline.components.feature_preprocessing.select_percentile_regression import SelectPercentileRegression
    using sklearn.feature_selection.SelectPercentile

    Parameters
    ----------
    percentile: Percent of features to keep, default = 10

    score_func: default = 'f_regression'
    supported mode ('f_regression', 'mutual_info_regression')

    seed: random seed, default = 1
    '''

    def __init__(
        self,
        percentile = 10,
        score_func = 'f_regression',
        seed = 1
    ) :
        self.percentile = int(float(percentile))
        self.seed = seed
        import sklearn.feature_selection

        if score_func == 'f_regression' :
            self.score_func = sklearn.feature_selection.f_regression
        elif score_func == 'mutual_info_regression' :
            self.score_func = partial(sklearn.feature_selection.mutual_info_regression, \
                random_state = self.seed)
            self.mode = 'percentile'
        else :
            raise ValueError('Not recognizing score_func, only support ("f_regression", "mutual_info_regression"), \
                get {}'.format(score_func))
        
        self.preprocessor = None

    def fit(self, X, y) :

        import sklearn.feature_selection

        self.preprocessor = sklearn.feature_selection.SelectPercentile(score_func = self.score_func, \
            percentile = self.percentile)
        self.preprocessor.fit(X, y)

        return self

    def transform(self, X) :

        import sklearn.feature_selection

        if self.preprocessor is None :
            raise NotImplementedError()

        _X = self.preprocessor.transform(X)

        if _X.shape[1] == 0 :
            warnings.warn('All features removed.')
        
        return _X

class SelectClassificationRates() :

    '''
    from autosklearn.pipeline.components.feature_preprocessing.select_rates_classification import SelectClassificationRates
    using sklearn.feature_selection.GenericUnivariateSelect

    Parameters
    ----------
    alpha: parameter of corresponding mode, default = 1e-5

    mode: Feature selection mode, default = 'fpr'
    supported mode ('percentile', 'k_best', 'fpr', 'fdr', 'fwe') 

    score_func: default = 'chi2'
    supported mode ('chi2', 'f_classif', 'mutual_info_classif')

    seed: random seed, default = 1
    '''

    def __init__(
        self,
        alpha = 1e-5,
        mode = 'fpr',
        score_func = 'chi2',
        seed = 1
    ) :
        self.alpha = alpha
        self.mode = mode
        self.seed = seed
        import sklearn.feature_selection

        if score_func == 'chi2' :
            self.score_func = sklearn.feature_selection.chi2
        elif score_func == 'f_classif' :
            self.score_func = sklearn.feature_selection.f_classif
        elif score_func == 'mutual_info_classif' :
            self.score_func = partial(sklearn.feature_selection.mutual_info_classif, \
                random_state = self.seed)
            self.mode = 'percentile'
        else :
            raise ValueError('Not recognizing score_func, supported ("chi2", "f_classif", "mutual_info_classif", \
                get {})'.format(score_func))
        
        self.preprocessor = None

    def fit(self, X, y) :
        import sklearn.feature_selection

        self.alpha = float(self.alpha)

        if self.score_func == sklearn.feature_selection.chi2 :
            X[X < 0] = 0

        self.preprocessor = sklearn.feature_selection.GenericUnivariateSelect(score_func = self.score_func, \
            param = self.alpha, mode = self.mode)
        self.preprocessor.fit(X, y)

        return self

    def transform(self, X) :

        import sklearn.feature_selection

        if self.score_func == sklearn.feature_selection.chi2 :
            X[X < 0] = 0

        if self.preprocessor is None :
            raise NotImplementedError()
        
        _X = self.preprocessor.transform(X)
        
        if _X.shape[1] == 0 :
            warnings.warn('All features removed.')

        return _X

class SelectRegressionRates() :

    '''
    from autosklearn.pipeline.components.feature_preprocessing.select_rates_regression import SelectRegressionRates
    using sklearn.feature_selection.GenericUnivariateSelect

    Parameters
    ----------
    alpha: parameter of corresponding mode, default = 1e-5

    mode: Feature selection mode, default = 'percentile'
    supported mode ('percentile', 'k_best', 'fpr', 'fdr', 'fwe') 

    score_func: default = 'f_regression'
    supported mode ('f_regression', 'mutual_info_regression')

    seed: random seed, default = 1
    '''

    def __init__(
        self,
        alpha = 1e-5,
        mode = 'percentile',
        score_func = 'f_regression',
        seed = 1
    ) :
        self.alpha = alpha
        self.mode = mode
        self.seed = seed
        import sklearn.feature_selection

        if score_func == 'f_regression' :
            self.score_func = sklearn.feature_selection.f_regression
        elif score_func == 'mutual_info_regression' :
            self.score_func = partial(sklearn.feature_selection.mutual_info_regression, \
                random_state = self.seed)
            self.mode = 'percentile'
        else :
            raise ValueError('Not recognizing score_func, only support ("f_regression", "mutual_info_regression"), \
                get {}'.format(score_func))
        
        self.preprocessor = None

    def fit(self, X, y) :

        import sklearn.feature_selection
        alpha = float(self.alpha)
        self.preprocessor = sklearn.feature_selection.GenericUnivariateSelect(score_func = self.score_func, \
            param = alpha, mode = self.mode)
        
        self.preprocessor.fit(X, y)
        
        return self

    def transform(self, X) :

        if self.preprocessor is None :
            raise NotImplementedError()
        
        _X = self.preprocessor.transform(X)

        if _X.shape[1] == 0 :
            warnings.warn('All features removed.')
        
        return _X

class TruncatedSVD() :
    
    '''
    from autosklearn.pipeline.components.feature_preprocessing.truncatedSVD import TruncatedSVD
    Truncated SVD using sklearn.decomposition.TruncatedSVD

    Parameters
    ----------
    n_components: Number of features to retain, default = None
    will be set to p - 1, and capped at p -1 for any input

    seed: random seed, default = 1
    '''

    def __init__(
        self,
        n_components = None,
        seed = 1
    ) :
        self.n_components = n_components
        self.seed = seed
        self.preprocessor = None

    def fit(self, X, y) :

        if self.n_components == None :
            self.n_components = X.shape[1] - 1
        else :
            self.n_components = int(self.n_components)
        n_components = min(self.n_components, X.shape[1] - 1) # cap n_components

        from sklearn.decomposition import TruncatedSVD
        self.preprocessor = TruncatedSVD(
            n_components, algorithm = 'randomized', random_state = self.seed
        )
        self.preprocessor.fit(X, y)

        return self
    
    def transform(self, X) :

        if self.preprocessor is None :
            raise NotImplementedError()
        
        return self.preprocessor.transform(X)

######################################################################################################################
