from multiprocessing.sharedctypes import Value
import warnings
import time
import numbers
import warnings
import numpy as np
from numpy.lib.arraysetops import isin
import pandas as pd
import scipy
import scipy.linalg
import sklearn
from sklearn.feature_selection._univariate_selection import f_classif
from sklearn.utils.extmath import stable_cumsum, svd_flip
from sympy import solve
import itertools

# feature selection from autosklearn
from autosklearn.pipeline.components.feature_preprocessing.no_preprocessing import NoPreprocessing
from autosklearn.pipeline.components.feature_preprocessing.densifier import Densifier
from autosklearn.pipeline.components.feature_preprocessing.extra_trees_preproc_for_classification import ExtraTreesPreprocessorClassification
from autosklearn.pipeline.components.feature_preprocessing.extra_trees_preproc_for_regression import ExtraTreesPreprocessorRegression
from autosklearn.pipeline.components.feature_preprocessing.fast_ica import FastICA
from autosklearn.pipeline.components.feature_preprocessing.feature_agglomeration import FeatureAgglomeration
from autosklearn.pipeline.components.feature_preprocessing.kernel_pca import KernelPCA
from autosklearn.pipeline.components.feature_preprocessing.kitchen_sinks import RandomKitchenSinks
from autosklearn.pipeline.components.feature_preprocessing.liblinear_svc_preprocessor import LibLinear_Preprocessor
from autosklearn.pipeline.components.feature_preprocessing.nystroem_sampler import Nystroem
from autosklearn.pipeline.components.feature_preprocessing.pca import PCA
from autosklearn.pipeline.components.feature_preprocessing.polynomial import PolynomialFeatures
from autosklearn.pipeline.components.feature_preprocessing.random_trees_embedding import RandomTreesEmbedding
from autosklearn.pipeline.components.feature_preprocessing.select_percentile import SelectPercentileBase
from autosklearn.pipeline.components.feature_preprocessing.select_percentile_classification import SelectPercentileClassification
from autosklearn.pipeline.components.feature_preprocessing.select_percentile_regression import SelectPercentileRegression
from autosklearn.pipeline.components.feature_preprocessing.select_rates_classification import SelectClassificationRates
from autosklearn.pipeline.components.feature_preprocessing.select_rates_regression import SelectRegressionRates
from autosklearn.pipeline.components.feature_preprocessing.truncatedSVD import TruncatedSVD

from ._utils import nan_cov, maxloc, empirical_covariance, _class_means, _class_cov

# Truncated SVD and PCA, both uses SVD to decompose the metrics, however, PCA focus on centered data,
# while Truncated SVD is beneficial on large sparse dataset. (two are the same if data already centered)

# class TruncatedSVD() :

#     def __init__(
#         self,
#         n_components = 2,
#         *,
#         algorithm="randomized",
#         n_iter=5,
#         random_state=None,
#         tol=0.0
#     ) :
#         self.n_compoents = n_components
#         self.algorithm = algorithm
#         self.n_iter = 5
#         self.random_state = random_state
#         self.tol = tol
#         from sklearn.decomposition import TruncatedSVD
#         self.model = TruncatedSVD(
#             n_components = self.n_compoents,
#             algorithm = self.algorithm,
#             n_iter = self.n_iter,
#             random_state = self.random_state,
#             tol = self.tol
#         )

#     def fit(self, X, y = None) :
#         self.model.fit(X, y = None)
    
#     def transform(self, X) :
#         return self.model.transform(X)

#     def fit_transform(self, X, y = None) :
#         return self.model.fit_transform(X, y)

#     def inverse_transform(self, X) :
#         return self.model.inverse_transform(X)   

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

    def fit(self, X) :

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

        self.means_ = _class_means(X, y)
        self.covariance_ = _class_cov(X, y, self.priors_)

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

    def fit(self, X) :
        
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
            _empty = (y == None).all()
        except AttributeError :
            _empty = (y == None)
        if _empty :
            raise ValueError('Must have response!')
        
        if self.criteria == 'Pearson' :          
            self._score = self._Pearson_Corr(X, y)
        elif self.criteria == 'MI' :
            self._score = self._MI(X)

    def _Pearson_Corr(self, X, y) :

        features = list(X.columns)
        result = []
        for _column in features :
            result.append((nan_cov(X[_column], y) / np.sqrt(nan_cov(X[_column]) * nan_cov(y)))[0][0])
        
        return result

    def _MI(self, X, y) :
        
        if len(X) != len(y) :
            raise ValueError('X and y not same size!')
 
        features = list(X.columns)
        _y_column = list(y.columns)
        result = []

        _y_pro = y.groupby(_y_column[0]).size().div(len(X)).values
        _H_y = - sum(item * np.log(item) for item in _y_pro)

        for _column in features :

            _X_y = pd.concat([X[_column], y], axis = 1)
            _pro = _X_y.groupby([_column, _y_column[0]]).size().div(len(X))
            _X_pro = X[_column].groupby(_column).size().div(len(X))
            _H_y_X = - sum(_pro[i] * np.log(_pro[i] / _X_pro.loc[_X_pro.index == _pro.index[i][0]]) \
                for i in range(len(X)))
            result.append(_H_y - _H_y_X)

        return result
    
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
    Adapative Sequential Forward Floating Selection (ASFFS)
    Mostly, ASFFS performs the same as Sequential Floating Forward Selection (SFFS),
    where only one feature is considered as a time. But, when the selected features are coming
    close to the predefined maximum, a adapative generalization limit will be activated, and 
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
            _objective_list.append(1 / _obj(y, _predict)) # the goal is to maximize the obejctive function

        return _subset[maxloc(_objective_list)], _objective_list[maxloc(_objective_list)]

    def _Backward_Objective(self, selected, o, X, y) :

        _subset = list(itertools.combinations(selected, o))
        _comb_subset = [[_full for _full in selected if _full not in item] for item in _subset] # remove new features from selected features 

        _objective_list = []
        if self.model == 'Linear' :
            from sklearn.linear_model import LinearRegression
            _model = LinearRegression()
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
            _objective_list.append(1 / _obj(y, _predict)) # the goal is to maximize the obejctive function

        return _subset[maxloc(_objective_list)], _objective_list[maxloc(_objective_list)]

    def fit(self, X, y) :
        
        n, p = X.shape
        features = list(X.columns)

        if self.n_components == None :
            _n_components = max(max(20, p), int(0.5 * p))
        else :
            _n_components = self.n_components
        if self.b == None :
            _b = max(5, int(0.05 * p))

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
                    break
                else :
                    if _o < _r :
                        _o += 1
                    else :
                        _k -= 1 # the marked in J_max and _subset_max are considered as best for _k features
                        _selected = self._subset_max[_k].copy() # read stored best subset
                        _unselected = features.copy()
                        for _f in _selected :
                            _unselected.remove(_f)
                        break
        
        self.selected_ = _selected
        return self

    def transform(self, X) :

        return X.loc[:, self.selected_]
            

# Genetic Algorithm (GA)
# CHCGA
