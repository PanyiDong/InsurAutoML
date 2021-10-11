from multiprocessing.sharedctypes import Value
import time
import numbers
import numpy as np
from numpy.lib.arraysetops import isin
import pandas as pd
import scipy
import scipy.linalg
import sklearn
from sklearn.utils.extmath import stable_cumsum
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

from ._utils import nan_cov, maxloc

class PCA_FeatureSelection() :

    '''
    Principal Component Analysis
    
    Use Singular Value Decomposition (SVD) to project data to a lower dimensional 
    space, and thus achieve feature selection.

    Methods used:
    Full SVD:
    Trucated SVD:
    Randomized truncated SVD:

    Parameters
    ----------
    n_components: remaining features after selection, default = None

    solver: the method to perform SVD, default = 'auto'
    all choices ('auto', 'full', 'truncated', 'randomized')

    seed: random seed, default = 1
    '''

    def __init__(
        self, 
        n_components = None, 
        solver = 'auto', 
        seed = 1
    ):
        self.n_components = n_components
        self.solver = solver
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

        if self.solver == 'full' :
            return self._fit_full(X, n_components)
        elif self.solver == 'truncated' :
            return self._fit_truncated(X, n_components)
        elif self.solver == 'randomized' :
            return self._fit_randomized(X, n_components)
        else :
            raise ValueError('No solver selected!')

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
        U, S, V = scipy.linalg.svd(X, full_matrices = False)

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

    def __init__(self, gamma = 1., n_components = 100, seed = None):
        self.gamma = gamma
        self.n_components = n_components
        self.seed = seed

    def fit(self, X) :
        
        if isinstance(X, list) :
            n_features = len(X[0])
        else :
            n_features = X.shape[1]

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

    threshold: threshold to retain features, default = 0.1
    '''

    def __init__(
        self,
        criteria = 'Pearson',
        threshold = 0.1
    ) :
        self.criteria = criteria
        self.threshold = threshold

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

    def _Pearson_Corr(X, y) :

        features = list(X.columns)
        result = []
        for _column in features :
            result.append(nan_cov(X[_column], y) / np.sqrt(nan_cov(X[_column]) * nan_cov(y)))
        
        return result

    def _MI(X, y) :
        
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

        return X.loc[:, self._score > self.criteria]

# FeatureWrapper

# Exhaustive search for optimal feature combination

# Sequential Feature Selection (SFS)
# Sequential Backward Selection (SBS)
# Sequential Floating Forward Selection (SFFS)
# Adapative Sequential Forward Floating Selection (ASFFS)
class ASFFS() :

    '''
    Adapative Sequential Forward Floating Selection (ASFFS)

    Parameters
    ----------
    '''

    def __init__(
        self,
        d = None,
        b = None,
        r_max = 5,
        Delta = 0,
        model = 'Linear',
        objective = 'MSE'
    ) :
        self.d = d
        self.b = b
        self.r_max = r_max
        self.Delta = Delta
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


    def fit(self, X, y) :
        
        n, p = X.shape
        features = list(X.columns)

        if self.d == None :
            _d = max(max(20, n), int(0.5 * n))
        if self.b == None :
            _b = max(5, int(0.05 * n))

        _k = 0
        self.J_max = [0 for _ in range(p)] # mark the most significant objective function value
        self._subset_max = [[] for _ in range(p)] # mark the best performing subset features
        _unselected = features
        _selected = [] # selected  feature stored here, not selected will be stored in features       

        while True :
            
            # Forward Phase
            _r = self.generalization_limit(_k, _d, _b)
            _o = 1
            
            while _o <= _r :
            
                _new_feature, _max_obj = self._Forward_Objective(_selected, _unselected, _o, X, y)

                if _max_obj > self.J_max[_k + _o] :
                    self.J_max[_k + _o] = _max_obj
                    _k += _o
                    for _f in _new_feature : # add new features and remove these features from the pool
                        _selected.append(_f)
                    _unselected.remove(_new_feature)
                    self._subset_max[_k + _o] = _selected
                else :
                    if _o < _r :
                        _o += 1
                    else :
                        _k += 1 # the marked in J_max and _subset_max are considered as best for _k features

            # Termination Condition
            if _k >= _d + self.Delta :
                break
            
            # Backward Phase
            _r = self.generalization_limit(_k, _d, _b)
            _o = 1

            while _o <= _r :
                _new_feature, _max_obj = self._Backward_Objective(_selected, _unselected, _o, X, y)

                if _max_obj > self.J_max[_k - _o] :
                    self.J_max[_k - _o] = _max_obj
                    _k -= _o
                    for _f in _new_feature : # add new features and remove these features from the pool
                        _unselected.append(_f)
                    _selected.remove(_new_feature)
                    self._subset_max[_k - _o] = _selected
                else :
                    if _o < _r :
                        _o += 1
                    else :
                        _k += 1 # the marked in J_max and _subset_max are considered as best for _k features


# Genetic Algorithm (GA)
# CHCGA
