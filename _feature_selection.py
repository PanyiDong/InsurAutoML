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
