from multiprocessing.sharedctypes import Value
import numbers
import numpy as np
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
        
