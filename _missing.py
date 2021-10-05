from cmath import nan
import random
import numpy as np
from numpy.lib.function_base import bartlett
import pandas as pd
import warnings
import sklearn
import sklearn.utils
import tensorflow as tf

from ._base import random_list

class  SimpleImputer() :

    '''
    Simple Imputer to fill nan values

    Parameters
    ----------
    method: the method to fill nan values, default = 'mean'
    supproted methods ['mean', 'zero', 'median', 'most frequent', constant]
    'mean' : fill columns with nan values using mean of non nan values
    'zero': fill columns with nan values using 0
    'median': fill columns with nan values using median of non nan values
    'most frequent': fill columns with nan values using most frequent of non nan values
    constant: fill columns with nan values using predefined values
    '''

    def __init__(
        self,
        method = 'mean'
    ) :
        self.method = method

    def fill(self, X) :

        _X = X.copy(deep = True)
        
        if _X .isnull().values.any() :
            features = list(X.columns)
            for _column in features :
                if X[_column].isnull().values.any() :
                    _X[_column] = self._fill(_X[_column])

        return _X

    def _fill(self, X) :

        if self.method == 'mean' :
            X = X.fillna(np.nanmean(X))
        elif self.method == 'zero' :
            X = X.fillna(0)
        elif self.method == 'median' :
            X = X.fillna(np.nanmedian(X))
        elif self.method == 'most frequent' :
            X = X.fillna(X.value_counts().index[0])
        else :
            X = X.fillna(self.method)

        return X

class DummyImputer() :

    '''
    Create dummy variable for nan values and fill the original feature with 0
    The idea is that there are possibilities that the nan values are critically related to response, create dummy
    variable to identify the relationship
    
    Parameters
    ----------
    force: whether to force dummy coding for nan values, default = False
    if force == True, all nan values will create dummy variables, otherwise, only nan values that creates impact 
    on response will create dummy variables

    threshold: threshold whether to create dummy variable, default = 0.1
    if mean of nan response and mean of non nan response is different above threshold, threshold will be created

    method: the method to fill nan values for columns not reaching threshold, default = 'mean'
    supproted methods ['mean', 'zero', 'median', 'most frequent', constant]
    'mean' : fill columns with nan values using mean of non nan values
    'zero': fill columns with nan values using 0
    'median': fill columns with nan values using median of non nan values
    'most frequent': fill columns with nan values using most frequent of non nan values
    constant: fill columns with nan values using predefined values
    '''

    def __init__(
        self,
        force = False,
        threshold = 0.1,
        method = 'zero'
    ) :
        self.force = force
        self.threshold = threshold
        self.method = method

    def fill(self, X, y) :

        _X = X.copy(deep = True)

        if _X .isnull().values.any() :
            _X = self._fill(_X, y)

        return _X
    
    def _fill(self, X, y) :

        features = list(X.columns)

        for _column in features :
            if X[_column].isnull().values.any() :
                _mean_nan = y[X[_column].isnull()].mean()
                _mean_non_nan = y[~X[_column].isnull()].mean()
                if abs(_mean_nan / _mean_non_nan - 1) >= self.threshold :
                    X[_column + '_nan'] = X[X[_column].isnull(), _column].astype(int)
                    X[_column] = X[_column].fillna(0)
                else :
                    if self.method == 'mean' :
                        X[_column] = X[_column].fillna(np.nanmean(X[_column]))
                    elif self.method == 'zero' :
                        X[_column] = X[_column].fillna(0)
                    elif self.method == 'median' :
                        X[_column] = X[_column].fillna(np.nanmedian(X[_column]))
                    elif self.method == 'most frequent' :
                        X[_column] = X[_column].fillna(X[_column].value_counts().index[0])
                    else :
                        X[_column] = X[_column].fillna(self.method)

class JointImputer() :

    '''
    Impute the missing values assume a joint distribution, default as multivariate Gaussian distribution

    Not yet ready
    '''

    def __init__(
        self,
        kernel = 'normal'
    ) :
        self.kernel = kernel

    def fill(self, X) :

        _X = X.copy(deep = True)

        if _X .isnull().values.any() :
            _X = self._fill(_X)
            
        return _X
    
    def _fill(self, X) :

        self._mean = X.mean(axis = 1, skipna = True)
        self._covaraince = X.cov()

        features = list(X.columns)
        for _column in features :
            if X[_column].isnull().values.any() :
                X[_column] = self._fill_column(X[_column])

    def _fill_column(self, df) :

        ###########################################################################################          

        return df

class MICE() :
    
    '''
    Multiple Imputation by chained equations (MICE)
    using single imputation to initialize the imputation step, and iteratively build regression/
    classification model to impute features with missing values [1]
    
    [1] Azur, M.J., Stuart, E.A., Frangakis, C. and Leaf, P.J., 2011. Multiple imputation by 
    chained equations: what is it and how does it work?. International journal of methods in 
    psychiatric research, 20(1), pp.40-49.

    Parameters
    ----------
    cycle: how many runs of regression/imputation to build the complete data, default = 10

    method: the method to initially fill nan values, default = 'mean'
    supproted methods ['mean', 'zero', 'median', 'most frequent', constant]
    'mean' : fill columns with nan values using mean of non nan values
    'zero': fill columns with nan values using 0
    'median': fill columns with nan values using median of non nan values
    'most frequent': fill columns with nan values using most frequent of non nan values
    constant: fill columns with nan values using predefined values

    seed: random seed, default = 1
    every random draw from the minority class will increase the random seed by 1
    '''

    def __init__(
        self,
        cycle = 10,
        method = 'mean',
        seed = 1
    ) :
        self.method = method
        self.cycle = cycle
        self.seed = seed

    def fill(self, X) :

        _X = X.copy(deep = True)
        
        if _X .isnull().values.any() :
            _X = self._fill(_X)
        else :
            warnings.warn('No nan values found, no change.')

        return _X

    def _fill(self, X) :

        features = list(X.columns)

        for _column in features :
            if (X[_column].dtype == np.object) or (str(X[_column].dtype) == 'category') :
                raise ValueError('MICE can only handle numerical filling, run encoding first!')

        self._missing_feature = [] # features contains missing values
        self._missing_vector = [] # vector with missing values, to mark the missing index
                                  # create _missing_table with _missing_feature
                                  # missing index will be 1, existed index will be 0

        for _column in features :
            if X[_column].isnull().values.any() :
                self._missing_feature.append(_column)
                self._missing_vector.append(X[_column].isnull().astype(int))

        self._missing_vector = np.array(self._missing_vector).T
        self._missing_table = pd.DataFrame(self._missing_vector, columns = self._missing_feature)

        X = SimpleImputer(method = self.method).fill(X) # initial filling for missing values
        
        random_feautres = random_list(self._missing_feature, self.seed) # the order to regress on missing features

        for _ in range(self.cycle) :
            X = self._cycle_impute(X, random_feautres)

        return X
    
    def _cycle_impute(self, X, random_feautres) :

        from sklearn.linear_model import LinearRegression, LogisticRegression, LassoCV

        features = list(X.columns)
    
        for _column in random_feautres :
            _subfeature = features
            _subfeature.remove(_column)
            X.loc[self._missing_table[_column] == 1, _column] = nan
            if len(X[_column].unique()) == 2 :
                fit_model = LogisticRegression()
            elif len(features) <= 15 :
                fit_model = LinearRegression()
            else :
                fit_model = LassoCV()
            fit_model.fit(X.loc[~X[_column].isnull(), _subfeature], X.loc[~X[_column].isnull(), _column])
            X.loc[X[_column].isnull(), _column] = fit_model.predict(X.loc[X[_column].isnull(), _subfeature])

        return X

class GAIN() :

    '''
    Generative Adversarial Imputation Nets (GAIN)
    train Generator (G) and Discriminator (D) to impute missing values

    Parameters
    ----------
    batch_size: sampling size from data

    hint_rate: hint rate

    alpha: penalty in optimizing Generator

    iterations: number of iterations
    '''

    def __init__(
        self,
        batch_size = None,
        hint_rate = None,
        alpha = None,
        iterations = None
    ) :
        self.batch_size = batch_size
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.iterations = iterations
    
    def mask_matrix(self, X) :

        '''
        mask matrix, m_{ij} = 1 where x_{ij} exists; m_{ij} = 0 otherwise
        '''
        return X.isnull().astype(int)