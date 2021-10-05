from cmath import nan
import random
from tqdm import tqdm
import numpy as np
from numpy.lib.function_base import bartlett
import pandas as pd
import warnings
import sklearn
import tensorflow as tf
#tf.compat.v1.disable_v2_behavior() # use tf < 2.0 functions

from ._base import random_index, random_list, feature_rounding
from ._scaling import MinMaxScale

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

    uni_class: unique classes in a column which will be considered as categorical class, default = 31
    round numerical to categorical in case after the imputation, the data type changes

    seed: random seed
    '''

    def __init__(
        self,
        batch_size = None,
        hint_rate = None,
        alpha = None,
        iterations = None,
        uni_class = 31,
        seed = 1
    ) :
        self.batch_size = batch_size
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.iterations = iterations
        self.uni_class = uni_class
        self.seed = seed
    
    def mask_matrix(self, X) :

        '''
        mask matrix, m_{ij} = 1 where x_{ij} exists; m_{ij} = 0 otherwise
        '''
        return X.isnull().astype(int)
    
    # initialize normal tensor by size
    def normal_initial(self, size) :

        _dim = size[0]
        return tf.random.normal(shape = size, stddev = 1 / tf.sqrt(_dim / 2))\
    
    # return random binary array by size
    def binary_sampler(self, p = 0.5, size = (1, 1)) :

        # allows only change row size with (n, )
        # cannot handle (, n)
        try :
            if size[0] == None :
                size[0] == 1
            elif size[1] == None :
                size[1] == 1
        except IndexError :
            size = (size[0], 1)

        _random_unit = np.random.uniform(low = 0, high = 1, size = size)
        return 1 * (_random_unit < p)
        
    # return random uniform array by size
    def uniform_sampler(self, low = 0, high = 1, size = (1, 1)) :
        
        # allows only change row size with (n, )
        # cannot handle (, n)
        try :
            if size[0] == None :
                size[0] == 1
            elif size[1] == None :
                size[1] == 1
        except IndexError :
            size = (size[0], 1)
        
        return np.random.uniform(low = low, high = high, size = size)

    # Generator
    def Generator(self, data, mask) :
        
        G_W1, G_W2, G_W3, G_b1, G_b2, G_b3 = self.theta_G
        _input = tf.concat(values = [data, mask], axis = 1) # concate data with mask
        G_h1 = tf.nn.relu(tf.matmul(_input, G_W1) + G_b1)
        G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
        G_pro = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3) # MinMax normalization

        return G_pro

    # Discriminator
    def Discriminator(self, data, mask) :
        
        D_W1, D_W2, D_W3, D_b1, D_b2, D_b3 = self.theta_D
        _input = tf.concat(values = [data, mask], axis = 1) # concate data with mask
        D_h1 = tf.nn.relu(tf.matmul(_input, D_W1) + D_b1)
        D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
        D_pro = tf.nn.sigmoid(tf.matmul(D_h2, D_W3) + D_b3) # MinMax normalization

        return D_pro

    def fill(self, data) :

        _data = data.copy(deep = True)
        n, p = _data.shape

        _h_dim = int(p) # Hidden state dimensions

        _mask = self.mask_matrix(_data)
        # scaling data to [0, 1]
        scaler = MinMaxScale()
        scaler.fit(_data)
        _data_scaled = scaler.transform(_data)
        
        # divide dataframe to np array for values and features names list
        _features = list(_data_scaled.columns)
        _data_scaled = _data_scaled.values

        # GAIN architecture
        _X = tf.compat.v1.placeholder(tf.float32, shape = [None, p]) # data
        _M = tf.compat.v1.placeholder(tf.float32, shape = [None, p]) # mask vector
        _H = tf.compat.v1.placeholder(tf.float32, shape = [None, p]) # hint vector

        # Generator Variables
        G_W1 = tf.Variable(self.normal_initial([p ** 2, _h_dim]))
        G_b1 = tf.Variable(tf.zeros(shape = [_h_dim]))
        G_W2 = tf.Variable(self.normal_initial([_h_dim, _h_dim]))
        G_b2 = tf.Variable(tf.zeros(shape = [_h_dim]))
        G_W3 = tf.Variable(self.normal_initial([_h_dim, p]))
        G_b3 = tf.Variable(tf.zeros(shape = [p]))

        self.theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

        # Discriminator Varaibles
        D_W1 = tf.Variable(self.normal_initial([p ** 2, _h_dim]))
        D_b1 = tf.Variable(tf.zeros(shape = [_h_dim]))
        D_W2 = tf.Variable(self.normal_initial([_h_dim, _h_dim]))
        D_b2 = tf.Variable(tf.zeros(shape = [_h_dim]))
        D_W3 = tf.Variable(self.normal_initial([_h_dim, p]))
        D_b3 = tf.Variable(tf.zeros(shape = [p]))

        self.theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

        # GAIN structure
        _G = self.Generator(_X, _M) # Generator
        _hat_X = _X * _M + _G * (1 - _M) # combine mask with observed data
        _D = self.Discriminator(_hat_X, _H) # Discriminator

        _D_loss = -tf.reduce_mean(_M * tf.log(_D + 1e-8) + \
            (1 - _M) * tf.log(1. - _D + 1e-8)) # Discriminator loss
        _G_loss_1 = -tf.reduce_mean((1 - _M) * tf.log(_D + 1e-8)) # Generator loss
        _MSE_loss = tf.reduce_mean((_M * _X - _X * _G) ** 2) / tf.reduce_mean(_M)
        _G_loss = _G_loss_1 + self.alpha * _MSE_loss

        # GAIN solver
        _G_solver = tf.compat.v1.train.AdamOptimizer().minimize(_D_loss, var_list = self.theta_G)
        _D_solver = tf.compat.v1.train.AdamOptimizer().minimize(_G_loss, var_list = self.theta_D)

        # Iterations
        sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())
        _seed = self.seed # initialize random seed

        for _run in tqdm(range(self.iterations)) :

            batch_index = random_index(self.batch_size, n, seed = _seed) # random sample batch
            _X_mb = _data_scaled[batch_index, :]
            _M_mb = _mask[batch_index, :]
            _Z_mb = self.uniform_sampler(low = 0, high = 0.01, size = (self.batch_size, p)) # random sample vector
            _H_mb_1 = self.binary_sampler(p = self.hint_rate, size = (self.batch_size, p))
            _H_mb = _M_mb + _H_mb_1 # sample hint vectors

            # combine random sample vector with observed data
            _X_mb = _M_mb * _X_mb + (1 - _M_mb) * _Z_mb
            _, _D_loss_now = sess.run([_D_solver, _D_loss], \
                feed_dict = {_M : _M_mb, _X : _X_mb, _H : _H_mb})
            _, _G_loss_now, _MSE_loss_now = sess.run([_G_solver, _G_loss, _MSE_loss], \
                feed_dict = {_M : _M_mb, _X : _X_mb, _H : _H_mb})

            _seed += 1

        # return imputed data
        _Z_mb = self.uniform_sampler(low = 0, high = 0.01, size = (self.batch_size, p))
        _M_mb = _mask
        _X_mb = _data_scaled
        _X_mb = _M_mb * _X_mb + (1 - _M_mb) * _Z_mb

        _imputed_data = sess.run([_G], feed_size = {_X : _X_mb, _M : _M_mb})[0]
        _imputed_data = _mask * _data_scaled + (1 - _mask) * _imputed_data
        
        # combine data with column names to dataframe
        _imputed_data = pd.DataFrame(_imputed_data, columns = _features)

        # Unscale the imputed data
        _imputed_data = scaler.inverse_transform(_imputed_data)
        _imputed_data = feature_rounding(_imputed_data)

        return _imputed_data
