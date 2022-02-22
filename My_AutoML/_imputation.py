import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings
import sklearn
from functools import partial
import multiprocessing
from multiprocessing import Pool

from ._scaling import MinMaxScale
from ._utils import formatting

# check if tensorflow exists
# if exists, import tensorflow
import importlib
tensorflow_spec = importlib.util.find_spec('tensorflow')
if tensorflow_spec is not None :
    import tensorflow as tf
    from tensorflow.python.types.core import Value
    tf.compat.v1.disable_eager_execution()
    #tf.compat.v1.disable_v2_behavior() # use tf < 2.0 functions

from ._utils import random_index, random_list, feature_rounding, nan_cov
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

        rows = list(X.index)
        for _row in rows :
            if X.loc[_row, :].isnull().values.any() :
                X.loc[_row, :] = self._fill_row(_row, X)

        return X

    def _fill_row(self, row_index, X) :

        '''
        for x = (x_{mis}, x_{obs})^{T} with \mu = (\mu_{mis}, \mu_{obs}).T and \Sigma = ((Sigma_{mis, mis}, 
        Sigma_{mis, obs}), (Sigma_{obs, Sigma}, Sigma_{obs, obs})),
        Conditional distribution x_{mis}|x_{obs} = a is N(\bar(\mu), \bar(\Sigma))
        where \bar(\mu) = \mu_{mis} + \Sigma_{mis, obs}\Sigma_{obs, obs}^{-1}(a - \mu_{obs})
        and \bar(\Sigma) = \Sigma_{mis, mis} - \Sigma_{mis, obs}\Sigma_{obs, obs}^{-1}\Sigma_{obs, mis}

        in coding, 1 = mis, 2 = obs for simpilicity
        '''

        _mis_column = np.argwhere(X.loc[row_index, :].isnull().values).T[0]
        _obs_column = [i for i in range(len(list(X.columns)))]
        for item in _mis_column :
            _obs_column.remove(item)

        _mu_1 = np.nanmean(X.iloc[:, _mis_column], axis = 0).T.reshape(len(_mis_column), 1)
        _mu_2 = np.nanmean(X.iloc[:, _obs_column], axis = 0).T.reshape(len(_obs_column), 1)

        _sigma_11 = nan_cov(X.iloc[:, _mis_column].values)
        _sigma_22 = nan_cov(X.iloc[:, _obs_column].values)
        _sigma_12 = nan_cov(X.iloc[:, _mis_column].values, y = X.iloc[:, _obs_column].values)
        _sigma_21 = nan_cov(X.iloc[:, _obs_column].values, y = X.iloc[:, _mis_column].values)
         
        _a = X.loc[row_index, ~X.loc[row_index, :].isnull()].values.T.reshape(len(_obs_column), 1)
        _mu = _mu_1 + _sigma_12 @ np.linalg.inv(_sigma_22) @ (_a - _mu_2)
        _mu = _mu[0] # multivariate_normal only accept 1 dimension mean
        _sigma = _sigma_11 - _sigma_12 @ np.linalg.inv(_sigma_22) @ _sigma_21

        X.loc[row_index, X.loc[row_index, :].isnull()] = np.random.multivariate_normal(mean = _mu, \
            cov = _sigma, size = (X.loc[row_index, :].isnull().values.sum(), 1))

        return X.loc[row_index, :]

class ExpectationMaximization() :
    
    '''
    Use Expectation Maximization (EM) to impute missing data[1]

    [1] Impyute.imputation.cs.em

    Parameters
    ----------
    iterations: maximum number of iterations for single imputation, default = 50

    threshold: threshold to early stop iterations, default = 0.01
    only early stop when iterations < self.iterations and change in the imputation < self.threshold

    seed: random seed, default = 1
    '''

    def __init__(
        self,
        iterations = 50,
        threshold = 0.01,
        seed = 1
    ) :
        self.iterations = iterations
        self.threshold = threshold
        self.seed = seed

    def fill(self, X) :

        self.iterations = int(self.iterations)
        self.threshold = float(self.threshold)

        _X = X.copy(deep = True)
        n = _X.shape[0]

        if _X .isnull().values.any() :
            _X = self._fill(_X)
            
        return _X

    def _fill(self, X) :
        
        features = list(X.columns)
        np.random.seed(self.seed)

        _missing_feature = [] # features contains missing values
        _missing_vector = [] # vector with missing values, to mark the missing index
                             # create _missing_table with _missing_feature
                             # missing index will be 1, existed index will be 0

        for _column in features :
            if X[_column].isnull().values.any() :
                _missing_feature.append(_column)
                _missing_vector.append(X[_column].loc[X[_column].isnull()].index.astype(int))

        _missing_vector = np.array(_missing_vector).T
        self._missing_table = pd.DataFrame(_missing_vector, columns = _missing_feature)

        for _column in list(self._missing_table.columns) :
            for _index in self._missing_table[_column] :
                X.loc[_index, _column] = self._EM_iter(X, _index, _column)

        return X
    
    def _EM_iter(self, X, index, column) :

        _mark = 1
        for _ in range(self.iterations) :
            _mu = np.nanmean(X.loc[:, column])
            _std = np.nanstd(X.loc[:, column])
            _tmp = np.random.normal(loc = _mu, scale = _std)
            _delta = np.abs(_tmp - _mark) / _mark
            if _delta < self.threshold and self.iterations > 10 :
                return _tmp
            X.loc[index, column] = _tmp
            _mark = _tmp
        return _tmp

class KNNImputer() :

    '''
    Use KNN to impute the missing values, further update: use cross validation to select best k [1]

    [1] Stekhoven, D.J. and Bühlmann, P., 2012. MissForest—non-parametric missing value imputation 
    for mixed-type data. Bioinformatics, 28(1), pp.112-118.

    Parameters
    ----------
    n_neighbors: list of k, default = None
    default will set to 1:10

    fold: cross validation number of folds, default = 10

    uni_class: unique class to be considered as categorical columns, default = 31
    '''

    def __init__(
        self,
        n_neighbors = None,
        fold = 10,
        uni_class = 31
    ) :
        self.n_neighbors = n_neighbors
        self.fold = fold
        self.uni_class = uni_class

    def fill(self, X) :

        features = list(X.columns)
        for _column in features :
            if len(X[_column].unique()) <= min(0.1 * len(X), self.uni_class) :
                raise ValueError('KNN Imputation not supported for categorical data!')

        _X = X.copy(deep = True)
        if _X.isnull().values.any() :
            _X = self._fill(_X)
        else :
            warnings.warn('No nan values found, no change.')

        return _X

    def _fill(self, X) :
        
        features = list(X.columns)

        self._missing_feature = [] # features contains missing values
        self._missing_vector = [] # vector with missing values, to mark the missing index
                                  # create _missing_table with _missing_feature
                                  # missing index will be 1, existed index will be 0

        for _column in features :
            if X[_column].isnull().values.any() :
                self._missing_feature.append(_column)
                self._missing_vector.append(X[_column].loc[X[_column].isnull()].index.astype(int))

        self._missing_vector = np.array(self._missing_vector).T
        self._missing_table = pd.DataFrame(self._missing_vector, columns = self._missing_feature)

        X = SimpleImputer(method = self.method).fill(X) # initial filling for missing values
        
        random_feautres = random_list(self._missing_feature, self.seed) # the order to regress on missing features
        _index = random_index(len(X.index)) # random index for cross validation
        _err = []

        for i in range(self.fold) :
            _test = X.iloc[i * int(len(X.index) / self.fold):int(len(X.index) / self.fold), :]
            _train = X
            _train.drop(labels = _test.index, axis = 0, inplace = True)
            _err.append(self._cross_validation_knn(_train, _test, random_feautres))

        _err = np.mean(np.array(_err), axis = 0) # mean of cross validation error
        self.optimial_k = np.array(_err).argmin()[0] + 1 # optimal k

        X = self._knn_impute(X, random_feautres, self.optimial_k)

        return X
    
    def _cross_validation_knn(self, _train, _test, random_feautres) : # cross validation to return error

        from sklearn.neighbors import KNeighborsRegressor
        if self.n_neighbors == None :
            n_neighbors = [i + 1 for i in range(10)]
        else :
            n_neighbors = self.n_neighbors
            
        _test_mark = _test.copy(deep = True)
        _err = []

        for _k in n_neighbors :
            _test = _test_mark.copy(deep = True)
            for _feature in random_feautres :
                _subfeatures = list(_train.columns)
                _subfeatures.drop(_feature, inplace = True)

                fit_model = KNeighborsRegressor(n_neighbors = _k)
                fit_model.fit(_train.loc[:, _subfeatures], _train.loc[:, _feature])
                _test.loc[:, _feature] = fit_model.predict(_test.loc[:, _subfeatures])
            _err.append(((_test - _test_mark) ** 2).sum())

        return _err


    def _knn_impute(self, X, random_feautres, k) :

        from sklearn.neighbors import KNeighborsRegressor

        features = list(X.columns)
        for _column in random_feautres :
            _subfeature = features
            _subfeature.remove(_column)
            X.loc[self._missing_table[_column] == 1, _column] = nan
            fit_model = KNeighborsRegressor(n_neighbors = k)
            fit_model.fit(X.loc[~X[_column].isnull(), _subfeature], X.loc[~X[_column].isnull(), _column])
            X.loc[X[_column].isnull(), _column] = fit_model.predict(X.loc[X[_column].isnull(), _subfeature])

        return X


class MissForestImputer() :

    '''
    Run Random Forest to impute the missing values [1]

    [1] Stekhoven, D.J. and Bühlmann, P., 2012. MissForest—non-parametric missing 
    value imputation for mixed-type data. Bioinformatics, 28(1), pp.112-118.

    Parameters
    ----------
    threshold: threshold to terminate iterations, default = 0
    At default, if difference between iterations increases, the iteration stops

    method: initial imputation method for missing values, default = 'mean'

    uni_class: column with unique classes less than uni_class will be considered as categorical, default = 31
    '''

    def __init__(
        self,
        threshold = 0,
        method = 'mean',
        uni_class = 31
    ) :
        self.threshold = threshold
        self.method = method
        self.uni_class = uni_class

    def _RFImputer(self, X) :

        from sklearn.ensemble import RandomForestRegressor
        
        _delta = [] # criteria of termination

        while True :
            for _column in list(self._missing_table.columns) :
                X_old = X.copy(deep = True)
                _subfeature = list(X_old.columns)
                _subfeature.remove(str(_column))
                _missing_index = self._missing_table[_column].tolist()
                RegModel = RandomForestRegressor()
                RegModel.fit(X.loc[~X.index.astype(int).isin(_missing_index), _subfeature], \
                    X.loc[~X.index.astype(int).isin(_missing_index), _column])
                _tmp_column = RegModel.predict(X.loc[X.index.astype(int).isin(_missing_index), _subfeature])
                X.loc[X.index.astype(int).isin(_missing_index), _column] = _tmp_column
                _delta.append(self._delta_cal(X, X_old))
                if len(_delta) >= 2 and _delta[-1] > _delta[-2] :
                    break
            if len(_delta) >= 2 and _delta[-1] > _delta[-2] :
                break

        return X
    
    # calcualte the difference between data newly imputed and before imputation
    def _delta_cal(self, X_new, X_old) :

        if (X_new.shape[0] != X_old.shape[0]) or (X_new.shape[1] != X_old.shape[1]) :
            raise ValueError('New and old data must have same size, get different!')

        _numerical_features = []
        _categorical_features = []
        for _column in list(self._missing_table.columns) :
            if len(X_old[_column].unique()) <= self.uni_class :
                _categorical_features.append(_column)
            else :
                _numerical_features.append(_column)
        
        _N_nume = 0
        _N_deno = 0
        _F_nume = 0
        _F_deno = 0
        
        if len(_numerical_features) > 0 :
            for _column in _numerical_features :
                _N_nume += ((X_new[_column] - X_old[_column]) ** 2).sum()
                _N_deno += (X_new[_column] ** 2).sum()
        
        if len(_categorical_features) > 0 :
            for _column in _categorical_features :
                _F_nume += (X_new[_column] != X_old[_column]).astype(int).sum()
                _F_deno += len(self._missing_table[_column])
        
        if len(_numerical_features) > 0 and len(_categorical_features) > 0 :
            return _N_nume / _N_deno + _F_nume / _F_deno
        elif len(_numerical_features) > 0 : 
            return _N_nume / _N_deno
        elif len(_categorical_features) > 0 :
            return _F_nume / _F_deno

    def fill(self, X) :

        _X = X.copy(deep = True)
        if _X.isnull().values.any() :
            _X = self._fill(_X)
        else :
            warnings.warn('No nan values found, no change.')

        return _X

    def _fill(self, X) :
        
        features = list(X.columns)

        for _column in features :
            if (X[_column].dtype == np.object) or (str(X[_column].dtype) == 'category') :
                raise ValueError('MICE can only handle numerical filling, run encoding first!')

        _missing_feature = [] # features contains missing values
        _missing_vector = [] # vector with missing values, to mark the missing index
                             # create _missing_table with _missing_feature
                             # missing index will be 1, existed index will be 0
        _missing_count = [] # counts for missing values

        for _column in features :
            if X[_column].isnull().values.any() :
                _missing_feature.append(_column)
                _missing_vector.append(X.loc[X[_column].isnull()].index.astype(int))
                _missing_count.append(X[_column].isnull().astype(int).sum())

        # reorder the missing features by missing counts increasing
        _order = np.array(_missing_count).argsort().tolist()
        _missing_count = np.array(_missing_count)[_order].tolist()
        _missing_feature = np.array(_missing_feature)[_order].tolist()
        _missing_vector = np.array(_missing_vector)[_order].T.tolist()

        self._missing_table = pd.DataFrame(_missing_vector, columns = _missing_feature)

        X = SimpleImputer(method = self.method).fill(X) # initial filling for missing values
        X = self._RFImputer(X)

        return X

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

        self.cycle = int(self.cycle)

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
                self._missing_vector.append(X.loc[X[_column].isnull()].index.astype(int))

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
            _missing_index = self._missing_table[_column].tolist()
            X.loc[X.index.astype(int).isin(_missing_index), _column] = np.nan
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
    train Generator (G) and Discriminator (D) to impute missing values [1]

    [1] Yoon, J., Jordon, J. and Schaar, M., 2018, July. Gain: Missing data imputation using 
    generative adversarial nets. In International Conference on Machine Learning (pp. 5689-5698). PMLR.
    github.com/jsyooon0823/GAIN

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
        batch_size = 128,
        hint_rate = 0.9,
        alpha = 100,
        iterations = 10000,
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
        return 1 - X.isnull().astype(int)
    
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
    def Discriminator(self, data, hint) :
        
        D_W1, D_W2, D_W3, D_b1, D_b2, D_b3 = self.theta_D
        _input = tf.concat(values = [data, hint], axis = 1) # concate data with mask
        D_h1 = tf.nn.relu(tf.matmul(_input, D_W1) + D_b1)
        D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
        D_pro = tf.nn.sigmoid(tf.matmul(D_h2, D_W3) + D_b3) # MinMax normalization

        return D_pro

    def fill(self, X) :

        _X = X.copy(deep = True)
        
        if _X .isnull().values.any() :
            _X = self._fill(_X)
        else :
            warnings.warn('No nan values found, no change.')

        return _X

    def _fill(self, data) :

        _data = data.copy(deep = True)
        n, p = _data.shape

        _h_dim = int(p) # Hidden state dimensions

        _mask = self.mask_matrix(_data).values
        # scaling data to [0, 1]
        scaler = MinMaxScale()
        scaler.fit(_data)
        _data_scaled = scaler.transform(_data)
        _data_scaled = _data_scaled.fillna(0)
        
        # divide dataframe to np array for values and features names list
        _features = list(_data_scaled.columns)
        _data_scaled = _data_scaled.values

        # GAIN architecture
        _X = tf.compat.v1.placeholder(tf.float32, shape = [None, p]) # data
        _M = tf.compat.v1.placeholder(tf.float32, shape = [None, p]) # mask vector
        _H = tf.compat.v1.placeholder(tf.float32, shape = [None, p]) # hint vector

        # Generator Variables
        G_W1 = tf.Variable(self.normal_initial([p * 2, _h_dim]))
        G_b1 = tf.Variable(tf.zeros(shape = [_h_dim]))
        G_W2 = tf.Variable(self.normal_initial([_h_dim, _h_dim]))
        G_b2 = tf.Variable(tf.zeros(shape = [_h_dim]))
        G_W3 = tf.Variable(self.normal_initial([_h_dim, p]))
        G_b3 = tf.Variable(tf.zeros(shape = [p]))

        self.theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

        # Discriminator Varaibles
        D_W1 = tf.Variable(self.normal_initial([p * 2, _h_dim]))
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

        _D_loss_tmp = -tf.reduce_mean(_M * tf.compat.v1.log(_D + 1e-8) + \
            (1 - _M) * tf.compat.v1.log(1. - _D + 1e-8)) # Discriminator loss
        _G_loss_tmp = -tf.reduce_mean((1 - _M) * tf.compat.v1.log(_D + 1e-8)) # Generator loss
        _MSE_loss = tf.reduce_mean((_M * _X - _M * _G) ** 2) / tf.reduce_mean(_M)
        _D_loss = _D_loss_tmp
        _G_loss = _G_loss_tmp + self.alpha * _MSE_loss

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
            _H_mb = _M_mb * _H_mb_1 # sample hint vectors

            # combine random sample vector with observed data
            _X_mb = _M_mb * _X_mb + (1 - _M_mb) * _Z_mb
            _, _D_loss_now = sess.run([_D_solver, _D_loss_tmp], \
                feed_dict = {_M : _M_mb, _X : _X_mb, _H : _H_mb})
            _, _G_loss_now, _MSE_loss_now = sess.run([_G_solver, _G_loss_tmp, _MSE_loss], \
                feed_dict = {_M : _M_mb, _X : _X_mb, _H : _H_mb})

            _seed += 1

        # return imputed data
        _Z_mb = self.uniform_sampler(low = 0, high = 0.01, size = (n, p))
        _M_mb = _mask
        _X_mb = _data_scaled
        _X_mb = _M_mb * _X_mb + (1 - _M_mb) * _Z_mb

        _imputed_data = sess.run([_G], feed_dict = {_X : _X_mb, _M : _M_mb})[0]
        _imputed_data = _mask * _data_scaled + (1 - _mask) * _imputed_data
        
        # combine data with column names to dataframe
        _imputed_data = pd.DataFrame(_imputed_data, columns = _features)

        # Unscale the imputed data
        _imputed_data = scaler.inverse_transform(_imputed_data)
        _imputed_data = feature_rounding(_imputed_data)

        return _imputed_data

class AAI_kNN(formatting, MinMaxScale) :
    
    """
    kNN Imputation/Neighborhood-based Collaborative Filtering with 
    Auto-adaptive Imputation/AutAI [1]
    AutAI's main idea is to distinguish what part of the dataset may
    be important for imputation.

    ----
    [1] Ren, Y., Li, G., Zhang, J. and Zhou, W., 2012, October. The 
    efficient imputation method for neighborhood-based collaborative 
    filtering. In Proceedings of the 21st ACM international conference 
    on Information and knowledge management (pp. 684-693).

    Parameters
    ----------
    columns: columns need imputation, default = []
    default will take all columns

    k: k nearest neighbors selected, default = 3
    Odd k preferred

    scaling: whether to perform scaling on the features, default = True

    similarity: how to calculate similarity among rows, default = 'PCC'
    support ['PCC', 'COS']

    AutAI: whether to use AutAI, default = True

    AutAI_tmp: whether AutAI is temporary imputation or permanent 
    imputation, default = True
    if True (temporary), AutAI imputation will not be preserved, but 
    can take much longer

    threads: number of threads to use, default = -1
    if -1, use all threads

    deep_copy: whether to deep_copy dataframe, default = False
    """

    def __init__(
        self,
        columns = [],
        k = 3,
        scaling = True,
        similarity = 'PCC',
        AutAI = True,
        AutAI_tmp = True,
        threads = -1,
        deep_copy = False,
    ) :
        self.columns = columns
        self.k = k
        self.scaling = scaling
        self.similarity = similarity
        self.AutAI = AutAI
        self.AutAI_tmp = AutAI_tmp
        self.threads = threads
        self.deep_copy = deep_copy

        self.fitted = False # whether fitted on train set
        self.train = pd.DataFrame() # store the imputed train set
    
    # calculate Pearson Correlation Coefficient/PCC
    # PCC = \sum_{i}(x_{i}-\mu_{x})(y_{i}-\mu_{y}) / 
    # \sqrt{\sum_{i}(x_{i}-\mu_{x})^{2}\sum_{i}(y_{i}-\mu_{y})^{2}}
    def Pearson_Correlation_Coefficient(self, x, y) :

        # convert to numpy array
        x = np.array(x)
        y = np.array(y)
        
        # get mean of x and y
        u_x = np.nanmean(x)
        u_y = np.nanmean(y)
        
        # get numerator and denominator
        numerator = np.nansum((x - u_x) * (y - u_y))
        denominator = np.sqrt(np.nansum((x - u_x) ** 2) * np.nansum((y - u_y) ** 2))
        
        # special case of denominator being 0
        if denominator == 0 :
            return 1
        else :
            return numerator / denominator
    
    # calculate Cosine-based similarity/COS
    # COS = x * y / (|x|*|y|)
    def Cosine_based_similarity(self, x, y) :
        
        # convert to numpy array
        x = np.array(x)
        y = np.array(y)
        
        # get numerator and denominator
        numerator = np.nansum(x * y)
        denominator = np.sqrt(np.nansum(x ** 2) * np.nansum(y ** 2))

        # special case of denominator being 0
        if denominator == 0 :
            return 1
        else :
            return numerator / denominator
    
    # get column values from k nearest neighbors
    def _get_k_neighbors(self, test, train, column) :

        similarity_list = []

        for index in list(train.index) :
            # get similarity between test and all rows in train
            if self.similarity == 'PCC' :
                similarity_list.append(
                    self.Pearson_Correlation_Coefficient(test.values, train.loc[index].values)
                )
            elif self.similarity == 'COS' :
                similarity_list.append(
                    self.Cosine_based_similarity(test.values, train.loc[index].values)
                )

        # get index of k largest similarity in list
        k_order = np.argsort(similarity_list)[-self.k:]
        # convert similarity list order to data index
        k_index = [list(train.index)[i] for i in k_order]

        # get k largest similarity
        k_similarity = [similarity_list[_index] for _index in k_order]
        
        # get the k largest values in the list
        k_values = [train.loc[_index, column] for _index in k_index]

        return k_values, k_index, k_similarity
    
    # AutAI imputation
    def _AAI_impute(self, X, index, column) :

        _X = X.copy(deep = self.deep_copy)

        # (index, column) gives a location of missing value
        # the goal is to find and impute relatively important data
        
        # get indexes where column values are not missing
        U_a = list(_X.loc[~_X[column].isnull()].index.astype(int))

        # find union of columns where both non_missing (each from above) 
        # and missing rows have data
        T_s = [] # ultimate union of features
        loc_non_missing_column = set(_X.columns[~_X.loc[index, :].isnull()])
  
        for _index in U_a :
            # get all intersection from U_a rows
            non_missing_columns = set(_X.columns[~_X.loc[_index, :].isnull()])
            intersection_columns = loc_non_missing_column.intersection(non_missing_columns)

            if not T_s : # if empty
                T_s = intersection_columns
            else :
                T_s = T_s.union(intersection_columns)

        T_s = list(T_s) # convert to list

        # range (U_a, T_s) considered important data

        # use kNN with weight of similarity for imputation
        for _column in T_s :
            for _index in list(set(X[X[_column].isnull()].index) & set(U_a)) :
                # get kNN column values, index, and similarity
                k_values, k_index, k_similarity = self._get_k_neighbors(
                    _X.loc[_index, :], _X.loc[~_X[_column].isnull()], _column
                )
        
                # normalize k_similarity
                k_similarity = [item / sum(k_similarity) for item in k_similarity]

                # get kNN row mean
                k_means = [np.nanmean(_X.loc[_index, :]) for _index in k_index]

                # calculate impute value
                _impute = np.nanmean(_X.loc[index, :])
                for i in range(self.k) :
                    _impute += k_similarity[i] * (k_values[i] - k_means[i])
                
                _X.loc[_index, _column] = _impute

        return _X 
    
    # pool tasks on the index chunks
    # every pool task works on part of the chunks
    def Pool_task(self, X, index_list) :

        _X = X.copy(deep = self.deep_copy)

        for _column in self.columns :
            # get missing rows
            # select in index_list and get missing rows
            missing = _X.loc[index_list].loc[_X[_column].isnull()]

            # make sure k is at least not larger than rows of non_missing
            self.k = min(self.k, len(_X) - len(missing))

            if missing.empty : # if no missing found in the column, skip
                pass
            else :
                for _index in list(missing.index) :
                    # if need AutAI, perform AutAI imputation first
                    # if fitted, no need for AutAI, directly run kNN imputation
                    if self.AutAI and not self.fitted :
                        if self.AutAI_tmp :
                            _X_tmp = self._AAI_impute(_X, _index, _column)
                            # get non-missing (determined by _column) rows
                            non_missing = _X_tmp.loc[~_X_tmp[_column].isnull()]
                        else :
                            _X = self._AAI_impute(_X, _index, _column)
                            # get non-missing (determined by _column) rows
                            non_missing = _X.loc[~_X[_column].isnull()]
                    elif not self.fitted :
                        # get non-missing (determined by _column) rows
                        non_missing = _X.loc[~_X[_column].isnull()]

                    # use kNN imputation for (_index, _column)
                    # if fitted, use imputed dataset for imputation
                    if not self.fitted :
                        k_values, _, _ = self._get_k_neighbors(
                            _X.loc[_index, :], non_missing, _column
                        )
                        _X.loc[_index, _column] = np.mean(k_values)
                    else :
                        k_values, _, _ = self._get_k_neighbors(
                            _X.loc[_index, :], self.train, _column
                        )
                        _X.loc[_index, _column] = np.mean(k_values)
            
        # return only the working part
        return _X.loc[index_list, :]

    def fill(self, X) :
        
        # make sure input is a dataframe
        if not isinstance(X, pd.DataFrame) :
            try : 
                X = pd.DataFrame(X)
            except :
                raise TypeError('Expect a dataframe, get {}.'.format(X))

        _X = X.copy(deep = self.deep_copy)

        # initialize columns
        self.columns = list(_X.columns) if not self.columns else self.columns

        # initialize number of working threads
        self.threads = multiprocessing.cpu_count() if self.threads == -1 else int(self.threads)

        if _X[self.columns].isnull().values.any() :
            _X = self._fill(_X)
        else :
            warnings.warn('No missing values found, no change.')

        return _X

    def _fill(self, X) :

        _X = X.copy(deep = self.deep_copy)

        # convert categorical to numerical
        formatter = formatting(columns = self.columns, inplace = True)
        formatter.fit(_X)
        
        # if scaling, use MinMaxScale to scale the features
        if self.scaling :
            scaling = MinMaxScale(columns = self.columns)
            _X = scaling.fit_transform(_X)

        # kNN imputation

        # parallelized pool workflow
        pool = Pool(processes = self.threads)
        # divide indexes to evenly sized chunks
        divide_index = np.array_split(list(_X.index), self.threads)

        # parallelized work
        pool_data = pool.map(partial(self.Pool_task, _X), divide_index)
        pool.close()
        pool.join()

        # concat the chunks of the datset
        _X = pd.concat(pool_data).sort_index()
        
        # convert self.fitted and store self.train
        self.fitted = True
        # only when empty need to store
        # stored train is imputed, formatted, scaled dataset
        self.train = _X.copy() if self.train.empty else self.train

        # if scaling, scale back
        if self.scaling :
            _X = scaling.inverse_transform(_X)
        
        # convert numerical back to categorical
        formatter.refit(_X)

        return _X

"""
For clustering methods, if get warning of empty mean (all cluster 
have nan for the feature) or reduction in number of clusters (get 
empty cluster), number of cluster is too large compared to the 
data size and should be decreased. However, no error will be 
raised, and all observations will be correctly assigned.
"""

class KMI(formatting, MinMaxScale) :

    """
    KMI/K-Means Imputation[1], the idea is to incorporate K-Means
    Clustering with kNN imputation

    ----
    [1] Hruschka, E.R., Hruschka, E.R. and Ebecken, N.F., 2004, December. 
    Towards efficient imputation by nearest-neighbors: A clustering-based 
    approach. In Australasian Joint Conference on Artificial Intelligence 
    (pp. 513-525). Springer, Berlin, Heidelberg.

    Parameters
    ----------
    """

    def __init__(
        self,
        scaling = True,
    ) :
        self.scaling = scaling
        
        raise NotImplementedError('Not implemented!')

class CMI(formatting, MinMaxScale) :

    """
    Clustering-based Missing Value Imputation/CMI[1], introduces the idea
    of clustering observations into groups and use the kernel statistsics 
    to impute the missing values for the groups.

    ----
    [1] Zhang, S., Zhang, J., Zhu, X., Qin, Y. and Zhang, C., 2008. Missing 
    value imputation based on data clustering. In Transactions on computational 
    science I (pp. 128-138). Springer, Berlin, Heidelberg.

    Parameters
    ----------
    columns: columns need imputation, default = []
    default will take all columns

    k: number of cluster groups, default = 10

    distance: metrics of calculating the distance between two rows, default = 'l2'
    used for selecting clustering groups,
    support ['l1', 'l2']

    delta: threshold to stop k Means Clustering, default = 0
    delta defined as number of group assignment changes for clustering
    0 stands for best k Means Clustering

    scaling: whether to use scaling before imputation, default = True

    seed: random seed, default = 1
    used for k group initialization

    threads: number of threads to use, default = -1
    if -1, use all threads

    deep_copy: whether to use deep copy, default = False
    """

    def __init__(
        self, 
        columns = [],
        k = 10,
        distance = 'l2',
        delta = 0,
        scaling = True,
        seed = 1,
        threads = -1,
        deep_copy = False,
    ) :
        self.columns = columns
        self.k = k
        self.distance = distance
        self.delta = delta
        self.scaling = scaling
        self.seed = seed
        self.threads = threads
        self.deep_copy = deep_copy

        np.random.seed(seed = self.seed)

        self.fitted = False # whether fitted on train set
        self.train = pd.DataFrame() # store the imputed train set
    
    # calculate distance between row and k group mean
    # 'l1' or 'l2' Euclidean distance
    def _distance(self, row, k) :

        if self.distance == 'l2' :
            return np.sqrt(np.nansum((row - self.k_means[k]) ** 2))
        elif self.distance == 'l1' :
            return np.nansum(np.abs(row - self.k_means[k]))

    # get the Gaussian kernel values
    def _kernel(self, row1, row2) :
        
        return np.prod(
            np.exp(-((row1 - row2) / self.bandwidth) ** 2 / 2) / np.sqrt(2 * np.pi)
        )
    
    # get k_means for group k
    def _get_k_means(self, data, k) :

        # get observations in _k group
        group_data = data.loc[np.where(self.group_assign == k)[0], :]

        if not group_data.empty :
            return np.nanmean(group_data, axis = 0)
        else :
            # return None array
            return np.array([None])

    # get group assign for the chunk of data (by index_list)
    def _get_group_assign(self, data, index_list) :

        result = []
        for _index in index_list :
            # get distance between row and every k groups
            distance = [self._distance(data.loc[_index, :], _k) for _k in range(self.k)]
            # assign the row to closest range group
            result.append(np.argsort(distance)[0])
        
        return result
    
    # assign clustering groups according to Euclidean distance
    # only support numerical features
    def _k_means_clustering(self, X) :

        _X = X.copy(deep = self.deep_copy)

        n, p = _X.shape # number of observations

        # make sure self.k is smaller than n
        self.k = min(self.k, n)
        
        # if not fitted (on train dataset), run complete k Means clustering
        # else, use train k_means for clustering
        if not self.fitted :
            
            # get bandwidth for the kernel function
            self.bandwidth = np.sum(np.abs(
                _X[self.columns].max() - _X[self.columns].min()
            ))

            # initialize k groups
            self.group_assign = np.random.randint(0, high = self.k, size = n)
            # initialize k means
            self.k_means = np.empty([self.k, p])

            # parallelized k means calculation
            pool = Pool(processes = min(int(self.k), self.threads))
            self.k_means = pool.map(partial(self._get_k_means, _X), list(range(self.k)))
            pool.close()
            pool.join()

            # if group empty, raise warning and set new k
            self.k_means = [item for item in self.k_means if item.all()]
            if len(self.k_means) < self.k :
                warnings.warn('Empty cluster found and removed.')
                self.k = len(self.k_means)

            while True :
                # store the group assignment
                # need deep copy, or the stored assignment will change accordingly
                previous_group_assign = self.group_assign.copy()

                # assign each observation to new group based on k_means
                pool = Pool(processes = self.threads)
                divide_index = np.array_split(list(_X.index), self.threads)
                self.group_assign = pool.map(partial(self._get_group_assign, _X), divide_index)
                # flatten 2d list to 1d
                self.group_assign = np.array(np.concatenate(self.group_assign).flat)
                pool.close()
                pool.join()

                # calculate the new k_means
                # parallelized k means calculation
                pool = Pool(processes = min(int(self.k), self.threads))
                self.k_means = pool.map(partial(self._get_k_means, _X), list(range(self.k)))
                pool.close()
                pool.join()

                # if group empty, raise warning and set new k
                self.k_means = [item for item in self.k_means if item.all()]
                if len(self.k_means) < self.k :
                    warnings.warn('Empty cluster found and removed.')
                    self.k = len(self.k_means)
                
                # if k Means constructed, break the loop
                if np.sum(np.abs(previous_group_assign - self.group_assign)) <= self.delta :
                    break
        else :
            # copy the train group assignment
            self.group_assign_train = self.group_assign.copy()
            
            self.group_assign = np.zeros(n) # re-initialize the group_assign (n may change)
            # assign each observation to new group based on k_means
            pool = Pool(processes = self.threads)
            divide_index = np.array_split(list(_X.index), self.threads)
            self.group_assign = pool.map(partial(self._get_group_assign, _X), divide_index)
            # flatten 2d list to 1d
            self.group_assign = np.array(np.concatenate(self.group_assign).flat)
            pool.close()
            pool.join()

    # pool tasks on the column chunks
    # every pool task works on part of the chunks
    def Pool_task(self, X, _column, non_missing_index, n, index_list) :

        _X = X.copy(deep = self.deep_copy)
    
        # find missing indexes belongs to _k group
        for _index in index_list :
            if not self.fitted :
                # get the kernel values
                kernel = np.array(
                    [self._kernel(
                        _X.loc[_index, _X.columns != _column], 
                        _X.loc[__index, _X.columns != _column]
                    )
                    for __index in non_missing_index]
                )
                # impute the missing_values
                _X.loc[_index, _column] = np.sum(
                    _X.loc[non_missing_index, _column] * kernel
                    ) / (np.sum(kernel) + n ** (-2))
            else :
                # get the kernel values
                kernel = np.array(
                    [self._kernel(
                        _X.loc[_index, _X.columns != _column], 
                        self.train.loc[__index, self.train.columns != _column]
                    )
                    for __index in non_missing_index]
                )
                # impute the missing_values
                _X.loc[_index, _column] = np.sum(
                    self.train.loc[non_missing_index, _column] * kernel
                ) / (np.sum(kernel) + n ** (-2))
        
        # return group data
        return _X.loc[index_list, _column]

    def fill(self, X) :

        # make sure input is a dataframe
        if not isinstance(X, pd.DataFrame) :
            try :
                X = pd.DataFrame(X)
            except :
                raise TypeError('Expect a dataframe, get {}.'.format(type(X)))

        _X = X.copy(deep = self.deep_copy)

        # initialize columns
        self.columns = list(_X.columns) if not self.columns else self.columns

        # initialize number of working threads
        self.threads = multiprocessing.cpu_count() if self.threads == -1 else int(self.threads)

        if _X[self.columns].isnull().values.any() :
            _X = self._fill(_X)
        else :
            warnings.warn('No missing values found, no change.')

        return _X

    def _fill(self, X) :

        _X = X.copy(deep = self.deep_copy)

        # convert categorical to numerical
        formatter = formatting(columns = self.columns, inplace = True)
        formatter.fit(_X)

        # if scaling, use MinMaxScale to scale the features
        if self.scaling :
            scaling = MinMaxScale(columns = self.columns)
            _X = scaling.fit_transform(_X)
        
        # imputation on formatted, scaled datasets
        # assign observations to self.k groups
        # get self.group_assign and self.k_means
        # if already fitted (working on test data now), get
        # k_means from train dataset and the group assignment
        # for train dataset
        self._k_means_clustering(_X)

        for _column in self.columns :
            for _k in range(self.k) :
                group_index = np.where(self.group_assign == _k)[0]
                n = len(group_index) # number of observations in the group
                # get the missing/non-missing indexes
                missing_index = list(set(_X[_X[_column].isnull()].index) & set(group_index))
                if not self.fitted : # if not fitted, use _X
                    non_missing_index = list(set(_X[~_X[_column].isnull()].index) & set(group_index))
                else : # if fitted, use self.train
                    # after imputation, there should be no missing in train
                    # so take all of them
                    non_missing_index = np.where(self.group_assign_train == _k)[0]
                
                # parallelize imputation
                divide_missing_index = np.array_split(missing_index, self.threads)
                pool = Pool(self.threads)
                imputation = pool.map(
                    partial(self.Pool_task, _X, _column, non_missing_index, n), 
                    divide_missing_index
                )
                pool.close()
                pool.join()

                imputation = pd.concat(imputation).sort_index()
                _X.loc[missing_index, _column] = imputation
        
        # convert self.fitted and store self.train
        self.fitted = True
        # only when empty need to store
        # stored train is imputed, formatted, scaled dataset
        self.train = _X.copy() if self.train.empty else self.train

        # if scaling, scale back
        if self.scaling :
            _X = scaling.inverse_transform(_X)
        
        # convert numerical back to categorical
        formatter.refit(_X)

        return _X

class k_Prototype_NN(formatting, MinMaxScale) :

    """
    Three clustering models are provided: k-Means Paradigm, k-Modes Paradigm,
    k-Prototypes Paradigm [1]

    Parameters
    ----------
    columns: columns need imputation, default = []
    default will take all columns

    k: number of cluster groups, default = 10

    distance: metrics of calculating the distance between two rows, default = 'l2'
    used for selecting clustering groups,
    support ['l1', 'l2']

    dissimilarity: how to calculate the dissimilarity for categorical columns, default = 'weighted'
    support ['simple', 'weighted']

    scaling: whether to use scaling before imputation, default = True

    numerics: numerical columns

    seed: random seed, default = 1
    used for k group initialization

    threads: number of threads to use, default = -1
    if -1, use all threads

    deep_copy: whether to use deep copy, default = False
    
    ----
    [1] Madhuri, R., Murty, M.R., Murthy, J.V.R., Reddy, P.P. and Satapathy, 
    S.C., 2014. Cluster analysis on different data sets using K-modes and 
    K-prototype algorithms. In ICT and Critical Infrastructure: Proceedings 
    of the 48th Annual Convention of Computer Society of India-Vol II 
    (pp. 137-144). Springer, Cham.
    """

    def __init__(
        self,
        columns = [],
        k = 10,
        distance = 'l2',
        dissimilarity = 'weighted',
        scaling = True,
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'],
        threads = -1,
        deep_copy = False,
        seed = 1,
    ) :
        self.columns = columns
        self.k = k
        self.distance = distance
        self.dissimilarity = dissimilarity
        self.scaling = scaling
        self.numerics = numerics
        self.threads = threads
        self.deep_copy = deep_copy
        self.seed = seed

        np.random.seed(self.seed)

        self.fitted = False # check whether fitted on train data
        self.models = {} # save fit models

    # calculate distance between row and k group mean
    # 'l1' or 'l2' Euclidean distance
    def _distance(self, row, k_centroids) :
        
        # if not defining np.float64, may get float and 
        # raise Error not able of iterating
        if self.distance == 'l2' :
            return np.sqrt(np.nansum(
                (row - k_centroids) ** 2, axis = 1, dtype = np.float64
            ))
        elif self.distance == 'l1' :
            return np.nansum(
                np.abs(row - k_centroids), axis = 1, dtype = np.float64
            )

    # calculate dissimilarity difference between row
    # and k group
    def _dissimilarity(self, row, k_centroids) :

        k, p = k_centroids.shape
        
        # simple dissimilarity, number of different categories
        if self.dissimilarity == 'simple' :
            return np.sum((row.values != k_centroids.values).astype(int), axis = 1)
        # weighted dissimilarity, weighted based on number of unique categories
        elif self.dissimilarity == 'weighted' :
            # set difference between row and k_centroids
            different_matrix = (row.values != k_centroids.values).astype(int)
            
            # initialize number counts
            row_count = np.empty(p)
            centroids_count = np.empty([k, p])
           
            for idx, _column in enumerate(list(k_centroids.columns)) :
                # get the category
                cate = row[_column]
                # find the corresponding count
                # if get missing value, count as 0
                row_count[idx] = self.categorical_table[_column][cate] if not cate else 0
                for _k in range(k) :
                    # get the category
                    cate = k_centroids.loc[_k, _column]
                    # find the corresponding count
                    centroids_count[_k, idx] = self.categorical_table[_column][cate]
            
            # calculate the weights based on number of categories
            weight = np.empty([k, p])
            # in case get denominator of 0
            for _p in range(p) :
                for _k in range(k) :
                    weight[_k, _p] = (row_count[_p] + centroids_count[_k, _p]) / \
                        (row_count[_p] * centroids_count[_k, _p]) if row_count[_p] != 0 else 0

            return np.nansum(np.multiply(weight, different_matrix), axis = 1)
    
    # calculate the measurement for given index
    def _get_group_assign(self, data, numerical_columns, categorical_columns, index_list) :

        group_assign = []

        for _index in index_list :
            measurement = self._distance(
                data.loc[_index, numerical_columns], self.k_centroids[numerical_columns]
            ) + self._dissimilarity(
                data.loc[_index, categorical_columns], self.k_centroids[categorical_columns]
            )
                
            # assign the observations to closest centroids
            group_assign.append(np.argsort(measurement)[0])

        return group_assign
    
    # calculate the k_centroids for group k
    def _get_k_centroids(self, data, numerical_columns, categorical_columns, k) :

        k_centroids = pd.DataFrame(index = [k], columns = data.columns)
        
        group_data = data.loc[np.where(self.group_assign == k)[0], :]
        # get column means of the group
        # if get empty group_data, no return
        if not group_data.empty :
            # centroids for numerical columns are mean of the features
            k_centroids[numerical_columns] = np.nanmean(
                group_data[numerical_columns], axis = 0
            )
            # centroids for categorical columns are modes of the features
            # in case multiple modes occur, get the first one
            k_centroids[categorical_columns] = \
                group_data[categorical_columns].mode(dropna = True).values[0]
        
        return k_centroids

    # assign clustering groups according to dissimilarity
    # compared to modes
    # best support categorical features
    def _k_modes_clustering(self, X) :

        _X = X.copy(deep = self.deep_copy)

        n, p = _X.shape # number of observations

        # make sure self.k is smaller than n
        self.k = min(self.k, n)

    # combine k_Means and k_Modes clustering for mixed 
    # numerical/categorical datasets
    # numerical columns will use k_means with distance
    # categorical columns will use k_modes with dissimilarity
    def _k_prototypes_clustering(self, X, numerical_columns, categorical_columns) :

        _X = X.copy(deep = self.deep_copy)

        n, p = _X.shape # number of observations

        # make sure self.k is smaller than n
        self.k = min(self.k, n)
        
        if not self.fitted :
            # create categorical count table
            # unique values descending according to number of observations
            self.categorical_table = {}
            for _column in categorical_columns :
                self.categorical_table[_column] = _X[_column].value_counts(
                    sort = True, ascending = False, dropna = True
                ).to_dict()
            
            # initialize clustering group assignments
            self.group_assign = np.zeros(n)
            # initialize the corresponding centroids
            # use dataframe to match the column names
            self.k_centroids = pd.DataFrame(index = range(self.k), columns = _X.columns)
            # random initialization
            for _column in list(_X.columns) :
                self.k_centroids[_column] = _X.loc[~_X[_column].isnull(), _column].sample(
                    n = self.k, replace = True, random_state = self.seed
                ).values

            while True :
                # calculate sum of Euclidean distance (numerical features) and 
                # dissimilarity difference (categorical features) for every 
                # observations among all centroids

                # parallelized calculation for group assignment
                pool = Pool(processes = self.threads)
                divide_list = np.array_split(list(_X.index), self.threads)
                self.group_assign = pool.map(
                    partial(
                        self._get_group_assign, _X, numerical_columns, categorical_columns
                    ), divide_list
                )
                # flatten 2d list to 1d
                self.group_assign = np.array(np.concatenate(self.group_assign).flat)
                pool.close()
                pool.join()
                
                # save k_centroids for comparison
                previous_k_centroids = self.k_centroids.copy()

                # recalculate the k_centroids
                # calculate the new k_means
                pool = Pool(processes = min(int(self.k), self.threads))
                self.k_centroids = pool.map(
                    partial(
                        self._get_k_centroids, _X, numerical_columns, categorical_columns
                    ), list(range(self.k))
                )
                # concat the k centroids to one dataframe
                self.k_centroids = pd.concat(self.k_centroids).sort_index()
                pool.close()
                pool.join()

                # if get empty cluster, sort index and renew k
                self.k_centroids.dropna(inplace = True)
                if len(self.k_centroids) < self.k :
                    self.k_centroids.reset_index(drop = True, inplace = True)
                    self.k = len(self.k_centroids)

                # stopping criteria
                # check whether same k (in case delete centroids in the process)
                if len(previous_k_centroids) == len(self.k_centroids) :
                    if np.all(previous_k_centroids.values == self.k_centroids.values) :
                        break
        # if fitted, use the trained k_centroids assigning groups
        else :
            # parallelized calculation for group assignment
            pool = Pool(processes = self.threads)
            divide_list = np.array_split(list(_X.index), self.threads)
            self.group_assign = pool.map(
                partial(self._get_group_assign, _X, numerical_columns, categorical_columns), divide_list
            )
            # flatten 2d list to 1d
            self.group_assign = np.array(np.concatenate(self.group_assign).flat)
            pool.close()
            pool.join()
    
    # impute on cluster k
    def _kNN_impute(self, data, k) :

        from sklearn.impute import KNNImputer
        # use 1-NN imputer with clustered groups
        # on train dataset, fit the models
        if k not in self.models.keys() :
            self.models[k] = KNNImputer(n_neighbors = 1)
            self.models[k].fit(data.loc[np.where(self.group_assign == k)[0], :])
        # impute the missing values
        data.loc[np.where(self.group_assign == k)[0], :] = self.models[k].transform(
            data.loc[np.where(self.group_assign == k)[0], :]
        )

        return data.loc[np.where(self.group_assign == k)[0], :]

    def fill(self, X) :
    
        # make sure input is a dataframe
        if not isinstance(X, pd.DataFrame) :
            try :
                X = pd.DataFrame(X)
            except :
                raise TypeError('Expect a dataframe, get {}.'.format(type(X)))

        _X = X.copy(deep = self.deep_copy)

        # initialize columns
        self.columns = list(_X.columns) if not self.columns else self.columns

        # initialize number of working threads
        self.threads = multiprocessing.cpu_count() if self.threads == -1 else int(self.threads)

        if _X[self.columns].isnull().values.any() :
            _X = self._fill(_X)
        else :
            warnings.warn('No missing values found, no change.')

        return _X

    def _fill(self, X) :

        _X = X.copy(deep = self.deep_copy)
        
        # all numerical columns
        numeric_columns = list(_X.select_dtypes(include = self.numerics).columns)
        # select numerical columns in self.columns
        numeric_columns = list(set(_X.columns) & set(numeric_columns))
        # select categorical columns in self.columns
        categorical_columns = list(set(_X.columns) - set(numeric_columns))

        # format columns
        # convert categorical to numerical, 
        # but no numerical manipulation
        formatter = formatting(columns = list(_X.columns), inplace = True)
        formatter.fit(_X)

        # if scaling, scaling the numerical columns
        if self.scaling :
            scaling = MinMaxScale(columns = numeric_columns)
            _X = scaling.fit_transform(_X)

        # imputation procedure
        # assign observations to clustering groups using 
        # k_Prototypes clustering
        self._k_prototypes_clustering(_X, numeric_columns, categorical_columns)

        # use the clustered groups to impute
        # parallelized the imputation process according to clusters
        pool = Pool(processes = min(int(self.k), self.threads))
        pool_data = pool.map(partial(self._kNN_impute, _X), list(range(self.k)))
        # concat pool_data and order according to index
        _X = pd.concat(pool_data).sort_index()
        pool.close()
        pool.join()

        # set fitted to true
        self.fitted = True

        # if scaling, scale back
        if self.scaling :
            _X = scaling.inverse_transform(_X)
        
        # make sure column types retains
        formatter.refit(_X)

        return _X