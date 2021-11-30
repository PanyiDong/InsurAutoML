import random
from tqdm import tqdm
import numpy as np
from numpy.lib.function_base import bartlett
import pandas as pd
import warnings
import sklearn

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
