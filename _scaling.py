from matplotlib.pyplot import sci
import numpy as np
import pandas as pd
import scipy
import scipy.stats
from sympy import SeqAdd
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

from ._encoding import DataEncoding

class NoScaling() :

    def fit(self, X) :
        return self

    def transform(self, X) :
        return X

class Standardize() :

    '''
    Standardize the dataset by column (each feature), using _x = (x - mean) / std

    Parameters
    ----------
    with_mean: whether to standardize with mean, default = True

    with_std: whether to standardize with standard variance, default = True
    '''

    def __init__(
        self,
        with_mean = True,
        with_std = True
    ) :
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X) :
        
        _X = X.copy(deep = True)
        # check for categorical type
        for _column in list(X.columns) :
            # if encounter string/category type, transform the data
            if X[_column].dtype == np.object or str(X[_column].dtype) == 'category' :
                preprocessor = DataEncoding(transform = False)
                preprocessor.fit(_X)
                _X = preprocessor.transform(_X)
                break

        n, p = _X.shape
        if self.with_mean == True :
            self._mean = [0 for _ in range(p)]
        if self.with_std == True :
            self._std = [0 for _ in range(p)]

        for i in range(p) :
            _data = _X.iloc[:, i].values
            n_notnan = n - np.isnan(_data).sum()
            _x_sum = 0
            _x_2_sum = 0
            for j in range(n) :
                _x_sum += (_data[j] if np.isnan(_data[j]) else 0)
                _x_2_sum += (_data[j] ** 2 if np.isnan(_data[j]) else 0)
            if self.with_mean == True :
                self._mean[i] = _x_sum / n_notnan
            if self.with_std == True :
                self._std[i] = np.sqrt((_x_2_sum - n_notnan * ((_x_sum / n_notnan) ** 2)) \
                    / (n_notnan - 1))

        return self

    def transform(self, X) :

        _X = X.copy(deep = True)
        if self.with_mean :
            _X -= self._mean
        if self.with_std :
            _X /= self._std

        return _X

class Normalize() :

    '''
    Normalize features with x / x_

    Parameters
    ----------
    norm: how to select x_, default = 'max'
    supported ['l1', 'l2', 'max']
    '''

    def __init__(
        self,
        norm = 'max', 
    ) :
        self.norm = norm

    def fit(self, X) :
        
        if self.norm not in ['l1', 'l2', 'max'] :
            raise ValueError('Not recognizing norm method!')
        
        _X = X.copy(deep = True)
        n, p = _X.shape
        self._scale = [0 for _ in range(p)]

        for i in range(p) :
            _data = _X.iloc[:, i].values
            if self.norm == 'max' :
                self._scale[i] = np.max(np.abs(_data))
            elif self.norm == 'l1' :
                self._scale[i] = np.abs(_data).sum()
            elif self.norm == 'l2' :
                self._scale[i] = (_data ** 2).sum()
        
        return self

    def transform(self, X) :

        _X = X.copy(deep = True)
        _X /= self._scale

        return _X

class RobustScale() :

    '''
    Use quantile to scale, x / (q_max - q_min)

    Parameters
    ----------
    with_centering: whether to standardize with median, default = True

    with_std: whether to standardize with standard variance, default = True

    quantile: (q_min, q_max), default = (25.0, 75.0)

    uni_variance: whether to set unit variance for scaled data, default = False
    '''

    def __init__(
        self,
        with_centering = True,
        with_scale = True,
        quantile = (25.0, 75.0),
        unit_variance = False
    ) :
        self.with_centering = with_centering
        self.with_scale = with_scale
        self.quantile = quantile
        self.unit_variance = unit_variance

    def fit(self, X) :

        q_min, q_max = self.quantile
        if q_min == None : # in case no input
            q_min = 25.0
        if q_max == None :
            q_max = 75.0
        if not 0 <= q_min <= q_max <= 100.0 :
            raise ValueError('Quantile not in range, get {0:.1f} and {1:.1f}!'.format(q_min, q_max))
        
        _X = X.copy(deep = True)
        n, p = _X.shape
        if self.with_centering == True :
            self._median = [0 for _ in range(p)]
        if self.with_scale == True :
            self._scale = [0 for _ in range(p)]

        for i in range(p) :
            _data = _X.iloc[:, i].values
            if self.with_centering == True :
                self._median[i] = np.nanmedian(_data)
            if self.with_scale == True :
                quantile = np.nanquantile(_data, (q_min, q_max))
                quantile = np.transpose(quantile)
                self._scale[i] = quantile[1] - quantile[0]
                if self.unit_variance == True :
                    self._scale[i] = self.scale[i] / (scipy.stats.norm.ppf(q_max) - scipy.stats.norm.ppf)
        
        return self

    def transform(self, X) :

        _X = X.copy(deep = True)

        if self.with_centering == True :
            _X -= self._median
        if self.with_scale == True :
            _X /= self._scale

        return _X

class MinMaxScale() :

    '''
    Use min_max value to scale the feature, x / (x_max - x_min)

    Parameters
    ----------
    feature_range: (feature_min, feature_max) to scale the feature, default = (0, 1)
    '''

    def __init__(
        self,
        feature_range = (0, 1)
    ) :
        self.feature_range = feature_range

    def fit(self, X) :

        _X = X.copy(deep = True)
        n, p = _X.shape

        self._min = [0 for _ in range(p)]
        self._max = [0 for _ in range(p)]

        for i in range(p) :
            _data = _X.iloc[:, i].values
            self._min[i] = np.nanmin(_data)
            self._max[i] = np.nanmax(_data)
        
        return self

    def transform(self, X) :

        f_min, f_max = self.feature_range
        if not f_min < f_max :
            raise ValueError('Minimum of feature range must be smaller than maximum!')

        _X = X.copy(deep = True)
        _X = (_X - self._min) / (np.array(self._max) - np.array(self._min))
        _X = _X * (f_max - f_min) + f_min

        return _X

    def inverse_transform(self, X) :

        f_min, f_max = self.feature_range
        if not f_min < f_max :
            raise ValueError('Minimum of feature range must be smaller than maximum!')

        _X = X.copy(deep = True)
        _X = (_X - f_min) / (f_max - f_min)
        _X = _X * (np.array(self._max) - np.array(self._min)) + self._min

        return _X

class Winsorization() :

    '''
    Limit feature to certain quantile (remove the effect of extreme values)
    if the response of extreme values are different than non extreme values above threshold, the feature will
    be capped

    Parameters
    ----------
    quantile: quantile to be considered as extreme, default = 0.95

    threshold: threshold to decide whether to cap feature, default = 0.1
    '''

    def __init__(
        self,
        quantile = 0.95,
        threshold = 0.1
    ) :
        self.quantile = quantile
        self.threshold = threshold

    def fit(self, X, y) :

        features = list(X.columns)
        self._quantile_list = []
        self._list = []

        for _column in features :
            quantile = np.nanquantile(X[_column], self.quantile, axis = 0)
            self._quantile_list.append(quantile)
            _above_quantile = y[X[_column] > quantile].mean()
            _below_quantile = y[X[_column] <= quantile].mean()
            if abs(_above_quantile / _below_quantile - 1) > self.threshold :
                self._list.append(True)
            else :
                self._list.append(False)
        
        return self

    def transform(self, X) :

        _X = X.copy(deep = True)
        features = list(_X.columns)
        i = 0

        for _column in features :
            if self._list[i] :
                _X.loc[_X[_column] > self._quantile_list[i], _column] = self._quantile_list[i]
            i += 1

        return _X
