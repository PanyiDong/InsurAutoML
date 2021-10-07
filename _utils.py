import numpy as np
import pandas as pd
from dateutil.parser import parse
import copy

# set response to [0, 1] class, random guess at 0.5
def random_guess(number, seed = None) :
    np.random.seed(seed)
    if number > 0.5 :
        return 1
    elif number < 0.5 :
        return 0
    else :
        return np.random.randint(0, 2)

# Return random index of a list (unique values only)
# from total draw n, default total = n
def random_index(n, total = None, seed = 1) :
    np.random.seed(seed)
    if total == None :
        total == n
    output = []
    vlist = [i for i in range(total)]
    for _ in range(n) :
        #np.random.seed(int(datetime.now().strftime("%H%M%S")))
        index = np.random.randint(0, high = len(vlist), size = 1)[0]
        output.append(vlist[index])
        vlist.pop(index)

    return output

# Return randomly shuffle of a list (unique values only)
def random_list(vlist, seed = 1) :
    np.random.seed(seed)
    output = []
    for _ in range(len(vlist)) :
        #np.random.seed(int(datetime.now().strftime("%H%M%S")))
        index = np.random.randint(0, high = len(vlist), size = 1)[0]
        output.append(vlist[index])
        vlist.pop(index)

    return output

# check if values in the dataframe is time string
# rule = 'any' will consider the column as date type as long as one value is date type,
# rule = 'all' will consider the column as date type only when all values are date type.
def is_date(df, rule = 'any') :

    def _is_date(string, fuzzy = False) :
        try:
            parse(string, fuzzy = fuzzy)
            return True
    
        except ValueError :
            return False
    
    _check = []
    for item in df.values :
        _check.append(_is_date(str(item[0])))
    if rule == 'any' :
        return any(_check)
    elif rule == 'all' :
        return all(_check)

# Round data for categorical features (in case after preprocessing/modification, the data changed)
def feature_rounding(X, uni_class = 20) :

    features = list(X.columns)
    _X = X.copy(deep = True)

    for _column in features :
        _unique = np.sort(_X[_column].dropna().unique())
        if len(_unique) <= uni_class :
            _X[_column] = np.round(_X[_column])

    return _X

# Train test split using test set percentage
def train_test_split(X, y, test_perc = 0.15, seed = 1) :
    
    '''
    return order: X_train, X_test, y_train, y_test
    '''

    n = len(X)
    index_list = random_index(n, seed)
    valid_index = index_list[:int(test_perc * n)]
    train_index = list(set([i for i in range(n)]) - set(valid_index))

    return X.iloc[train_index], X.iloc[valid_index], y.iloc[train_index], \
        y.iloc[valid_index]

# Return location of minimum values
def minloc(vlist) :
    if len(vlist) == 0 :
        raise ValueError('Invalid List!')
    elif len(vlist) == 1 :
        return 0
    else :
        result = 0
        for i in range(len(vlist) - 1) :
            if vlist[i + 1] < vlist[result] :
                result = i + 1
            else :
                continue
        return result

# return non-nan covariance matrix between X and y, (return covariance of X if y = None)
# default calculate at columns (axis = 0), axis = 1 at rows
def nan_cov(X, y = None, axis = 0) :

    if y == None  :
        y = copy.deepcopy(X)
    else :
        y = y

    if axis == 0 :
        if len(X) != len(y) :
            raise ValueError('X and y must have same length of rows!')
    elif axis == 1 :
        if len(X[0]) != len(y[0]) :
            raise ValueError('X and y must have same length of columns!')

    X = np.array(X)
    y = np.array(y)

    _x_mean = np.nanmean(X, axis = axis)
    _y_mean = np.nanmean(y, axis = axis)

    _cov = np.array([[0. for _i in range(y.shape[1 - axis])] for _j in range(X.shape[1 - axis])]) # initialize covariance matrix

    for i in range(_cov.shape[0]) :
        for j in range(_cov.shape[1]) :
            if axis == 0 :
                _cov[i, j] = np.nansum((X[:, i] - _x_mean[i]) * (y[:, j] - _y_mean[j])) / (len(X) - 1)
            elif axis == 1 :
                _cov[i, j] = np.nansum((X[i, :] - _x_mean[i]) * (y[j, :] - _y_mean[j])) / (len(X[0]) - 1)

    return _cov