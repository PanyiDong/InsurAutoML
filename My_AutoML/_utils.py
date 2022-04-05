"""
File: _utils.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Relative Path: /_utils.py
File Created: Friday, 25th February 2022 6:13:42 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Tuesday, 5th April 2022 11:44:26 am
Modified By: Panyi Dong (panyid2@illinois.edu)

-----
MIT License

Copyright (c) 2022 - 2022, Panyi Dong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import importlib
import time
import warnings
import numpy as np
import pandas as pd
import scipy.stats
from dateutil.parser import parse
import copy

pytorch_spec = importlib.util.find_spec("torch")
if pytorch_spec is not None:
    import torch
    from torch.utils.data import TensorDataset, DataLoader

# set response to [0, 1] class, random guess at 0.5
def random_guess(number, seed=1):
    if seed != None:
        np.random.seed(seed)
    if number > 0.5:
        return 1
    elif number < 0.5:
        return 0
    else:
        return np.random.randint(0, 2)


# Return random index of a list (unique values only)
# from total draw n, default total = n
def random_index(n, total=None, seed=1):
    if seed is not None:
        np.random.seed(seed)
    if total is None:
        total = n
    output = []
    vlist = [i for i in range(total)]
    for _ in range(n):
        # np.random.seed(int(datetime.now().strftime("%H%M%S")))
        index = np.random.randint(0, high=len(vlist), size=1)[0]
        output.append(vlist[index])
        vlist.pop(index)

    return output


# Return randomly shuffle of a list (unique values only)
def random_list(vlist, seed=1):
    if seed != None:
        np.random.seed(seed)
    output = []
    for _ in range(len(vlist)):
        # np.random.seed(int(datetime.now().strftime("%H%M%S")))
        index = np.random.randint(0, high=len(vlist), size=1)[0]
        output.append(vlist[index])
        vlist.pop(index)

    return output


# check if values in the dataframe is time string
# rule = 'any' will consider the column as date type as long as one value is date type,
# rule = 'all' will consider the column as date type only when all values are date type.
def is_date(df, rule="any"):
    def _is_date(string, fuzzy=False):
        try:
            parse(string, fuzzy=fuzzy)
            return True

        except ValueError:
            return False

    _check = []
    for item in df.values:
        _check.append(_is_date(str(item[0])))
    if rule == "any":
        return any(_check)
    elif rule == "all":
        return all(_check)


# Round data for categorical features (in case after preprocessing/modification, the data changed)
def feature_rounding(X, uni_class=20):

    features = list(X.columns)
    _X = X.copy(deep=True)

    for _column in features:
        _unique = np.sort(_X[_column].dropna().unique())
        if len(_unique) <= uni_class:
            _X[_column] = np.round(_X[_column])

    return _X


# Train test split using test set percentage
def train_test_split(X, y, test_perc=0.15, seed=1):

    """
    return order: X_train, X_test, y_train, y_test
    """

    n = len(X)
    index_list = random_index(n, seed=seed)
    valid_index = index_list[: int(test_perc * n)]
    train_index = list(set([i for i in range(n)]) - set(valid_index))

    return (
        X.iloc[train_index],
        X.iloc[valid_index],
        y.iloc[train_index],
        y.iloc[valid_index],
    )


# Return location of minimum values
def minloc(vlist):
    if len(vlist) == 0:
        raise ValueError("Invalid List!")
    elif len(vlist) == 1:
        return 0
    else:
        result = 0
        for i in range(len(vlist) - 1):
            if vlist[i + 1] < vlist[result]:
                result = i + 1
            else:
                continue
        return result


# Return location of maximum values
def maxloc(vlist):
    if len(vlist) == 0:
        raise ValueError("Invalid List!")
    elif len(vlist) == 1:
        return 0
    else:
        result = 0
        for i in range(len(vlist) - 1):
            if vlist[i + 1] > vlist[result]:
                result = i + 1
            else:
                continue
        return result


# return the index of Boolean list or {0, 1} list
# default 1 consider as True
def True_index(X, _true=[True, 1]):

    result = [i for i, value in enumerate(X) if value in _true]

    return result


# return non-nan covariance matrix between X and y, (return covariance of X if y = None)
# default calculate at columns (axis = 0), axis = 1 at rows
def nan_cov(X, y=None, axis=0):

    try:
        _empty = (y == None).all().values[0]
    except AttributeError:
        _empty = y == None

    if _empty:
        y = copy.deepcopy(X)
    else:
        y = y

    if axis == 0:
        if len(X) != len(y):
            raise ValueError("X and y must have same length of rows!")
    elif axis == 1:
        if len(X[0]) != len(y[0]):
            raise ValueError("X and y must have same length of columns!")

    X = np.array(X)
    y = np.array(y)

    # reshape the X/y
    try:
        X.shape[1]
    except IndexError:
        X = X.reshape(len(X), 1)

    try:
        y.shape[1]
    except IndexError:
        y = y.reshape(len(y), 1)

    _x_mean = np.nanmean(X, axis=axis)
    _y_mean = np.nanmean(y, axis=axis)

    _cov = np.array(
        [[0.0 for _i in range(y.shape[1 - axis])] for _j in range(X.shape[1 - axis])]
    )  # initialize covariance matrix

    for i in range(_cov.shape[0]):
        for j in range(_cov.shape[1]):
            if axis == 0:
                _cov[i, j] = np.nansum(
                    (X[:, i] - _x_mean[i]) * (y[:, j] - _y_mean[j])
                ) / (len(X) - 1)
            elif axis == 1:
                _cov[i, j] = np.nansum(
                    (X[i, :] - _x_mean[i]) * (y[j, :] - _y_mean[j])
                ) / (len(X[0]) - 1)

    return _cov


# return class (unique in y) mean of X
def class_means(X, y):

    _class = np.unique(y)
    result = []

    for _cl in _class:
        data = X.loc[y.values == _cl]
        result.append(np.mean(data, axis=0).values)

    return result


# return maximum likelihood estimate for covaraiance
def empirical_covariance(X, *, assume_centered=False):

    X = np.asarray(X)

    if X.ndim == 1:
        X = np.reshape(X, (1, -1))

    if X.shape[0] == 1:
        warnings.warn("Only one data sample available!")

    if assume_centered:
        covariance = np.dot(X.T, X) / X.shape[0]
    else:
        covariance = np.cov(X.T, bias=1)

    if covariance.ndim == 0:
        covariance = np.array([[covariance]])
    return covariance


# return weighted within-class covariance matrix
def class_cov(X, y, priors):

    _class = np.unique(y)
    cov = np.zeros(shape=(X.shape[1], X.shape[1]))
    for idx, _cl in enumerate(_class):
        _data = X.loc[y.values == _cl, :]
        cov += priors[idx] * empirical_covariance(_data)
    return cov


# return Pearson Correlation Coefficients
def Pearson_Corr(X, y):

    features = list(X.columns)
    result = []
    for _column in features:
        result.append(
            (nan_cov(X[_column], y) / np.sqrt(nan_cov(X[_column]) * nan_cov(y)))[0][0]
        )

    return result


# return Mutual Information
def MI(X, y):

    if len(X) != len(y):
        raise ValueError("X and y not same size!")

    features = list(X.columns)
    _y_column = list(y.columns)
    result = []

    _y_pro = y.groupby(_y_column[0]).size().div(len(X)).values
    _H_y = -sum(item * np.log(item) for item in _y_pro)

    for _column in features:

        _X_y = pd.concat([X[_column], y], axis=1)
        _pro = (
            _X_y.groupby([_column, _y_column[0]]).size().div(len(X))
        )  # combine probability (x, y)
        _pro_val = _pro.values  # take only values
        _X_pro = X[[_column]].groupby(_column).size().div(len(X))  # probability (x)
        _H_y_X = -sum(
            _pro_val[i]
            * np.log(
                _pro_val[i] / _X_pro.loc[_X_pro.index == _pro.index[i][0]].values[0]
            )
            for i in range(len(_pro))
        )
        result.append(_H_y - _H_y_X)

    return result


# return t-statistics of dataset, only two groups dataset are suitable
def t_score(X, y, fvalue=True, pvalue=False):

    if len(X) != len(y):
        raise ValueError("X and y not same size!")

    features = list(X.columns)
    _y_column = list(y.columns)[0]  # only accept one column of response

    _group = y[_y_column].unique()
    if len(_group) != 2:
        raise ValueError(
            "Only 2 group datasets are acceptable, get {}.".format(len(_group))
        )

    _f = []
    _p = []

    for _col in features:
        t_test = scipy.stats.ttest_ind(
            X.loc[y[_y_column] == _group[0], _col],
            X.loc[y[_y_column] == _group[1], _col],
        )
        if fvalue:
            _f.append(t_test[0])
        if pvalue:
            _p.append(t_test[1])

    if fvalue and pvalue:
        return _f, _p
    elif fvalue:
        return _f
    elif pvalue:
        return _p


# return ANOVA of dataset, more than two groups dataset are suitable
def ANOVA(X, y, fvalue=True, pvalue=False):

    if len(X) != len(y):
        raise ValueError("X and y not same size!")

    features = list(X.columns)
    _y_column = list(y.columns)[0]  # only accept one column of response

    _group = y[_y_column].unique()

    _f = []
    _p = []

    for _col in features:
        _group_value = []
        for _g in _group:
            _group_value.append(X.loc[y[_y_column] == _g, _col].flatten())
        _test = scipy.stats.f_oneway(*_group_value)
        if fvalue:
            _f.append(_test[0])
        if pvalue:
            _p.append(_test[1])

    if fvalue and pvalue:
        return _f, _p
    elif fvalue:
        return _f
    elif pvalue:
        return _p


# transform between numpy array and pandas dataframe
# to deal with some problems where dataframe will be converted to array using sklearn objects
class as_dataframe:
    def __init__(self):
        self.design_matrix = None  # record the values of dataframe
        self.columns = None  # record column heads for the dataframe

    def to_array(self, X):

        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input should be dataframe!")

        self.design_matrix = X.values
        self.columns = list(X.columns)

        return self.design_matrix

    def to_df(self, X=None, columns=None):

        if not isinstance(X, np.ndarray):
            if not X:
                X = self.design_matrix  # using original values from dataframe
            else:
                raise TypeError("Input should be numpy array!")

        try:
            _empty = (columns == None).all()
        except AttributeError:
            _empty = columns == None

        if _empty:
            columns = self.columns

        if len(columns) != X.shape[1]:
            raise ValueError(
                "Columns of array {} does not match length of columns {}!".format(
                    X.shape[1], len(columns)
                )
            )

        return pd.DataFrame(X, columns=columns)


# determine the task types
def type_of_task(y):

    if isinstance(y, pd.DataFrame):
        y = y.values

    if y.dtype.kind == "f" and np.any(y != y.astype(int)):
        return "continuous"  # assign for regression tasks

    if y.dtype.kind in ["i", "u"] and len(np.unique(y)) >= 0.5 * len(y):
        return "integer"  # assign for regression tasks

    if (len(np.unique(y)) > 2) or (y.ndim >= 2 and len(y[0]) > 1):
        return "multiclass"  # assign for classification tasks
    else:
        return "binary"  # assign for regression tasks


# formatting the type of features in a dataframe
# to ensure the consistency of the features,
# avoid class type (encoded as int) becomes continuous type
# older version
# class formatting:
#     def __init__(self, allow_str=False):
#         self.allow_str = allow_str

#         self.category_table = None

#     def fit(self, X):
#         # get dtype of the features
#         self.dtype_table = X.dtypes.values

#         if not self.allow_str:  # if not allow str, transform string types to int
#             for i in range(len(self.dtype_table)):
#                 if self.dtype_table[i] == object:
#                     self.dtype_table[i] = np.int64

#         return self

#     def transform(self, X):

#         for i in range(len(self.dtype_table)):
#             X.iloc[:, i] = X.iloc[:, i].astype(self.dtype_table[i])

#         return X
# new version of formatting
class formatting:

    """
    Format numerical/categorical columns

    Parameters
    ----------
    numerics: numerical columns

    nas: different types of missing values

    allow_string: whether allow string to store in dataframe, default = False

    inplace: whether to replace dataset in fit step, default = True

    Example
    -------
    >> a = pd.DataFrame({
    >>     'column_1': [1, 2, 3, np.nan],
    >>     'column_2': ['John', np.nan, 'Amy', 'John'],
    >>     'column_3': [np.nan, '3/12/2000', '3/13/2000', np.nan]
    >> })

    >> formatter = formatting(columns = ['column_1', 'column_2'], inplace = True)
    >> formatter.fit(a)
    >> print(a)

       column_1  column_2   column_3
    0       1.0       0.0        NaN
    1       2.0       NaN  3/12/2000
    2       3.0       1.0  3/13/2000
    3       NaN       0.0        NaN

    >> a.loc[2, 'column_2'] = 2.6
    >> formatter.refit(a)
    >> print(a)

       column_1 column_2   column_3
    0       1.0      Amy        NaN
    1       2.0      NaN  3/12/2000
    2       3.0      Amy  3/13/2000
    3       NaN     John        NaN
    """

    def __init__(
        self,
        columns=[],
        numerics=["int16", "int32", "int64", "float16", "float32", "float64"],
        nas=[np.nan, None, "nan", "NaN", "NA", "novalue", "None", "none"],
        allow_string=False,
        inplace=True,
    ):
        self.columns = columns
        self.numerics = numerics
        self.nas = nas
        self.allow_string = allow_string
        self.inplace = inplace

        self.type_table = {}  # store type of every column
        self.unique_table = {}  # store unique values of categorical columns

    # factorize data without changing values in nas
    # pd.factorize will automatically convert missing values
    def factorize(self, data):

        # get all unique values, including missing types
        raw_unique = pd.unique(data)
        # remove missing types
        # since nan != nan, convert it to string for comparison
        unique_values = [item for item in raw_unique if str(item) not in self.nas]

        # add unique values to unique_table
        self.unique_table[data.name] = unique_values

        # create categorical-numerical table
        unique_map = {}
        for idx, item in enumerate(unique_values):
            unique_map[item] = idx

        # mapping categorical to numerical
        data = data.replace(unique_map)

        return data

    # make sure the category seen in observed data
    def unify_cate(self, x, list):

        if not x in list and str(x) not in self.nas:
            x = np.argmin(np.abs([item - x for item in list]))

        return x

    def fit(self, X):

        # make sure input is a dataframe
        if not isinstance(X, pd.DataFrame):
            try:
                X = pd.DataFrame(X)
            except:
                raise TypeError("Expect a dataframe, get {}.".format(type(X)))

        # if not specified, get all columns
        self.columns = list(X.columns) if not self.columns else self.columns

        for _column in self.columns:
            self.type_table[_column] = X[_column].dtype
            # convert categorical to numerics
            if X[_column].dtype not in self.numerics:
                if self.inplace:
                    X[_column] = self.factorize(X[_column])
                else:
                    self.factorize(X[_column])

    def refit(self, X):

        for _column in self.columns:
            # if numerical, refit the dtype
            if self.type_table[_column] in self.numerics:
                X[_column] = X[_column].astype(self.type_table[_column])
            else:
                # if column originally belongs to categorical,
                # but converted to numerical, convert back
                if X[_column].dtype in self.numerics:
                    # get all possible unique values in unique_table
                    unique_num = np.arange(len(self.unique_table[_column]))
                    # make sure all values have seen in unique_table
                    X[_column] = X[_column].apply(
                        lambda x: self.unify_cate(x, unique_num)
                    )

                    # get unique category mapping, from numerical-> categorical
                    unique_map = dict(zip(unique_num, self.unique_table[_column]))
                    X[_column] = X[_column].map(
                        unique_map
                    )  # convert numerical-> categorical

                # refit dtype, for double checking
                X[_column] = X[_column].astype(self.type_table[_column])


# define a Timer to record efficiency
# enable multiple running times for comparison
class Timer:
    def __init__(self):
        self.times = []
        self.start()

    def start(self):  # start the timer
        self.tik = time.time()

    def stop(self):  # stop the timer and record the time
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()


def unify_nan(dataset, columns=[], nas=["novalue", "None", "none"], replace=False):

    """
    unify missing values
    can specify columns. If None, all missing columns will unify

    nas: define the searching criteria of missing

    replace: whether to replace the missing columns, default = False
    if False, will create new column with _useNA ending

    Example
    -------
    >> data = np.arange(15).reshape(5, 3)
    >> data = pd.DataFrame(data, columns = ['column_1', 'column_2', 'column_3'])
    >> data.loc[:, 'column_1'] = 'novalue'
    >> data.loc[3, 'column_2'] = 'None'
    >> data

      column_1 column_2  column_3
    0  novalue        1         2
    1  novalue        4         5
    2  novalue        7         8
    3  novalue     None        11
    4  novalue       13        14

    >> data = unify_nan(data)
    >> data

      column_1 column_2  column_3  column_1_useNA  column_2_useNA
    0  novalue        1         2             NaN             1.0
    1  novalue        4         5             NaN             4.0
    2  novalue        7         8             NaN             7.0
    3  novalue     None        11             NaN             NaN
    4  novalue       13        14             NaN            13.0
    """

    # if not specified, all columns containing nas values will add to columns
    if not columns:
        columns = []
        for column in list(dataset.columns):
            if dataset[column].isin(nas).any():  # check any values in nas
                columns.append(column)

    # if only string for one column is available, change it to list
    if isinstance(columns, str):
        columns = [columns]

    # nas dictionary
    nas_dict = {}
    for _nas in nas:
        nas_dict[_nas] = np.nan

    # unify the nan values
    for _column in columns:
        if replace:  # if replace, replace missing column with unified nan one
            dataset[_column] = dataset[_column].replace(nas_dict)
        else:
            dataset[_column + "_useNA"] = dataset[_column].replace(nas_dict)

    return dataset


def remove_index_columns(
    data, index=[], columns=[], axis=1, threshold=1, reset_index=True, save=False
):

    """
    delete columns/indexes with majority being nan values
    limited/no information these columns/indexes provided

    Parameters
    ----------
    data: input data

    index: whether to specify index range, default = []
    default will include all indexes

    columns: whether to specify column range, default = []
    default will include all columns

    axis: on which axis to remove, default = 1
    axis = 1, remove columns; axis = 0, remove indexes

    threshold: criteria of missing percentage, whether to remove column, default = 1
    accpetable types: numeric in [0, 1], or list of numerics
    if both columns and threshold are lists, two can be combined corresponding

    reset_index: whether to reset index after dropping

    save: save will store the removing columns to another file
    """

    remove_index = []  # store index need removing
    remove_column = []  # store columns need removing

    # make sure it's dataframe
    if not isinstance(data, pd.DataFrame):
        try:
            data = pd.DataFrame(data)
        except:
            raise TypeError("Expect a dataframe, get {}.".format(type(data)))

    n, p = data.shape  # number of observations/features in the dataset

    if axis == 1:  # remove columns
        if not columns and index:  # in case users confuse index for columns
            columns = index
        else:
            columns = list(data.columns) if not columns else columns
    elif axis == 0:  # remove index
        if not index and columns:  # in case users confuse columns for index
            index = columns
        else:
            index = list(data.index) if not index else index

    if isinstance(threshold, list):
        # if threshold a list, use specified threshold list for each feature
        if axis == 1:  # remove columns
            if len(columns) != len(threshold):
                raise ValueError(
                    "Columns and threshold should be same size, get {} and {}.".format(
                        len(columns), len(threshold)
                    )
                )
            for _column, _threshold in zip(columns, threshold):
                # only delete column when missing percentage larger than threshold
                if data[_column].isnull().values.sum() / n >= _threshold:
                    remove_column.append(_column)
        elif axis == 0:  # remove index
            if len(index) != len(threshold):
                raise ValueError(
                    "Indexes and threshold should be same size, get {} and {}.".format(
                        len(index), len(threshold)
                    )
                )
            for _index, _threshold in zip(index, threshold):
                if data.loc[_index, :].isnull().values.sum() / p >= _threshold:
                    remove_index.append(_index)
    else:
        if axis == 1:  # remove columns
            for _column in columns:
                if data[_column].isnull().values.sum() / n >= threshold:
                    remove_column.append(_column)
        elif axis == 0:  # remove indexes
            for _index in index:
                if data.loc[_index, :].isnull().values.sum() / p >= threshold:
                    remove_index.append(_index)

    # save the removing columns to another file
    if save:
        if axis == 1:
            data[remove_column].to_csv(
                "Removing_data(Limited_Information).csv", index=False
            )
        elif axis == 0:
            data[remove_index].to_csv(
                "Removing_data(Limited_Information).csv", index=False
            )

    if axis == 1:  # remove columns
        data.drop(remove_column, axis=1, inplace=True)
    elif axis == 0:  # remove index
        data.drop(remove_index, axis=0, inplace=True)

    if reset_index:  # whether to reset index
        data.reset_index(drop=True, inplace=True)

    return data


# get missing matrix
def get_missing_matrix(
    data, nas=["nan", "NaN", "NaT", "NA", "novalue", "None", "none"], missing=1
):

    """
    Get missing matrix for datasets

    Parameters
    ----------
    data: data containing missing values,
    acceptable type: pandas.DataFrame, numpy.ndarray

    nas: list of different versions of missing values
    (convert to string, since not able to distinguish np.nan)

    missing: convert missing indexes to 1/0, default = 1

    Example
    -------
    >> a = pd.DataFrame({
    >>     'column_1' : [1, 2, 3, np.nan, 5, 'NA'],
    >>     'column_2' : [7, 'novalue', 'none', 10, 11, None],
    >>     'column_3' : [np.nan, '3/12/2000', '3/13/2000', np.nan, '3/12/2000', '3/13/2000']
    >> })
    >> a['column_3'] = pd.to_datetime(a['column_3'])
    >> print(get_missing_matrix(a))

    [[0 0 1]
     [0 1 0]
     [0 1 0]
     [1 0 1]
     [0 0 0]
     [1 1 0]]
    """

    # make sure input becomes array
    # if data is dataframe, get only values
    if isinstance(data, pd.DataFrame):
        data = data.values

    # if not numpy.array, raise Error
    if not isinstance(data, np.ndarray):
        raise TypeError("Expect a array, get {}.".format(type(data)))

    # since np.nan != np.nan, convert data to string for comparison
    missing_matrix = np.isin(data.astype(str), nas)

    # if missing == 1 :
    #     missing_matrix = missing_matrix.astype(int)
    # elif missing == 0 :
    #     missing_matrix = 1 - missing_matrix.astype(int)

    # convert True/False array to 1/0 array
    # one line below works the same as above
    missing_matrix = np.abs(1 - missing - missing_matrix.astype(int))

    return missing_matrix


# text preprocessing
# build a vocabulary from text
def text_processing(
    data,
    batch_size=32,
    shuffle=True,
    return_offset=False,
):

    from torchtext.data.utils import get_tokenizer
    from torchtext.vocab import build_vocab_from_iterator

    tokenizer = get_tokenizer("basic_english")
    data_iter = iter(data)

    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)

    # define vocabulary
    vocab = build_vocab_from_iterator(yield_tokens(data_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    # tokenize data and build vocab
    text_pipeline = lambda x: vocab(tokenizer(x))
    # label_pipeline = lambda x: int(x) - 1

    # return text, label, and offset (optional)
    def collate_batch(batch):
        text_list, label_list, offsets = [], [], []
        for (_text, _label) in batch:
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            label_list.append(_label)
            if return_offset:
                offsets.append(processed_text.size(0))

        text_list = torch.cat(text_list)
        label_list = torch.tensor(label_list, dtype=torch.int64)
        if return_offset:
            offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)

        if return_offset:
            return text_list.to(device), label_list.to(device), offsets.to(device)
        else:
            return text_list.to(device), label_list.to(device)

    # load data to DataLoader
    data_loader = DataLoader(
        data_iter, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch
    )

    return data_loader, vocab
