import os
import shutil
import glob
from dateutil.parser import parse
import numpy as np
import pandas as pd
import warnings
import sklearn
from sklearn.tree import ExtraTreeClassifier

import autosklearn
import autosklearn.classification
import autosklearn.regression

# R environment
import rpy2
import rpy2.robjects as ro
from rpy2.robjects import Formula, pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr


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

class no_processing() :

    '''
    No processing on data, asa comparison
    '''

    def fit(self, X) :
        return self
    
    def transform(self, X) :
        return X

    def fit_transform(self, X) :
        return X

class load_data() :

    '''
    load all data files to dict

    Parameters
    ----------
    path: path of the files to search for, can be list of paths

    data_type: matching data file types, default = 'all'
    supported types ('all', '.csv', '.asc', '.data', '.rda', '.rdata')

    '''

    def __init__(
        self,
        data_type = 'all'
    ) :
        self.data_type = data_type
        self.database = {}

    
    def load(self, path, filename = None) :
        
        if isinstance(path, list) :  # add / at the end of path
            path = [(_path if _path[-1] == '/' else _path + '/') for _path in path]
        else :
            path = [(path if path[-1] == '/' else path + '/')]
        
        for _path in path :
            self._main(_path, filename)
        return self.database

    def _main(self, path, filename) :
        
        # initilize path sets
        _csv_files = []
        _data_files = []
        _rda_files = []
        _rdata_files = []

        # load .csv/.data files in the path
        if self.data_type == '.csv' or self.data_type == '.data' or \
            self.data_type == '.asc' or self.data_type == 'all' :
            if self.data_type == '.csv' or self.data_type == 'all' :
                if filename == None :
                    _csv_files = glob.glob(path + '*.csv')
                else :
                    _csv_files = glob.glob(path + filename + '.csv')
            if self.data_type == '.data' or self.data_type == 'all' :
                if filename == None :
                    _data_files = glob.glob(path + '*.data')
                else :
                    _data_files = glob.glob(path + filename + '.data')
            if self.data_type == '.asc' or self.data_type == 'all' :
                if filename == None :
                    _data_files = glob.glob(path + '*.asc')
                else :
                    _data_files = glob.glob(path + filename + '.asc')

            if not _csv_files and self.data_type == '.csv' :
                warnings.warn('No .csv files found!')
            elif not _data_files and self.data_type == '.data' :
                warnings.warn('No .data file found!')
            elif not _data_files and self.data_type == '.asc' :
                warnings.warn('No .asc file found!')
            elif _csv_files + _data_files :
                for _data_path in (_csv_files + _data_files) :
                    _filename = _data_path.split('/')[-1]
                    self.database[_filename.split('.')[0]] = pd.read_csv(_data_path)

        # load .rda/.rdata files in the path
        if self.data_type == '.rda' or self.data_type == '.rdata' or self.data_type == 'all' :
            if self.data_type == '.rda' or self.data_type == 'all' :
                if filename == None :
                    _rda_files = glob.glob(path + '*.rda')
                else :
                    _rda_files = glob.glob(path + filename + '.rda')
            if self.data_type == '.rdata' or self.data_type == 'all' :
                if filename == None :
                    _rdata_files = glob.glob(path + '*.rdata')
                else :
                    _rdata_files = glob.glob(path + filename + '.rdata')
            
            if not _rda_files and self.data_type == '.rda' :
                warnings.warn('No .rda file found!')
            elif not _rdata_files and self.data_type == '.rdata' :
                warnings.warn('No .rdata file found!')
            elif _rda_files + _rdata_files :
                for _data_path in (_rda_files + _rdata_files) :
                    _filename = _data_path.split('/')[-1]
                    ro.r('load("' + _data_path + '")')
                    ro.r('rdata = ' + _filename.split('.')[0])
                    with localconverter(ro.default_converter + pandas2ri.converter):
                        self.database[_filename.split('.')[0]] = \
                            ro.conversion.rpy2py(ro.r.rdata)

        if self.data_type == 'all' and not self.database :
            warnings.warn('No file found!')