from calendar import c
import numpy as np
import pandas as pd
from sklearn import preprocessing
from ._utils import is_date

class DataEncoding():

    '''
    Data preprocessing
    1. convert string type features to numerical categorical/dummy variables
    2. transform non-categorical features
    3. refit for test data (in cases train/test data is already divided), 
    using category table recorded while convert train data, only deal with non nan values

    Parameters
    ----------
    df: data

    dummy_coding: whether to use dummy variables, default = False
    if True, convert string categories to numerical categories(0, 1, 2, ...)

    transform: how to transform numerical features, default = False
    'standardize', 'center', 'log' are available

    '''

    def __init__(self, dummy_coding=False, transform=False):
        self.dummy_coding = dummy_coding
        self.transform = transform

    def fit(self, _df):
        df = _df.copy(deep = True)
        features = list(df.columns)
        self.category = pd.DataFrame()
        self.mean_scaler = {}
        self.sigma_scaler = {}
        for column in features:
            if df[column].dtype == object and is_date(df[[column]]) and len(df[column].dropna().unique()) > 31:
                df[column] = pd.to_numeric(pd.to_datetime(df[column]))
            elif (df[column].dtype == object) or (str(df[column].dtype) == 'category') :
                # dummy coding for string categorical features
                if str(df[column].dtype) == "category" :
                    df[column] = df[column].astype(str)
                if self.dummy_coding == True:
                    unique_value = np.sort(df[column].dropna().unique())
                    if self.category.empty:
                        self.category = pd.DataFrame(
                            {column: unique_value})
                    else:
                        self.category = pd.concat([self.category, pd.DataFrame({column: unique_value})],
                                                  axis=1)
                    for elem in unique_value:
                        df[column + '_' +
                            str(elem)] = (df[column] == elem).astype(int)
                else:
                    unique_value = np.sort(df[column].dropna().unique())
                    if self.category.empty:
                        self.category = pd.DataFrame(
                            {column: unique_value})
                    else:
                        self.category = pd.concat([self.category, pd.DataFrame({column: unique_value})],
                                                  axis=1)
                    for i in range(len(unique_value)):
                        df.loc[df[column] == unique_value[i], column] = i
                    df.loc[~df[column].isnull(), column] = df.loc[~df[column].isnull(), column].astype(int)
            else :
                df.loc[~df[column].isnull(), column] = df.loc[~df[column].isnull(), column].astype(float)
                # standardardize numerical features
                if self.transform == 'standardize':
                    standard_scaler = preprocessing.StandardScaler().fit(
                        df[[column]].values)
                    # save scale map for scale back
                    self.mean_scaler.update({column: standard_scaler.mean_[0]})
                    self.sigma_scaler.update(
                        {column: standard_scaler.scale_[0]})
                    df[column] = standard_scaler.transform(df[[column]].values)
                elif self.transform == 'center':
                    standard_scaler = preprocessing.StandardScaler().fit(
                        df[[column]].values)
                    # save scale map for scale back
                    self.mean_scaler.update({column: standard_scaler.mean_[0]})
                    df.loc[~df[column].isnull(), column] = df.loc[~df[column].isnull(), column] \
                        - standard_scaler.mean_[0]
                elif self.transform == 'log':
                    df.loc[~df[column].isnull(), column] = np.log(df.loc[~df[column].isnull(), column])

        # remove categorical variables
        if self.dummy_coding == True:
            df.drop(columns=list(self.category.columns), inplace=True)
        
        return df

    def refit(self, _df) :
        df = _df.copy(deep = True)
        if self.category.empty :
            raise ValueError('No category table exists!')
        categorical_features = list(self.category.columns)
        for column in list(df.columns) :
            if df[column].dtype == object and is_date(df[[column]]) and len(df[column].dropna().unique()) > 31 :
                df[column] = pd.to_numeric(pd.to_datetime(df[column]))
            elif df[column].dtype == object or str(df[column].dtype) == 'category':
                if str(df[column].dtype) == 'category' :
                    df[column] = df[column].astype(str)
                if column in categorical_features:  # map categorical testdata based on category
                    unique_values = self.category.loc[self.category[column].notnull(), column].values  # Select only non nan values
                    if self.dummy_coding == True :
                        for value in unique_values:
                            df[str(column) + '_' + str(value)] = (df[column] == value).astype(int)
                            # Notice: categorical values not appear in traindata will be dropped,
                            # it's not a problem seems it can not be trained if it's not even in train data
                            # if False in (np.sort(unique_values) == np.sort(df[column].unique())) :
                            #     raise ValueError('Testdata has unkown categories!')
                    else :
                        for i in range(len(unique_values)) :
                            df.loc[df[column] == unique_values[i], column] = i
                        df.loc[~df[column].isin(unique_values), column] = np.NaN
                        df.loc[~df[column].isnull(), column] = df.loc[~df[column].isnull(), column].astype(int)
            else :
                df.loc[~df[column].isnull(), column] = df.loc[~df[column].isnull(), column].astype(float)
                # standardardize numerical features
                if self.transform == 'standardize':
                    standard_scaler = preprocessing.StandardScaler().fit(
                        df[[column]].values)
                    # save scale map for scale back
                    self.mean_scaler.update({column: standard_scaler.mean_[0]})
                    self.sigma_scaler.update(
                        {column: standard_scaler.scale_[0]})
                    df[column] = standard_scaler.transform(df[[column]].values)
                elif self.transform == 'center':
                    standard_scaler = preprocessing.StandardScaler().fit(
                        df[[column]].values)
                    # save scale map for scale back
                    self.mean_scaler.update({column: standard_scaler.mean_[0]})
                    df.loc[~df[column].isnull(), column] = df.loc[~df[column].isnull(), column] \
                        - standard_scaler.mean_[0]
                elif self.transform == 'log':
                    df.loc[~df[column].isnull(), column] = np.log(df.loc[~df[column].isnull(), column])

        # remove categorical variables
        if self.dummy_coding == True:
            df.drop(columns=list(self.category.columns), inplace=True)
        
        return df


class CategoryShift() :

    '''
    Add 3 to every cateogry

    Parameters
    ----------
    seed: random seed
    '''

    def __init__(
        self,
        seed = 1
    ) :
        self.seed = seed

    def fit(self, X) :
        
        # Check data type
        columns = list(X.columns)
        for _column in columns :
            if X[_column].dtype == object :
                raise ValueError('Cannot handle object type!')
            elif str(X[_column].dtype) == 'category' :
                raise ValueError('Cannot handle categorical type!')

    def transform(self, X) :
        
        _X = X.copy(deep = True)
        _X += 3
        return _X
