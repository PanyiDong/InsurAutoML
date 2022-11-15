"""
File Name: _data.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_utils/_data.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 14th November 2022 9:23:07 pm
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

from typing import Any, Tuple, List, Union
import os
import re
import ast
import json
import numpy as np
import pandas as pd

from ._base import random_index
from InsurAutoML._constant import UNI_CLASS

# string list to list
def str2list(item: str) -> list:

    try:
        return ast.literal_eval(item)
    except:
        return item


# string dict to dict
def str2dict(item: str) -> dict:

    try:
        return json.loads(item)
    except:
        return item


# Train test split using test set percentage
def train_test_split(
    X: Any, y: Any, test_perc: float = 0.15, seed: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

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


# transform between numpy array and pandas dataframe
# to deal with some problems where dataframe will be converted to array using sklearn objects
class as_dataframe:
    def __init__(self) -> None:
        self.design_matrix = None  # record the values of dataframe
        self.columns = None  # record column heads for the dataframe

    def to_array(self, X: pd.DataFrame) -> np.ndarray:

        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input should be dataframe!")

        self.design_matrix = X.values
        self.columns = list(X.columns)

        return self.design_matrix

    def to_df(self, X: Any = None, columns: List = None) -> pd.DataFrame:

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
        columns: List = [],
        numerics: List[str] = [
            "int16",
            "int32",
            "int64",
            "float16",
            "float32",
            "float64",
        ],
        nas: List = [np.nan, None, "nan", "NaN", "NA", "novalue", "None", "none"],
        allow_string: bool = False,
        inplace: bool = True,
    ) -> None:
        self.columns = columns
        self.numerics = numerics
        self.nas = nas
        self.allow_string = allow_string
        self.inplace = inplace

        self.type_table = {}  # store type of every column
        self.unique_table = {}  # store unique values of categorical columns

    # factorize data without changing values in nas
    # pd.factorize will automatically convert missing values
    def factorize(self, data: pd.Series) -> pd.Series:

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
    def unify_cate(self, x: Any, list: List) -> Any:

        if not x in list and str(x) not in self.nas:
            x = np.argmin(np.abs([item - x for item in list]))

        return x

    def fit(self, X: pd.DataFrame) -> None:

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

    def refit(self, X: pd.DataFrame) -> Union[None, pd.DataFrame]:

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

        if not self.inplace:
            return X


def unify_nan(
    dataset: pd.DataFrame,
    columns: List = [],
    nas: List[str] = ["novalue", "None", "none"],
    replace: bool = False,
) -> pd.DataFrame:

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
    data: pd.DataFrame,
    index: List = [],
    columns: List = [],
    axis: int = 1,
    threshold: float = 1,
    reset_index: bool = True,
    save: bool = False,
) -> pd.DataFrame:

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
    data: pd.DataFrame,
    nas: List[str] = ["nan", "NaN", "NaT", "NA", "novalue", "None", "none"],
    missing: int = 1,
) -> pd.DataFrame:

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


# determine whether the data contains imbalanced data
# if value, returns the column header and majority class from the unbalanced dataset
def is_imbalance(
    data: pd.DataFrame, threshold: float, value: bool = False
) -> Union[bool, Tuple[str, float]]:

    features = list(data.columns)

    for _column in features:
        unique_values = data[_column].unique()

        if len(unique_values) == 1:  # only one class exists
            if value:
                return None, None
            else:
                return False

        for _value in unique_values:
            if (
                len(data.loc[data[_column] == _value, _column]) / len(data[_column])
                > threshold
            ):
                if value:
                    return _column, _value
                else:
                    return True
    if value:
        return None, None
    else:
        return False


# return the distance between sample and the table sample points
# notice: the distance betwen sample and sample itself will be included, be aware to deal with it
# supported norm ['l1', 'l2']
def LinkTable(sample: pd.DataFrame, table: pd.DataFrame, norm: str = "l2") -> List:

    if sample.shape[1] != table.shape[1]:
        raise ValueError("Not same size of columns!")

    _sample = sample.values
    features = list(table.columns)
    _table = table.copy(deep=True)
    _linktable = []

    for sample_point in _sample:
        for i in range(len(features)):
            if norm == "l2":
                # print(sample[_column], sample[_column][0])
                _table.iloc[:, i] = (_table.iloc[:, i] - sample_point[i]) ** 2
            if norm == "l1":
                _table.iloc[:, i] = np.abs(_table.iloc[:, i] - sample_point[i])
        _linktable.append(_table.sum(axis=1).values.tolist())

    return _linktable


class ExtremeClass:

    """
    remove the features where only one unique class exists, since no information provided by the feature

    Parameters
    ----------
    extreme_threshold: default = 1
    the threshold percentage of which the class holds in the feature will drop the feature
    """

    def __init__(self, extreme_threshold: float = 1) -> None:
        self.extreme_threshold = extreme_threshold

    def cut(self, X: pd.DataFrame) -> pd.DataFrame:

        _X = X.copy(deep=True)

        features = list(_X.columns)
        for _column in features:
            unique_values = sorted(_X[_column].dropna().unique())
            for _value in unique_values:
                if (
                    len(X.loc[X[_column] == _value, _column]) / len(X)
                    >= self.extreme_threshold
                ):
                    _X.remove(labels=_column, inplace=True)
                    break
        return _X


# convert a (n_samples, n_classes) array to a (n_samples, 1) array
# assign the class with the largest probability to the sample
# common use of this function is to convert the prediction of the class
# from neural network to actual predictions
def assign_classes(list: List) -> List:

    return np.array([np.argmax(item) for item in list])


# softmax function that can handle 1d data
def softmax(df: pd.DataFrame) -> pd.DataFrame:

    if len(df.shape) == 1:
        ppositive = 1 / (1 + np.exp(-df))
        ppositive[ppositive > 0.999999] = 1
        ppositive[ppositive < 0.0000001] = 0
        return np.transpose(np.array((1 - ppositive, ppositive)))
    else:
        tmp = df - np.max(df, axis=1).reshape((-1, 1))
        tmp = np.exp(tmp)
        return tmp / np.sum(tmp, axis=1).reshape((-1, 1))


# plot high dimensional data
def plotHighDimCluster(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.DataFrame, np.ndarray],
    plot: bool = True,
    method: str = "PCA",
    dim: int = 2,
    save: bool = False,
    path: str = "tmp",
) -> None:

    import matplotlib.pyplot as plt

    # check input shape:
    # 1. X must be high dimensional data
    # 2. y must be 1-dimensional labels
    if X.shape[1] <= 3:
        raise ValueError(
            "X must be high dimensional data, dimension = {} have no need for this plotting".format(
                X.shape[1]
            )
        )

    if len(X) != len(y):
        raise ValueError("X and y must have the same length")

    # method PCA
    if method == "PCA":
        from sklearn.decomposition import PCA

        # initiate PCA
        pca = PCA(n_components=dim)
        project_result = pca.fit_transform(X)
    elif method == "TSNE":
        # if too many number of features, use PCA first to reduce computational cost
        if X.shape[1] > 100:
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE

            pca = PCA(n_components=100)
            tsne = TSNE(n_components=dim)
            project_result = pca.fit_transform(X)
            project_result = tsne.fit_transform(project_result)
        else:
            from sklearn.manifold import TSNE

            tsne = TSNE(n_components=dim)
            project_result = tsne.fit_transform(X)

    # initiate storing dataframe
    df = pd.DataFrame()
    for _dim in range(dim):
        df["PCA_" + str(_dim)] = project_result[:, _dim]

    if dim == 2:
        plt.figure()
        plt.scatter(df["PCA_0"], df["PCA_1"], c=y, cmap="RdYlBu", alpha=0.4)
        plt.xlabel("PCA_1")
        plt.ylabel("PCA_2")
    if dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(df["PCA_0"], df["PCA_1"], df["PCA_2"], c=y, cmap="RdYlBu", alpha=0.4)
        ax.set_xlabel("PCA_1")
        ax.set_ylabel("PCA_2")
        ax.set_zlabel("PCA_3")

    if save:
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(os.path.join(path, "HighDim.png").replace("\\", "/"))

    if plot:
        plt.show()
    plt.close()


# distinguish from numerical categorical and numerical continuous
def numerical_categorical_continuous(df: pd.DataFrame) -> str:

    if len(pd.unique(df)) <= min(UNI_CLASS, 0.1 * len(df)):
        return "numerical_categorical"
    else:
        return "continuous"


# distinguish the data type of each column between string categorical and text
def string_categorical_text(df: pd.DataFrame) -> str:

    # # check if the input is a dataframe
    # if not isinstance(df, pd.DataFrame):
    #     try:
    #         df = pd.DataFrame(df)
    #     except:
    #         raise ValueError("Input must be a dataframe or convertible to dataframe")

    if len(pd.unique(df)) <= min(UNI_CLASS, 0.1 * len(df)):
        return "string_categorical"
    else:
        return "text"


# distinguish numerical and categorical features
# one column at a time
def feature_type(df: Union[pd.DataFrame, pd.Series]) -> str:

    if not df.dtype == "object":
        return numerical_categorical_continuous(df)
    else:
        return string_categorical_text(df)


# text preprocessing
def text_preprocessing(text: str) -> list:

    # remove anything inside brackets
    extract_txt = re.sub("[\(\[].*?[\)\]]", "", text)
    # get only alphabets and numerics info
    use_txt = re.sub("[^a-zA-Z0-9]", " ", extract_txt)
    # get tokens of the text
    token_txt = use_txt.split(" ")
    # get lower case of the tokens
    lower_txt = [item.lower() for item in token_txt]

    return [item for item in lower_txt if item != ""]


# use word2vec to convert a text column to vector column
def text2vec(
    df: Union[pd.DataFrame, pd.Series],
    method: str = "Word2Vec",
    pretrained: str = None,
    dim: int = 100,
    how: str = "sum",
    return_type: str = "df",
):
    # get prefix
    if isinstance(df, pd.DataFrame):
        prefix = df.columns[0]
    elif isinstance(df, pd.Series):
        prefix = df.name

    # method to combine the word (in sentences/phrases) vectors
    how_dict = {
        "sum": np.sum,
        "mean": np.mean,
    }

    # pretrained word2vec model
    pretrained_str = [
        "fasttext-wiki-news-subwords-300",
        "conceptnet-numberbatch-17-06-300",
        "word2vec-ruscorpora-300",
        "word2vec-google-news-300",
        "glove-wiki-gigaword-50",
        "glove-wiki-gigaword-100",
        "glove-wiki-gigaword-200",
        "glove-wiki-gigaword-300",
        "glove-twitter-25",
        "glove-twitter-50",
        "glove-twitter-100",
        "glove-twitter-200",
    ]

    # apply text preprocessing
    preprocessed_df = df.apply(lambda x: text_preprocessing(x))

    if method == "Word2Vec":
        # if not pretrained, train the model
        if pretrained is None:
            # build word2vec model
            from gensim.models import Word2Vec

            model = Word2Vec(
                preprocessed_df, vector_size=dim, window=5, min_count=1, workers=4
            )
            KeyVectors = model.wv
            print(KeyVectors)
        else:
            # check if the pretrained model is valid
            if pretrained not in pretrained_str:
                raise ValueError(
                    "Pretrained model must be one of {}, get {}.".format(
                        pretrained_str, pretrained
                    )
                )

            # load the pretrained model
            import gensim.downloader

            KeyVectors = gensim.downloader.load(pretrained)

    # apply word2vec to convert text to vector
    vectorized_df = preprocessed_df.apply(
        lambda x: how_dict[how](KeyVectors[x], axis=0)
    )

    # expand the vector column to multiple columns by dimension
    if return_type == "df":
        return pd.DataFrame(
            vectorized_df.values.tolist(),
            index=vectorized_df.index,
            columns=[prefix + "_" + str(i) for i in range(dim)],
        )
    elif return_type == "pt":
        import torch

        return torch.tensor(vectorized_df.values.tolist())
