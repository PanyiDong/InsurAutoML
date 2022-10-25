"""
File: _base.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Latest Version: 0.2.0
Relative Path: /My_AutoML/_utils/_base.py
File Created: Wednesday, 6th April 2022 12:01:20 am
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Sunday, 25th September 2022 7:51:20 pm
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

import warnings
import time
import numpy as np
import pandas as pd
from dateutil.parser import parse

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

    # if number of samples is larger than limit, raise error
    if total < n:
        raise ValueError(
            "Total number of samples must be greater than or equal to the number of samples to be drawn. Got total={}, n={}".format(
                total, n
            )
        )
    # if number of samples is 0, raise error
    elif n == 0:
        raise ValueError(
            "Number of samples to be drawn must be greater than 0. Got n={}.".format(n)
        )

    # NOTE: Sep. 25, 2022
    # Use native numpy function to speed up. Original code is decrypted.
    # output = []
    # vlist = [i for i in range(total)]
    # for _ in range(n):
    #     # np.random.seed(int(datetime.now().strftime("%H%M%S")))
    #     index = np.random.randint(0, high=len(vlist), size=1)[0]
    #     output.append(vlist[index])
    #     vlist.pop(index)

    return np.random.choice(total, n, replace=False)


# Return randomly shuffle of a list (unique values only)
def random_list(vlist, seed=1):
    if seed != None:
        np.random.seed(seed)

    # NOTE: Sep. 25, 2022
    # Use native numpy function to speed up. Original code is decrypted.
    # output = []
    # for _ in range(len(vlist)):
    #     # np.random.seed(int(datetime.now().strftime("%H%M%S")))
    #     index = np.random.randint(0, high=len(vlist), size=1)[0]
    #     output.append(vlist[index])
    #     vlist.pop(index)

    return np.random.permutation(vlist)


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


# Return location of minimum values
def minloc(vlist):

    # make sure input is np.array
    if not isinstance(vlist, np.ndarray):
        vlist = np.array(vlist)

    if len(vlist) == 0:
        raise ValueError("Invalid List!")
    # NOTE: Sep. 25, 2022
    # Use native numpy function to speed up. Original code is decrypted.
    # elif len(vlist) == 1:
    #     return 0
    # else:
    #     result = 0
    #     for i in range(len(vlist) - 1):
    #         if vlist[i + 1] < vlist[result]:
    #             result = i + 1
    #         else:
    #             continue
    #     return result
    else:

        # check whether multi-dimensional array
        if len(vlist.shape) > 1:
            warnings.warn(
                "Input is multi-dimensional array, return min location of the first dimension."
            )

        return np.argmin(vlist, axis=0)


# Return location of maximum values
def maxloc(vlist):

    # make sure input is np.array
    if not isinstance(vlist, np.ndarray):
        vlist = np.array(vlist)

    if len(vlist) == 0:
        raise ValueError("Invalid List!")
    # NOTE: Sep. 25, 2022
    # Use native numpy function to speed up. Original code is decrypted.
    # elif len(vlist) == 1:
    #     return 0
    # else:
    #     result = 0
    #     for i in range(len(vlist) - 1):
    #         if vlist[i + 1] > vlist[result]:
    #             result = i + 1
    #         else:
    #             continue
    #     return result
    else:

        # check whether multi-dimensional array
        if len(vlist.shape) > 1:
            warnings.warn(
                "Input is multi-dimensional array, return max location of the first dimension."
            )

        return np.argmax(vlist, axis=0)


# return the index of Boolean list or {0, 1} list
# default 1 consider as True
def True_index(X, _true=[True, 1]):

    result = [i for i, value in enumerate(X) if value in _true]

    return result


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


# determine whether using a python terminal environment or
# a jupyter notebook environment
def type_of_script():

    try:
        ipy_str = str(type(get_ipython()))
        if "zmqshell" in ipy_str:
            return "jupyter"
        if "terminal" in ipy_str:
            return "ipython"
    except:
        return "terminal"


# determine whether a method exists in a class
def has_method(obj, name):
    return callable(getattr(obj, name, None))


# check if is None
def is_none(item, pat=[None, "None", "none", "NONE"]):

    if item in pat:
        return True
    else:
        return False
