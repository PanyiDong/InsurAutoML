"""
File Name: _base.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_base.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 14th November 2022 8:20:52 pm
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

import os
import glob
import numpy as np
import pandas as pd
import warnings

# R environment
# check if rpy2 available
# if not, can not read R type data
import importlib

rpy2_spec = importlib.util.find_spec("rpy2")
if rpy2_spec is not None:
    import rpy2
    import rpy2.robjects as ro
    from rpy2.robjects import Formula, pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr


class no_processing:

    """
    No processing on data, asa comparison
    """

    def __init__(self):

        self._fitted = False  # record whether the method has been fitted

    def fit(self, X, y=None):

        self._fitted = True

        return self

    def fill(self, X):

        self._fitted = True

        return X

    def transform(self, X):

        return X

    def fit_transform(self, X, y=None):

        self._fitted = True

        if isinstance(y, pd.DataFrame):
            _empty = y.isnull().all().all()
        elif isinstance(y, pd.Series):
            _empty = y.isnull().all()
        elif isinstance(y, np.ndarray):
            _empty = np.all(np.isnan(y))
        else:
            _empty = y == None

        if _empty:
            return X
        else:
            return X, y

    def inverse_transform(self, X):

        self._fitted = False

        return X


class load_data:

    """
    load all data files to dict

    Parameters
    ----------
    path: path of the files to search for, can be list of paths

    data_type: matching data file types, default = 'all'
    supported types ('all', '.csv', '.asc', '.data', '.rda', '.rdata')

    """

    def __init__(self, data_type="all"):
        self.data_type = data_type
        self.database = {}

    def load(self, path, filename=None):

        if isinstance(path, list):  # add / at the end of path
            path = [
                (_path if (_path == "" or _path[-1] == "/") else _path + "/")
                for _path in path
            ]
        else:
            path = [(path if (path == "" or path[-1] == "/") else path + "/")]

        for _path in path:
            self._main(_path, filename)
        return self.database

    def _main(self, path, filename):

        # initialize path sets
        _csv_files = []
        _data_files = []
        _rda_files = []
        _rdata_files = []

        # load .csv/.data files in the path
        if (
            self.data_type == ".csv"
            or self.data_type == ".data"
            or self.data_type == ".asc"
            or self.data_type == "all"
        ):
            if self.data_type == ".csv" or self.data_type == "all":
                if filename == None:
                    _csv_files = glob.glob(path + "*.csv")
                elif isinstance(filename, list):
                    _csv_files = []
                    for _filename in filename:
                        _csv_files += glob.glob(path + _filename + ".csv")
                else:
                    _csv_files = glob.glob(path + filename + ".csv")
            if self.data_type == ".data" or self.data_type == "all":
                if filename == None:
                    _data_files = glob.glob(path + "*.data")
                elif isinstance(filename, list):
                    _data_files = []
                    for _filename in filename:
                        _data_files += glob.glob(path + _filename + ".data")
                else:
                    _data_files = glob.glob(path + filename + ".data")
            if self.data_type == ".asc" or self.data_type == "all":
                if filename == None:
                    _data_files = glob.glob(path + "*.asc")
                elif isinstance(filename, list):
                    _data_files = []
                    for _filename in filename:
                        _data_files += glob.glob(path + _filename + ".asc")
                else:
                    _data_files = glob.glob(path + filename + ".asc")

            if not _csv_files and self.data_type == ".csv":
                warnings.warn("No .csv files found!")
            elif not _data_files and self.data_type == ".data":
                warnings.warn("No .data file found!")
            elif not _data_files and self.data_type == ".asc":
                warnings.warn("No .asc file found!")
            elif _csv_files + _data_files:
                for _data_path in _csv_files + _data_files:
                    # in linux path, the path separator is '/'
                    # in windows path, the path separator is '\\'
                    # _filename = (
                    #     _data_path.split("/")[-1]
                    #     if "/" in _data_path
                    #     else _data_path.split("\\")[-1]
                    # )
                    # use os.path.split for unify path separator
                    _filename = os.path.split(_data_path)[-1]
                    self.database[_filename.split(".")[0]] = pd.read_csv(_data_path)

        # load .rda/.rdata files in the path
        # will not read any files if rpy2 is not available
        if rpy2_spec is None:
            if self.data_type == ".rda" or self.data_type == ".rdata":
                raise ImportError("Require rpy2 package, package not found!")
            if self.data_type == "all":
                pass
        else:
            if (
                self.data_type == ".rda"
                or self.data_type == ".rdata"
                or self.data_type == "all"
            ):
                if self.data_type == ".rda" or self.data_type == "all":
                    if filename == None:
                        _rda_files = glob.glob(path + "*.rda")
                    elif isinstance(filename, list):
                        _rda_files = []
                        for _filename in filename:
                            _rda_files += glob.glob(path + _filename + ".rda")
                    else:
                        _rda_files = glob.glob(path + filename + ".rda")
                if self.data_type == ".rdata" or self.data_type == "all":
                    if filename == None:
                        _rdata_files = glob.glob(path + "*.rdata")
                    elif isinstance(filename, list):
                        _rdata_files = []
                        for _filename in filename:
                            _rdata_files += glob.glob(path + _filename + ".rdata")
                    else:
                        _rdata_files = glob.glob(path + filename + ".rdata")

                if not _rda_files and self.data_type == ".rda":
                    warnings.warn("No .rda file found!")
                elif not _rdata_files and self.data_type == ".rdata":
                    warnings.warn("No .rdata file found!")
                elif _rda_files + _rdata_files:
                    for _data_path in _rda_files + _rdata_files:
                        # in linux path, the path separator is '/'
                        # in windows path, the path separator is '\\'
                        # _filename = (
                        #     _data_path.split("/")[-1]
                        #     if "/" in _data_path
                        #     else _data_path.split("\\")[-1]
                        # )
                        # use os.path.split for unify path separator
                        _filename = os.path.split(_data_path)[-1]
                        ro.r('load("' + _data_path + '")')
                        ro.r("rdata = " + _filename.split(".")[0])
                        with localconverter(ro.default_converter + pandas2ri.converter):
                            self.database[
                                _filename.split(".")[0]
                            ] = ro.conversion.rpy2py(ro.r.rdata)

        if self.data_type == "all" and not self.database:
            warnings.warn("No file found!")
