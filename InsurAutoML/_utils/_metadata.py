"""
File: _metadata.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_utils/_metadata.py
File: _metadata.py
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Tuesday, 15th November 2022 6:11:20 pm
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

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import warnings
from typing import Any, Dict, Tuple, List

from InsurAutoML._constant import UNI_CLASS, UNIQUE_FULLTYPE

# get subtype of int
def meta_map_int(data: pd.Series) -> str:
    if len(np.unique(data)) <= min(UNI_CLASS, 0.1 * len(data)):
        return "Categorical"
    else:
        return "Numerical"


# get subtype of text
def meta_map_object(data: pd.Series) -> str:
    # judge by unique values proportion
    if len(data.unique()) <= min(UNI_CLASS, 0.1 * len(data)):
        return "Categorical"
    # judge by average length
    try:
        txt_avg_len = pd.Series(data.unique()).str.split().str.len().mean()
        if txt_avg_len >= 3:
            return "Text"
    except:
        return "Categorical"


class get_details:
    def __init__(self, type: str, subtype: str) -> None:
        self.type = type
        self.subtype = subtype

    def get(self, data: pd.Series) -> Dict[Tuple, List] or dict:

        if self.type == "Float" or self.subtype == "Numerical":
            return self._get_details_numerical(data)
        elif self.subtype == "Categorical":
            return self._get_details_categorical(data)
        elif self.type in ["Datetime", "Path"] or self.subtype == "Text":
            return {}

    # get details of numerical data
    @staticmethod
    def _get_details_numerical(data: pd.Series) -> Dict[str, np.ndarray]:

        return {
            "mean": np.mean(data),
            "std": np.std(data),
            "min": np.min(data),
            "max": np.max(data),
        }

    # get details of categorical data
    @staticmethod
    def _get_details_categorical(data: pd.Series) -> Dict[str, dict]:
        return {
            "unique_cout": {
                key: value for key, value in zip(*np.unique(data, return_counts=True))
            }
        }


class MetaData:

    _unique_fulltype = UNIQUE_FULLTYPE

    def __init__(self, data: Any = None) -> None:

        # if fed data, get metadata from data
        if data is not None:
            self.get(data)

    def __repr__(self) -> str:
        # check whether metadata generated
        self._check_metadata()

        return str(self.metadata)

    def __str__(self) -> str:
        # check whether metadata generated
        self._check_metadata()

        return str(self.metadata)

    def _check_metadata(self):
        if not hasattr(self, "metadata"):
            raise AttributeError("Metadata not generated yet. Please use get() first.")

    def get(self, data: Any) -> Dict[Tuple, List] or dict:

        if isinstance(data, pd.DataFrame):
            return self.get_from_df(data)
        else:
            return self._get_from_others(data)

    def update(
        self, data: pd.DataFrame, names: List[str] = None
    ) -> Dict[Tuple, List] or dict:

        # check whether metadata generated
        self._check_metadata()

        # if names not provided, update all columns
        if names is None:
            names = data.columns

        for name in names:
            # remove registered metadata
            for key in self.metadata.keys():
                if name in self.metadata[key]:
                    self.metadata[key].remove(name)
                    # if no column has this metadata, remove the metadata
                    if len(self.metadata[key]) == 0:
                        del self.metadata[key]
                    break
            # no need to remove registered details
            # since it's updated per column

            # get the full type
            type, subtype = self._get_fulltype(data[name])

            # get the details
            details = get_details(type, subtype).get(data[name])

            # register the metadata
            self.register(name, (type, subtype), details)

        return self.metadata

    def force_update(
        self, names: List[str] or str, fulltypes: List[Tuple[str, str]] or str
    ) -> Dict[Tuple, List] or dict:

        # check whether metadata generated
        self._check_metadata()

        # if names is a string, convert it to a list
        if isinstance(names, str):
            names = [names]
        # if fulltypes is a string, convert it to a list
        if isinstance(fulltypes, str):
            fulltypes = [fulltypes]

        # check length of names and fulltypes
        if len(names) != len(fulltypes):
            raise ValueError(
                "Length of names and fulltypes must be the same. Got {} and {}.".format(
                    len(names), len(fulltypes)
                )
            )

        for name, fulltype in zip(names, fulltypes):
            # remove registered metadata
            for key in self.metadata.keys():
                if name in self.metadata[key]:
                    self.metadata[key].remove(name)
                    # if no column has this metadata, remove the metadata
                    if len(self.metadata[key]) == 0:
                        del self.metadata[key]
                    break
            # no need to remove registered details
            # since it's updated per column

            # register the metadata
            self.register(name, fulltype)

        return self.metadata

    def register(self, name: str, fulltype: dict, details: dict = None) -> None:

        # check whether metadata generated
        self._check_metadata()

        # register the metadata
        if fulltype in self.metadata.keys():
            self.metadata[fulltype] += [name]
        else:
            self.metadata[fulltype] = [name]

        # register the details
        if details is not None:
            self.details[name] = details
        else:
            self.details[name] = {}

    def _get_from_others(self, data: Any) -> Dict[Tuple, List] or dict:

        # convert to dataframe
        try:
            data = pd.DataFrame(data)
        except:
            raise TypeError(
                "The input data type {} cannot be converted to a dataframe.".format(
                    type(data)
                )
            )
        data.columns = ["col_" + str(i) for i in range(data.shape[1])]

        return self.get_from_df(data)

    @staticmethod
    def _check_fulltype(metadata: dict, ref: list) -> None:

        for key in metadata.keys():
            if key not in ref:
                raise TypeError(
                    "The fulltype {} is not supported. All supported fulltypes are {}".format(
                        key, ref
                    )
                )

    @staticmethod
    def _get_fulltype(data: pd.Series) -> str:

        _column = data.name  # get column name
        _dtype = data.dtypes  # get column dtype

        # Integer
        if "int" in str(_dtype):
            type = "Int"
            # check is Int_Numerical or Int_Categorical
            subtype = meta_map_int(data)
        # Float
        elif "float" in str(_dtype):

            # in case int registed as float
            # if nan values, fillna first, since those values will not impact our judgement
            if max(np.abs(data.fillna(0) - data.fillna(0).astype(int))) < 1e-6:
                warnings.warn(
                    "The column {} is registered as float, but it is actually int. Convert it to int".format(
                        _column
                    )
                )
                data = data.astype(int)
                type = "Int"
                # check is Int_Numerical or Int_Categorical
                subtype = meta_map_int(data)

            type = "Float"
            subtype = ""
        # Datetime
        elif "datetime" in str(_dtype):
            type = "Datetime"
            subtype = ""
        # Object
        elif _dtype == "object":

            # check if is path
            if os.path.isfile(data[0]):
                type = "Path"
                subtype = ""

            type = "Object"
            subtype = meta_map_object(data)
        else:
            raise TypeError(
                "The data type {} of column {} is not supported.".format(
                    _dtype, _column
                )
            )

        return type, subtype

    def get_from_df(self, data: pd.DataFrame) -> Dict[Tuple, List] or dict:

        # initialize the metadata
        self.metadata = {}
        # initialize the details
        self.details = {}

        for _column in data.columns:

            # get type and subtype
            type, subtype = self._get_fulltype(data[_column])

            # get the details
            details = get_details(type, subtype).get(data[_column])
            # register the metadata and details
            self.register(_column, (type, subtype), details)

            # check unique fulltype
            self._check_fulltype(self.metadata, ref=self._unique_fulltype)

        return self.metadata
