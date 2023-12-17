"""
File Name: base.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.5
Relative Path: /InsurAutoML/feature_selection/base.py
File Created: Monday, 29th May 2023 2:25:52 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Thursday, 1st June 2023 11:12:26 pm
Modified By: Panyi Dong (panyid2@illinois.edu)

-----
MIT License

Copyright (c) 2023 - 2023, Panyi Dong

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
from typing import Union
import numpy as np
import pandas as pd


class BaseFeatureSelection:
    def __init__(self) -> None:
        pass

    def _check_feature_names(self, X: Union[pd.DataFrame, np.ndarray]) -> None:
        if not hasattr(self, "feature_names_in_"):
            if isinstance(X, pd.DataFrame):
                self.feature_names_in_ = X.columns.tolist()
            else:
                self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> BaseFeatureSelection:
        raise NotImplementedError

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        raise NotImplementedError

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray] = None,
    ) -> Union[pd.DataFrame, np.ndarray]:
        self.fit(X, y)
        return self.transform(X)
