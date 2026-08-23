"""
File Name: utils.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: split
Latest Version: <<projectversion>>
Relative Path: /utils.py
File Created: Monday, 15th September 2025 3:14:01 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 15th September 2025 3:14:16 pm
Modified By: Panyi Dong (panyid2@illinois.edu)

-----
MIT License

Copyright (c) 2025 - 2025, Panyi Dong

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

import pandas as pd
import numpy as np


class HelmertEncoding:

    def __init__(self):
        self.unique_levels = {}
        self.helmert_matrices = {}
        self.fitted = False

    @staticmethod
    def contr_helmert(n: int):
        contr = np.zeros((n, n - 1))
        contr[1:][np.diag_indices(n - 1)] = np.arange(1, n)
        contr[np.triu_indices(n - 1)] = -1

        return contr

    def fit_col(self, data):
        levels = data.unique()
        self.unique_levels[data.name] = levels
        n_levels = len(levels)

        if n_levels < 2:
            raise ValueError("Need at least two levels for Helmert encoding.")

        # build contrast matrix
        self.helmert_matrices[data.name] = self.contr_helmert(n_levels)

    def transform_col(self, data):
        index = data.map(
            {lev: idx for idx, lev in enumerate(self.unique_levels[data.name])}
        ).to_numpy()
        # transform to helmert encoding
        data_ = self.helmert_matrices[data.name][index, :]

        return data_

    def fit(self, data):
        data = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
        for col in data.columns:
            if data[col].dtype == "object" or str(data[col].dtype).startswith(
                "category"
            ):
                self.fit_col(data[col])
        self.fitted = True

    def transform(self, data):
        if not self.fitted:
            raise ValueError("The encoder has not been fitted yet.")

        data = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
        transformed_cols = []
        for col in data.columns:
            if col in self.unique_levels.keys():
                transformed_cols.append(self.transform_col(data[col]))
            else:
                transformed_cols.append(data[[col]].to_numpy())

        return np.hstack(transformed_cols)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
