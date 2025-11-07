"""
File Name: split.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.6
Relative Path: /InsurAutoML/ext/suppsplit/suppsplit/split.py
File Created: Thursday, 11th September 2025 3:07:38 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Thursday, 6th November 2025 8:02:30 pm
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

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .utils import HelmertEncoding
from ...suppsplit.suppsplit_cpp import sp_cpp, subsample  # the compiled C++ extension


def jitter(x, factor: float = 1, amount: float = None):
    sort_unique = np.sort(np.unique(x))
    smallest_diff = sort_unique[1] - sort_unique[0]
    if amount is None:
        amount = smallest_diff / 5
    return x + np.random.uniform(-1, 1, size=x.shape) * amount * factor


def compute_sp(n, p, dist_samp, num_subsamp, iter_max=500, tol=1e-10, n_threads=None):
    rnd_flg = num_subsamp < dist_samp.shape[0]
    n_threads = (
        max(n_threads, os.cpu_count()) if n_threads is not None else os.cpu_count()
    )
    wts = np.ones(dist_samp.shape[0])
    n0 = float(n * p)
    bd = np.column_stack([dist_samp.min(axis=0), dist_samp.max(axis=0)])

    # handle duplicate
    if pd.DataFrame(dist_samp).duplicated().any():
        dist_samp = jitter(dist_samp)
        # clip dist_samp to avoid numerical issues
        for j in range(p):
            dist_samp[:, j] = np.clip(dist_samp[:, j], bd[j, 0], bd[j, 1])

    # initialize design
    ini = dist_samp[np.random.choice(dist_samp.shape[0], size=n, replace=False), :]
    ini = jitter(ini)
    for j in range(p):
        ini[:, j] = np.clip(ini[:, j], bd[j, 0], bd[j, 1])

    if p == 1:
        ini = ini.reshape(-1, 1)

    return sp_cpp(
        ini,
        dist_samp,
        True,
        bd,
        num_subsamp,
        iter_max,
        tol,
        n_threads,
        n0,
        wts,
        rnd_flg,
    )


def data_format(data):
    data = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
    if data.isnull().values.any():
        raise ValueError("Dataset contains missing values.")

    cols = []
    for col in data.columns:
        if pd.api.types.is_categorical_dtype(data[col]) or data[col].dtype == "object":
            enc = HelmertEncoding()
            trans = enc.fit_transform(data[[col]])
            cols.append(trans)
        elif np.issubdtype(data[col].dtype, np.number):
            if (data[col] == data[col].iloc[0]).all():
                continue
            cols.append(data[[col]].to_numpy())
        else:
            raise ValueError("Unsupported column type.")

    D = np.hstack(cols)
    sc = StandardScaler()
    sc.fit(D)
    sc.scale_ = np.std(D, axis=0, ddof=1)  # R scale function use ddof=1
    return sc.transform(D)


def SPlit(
    data,
    split_ratio=0.2,
    kappa=None,
    max_iterations=500,
    tolerance=1e-10,
    n_threads=None,
):
    if not (0 < split_ratio < 1):
        raise ValueError("split_ratio must be in (0, 1).")

    data_ = data_format(data)
    # data_ = pd.read_csv("data.csv").to_numpy()  # for debugging
    n = round(min(split_ratio, 1 - split_ratio) * data_.shape[0])

    # set kappa
    if kappa is None:
        kappa = data_.shape[0]
    else:
        assert kappa > 0, "kappa must be positive."
        kappa = min(data_.shape[0], int(np.ceil(kappa * n)))

    sp_ = compute_sp(
        n,
        data_.shape[1],
        data_,
        kappa,
        iter_max=max_iterations,
        tol=tolerance,
        n_threads=n_threads,
    )

    return subsample(data_, sp_) - 1
