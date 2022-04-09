"""
File: _imputer_hyperparameter.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Latest Version: 0.2.0
Relative Path: /My_AutoML/_hyperparameters/_hyperopt/_imputer_hyperparameter.py
File Created: Tuesday, 5th April 2022 11:02:55 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Friday, 8th April 2022 10:23:00 pm
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

import numpy as np
from hyperopt import hp
from hyperopt.pyll import scope

# imputer
imputer_hyperparameter = [
    {
        "imputer": "SimpleImputer",
        "method": hp.choice(
            "SimpleImputer_method", ["mean", "zero", "median", "most frequent"]
        ),
    },
    {"imputer": "DummyImputer"},
    {"imputer": "JointImputer"},
    {
        "imputer": "ExpectationMaximization",
        "iterations": hp.quniform("ExpectationMaximization_iterations", 10, 100, 1),
        "threshold": hp.uniform(
            "ExpectationMaximization_iterations_threshold", 1e-5, 1
        ),
    },
    {
        "imputer": "KNNImputer",
        "n_neighbors": scope.int(hp.quniform("KNNImputer_n_neighbors", 1, 15, 1)),
        "fold": scope.int(hp.quniform("KNNImputer_fold", 5, 15, 1)),
    },
    {"imputer": "MissForestImputer"},
    {"imputer": "MICE", "cycle": hp.quniform("MICE_cycle", 5, 20, 1)},
    {"imputer": "GAIN"},
    # {"imputer": "AAI_kNN"},
    # {"imputer": "KMI"},
    # {"imputer": "CMI"},
    # {"imputer": "k_Prototype_NN"},
]
