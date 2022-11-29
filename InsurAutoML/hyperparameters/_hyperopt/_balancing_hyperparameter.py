"""
File: _balancing_hyperparameter.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Latest Version: 0.2.0
Relative Path: /My_AutoML/_hyperparameters/_hyperopt/_balancing_hyperparameter.py
File Created: Tuesday, 5th April 2022 11:04:29 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Friday, 8th April 2022 10:22:35 pm
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

# balancing
# if the imbalance threshold small, TomekLink will take too long
balancing_hyperparameter = [
    {"balancing": "no_processing"},
    {
        "balancing": "SimpleRandomOverSampling",
        "imbalance_threshold": hp.uniform(
            "SimpleRandomOverSampling_imbalance_threshold", 0.8, 1
        ),
    },
    {
        "balancing": "SimpleRandomUnderSampling",
        "imbalance_threshold": hp.uniform(
            "SimpleRandomUnderSampling_imbalance_threshold", 0.8, 1
        ),
    },
    {
        "balancing": "TomekLink",
        "imbalance_threshold": hp.uniform("TomekLink_imbalance_threshold", 0.8, 1),
    },
    {
        "balancing": "EditedNearestNeighbor",
        "imbalance_threshold": hp.uniform(
            "EditedNearestNeighbor_imbalance_threshold", 0.8, 1
        ),
        "k": scope.int(hp.quniform("EditedNearestNeighbor_k", 1, 7, 1)),
    },
    {
        "balancing": "CondensedNearestNeighbor",
        "imbalance_threshold": hp.uniform(
            "CondensedNearestNeighbor_imbalance_threshold", 0.8, 1
        ),
    },
    {
        "balancing": "OneSidedSelection",
        "imbalance_threshold": hp.uniform(
            "OneSidedSelection_imbalance_threshold", 0.8, 1
        ),
    },
    {
        "balancing": "CNN_TomekLink",
        "imbalance_threshold": hp.uniform("CNN_TomekLink_imbalance_threshold", 0.8, 1),
    },
    {
        "balancing": "Smote",
        "imbalance_threshold": hp.uniform("Smote_imbalance_threshold", 0.8, 1),
        "k": scope.int(hp.quniform("Smote_k", 1, 10, 1)),
    },
    {
        "balancing": "Smote_TomekLink",
        "imbalance_threshold": hp.uniform(
            "Smote_TomekLink_imbalance_threshold", 0.8, 1
        ),
        "k": scope.int(hp.quniform("Smote_TomekLink_k", 1, 10, 1)),
    },
    {
        "balancing": "Smote_ENN",
        "imbalance_threshold": hp.uniform("Smote_ENN_imbalance_threshold", 0.8, 1),
        "k": scope.int(hp.quniform("Smote_ENN_k", 1, 10, 1)),
    },
]
