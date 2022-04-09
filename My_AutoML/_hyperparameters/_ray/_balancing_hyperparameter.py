"""
File: _balancing_hyperparameter.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Latest Version: 0.2.0
Relative Path: /My_AutoML/_hyperparameters/_ray/_balancing_hyperparameter.py
File Created: Wednesday, 6th April 2022 10:06:01 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Friday, 8th April 2022 10:23:34 pm
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

from ray import tune

# balancing
# if the imbalance threshold small, TomekLink will take too long
balancing_hyperparameter = [
    {"balancing_1": "no_processing"},
    {
        "balancing_2": "SimpleRandomOverSampling",
        "SimpleRandomOverSampling_imbalance_threshold": tune.uniform(0.8, 1),
    },
    {
        "balancing_3": "SimpleRandomUnderSampling",
        "SimpleRandomUnderSampling_imbalance_threshold": tune.uniform(0.8, 1),
    },
    {
        "balancing_4": "TomekLink",
        "TomekLink_imbalance_threshold": tune.uniform(0.8, 1),
    },
    {
        "balancing_5": "EditedNearestNeighbor",
        "EditedNearestNeighbor_imbalance_threshold": tune.uniform(0.8, 1),
        "EditedNearestNeighbor_k": tune.qrandint(1, 7, 1),
    },
    {
        "balancing_6": "CondensedNearestNeighbor",
        "CondensedNearestNeighbor_imbalance_threshold": tune.uniform(0.8, 1),
    },
    {
        "balancing_7": "OneSidedSelection",
        "OneSidedSelection_imbalance_threshold": tune.uniform(0.8, 1),
    },
    {
        "balancing_8": "CNN_TomekLink",
        "CNN_TomekLink_imbalance_threshold": tune.uniform(0.8, 1),
    },
    {
        "balancing_9": "Smote",
        "Smote_imbalance_threshold": tune.uniform(0.8, 1),
        "Smote_k": tune.qrandint(1, 10, 1),
    },
    {
        "balancing_10": "Smote_TomekLink",
        "Smote_TomekLink_imbalance_threshold": tune.uniform(0.8, 1),
        "Smote_TomekLink_k": tune.qrandint(1, 10, 1),
    },
    {
        "balancing_11": "Smote_ENN",
        "Smote_ENN_imbalance_threshold": tune.uniform(0.8, 1),
        "Smote_ENN_k": tune.qrandint(1, 10, 1),
    },
]
