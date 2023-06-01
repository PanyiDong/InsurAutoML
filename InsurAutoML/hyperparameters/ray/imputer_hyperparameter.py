"""
File Name: imputer_hyperparameter.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.5
Relative Path: /InsurAutoML/hyperparameters/ray/imputer_hyperparameter.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Thursday, 1st June 2023 9:38:05 am
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
from ...utils.base import format_hyper_dict

SIMPLEIMPUTER = {
    "imputer": "SimpleImputer",
    "method": tune.choice(["mean", "zero", "median", "most frequent"]),
}
DUMMYIMPUTER = {"imputer": "DummyImputer"}
JOINTIMPUTER = {"imputer": "JointImputer"}
EXPECTATIONMAXIMIZATION = {
    "imputer": "ExpectationMaximization",
    "iterations": tune.qrandint(10, 100, 1),
    "threshold": tune.uniform(1e-5, 1),
}
KNNIMPUTER = {
    "imputer": "KNNImputer",
    "n_neighbors": tune.qrandint(1, 15, 1),
    "fold": tune.qrandint(5, 15, 1),
}
MISSFORESTIMPUTER = {
    "imputer": "MissForestImputer",
    "threshold": tune.loguniform(1e-5, 10),
    "method": tune.choice(["mean", "zero", "median", "most frequent"]),
}
MICE = {
    "imputer": "MICE",
    "method": tune.choice(["mean", "zero", "median", "most frequent"]),
    "cycle": tune.qrandint(5, 20, 1),
}
GAIN = {
    "imputer": "GAIN",
}
AAIKNN = {
    "imputer": "AAI_kNN",
}
KMI = {
    "imputer": "KMI",
}
CMI = {
    "imputer": "CMI",
}
KPROTOTYPENN = {
    "imputer": "k_Prototype_NN",
}

# imputer
imputer_hyperparameter = [
    SIMPLEIMPUTER,
    DUMMYIMPUTER,
    JOINTIMPUTER,
    EXPECTATIONMAXIMIZATION,
    KNNIMPUTER,
    MISSFORESTIMPUTER,
    MICE,
    GAIN,
    # AAIKNN, # this methods are not efficient enough
    # KMI,
    # CMI,
    # KPROTOTYPENN,
]

# deprecated, add custom hyperparameter construction by search algorithm in AutoTabularBase class
# imputer_hyperparameter = [
#     format_hyper_dict(dict, order + 1, ref = "imputer")
#     for order, dict in enumerate(imputer_hyperparameter)
# ]

if __name__ == "__main__":
    pass
