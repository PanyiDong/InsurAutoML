"""
File: _balancing_hyperparameter.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_hyperparameters/_ray/_balancing_hyperparameter.py
File: _balancing_hyperparameter.py
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 7th November 2022 3:49:32 pm
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
from InsurAutoML._utils._base import format_hyper_dict

NOPROCESSING = {
    "balancing": "no_processing",
}
SIMPLERANDOMOVERSAMPLING = {
    "balancing": "SimpleRandomOverSampling",
    "imbalance_threshold": tune.uniform(0.8, 1),
}
SIMPLERANDOMUNDERSAMPLING = {
    "balancing": "SimpleRandomUnderSampling",
    "imbalance_threshold": tune.uniform(0.8, 1),
}
TOMEKLINK = {
    "balancing": "TomekLink",
    "imbalance_threshold": tune.uniform(0.8, 1),
}
EDITEDNEARESTNEIGHBOR = {
    "balancing": "EditedNearestNeighbor",
    "imbalance_threshold": tune.uniform(0.8, 1),
    "k": tune.qrandint(1, 7, 1),
}
CONDENSEDNEARESTNEIGHBOR = {
    "balancing": "CondensedNearestNeighbor",
    "imbalance_threshold": tune.uniform(0.8, 1),
}
ONESIDEDSELECTION = {
    "balancing": "OneSidedSelection",
    "imbalance_threshold": tune.uniform(0.8, 1),
}
CNNTOMEKLINK = {
    "balancing": "CNN_TomekLink",
    "imbalance_threshold": tune.uniform(0.8, 1),
}
SMOTE = {
    "balancing": "Smote",
    "imbalance_threshold": tune.uniform(0.8, 1),
    "k": tune.qrandint(1, 10, 1),
}
SMOTETOMEKLINK = {
    "balancing": "Smote_TomekLink",
    "imbalance_threshold": tune.uniform(0.8, 1),
    "k": tune.qrandint(1, 10, 1),
}
SMOTEENN = {
    "balancing": "Smote_ENN",
    "imbalance_threshold": tune.uniform(0.8, 1),
    "k": tune.qrandint(1, 10, 1),
}

# balancing
# initialize the hyperparameter dictionary
balancing_hyperparameter = [
    NOPROCESSING,
    SIMPLERANDOMOVERSAMPLING,
    SIMPLERANDOMUNDERSAMPLING,
    TOMEKLINK,   # if the imbalance threshold small, TomekLink will take too long
    EDITEDNEARESTNEIGHBOR,
    CONDENSEDNEARESTNEIGHBOR,
    ONESIDEDSELECTION,
    CNNTOMEKLINK,
    SMOTE,
    SMOTETOMEKLINK,
    SMOTEENN,
]

balancing_hyperparameter = [
    format_hyper_dict(dict, order + 1, ref = "balancing") 
    for order, dict in enumerate(balancing_hyperparameter)
]

if __name__ == "__main__":
    pass
