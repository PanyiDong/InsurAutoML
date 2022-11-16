"""
File Name: __init__.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_imputation/__init__.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 14th November 2022 8:07:20 pm
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

from ._base import SimpleImputer, DummyImputer, JointImputer
from ._multiple import ExpectationMaximization, KNNImputer, MissForestImputer, MICE

# from ._clustering import AAI_kNN, KMI, CMI, k_Prototype_NN


imputers = {
    "SimpleImputer": SimpleImputer,
    #    'DummyImputer' : DummyImputer,
    "JointImputer": JointImputer,
    "ExpectationMaximization": ExpectationMaximization,
    "KNNImputer": KNNImputer,
    "MissForestImputer": MissForestImputer,
    "MICE": MICE,
    # "AAI_kNN": AAI_kNN, # extremely slow (all below)
    # "KMI": KMI, # not implemented
    # "CMI": CMI,
    # "k_Prototype_NN": k_Prototype_NN,
}
# Markov Chain Monte Carlo (MCMC)

import importlib

# check tensorflow/pytorch installed only import nn-based methods
# when tensorflow/pytorch is installed
tensorflow_spec = importlib.util.find_spec("tensorflow")
torch_spec = importlib.util.find_spec("torch")
if tensorflow_spec is not None or torch_spec is not None:
    from ._nn import GAIN

    imputers["GAIN"] = GAIN
