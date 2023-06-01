"""
File Name: scaling_hyperparameter.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.5
Relative Path: /InsurAutoML/hyperparameters/ray/scaling_hyperparameter.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Thursday, 1st June 2023 9:39:13 am
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

NOPROCESSING = {"scaling": "no_processing"}
STANDARDIZE = {"scaling": "Standardize"}
NORMALIZE = {"scaling": "Normalize"}
ROBUSTSCALE = {"scaling": "RobustScale"}
POWERTRANSFORMER = {
    "scaling": "PowerTransformer",
    "method": tune.choice(["yeo-johnson"]),
}
QUANTILETRANSFORMER = {"scaling": "QuantileTransformer"}
MINMAXSCALE = {"scaling": "MinMaxScale"}
WINSORIZATION = {"scaling": "Winsorization"}
FEATUREMANIPULATION = {"scaling": "Feature_Manipulation"}
FEATURETRUNCATION = {"scaling": "Feature_Truncation"}

# scaling
# initialize the hyperparameter dictionary
scaling_hyperparameter = [
    NOPROCESSING,
    STANDARDIZE,
    NORMALIZE,
    ROBUSTSCALE,
    POWERTRANSFORMER,
    QUANTILETRANSFORMER,
    MINMAXSCALE,
    WINSORIZATION,
    FEATUREMANIPULATION,
    FEATURETRUNCATION,
]

# deprecated, add custom hyperparameter construction by search algorithm in AutoTabularBase class
# scaling_hyperparameter = [
#     format_hyper_dict(dict, order + 1, ref = "scaling")
#     for order, dict in enumerate(scaling_hyperparameter)
# ]

if __name__ == "__main__":
    pass
