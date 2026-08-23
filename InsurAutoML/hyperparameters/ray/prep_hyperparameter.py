"""
File Name: prep_hyperparameter.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Actuarial and Risk Management Sciences, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.6
Relative Path: /InsurAutoML/hyperparameters/ray/prep_hyperparameter.py
File Created: Wednesday, 3rd December 2025 7:04:54 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 15th December 2025 11:23:50 am
Modified By: Panyi Dong (panyid2@illinois.edu)

-----
MIT License

Copyright (c) 2025, Panyi Dong

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

CompletePrepPipeline = {
    "complete_prep": "CompletePrepPipeline",
    "missing_threshold": tune.uniform(0.5, 0.9),
    "cc_threshold": tune.uniform(0.01, 0.99),
}

MissingPrepPipeline = {
    "missing_prep": "MissingPrepPipeline",
    "missing_threshold": tune.uniform(0.5, 0.9),
    "twin_r": tune.qrandint(5, 100, 1),
    "imputation_max_iter": tune.qrandint(1, 20, 1),
    "imputation_n_estimators": tune.qrandint(10, 100, 1),
    "cc_threshold": tune.uniform(0.01, 0.99),
}

complete_prep_hyperparameter = [
    CompletePrepPipeline,
]

missing_prep_hyperparameter = [
    MissingPrepPipeline,
]

if __name__ == "__main__":
    pass
