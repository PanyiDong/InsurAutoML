"""
File Name: space.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/nn/utils/hpSpace/space.py
File Created: Friday, 25th November 2022 11:43:31 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 5th December 2022 4:56:51 pm
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

search_space = {
    "num_epochs": {"_type": "randint", "_value": [10, 100]},
    "batch_size": {"_type": "choice", "_value": [8, 16, 32, 64, 128, 256, 512]},
    "optimizer": {"_type": "choice", "_value": ["Adagrad", "Adam", "ASGD", "RMSprop", "SGD"]},
    "lr": {"_type": "loguniform", "_value": [1e-8, 0.1]},
    "weight_decay ": {"_type": "loguniform", "_value": [1e-7, 0.1]},
    "lr_scheduler": {"_type": "choice", "_value": ["None", "StepLR", "ConstantLR", "LinearLR", "ExponentialLR", "PolynomialLR", "ReduceLROnPlateau"]},
    "step_size": {"_type": "qloguniform", "_value": [2, 100, 1]},
    "gamma": {"_type": "uniform", "_value": [0.1, 0.99]},
    "factor": {"_type": "uniform", "_value": [0.1, 0.99]},
    "total_iters": {"_type": "qloguniform", "_value": [2, 100, 1]},
    "power": {"_type": "qrandint", "_value": [1, 5]},
}
