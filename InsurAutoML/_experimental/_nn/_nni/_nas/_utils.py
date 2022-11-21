"""
File Name: _utils.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_experimental/_nn/_nni/_nas/_utils.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Sunday, 20th November 2022 11:23:52 pm
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


from collections import OrderedDict
from typing import Callable
import torch
import torch.nn as nn
import torch.optim as optim

import nni.retiarii.strategy as strategy
import nni.retiarii.nn.pytorch as nninn


##########################################################################
# Constant

ACTIVATIONS = OrderedDict(
    [
        ("ReLU", nninn.ReLU()),
        ("Sigmoid", nninn.Sigmoid()),
        ("Tanh", nninn.Tanh()),
    ]
)

RNN_TYPES = OrderedDict(
    [
        ("RNN", nninn.RNN),
        ("LSTM", nninn.LSTM),
        ("GRU", nninn.GRU),
    ]
)

##########################################################################
# utils


def how_to_init(m, how="uniform"):

    if how == "uniform":
        nninn.init.uniform_(m.weight)
    elif how == "normal":
        nninn.init.normal_(m.weight)
    elif how == "one":
        nninn.init.ones_(m.weight)
    elif how == "zero":
        nninn.init.zeros_(m.weight)
    elif how == "eye":
        nninn.init.eye_(m.weight)
    elif how == "xavier_uniform":
        nninn.init.xavier_uniform_(m.weight)
    elif how == "xavier_normal":
        nninn.init.xavier_normal_(m.weight)
    elif how == "kaiming_uniform":
        nninn.init.kaiming_uniform_(m.weight)
    elif how == "kaiming_normal":
        nninn.init.kaiming_normal_(m.weight)
    elif how == "trunc_normal":
        nninn.init.trunc_normal_(m.weight)
    elif how == "orthogonal":
        nninn.init.orthogonal_(m.weight)
    else:
        raise ValueError("Unrecognized init method {}".format(how))


def get_search_space(search_space):

    if search_space == "MLP":
        from ._baseSpace import MLPBaseSpace

        return MLPBaseSpace
    elif isinstance(search_space, Callable):
        return search_space


def get_optimizer(optimizer):

    # get optimizer
    if optimizer == "SGD":
        optimizer = optim.SGD
    elif optimizer == "Adam":
        optimizer = optim.Adam
    elif optimizer == "Adagrad":
        optimizer = optim.Adagrad
    elif optimizer == "LBFGS":
        optimizer = optim.LBFGS
    elif optimizer == "RMSprop":
        optimizer = optim.RMSprop
    elif optimizer == "Rprop":
        optimizer = optim.Rprop
    elif isinstance(optimizer, Callable):
        pass
    else:
        raise ValueError(
            "optimizer must be one of SGD, Adam, Adagrad, LBFGS, RMSprop, Rprop, get {}".format(
                optimizer
            )
        )
    return optimizer


def get_scheduler(lr_scheduler):
    # get lr_scheduler
    if lr_scheduler is None or lr_scheduler == "None":
        lr_scheduler = None
    elif lr_scheduler == "Constant":
        lr_scheduler = optim.lr_scheduler.ConstantLR
    elif lr_scheduler == "Step":
        lr_scheduler = optim.lr_scheduler.StepLR
    elif lr_scheduler == "Linear":
        lr_scheduler = optim.lr_scheduler.LinearLR
    elif lr_scheduler == "Exponential":
        lr_scheduler = optim.lr_scheduler.ExponentialLR
    elif lr_scheduler == "CosineAnnealing":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR
    elif isinstance(lr_scheduler, Callable):
        pass
    else:
        raise ValueError(
            "lr_scheduler must be one of None, Constant, Step, Linear, Exponential, CosineAnnealing, get {}".format(
                lr_scheduler
            )
        )

    return lr_scheduler


def get_criterion(criterion):

    if criterion in ["MAE", "L1", "L1Loss"]:
        return nn.L1Loss
    elif criterion in ["MSE", "MSELoss"]:
        return nn.MSELoss
    elif criterion in ["CrossEntropy", "CrossEntropyLoss"]:
        return nn.CrossEntropyLoss
    elif criterion in ["NLLLoss", "NLL"]:
        return nn.NLLLoss
    elif criterion in ["BCE", "BCELoss"]:
        return nn.BCELoss
    elif criterion in ["BCEWithLogits", "BCEWithLogitsLoss"]:
        return nn.BCEWithLogitsLoss
    elif criterion in ["Huber", "HuberLoss"]:
        return nn.HuberLoss
    elif criterion in ["SmoothL1", "SmoothL1Loss"]:
        return nn.SmoothL1Loss
    elif isinstance(criterion, Callable):
        return criterion
    else:
        raise ValueError("Unrecognized criterion {}".format(criterion))


def get_strategy(strategyStr):

    if strategyStr == "Random":
        return strategy.Random(dedup=True)
    elif strategyStr == "Grid":
        return strategy.GridSearch(shuffle=True)
    elif strategyStr == "Evolution":
        return strategy.RegularizedEvolution()
    elif strategyStr == "TPE":
        return strategy.TPE()
    elif strategyStr == "RL":
        return strategy.PolicyBasedRL()
    elif isinstance(strategyStr, Callable):
        return strategyStr()
    else:
        raise ValueError("Unrecognized strategy {}".format(strategyStr))
