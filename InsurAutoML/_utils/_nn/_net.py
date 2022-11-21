"""
File Name: _net.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_utils/_nn/_net.py
File Created: Saturday, 19th November 2022 10:03:31 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 21st November 2022 2:13:24 pm
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

import torch
import torch.nn as nn
import nni.retiarii.nn.pytorch as nninn
from nni.retiarii import model_wrapper

from ..._experimental._nn._nni._nas._baseSpace import MLPBaseSpace, MLPHead

"""
List of methods:
                    Txt           Cat           Con
FusToken:      tokenize      encoding      encoding
FusEmbed:     embedding     embedding     embedding
FusModel:         model         model         model

FusToken: Txt/Cat/Con are only tokenized to numerics and fuse those heads together for whole model training.
FusEmbed: Txt/Cat/Con are further embedded with a few layers of NNs and fuse those embeddings together for later half of model training.
FusModel: Txt/Cat/Con are almost trained separately and just fused together for unified prediction.
"""


@model_wrapper
class FusTokenNet(MLPBaseSpace):

    def __init__(
        self,
        inputSize,
        outputSize,
    ):
        super(FusTokenNet, self).__init__(inputSize, outputSize)


class FusEmbedNet:

    def __init__(
        self,
        inputSize,
        outputSize,
    ):
        super(FusEmbedNet, self).__init__(inputSize, outputSize)
