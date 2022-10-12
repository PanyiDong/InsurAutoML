"""
File: _buildmodel.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Latest Version: 0.2.1
Relative Path: /My_AutoML/_nn/_nni/_nas/_buildmodel.py
File Created: Tuesday, 11th October 2022 4:02:00 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Tuesday, 11th October 2022 4:16:24 pm
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

from ._utils import ACTIVATIONS


def build_model(config, inputSize, outputSize, architecture="MLP"):

    if architecture == "MLP":
        return build_mlp(config, inputSize, outputSize)


def build_mlp(config, inputSize, outputSize):

    layers = []
    # input layer
    layers.append(nn.Linear(inputSize, config["hiddenSize_at_layer_0"]))

    # hidden layers
    for layer_id in range(config["num_hidden_layers"]):
        layers.append(
            nn.Linear(
                config[f"hiddenSize_at_layer_{layer_id}"],
                config[f"hiddenSize_at_layer_{layer_id + 1}"],
            )
        )
        layers.append(ACTIVATIONS[config[f"activation_at_layer_{layer_id}"]])
        layers.append(nn.Dropout(config[f"p_dropout_at_layer_{layer_id}"]))

    # output layer
    layers.append(nn.LazyLinear(outputSize))
    layers.append(nn.Softmax())

    model = nn.Sequential(*layers)

    return model
