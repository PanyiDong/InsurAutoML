"""
File Name: _wholeSpace.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_experimental/_nn/_nni/_nas/_baseSpace/_wholeSpace.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Tuesday, 22nd November 2022 4:47:02 pm
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

import nni
import nni.retiarii.nn.pytorch as nninn
from nni.retiarii import model_wrapper, fixed_arch

from .._utils import ACTIVATIONS, RNN_TYPES, how_to_init
# from .._buildModel import build_mlp

##########################################################################
# MLP Space


# @model_wrapper
class MLPBaseSpace(nninn.Module):
    def __init__(
        self,
        inputSize,
        outputSize,
        prefix="",
    ):
        super().__init__()

        # initialize the structure of the neural network
        self.net = []

        # add input layer
        self.net.append(
            nninn.Linear(
                inputSize,
                nninn.ValueChoice(
                    [32, 64, 128, 256, 512], label=f"{prefix}hiddenSize_at_layer_0"
                ),
            )
        )

        # add hidden layers
        # each hidden layer has a linear layer, a activation layer and possibly
        # a dropout layer
        self.net.append(
            nninn.Repeat(
                lambda index: nninn.Sequential(
                    nninn.Linear(nninn.ValueChoice([32, 64, 128, 256, 512], label=f"{prefix}hiddenSize_at_layer_{index}",), nninn.ValueChoice(
                        [32, 64, 128, 256, 512], label=f"{prefix}hiddenSize_at_layer_{index + 1}",),),
                    nninn.LayerChoice(
                        ACTIVATIONS, label=f"{prefix}activation_at_layer_{index}"),
                    nninn.Dropout(nninn.ValueChoice([0.0, 0.2, 0.5, 0.9], label=f"{prefix}p_dropout_at_layer_{index}")),),
                nninn.ValueChoice([1, 2, 3, 4, 5, 6], label=f"{prefix}num_hidden_layers"), label=f"{prefix}hidden_layers",)
        )

        # add output layer
        # use lazy linear to automatically initialize the input size
        self.net.append(nninn.LazyLinear(outputSize))

        # add softmax layer
        self.net.append(nninn.Softmax())

        self.net = nninn.Sequential(*self.net)

    def forward(self, X):

        if isinstance(X, list):
            if len(X) > 1:
                raise ValueError(
                    "Expect one tensor, get {} tensors".format(len(X)))
            X = X[0]

        return self.net(X)

    def init_weight(self, how="xavier_normal"):

        for m in self.modules():
            if isinstance(m, nninn.Linear):
                how_to_init(m, how)

    @staticmethod
    def build_model(config_path, inputSize, outputSize):

        with fixed_arch(config_path):
            return MLPBaseSpace(inputSize, outputSize)


# @model_wrapper
class MLPLintSpace(nninn.Module):
    def __init__(
        self,
        inputSize,
        outputSize,
    ):
        super().__init__()

        # initialize the structure of the neural network
        self.net = []

        # add input layer
        self.net.append(nninn.Linear(inputSize, nninn.ValueChoice(
            [4, 8, 16, 32, 64], label="hiddenSize_at_layer_0"), ))

        # add hidden layers
        # each hidden layer has a linear layer, a activation layer and possibly
        # a dropout layer
        self.net.append(
            nninn.Repeat(
                lambda index: nninn.Sequential(
                    nninn.Linear(nninn.ValueChoice([4, 8, 16, 32, 64], label=f"hiddenSize_at_layer_{index}",),
                                 nninn.ValueChoice([4, 8, 16, 32, 64], label=f"hiddenSize_at_layer_{index + 1}",),),
                    nninn.LayerChoice(
                        ACTIVATIONS, label=f"activation_at_layer_{index}"),
                    nninn.Dropout(nninn.ValueChoice([0.0, 0.2, 0.5, 0.9], label=f"p_dropout_at_layer_{index}")),),
                nninn.ValueChoice([1, 2, 3], label="num_hidden_layers"), label="hidden_layers",)
        )

        # add output layer
        # use lazy linear to automatically initialize the input size
        self.net.append(nninn.LazyLinear(outputSize))

        # add softmax layer
        self.net.append(nninn.Softmax())

        self.net = nninn.Sequential(*self.net)

    def forward(self, X):

        if isinstance(X, list):
            if len(X) > 1:
                raise ValueError(
                    "Expect one tensor, get {} tensors".format(len(X)))
            X = X[0]

        return self.net(X)

    def init_weight(self, how="xavier_normal"):

        for m in self.modules():
            if isinstance(m, nninn.Linear):
                how_to_init(m, how)

    @staticmethod
    def build_model(config_path, inputSize, outputSize):

        with fixed_arch(config_path):
            return MLPLintSpace(inputSize, outputSize)


##########################################################################
# RNN Space


# @nni.trace
# @model_wrapper
class RNNBaseSpace(nninn.Module):
    def __init__(
        self,
        inputSize,
        outputSize,
    ):
        super().__init__()

        # initialize the structure of the neural network
        self.prenet = []
        self.RNNnet = []
        self.postnet = []

        # embedding layer
        self.prenet.append(nninn.Embedding(inputSize, nninn.ValueChoice(
            [128, 256, 512, 1024], label="embedding_size"), ))

        # RNN parameters
        num_RNN_layers = nninn.ValueChoice(
            [
                1,
                2,
                3,
            ],
            label="num_RNN_layers",
        )
        bidirectional = nninn.ValueChoice([True, False], label="bidirectional")
        p_dropout = nninn.ValueChoice([0.0, 0.2, 0.5, 0.9], label="p_dropout")
        # RNN layer
        self.RNNnet.append(
            nninn.LayerChoice(
                [
                    RNNstructure(
                        input_size=nninn.ValueChoice(
                            [128, 256, 512, 1024], label="embedding_size"
                        ),
                        hidden_size=nninn.ValueChoice(
                            [64, 128, 256, 512], label="hidden_size"
                        ),
                        num_layers=num_RNN_layers,
                        dropout=p_dropout,
                        bidirectional=bidirectional,
                    )
                    for RNNstructure in RNN_TYPES.values()
                ]
            )
        )

        # linear layer
        self.postnet.append(
            nninn.Repeat(
                lambda index: nninn.Sequential(
                    nninn.LazyLinear(
                        nninn.ValueChoice(
                            [32, 64, 128, 256, 512],
                            label=f"size_at_dense_layer_{index}",
                        )
                    ),
                    nninn.LayerChoice(
                        ACTIVATIONS, label=f"activation_at_dense_layer_{index}"
                    ),
                    nninn.Dropout(
                        nninn.ValueChoice(
                            [0.0, 0.2, 0.5, 0.9],
                            label=f"p_dropout_at_dense_layer_{index}",
                        )
                    ),
                ),
                nninn.ValueChoice([0, 1, 2], label="num_dense_layers"),
                label="dense_layers",
            )
        )

        # output layer
        self.postnet.append(nninn.LazyLinear(outputSize))
        # softmax layer
        self.postnet.append(nninn.Softmax())

        self.prenet = nninn.Sequential(*self.prenet)
        self.RNNnet = nninn.Sequential(*self.RNNnet)
        self.postnet = nninn.Sequential(*self.postnet)

    def forward(self, input, hidden):

        # embedding layer
        output = self.prenet(input)
        # RNN layer
        output, hidden = self.RNNnet(output, hidden)
        # post layer
        output = self.postnet(output)

        return output

    def init_weight(self, how="xavier_normal"):

        for m in self.modules():
            if (
                isinstance(m, nninn.Linear)
                or isinstance(m, nninn.Embedding)
                or isinstance(m, nninn.LazyLinear)
                or isinstance(m, nninn.RNN)
                or isinstance(m, nninn.LSTM)
                or isinstance(m, nninn.GRU)
            ):
                how_to_init(m, how)

    @staticmethod
    def build_model(config_path, inputSize, outputSize):

        with fixed_arch(config_path):
            return RNNBaseSpace(inputSize, outputSize)
