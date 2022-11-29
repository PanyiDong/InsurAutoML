"""
File Name: head.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/nn/utils/baseSpace/head.py
File Created: Sunday, 13th November 2022 5:15:00 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Tuesday, 29th November 2022 3:55:50 pm
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
import nni
import nni.retiarii.nn.pytorch as nninn
from nni.retiarii import model_wrapper, fixed_arch

from ..args import ACTIVATIONS, RNN_TYPES, how_to_init

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

"""
List of classes need for multimodal neural architecture search.
1. entire network space (whole vs lint): use ones in _wholeSpace.py
2. heads (before fuse)
3. prediction (after fuse for FusModel)
"""


# @model_wrapper
class MLPHead(nninn.Module):
    def __init__(
        self,
        inputSize,
        outputSize,
        vocabSize: int = 0,
        prefix="",
    ):
        super().__init__()
        self.inputSize = inputSize
        self.outputSize = outputSize
        # initialize the structure of the neural network
        self.net = []

        # add input layer
        self.net.append(nninn.Linear(inputSize, nninn.ValueChoice(
            [4, 8, 16, 32, 64], label=f"{prefix}hiddenSize_at_layer_0"), ))

        # add hidden layers
        # each hidden layer has a linear layer, a activation layer and possibly
        # a dropout layer
        self.net.append(
            nninn.Repeat(
                lambda index: nninn.Sequential(
                    nninn.Linear(
                        nninn.ValueChoice(
                            [4, 8, 16, 32, 64],
                            label=f"{prefix}hiddenSize_at_layer_{index}",
                        ),
                        nninn.ValueChoice(
                            [4, 8, 16, 32, 64],
                            label=f"{prefix}hiddenSize_at_layer_{index + 1}",
                        ),
                    ),
                    nninn.LayerChoice(
                        ACTIVATIONS, label=f"{prefix}activation_at_layer_{index}"
                    ),
                    nninn.Dropout(
                        nninn.ValueChoice(
                            [0.0, 0.2, 0.5, 0.9], label=f"{prefix}p_dropout_at_layer_{index}"
                        )
                    ),
                ),
                nninn.ValueChoice(
                    [1, 2, 3], label=f"{prefix}num_hidden_layers"),
                label=f"{prefix}hidden_layers",
            )
        )

        # add output layer
        # use lazy linear to automatically initialize the input size
        self.net.append(nninn.LazyLinear(outputSize))

        # add softmax layer
        self.net.append(nninn.Softmax())

        self.net = nninn.Sequential(*self.net)

    def forward(self, X):

        # make sure input to embedding layer is float
        X = X.to(torch.float32)

        return self.net(X)

    def init_weight(self, how="xavier_normal"):

        for m in self.modules():
            if isinstance(m, nninn.Linear):
                how_to_init(m, how)

    @staticmethod
    def build_model(config_path, inputSize, outputSize):

        with fixed_arch(config_path):
            return MLPHead(inputSize, outputSize)


class RNNHead(nninn.Module):
    def __init__(
        self,
        inputSize,
        outputSize,
        vocabSize: int,
        prefix="",
    ):
        super().__init__()

        # initialize the structure of the neural network
        self.prenet = []
        self.RNNnet = []
        self.postnet = []

        # embedding layer
        self.prenet.append(nninn.Embedding(vocabSize, nninn.ValueChoice(
            [32, 64, 128, 256], label=f"{prefix}embedding_size"), ))

        # RNN parameters
        num_RNN_layers = nninn.ValueChoice(
            [1, 2],
            label=f"{prefix}num_RNN_layers",
        )
        self._bidirectional = nninn.ValueChoice(
            [True, False], label=f"{prefix}bidirectional")
        p_dropout = nninn.ValueChoice(
            [0.0, 0.2, 0.5, 0.9], label=f"{prefix}p_dropout")
        # RNN layer
        self.RNNnet = nninn.LayerChoice(
            [
                RNNstructure(
                    input_size=nninn.ValueChoice(
                        [32, 64, 128, 256], label=f"{prefix}embedding_size"
                    ),
                    hidden_size=nninn.ValueChoice(
                        [16, 32, 64, 128], label=f"{prefix}hidden_size"
                    ),
                    num_layers=num_RNN_layers,
                    dropout=p_dropout,
                    bidirectional=self._bidirectional,
                    batch_first=True,
                )
                for RNNstructure in RNN_TYPES.values()
            ], label=f"{prefix}RNN_type"
        )

        # linear layer
        self.postnet.append(
            nninn.Repeat(
                lambda index: nninn.Sequential(
                    nninn.LazyLinear(
                        nninn.ValueChoice(
                            [16, 32, 64, 128],
                            label=f"{prefix}size_at_dense_layer_{index}",
                        )
                    ),
                    nninn.LayerChoice(
                        ACTIVATIONS, label=f"{prefix}activation_at_dense_layer_{index}"
                    ),
                    nninn.Dropout(
                        nninn.ValueChoice(
                            [0.0, 0.2, 0.5, 0.9],
                            label=f"{prefix}p_dropout_at_dense_layer_{index}",
                        )
                    ),
                ),
                nninn.ValueChoice(
                    [0, 1, 2], label=f"{prefix}num_dense_layers"),
                label=f"{prefix}dense_layers",
            )
        )

        # output layer
        self.postnet.append(nninn.LazyLinear(outputSize))
        # softmax layer
        self.postnet.append(nninn.Softmax())

        self.prenet = nninn.Sequential(*self.prenet)
        # self.RNNnet = nninn.Sequential(*self.RNNnet)
        self.postnet = nninn.Sequential(*self.postnet)

    def forward(self, input, hidden):

        if isinstance(input, list):
            if len(input) > 1:
                raise ValueError(
                    "Expect one tensor, get {} tensors".format(len(input)))
            input = input[0]

        # make sure input to embedding layer is long
        input = input.long()

        # embedding layer
        output = self.prenet(input)
        # RNN layer
        output, hidden = self.RNNnet(output, hidden)
        # post layer
        output = self.postnet(output)[:, -1, :]

        return output, hidden

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

    def init_hidden(self, batch_size):
        num_directions = 2 if self._bidirectional else 1
        if isinstance(self.RNNnet, nninn.LSTM):
            return (
                torch.zeros(
                    self.RNNnet.num_layers * num_directions, batch_size, self.RNNnet.hidden_size,
                ),
                torch.zeros(
                    self.RNNnet.num_layers * num_directions, batch_size, self.RNNnet.hidden_size,
                )
            )
        else:
            return torch.zeros(
                self.RNNnet.num_layers * num_directions, batch_size, self.RNNnet.hidden_size,
            )

    @staticmethod
    def build_model(config_path, inputSize, outputSize):

        with fixed_arch(config_path):
            return RNNHead(inputSize, outputSize)
