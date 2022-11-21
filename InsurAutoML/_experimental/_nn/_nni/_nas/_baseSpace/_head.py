"""
File Name: _head.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_experimental/_nn/_nni/_nas/_baseSpace/_head.py
File Created: Sunday, 13th November 2022 5:15:00 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Saturday, 19th November 2022 10:06:10 pm
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
from nni.retiarii import model_wrapper

from .._utils import ACTIVATIONS, RNN_TYPES, how_to_init

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


@model_wrapper
class MLPHead(nninn.Module):
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
                    nninn.Linear(
                        nninn.ValueChoice(
                            [4, 8, 16, 32, 64],
                            label=f"hiddenSize_at_layer_{index}",
                        ),
                        nninn.ValueChoice(
                            [4, 8, 16, 32, 64],
                            label=f"hiddenSize_at_layer_{index + 1}",
                        ),
                    ),
                    nninn.LayerChoice(
                        ACTIVATIONS, label=f"activation_at_layer_{index}"
                    ),
                    nninn.Dropout(
                        nninn.ValueChoice(
                            [0.0, 0.2, 0.5, 0.9], label=f"p_dropout_at_layer_{index}"
                        )
                    ),
                ),
                nninn.ValueChoice([1, 2, 3], label="num_hidden_layers"),
                label="hidden_layers",
            )
        )

        # add output layer
        # use lazy linear to automatically initialize the input size
        self.net.append(nninn.LazyLinear(outputSize))

        # add softmax layer
        self.net.append(nninn.Softmax())

        self.net = nninn.Sequential(*self.net)

    def forward(self, X):

        return self.net(X)

    def init_weight(self, how="xavier_normal"):

        for m in self.modules():
            if isinstance(m, nninn.Linear):
                how_to_init(m, how)
