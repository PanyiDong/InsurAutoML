"""
File: _rnn.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Last Version: 0.2.1
Relative Path: /My_AutoML/_utils/_nas/_enas/_rnn.py
File Created: Saturday, 16th July 2022 2:54:09 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 24th October 2022 10:52:45 pm
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
from torch.nn import Parameter

from InsurAutoML._utils._nas._ops import activation_dict


class CustomRNNCell(nn.Module):

    """
    Custom RNN Cell whose structure is defined by a list of nodes.
    """

    def __init__(
        self,
        cellList,
        inputSize,
        hiddenSize,
        activationList=list(activation_dict.values()),
        bias=True,
        verbose=True,
    ):
        super().__init__()
        self.bias = bias
        self.verbose = verbose

        # N nodes in a cell structure
        N = (len(cellList) + 1) // 2

        # check whether the number of nodes is odd
        if len(cellList) != 2 * N - 1:
            print(
                "Number of node information {} generated is not odd. Only first {} nodes and {} connections are used.".format(
                    len(cellList),
                    N,
                    N -
                    1))

        # activation functions at each node
        self.activations = [cellList[2 * i] for i in range(N)]
        # convert from list to actual activation functions
        self.activations = [activationList[i] for i in self.activations]

        # connections between nodes
        self.connections = [cellList[2 * i + 1] for i in range(N - 1)]

        # input weight
        self.inputWeights = Parameter(torch.randn(inputSize, hiddenSize))
        # # register parameters
        # self.register_parameter(self.inputWeights)

        # hidden weight
        self.hiddenWeights = Parameter(torch.randn(N, hiddenSize, hiddenSize))
        # # register parameters
        # self.register_parameter(self.hiddenWeights)

        # hidden bias
        if self.bias:
            self.hiddenBias = Parameter(torch.randn(N, hiddenSize))
            # # register parameters
            # self.register_parameter(self.hiddenBias)

    def forward(self, input, hidden):

        # N nodes in a cell structure
        N = len(self.activations)

        # initialize node values
        node_value = [0 for _ in range(N)]
        node_not_connected = [i for i in range(N)]

        for idx in range(N):
            # first node: combine input with hidden
            if idx == 0:
                # whether to use bias
                if self.bias:
                    node_value[idx] = self.activations[idx](
                        torch.mm(input, self.inputWeights)
                        + torch.mm(hidden, self.hiddenWeights[idx, :, :].clone())
                        + self.hiddenBias[idx, :].clone()
                    )
                else:
                    node_value[idx] = self.activations[idx](
                        torch.mm(input, self.inputWeights)
                        + torch.mm(hidden, self.hiddenWeights[idx, :, :].clone())
                    )
            # from the second node: only use previous hidden
            else:
                # activation at node idx
                # whether to use bias
                if self.bias:
                    node_value[idx] = self.activations[idx](
                        torch.mm(
                            node_value[idx - 1],
                            self.hiddenWeights[idx, :, :].clone(),
                        )
                        + self.hiddenBias[idx, :].clone()
                    )
                else:
                    node_value[idx] = self.activations[idx](
                        torch.mm(
                            node_value[idx - 1],
                            self.hiddenWeights[idx, :, :].clone(),
                        )
                    )

                # check whether node to connect is above idx
                if self.connections[idx - 1] > idx and self.verbose:
                    print(
                        "Node {} is connected to node {}, which is in the future.".format(
                            idx, self.connections[idx - 1]
                        )
                    )
                    # force the connection to the last node
                    self.connections[idx - 1] = idx - 1

                # get connection from previous node
                node_value[idx] = (
                    node_value[idx] + node_value[self.connections[idx - 1]]
                )

                # remove connected node idx
                if self.connections[idx - 1] in node_not_connected:
                    node_not_connected.remove(self.connections[idx - 1])

        # get next hidden state
        # find all nodes that are not further connected
        final_list = [node_value[i] for i in node_not_connected]

        # stack all left nodes and return the average
        # since the nodes are stacked, the average dim is 0
        return torch.mean(torch.stack(final_list), dim=0)


class CustomRNN(nn.Module):
    def __init__(
        self,
        cellList,
        inputSize,
        hiddenSize,
        num_layers=1,
        bias=True,
        p_dropout=0,
        batch_first=True,
    ):
        super().__init__()

        self.cellList = cellList
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.num_layers = num_layers
        self.bias = bias
        self.p_dropout = p_dropout
        self.batch_first = batch_first

        self.module = None

    def forward(self, input, hidden):

        # check shape of input and hidden
        # if not match, raise error
        # last dim of input must be inputSize
        if input.size()[-1] != self.inputSize:
            raise ValueError(
                "Last dimension of input must be equal to inputSize. Expect {} but got {}".format(
                    self.inputSize, input.size()[-1]
                )
            )
        # last dim of hidden must be hiddenSize
        if hidden.size()[-1] != self.hiddenSize:
            raise ValueError(
                "Last dimension of hidden must be equal to hiddenSize. Expect {} but got {}".format(
                    self.hiddenSize, hidden.size()[-1]
                )
            )
        # first dim of hidden must be num_layers
        if hidden.size()[0] != self.num_layers:
            raise ValueError(
                "First dimension of hidden must be equal to num_layers. Expect {} but got {}".format(
                    self.num_layers, hidden.size()[0]))
        # if 2d input, hidden must be 2d
        if len(input.size()) == 2 and len(hidden.size()) != 2:
            raise ValueError(
                "Hidden must be 3d given 2d input. Get {}d hidden.".format(
                    len(hidden.size())
                )
            )
        # if 3d input, hidden must be 3d
        elif len(input.size()) == 3 and len(hidden.size()) != 3:
            raise ValueError(
                "Hidden must be 3d given 3d input. Get {}d hidden.".format(
                    len(hidden.size())
                )
            )

        # if 3d input and batch_first is True: first dim of input must equal to second dim of hidden
        # False: second dim of input must equal to second dim of hidden
        if (
            len(input.size()) == 3
            and self.batch_first
            and not input.size()[0] == hidden.size()[1]
        ):
            raise ValueError(
                "Second dimension of hidden must equal to first dimension of input. Expect {} but got {}".format(
                    input.size()[0], hidden.size()[1]))
        elif (
            len(input.size()) == 3
            and not self.batch_first
            and not input.size()[1] == hidden.size()[1]
        ):
            raise ValueError(
                "Second dimension of hidden must equal to second dimension of input. Expect {} but got {}".format(
                    input.size()[1], hidden.size()[1]))

        # check whether contains batch dim
        is_batch = False if len(input.size()) == 2 else True

        # if the module is not initialized, initialize it
        if self.module is None:
            self.module = []

            # find length of sequence for the architecture
            if not is_batch or (is_batch and not self.batch_first):
                seq_len = input.size(0)
            elif is_batch and self.batch_first:
                seq_len = input.size(1)
            else:
                raise ValueError(
                    "Only input size 2d or 3d tensor supported. Input size {} not supported.".format(
                        input.size()))

            # iterate through the layers
            for idx in range(self.num_layers):
                # at first layer, (input, hidden) is provided
                if idx == 0:
                    self.module.append(
                        nn.ModuleList(
                            [
                                CustomRNNCell(
                                    self.cellList,
                                    self.inputSize,
                                    self.hiddenSize,
                                    bias=self.bias,
                                )
                                for _ in range(seq_len)
                            ]
                        )
                    )
                # at other layers, (hidden, hidden) is provided
                else:
                    self.module.append(
                        nn.ModuleList(
                            [
                                CustomRNNCell(
                                    self.cellList,
                                    self.hiddenSize,
                                    self.hiddenSize,
                                    bias=self.bias,
                                )
                                for _ in range(seq_len)
                            ]
                        )
                    )
                # if p_dropout is not 0, add dropout layer
                if self.p_dropout > 0:
                    self.module.append(nn.Dropout(self.p_dropout))

            # convert the list to a module
            self.module = nn.Sequential(*self.module)

        # forward the module
        hidden_layer = 0
        hidden_output = []
        for module in self.module.modules():
            # if is a ModuleList, expect stacked RNN cells
            if isinstance(module, nn.ModuleList):
                _output = []
                # get hidden at the layer
                _hidden = (
                    hidden[hidden_layer, :].view(1, -1)
                    if not is_batch
                    else hidden[hidden_layer, :, :]
                )
                for idx, cell in enumerate(module):
                    # if the first layer, input is provided
                    if hidden_layer == 0:
                        # get corresponding input at time step
                        if not is_batch:
                            _hidden = cell(input[idx, :].view(1, -1), _hidden)
                        # if 3d input, get the corresponding dim
                        if is_batch:
                            _hidden = cell(
                                input[:, idx, :]
                                if self.batch_first
                                else input[idx, :, :],
                                _hidden,
                            )
                    # if other layers, hidden is provided
                    else:
                        # get corresponding input at time step
                        if not is_batch:
                            _hidden = cell(output[idx, :].view(1, -1), _hidden)
                        # if 3d input, get the corresponding dim
                        # after one layer of RNN, the input will be reshaped to
                        # (seq_len, batch_size, hidden_size)
                        if is_batch:
                            _hidden = cell(
                                output[idx, :, :],
                                _hidden,
                            )

                    _output.append(_hidden)

                # stack all outputs from all time steps
                output = torch.stack(_output, dim=0)
                # record the last hidden state from each layer
                hidden_output.append(_hidden)
                # increase the layer
                hidden_layer += 1
            # if dropout layer found, forward it
            elif isinstance(module, nn.Dropout):
                output = module(output)

        hidden_output = torch.stack(hidden_output, dim=0)

        if is_batch:
            return (
                output.permute(1, 0, 2) if self.batch_first else output,
                hidden_output,
            )
        else:
            return (
                output[:, -1, :],
                hidden_output[:, -1, :],
            )
