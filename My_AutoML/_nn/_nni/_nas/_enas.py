"""
File: _enas.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Last Version: 0.2.1
Relative Path: /My_AutoML/_nn/_nni/_nas/_enas.py
File Created: Wednesday, 20th July 2022 2:48:16 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Saturday, 6th August 2022 10:45:57 pm
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
import torch.nn.functional as F

from nni.nas.pytorch import mutables
from nni.retiarii.oneshot.pytorch.random import PathSamplingInputChoice

from ._ops import (
    SepConvBN,  # base space
    AvgPool,
    MaxPool,
    FullConvBN,  # full space
    AvgPoolBN,
    MaxPoolBN,
    ChannelCalibration,
    ReductionLayer,
    AuxiliaryHead,
    FactorizedReduce,
)
from ._utils import how_to_init

###################################################################################################################
# MLP base structure


###################################################################################################################
# CNN related components/structures


class CNNCell(nn.Module):
    def __init__(
        self,
        cellName,
        previousLabel,
        channel,
    ):
        super().__init__()

        self.input = mutables.InputChoice(
            choose_from=previousLabel,
            n_chosen=1,
            return_mask=True,
            key=cellName + "_input",
        )

        self.operation = mutables.LayerChoice(
            [
                SepConvBN(channel, channel, kernel_size=3, stride=1),
                SepConvBN(channel, channel, kernel_size=5, stride=2),
                AvgPool(kernel_size=3, stride=1, padding=1),
                MaxPool(kernel_size=3, stride=1, padding=1),
                nn.Identity(),
            ],
            key=cellName + "_operation",
        )

    def forward(self, X):

        # get input from previous layer
        output = self.input(X)

        # check whether to use operation
        if isinstance(self.input, PathSamplingInputChoice):
            return output, self.input.mask
        else:
            _input, _mask = output
            return self.operation(_input), _mask


class CNNNode(mutables.MutableScope):
    def __init__(
        self,
        nodeName,
        previousNodeName,
        channel,
    ):
        super().__init__(nodeName)

        self.cell_1 = CNNCell(nodeName + "_1", previousNodeName, channel)
        self.cell_2 = CNNCell(nodeName + "_2", previousNodeName, channel)

    def forward(self, X):

        out_1, mask_1 = self.cell_1(X)
        out_2, mask_2 = self.cell_2(X)

        return out_1 + out_2, mask_1 | mask_2


###################################################################################################################
# CNN base space


class CNNBaseLayer(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_channels1,
        in_channels2,
        out_channel,
        reduction=True,
    ):
        super().__init__()

        # reshape channels for two in_channels
        self.preprocess1 = ChannelCalibration(in_channels1, out_channel)
        self.preprocess2 = ChannelCalibration(in_channels2, out_channel)

        # number of nodes in this layer
        self.num_nodes = num_nodes
        # name of the node, whether reduction is used
        name_prefix = "reduced" if reduction else "normal"
        # node labels initialization
        node_labels = [mutables.InputChoice.NO_KEY, mutables.InputChoice.NO_KEY]

        # initialize nodes
        self.nodes = nn.ModuleList()

        # set up nodes
        for i in range(num_nodes):
            # get latest node label
            node_labels.append("{}_node_{}".format(name_prefix, i))
            # add node to the module
            # available previous nodes are everything but the latest one (not constructed yet)
            self.nodes.append(CNNNode(node_labels[-1], node_labels[:-1], out_channel))

        self.conv_weight = nn.Parameter(
            torch.zeros(out_channel, self.num_nodes + 2, out_channel, 1, 1),
            requires_grad=True,
        )
        self.reset_conv_weight()
        # batch normalization
        self.bn = nn.BatchNorm2d(out_channel, affine=False)

    # previous2 is from the previous layer
    # previous1 is from two layers ahead
    def forward(self, previous1, previous2):

        # reset channels so that two previous layers can be connected
        nodeValue = [self.preprocess1(previous1), self.preprocess2(previous2)]

        # make sure the devices used are the same, otherwise it will cause error
        record_mask = torch.zeros(
            self.num_nodes + 2, dtype=torch.bool, device=previous2.device
        )

        # forward phase: 1. feed forward through the nodes, 2. update the masks
        for i in range(self.num_nodes):
            # get output from each node
            out, mask = self.nodes[i](nodeValue)
            # update used mask
            record_mask[: mask.size(0)] |= mask.to(out.device)
            # update previous layer
            nodeValue.append(out)

        # concat all unused(unmasked) nodes
        unused_nodes = torch.cat(
            [out for used, out in zip(record_mask, nodeValue) if not used], dim=1
        )
        # use ReLU activation
        unused_nodes = F.relu(unused_nodes)

        # get conv weights
        conv_weight = self.conv_weight[:, ~record_mask, :, :, :]
        conv_weight = conv_weight.view(conv_weight.size(0), -1, 1, 1)

        out = F.conv2d(unused_nodes, conv_weight)

        return previous2, self.bn(out)

    def reset_conv_weight(self, how="kaiming_normal"):

        if how == "kaiming_normal":
            nn.init.kaiming_normal_(self.conv_weight)


class CNNBaseSpace(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=24,
        output_size=10,
        num_layers=2,
        num_nodes=5,
        p_dropout=0.0,
        use_aux_head=False,
    ):
        super().__init__()
        # initialize parameters
        self.num_layers = num_layers
        self.use_aux_head = use_aux_head

        # preprocess layers
        self.preprocess = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels * 3,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels * 3),
        )

        # pool related parameters
        pool_distance = self.num_layers // 3
        pool_layers = [pool_distance, 2 * pool_distance + 1]

        self.net = nn.ModuleList()
        # set channels for previous layers
        c_pp = c_p = out_channels * 3
        # set channels for current layers
        c_cur = out_channels

        # set up network
        for id in range(self.num_layers + 2):
            reduction = False  # whether a reduction layer
            # for pool layers, use reduction
            if id in pool_layers:
                c_cur = c_p * 2
                reduction = True
                self.net.append(ReductionLayer(c_pp, c_p, c_cur))
                c_pp = c_p = c_cur
            # normal layers
            self.net.append(CNNBaseLayer(num_nodes, c_pp, c_p, c_cur, reduction))

            # whether to use aux head
            if self.use_aux_head and id == pool_distance[-1] + 1:
                self.net.append(AuxiliaryHead(c_cur, output_size))

            # reset channels
            c_pp, c_p = c_p, c_cur

        # layers after the CNN layers
        # Average pooling layer
        self.gap = nn.AdaptiveAvgPool2d(1)
        # ReLU activation
        self.relu = nn.ReLU()
        # dropout layers
        self.dropout = nn.Dropout(p_dropout)
        # dense linear layer
        self.linear = nn.Linear(c_cur, output_size)

        self.reset_parameters()

    def forward(self, X):

        x0 = X.size(0)

        previous = current = self.preprocess(X)
        aux_logits = None

        # forward through the CNN layers
        for layer in self.net:
            if isinstance(layer, AuxiliaryHead):
                if self.training:
                    aux_logits = layer(current)
            else:
                previous, current = layer(previous, current)

        output = self.gap(self.relu(current)).view(x0, -1)
        output = self.linear(self.dropout(output))

        if aux_logits is not None:
            return output, aux_logits
        return output

    def reset_parameters(self, how="kaiming_normal"):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                how_to_init(m, how)


###################################################################################################################
# CNN full space


class CNNFullLayer(mutables.MutableScope):
    def __init__(
        self,
        key,
        previous_labels,
        in_channels,
        out_channels,
    ):
        super().__init__(key)

        self.in_channels = in_channels
        self.out_channels = out_channels

        # layer choices
        self.layer = mutables.LayerChoice(
            [
                FullConvBN(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    separable=False,
                ),
                FullConvBN(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    separable=True,
                ),
                FullConvBN(
                    in_channels,
                    out_channels,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    separable=False,
                ),
                FullConvBN(
                    in_channels,
                    out_channels,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    separable=True,
                ),
                AvgPoolBN(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                ),
                MaxPoolBN(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                ),
            ]
        )

        # whether to skip connection from previous layer
        if len(previous_labels) > 0:
            self.skipconnect = mutables.InputChoice(
                choose_from=previous_labels, n_chosen=None
            )
        else:
            self.skipconnect = None

        # batch normalization layer
        self.bn = nn.BatchNorm2d(out_channels, affine=False)

    def forward(self, X):  # X is a list from previous layer
        # forward through the layer
        output = self.layer(X[-1])
        # find connections to previous layers if needed
        if self.skipconnect is not None:
            connection = self.skipconnect(X[:-1])
            if connection is not None:
                output = output + connection

        # batch normalization
        return self.bn(output)


class CNNFullSpace(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=24,
        output_size=10,
        num_layers=12,
        p_dropout=0.0,
    ):
        super().__init__()
        # initialize parameters
        self.num_layers = num_layers

        # preprocess layers
        self.preprocess = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        # pool related parameters
        pool_distance = self.num_layers // 3
        self.pool_layer_idx = [pool_distance - 1, 2 * pool_distance - 1]

        # initialize layers
        self.net = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        labels = []

        for id in range(self.num_layers):
            labels.append("layer_{}".format(id))
            # if in pool layers, add a FactorizedReduce layer
            if id in self.pool_layer_idx:
                self.pool_layers.append(FactorizedReduce(out_channels, out_channels))
            self.net.append(
                CNNFullLayer(labels[-1], labels[:-1], out_channels, out_channels)
            )

        # layers after the CNN layers
        # Average pooling layer
        self.gap = nn.AdaptiveAvgPool2d(1)
        # dropout layers
        self.dropout = nn.Dropout(p_dropout)
        # dense linear layer
        self.linear = nn.Linear(out_channels, output_size)

    def forward(self, X):

        x0 = X.size(0)
        # forward preprocess layer
        current = self.preprocess(X)
        layers = [current]

        for id in range(self.num_layers):
            # forward through the CNN layers
            current = self.net[id](layers)
            layers.append(current)
            # forward through the pool layers
            if id in self.pool_layer_idx:
                for idx, layer in enumerate(layers):
                    layers[idx] = self.pool_layers[self.pool_layer_idx.index(id)](layer)
                current = layers[-1]

        # forward through the postprocess layers
        # Average pooling layer
        output = self.gap(current).view(x0, -1)
        # dropout layers and dense linear layer
        output = self.linear(self.dropout(output))

        return output


###################################################################################################################
# RNN base space


class RNNCell(nn.Module):
    def __init__(
        self,
        cellName,
        previousLabel,
        inputSize,
        outputSize,
    ):
        super().__init__()

        self.input = mutables.InputChoice(
            choose_from=previousLabel,
            n_chosen=1,
            return_mask=True,
            key=cellName + "_input",
        )

        self.inputWeight = nn.Parameter(torch.Tensor(inputSize, outputSize))
        self.hiddenWeight = nn.Parameter(torch.Tensor(outputSize, outputSize))
        self.bias = nn.Parameter(torch.Tensor(outputSize))

        self.operation = mutables.LayerChoice(
            [
                nn.ReLU(),
                nn.Sigmoid(),
                nn.Tanh(),
            ],
            key=cellName + "_activation",
        )

    def forward(self, input, hidden):  # input here is just hidden state

        # get input from previous layer
        output = self.input(input, hidden)

        # check whether to use operation
        # if PathSamplingInputChoice, no need for operation and no mask returned
        if isinstance(self.input, PathSamplingInputChoice):

            # unpack the output
            output, hidden = output

            return (output, hidden), self.input.mask
        else:
            # unpack the output
            output, hidden = output
            (output, hidden), mask = output

            # apply linear transformation and activation
            output = self.operation(
                torch.mul(output, self.inputWeight)
                + torch.mul(hidden, self.hiddenWeight)
                + self.bias
            )
            return (self.operation(output), hidden), mask


class RNNNode(mutables.MutableScope):
    def __init__(
        self,
        nodeName,
        previousNodeName,
        channel,
    ):
        super().__init__(nodeName)

        self.cell_1 = RNNCell(nodeName + "_1", previousNodeName, channel)
        self.cell_2 = RNNCell(nodeName + "_2", previousNodeName, channel)

    def forward(self, input, hidden):

        (output_1, hidden_1), mask_1 = self.cell_1(input, hidden)
        (output_2, hidden_2), mask_2 = self.cell_2(input, hidden)

        # take average from previous cells
        return (
            torch.mean(torch.stack([output_1, output_2]), dim=0),
            torch.mean(torch.stack([hidden_1, hidden_2]), dim=0),
        ), mask_1 | mask_2


###################################################################################################################
# RNN full space
