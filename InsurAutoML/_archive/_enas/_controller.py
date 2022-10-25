"""
File: _controller.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Last Version: 0.2.1
Relative Path: /My_AutoML/_utils/_nas/_enas/_controller.py
File Created: Saturday, 16th July 2022 11:00:51 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 24th October 2022 10:52:48 pm
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

from InsurAutoML._utils._nas._ops import activation_dict
from InsurAutoML._utils._nas._enas._utils import decision_mask


class Controller(nn.Module):
    def __init__(
        self,
        inputSize=10,
        hiddenSize=100,
        n_nodes=4,
        n_activations=len(activation_dict),
        num_layers=1,
        batch_first=True,
    ):
        super().__init__()

        outputSize = max(n_nodes - 1, n_activations)

        self.inputSize = inputSize
        self.n_nodes = n_nodes
        self.n_activations = n_activations
        self.num_layers = num_layers
        self.hiddenSize = hiddenSize
        self.embed = nn.Embedding(inputSize, hiddenSize)
        self.lstm = nn.LSTM(
            hiddenSize, hiddenSize, num_layers=num_layers, batch_first=batch_first
        )
        self.linear = nn.Linear(hiddenSize, outputSize)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, hidden):

        output = self.embed(input)
        output, (hx, cx) = self.lstm(output, hidden)
        output = self.linear(output)
        # get output from last time step
        output = output[-1, :, :]

        # apply mask to output
        mask = decision_mask(n_nodes=self.n_nodes, n_activations=self.n_activations)
        output = self.softmax(output) * mask

        return torch.argmax(output, dim=1), (hx, cx)

    def sample(self, with_details=True):

        input = self.init_input("zeros")
        hidden = self.init_hidden()

        logits, hidden = self.forward(input, hidden)

        # get probs and log_probs
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # get entropy
        entropy = -(log_probs * probs).sum(dim=-1)

        if with_details:
            return logits, log_probs, entropy
        else:
            return logits

    def init_input(self, initialization="zeros"):

        if initialization == "zeros":
            return torch.zeros([1, 2 * self.n_nodes - 1], dtype=torch.long)
        elif initialization == "random":
            return torch.rand([1, 2 * self.n_nodes - 1], dtype=torch.long)
        elif initialization == "normal":
            return torch.randn([1, 2 * self.n_nodes - 1], dtype=torch.long)

    def init_hidden(self):
        return (
            torch.zeros(
                self.num_layers,
                self.hiddenSize,
                dtype=torch.float,
            ),
            torch.zeros(
                self.num_layers,
                self.hiddenSize,
                dtype=torch.float,
            ),
        )
