"""
File: _trainer.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Last Version: 0.2.1
Relative Path: /InsurAutoML/archive/_enas/_trainer.py
File Created: Sunday, 17th July 2022 3:19:34 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 28th November 2022 11:43:32 pm
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
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from InsurAutoML.utils._nas._enas._RNN import CustomRNNCell, CustomRNN
from InsurAutoML.utils._nas._enas._controller import Controller
from InsurAutoML.utils.tensor import repackage_hidden

inputSize = 512
outputSize = 2

controller = Controller()

train = TensorDataset(
    torch.randn(1024, 20, 512),
    torch.randint(0, 2, (1024, )),
)
valid = TensorDataset(
    torch.randn(32, 20, 512),
    torch.randint(0, 2, (32, )),
)
test = TensorDataset(
    torch.randn(32, 20, 512),
    torch.randint(0, 2, (32, )),
)

train = DataLoader(train, batch_size=32, shuffle=True)
valid = DataLoader(valid, batch_size=32)
test = DataLoader(test, batch_size=32)

torch.autograd.set_detect_anomaly(True)

for _ in range(10):

    hidden = torch.randn(2, 32, outputSize)
    inputList, log_probs, entropy = controller.sample()
    print(inputList)
    layer = CustomRNN(
        inputList,
        inputSize,
        outputSize,
        num_layers=2,
        batch_first=True)

    controller_optimizer = optim.Adam(controller.parameters(), lr=0.001)
    model_loss = nn.CrossEntropyLoss()
    controller_loss = nn.SmoothL1Loss()

    with torch.no_grad():
        for X, y in valid:
            output, hidden = layer(X, hidden)
            output = output[:, -1, :]
            pre_controller_loss = model_loss(output, y)

    model_optimizer = optim.SGD(layer.parameters(), lr=0.001)

    for _ in range(2):
        hidden = torch.randn(2, 32, outputSize)
        for X, y in train:
            hidden = repackage_hidden(hidden)  # detach gradient
            model_optimizer.zero_grad()
            output, hidden = layer(X, hidden)
            output = output[:, -1, :]
            loss = model_loss(output, y)
            loss.backward(retain_graph=True)
            model_optimizer.step()

    with torch.no_grad():
        for X, y in valid:
            output, hidden = layer(X, hidden)
            output = output[:, -1, :]
            after_controller_loss = model_loss(output, y)

    # calculate reward
    np_entropy = entropy.detach().cpu().numpy()

    controller_optimizer.zero_grad()
    # controller_loss = torch.mul(inputList, pre_controller_loss.item() - after_controller_loss.item())
    after_controller_loss.backward(retain_graph=True)
    controller_optimizer.step()

    print(inputList, after_controller_loss.item())
