"""
File: _trainer.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Last Version: 0.2.1
Relative Path: /My_AutoML/_utils/_nas/_nni/_trainer.py
File Created: Sunday, 17th July 2022 9:40:10 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Tuesday, 19th July 2022 11:37:45 pm
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
from torch.utils.data import DataLoader
import nni
from nni.retiarii.evaluator import FunctionalEvaluator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from ._utils import get_optimizer, get_criterion


@nni.trace
def epochEvaluation(
    model,
    dataloader,
    optimizer,
    criteria,
    epoch,
    mode="train",
):
    # initialize loss and accuracy
    report_loss = 0
    accuracy = 0

    # set model mode
    if mode == "train":
        model.train()
        for idx, (input, label) in enumerate(dataloader):
            input = input.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(input)
            loss = criteria(output, label)
            loss.backward()
            optimizer.step()

    elif mode == "eval":
        model.eval()

        with torch.no_grad():
            for idx, (input, label) in enumerate(dataloader):
                input = input.to(device)
                label = label.to(device)
                output = model(input)
                loss = criteria(output, label)
                report_loss += loss.item()
                accuracy += (output.argmax(dim=1) == label).sum().item()

        report_loss /= len(dataloader.dataset)
        accuracy = 100.0 * accuracy / len(dataloader.dataset)

        return accuracy


@nni.trace
def modelEvaluation(
    model_cls,
    trainloader,
    testloader,
    optimizer,
    criteria,
    num_epoch,
):
    model = model_cls()
    model.to(device)

    optimizer = optimizer(model.parameters())

    for epoch in range(num_epoch):
        epochEvaluation(
            model,
            trainloader,
            optimizer,
            criteria,
            epoch,
            mode="train",
        )
        accuracy = epochEvaluation(
            model,
            testloader,
            optimizer,
            criteria,
            epoch,
            mode="eval",
        )
        nni.report_intermediate_result(accuracy)

    nni.report_final_result(accuracy)


def get_evaluator(
    trainset,
    testset,
    batchSize=32,
    optimizer="Adam",
    criterion=nn.CrossEntropyLoss,
    num_epoch=10,
):
    return FunctionalEvaluator(
        modelEvaluation,
        trainloader=nni.trace(DataLoader)(
            trainset, batch_size=batchSize, shuffle=True, drop_last=True
        ),
        testloader=nni.trace(DataLoader)(testset, batch_size=batchSize),
        optimizer=get_optimizer(optimizer),
        criteria=get_criterion(criterion)(),
        num_epoch=num_epoch,
    )
