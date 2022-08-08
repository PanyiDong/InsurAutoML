"""
File: _utilsPL.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Last Version: 0.2.1
Relative Path: /My_AutoML/_utils/_nas/_nni/_utilsPL.py
File Created: Tuesday, 19th July 2022 2:04:35 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Tuesday, 19th July 2022 11:37:57 pm
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

# pytorch-lightning version of evaluator

import importlib

pl_spec = importlib.util.find_spec("pytorch_lightning")
# if pl not found, raise error
if pl_spec is None:
    raise ImportError(
        "Use of pytorch-lightning evaluator requires pytorch-lightning to be installed. \
        Use `pip install pytorch-lightning` to install."
    )

import torch
import nni
import nni.retiarii.evaluator.pytorch.lightning as pl

from ._utils import get_optimizer, get_criterion, get_scheduler


@nni.trace
class plEvaluator(pl.LightningModule):
    def __init__(
        self,
        model,
        criterion="CrossEntropyLoss",
        optimizer="Adam",
        optimizer_lr=1e-3,
        lr_scheduler="None",
    ):
        super().__init__()

        # set parameters
        self.model = model
        self.criterion = get_criterion(criterion)()
        self.optimizer = get_optimizer(optimizer)
        self.optimizer_lr = optimizer_lr
        self.lr_scheduler = get_scheduler(lr_scheduler)

    def forward(self, X):

        return self.model(X)

    def training_step(self, batch, batch_idx):
        input, label = batch  # parse the input batch
        output = self.model(input)  # forward phasepip
        loss = self.criterion(output, label)  # compute loss
        self.log("train_loss", loss)  # Logging to TensorBoard by default

        return loss

    def validation_step(self, batch, batch_idx):
        input, label = batch  # parse the input batch
        output = self.model(input)  # forward phase
        print(output, label)
        loss = self.criterion(output, label)  # compute loss
        self.log("val_loss", loss)  # Logging to TensorBoard by default

    def configure_optimizers(self):

        # initialize optimizer
        self.optimizer = self.optimizer(self.parameters(), self.optimizer_lr)

        if self.lr_scheduler is None:
            return self.optimizer
        else:
            return {
                "optimizer": self.optimizer,
                "scheduler": self.lr_scheduler(self.optimizer),
            }

    def on_validation_epoch_end(self):
        nni.report_intermediate_result(self.trainer.callback_metrics["val_loss"].item())

    def teardown(self, stage):
        if stage == "fit":
            nni.report_final_result(self.trainer.callback_metrics["val_loss"].item())


def get_evaluator(
    model,
    train_set,
    test_set,
    batchSize=32,
    criterion="CrossEntropyLoss",
    optimizer="Adam",
    optimizer_lr=1e-3,
    lr_scheduler="None",
    num_epochs=10,
):
    return pl.Lightning(
        lightning_module=plEvaluator(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            optimizer_lr=optimizer_lr,
            lr_scheduler=lr_scheduler,
        ),
        trainer=pl.Trainer(max_epochs=num_epochs, gpus=torch.cuda.device_count()),
        train_dataloader=pl.DataLoader(
            train_set,
            batch_size=batchSize,
            shuffle=True,
            drop_last=True,
        ),
        val_dataloaders=pl.DataLoader(
            test_set,
            batch_size=batchSize,
        ),
    )
