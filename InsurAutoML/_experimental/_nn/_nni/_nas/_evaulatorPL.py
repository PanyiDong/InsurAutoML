"""
File Name: _evaulatorPL.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_experimental/_nn/_nni/_nas/_evaulatorPL.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Thursday, 24th November 2022 3:51:55 pm
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

from collections.abc import Iterable
import nni
import nni.retiarii.evaluator.pytorch.lightning as pl
# from nni.retiarii.evaluator.pytorch import DataLoader
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import importlib

from InsurAutoML._experimental._nn._nni._nas._utils import tensor_accuracy
from ._utils import get_optimizer, get_criterion, get_scheduler

# serialized version of dataloader
DataLoader = nni.trace(DataLoader)

pl_spec = importlib.util.find_spec("pytorch_lightning")
# if pl not found, raise error
if pl_spec is None:
    raise ImportError(
        "Use of pytorch-lightning evaluator requires pytorch-lightning to be installed. \
        Use `pip install pytorch-lightning` to install.")


def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


@nni.trace
class plEvaluator(pl.LightningModule):

    ACCURACY_TYPE = ["binary", "multiclass"]

    def __init__(
        self,
        model,
        type_of_task="multiclass",
        batch_size=32,
        criterion="CrossEntropyLoss",
        optimizer="Adam",
        optimizer_lr=1e-3,
        lr_scheduler="None",
    ):
        super().__init__()

        # set parameters
        self.model = model
        self.type_of_task = type_of_task
        self.batch_size = batch_size
        self.criterion = get_criterion(criterion)()
        self.optimizer = get_optimizer(optimizer)
        self.optimizer_lr = optimizer_lr
        self.lr_scheduler = get_scheduler(lr_scheduler)

    def forward(self, X):
        print(self.model)
        if self._hidden is not None:
            # only get the values of hidden, no need for gradient
            self._hidden = repackage_hidden(self._hidden)
            output, self._hidden = self.model(X, self._hidden)
        else:
            output = self.model(X)

        return output

    def training_step(self, batch, batch_idx):

        # initialize hidden state for each epoch
        if not hasattr(self, '_train_epoch'):
            self._train_epoch = self.current_epoch
            self.init_hidden()
        elif self._train_epoch != self.current_epoch:
            self._train_epoch = self.current_epoch
            self.init_hidden()

        input, label = batch  # parse the input batch
        # forward phase
        output = self(input)

        loss = self.criterion(output, label)  # compute loss
        self.log("train_loss", loss)  # Logging to TensorBoard by default

        return loss

    def validation_step(self, batch, batch_idx):

        # initialize hidden state for each epoch
        if not hasattr(self, '_valid_epoch'):
            self._valid_epoch = self.current_epoch
            self.init_hidden()
        elif self._valid_epoch != self.current_epoch:
            self._valid_epoch = self.current_epoch
            self.init_hidden()

        input, label = batch  # parse the input batch
        output = self(input)
        loss = self.criterion(output, label)  # compute loss
        # # if classification job, use accuracy
        # if self.type_of_task in self.ACCURACY_TYPE:
        #     # Logging to TensorBoard by default
        #     self.log("val_loss", tensor_accuracy(output, label))
        # # for regression job, use -criterion to satisfy maximization setting
        # else:
        #     self.log("val_loss", -loss)  # Logging to TensorBoard by default
        self.log("val_loss", -loss)

        return {"- val_loss": -loss}

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
        nni.report_intermediate_result(
            self.trainer.callback_metrics["val_loss"].item())

    def teardown(self, stage):
        if stage == "fit":
            nni.report_final_result(
                self.trainer.callback_metrics["val_loss"].item())

    def init_hidden(self):

        if hasattr(self.model, 'init_hidden'):
            self._hidden = self.model.init_hidden(self.batch_size)
        else:
            self._hidden = None


# @nni.trace
def get_evaluator(
    model,
    train_set,
    test_set,
    batchSize=32,
    # type_of_task="multiclass",
    criterion="CrossEntropyLoss",
    optimizer="Adam",
    optimizer_lr=1e-3,
    lr_scheduler="None",
    num_epochs=10,
    log_dir="tmp/NAS"
):
    # setup pytorch_lightning logger
    logger = TensorBoardLogger(log_dir, name="pl_logs")

    return pl.Lightning(
        lightning_module=plEvaluator(
            model=model,
            # type_of_task=type_of_task,
            batch_size=batchSize,
            criterion=criterion,
            optimizer=optimizer,
            optimizer_lr=optimizer_lr,
            lr_scheduler=lr_scheduler,
        ),
        trainer=pl.Trainer(
            logger=logger,
            # early stopping when loss is nan
            callbacks=[EarlyStopping(
                monitor="val_loss", mode="min", check_finite=True)],
            max_epochs=num_epochs,
            gpus=torch.cuda.device_count()),
        train_dataloader=pl.DataLoader(
            train_set,
            batch_size=batchSize,
            shuffle=True,
            drop_last=True,
        ),
        val_dataloaders=pl.DataLoader(
            test_set,
            batch_size=batchSize,
            drop_last=True,
        ),
    )
