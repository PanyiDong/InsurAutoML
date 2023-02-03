"""
File Name: evaluator.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/nn/hpo/evaluator.py
File Created: Monday, 5th December 2022 2:49:26 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 5th December 2022 5:08:24 pm
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

import argparse
from typing import Union, Callable, List, Tuple, Dict
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import nni

from ..utils.args import repackage_hidden
from ..utils.hpSpace import search_space

# get training arguments
parser = argparse.ArgumentParser(description='HPO training for InsurAutoML')
parser.add_argument("--log_dir", type=str,
                    default="tmp/HPO", help="log directory")
parser.add_argument("--use_gpu", type=bool,
                    default=True, help="use GPU or not")
args = parser.parse_args()

# get arguments
log_dir = args.log_dir
use_gpu = args.use_gpu


class HPOEvaluator(pl.LightningModule):

    def __init__(
        self,
        model: Union[str, Callable],
        use_gpu: bool = True,
    ) -> None:
        super().__init__()

        self.model = model
        self.use_gpu = use_gpu

    @staticmethod
    def _get_lr_sch_args(lr_sch: str, params: Dict[str, Union[str, float, int]]) -> Dict[str, Union[str, float, int]]:

        if lr_sch == "StepLR":
            return {"step_size": params["step_size"], "gamma": params["gamma"]}
        elif lr_sch == "ConstantLR":
            return {"factor": params["factor"], "total_iters": params["total_iters"]}
        elif lr_sch == "LinearLR":
            return {"start_factor": params["factor"], "total_iters": params["total_iters"]}
        elif lr_sch == "ExponentialLR":
            return {"gamma": params["gamma"]}
        elif lr_sch == "PolynomialLR":
            return {"power": params["power"], "total_iters": params["total_iters"]}
        elif lr_sch == "ReduceLROnPlateau":
            return {"factor": params["factor"]}

    def forward(self, X: Union[torch.Tensor, List[torch.Tensor]]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self._hidden is not None:
            # only get the values of hidden, no need for gradient
            self._hidden = repackage_hidden(self._hidden)
            output, self._hidden = self.model(X, self._hidden)
        else:
            output = self.model(X)

        return output

    def training_step(self, batch: Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor], batch_idx: int) -> torch.Tensor:

        input, label = batch  # parse the input batch
        # forward phase
        output = self(input)

        loss = self.criterion(output, label)  # compute loss
        self.log("train_loss", loss)  # Logging to TensorBoard by default

        return loss

    def validation_step(self, batch: Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:

        input, label = batch  # parse the input batch
        output = self(input)
        loss = self.criterion(output, label)  # compute loss
        self.log("val_loss", -loss)

        return {"val_loss": -loss}

    def configure_optimizers(self) -> Union[Callable, Dict[str, Callable]]:

        # initialize optimizer
        self.optimizer = getattr(optim, params["optimizer"])(
            self.parameters(), params["lr"], weight_decay=params["weight_decay"])
        self.lr_scheduler = getattr(optim.lr_scheduler, params["lr_scheduler"]) if str(
            params["lr_scheduler"]) is not "None" else None

        if self.lr_scheduler is None:
            return self.optimizer
        else:
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": self.lr_scheduler(self.optimizer, **self._get_lr_sch_args(params["lr_scheduler"], params)),
            }

    def on_train_epoch_start(self) -> None:
        # update train_epoch number
        if not hasattr(self, '_train_epoch'):
            self._train_epoch = self.current_epoch
        else:
            self._train_epoch += 1
        # initialize hidden state for each epoch
        self.init_hidden()

    def on_train_epoch_end(self) -> None:
        # scheduler step for ReduceLROnPlateau
        _sch = self.lr_schedulers()
        if isinstance(_sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            _sch.step(self.trainer.callback_metrics["val_loss"])

    def on_validation_epoch_start(self) -> None:
        # update valid_epoch number
        if not hasattr(self, '_valid_epoch'):
            self._valid_epoch = self.current_epoch
        else:
            self._valid_epoch = self.current_epoch
        # initialize hidden state for each epoch
        self.init_hidden()

    def on_validation_epoch_end(self) -> None:
        nni.report_intermediate_result(
            self.trainer.callback_metrics["val_loss"].item())

    def teardown(self, stage) -> None:
        if stage == "fit":
            nni.report_final_result(
                self.trainer.callback_metrics["val_loss"].item())

    def init_hidden(self) -> None:

        if hasattr(self.model, 'init_hidden'):
            self._hidden = self.model.init_hidden(params["batch_size"])
            # put hidden state on GPU if available and use_gpu is True
            self._hidden = tuple([item.to(self._device) for item in self._hidden]) if isinstance(
                self._hidden, (list, tuple)) else self._hidden.to(self._device)
        else:
            self._hidden = None


if __name__ == "__main__":

    # init search space
    params = search_space
    # get next hyper-parameter
    optimized_params = nni.get_next_parameter()
    params.update(optimized_params)

    # logger
    logger = TensorBoardLogger(log_dir, name="pl_logs")
    # gpu_setting
    accelerator_config = {"accelerator": "gpu", "devices": torch.cuda.device_count(
    )} if use_gpu else {"accelerator": "cpu"}

    trainer = Trainer(
        logger=logger,
        # early stopping when loss is nan
        callbacks=[EarlyStopping(
            monitor="val_loss", mode="min", check_finite=True)],
        max_epochs=params["num_epochs"],
        **accelerator_config,
    )
