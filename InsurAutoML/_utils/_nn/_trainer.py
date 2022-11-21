"""
File Name: _trainer.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_utils/_nn/_trainer.py
File Created: Saturday, 19th November 2022 6:30:24 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Sunday, 20th November 2022 3:04:27 pm
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

from __future__ import annotations

from typing import Callable, Union, Tuple
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from nni.retiarii.evaluator.pytorch import DataLoader
# from torch.utils.data import DataLoader
try:
    import pytorch_lightning as pl
except ImportError:
    pl = None

logger = logging.getLogger(__name__)

##########################################################################
# Pytorch trainer


class TorchTrainer:

    """Pytorch trainer for neural networks."""

    def __init__(
        self,
        model: Union[str, nn.Module],
        optimizer: Union[str, optim.Optimizer],
        criterion: Union[str, nn.Module],
        save: bool = True,
        path: str = "tmp/model.pth",
        checkpoint: bool = True,
        use_gpu=True,
    ) -> None:

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.save = save
        self.path = path
        self.checkpoint = checkpoint

        # if use_gpu but no gpu available, use cpu but log a warning
        logger.warn("GPU is not available. Using CPU instead.")

        # get device
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() and use_gpu else "cpu")

        # put model to device
        self.model.to(self._device)

    def train(
            self,
            dataloader: DataLoader,
            n_epochs: int = 10,
            progressbar: bool = True) -> None:

        # if show progress bar, use tqdm to wrap dataloader
        if progressbar:
            from tqdm import tqdm
            dataloader = tqdm(dataloader, position=0, unit="batch")

        for epoch in range(n_epochs):
            for idx, batch in enumerate(dataloader):
                # get input and target
                # in case there are multiple inputs, last one is target
                input, target = batch[:-1], batch[-1]
                # put input and target to device
                input, target = input.to(self._device), target.to(self._device)

                # backward pass
                self.optimizer.zero_grad()
                # forward pass
                output = self.model(*input)
                # calculate loss
                loss = self.criterion(output, target)

                # back-propagation and update weights
                loss.backward()
                self.optimizer.step()

            # if checkpoint, save model
            if self.checkpoint:
                torch.save(self.model.state_dict(), self.path)

        # if save, save model dict
        if self.save:
            torch.save(self.model.state_dict(), self.path)

    def fit(self, dataloader: DataLoader, n_epochs: int = 10,
            progressbar: bool = True) -> Union[np.ndarray, torch.Tensor]:
        self.train(dataloader, n_epochs, progressbar)

        return self

    def predict(self, dataloader: DataLoader, to_numpy=True):

        output = []

        for idx, batch in enumerate(dataloader):
            # get input and target
            # in case there are multiple inputs, last one is target
            input, target = batch[:-1], batch[-1]
            # put input and target to device
            input, target = input.to(self._device), target.to(self._device)

            # forward pass
            with torch.no_grad():
                output.append(self.model(*input))

        # concat output list
        output = torch.cat(output, dim=0)

        # if to_numpy, convert to numpy array
        if to_numpy:
            return output.detach().cpu().numpy()
        # else, return a cpu tensor
        else:
            return output.detach().cpu()

##########################################################################
# Pytorch lightning trainer


class plModule(pl.LightningModule):

    def __init__(self, model: Union[str, nn.Module],
                 optimizer: Union[str, optim.Optimizer],
                 criterion: Union[str, nn.Module],
                 save: bool = True,
                 path: str = "tmp/model.pth",
                 checkpoint: bool = True,
                 use_gpu=True,
                 ) -> None:
        super(plModule, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.save = save
        self.path = path
        self.checkpoint = checkpoint

        # if use_gpu but no gpu available, use cpu but log a warning
        logger.warn("GPU is not available. Using CPU instead.")

        # get device
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() and use_gpu else "cpu")

        # put model to device
        self.model.to(self._device)

    def configure_optimizers(self):
        return self.optimizer

    def training_step(self, batch: Tuple, batch_idx: int) -> pl.TrainResult:
        input, target = batch[:-1], batch[-1]
        output = self.model(*input)
        return pl.TrainResult(loss=self.criterion(output, target))

    def validation_step(self, batch: Tuple, batch_idx: int) -> pl.EvalResult:
        input, target = batch[:-1], batch[-1]
        with torch.no_grad():
            output = self.model(*input)

        return pl.EvalResult(loss=self.criterion(output, target))


class plTrainer(pl.Trainer):

    def __init__(self,) -> None:
        super(plTrainer, self).__init__()
