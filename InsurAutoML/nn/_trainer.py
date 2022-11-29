"""
File Name: _trainer.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/nn/_trainer.py
File Created: Saturday, 26th November 2022 12:26:39 am
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 28th November 2022 11:40:04 pm
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

import os
import shutil
import logging

from .nas import NasTrainer
from .utils.args import get_strategy, get_search_space

logger = logging.getLogger(__name__)


class AutoMultiModalTabular(NasTrainer):

    def __init__(
        self,
        preprocessor=None,
        search_space="MLPNet",
        max_evals=10,
        timeout=3600 * 24,
        optimizer="Adam",
        optimizer_lr=1e-3,
        lr_scheduler="None",
        criterion=None,
        # evaluator="base",
        search_strategy="Random",
        batch_size=32,
        num_epoch=10,
        valid_perc=0.15,
        task_name="MultiModal",
        temp_directory="tmp",
        verbose=0,
        use_gpu=True,
    ):
        self.preprocessor = preprocessor
        self.search_space = get_search_space(search_space)
        self.max_evals = max_evals
        self.timeout = timeout
        self.optimizer = optimizer
        self.optimizer_lr = optimizer_lr
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.optimizer = optimizer
        # self.evaluator = evaluator
        self.search_strategy = get_strategy(search_strategy)
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.valid_perc = valid_perc
        self.task_name = task_name
        self.temp_directory = temp_directory
        self.verbose = verbose
        self.use_gpu = use_gpu

        # if folder existed, empty it
        if os.path.isdir(os.path.join(self.temp_directory, self.task_name)):
            shutil.rmtree(os.path.join(self.temp_directory, self.task_name))
        # if folder not exist, create it
        if not os.path.exists(os.path.join(self.temp_directory, self.task_name)):
            os.makedirs(os.path.join(self.temp_directory, self.task_name))

        # initiate the nas trainer
        super(AutoMultiModalTabular, self).__init__(
            preprocessor=self.preprocessor,
            search_space=self.search_space,
            max_evals=self.max_evals,
            timeout=self.timeout,
            optimizer=self.optimizer,
            optimizer_lr=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            criterion=self.criterion,
            # evaluator="base",
            search_strategy=self.search_space,
            batch_size=self.batch_size,
            num_epoch=self.num_epoch,
            valid_perc=self.valid_perc,
            task_name=os.path.join(self.task_name, "NAS"),
            temp_directory=self.temp_directory,
            verbose=self.verbose,
            use_gpu=self.use_gpu,
        )

    def train(self, train, valid=None, inputSize=None, outputSize=None, **kwargs):

        # NAS training
        super(AutoMultiModalTabular, self).train(
            train, valid, inputSize, outputSize, **kwargs)
