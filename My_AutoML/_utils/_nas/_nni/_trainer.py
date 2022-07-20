"""
File: _trainer.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Last Version: 0.2.1
Relative Path: /My_AutoML/_utils/_nas/_nni/_trainer.py
File Created: Tuesday, 19th July 2022 2:06:36 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Tuesday, 19th July 2022 9:22:03 pm
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

import json
import torch
import nni
from nni.retiarii import fixed_arch
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig

from My_AutoML._utils._nas._nni._utils import get_strategy, get_search_space
from My_AutoML._utils._tensor import CustomTensorDataset


class Trainer(object):
    def __init__(
        self,
        search_space="MLP",
        max_evals=10,
        timeout=3600 * 24,
        optimizer="Adam",
        optimizer_lr=1e-3,
        lr_scheduler="None",
        criterion="CrossEntropyLoss",
        evaluator="base",
        search_strategy="Random",
        batch_size=32,
        num_epoch=10,
    ):
        self.search_space = get_search_space(search_space)
        self.max_evals = max_evals
        self.timeout = timeout
        self.optimizer = optimizer
        self.optimizer_lr = optimizer_lr
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.search_strategy = get_strategy(search_strategy)
        self.batch_size = batch_size
        self.num_epoch = num_epoch

    @nni.trace
    def _prep_loader(data, batch_size=32, mode="train"):

        from torch.utils.data import DataLoader

        if mode == "train":
            return DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
        elif mode == "test":
            return DataLoader(data, batch_size=batch_size)

    @nni.trace
    def _prep_dataset(X, y):

        # format to tensor
        if not isinstance(X, torch.Tensor) or not isinstance(y, torch.Tensor):
            X = torch.as_tensor(X)
            y = torch.as_tensor(y)

        return CustomTensorDataset(X, y, format="tuple")

    def train(self, train, test):

        # get evaluator
        # base form evaluator
        if self.evaluator == "base":

            from ._evaluator import get_evaluator

            self.evaluator = get_evaluator(
                trainset=train,
                testset=test,
                batchSize=self.batch_size,
                optimizer=self.optimizer,
                criterion=self.criterion,
                num_epoch=self.num_epoch,
            )
        # pytorch-lightning form evaluator
        elif self.evaluator == "pl":
            from ._evaulatorPL import get_evaluator

            self.evaluator = get_evaluator(
                model=self.search_space,
                train_set=train,
                test_set=test,
                batchSize=self.batch_size,
                criterion=self.criterion,
                optimizer=self.optimizer,
                optimizer_lr=self.optimizer_lr,
                lr_scheduler=self.lr_scheduler,
                num_epochs=self.num_epoch,
            )

        exp = RetiariiExperiment(
            self.search_space,
            self.evaluator,
            [],
            self.search_strategy,
        )
        exp_config = RetiariiExeConfig("local")
        exp_config.experiment_name = "nas_test"
        # exp_config.execution_engine = "oneshot"
        exp_config.max_trial_number = self.max_evals
        exp_config.max_trial_duration = self.timeout
        exp_config.trial_concurrency = 1
        exp_config.trial_gpu_number = torch.cuda.device_count()
        exp_config.training_service.use_active_gpu = True
        exp.run(exp_config, wait_completion=True)
        exp.export_data()
        exp.stop()

        for model_dict in exp.export_top_models(formatter="dict"):
            with open("optimal_architecture.json", "w") as outfile:
                json.dump(model_dict, outfile)

    def optimal_loading(self):

        with fixed_arch("optimal_architecture.json"):
            return self.search_space
