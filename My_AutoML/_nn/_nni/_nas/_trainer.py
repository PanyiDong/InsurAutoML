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
Last Modified: Tuesday, 11th October 2022 8:42:02 pm
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
import json
import shutil
import numpy as np
import pandas as pd
import torch
import nni
from nni.retiarii import fixed_arch
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig

from My_AutoML._utils._tensor import CustomTensorDataset
from ._utils import get_strategy, get_search_space
from ._buildModel import build_model


NNI_VERBOSE = [
    # "fatal",
    "error",
    "warning",
    "info",
    "debug",
    "trace",
]


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
        task_name="NAS",
        temp_directory="tmp",
        verbose=0,
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
        self.task_name = task_name
        self.temp_directory = temp_directory
        self.verbose = verbose

    def train(self, train, test, inputSize, outputSize):

        # must use function to wrap the dataset for serialization
        @nni.trace
        def _prep_dataset(X, y):
            # format to tensor
            if not isinstance(X, torch.Tensor) or not isinstance(y, torch.Tensor):
                # dataframe cannot be directly converted to tensor
                if isinstance(X, pd.DataFrame) or isinstance(y, pd.DataFrame):
                    X = torch.tensor(X.values, dtype=torch.float32)
                    y = torch.tensor(y.values)
                else:
                    X = torch.tensor(X, dtype=torch.float32)
                    y = torch.tensor(y)

            return CustomTensorDataset(X, y, format="tuple")

        # @nni.trace
        def _prep_loader(data, batch_size=32, mode="train"):

            from torch.utils.data import DataLoader

            if mode == "train":
                return DataLoader(
                    data, batch_size=batch_size, shuffle=True, drop_last=True
                )
            elif mode == "test":
                return DataLoader(data, batch_size=batch_size)

        # unpack data
        # (X_train, y_train), (X_test, y_test) = train, test
        # inputSize = X_train.shape[1]
        # outputSize = len(np.unique(y_train))

        # if folder existed, empty it
        if os.path.isdir(os.path.join(self.temp_directory, self.task_name)):
            shutil.rmtree(os.path.join(self.temp_directory, self.task_name))
        # if folder not exist, create it
        if not os.path.exists(os.path.join(self.temp_directory, self.task_name)):
            os.makedirs(os.path.join(self.temp_directory, self.task_name))

        # get evaluator
        # base form evaluator
        if self.evaluator == "base":

            from ._evaluator import get_evaluator

            self.evaluator = get_evaluator(
                # trainset=_prep_dataset(X_train, y_train),
                # testset=_prep_dataset(X_test, y_test),
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
                model=nni.trace(self.search_space)(inputSize, outputSize),
                # train_set=_prep_dataset(X_train, y_train),
                # test_set=_prep_dataset(X_test, y_test),
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
            nni.trace(self.search_space)(inputSize, outputSize),
            self.evaluator,
            [],
            self.search_strategy,
        )
        exp_config = RetiariiExeConfig("local")
        exp_config.experiment_name = self.task_name
        # exp_config.execution_engine = "oneshot"
        exp_config.max_trial_number = self.max_evals
        exp_config.max_trial_duration = self.timeout
        exp_config.trial_concurrency = 1
        exp_config.trial_gpu_number = torch.cuda.device_count()
        exp_config.training_service.use_active_gpu = True
        # exp_config.experiment_working_directory = os.path.join(
        #     self.temp_directory, self.task_name
        # )
        # exp_config.experiment_name = self.task_name
        exp_config.log_level = NNI_VERBOSE[self.verbose]
        exp.run(exp_config)
        exp.export_data()
        exp.stop()

        # save the best model architecture
        # initialize the best model
        for model_dict in exp.export_top_models(formatter="dict"):
            with open(
                os.path.join(
                    self.temp_directory, self.task_name, "optimal_architecture.json"
                ),
                "w",
            ) as outfile:
                # build model
                init_model = build_model(model_dict, inputSize, outputSize, "MLP")
                torch.save(
                    init_model,
                    os.path.join(
                        self.temp_directory, self.task_name, "init_optimal_model.pt"
                    ),
                )
                # save the model dict
                json.dump(model_dict, outfile)

    def optimal_loading(self):

        with fixed_arch("optimal_architecture.json"):
            return self.search_space
