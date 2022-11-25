"""
File Name: _trainer.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_experimental/_nn/_nni/_nas/_trainer.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Thursday, 24th November 2022 6:54:20 pm
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

from typing import Union, List, Dict, Any
import os
import json
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import nni
from nni.retiarii import fixed_arch
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig

from InsurAutoML._utils._tensor import SerialTensorDataset
from InsurAutoML._utils import type_of_task, train_test_split
from InsurAutoML._experimental._nn._nni._nas._utils import get_strategy, get_search_space
from InsurAutoML._experimental._nn._nni._nas._evaulatorPL import get_evaluator

DataLoader = nni.trace(DataLoader)

NNI_VERBOSE = [
    # "fatal",
    "error",
    "warning",
    "info",
    "debug",
    "trace",
]


class Trainer:

    def __init__(
        self,
        preprocessor=None,
        search_space="MLP",
        max_evals=10,
        timeout=3600 * 24,
        optimizer="Adam",
        optimizer_lr=1e-3,
        lr_scheduler="None",
        criterion=None,
        evaluator="base",
        search_strategy="Random",
        batch_size=32,
        num_epoch=10,
        valid_perc=0.15,
        task_name="NAS",
        temp_directory="tmp",
        verbose=0,
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
        self.evaluator = evaluator
        self.search_strategy = get_strategy(search_strategy)
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.valid_perc = valid_perc
        self.task_name = task_name
        self.temp_directory = temp_directory
        self.verbose = verbose

     # must use function to wrap the dataset for serialization
    @staticmethod
    def _prep_dataset(X, y):
        # format to tensor
        if not isinstance(
                X, torch.Tensor) or not isinstance(
                y, torch.Tensor):
            # dataframe cannot be directly converted to tensor
            if isinstance(X, pd.DataFrame) or isinstance(y, pd.DataFrame):
                X = torch.tensor(X.values, dtype=torch.float32)
                y = torch.tensor(y.values)
            else:
                X = torch.tensor(X, dtype=torch.float32)
                y = torch.tensor(y)

        return SerialTensorDataset(X, y)

    @staticmethod
    def _prep_loader(data, batch_size=32, mode="train"):

        if mode == "train":
            return DataLoader(
                data, batch_size=batch_size, shuffle=True  # , drop_last=True
            )
        elif mode == "test":
            return DataLoader(data, batch_size=batch_size)

    @staticmethod
    def _tensor_formatter(data: Any) -> torch.Tensor:

        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            return torch.tensor(data.values)
        elif isinstance(data, np.ndarray):
            return torch.tensor(data)
        elif isinstance(data, torch.Tensor):
            return data
        else:
            raise TypeError(
                "Unsupported data type for tensor formatter. Expected pd.DataFrame, np.ndarray, or torch.Tensor, but got %s." %
                type(data))

    def _data_formatter(self,
                        data: Union[List,
                                    Dict,
                                    Any]) -> Union[List[torch.Tensor],
                                                   Dict[str,
                                                        torch.Tensor],
                                                   torch.Tensor]:
        if isinstance(data, list):
            return [self._tensor_formatter(item) for item in data]
        if isinstance(data, dict):
            return {
                key: self._tensor_formatter(value) for key,
                value in data.items()}
        else:
            return self._tensor_formatter(data)

    def _setup_data(self, train, valid):

        # unpack data
        if valid is not None:
            (X_train, y_train), (X_valid, y_valid) = train, valid
        # if valid not provided, split from train set
        else:
            X_train, y_train = train
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_train, y_train, test_perc=self.valid_perc)
        # get task type
        task_type = type_of_task(y_train)

        # get default loss function
        if task_type in ["binary", "multiclass"] and self.criterion is None:
            self.criterion = "CrossEntropyLoss"
        elif task_type in ["integer", "continuous"] and self.criterion is None:
            self.criterion = "MSELoss"
        else:
            raise ValueError("Unrecognized task type %s." % task_type)

        # if preprocessor found, use it to preprocess data
        if self.preprocessor is not None:
            X_train = self.preprocessor.fit_transform(X_train)
            X_valid = self.preprocessor.transform(X_valid)

        # format data
        X_train, y_train = self._data_formatter(
            X_train), self._data_formatter(y_train)
        X_valid, y_valid = self._data_formatter(
            X_valid), self._data_formatter(y_valid)

        return X_train, y_train, X_valid, y_valid  # , task_type

    @staticmethod
    def _get_sizes(X_train, y_train, inputSize, outputSize):
        # get input and output Size
        # if RNN net, need to use vocab size as inputSize
        if isinstance(X_train, torch.Tensor):
            # if self.search_space.__name__ in ["RNNet", "LiteRNNet"]:
            #     inputSize = self.preprocessor.vocab_size if inputSize is None else inputSize
            # else:
            inputSize = X_train.size(
                dim=-1) if inputSize is None else inputSize
        # format list of tensor to tensor by order
        elif isinstance(X_train, list):
            # if self.search_space.__name__ in ["RNNet", "LiteRNNet"]:
            #     inputSize = {idx: _X_train.size(dim=-1) if idx > 0 else self.preprocessor.vocab_size for idx,
            #                  _X_train in enumerate(X_train)} if inputSize is None else inputSize
            # else:
            inputSize = {idx: _X_train.size(
                dim=-1) for idx, _X_train in enumerate(X_train)} if inputSize is None else inputSize
        elif isinstance(X_train, dict):
            # if self.search_space.__name__ in ["RNNet", "LiteRNNet"]:
            #     inputSize = {key: value.size(dim=-1) if key != "txt" else self.preprocessor.vocab_size for key,
            #                  value in X_train.items()} if inputSize is None else inputSize
            #     outputSize = len(np.unique(y_train)
            #                      ) if outputSize is None else outputSize
            # else:
            inputSize = {key: value.size(
                dim=-1) for key, value in X_train.items()} if inputSize is None else inputSize
        outputSize = len(np.unique(y_train)
                         ) if outputSize is None else outputSize

        return inputSize, outputSize

    # def train(self, train, test, inputSize, outputSize):
    def train(self, train, valid=None, inputSize=None, outputSize=None, **kwargs):

        # setup data
        X_train, y_train, X_valid, y_valid = self._setup_data(train, valid)

        # get inputSize, outputSize and vocabSize
        inputSize, outputSize = self._get_sizes(
            X_train, y_train, inputSize, outputSize)
        vocabSize = self.preprocessor.vocab_size if self.preprocessor is not None else None

        # if folder existed, empty it
        if os.path.isdir(os.path.join(self.temp_directory, self.task_name)):
            shutil.rmtree(os.path.join(self.temp_directory, self.task_name))
        # if folder not exist, create it
        if not os.path.exists(
                os.path.join(self.temp_directory, self.task_name)):
            os.makedirs(os.path.join(self.temp_directory, self.task_name))

        # get dataset
        if not os.path.exists(os.path.join(self.temp_directory, self.task_name, "train")):
            os.makedirs(os.path.join(
                self.temp_directory, self.task_name, "train"))
        if not os.path.exists(os.path.join(self.temp_directory, self.task_name, "valid")):
            os.makedirs(os.path.join(
                self.temp_directory, self.task_name, "valid"))
        trainset = SerialTensorDataset(X_train, y_train, path=os.path.join(
            self.temp_directory, self.task_name, "train"))
        validset = SerialTensorDataset(X_valid, y_valid, path=os.path.join(
            self.temp_directory, self.task_name, "valid"))

        # Update: Nov. 20, 2022
        # Force using pytorch_lightning evaluator
        # get evaluator
        # # base form evaluator
        # if self.evaluator == "base":

        #     from ._evaluator import get_evaluator

        #     self.evaluator = get_evaluator(
        #         trainset=trainset,
        #         testset=testset,
        #         # trainset=_prep_loader(train),
        #         # testset=_prep_loader(test),
        #         batchSize=self.batch_size,
        #         optimizer=self.optimizer,
        #         criterion=self.criterion,
        #         num_epoch=self.num_epoch,
        #     )

        # # pytorch-lightning form evaluator
        # elif self.evaluator == "pl":
        self.evaluator = get_evaluator(
            model=self.search_space(inputSize, outputSize, vocabSize),
            train_set=trainset,
            test_set=validset,
            # train_set=_prep_dataset(X_train, y_train),
            # test_set=_prep_dataset(X_test, y_test),
            batchSize=self.batch_size,
            # type_of_task=task_type,
            criterion=self.criterion,
            optimizer=self.optimizer,
            optimizer_lr=self.optimizer_lr,
            lr_scheduler=self.lr_scheduler,
            num_epochs=self.num_epoch,
            log_dir=os.path.join(self.temp_directory, self.task_name),
        )

        # setup experiment settings
        exp = RetiariiExperiment(
            self.search_space(inputSize, outputSize, vocabSize),
            self.evaluator,
            [],
            self.search_strategy,
        )
        # setup experiment config
        exp_config = RetiariiExeConfig("local")
        # experiment name
        exp_config.experiment_working_directory = os.path.join(
            self.temp_directory, self.task_name)
        exp_config.experiment_name = self.task_name
        # exp_config.execution_engine = "oneshot"
        # number of trials
        exp_config.max_trial_number = self.max_evals
        # time budget
        exp_config.max_experiment_duration = self.timeout
        exp_config.trial_concurrency = 1
        exp_config.trial_gpu_number = torch.cuda.device_count()
        exp_config.training_service.use_active_gpu = True
        # exp_config.experiment_working_directory = os.path.join(
        #     self.temp_directory, self.task_name
        # )
        exp_config.log_level = NNI_VERBOSE[self.verbose]
        exp.run(exp_config)
        exp.export_data()
        exp.stop()

        # save the best model architecture
        # initialize the best model
        for model_dict in exp.export_top_models(
                top_k=1, optimize_mode='maximize', formatter="dict"):
            # save the model dict
            with open(os.path.join(self.temp_directory, self.task_name, "optimal_architecture.json"), "w",) as outfile:
                json.dump(model_dict, outfile)
            # build init model for HPO
            init_model = self.build_model(inputSize, outputSize, vocabSize)
            # save the model dict
            torch.save(
                init_model.state_dict(),
                os.path.join(self.temp_directory,
                             self.task_name, "init_optimal_model.pth"),
            )

    def build_model(self, inputSize, outputSize, vocabSize):

        with fixed_arch(os.path.join(self.temp_directory, self.task_name, "optimal_architecture.json")):
            net = self.search_space(inputSize, outputSize, vocabSize)

        return net
