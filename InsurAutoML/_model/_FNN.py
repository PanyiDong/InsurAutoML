"""
File Name: _FNN.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_model/_FNN.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 14th November 2022 8:14:58 pm
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

from typing import Union
import warnings
import numpy as np
import pandas as pd

from InsurAutoML._utils._data import assign_classes

# check if pytorch exists
# if exists, import pytorch
import importlib

pytorch_spec = importlib.util.find_spec("torch")
if pytorch_spec is not None:
    import torch
    from torch import nn
    from torch import optim
    from torch.utils.data import TensorDataset, DataLoader

####################################################################################################
# Feed Forward Neural Network models

####################################################################################################
# Multi-Layer Perceptron Model
# 1. MLP_Model, forward phase
# 2. MLP_Base, base training/evaluation phase
# 3. MLP_Classifier, MLP specified for classification tasks
# 4. MLP_Regressor, MLP specified for regression tasks
class MLP_Model(nn.Module):

    """
    Flexible Multi-Layer Perceptron model

    Parameters
    ----------
    input_size: input shape, for tabular data, input_size equals number of features

    hidden_layer: number of hidden layers

    hidden_size: number of neurons in each hidden layer

    output_size: output shape, for classification, output_size equals number of classes;
    for regression, output_size equals 1

    softmax: if True, add softmax function (for classification), default is False

    activation: activation functions, default: "ReLU"
    support activation ["ReLU", "Tanh", "Sigmoid"]
    """

    def __init__(
        self,
        input_size: int,
        hidden_layer: int,
        hidden_size: int,
        output_size: int,
        softmax: bool = False,
        activation: str = "ReLU",
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_layer = hidden_layer
        self.hidden_size = hidden_size
        self.softmax = softmax
        self.output_size = output_size

        # specify activation function
        if activation in ["ReLU", "Tanh", "Sigmoid"]:
            self.activation = getattr(nn, activation)()
        # if activation == "ReLU":
        #     self.activation = nn.ReLU()
        # elif activation == "Tanh":
        #     self.activation = nn.Tanh()
        # elif activation == "Sigmoid":
        #     self.activation = nn.Sigmoid()
        else:
            raise ValueError(
                "Activation function not supported, please choose from [ReLU, Tanh, Sigmoid]"
            )

        self.forward_model = []  # sequential model

        # at first layer, from input layer to first hidden layer
        # add activation function
        self.forward_model.append(nn.Linear(self.input_size, self.hidden_size))
        self.forward_model.append(self.activation)

        # in the middle layer, from previous hidden layer to next hidden layer
        for _ in range(self.hidden_layer):
            self.forward_model.append(nn.Linear(self.hidden_size, self.hidden_size))
            self.forward_model.append(self.activation)

        # at last layer, from last hidden layer to output layer
        # no activation function
        self.forward_model.append(nn.Linear(self.hidden_size, self.output_size))

        # if softmax is True, add softmax function
        if self.softmax:
            self.forward_model.append(nn.Softmax())

        self.forward_model = nn.Sequential(*self.forward_model)

    def forward(self, X: torch.Tensor) -> torch.Tensor:

        return self.forward_model(X)


# Multi-Layer Perceptron base model fit/predict (training/evaluation)
class MLP_Base:

    """
    Multi-Layer Perceptron base model

    Parameters
    ----------
    input_size: input shape, for tabular data, input_size equals number of features

    hidden_layer: number of hidden layers

    hidden_size: number of neurons in each hidden layer

    output_size: output shape, for classification, output_size equals number of classes;
    for regression, output_size equals 1

    softmax: if True, add softmax function (for classification), default is False

    activation: activation functions, default: "ReLU"
    support activation ["ReLU", "Tanh", "Sigmoid"]

    learning_rate: learning rate, default: None

    optimizer: optimizer, default: "Adam"
    support optimizer ["Adam", "SGD"]

    criteria: criteria, default: "MSE"
    support criteria ["MSE", "CrossEntropy", "MAE"]

    batch_size: batch size, default: 32

    num_epochs: number of epochs, default: 20

    is_cuda: whether to use cuda, default: True
    """

    def __init__(
        self,
        input_size: int,
        hidden_layer: int,
        hidden_size: int,
        output_size: int,
        softmax: bool = False,
        activation: str = "ReLU",
        learning_rate: float = None,
        optimizer: str = "Adam",
        criteria: str = "MSE",
        batch_size: int = 32,
        num_epochs: int = 20,
        is_cuda: bool = True,
        seed: int = 1,
    ) -> None:
        self.input_size = input_size
        self.hidden_layer = hidden_layer
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.softmax = softmax
        self.activation = activation
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.criteria = criteria
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.is_cuda = is_cuda
        self.seed = seed

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> MLP_Base:

        # set seed
        torch.manual_seed(self.seed)

        # use cuda if detect GPU and is_cuda is True
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.is_cuda else "cpu"
        )

        # if try cuda and no cuda available, raise warning
        if self.is_cuda and str(self.device) == "cpu":
            warnings.warn("No GPU detected, use CPU for training.")

        # load model
        self.model = MLP_Model(
            input_size=self.input_size,
            hidden_layer=self.hidden_layer,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            softmax=self.softmax,
            activation=self.activation,
        ).to(self.device)

        # specify optimizer
        if self.optimizer == "Adam":
            lr = 0.001 if self.learning_rate is None else self.learning_rate
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif self.optimizer == "SGD":
            lr = 0.1 if self.learning_rate is None else self.learning_rate
            optimizer = optim.SGD(self.model.parameters(), lr=lr)

        # specify loss function
        if self.criteria == "MSE":
            criteria = nn.MSELoss()
        elif self.criteria == "MAE":
            criteria = nn.L1Loss()
        elif self.criteria == "CrossEntropy":
            criteria = nn.CrossEntropyLoss()
        elif self.criteria == "NegativeLogLikelihood":
            criteria = nn.NLLLoss()
        else:
            raise ValueError("Not recognized criteria: {}.".format(self.criteria))

        # load dataset to TensorDataset
        train_tensor = TensorDataset(X, y)
        # load dataset to DataLoader
        train_loader = DataLoader(
            train_tensor, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        # train model
        for epoch in range(self.num_epochs):

            for batch_idx, (data, target) in enumerate(train_loader):
                # load batch to device
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)  # forward
                loss = criteria(output, target)  # calculate loss
                loss.backward()  # backpropagation
                optimizer.step()  # update parameters

        self._fitted = True

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:

        # load test dataset to DataLoader
        if isinstance(X, pd.DataFrame):
            test_tensor = TensorDataset(torch.as_tensor(X.values, dtype=torch.float32))
        else:
            test_tensor = TensorDataset(torch.as_tensor(X, dtype=torch.float32))

        test_loader = DataLoader(test_tensor, batch_size=len(test_tensor))

        # predict
        for batch_idx, [data] in enumerate(test_loader):

            with torch.no_grad():
                results = self.model(data.to(self.device))

        return results.cpu().numpy()  # make prediction to numpy array


# Multi-Layer Perceptron classifier
class MLP_Classifier(MLP_Base):

    """
    Multi-Layer Perceptron classification model

    Parameters
    ----------
    hidden_layer: number of hidden layers

    hidden_size: number of neurons in each hidden layer

    softmax: if True, add softmax function (for classification), default is True

    activation: activation functions, default: "ReLU"
    support activation ["ReLU", "Tanh", "Sigmoid"]

    learning_rate: learning rate, default: None

    optimizer: optimizer, default: "Adam"
    support optimizer ["Adam", "SGD"]

    criteria: criteria, default: "CrossEntropy"
    support criteria ["CrossEntropy"]

    batch_size: batch size, default: 32

    num_epochs: number of epochs, default: 20

    is_cuda: whether to use cuda, default: True
    """

    def __init__(
        self,
        hidden_layer: int,
        hidden_size: int,
        softmax: bool = True,
        activation: str = "ReLU",
        learning_rate: float = None,
        optimizer: str = "Adam",
        criteria: str = "CrossEntropy",
        batch_size: int = 32,
        num_epochs: int = 20,
        is_cuda: bool = True,
        seed: int = 1,
    ) -> None:
        self.hidden_layer = hidden_layer
        self.hidden_size = hidden_size
        self.softmax = softmax
        self.activation = activation
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.criteria = criteria
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.is_cuda = is_cuda
        self.seed = seed

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> MLP_Classifier:

        self.input_size = X.shape[1]  # number of features as input size
        self.output_size = len(pd.unique(y))  # unique classes as output size

        # make sure losses are classification type
        if self.criteria not in ["CrossEntropy", "NegativeLogLikelihood"]:
            raise ValueError("Loss must be CrossEntropy!")

        super().__init__(
            input_size=self.input_size,
            hidden_layer=self.hidden_layer,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            softmax=self.softmax,
            activation=self.activation,
            learning_rate=self.learning_rate,
            optimizer=self.optimizer,
            criteria=self.criteria,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            is_cuda=self.is_cuda,
            seed=self.seed,
        )

        # convert to tensors
        X = torch.as_tensor(
            X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float
        )
        y = torch.as_tensor(
            y.values if isinstance(y, pd.DataFrame) else y, dtype=torch.long
        )

        self._fitted = True

        return super().fit(X, y)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:

        # need to wrap predict function to convert output format
        return assign_classes(super().predict(X))

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:

        # not need to use argmax to select the one class
        # but to return full probability
        return super().predict(X)


# Multi-Layer Perceptron regressor
class MLP_Regressor(MLP_Base):

    """
    Multi-Layer Perceptron regression model

    Parameters
    ----------
    hidden_layer: number of hidden layers

    hidden_size: number of neurons in each hidden layer

    softmax: if True, add softmax function (for classification), default is False

    activation: activation functions, default: "ReLU"
    support activation ["ReLU", "Tanh", "Sigmoid"]

    learning_rate: learning rate, default: None

    optimizer: optimizer, default: "Adam"
    support optimizer ["Adam", "SGD"]

    criteria: criteria, default: "MSE"
    support criteria ["MSE", "MAE"]

    batch_size: batch size, default: 32

    num_epochs: number of epochs, default: 20

    is_cuda: whether to use cuda, default: True
    """

    def __init__(
        self,
        hidden_layer: int,
        hidden_size: int,
        softmax: bool = False,
        activation: str = "ReLU",
        learning_rate: float = None,
        optimizer: str = "Adam",
        criteria: str = "MSE",
        batch_size: int = 32,
        num_epochs: int = 20,
        is_cuda: bool = True,
        seed: int = 1,
    ) -> None:
        self.hidden_layer = hidden_layer
        self.hidden_size = hidden_size
        self.softmax = softmax
        self.activation = activation
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.criteria = criteria
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.is_cuda = is_cuda
        self.seed = seed

        self._fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> MLP_Regressor:

        self.input_size = X.shape[1]  # number of features as input size
        self.output_size = 1  # output size is 1 (regression purpose)

        # make sure losses are regression type
        if self.criteria not in ["MSE", "MAE"]:
            raise ValueError("Loss must be MSE or MAE!")

        super().__init__(
            input_size=self.input_size,
            hidden_layer=self.hidden_layer,
            hidden_size=self.hidden_size,
            softmax=self.softmax,
            output_size=self.output_size,
            activation=self.activation,
            learning_rate=self.learning_rate,
            optimizer=self.optimizer,
            criteria=self.criteria,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            is_cuda=self.is_cuda,
            seed=self.seed,
        )

        # convert to tensors
        X = torch.as_tensor(
            X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float
        )
        y = torch.as_tensor(
            y.values if isinstance(y, pd.DataFrame) else y, dtype=torch.float
        )

        self._fitted = True

        return super().fit(X, y)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:

        return super().predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:

        return NotImplementedError("predict_proba is not implemented for regression.")
