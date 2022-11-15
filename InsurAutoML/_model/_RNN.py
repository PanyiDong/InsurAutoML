"""
File Name: _RNN.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_model/_RNN.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 14th November 2022 8:15:30 pm
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

from typing import Union, Tuple
import warnings
import numpy as np
import pandas as pd

from InsurAutoML._utils._data import assign_classes
from InsurAutoML._utils._tensor import repackage_hidden

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
# RNN models
# Common RNN/LSTM/GRU models supported


class RNN_Net(nn.Module):

    """
    Recurrent Neural Network (RNN) model structure

    Parameters
    ----------
    input_size: input size

    hidden_size: dimension of the hidden layer

    output_size: output size of the model, for classification tasks, this is the number of classes;
    for regression tasks, this is 1

    n_layers: number of hidden layers, default = 1

    RNN_unit: RNN unit type, default = "RNN"
    support type ["RNN", "LSTM", "GRU"]

    activation: activation function, default = "Sigmoid"
    support type ["ReLU", "Tanh", "Sigmoid"]

    dropout: dropout rate in fully-connected layers, default = 0.2

    device: string or torch.device cpu/gpu for the training
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        n_layers: int = 1,
        RNN_unit: str = "RNN",
        activation: str = "Sigmoid",
        dropout: float = 0.2,
        use_softmax: bool = False,
        device: Union[str, torch.device] = torch.device("cuda"),
    ) -> None:
        super().__init__()

        # assign device
        self.device = torch.device(device) if isinstance(device, str) else device

        # hidden size and hidden layers
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # embedding layer
        # for tabular, no need for embedding layer
        # self.embedding = nn.Embedding(input_size, input_size)

        # select RNN unit from ["RNN", "LSTM", "GRU"]
        self.RNN_unit = RNN_unit
        if self.RNN_unit in ["RNN", "LSTM", "GRU"]:
            self.rnn = getattr(nn, self.RNN_unit)(
                input_size, hidden_size, n_layers, batch_first=True
            )
        # if self.RNN_unit == "RNN":
        #     self.rnn = nn.RNN(input_size, hidden_size, n_layers, batch_first=True)
        # elif self.RNN_unit == "LSTM":
        #     self.rnn = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
        # elif self.RNN_unit == "GRU":
        #     self.rnn = nn.GRU(input_size, hidden_size, n_layers, batch_first=True)
        else:
            raise TypeError("Not recognizing RNN unit type!")

        # linear connection after RNN layers
        self.hidden2tag = nn.Linear(hidden_size, output_size)

        # activation function
        if activation in ["ReLU", "Tanh", "Sigmoid"]:
            self.activation = getattr(nn, activation)()
        # if activation == "ReLU":
        #     self.activation = nn.ReLU()
        # elif activation == "Tanh":
        #     self.activation = nn.Tanh()
        # elif activation == "Sigmoid":
        #     self.activation = nn.Sigmoid()
        else:
            raise TypeError("Not recognizing activation function!")

        self.dropout = nn.Dropout(p=dropout)  # dropout layer
        self.use_softmax = use_softmax
        if self.use_softmax:
            self.softmax_layer = nn.LogSoftmax(dim=1)  # softmax layer

    def forward(
        self, input: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:

        # embeds = self.embedding(input)
        if self.RNN_unit in ["LSTM"]:
            rnn_out, (rnn_hidden, rnn_cell) = self.rnn(input, hidden)
        elif self.RNN_unit in ["RNN", "GRU"]:
            rnn_out, rnn_hidden = self.rnn(input, hidden)
        tag_space = self.hidden2tag(rnn_out)
        tag_scores = self.dropout(self.activation(tag_space))
        if self.use_softmax:
            tag_scores = self.softmax_layer(
                tag_scores
            )  # [:, -1, :]  # keep full output here

        if self.RNN_unit in ["LSTM"]:
            return tag_scores, (rnn_hidden, rnn_cell)
        elif self.RNN_unit in ["RNN", "GRU"]:
            return tag_scores, rnn_hidden

    def init_hidden(
        self, batch_size: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        h0 = torch.zeros((self.n_layers, batch_size, self.hidden_size)).to(self.device)

        # if LSTM, need (h0, c0)
        if self.RNN_unit in ["LSTM"]:
            c0 = torch.zeros((self.n_layers, batch_size, self.hidden_size)).to(
                self.device
            )

            return (h0, c0)
        # if not (RNN, GRU), need h0
        if self.RNN_unit in ["RNN", "GRU"]:

            return h0


class RNN_Base:

    """
    Recurrent Neural Network (RNN) models for classification tasks, training/evaluation

    Parameters
    ----------
    input_size: input size, default = 1

    hidden_size: dimension of the hidden layer, default = 256

    output_size: output size, default = 1
    will be assigned to number of classes

    n_layers: number of hidden layers, default = 1

    RNN_unit: RNN unit type, default = "RNN"
    support type ["RNN", "LSTM", "GRU"]

    activation: activation function, default = "Sigmoid"
    support type ["ReLU", "Tanh", "Sigmoid"]

    dropout: dropout rate in fully-connected layers, default = 0.2

    learning_rate: learning rate for the optimizer, default = None

    optimizer: optimizer for training, default = "Adam"
    support type ["Adam", "SGD"]

    criteria: loss function, default = "CrossEntropy"
    support type ["CrossEntropy", "NegativeLogLikelihood"]

    batch_size: batch size for training, default = 32

    num_epochs: number of epochs for training, default = 20

    is_cuda: whether to use GPU for training, default = True

    seed: random seed, default = 1
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 256,
        output_size: int = 1,
        n_layers: int = 1,
        RNN_unit: str = "RNN",
        activation: str = "Sigmoid",
        dropout: float = 0.2,
        learning_rate: float = None,
        optimizer: str = "Adam",
        criteria: str = "CrossEntropy",
        batch_size: int = 32,
        num_epochs: int = 20,
        use_softmax: bool = False,
        is_cuda: bool = True,
        seed: int = 1,
    ) -> None:
        # model parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.RNN_unit = RNN_unit
        self.activation = activation
        self.dropout = dropout
        self.use_softmax = use_softmax

        # training parameters
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
    ) -> RNN_Base:

        # set seed
        torch.manual_seed(self.seed)

        # use cuda if detect GPU and is_cuda is True
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.is_cuda else "cpu"
        )

        # if try cuda and no cuda available, raise warning
        if self.is_cuda and str(self.device) == "cpu":
            warnings.warn("No GPU detected, use CPU for training.")

        # make sure RNN unit is supported
        if self.RNN_unit not in ["RNN", "LSTM", "GRU"]:
            raise ValueError("RNN unit must be RNN, LSTM or GRU!")

        # self.output_size = len(pd.unique(y))  # unique classes as output size

        # load model
        self.model = RNN_Net(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            n_layers=self.n_layers,
            RNN_unit=self.RNN_unit,
            activation=self.activation,
            dropout=self.dropout,
            use_softmax=self.use_softmax,
            device=self.device,
        ).to(self.device)

        # specify optimizer
        if self.optimizer == "Adam":
            lr = 0.001 if self.learning_rate is None else self.learning_rate
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif self.optimizer == "SGD":
            lr = 0.1 if self.learning_rate is None else self.learning_rate
            optimizer = optim.SGD(self.model.parameters(), lr=lr)

        # specify loss function
        if self.criteria == "CrossEntropy":
            criteria = nn.CrossEntropyLoss()
        elif self.criteria == "MSE":
            criteria = nn.MSELoss()
        elif self.criteria == "MAE":
            criteria = nn.L1Loss()
        elif self.criteria == "NegativeLogLikelihood":
            criteria = nn.NLLLoss()
        else:
            raise ValueError("Not recognized criteria: {}.".format(self.criteria))

        # load data to DataLoader
        train_tensor = TensorDataset(X, y)

        train_loader = DataLoader(
            train_tensor, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        # training process
        for _ in range(self.num_epochs):
            # initialize hidden state for each batch
            h = self.model.init_hidden(self.batch_size)
            for batch_idx, (data, target) in enumerate(train_loader):

                # put data, target to device
                data = data.to(self.device)
                target = target.to(self.device)

                # only get the values, no need for gradient
                h = repackage_hidden(h)

                self.model.zero_grad()
                output, h = self.model(data, h)  # forward step
                # only use last output for classification (at last time T)
                loss = criteria(output[:, -1, :], target)  # calculate loss
                loss.backward()  # backpropagation
                optimizer.step()  # update parameters

        self._fitted = True

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:

        # convert to tensor
        X = torch.as_tensor(
            X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float
        )
        X.unsqueeze_(-1)  # expand to 3d tensor

        # load data to TensorDataset
        test_tensor = TensorDataset(X)
        # load data to DataLoader
        test_loader = DataLoader(test_tensor, batch_size=len(test_tensor))

        # initialize hidden state
        h = self.model.init_hidden(len(test_tensor))

        # predict
        for batch_idx, [data] in enumerate(test_loader):

            with torch.no_grad():
                results, h = self.model(data.to(self.device), h)

        return results[:, -1, :].cpu().numpy()  # return prediction to cpu


class RNN_Classifier(RNN_Base):

    """
    RNN Classifier

    Parameters
    ----------
    input_size: input size, default = 1

    hidden_size: dimension of the hidden layer, default = 256

    output_size: output size, default = 1
    will be assigned to number of classes

    n_layers: number of hidden layers, default = 1

    RNN_unit: RNN unit type, default = "RNN"
    support type ["RNN", "LSTM", "GRU"]

    activation: activation function, default = "Sigmoid"
    support type ["ReLU", "Tanh", "Sigmoid"]

    dropout: dropout rate in fully-connected layers, default = 0.2

    learning_rate: learning rate for the optimizer, default = None

    optimizer: optimizer for training, default = "Adam"
    support type ["Adam", "SGD"]

    criteria: loss function, default = "CrossEntropy"
    support type ["CrossEntropy", "NegativeLogLikelihood"]

    batch_size: batch size for training, default = 32

    num_epochs: number of epochs for training, default = 20

    is_cuda: whether to use GPU for training, default = True

    seed: random seed, default = 1
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 256,
        output_size: int = 1,
        n_layers: int = 1,
        RNN_unit: str = "RNN",
        activation: str = "Sigmoid",
        dropout: float = 0.2,
        learning_rate: float = None,
        optimizer: str = "Adam",
        criteria: str = "CrossEntropy",
        batch_size: int = 32,
        num_epochs: int = 20,
        is_cuda: bool = True,
        seed: int = 1,
    ) -> None:
        # model parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.RNN_unit = RNN_unit
        self.activation = activation
        self.dropout = dropout

        # training parameters
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.criteria = criteria
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.is_cuda = is_cuda
        self.seed = seed

        self._fitted = False

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> RNN_Classifier:

        # get unique classes
        self.output_size = len(pd.unique(y))

        # convert data to tensor
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(
                X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float
            )
            X.unsqueeze_(-1)  # expand to 3d tensor
        if not isinstance(y, torch.Tensor):
            y = torch.as_tensor(
                y.values if isinstance(y, pd.DataFrame) else y, dtype=torch.long
            )

        super().__init__(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            n_layers=self.n_layers,
            RNN_unit=self.RNN_unit,
            activation=self.activation,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
            optimizer=self.optimizer,
            criteria=self.criteria,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            use_softmax=True,
            is_cuda=self.is_cuda,
            seed=self.seed,
        )

        self._fitted = True

        return super().fit(X, y)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:

        # need to assign prediction to classes
        return assign_classes(super().predict(X))

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:

        # not need to use argmax to select the one class
        # but to return full probability
        return super().predict(X)


class RNN_Regressor(RNN_Base):

    """
    RNN Classifier

    Parameters
    ----------
    input_size: input size, default = 1

    hidden_size: dimension of the hidden layer, default = 256

    output_size: output size, default = 1
    will be assigned to 1

    n_layers: number of hidden layers, default = 1

    RNN_unit: RNN unit type, default = "RNN"
    support type ["RNN", "LSTM", "GRU"]

    activation: activation function, default = "Sigmoid"
    support type ["ReLU"]

    dropout: dropout rate in fully-connected layers, default = 0.2

    learning_rate: learning rate for the optimizer, default = None

    optimizer: optimizer for training, default = "Adam"
    support type ["Adam", "SGD"]

    criteria: loss function, default = "MSE"
    support type ["MSE", "MAE"]

    batch_size: batch size for training, default = 32

    num_epochs: number of epochs for training, default = 20

    is_cuda: whether to use GPU for training, default = True

    seed: random seed, default = 1
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 256,
        output_size: int = 1,
        n_layers: int = 1,
        RNN_unit: str = "RNN",
        activation: str = "Sigmoid",
        dropout: float = 0.2,
        learning_rate: float = None,
        optimizer: str = "Adam",
        criteria: str = "MSE",
        batch_size: int = 32,
        num_epochs: int = 20,
        is_cuda: bool = True,
        seed: int = 1,
    ) -> None:
        # model parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.RNN_unit = RNN_unit
        self.activation = activation
        self.dropout = dropout

        # training parameters
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.criteria = criteria
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.is_cuda = is_cuda
        self.seed = seed

        self._fitted = False

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> RNN_Regressor:

        # get unique classes
        self.output_size = 1

        # convert data to tensor
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(
                X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float
            )
            X.unsqueeze_(-1)  # expand to 3d tensor
        if not isinstance(y, torch.Tensor):
            y = torch.as_tensor(
                y.values if isinstance(y, pd.DataFrame) else y, dtype=torch.float
            )

        super().__init__(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            n_layers=self.n_layers,
            RNN_unit=self.RNN_unit,
            activation=self.activation,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
            optimizer=self.optimizer,
            criteria=self.criteria,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            use_softmax=False,
            is_cuda=self.is_cuda,
            seed=self.seed,
        )

        self._fitted = True

        return super().fit(X, y)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:

        return super().predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:

        raise NotImplementedError("predict_proba is not implemented for regression.")
