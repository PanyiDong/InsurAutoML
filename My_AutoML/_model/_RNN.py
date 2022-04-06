"""
File: _RNN.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Relative Path: /My_AutoML/_model/_RNN.py
File Created: Tuesday, 5th April 2022 11:46:25 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Tuesday, 5th April 2022 11:48:21 pm
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

import numpy as np
import pandas as pd

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
# RNN/LSTM/GRU models supported


class RNN_Model(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_size,
        hidden_size,
        output_size,
        n_layers=1,
        RNN_unit="RNN",
        activation="Sigmoid",
        dropout=0.2,
    ):
        super().__init__()

        # hidden size and hidden layers
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # select RNN unit from ["RNN", "LSTM", "GRU"]
        if RNN_unit == "RNN":
            self.rnn = nn.RNN(embedding_size, hidden_size, n_layers, batch_first=True)
        elif RNN_unit == "LSTM":
            self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers, batch_first=True)
        elif RNN_unit == "GRU":
            self.rnn = nn.GRU(embedding_size, hidden_size, n_layers, batch_first=True)

        # linear connection after RNN layers
        self.hidden2tag = nn.Linear(hidden_size, output_size)

        # activation function
        if activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "Tanh":
            self.activation == nn.Tanh()
        elif activation == "Sigmoid":
            self.activation = nn.Sigmoid()

        self.dropout = nn.Dropout(p=dropout)  # dropout layer
        self.softmax = nn.LogSoftmax(dim=1)  # softmax layer

    def forward(self, input, hidden):

        embeds = self.embedding(input)
        rnn_out, (rnn_hiddne, rnn_cell) = self.rnn(embeds, hidden)
        tag_space = self.hidden2tag(rnn_out)
        tag_space = self.dropout(self.activation(tag_space))
        tag_scores = self.softmax(tag_space)[:, -1, :]

        return tag_scores, (rnn_hiddne, rnn_cell)

    def init_hidden(self, batch_size):
        h0 = torch.zeros((self.n_layers, batch_size, self.hidden_size))
        c0 = torch.zeros((self.n_layers, batch_size, self.hidden_size))

        return (h0, c0)


class RNN_Classifier(RNN_Model):
    def __init__(
        self,
        embedding_size=512,
        hidden_size=256,
        n_layers=1,
        RNN_unit="RNN",
        activation="Sigmoid",
        dropout=0.2,
        learning_rate=None,
        optimizer="Adam",
        criteria="CrossEntropy",
        batch_size=32,
        num_epochs=20,
        is_cuda=True,
        seed=1,
    ):
        # model parameters
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
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

    def fit(self, X, y):

        # set seed
        torch.manual_seed(self.seed)

        # use cuda if detect GPU and is_cuda is True
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.is_cuda else "cpu"
        )

        # make sure RNN unit is supported
        if self.RNN_unit not in ["RNN", "LSTM", "GRU"]:
            raise ValueError("RNN unit must be RNN, LSTM or GRU!")

        self.output_size = len(pd.unique(y))  # unique classes as output size

        # load model
        self.model = RNN_Model(
            vocab_size=len(X.vocab),
            embedding_size=self.embedding_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            n_layers=self.n_layers,
            RNN_unit=self.RNN_unit,
            activation=self.activation,
            dropout=self.dropout,
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
        else:
            raise ValueError("Not recognized criteria: {}.".format(self.criteria))

        # load data to DataLoader
        if isinstance(X, pd.DataFrame) or isinstance(y, pd.DataFrame):
            train_tensor = TensorDataset(
                torch.as_tensor(X.values, dtype=torch.float32),
                torch.as_tensor(y.values, dtype=torch.float32),
            )
        else:
            train_tensor = TensorDataset(
                torch.as_tensor(X, dtype=torch.float32),
                torch.as_tensor(y, dtype=torch.float32),
            )

        train_loader = DataLoader(
            train_tensor, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        # training process
        for _ in range(self.num_epochs):
            # initialize hidden state for each batch
            h = self.model.init_hidden(self.batch_size).to(self.device)
            for batch_idx, (data, target) in enumerate(train_loader):

                # put data, target to device
                data = data.to(self.device)
                target = target.to(self.device)

                self.model.zero_grad()
                output, h = self.model(data, h)  # forward step
                loss = criteria(output, target)  # calculate loss
                loss.backward()  # backpropagation
                optimizer.step()  # update parameters

        return self

    def transform(self, X, y=None):

        # load data to DataLoader
        if isinstance(X, pd.DataFrame) or isinstance(y, pd.DataFrame):
            test_tensor = TensorDataset(
                torch.as_tensor(X.values, dtype=torch.float32),
            )
        else:
            test_tensor = TensorDataset(
                torch.as_tensor(X, dtype=torch.float32),
            )

        test_loader = DataLoader(test_tensor, batch_size=len(test_tensor))

        # initialize hidden state
        h = self.model.init_hidden(len(test_tensor))

        # predict
        for batch_idx, [data] in enumerate(test_loader):

            with torch.no_grad():
                results = self.model(data.to(self.device))

        return results.cpu().numpy()  # return prediction to cpu
