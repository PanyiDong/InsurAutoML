"""
File: _model.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Relative Path: /_model.py
File Created: Friday, 25th February 2022 6:13:42 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 4th April 2022 11:10:29 pm
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
import autosklearn.pipeline.components.classification
import autosklearn.pipeline.components.regression

# check if pytorch exists
# if exists, import pytorch
import importlib

from sklearn.utils import shuffle

tensorflow_spec = importlib.util.find_spec("torch")
if tensorflow_spec is not None:
    import torch
    from torch import nn
    from torch import optim
    from torch.utils.data import TensorDataset, DataLoader

####################################################################################################
# self-defined models

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

    output_size: output shape, for classification, output_size equals number of classes; for regression, output_size equals 1

    activation: activation functions, default: "ReLU"
    support activation ["ReLU", "Tanh", "Sigmoid"]
    """

    def __init__(
        self,
        input_size,
        hidden_layer,
        hidden_size,
        output_size,
        activation,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_layer = hidden_layer
        self.hidden_size = hidden_size
        self.output_size = output_size

        # specify activation function
        if activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "Tanh":
            self.activation = nn.Tanh()
        elif activation == "Sigmoid":
            self.activation = nn.Sigmoid()

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

        self.forward_model = nn.Sequential(*self.forward_model)

    def forward(self, X):

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

    output_size: output shape, for classification, output_size equals number of classes; for regression, output_size equals 1

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
        input_size,
        hidden_layer,
        hidden_size,
        output_size,
        activation="ReLU",
        learning_rate=None,
        optimizer="Adam",
        criteria="MSE",
        batch_size=32,
        num_epochs=20,
        is_cuda=True,
    ):
        self.input_size = input_size
        self.hidden_layer = hidden_layer
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.criteria = criteria
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.is_cuda = is_cuda

    def fit(self, X, y):

        # use cuda if detect GPU and is_cuda is True
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.is_cuda else "cpu"
        )

        # load model
        self.model = MLP_Model(
            input_size=self.input_size,
            hidden_layer=self.hidden_layer,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
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
        elif self.criteria == "CrossEntropy":
            criteria = nn.CrossEntropyLoss()
        elif self.criteria == "MAE":
            criteria = nn.L1Loss()
        else:
            raise ValueError("Not recognized criteria: {}.".format(self.criteria))

        # load dataset to DataLoader
        if isinstance(X, pd.DataFrame):
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

        return self

    def transform(self, X, y=None):

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
        hidden_layer,
        hidden_size,
        activation="ReLU",
        learning_rate=None,
        optimizer="Adam",
        criteria="CrossEntropy",
        batch_size=32,
        num_epochs=20,
        is_cuda=True,
    ):
        self.hidden_layer = hidden_layer
        self.hidden_size = hidden_size
        self.activation = activation
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.criteria = criteria
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.is_cuda = is_cuda

    def fit(self, X, y):

        self.input_size = X.shape[1]  # number of features as input size
        self.output_size = len(pd.unique(y))  # unique classes as output size

        # make sure losses are classification type
        if self.criteria not in ["CrossEntropy"]:
            raise ValueError("Loss must be CrossEntropy!")

        super().__init__(
            input_size=self.input_size,
            hidden_layer=self.hidden_layer,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            activation=self.activation,
            learning_rate=self.learning_rate,
            optimizer=self.optimizer,
            criteria=self.criteria,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            is_cuda=self.is_cuda,
        )

        return super().fit(X, y)

    def transform(self, X, y=None):

        return super().transform(X, y)


# Multi-Layer Perceptron regressor
class MLP_Regressor(MLP_Base):

    """
    Multi-Layer Perceptron regression model

    Parameters
    ----------
    hidden_layer: number of hidden layers

    hidden_size: number of neurons in each hidden layer

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
        hidden_layer,
        hidden_size,
        activation="ReLU",
        learning_rate=None,
        optimizer="Adam",
        criteria="MSE",
        batch_size=32,
        num_epochs=20,
        is_cuda=True,
    ):
        self.hidden_layer = hidden_layer
        self.hidden_size = hidden_size
        self.activation = activation
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.criteria = criteria
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.is_cuda = is_cuda

    def fit(self, X, y):

        self.input_size = X.shape[1]  # number of features as input size
        self.output_size = 1  # output size is 1 (regression purpose)

        # make sure losses are regression type
        if self.criteria not in ["MSE", "MAE"]:
            raise ValueError("Loss must be MSE or MAE!")

        super().__init__(
            input_size=self.input_size,
            hidden_layer=self.hidden_layer,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            activation=self.activation,
            learning_rate=self.learning_rate,
            optimizer=self.optimizer,
            criteria=self.criteria,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            is_cuda=self.is_cuda,
        )

        return super().fit(X, y)

    def transform(self, X, y=None):

        return super().transform(X, y)

####################################################################################################
# classifiers

classifiers = {
    # classification models from autosklearn
    "AdaboostClassifier": autosklearn.pipeline.components.classification.adaboost.AdaboostClassifier,
    "BernoulliNB": autosklearn.pipeline.components.classification.bernoulli_nb.BernoulliNB,
    "DecisionTree": autosklearn.pipeline.components.classification.decision_tree.DecisionTree,
    "ExtraTreesClassifier": autosklearn.pipeline.components.classification.extra_trees.ExtraTreesClassifier,
    "GaussianNB": autosklearn.pipeline.components.classification.gaussian_nb.GaussianNB,
    "GradientBoostingClassifier": autosklearn.pipeline.components.classification.gradient_boosting.GradientBoostingClassifier,
    "KNearestNeighborsClassifier": autosklearn.pipeline.components.classification.k_nearest_neighbors.KNearestNeighborsClassifier,
    "LDA": autosklearn.pipeline.components.classification.lda.LDA,
    "LibLinear_SVC": autosklearn.pipeline.components.classification.liblinear_svc.LibLinear_SVC,
    "LibSVM_SVC": autosklearn.pipeline.components.classification.libsvm_svc.LibSVM_SVC,
    "MLPClassifier": autosklearn.pipeline.components.classification.mlp.MLPClassifier,
    "MultinomialNB": autosklearn.pipeline.components.classification.multinomial_nb.MultinomialNB,
    "PassiveAggressive": autosklearn.pipeline.components.classification.passive_aggressive.PassiveAggressive,
    "QDA": autosklearn.pipeline.components.classification.qda.QDA,
    "RandomForest": autosklearn.pipeline.components.classification.random_forest.RandomForest,
    "SGD": autosklearn.pipeline.components.classification.sgd.SGD,
    # self-defined models
    "MLP_Classifier": MLP_Classifier,
}

# regressors
regressors = {
    # regression models from autosklearn
    "AdaboostRegressor": autosklearn.pipeline.components.regression.adaboost.AdaboostRegressor,
    "ARDRegression": autosklearn.pipeline.components.regression.ard_regression.ARDRegression,
    "DecisionTree": autosklearn.pipeline.components.regression.decision_tree.DecisionTree,
    "ExtraTreesRegressor": autosklearn.pipeline.components.regression.extra_trees.ExtraTreesRegressor,
    "GaussianProcess": autosklearn.pipeline.components.regression.gaussian_process.GaussianProcess,
    "GradientBoosting": autosklearn.pipeline.components.regression.gradient_boosting.GradientBoosting,
    "KNearestNeighborsRegressor": autosklearn.pipeline.components.regression.k_nearest_neighbors.KNearestNeighborsRegressor,
    "LibLinear_SVR": autosklearn.pipeline.components.regression.liblinear_svr.LibLinear_SVR,
    "LibSVM_SVR": autosklearn.pipeline.components.regression.libsvm_svr.LibSVM_SVR,
    "MLPRegressor": autosklearn.pipeline.components.regression.mlp.MLPRegressor,
    "RandomForest": autosklearn.pipeline.components.regression.random_forest.RandomForest,
    "SGD": autosklearn.pipeline.components.regression.sgd.SGD,
    # self-defined models
    "MLP_Regressor": MLP_Regressor,
}  # LibSVM_SVR, MLP and SGD have problems of requiring inverse_transform of StandardScaler while having 1D array
# https://github.com/automl/auto-sklearn/issues/1297
# problem solved