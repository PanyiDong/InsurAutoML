from cProfile import label
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer

import warnings

from ._utils import type_of_task

device = 'cuda' if torch.cuda.is_available() else 'cpu' # check if gpu available

#############################################################################################

#############################################################################################
# Linear Neural Network (LNN)
class LNN :

    '''
    inputSize:
    
    outputSize:

    criteria: loss function, default = 'MSE'

    optimizer: optimizer used for backpropagation, default = 'SGD'

    learning_rate: learning rate for gradient descent, default = 0.03

    max_iter: maximum iterations allowed, default = 1000
    '''

    def __init__(
        self, 
        inputSize = 1, 
        outputSize = 1,
        criteria = 'MSE',
        optimizer = 'SGD',
        learning_rate = 0.03,
        max_iter = 100
    ):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.criteria = criteria
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit(self, X, y) :

        if self.inputSize != X.shape[1] :
            self.inputSize = X.shape[1]

        # initialize forward steps
        self.net = nn.Linear(self.inputSize, self.outputSize)

        # converting inputs to pytorch tensors
        if isinstance(X, np.ndarray) :
            inputs = torch.from_numpy(X).to(device)
        elif isinstance(X, pd.DataFrame) :
            inputs = torch.from_numpy(X.values).to(device)

        if isinstance(y, np.ndarray) :
            labels = torch.from_numpy(y).reshape(-1, 1).to(device)
        elif isinstance(y, pd.DataFrame) :
            labels = torch.from_numpy(y.values).reshape(-1, 1).to(device)

        # define the loss function and optimizer
        if self.criteria == 'MSE' :
            criterion = nn.MSELoss()

        if self.optimizer == 'SGD' :
            optimizer = torch.optim.SGD(self.net.parameters(), lr = self.learning_rate)

        Losses = [] # record losses for early stopping or 

        for _ in range(self.max_iter) :
    
            # clear gradient buffers
            optimizer.zero_grad()

            # calculate output using inputs
            outputs = self.net(inputs)

            # get loss from outputs
            loss = criterion(outputs, labels)
            Losses.append(loss.item())

            # get gradient for parameters
            loss.backward()

            # update parameters
            optimizer.step()

    def predict(self, X) :

        # converting inputs to pytorch tensors
        if isinstance(X, np.ndarray) :
            inputs = torch.from_numpy(X).to(device)
        elif isinstance(X, pd.DataFrame) :
            inputs = torch.from_numpy(X.values).to(device)

        with torch.no_grad() :

            return self.net(inputs).detach().cpu().numpy() # convert to numpy


#############################################################################################
# Multilayer Perceptrons (MLP)
class MLP :
    
    '''
    inputSize:
    
    outputSize:

    criteria: loss function, default = 'MSE'

    optimizer: optimizer used for backpropagation, default = 'SGD'

    learning_rate: learning rate for gradient descent, default = 0.03

    max_iter: maximum iterations allowed, default = 1000
    '''

    def __init__(
        self, 
        inputSize = 1, 
        hiddenSize = 5,
        outputSize = 1,
        criteria = 'MSE',
        optimizer = 'SGD',
        learning_rate = 0.03,
        max_iter = 1000
    ):
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.criteria = criteria
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit(self, X, y) :

        if self.inputSize != X.shape[1] :
            self.inputSize = X.shape[1]

        # initialize forward steps
        self.net =  nn.Sequential(
            nn.Linear(self.inputSize, self.hiddenSize),
            nn.ReLU(),
            nn.Linear(self.hiddenSize, self.hiddenSize),
            nn.ReLU(),
            nn.Linear(self.hiddenSize, self.outputSize)
        )

        # converting inputs to pytorch tensors
        if isinstance(X, np.ndarray) :
            inputs = torch.from_numpy(X).to(device)
        elif isinstance(X, pd.DataFrame) :
            inputs = torch.from_numpy(X.values).to(device)

        if isinstance(y, np.ndarray) :
            labels = torch.from_numpy(y).reshape(-1, 1).to(device)
        elif isinstance(y, pd.DataFrame) :
            labels = torch.from_numpy(y.values).reshape(-1, 1).to(device)

        # define the loss function and optimizer
        if self.criteria == 'MSE' :
            criterion = nn.MSELoss()
        elif self.criteria == 'CrossEntropy' :
            criterion = nn.CrossEntropyLoss()

        if self.optimizer == 'SGD' :
            optimizer = torch.optim.SGD(self.net.parameters(), lr = self.learning_rate)
        elif self.optimizer == 'Adam' :
            optimizer = torch.optim.Adam(self.net.parameters(), lr = self.learning_rate)

        Losses = [] # record losses for early stopping or 

        for _ in range(self.max_iter) :
    
            # clear gradient buffers
            optimizer.zero_grad()

            # calculate output using inputs
            outputs = self.net(inputs)

            # get loss from outputs
            loss = criterion(outputs, labels)
            Losses.append(loss.item())

            # get gradient for parameters
            loss.backward()

            # update parameters
            optimizer.step()

    def predict(self, X) :

        # converting inputs to pytorch tensors
        if isinstance(X, np.ndarray) :
            inputs = torch.from_numpy(X).to(device)
        elif isinstance(X, pd.DataFrame) :
            inputs = torch.from_numpy(X.values).to(device)

        with torch.no_grad() :

            return self.net(inputs).detach().cpu().numpy() # convert to numpy

#############################################################################################