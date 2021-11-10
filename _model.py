import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import warnings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # check if gpu available

#############################################################################################
# utils

# Parameters initialization
# perform apply to network
# def init_normal(m) :
#     if type(m) == nn.Linear :
#         nn.init.normal_(m.weight, mean = 0, std = 0.01) # weight parameters as normal variables
#         nn.init.zeros_(m.bias) # bias parameters as zeros

# def init_constant(m) :
#     if type(m) == nn.Linear :
#         nn.init.constant_(m.weight, 1) # weight parameters as constants
#         nn.init.zeros_(m.bias) # bias parameters as zeros

# def xavier(m) :
#     if type(m) == nn.Linear :
#         nn.init.xavier_uniform_(m.weight) # Xavier initializer

#############################################################################################
# Common layers

# Center the layer values, no parameters needed
class CenteredLayer(nn.Module) :

    def __init__(self) :
        super().__init__()

    def forward(self, X) :
        return X - X.mean()

# Linear layer
class LinearLayer(nn.Module) :

    def __init__(
        self,
        inputSize,
        outputSize,
    ) :
        super().__init__()
        self.weight = nn.Parameter(torch.randn(inputSize, outputSize))
        self.bias = nn.Parameter(torch.randn(outputSize, ))

    def forward(self, X) :

        output = torch.matmul(X, self.weight.data) + self.bias.data
        return output

#############################################################################################
# Linear Neural Network (LNN)
# forward step
class _LNN(nn.Module) :

    def __init__(
        self,
        inputSize,
        outputSize,
    ) :
        super().__init__()
        self.Linear = nn.Linear(inputSize, outputSize)

    def forward(self, X):

        output = self.Linear(X)
        return output
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
        self._model = _LNN(self.inputSize, self.outputSize).to(device)

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
            optimizer = torch.optim.SGD(self._model.parameters(), lr = self.learning_rate)
        elif self.optimizer == 'Adam' :
            optimizer = torch.optim.Adam(self._model.parameters(), lr = self.learning_rate)

        Losses = [] # record losses for early stopping or 

        for _ in range(self.max_iter) :
    
            # clear gradient buffers
            optimizer.zero_grad()

            # calculate output using inputs
            outputs = self._model(inputs)

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

            return self._model(inputs).detach().cpu().numpy() # convert to numpy


#############################################################################################
# Multilayer Perceptrons (MLP)
# forward step
class _MLP(nn.Module) :

    def __init__(
        self,
        inputSize, 
        hiddenSize,
        outputSize,
    ) :
        super().__init__()
        self.linear1 = nn.Linear(inputSize, hiddenSize)
        self.linear2 = nn.Linear(hiddenSize, hiddenSize)
        self.linear3 = nn.Linear(hiddenSize, outputSize)
        self.ReLU = nn.ReLU

    def farward(self, X) :
        
        output = self.linear1(X)
        output = self.ReLU(output)
        output = self.linear2(output)
        output = self.ReLU(output)
        output = self.linear3(output)

        return output

# optimization step
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
        self._model =  _MLP(self.inputSize, self.hiddenSize, self.outputSize).to(device)

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
            optimizer = torch.optim.SGD(self._model.parameters(), lr = self.learning_rate)
        elif self.optimizer == 'Adam' :
            optimizer = torch.optim.Adam(self._model.parameters(), lr = self.learning_rate)

        Losses = [] # record losses for early stopping or 

        for _ in range(self.max_iter) :
    
            # clear gradient buffers
            optimizer.zero_grad()

            # calculate output using inputs
            outputs = self._model(inputs)

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

            return self._model(inputs).detach().cpu().numpy() # convert to numpy

#############################################################################################