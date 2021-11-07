from cProfile import label
import numpy as np
import pandas as pd
import torch
import torch.optim
from torch import nn

import warnings

from My_AutoML._utils import type_of_task

device = 'cuda' if torch.cuda.is_available() else 'cpu' # check if gpu available

#############################################################################################
# Linear Neural Network
# forward step
class _LNN(nn.Module) :
    
    def __init__(
        self,
        inputSize,
        outputSize
    ):
        super().__init__()
        self.linear = nn.Linear(inputSize, outputSize)

        if torch.cuda.is_available() : # convert to cuda
            super().cuda()

    def forward(self, X) :
        out = self.linear(X)
        return out

# optimization step
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

        # initialize forward steps
        self._model = _LNN(inputSize = self.inputSize, outputSize = self.outputSize).to(device)

        if self.inputSize != X.shape[1] :
            self.inputSize = X.shape[1]

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
            optimizer = torch.optim.SGD(self._model.parameters(), lr = self.learning_rate)

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
