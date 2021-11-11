from typing import ForwardRef
import numpy as np
import pandas as pd
from pandas.core.algorithms import isin
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
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

# convert numpy array to pytorch tensor
def to_tensor(X, y, batch_size = 10) :

    if isinstance(X, pd.DataFrame) :
        X = X.values
    if isinstance(y, pd.DataFrame) :
        y = y.values

    inputs = torch.tensor(X).to(device)
    labels = torch.tensor(y).to(device)
    dataset = TensorDataset(inputs, labels)

    return DataLoader(dataset, batch_size = batch_size, shuffle = True)

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

# convolutional layer
# 2D Cross-correlation operation
def corr2d(X, K) :
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]) :
        for j in range(Y.shape[1]) :
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()

    return Y

class Conv2D(nn.Module) :

    def __init__(self, kernel_size) :
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size)) # random weight
        self.bias = nn.Parameter(torch.zeros(1)) # zero bias

    def forward(self, X) :
        return corr2d(X, self.weight) + self.bias

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
# Convolutional Neural Network (CNN)

# Ideas: Translation Invariance and Locality
# 1. Translation Invariance: Network should respond similarily to patches
# 2. Locality: Network should focus on local regions

# Padding: create surrounding empty values to increase the inputsize so reduction will be smaller 
# or even retain the inputsize, the boundaries will be preserved
# Stride: skip intermediate locations to improve efficiency or the purpose of downsample

# Max Pooling, Average Pooling

# LeNet-5
# Layers: Two Convolutional blocks, follow by three fully connected linear layers
# each Convolutional block consists of Convolutional layer, Activation function, AveragePool layer
class LeNet(nn.Module) :

    def __init__(
        self,
        Conv_inChannel = (1, 6),
        Conv_outChannel = (6, 16),
        Conv_kernel = (5, 5),
        Conv_padding = (2, 0),
        Conv_stride = (1, 1),
        Pool_kernel = (2, 2),
        Pool_padding = (0, 0),
        Pool_stride = (2, 2),
        denseSize = (120, 84, 10),
    ) :
        super().__init__()
        self.Con2d1 = nn.Conv2d(
            Conv_inChannel[0], Conv_outChannel[0], kernel_size = Conv_kernel[0],
            padding = Conv_padding[0], stride = Conv_stride[0]
        )
        self.Con2d2 = nn.Conv2d(
            Conv_inChannel[1], Conv_outChannel[1], kernel_size = Conv_kernel[1], 
            padding = Conv_padding[1], stride = Conv_stride[1]
        )
        self.AvgPool2d1 = nn.AvgPool2d(
            kernel_size = Pool_kernel[0], padding = Pool_padding[0], stride = Pool_stride[0]
        )
        self.AvgPool2d2 = nn.AvgPool2d(
            kernel_size = Pool_kernel[1], padding = Pool_padding[1], stride = Pool_stride[1]
        )
        self.Sigmoid = nn.Sigmoid()

        self.Conv_block = nn.Sequential(
            self.Con2d1, self.Sigmoid, self.AvgPool2d1, # first Convolution block
            self.Con2d2, self.Sigmoid, self.AvgPool2d2, # second Convolution block
            nn.Flatten() # flatten for linear layer
        )

        self.denseSize = denseSize # mark the size of linear, self.Linear1 requires change of size
        self.Linear1 = nn.Linear(120, denseSize[0])
        self.Linear2 = nn.Linear(denseSize[0], denseSize[1])
        self.Linear3 = nn.Linear(denseSize[1], denseSize[2])

    def forward(self, X) :
        
        # Convolution block
        output = self.Conv_block(X)

        # three fully-connected linear layers
        self.Linear1 = nn.Linear(output.size(dim = 1), self.denseSize[0])
        output = self.Linear1(output)
        output = self.Linear2(output)
        output = self.Linear3(output)

        return output

# Deep Convolutional Neural Networks
# AlexNet

# Networks using Blocks
# VGG (Visual Geometry Group) Network

# NiN (Network in Network)

# Network with Parallel Concatenations
# GoogLeNet

# ResNet (Residual Networks)

# DenseNet (Densely Connected Networks)

#############################################################################################
# Recurrent Neural Network (RNN)
