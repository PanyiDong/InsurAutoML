import collections
import numpy as np
import pandas as pd
from pandas.core.algorithms import isin
import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.utils.data import DataLoader, TensorDataset, Dataset
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
        
    torch_dataset = TensorDataset(inputs, labels)

    return DataLoader(torch_dataset, batch_size = batch_size, shuffle = True)

# count token frequencies
def count_corpus(tokens) :

    if len(tokens) == 0 or isinstance(tokens[0], list) :
        tokens = [token for line in tokens for token in line]

    return collections.Counter(tokens)

# create vocabulary for text tokens
class Vocab :

    def __init__(
        self,
        tokens = None,
        min_freq = 0,
        reserved_tokens = None
    ) :
        if tokens is None :
            tokens = []
        if reserved_tokens is None :
            reserved_tokens = []

        # Sort text tokens by frequencies
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key = lambda x: x[1], reserve = True)

        # store unknown tokens at index 0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {
            token: idx for idx, token in enumerate(self.idx_to_token)
        }
        for token, freq in self._token_freqs :
            if freq < min_freq :
                break
            if token not in self.token_to_idx :
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self) :
        return len(self.idx_to_token)
    
    def __getitem__(self, tokens) :
        if not isinstance(tokens, (list, tuple)) :
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices) :
        if not isinstance(indices, (list, tuple)) :
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self) :
        return 0

    @property
    def token_freqs(self) :
        return self._token_freqs

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

# Batch Normalization
# improve network stability
class BatchNorm(nn.Module) :

    '''
    num_features: number of outputs for a fully-connected layer/number of output channels for convolution layer

    num_dims: 2 for fully-connected layer/4 for convolution layer
    '''

    def __init__(
        self,
        num_features = 1,
        num_dims = 4
    ) :
        super().__init__()

        if num_dims == 2 :
            shape = (1, num_features)
        elif num_dims == 4 :
            shape = (1, num_features, 1, 1)
        else :
            raise ValueError('num_dims only accept 2 or 4, get {}.'.format(num_dims))
        
        # initialize scale/shift parameters
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

        # initialize moving average for prediction
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum) :

        if not torch.is_grad_enabled() : # at prediction stage, no need for batch calculation
            # calculate standardization using moving mean/variance
            X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
        else :
            assert len(X.shape) in (2, 4)
            if len(X.shape) == 2 : # for fully-connected layer
                mean = X.mean(dim = 0)
                var = ((X - mean) ** 2).mean(dim = 0)
            else : # for 2d convolution layer
                mean = X.mean(dim = (0, 2, 3), keepdim = True)
                var = ((X - mean) ** 2).mean(dim = (0, 2, 3), keepdim = True)
            
            # batch standardization
            # calculate standardization using batch mean/variance
            X_hat = (X - mean) / torch.sqrt(var + eps)

            # update mean/variance using moving average
            moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
            moving_var = momentum * moving_var + (1.0 - momentum) * var

        Y = gamma * X_hat + beta # scale and shift

        return Y, moving_mean.data, moving_var.data

    def forward(self, X) :

        # unify the device
        if self.moving_mean.device != X.device :
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)

        # update batch normalization
        Y, self.moving_mean, self.moving_var = self.batch_norm(
            X, self.gamma, self.beta, self.moving_mean, self.moving_var, eps = 1e-5, momentum = 0.9
        )

        return Y

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
        Conv_Channel = (1, 6, 16),
        Conv_kernel = (5, 5),
        Conv_padding = (2, 0),
        Conv_stride = (1, 1),
        Pool_kernel = (2, 2),
        Pool_padding = (0, 0),
        Pool_stride = (2, 2),
        LinearSize = (400, 120, 84, 10),
    ) :
        super().__init__()
        self.Con2d1 = nn.Conv2d(
            Conv_Channel[0], Conv_Channel[1], kernel_size = Conv_kernel[0],
            padding = Conv_padding[0], stride = Conv_stride[0], device = device
        )
        self.Con2d2 = nn.Conv2d(
            Conv_Channel[1], Conv_Channel[2], kernel_size = Conv_kernel[1], 
            padding = Conv_padding[1], stride = Conv_stride[1], device = device
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

        self.LinearSize = LinearSize # mark the size of linear, self.Linear1 requires change of size
        self.Linear1 = nn.Linear(LinearSize[0], LinearSize[1], device = device)
        self.Linear2 = nn.Linear(LinearSize[1], LinearSize[2], device = device)
        self.Linear3 = nn.Linear(LinearSize[2], LinearSize[3], device = device)

    def forward(self, X) :
        
        # Convolution block
        output = self.Conv_block(X.to(device))

        # three fully-connected linear layers
        self.Linear1 = nn.Linear(output.size(dim = 1), self.LinearSize[0], device = device)
        output = self.Linear1(output)
        output = self.Linear2(output)
        output = self.Linear3(output)

        return output

# Deep Convolutional Neural Networks
# Deeper layers version of CNN
# AlexNet
class AlexNet(nn.Module) :

    def __init__(
        self,
        Conv_Channel = (1, 96, 256, 384, 384, 256),
        Conv_kernel = (11, 5, 3, 3, 3),
        Conv_padding = (1, 2, 1, 1, 1),
        Conv_stride = (4, 1, 1, 1, 1),
        Pool_kernel = (3, 3, 3),
        Pool_padding = (0, 0, 0),
        Pool_stride = (2, 2, 2),
        LinearSize = (6400, 4096, 4096, 10),
        Dropout_p = (0.5, 0.5)
    ) :
        super().__init__()

        # five layers of Convolution
        self.Conv2d1 = nn.Conv2d(
            Conv_Channel[0], Conv_Channel[1], kernel_size = Conv_kernel[0],
            padding = Conv_padding[0], stride = Conv_stride[0], device = device
        )
        self.Conv2d2 = nn.Conv2d(
            Conv_Channel[1], Conv_Channel[2], kernel_size = Conv_kernel[1], 
            padding = Conv_padding[1], stride = Conv_stride[1], device = device
        )
        self.Conv2d3 = nn.Conv2d(
            Conv_Channel[2], Conv_Channel[3], kernel_size = Conv_kernel[2], 
            padding = Conv_padding[2], stride = Conv_stride[2], device = device
        )
        self.Conv2d4 = nn.Conv2d(
            Conv_Channel[3], Conv_Channel[4], kernel_size = Conv_kernel[3], 
            padding = Conv_padding[3], stride = Conv_stride[3], device = device
        )
        self.Conv2d5 = nn.Conv2d(
            Conv_Channel[4], Conv_Channel[5], kernel_size = Conv_kernel[4], 
            padding = Conv_padding[4], stride = Conv_stride[4], device = device
        )

        # two maxpool layer
        self.MaxPool2d1 = nn.MaxPool2d(
            kernel_size = Pool_kernel[0], padding = Pool_padding[0], stride = Pool_stride[0]
        )
        self.MaxPool2d2 = nn.MaxPool2d(
            kernel_size = Pool_kernel[1], padding = Pool_padding[1], stride = Pool_stride[1]
        )
        self.MaxPool2d3 = nn.MaxPool2d(
            kernel_size = Pool_kernel[2], padding = Pool_padding[2], stride = Pool_stride[2]
        )

        # activation function
        self.ReLU = nn.ReLU()

        # Convolution block
        self.Conv_block = nn.Sequential(
            self.Conv2d1, self.ReLU, self.MaxPool2d1,
            self.Conv2d2, self.ReLU, self.MaxPool2d2,
            self.Conv2d3, self.ReLU,
            self.Conv2d4, self.ReLU,
            self.Conv2d5, self.ReLU, self.MaxPool2d3,
            nn.Flatten() # flatten so linear layer can be incorporated
        )

        # fully-connected linear layers
        self.LinearSize = LinearSize
        self.Linear1 = nn.Linear(LinearSize[0], LinearSize[1], device = device)
        self.Linear2 = nn.Linear(LinearSize[1], LinearSize[2], device = device)
        self.Linear3 = nn.Linear(LinearSize[2], LinearSize[3], device = device)

        # Dropout to reduce network complexity
        self.Dropout1 = nn.Dropout(p = Dropout_p[0])
        self.Dropout2 = nn.Dropout(p = Dropout_p[1])

    def forward(self, X) :

        # Convolution block
        output = self.Conv_block(X.to(device))

        # Three fully-connected layers
        self.Linear1 = nn.Linear(output.size(dim = 1), self.LinearSize[1], device = device)
        self.Dense_block = nn.Sequential(
            self.Linear1, self.ReLU, self.Dropout1,
            self.Linear2, self.ReLU, self.Dropout2,
            self.Linear3
        )

        output = self.Dense_block(output)

        return output

# Networks using Blocks
# Blocks of Convolution layers
# VGG (Visual Geometry Group) Network
class VGG_11(nn.Module) :

    '''
    VGG-11: 8 convolutional layers (each consists numbers of convolution, activation function and one final max pooling) 
    and 3 fully-connected layers (Linear layers)
    
    Conv_num: sum of the tuple is total number of convolution layers, elements larger than 1 indicates repeating layer
    Conv_Channel tuple with size of (number of Conv_num + 1)
    Conv_kernel tuple with size of (number of Conv_num)
    Conv_padding tuple with size of (number of Conv_num)
    Conv_stride tuple with size of (number of Conv_num)

    Pool_kernel tuple with size of (number of Conv_num)
    Pool_padding tuple with size of (number of Conv_num)
    Pool_stride tuple with size of (number of Conv_num)
    '''

    def __init__(
        self,
        Conv_num = (1, 1, 2, 2, 2),
        Conv_Channel = (1, 64, 128, 256, 512, 512),
        Conv_kernel = (3, 3, 3, 3, 3),
        Conv_padding = (1, 1, 1, 1, 1),
        Conv_stride = (1, 1, 1, 1, 1),
        Pool_kernel = (2, 2, 2, 2, 2),
        Pool_padding = (0, 0, 0, 0, 0),
        Pool_stride = (2, 2, 2, 2, 2),
        LinearSize = (25088, 4096, 4096, 10),
        Dropout_p = (0.5, 0.5)
    ) :
        super().__init__()
        
        # check size of initialization
        if len(Conv_Channel) != len(Conv_num) + 1:
            raise ValueError(
                'Length of channels must be 1 larger than number of convolution layers, expect {}, get{}.'
                .format(len(Conv_num) + 1, len(Conv_Channel))
            )

        if len(Conv_kernel) != len(Conv_num) :
            raise ValueError(
                'Length of convolution kernels must be the same as the number of convolution layers, expect {}, get{}.'
                .format(len(Conv_num), len(Conv_kernel))
            )

        if len(Conv_padding) != len(Conv_num) :
            raise ValueError(
                'Length of convolution paddings must be the same as the number of convolution layers, expect {}, get{}.'
                .format(len(Conv_num), len(Conv_padding))
            )

        if len(Conv_stride) != len(Conv_num) :
            raise ValueError(
                'Length of convolution strides must be the same as the number of convolution layers, expect {}, get{}.'
                .format(len(Conv_num), len(Conv_stride))
            )

        if len(Pool_kernel) != len(Conv_num) :
            raise ValueError(
                'Length of pooling kernels must be the same as the number of convolution layers, expect {}, get{}.'
                .format(len(Conv_num), len(Pool_kernel))
            )

        if len(Pool_padding) != len(Conv_num) :
            raise ValueError(
                'Length of pooling paddings must be the same as the number of convolution layers, expect {}, get{}.'
                .format(len(Conv_num), len(Pool_padding))
            )

        if len(Pool_stride) != len(Conv_num) :
            raise ValueError(
                'Length of pooling strides must be the same as the number of convolution layers, expect {}, get{}.'
                .format(len(Conv_num), len(Pool_stride))
            )

        self.Conv_block = []
        self.ReLU = nn.ReLU() # activation function
        # create Convolution blocks
        for i in range(len(Conv_num)) :
            in_channel = Conv_Channel[i]
            out_channel = Conv_Channel[i + 1]
            for _ in range(Conv_num[i]) :
                self.Conv_block.append(
                    nn.Conv2d(in_channel, out_channel, kernel_size = Conv_kernel[i],
                    padding = Conv_padding[i], stride = Conv_stride[i], device = device)
                )
                self.Conv_block.append(self.ReLU)
                in_channel = out_channel
            self.Conv_block.append(
                nn.MaxPool2d(kernel_size = Pool_kernel[i], padding = Pool_padding[i], stride = Pool_stride[i])
            )

        self.Conv_block = nn.Sequential(*self.Conv_block, nn.Flatten()) # convert list of Convolution block to Sequential
        
        # fully-connected linear layers
        self.LinearSize = LinearSize
        self.Linear1 = nn.Linear(LinearSize[0], LinearSize[1], device = device)
        self.Linear2 = nn.Linear(LinearSize[1], LinearSize[2], device = device)
        self.Linear3 = nn.Linear(LinearSize[2], LinearSize[3], device = device)

        # Dropout to reduce network complexity
        self.Dropout1 = nn.Dropout(p = Dropout_p[0])
        self.Dropout2 = nn.Dropout(p = Dropout_p[1])

    def forward(self, X) :

        # Convolution block
        output = self.Conv_block(X.to(device))

        # Three fully-connected layers
        self.Linear1 = nn.Linear(output.size(dim = 1), self.LinearSize[1], device = device)
        self.Dense_block = nn.Sequential(
            self.Linear1, self.ReLU, self.Dropout1,
            self.Linear2, self.ReLU, self.Dropout2,
            self.Linear3
        )

        output = self.Dense_block(output)

        return output

# NiN (Network in Network)
# apply fully-connected layer on each pixel location
# in implementation, apply two layers 1 * 1 convolution layer with 
# activation function after convolutional layer
class NiN(nn.Module) :

    def __init__(
        self,
        Conv_channel = (1, 96, 256, 384, 10),
        Conv_kernel = (11, 5, 3, 3),
        Conv_padding = (0, 2, 1, 1),
        Conv_stride = (4, 1, 1, 1),
        Pool_kernel = (3, 3, 3),
        Pool_padding = (0, 0, 0),
        Pool_stride = (2, 2, 2),
        Dropout_p = 0.5
    ) :
        # check size of initialization
        if len(Conv_kernel) != len(Conv_channel) - 1:
            raise ValueError(
                'Length of convolution kernels must be the same as the number of convolution channels - 1, expect {}, get{}.'
                .format(len(Conv_channel) - 1, len(Conv_kernel))
            )

        if len(Conv_padding) != len(Conv_channel) - 1 :
            raise ValueError(
                'Length of convolution paddings must be the same as the number of convolution channels - 1, expect {}, get{}.'
                .format(len(Conv_channel) - 1, len(Conv_padding))
            )

        if len(Conv_stride) != len(Conv_channel) - 1 :
            raise ValueError(
                'Length of convolution strides must be the same as the number of convolution channels - 1, expect {}, get{}.'
                .format(len(Conv_channel) - 1, len(Conv_stride))
            )

        if len(Pool_kernel) != len(Conv_channel) - 2 :
            raise ValueError(
                'Length of pooling kernels must be the same as the number of convolution channels - 2, expect {}, get{}.'
                .format(len(Conv_channel) - 2, len(Pool_kernel))
            )

        if len(Pool_padding) != len(Conv_channel) - 2 :
            raise ValueError(
                'Length of pooling paddings must be the same as the number of convolution channels - 2, expect {}, get{}.'
                .format(len(Conv_channel) - 2, len(Pool_padding))
            )

        if len(Pool_stride) != len(Conv_channel) - 2 :
            raise ValueError(
                'Length of pooling strides must be the same as the number of convolution channels - 2, expect {}, get{}.'
                .format(len(Conv_channel) - 2, len(Pool_stride))
            )
        
        super().__init__()
        
        self.ReLU = nn.ReLU() # activation function
        self.Conv_block = []
        for i in range(len(Conv_channel) - 1) :
            self.Conv_block.append(
                nn.Conv2d(
                    Conv_channel[i], Conv_channel[i + 1], kernel_size = Conv_kernel[i], 
                    padding = Conv_padding[i], stride = Conv_stride[i], device = device
                )
            )
            self.Conv_block.append(self.ReLU)
            self.Conv_block.append(nn.Conv2d(
                Conv_channel[i + 1], Conv_channel[i + 1], kernel_size = 1, device = device
            ))
            self.Conv_block.append(self.ReLU)
            self.Conv_block.append(nn.Conv2d(
                Conv_channel[i + 1], Conv_channel[i + 1], kernel_size = 1, device = device
            ))
            self.Conv_block.append(self.ReLU)
            if i < len(Conv_channel) - 3 :
                self.Conv_block.append(nn.MaxPool2d(
                    Pool_kernel[i], padding = Pool_padding[i], stride = Pool_stride[i]
                ))
            elif i == len(Conv_channel) - 3 : 
                self.Conv_block.append(nn.MaxPool2d(
                    Pool_kernel[i], padding = Pool_padding[i], stride = Pool_stride[i]
                ))
                self.Conv_block.append(nn.Dropout(Dropout_p))
            else :
                self.Conv_block.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.net = nn.Sequential(*self.Conv_block, nn.Flatten())

    def forward(self, X) :

        return self.net(X.to(device))

# Network with Parallel Concatenations
# GoogLeNet
# Introduce the Inception Blocks, where multiple convolution layers are concatenated to produce the final output
# GooLeNet is built using 9 Inception Blocks and global average pooling

# ResNet (Residual Networks)

# DenseNet (Densely Connected Networks)

#############################################################################################
# Recurrent Neural Network (RNN)
class RNN(nn.Module):

    '''
    one RNN layer + fully-connected linear layer
    '''

    def __init__(
        self, 
        input_size, 
        output_size, 
        hidden_dim, 
        n_layers
    ):
        super(RNN, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        # Fully connected layer
        self.linear = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)
        
        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = nn.Flatten(out)
        out = self.linear(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):

        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden

# Early observations can have substantial impact on later observations, RNN layer can result in 
# vanishing or exploding gradients. To avoid the circumstances, modified models are required

# Gated Recurrent Units (GRU)
# introduce Reset Gate and Update Gate to select how much of previous state should be retained or 
# how much of new state should be copied
# Reset Gate captures short-term dependencies
# Update Gate captures long-term dependencies

# Long Short-Term Memory (LSTM)