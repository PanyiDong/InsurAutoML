import numpy as np
import pandas as pd
import torch
from torch import nn

class LNN(nn.Module) :
    
    def __init__(
        self,
        inputSize,
        outputSize
    ):
        super().__init__()
        self.linear = nn.Linear(inputSize, outputSize)

        if torch.cuda.is_available() :
            super().cuda()

    def forward(self, X) :
        out = self.linear(X)
        return out

X = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
y = np.array([1., 2., 3., 4.])

learning_rate = 0.03,
max_iter = 1000

# initialize forward steps
model = LNN(inputSize = 4, outputSize = 1)
        
# define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# converting inputs to pytorch tensors
if torch.cuda.is_available() :
    inputs = torch.from_numpy(X).cuda()
    labels = torch.from_numpy(y).cuda()
else :
    inputs = torch.from_numpy(X)
    labels = torch.from_numpy(y)

for _iter in range(max_iter) :

    # clear gradient buffers
    optimizer.zero_grad()

    # calculate output using inputs
    outputs = model(inputs)

    # get loss from outputs
    loss = criterion(outputs, labels)

    # get gradient for parameters
    loss.backward()

    # update parameters
    optimizer.step()


