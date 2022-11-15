"""
File Name: _CNN.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_model/_CNN.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 14th November 2022 8:14:48 pm
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

import importlib

pytorch_spec = importlib.util.find_spec("torch")
if pytorch_spec is not None:
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
    import torch.optim as optim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # common components in CNN architecture
# conv2d = nn.Conv2d(
#     in_channels=3,
#     out_channels=6,
#     kernel_size=5,
#     stride = 1,
#     padding = 0,
# )
# maxpool2d = nn.MaxPool2d(
#     kernel_size=2,
#     stride = None,
#     padding = 0,
# )

###################################################################################################################
# Classical CNN architecture

# LeNet5
class LeNet5(nn.Module):
    def __init__(
        self,
        inputSize=1,
        outputSize=10,
    ):
        super().__init__()

        self.Conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=inputSize, out_channels=6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.Flatten = nn.Flatten()

        self.dense_layer = nn.Sequential(
            nn.Linear(in_features=400, out_features=120),
            nn.Sigmoid(),
            nn.Linear(in_features=120, out_features=84),
            nn.Sigmoid(),
            nn.Linear(in_features=84, out_features=outputSize),
        )

    def forward(self, X):

        output = self.Conv_layer(X)
        output = self.Flatten(output)
        output = self.dense_layer(output)

        return output

    # initialize the model
    def init_weight(self):

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)


# AlexNet
class AlexNet(nn.Module):
    def __init__(
        self,
        inputSize=1,
        outputSize=10,
    ):
        super().__init__()

        self.Conv_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=inputSize,
                out_channels=96,
                kernel_size=11,
                stride=4,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.Flatten = nn.Flatten()

        self.dense_layer = nn.Sequential(
            nn.Linear(6400, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, outputSize),
        )

    def forward(self, X):

        output = self.Conv_layer(X)
        output = self.Flatten(output)
        output = self.dense_layer(output)

        return output

    # initialize the model
    def init_weight(self):

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)


# VGG extendable network
class VGG(nn.Module):
    def __init__(
        self,
        inputSize=1,
        conv_architecture=[(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)],
        outputSize=10,
        p_dropout=0.5,
    ):
        super().__init__()

        # set the convolutional layers
        conv_blocks = []
        in_channel = inputSize

        for idx, (num_block, out_channel) in enumerate(conv_architecture):
            conv_blocks.append(
                self.vgg_block(
                    num_blocks=num_block,
                    in_channels=in_channel,
                    out_channels=out_channel,
                )
            )
            in_channel = out_channel

        self.Conv_layer = nn.Sequential(*conv_blocks)

        # set flatten layer
        self.Flatten = nn.Flatten()

        # set dense layer
        self.dense_layer = nn.Sequential(
            nn.Linear(in_features=out_channel * 7 * 7, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=p_dropout),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=p_dropout),
            nn.Linear(in_features=4096, out_features=outputSize),
        )

    # generate a vgg block
    def vgg_block(self, num_blocks, in_channels, out_channels):

        block = []
        for _ in range(num_blocks):
            block.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
            block.append(nn.ReLU())
            in_channels = out_channels

        block.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return nn.Sequential(*block)

    def forward(self, X):

        output = self.Conv_layer(X)
        output = self.Flatten(output)
        output = self.dense_layer(output)

        return output

    def init_weight(self):

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)


# common variations of VGG
# VGG-11
class VGG_11(VGG):
    def __init__(
        self,
        inputSize=1,
        outputSize=10,
        p_dropout=0.5,
    ):
        super().__init__(
            inputSize=inputSize,
            conv_architecture=[(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)],
            outputSize=outputSize,
            p_dropout=p_dropout,
        )


# VGG-13
class VGG_13(VGG):
    def __init__(
        self,
        inputSize=1,
        outputSize=10,
        p_dropout=0.5,
    ):
        super().__init__(
            inputSize=inputSize,
            conv_architecture=[(2, 64), (2, 128), (2, 256), (2, 512), (2, 512)],
            outputSize=outputSize,
            p_dropout=p_dropout,
        )


# VGG-16
class VGG_16(VGG):
    def __init__(
        self,
        inputSize=1,
        outputSize=10,
        p_dropout=0.5,
    ):
        super().__init__(
            inputSize=inputSize,
            conv_architecture=[(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)],
            outputSize=outputSize,
            p_dropout=p_dropout,
        )


# VGG-19
class VGG_19(VGG):
    def __init__(
        self,
        inputSize=1,
        outputSize=10,
        p_dropout=0.5,
    ):
        super().__init__(
            inputSize=inputSize,
            conv_architecture=[(2, 64), (2, 128), (4, 256), (4, 512), (4, 512)],
            outputSize=outputSize,
            p_dropout=p_dropout,
        )


# NiN
class NiN(nn.Module):
    def __init__(
        self,
        inputSize=1,
        outputSize=10,
        p_dropout=0.5,
    ):
        super().__init__()

        # set the convolutional layers
        self.Conv_layer = nn.Sequential(
            self.nin_block(
                inputSize, out_channels=96, kernel_size=11, stride=4, padding=0
            ),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.nin_block(
                in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2
            ),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.nin_block(
                in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1
            ),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(p=p_dropout),
            self.nin_block(
                in_channels=384,
                out_channels=outputSize,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # set flatten layer
        self.Flatten = nn.Flatten()

    def nin_block(self, in_channels, out_channels, kernel_size, stride, padding):

        block = []
        block.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        )
        block.append(nn.ReLU())
        block.append(
            nn.Conv2d(
                in_channels=out_channels, out_channels=out_channels, kernel_size=1
            )
        )
        block.append(nn.ReLU())
        block.append(
            nn.Conv2d(
                in_channels=out_channels, out_channels=out_channels, kernel_size=1
            )
        )
        block.append(nn.ReLU())

        return nn.Sequential(*block)

    def forward(self, X):

        output = self.Conv_layer(X)
        output = self.Flatten(output)

        return output

    def init_weight(self):

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)


# GoogleNet
class Inception(nn.module):
    def __init__(
        self,
        in_channels,
        out_channels1,
        out_channels2,
        out_channels3,
        out_channels4,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # path 1: 1x1 convolution
        self.p1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels1, kernel_size=1), nn.ReLU()
        )

        # path 2: 1x1 convolution + 3x3 convolution
        self.p2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels2[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels2[0], out_channels2[1], kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # path 3: 1x1 convolution + 5x5 convolution
        self.p3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels3[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels3[0], out_channels3[1], kernel_size=5, padding=2),
            nn.ReLU(),
        )

        # path 4: 3x3 max pooling + 1x1 convolution
        self.p4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels4, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, X):

        result1 = self.p1(X)
        result2 = self.p2(X)
        result3 = self.p3(X)
        result4 = self.p4(X)

        return torch.cat([result1, result2, result3, result4], dim=1)


class GoogleNet(nn.Module):
    def __init__(
        self,
        inputSize=1,
        outputSize=10,
    ):
        super().__init__()

        # first block
        # 7x7 convolution + 3x3 max pooling
        self.block1 = nn.Sequential(
            nn.Conv2d(inputSize, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # second block
        # 1x1 convolution + 3x3 convolution + 3x3 max pooling
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # third block
        # two inception modules + 3x3 max pooling
        self.block3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # fourth block
        # five inception modules + 3x3 max pooling
        self.block4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (144, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # fifth block
        # two inception modules + 1x1 average pooling
        self.block5 = nn.Sequential(
            Inception(832, 256, (128, 320), (32, 128), 128),
            Inception(832, 384, (128, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # flatten layer
        self.Flatten = nn.Flatten()

        # fully connected layer
        self.Dense = nn.Linear(1024, outputSize)

    def forward(self, X):

        output = self.block1(X)
        output = self.block2(output)
        output = self.block3(output)
        output = self.block4(output)
        output = self.block5(output)
        output = self.Flatten(output)
        output = self.Dense(output)

        return output


# ResNet
class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        use_1x1conv=False,
        strides=1,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=strides, padding=1
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=strides
            )
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):

        output = F.relu(self.bn1(self.conv1(X)))
        output = self.bn2(self.conv2(output))
        if self.conv3:
            residual = self.conv3(X)
        output += residual

        return F.relu(output)


class ResNet(nn.Module):
    def __init__(
        self,
        inputSize=1,
        outputSize=10,
    ):
        super().__init__()

        # first convolutional layer
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(inputSize, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # second convolutional layer
        self.conv_block2 = nn.Sequential(*self.resblock(64, 64, 2, first_block=True))
        # third convolutional layer
        self.conv_block3 = nn.Sequential(*self.resblock(64, 128, 2))
        # fourth convolutional layer
        self.conv_block4 = nn.Sequential(*self.resblock(128, 256, 2))
        # fifth convolutional layer
        self.conv_block5 = nn.Sequential(*self.resblock(256, 512, 2))

        self.net = nn.Sequential(
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            self.conv_block4,
            self.conv_block5,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, outputSize),
        )

    def resblock(self, in_channels, out_channels, num_residuals, first_block=False):

        block = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                block.append(
                    ResBlock(in_channels, out_channels, use_1x1conv=True, strides=2)
                )
            else:
                block.append(ResBlock(out_channels, out_channels))
        return block

    def forward(self, X):

        return self.net(X)
