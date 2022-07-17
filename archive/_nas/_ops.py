"""
File: _cnn_component.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Last Version: 0.2.1
Relative Path: /My_AutoML/_utils/_enas/_cnn_component.py
File Created: Friday, 15th July 2022 6:09:24 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Sunday, 17th July 2022 2:02:58 pm
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

import torch
import torch.nn as nn

################################################################################################################
# constants and dictionaries

activation_dict = {
    # "identity": nn.Identity(),
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
}

################################################################################################################
# basic components


class StdConvBN(nn.Module):

    """
    standard 1x1 convolution with batch normalization and ReLU activation
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
    ):
        super().__init__()

        self.module = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=0
            ),
            nn.BatchNorm2d(out_channels, affine=False),
            nn.ReLU(),
        )

    def forward(self, X):
        return self.module(X)


class DepthwiseSeparableConv(nn.Module):

    """
    Depth-wise Separable Convolution
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        depth=1,
        kernel_size=3,
        stride=1,
        padding=0,
        bias=False,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels * depth,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(
            in_channels * depth, out_channels, kernel_size=1, bias=bias
        )

    def forward(self, X):

        out = self.depthwise(X)
        out = self.pointwise(out)

        return out


class FactorizedReduce(nn.Module):

    """
    Factorized Reduction: reduce the number of channels and reshape the channels
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=2,
        padding=0,
        affine=False,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.postprocess = nn.BatchNorm2d(out_channels, affine=affine)

    def forward(self, x):
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.postprocess(out)
        return out


################################################################################################################
# combination components


class AvgPoolBN(nn.Module):

    """
    1. StdConvBN
    2. average pooling with batch normalization
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        affine=False,
    ):
        super().__init__()

        self.module = nn.Sequential(
            StdConvBN(in_channels, out_channels),
            nn.AvgPool2d(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels, affine=affine),
        )

    def forward(self, X):

        return self.module(X)


class MaxPoolBN(nn.Module):

    """
    1. StdConvBN
    2. max pooling with batch normalization
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        affine=False,
    ):
        super().__init__()

        self.module = nn.Sequential(
            StdConvBN(in_channels, out_channels),
            nn.MaxPool2d(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels, affine=affine),
        )

    def forward(self, X):

        return self.module(X)


class SepConvBN(nn.Module):

    """
    Depth-wise Separable Convolution followed by Batch Normalization
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
    ):
        super().__init__()

        self.module = nn.Sequential(
            nn.ReLU(),
            DepthwiseSeparableConv(
                in_channels,
                out_channels,
                depth=1,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels, affine=True),
        )

    def forward(self, X):

        return self.module(X)


class FullConvBN(nn.Module):

    """
    1. StdConvBN
    2. Conv layer followed by BatchNorm and ReLU
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        separable=True,
    ):
        self.preprocess = StdConvBN(in_channels, out_channels)
        if separable:
            self.conv = DepthwiseSeparableConv(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        else:
            self.conv = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )

        self.module = nn.Sequential(
            self.preprocess,
            self.conv,
            nn.BatchNorm2d(out_channels, affine=False),
            nn.ReLU(),
        )

    def forward(self, X):

        return self.module(X)
