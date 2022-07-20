"""
File: _utils.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Last Version: 0.2.1
Relative Path: /My_AutoML/_utils/_nas/_utils.py
File Created: Saturday, 16th July 2022 11:36:05 am
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Tuesday, 19th July 2022 12:22:59 am
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

from ._ops import StdConvBN


class ChannelCalibration(nn.Module) :
    
    """
    Reset channel size for concat
    """
    
    def __init__(
        self,
        in_channels,
        out_channels,
    ) :
        super().__init__()
        
        self.need_calib = False
        if in_channels != out_channels :
            self.need_calib = True
            self.module = StdConvBN(in_channels, out_channels)
            
    def forward(self, X) :
        
        if self.need_calib :
            return self.module(X)
        else :
            return X
        