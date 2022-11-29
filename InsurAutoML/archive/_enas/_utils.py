"""
File: _utils.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Last Version: 0.2.1
Relative Path: /My_AutoML/_utils/_nas/_enas/_utils.py
File Created: Saturday, 16th July 2022 10:34:07 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Saturday, 16th July 2022 11:42:24 pm
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


def decision_mask(n_nodes, n_activations):
    """
    mask for decision nodes
    """

    dim_2 = max(n_nodes - 1, n_activations)

    # initialize mask
    mask = torch.ones(2 * n_nodes - 1, dim_2)
    # set node masks for unavailable activations
    mask[::2, n_activations:] = 0
    # set connection masks for future connections
    mask[1::2, :] = torch.tril(torch.ones(n_nodes - 1, dim_2), diagonal=0)

    return mask
