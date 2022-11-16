"""
File Name: _tensor.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_utils/_tensor.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 14th November 2022 9:46:39 pm
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

from typing import Union, Dict, Tuple, List
import torch
from torch.utils.data import Dataset

# detach tensor from the computation graph
def repackage_hidden(h):

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


class CustomTensorDataset(Dataset):

    """
    Custom Tensor Dataset
    """

    def __init__(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        format: str = "tuple",
    ) -> None:
        self.inputs = inputs
        self.labels = labels
        self.format = format

        if len(self.inputs) != len(self.labels):
            raise ValueError("inputs and labels must have the same length")

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Union[Dict, Tuple, List]:
        if self.format == "dict":
            return {"input": self.inputs[idx], "label": self.labels[idx]}
        elif self.format == "tuple":
            return (self.inputs[idx], self.labels[idx])
        elif self.format == "list":
            return [self.inputs[idx], self.labels[idx]]

    def inputSize(self) -> int:

        return self.inputs.size()[-1]

    def outputSize(self) -> int:

        return len(self.labels.unique())
