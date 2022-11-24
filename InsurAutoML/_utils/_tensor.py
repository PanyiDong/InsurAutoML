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
Last Modified: Thursday, 24th November 2022 12:26:40 am
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
import os
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
            raise ValueError(
                "inputs and labels must have the same length. Get {} and {}".format(
                    len(inputs), len(labels)))

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Union[Dict, Tuple, List]:
        if self.format == "dict":
            return {"input": self.inputs[idx], "label": self.labels[idx]}
        elif self.format == "tuple":
            return (self.inputs[idx], self.labels[idx])
        elif self.format == "list":
            return [self.inputs[idx], self.labels[idx]]

    @property
    def inputSize(self) -> int:

        return self.inputs.size()[-1]

    @property
    def outputSize(self) -> int:

        return len(self.labels.unique())


# class BaseTensorDataset(Dataset):

#     """Custom Tensor Dataset that can be used for serialization for tensor Dataset."""

#     def __init__(
#         self,
#         inputs: torch.Tensor,
#         labels: torch.Tensor,
#         path: str = "tmp",
#         format: str = "tuple",
#     ) -> None:
#         self.path = path
#         self.format = format

#         if len(inputs) != len(labels):
#             raise ValueError(
#                 "inputs and labels must have the same length. Get {} and {}".format(
#                     len(inputs), len(labels)))

#         # get paths and save tensors for inputs
#         self.input = os.path.join(self.path, "input.pt")
#         torch.save(inputs, self.input)

#         # get paths and save tensors for labels
#         self.labels = os.path.join(self.path, "labels.pt")
#         torch.save(labels, self.labels)

#     def __len__(self) -> int:
#         return len(torch.load(self.input))

#     def __getitem__(self, idx: int) -> Union[Dict, Tuple, List]:
#         input, label = torch.load(self.input)[
#             idx], torch.load(self.labels)[idx]

#         if self.format == "dict":
#             return {"input": input, "label": label}
#         elif self.format == "tuple":
#             return (input, label)
#         elif self.format == "list":
#             return [input, label]


class ListTensorDataset(Dataset):

    """Custom Tensor Dataset that can be used for serialization for List of tensor Dataset."""

    def __init__(
        self,
        inputs: List[torch.Tensor],
        labels: torch.Tensor,
        path: str = "tmp",
        format: str = "tuple",
    ) -> None:
        self.path = path
        self.format = format

        self._input_format = "list"
        if len(inputs[0]) != len(labels):
            raise ValueError(
                "inputs and labels must have the same length. Get {} and {}".format(
                    len(inputs[0]), len(labels)))

        # make dir for list/dict type
        if not os.path.exists(os.path.join(path, "input")):
            os.makedirs(os.path.join(path, "input"))

        # get paths and save tensors for inputs
        self.input = [os.path.join(
            self.path, "input/{}.pt".format(i)) for i in range(len(inputs))]
        for _input, _path in zip(inputs, self.input):
            torch.save(_input, _path)

        # get paths and save tensors for labels
        self.labels = os.path.join(self.path, "labels.pt")
        torch.save(labels, self.labels)

    def __len__(self) -> int:
        return len(torch.load(self.input[0]))

    def __getitem__(self, idx: int) -> Union[Dict, Tuple, List]:
        input, label = [torch.load(_input)[idx] for _input in self.input], torch.load(
            self.labels)[idx]

        if self.format == "dict":
            return {"input": input, "label": label}
        elif self.format == "tuple":
            return (input, label)
        elif self.format == "list":
            return [input, label]


class DictTensorDataset(Dataset):

    """Custom Tensor Dataset that can be used for serialization for dict of tensor Dataset."""

    def __init__(
        self,
        inputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        path: str = "tmp",
        format: str = "tuple",
    ) -> None:
        self.path = path
        self.format = format

        self._input_format = "dict"
        if len(list(inputs.values())[0]) != len(labels):
            raise ValueError(
                "inputs and labels must have the same length. Get {} and {}".format(
                    len(list(inputs.values())[0]), len(labels)))

        # make dir for list/dict type
        if not os.path.exists(os.path.join(path, "input")):
            os.makedirs(os.path.join(path, "input"))

        # get paths and save tensors for inputs
        self.input = [os.path.join(
            self.path, "input/{}.pt".format(i)) for i in inputs.keys()]
        for _input, _path in zip(inputs.values(), self.input):
            torch.save(_input, _path)

        # get paths and save tensors for labels
        self.labels = os.path.join(self.path, "labels.pt")
        torch.save(labels, self.labels)

    def __len__(self) -> int:
        return len(torch.load(self.input[0]))

    def __getitem__(self, idx: int) -> Union[Dict, Tuple, List]:
        input, label = [torch.load(_input)[idx] for _input in self.input], torch.load(
            self.labels)[idx]

        if self.format == "dict":
            return {"input": input, "label": label}
        elif self.format == "tuple":
            return (input, label)
        elif self.format == "list":
            return [input, label]


class SerialTensorDataset(ListTensorDataset, DictTensorDataset):

    """Custom Tensor Dataset that can be used for serialization."""

    def __init__(
        self,
        inputs: Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]],
        labels: torch.Tensor,
        path: str = "tmp",
        format: str = "tuple",
    ) -> None:

        if isinstance(inputs, torch.Tensor):
            super(SerialTensorDataset, self).__init__(
                [inputs], labels, path, format)
        elif isinstance(inputs, list):
            super(SerialTensorDataset, self).__init__(
                inputs, labels, path, format)
        elif isinstance(inputs, dict):
            super(ListTensorDataset, self).__init__(
                inputs, labels, path, format)
