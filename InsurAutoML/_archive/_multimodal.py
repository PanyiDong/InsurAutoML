"""
File: _multimodal.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_archive/_multimodal.py
File: _multimodal.py
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Sunday, 13th November 2022 12:38:01 am
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

from typing import List, Tuple, Dict, Union, Any
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


def TxtTokenize(
    data: Union[pd.DataFrame, np.ndarray],
    model_name: str = "bert-base-uncased",
    max_length: int = 128,
):
    # get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer mapping function

    def _tokenizer(text: str) -> List[int]:
        return tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]

    # tokenize the data
    data = data.applymap(lambda x: _tokenizer(x))

    # convert to tensor
    return torch.cat([torch.tensor(data[col].to_list())
                      for col in data.columns], dim=1)


# The purpose of this function is to make sure the unique values are consecutive
# Example: [8, 9, 4, 9, 4, 8, 2, 0, 6, 1] -> [5, 6, 3, 6, 3, 5, 2, 0, 4, 1]
# As seen, the missing 10 classes are reduced to 7 classes
def CatConsec(
    data: torch.Tensor,
):
    # get sorted (by ascending order) and indices
    sorted, indices = data.sort(dim=0, descending=False)
    # replicate the first row to make sure the first/second row diff is 0
    data_pad = torch.cat([sorted[0, :].unsqueeze(0), sorted], dim=0)
    # get the difference between each row and reduce by 1
    # since the normal step is 1, we don't need modification for those
    data_diff = F.relu(torch.diff(data_pad, dim=0) - 1)
    # get the cumulative sum of the difference
    data_cumsum = torch.cumsum(data_diff, dim=0)
    # initialize the result
    result = torch.zeros(data.size(), dtype=torch.int64)

    # distribute the indices to the unique values
    return result.scatter_(0, indices, (sorted - data_cumsum))


# Update: Nov. 13, 2022
# Function version decrypted to avoid inconsistency for train/test split
def CatOffsetEncoding(
    data: Union[pd.DataFrame, np.ndarray, torch.Tensor],
    unique_classes: List[int] = None,
    starting_offset: int = 0,
) -> torch.Tensor:
    """
    This function is used to convert categorical data into Encoding.

    Parameters
    ----------
    data: input data, must be a 2D tensor.

    unique_classes: the number of unique classes for each column.

    starting_offset: the starting offset for each column. This is used to make sure
    different encodings have different range of values.
    """

    # convert data to torch tensor
    data = torch.tensor(
        data.values if isinstance(
            data, pd.DataFrame) else data)

    # check data dimension
    if len(data.size()) != 2:
        raise TypeError(
            "Data must be a 2D tensor. Got {}D tensor.".format(len(data.size()))
        )

    # if unique_classes not passed, get unique classes from data
    if unique_classes is None:
        # if not passed, check the unique values to start from 0 with step 1
        # make sure the unique values are consecutive
        data = CatConsec(data)

        unique_classes = [len(torch.unique(t)) for t in torch.unbind(data.T)]

    # get offset encoding
    # cumsum of each column number of unique classes
    cat_offset = F.pad(torch.tensor(unique_classes),
                       (1, 0), value=starting_offset)
    cat_offset = cat_offset.cumsum(dim=-1)[:-1]

    return data + cat_offset


def NumOffsetEncoding(
    data: Union[pd.DataFrame, np.ndarray, torch.Tensor],
    num_classes: int = 10,
    starting_offset: int = 0,
) -> torch.Tensor:
    """
    This function is used to convert numerical data into categorical Encoding.

    Parameters
    ----------
    data: input data, must be a 2D tensor.

    num_classes: the number of unique classes for each column.

    starting_offset: the starting offset for each column. This is used to make sure
    different encodings have different range of values.
    """

    # convert data to torch tensor
    data = torch.tensor(
        data.values if isinstance(
            data, pd.DataFrame) else data)

    # check data dimension
    if len(data.size()) != 2:
        raise TypeError(
            "Data must be a 2D tensor. Got {}D tensor.".format(len(data.size()))
        )

    # convert numerical data into categorical data
    # use min-max normalization and multiply by num_classes
    # then convert to int to get categories
    vmax = torch.max(data, dim=0)[0]
    vmin = torch.min(data, dim=0)[0]

    data = (data - vmin.unsqueeze(0)) / \
        (vmax - vmin).unsqueeze(0) * num_classes
    data = data.int()

    return CatOffsetEncoding(
        data, [num_classes for _ in range(data.size()[1])], starting_offset
    )
