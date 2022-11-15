"""
File Name: _multiProc.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_utils/_nn/_multiProc.py
File Created: Saturday, 12th November 2022 4:41:07 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 14th November 2022 8:17:13 pm
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

from __future__ import annotations

from typing import List, Tuple, Dict, Union, Any
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from InsurAutoML._utils import (
    MetaData,
    formatting,
)
from InsurAutoML._constant import FULLTYPE_MAPPING

"""
List of methods:
                    Txt           Cat           Con
FusToken:      tokenize      encoding      encoding
FusEmbed:     embedding     embedding     embedding
FusModel:         model         model         model

FusToken: Txt/Cat/Con are only tokenized to numerics and fuse those heads together for whole model training.
FusEmbed: Txt/Cat/Con are further embedded with a few layers of NNs and fuse those embeddings together for later half of model training.
FusModel: Txt/Cat/Con are almost trained separately and just fused together for unified prediction.
"""


class NoProc:
    def __init__(self) -> None:
        self._fittted = False

    def fit(
        self,
        data: Union[pd.DataFrame, np.ndarray],
    ) -> NoProc:

        # record vocab size
        self._vocab_size = 0

        self._fitted = True

        return self

    def transform(
        self,
        data: Union[pd.DataFrame, np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        # if not fitted, raise error
        if not self._fitted:
            raise RuntimeError("NoProc is not fitted yet. Run fit() first.")

        # convert data to torch tensor
        data = torch.tensor(data.values if isinstance(data, pd.DataFrame) else data)

        # check data dimension
        if len(data.size()) != 2:
            raise TypeError(
                "Data must be a 2D tensor. Got {}D tensor.".format(len(data.size()))
            )

        return data


class TxtTokenize:
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 128,
        starting_offset: int = 0,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.starting_offset = starting_offset

        self._fitted = False

    # tokenizer mapping function
    def _tokenizer(self, text: str) -> List[int]:
        return self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]

    def fit(
        self,
        data: Union[pd.DataFrame, np.ndarray],
    ) -> TxtTokenize:
        # get tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # record vocab size
        self._vocab_size = self.tokenizer.vocab_size

        self._fitted = True

        return self

    def transform(
        self,
        data: Union[pd.DataFrame, np.ndarray],
    ):
        # if not fitted, raise error
        if not self._fitted:
            raise RuntimeError("TxtTokenize is not fitted yet. Run fit() first.")

        # tokenize the data
        data = data.applymap(lambda x: self._tokenizer(x))

        # convert to tensor
        return (
            torch.cat(
                [torch.tensor(data[col].to_list()) for col in data.columns], dim=1
            )
            + self.starting_offset
        )


class CatOffsetEncoding:

    """
    This function is used to convert categorical data into Encoding.

    Parameters
    ----------
    unique_classes: the number of unique classes for each column.

    starting_offset: the starting offset for each column. This is used to make sure
    different encodings have different range of values.
    """

    def __init__(
        self,
        unique_classes: List[int] = None,
        starting_offset: int = 0,
    ) -> None:
        self.unique_classes = unique_classes
        self.starting_offset = starting_offset

        self._fitted = False

    # # The purpose of this function is to make sure the unique values are consecutive
    # # Example: [8, 9, 4, 9, 4, 8, 2, 0, 6, 1] -> [5, 6, 3, 6, 3, 5, 2, 0, 4, 1]
    # # As seen, the missing 10 classes are reduced to 7 classes
    # def CatConsec(
    #     data: torch.Tensor,
    # ):
    #     # get sorted (by ascending order) and indices
    #     sorted, indices = data.sort(dim=0, descending=False)
    #     # replicate the first row to make sure the first/second row diff is 0
    #     data_pad = torch.cat([sorted[0, :].unsqueeze(0), sorted], dim=0)
    #     # get the difference between each row and reduce by 1
    #     # since the normal step is 1, we don't need modification for those
    #     data_diff = F.relu(torch.diff(data_pad, dim=0) - 1)
    #     # get the cumulative sum of the difference
    #     data_cumsum = torch.cumsum(data_diff, dim=0)
    #     # initialize the result
    #     result = torch.zeros(data.size(), dtype=torch.int64)

    #     # distribute the indices to the unique values
    #     return result.scatter_(0, indices, (sorted - data_cumsum))

    def fit(
        self,
        data: Union[pd.DataFrame, np.ndarray, torch.Tensor],
    ) -> CatOffsetEncoding:
        # convert data to torch tensor
        data = torch.tensor(data.values if isinstance(data, pd.DataFrame) else data)

        # check data dimension
        if len(data.size()) != 2:
            raise TypeError(
                "Data must be a 2D tensor. Got {}D tensor.".format(len(data.size()))
            )

        # if unique_classes not passed, get unique classes from data
        if self.unique_classes is None:
            # Update: Nov. 13, 2022
            # It's not optimal to convert, ignore this problem
            # # if not passed, check the unique values to start from 0 with step 1
            # # make sure the unique values are consecutive
            # data = self.CatConsec(data)

            self.unique_classes = [len(torch.unique(t)) for t in torch.unbind(data.T)]

        # get offset encoding
        # cumsum of each column number of unique classes
        cat_offset = F.pad(
            torch.tensor(self.unique_classes), (1, 0), value=self.starting_offset
        )
        self.cat_offset = cat_offset.cumsum(dim=-1)[:-1]

        # record vocab size
        self._vocab_size = sum(self.unique_classes)

        self._fitted = True

        return self

    def transform(
        self,
        data: Union[pd.DataFrame, np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        # if not fitted, raise error
        if not self._fitted:
            raise RuntimeError("CatOffsetEncoding is not fitted yet. Run fit() first.")

        # convert data to torch tensor
        data = torch.tensor(data.values if isinstance(data, pd.DataFrame) else data)

        # check data dimension
        if len(data.size()) != 2:
            raise TypeError(
                "Data must be a 2D tensor. Got {}D tensor.".format(len(data.size()))
            )

        return data + self.cat_offset


class NumOffsetEncoding(CatOffsetEncoding):

    """
    This function is used to convert numerical data into categorical Encoding.

    Parameters
    ----------
    num_classes: the number of unique classes for each column.

    starting_offset: the starting offset for each column. This is used to make sure
    different encodings have different range of values.
    """

    def __init__(
        self,
        num_classes: int = 10,
        starting_offset: int = 0,
    ) -> None:
        self.num_classes = num_classes
        self.starting_offset = starting_offset

        self._fitted = False

    def fit(
        self,
        data: Union[pd.DataFrame, np.ndarray, torch.Tensor],
    ) -> NumOffsetEncoding:
        # convert data to torch tensor
        data = torch.tensor(data.values if isinstance(data, pd.DataFrame) else data)

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
        self._range = (vmin.unsqueeze(0), vmax.unsqueeze(0))

        # initialize and fit CatOffsetEncoding
        super(NumOffsetEncoding, self).__init__(
            unique_classes=[self.num_classes for _ in range(data.size()[1])],
            starting_offset=self.starting_offset,
        )
        super(NumOffsetEncoding, self).fit(data)

        self._fitted = True

        return self

    def transform(
        self,
        data: Union[pd.DataFrame, np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        # if not fitted, raise error
        if not self._fitted:
            raise RuntimeError("NumOffsetEncoding is not fitted yet. Run fit() first.")

        # convert data to torch tensor
        data = torch.tensor(data.values if isinstance(data, pd.DataFrame) else data)

        # check data dimension
        if len(data.size()) != 2:
            raise TypeError(
                "Data must be a 2D tensor. Got {}D tensor.".format(len(data.size()))
            )

        # unpack range
        (vmin, vmax) = self._range

        data = (data - vmin) / (vmax - vmin) * self.num_classes
        data = data.int()

        return super(NumOffsetEncoding, self).transform(data)


class PipProc:
    def __init__(
        self,
        components: List[
            Tuple[Tuple, Union[TxtTokenize, CatOffsetEncoding, NumOffsetEncoding]]
        ],
    ) -> None:
        self.components = components

        self._fitted = False

    def fit(
        self,
    ) -> PipProc:
        for fulltype, component in self.components:
            # every component should be already fitted
            if not component._fitted:
                raise RuntimeError(
                    "Component for {} is not fitted yet.".format(fulltype)
                )
            # component.fit(X)

        self._fitted = True

        return self

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, torch.Tensor]:
        # if not fitted, raise error
        if not self._fitted:
            raise RuntimeError("PipProc is not fitted yet. Run fit() first.")

        # if metadata not provided, then generate it
        if metadata is None:
            metadata = self.get(X)

        # result = []
        # for fulltype, component in self.components:
        #     result.append(component.transform(X[metadata[fulltype]]))

        result = {}
        # Orders of preprocessing
        # text -> cat -> con
        # text
        for fulltype, component in self.components:
            key = FULLTYPE_MAPPING[fulltype]
            # if already have, concat with existed ones
            if key in result.keys():
                result[key] = torch.cat(
                    [result[key], component.transform(X[metadata[fulltype]])], dim=1
                )
            # else, create new one
            else:
                result[key] = component.transform(X[metadata[fulltype]])

        # concatenate the results
        return result


class MultiPreprocessing(MetaData, formatting):

    _support_fulltype = [
        ("Object", "Text"),
        ("Int", "Numerical"),
        ("Int", "Categorical"),
        ("Float", ""),
    ]

    def __init__(
        self,
        method="FusToken",
    ) -> None:
        self.method = method

        self._fitted = False

    def _fusProc(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        **kwargs,
    ):
        starting_offset = 0  # initialize starting offset
        pip_components = []  # initialize list of components

        # Orders of preprocessing
        # text -> cat -> con
        # text:
        if ("Object", "Text") in self.metadata.keys():
            TxtProc = TxtTokenize(
                starting_offset=0,
                **kwargs,
            )
            TxtProc.fit(X[self.metadata[("Object", "Text")]])
            pip_components += [(("Object", "Text"), TxtProc)]
            starting_offset += TxtProc._vocab_size
        # cat:
        # Object Categorical, convert to Int Categorical
        if ("Object", "Categorical") in self.metadata.keys():
            super(MetaData, self).__init__(
                columns=self.metadata[("Object", "Categorical")]
            )
            super(MetaData, self).fit(X)

            # update metadata
            self.update(X, names=self.metadata[("Object", "Categorical")])
        # Int Categorical
        if ("Int", "Categorical") in self.metadata.keys():
            CatProc = CatOffsetEncoding(
                starting_offset=starting_offset,
            )
            CatProc.fit(X[self.metadata[("Int", "Categorical")]])
            pip_components += [(("Int", "Categorical"), CatProc)]
            starting_offset += CatProc._vocab_size
        # con:
        # Int Numerical
        if ("Int", "Numerical") in self.metadata.keys():
            NumProc = NumOffsetEncoding(
                starting_offset=starting_offset,
            )
            NumProc.fit(X[self.metadata[("Int", "Numerical")]])
            pip_components += [(("Int", "Numerical"), NumProc)]
        # Float
        if ("Float", "") in self.metadata.keys():
            NumProc = NumOffsetEncoding(
                starting_offset=starting_offset,
            )
            NumProc.fit(X[self.metadata[("Float", "")]])
            pip_components += [(("Float", ""), NumProc)]

        return PipProc(pip_components)

    def _nofusProc(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        **kwargs,
    ):
        pip_components = []  # initialize list of components

        # Orders of preprocessing
        # text -> cat -> con
        # text:
        if ("Object", "Text") in self.metadata.keys():
            TxtProc = TxtTokenize(
                **kwargs,
            )
            TxtProc.fit(X[self.metadata[("Object", "Text")]])
            pip_components += [(("Object", "Text"), TxtProc)]
        # cat:
        # Object Categorical, convert to Int Categorical
        if ("Object", "Categorical") in self.metadata.keys():
            super(MetaData, self).__init__(
                columns=self.metadata[("Object", "Categorical")]
            )
            super(MetaData, self).fit(X)

            # update metadata
            self.update(X, names=self.metadata[("Object", "Categorical")])
        # Int Categorical
        if ("Int", "Categorical") in self.metadata.keys():
            CatProc = NoProc()
            CatProc.fit(X[self.metadata[("Int", "Categorical")]])
            pip_components += [(("Int", "Categorical"), CatProc)]
        # con:
        # Int Numerical
        if ("Int", "Numerical") in self.metadata.keys():
            NumProc = NoProc()
            NumProc.fit(X[self.metadata[("Int", "Numerical")]])
            pip_components += [(("Int", "Numerical"), NumProc)]
        # Float
        if ("Float", "") in self.metadata.keys():
            NumProc = NoProc()
            NumProc.fit(X[self.metadata[("Float", "")]])
            pip_components += [(("Float", ""), NumProc)]

        return PipProc(pip_components)

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        metadata: Dict[str, Any] = None,
        **kwargs,
    ) -> MultiPreprocessing:

        # if metadata not provided, then generate it
        if metadata is None:
            super(MultiPreprocessing, self).__init__(X)
        else:
            self.metadata = metadata

        # text:

        self._pipe = (
            self._fusProc(X, **kwargs)
            if self.method == "FusToken"
            else self._nofusProc(X, **kwargs)
        )
        self._pipe.fit()

        self._fitted = True

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Dict[str, torch.Tensor]:

        if not self._fitted:
            raise RuntimeError("The preprocessing is not fitted. Run fit() first.")

        result = self._pipe.transform(X, self.metadata)

        # update metadata
        self.update(X)

        return result

    def fit_transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Dict[str, torch.Tensor]:

        self.fit(X)
        return self.transform(X)


if __name__ == "__main__":
    pass
