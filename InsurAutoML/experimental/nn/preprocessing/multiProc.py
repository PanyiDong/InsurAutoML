"""
File Name: multiProc.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/preprocessing/multiProc.py
File Created: Wednesday, 16th November 2022 7:39:46 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Tuesday, 6th December 2022 11:33:08 pm
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

from typing import Dict, Union, Any
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from InsurAutoML.utils import (
    MetaData,
    formatting,
)
from .utils import NoProc, TxtTokenize, CatOffsetEncoding, NumOffsetEncoding, PipProc

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


class MultiPreprocessing(MetaData, formatting):

    """
    This class is used to preprocess data for multi-modal learning.

    Can handle text, categorical, and numerical data.

    Parameters
    ----------
    methods: methods used for preprocessing, default = "FusToken"
    all methods: ["FusToken", "FusEmbed", "FusModel"]
    """

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
            # get and fit text tokenizer
            TxtProc = TxtTokenize(
                starting_offset=0,
                **kwargs,
            )
            TxtProc.fit(X[self.metadata[("Object", "Text")]])
            # put into pipeline
            pip_components += [(("Object", "Text"), TxtProc)]
            # update starting offset
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
            starting_offset += NumProc._vocab_size
        # Float
        if ("Float", "") in self.metadata.keys():
            NumProc = NumOffsetEncoding(
                starting_offset=starting_offset,
            )
            NumProc.fit(X[self.metadata[("Float", "")]])
            pip_components += [(("Float", ""), NumProc)]
            starting_offset += NumProc._vocab_size

        # for FusToken method, cat/con/txt all count for vocab_size
        self.vocab_size = starting_offset + 1  # add 1 for padding

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
            # for non-FusToken methods, vocab_size only counts for text data
            self.vocab_size = TxtProc._vocab_size
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

    def transform(self,
                  X: Union[pd.DataFrame,
                           np.ndarray]) -> Dict[str,
                                                torch.Tensor]:

        if not self._fitted:
            raise RuntimeError(
                "The preprocessing is not fitted. Run fit() first.")

        result = self._pipe.transform(X, self.metadata)

        # update metadata
        self.update(X)

        # if FusToken, concat all sections to one tensor
        if self.method == "FusToken":
            return torch.cat(list(result.values()), dim=1)
        # else, output separately
        else:
            return result

    def fit_transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:

        self.fit(X)
        return self.transform(X)


class FusTokenProc(MultiPreprocessing):
    def __init__(self, **kwargs):
        super(FusTokenProc, self).__init__(method="FusToken", **kwargs)


class FusEmbedProc(MultiPreprocessing):
    def __init__(self, **kwargs):
        super(FusEmbedProc, self).__init__(method="FusEmbed", **kwargs)


class FusModelProc(MultiPreprocessing):
    def __init__(self, **kwargs):
        super(FusModelProc, self).__init__(method="FusModel", **kwargs)


if __name__ == "__main__":
    pass
