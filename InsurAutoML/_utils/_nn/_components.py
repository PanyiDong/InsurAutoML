"""
File Name: _components.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_utils/_nn/_components.py
File Created: Tuesday, 22nd November 2022 4:42:24 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Tuesday, 22nd November 2022 4:47:45 pm
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

from typing import Union
import nni.retiarii.nn.pytorch as nninn
# from nni.retiarii import model_wrapper

from ..._experimental._nn._nni._nas._baseSpace import MLPBaseSpace, MLPHead

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

# Update: Nov. 22, 2022
# only apply model_wrapper to highest level of the model

############################################################################
# Fusion Heads


class TxtHead(MLPHead):

    def __init__(
        self,
        inputSize: int,
        outputSize: int,
        prefix="txt_",
    ) -> None:
        super(TxtHead, self).__init__(inputSize, outputSize, prefix=prefix)


class CatHead(MLPHead):

    def __init__(
        self,
        inputSize: int,
        outputSize: int,
        prefix="cat_",
    ) -> None:
        super(CatHead, self).__init__(inputSize, outputSize, prefix=prefix)


class ConHead(MLPHead):

    def __init__(
        self,
        inputSize: int,
        outputSize: int,
        prefix="con_",
    ) -> None:
        super(ConHead, self).__init__(inputSize, outputSize, prefix=prefix)

############################################################################
# Fusion nets


class TxtNet(MLPBaseSpace):

    def __init__(
        self,
        inputSize: int,
        outputSize: Union[int, nninn.InputChoice],
        prefix="txt_",
    ) -> None:
        super(TxtNet, self).__init__(inputSize, outputSize, prefix=prefix)


class CatNet(MLPBaseSpace):

    def __init__(
        self,
        inputSize: int,
        outputSize: Union[int, nninn.InputChoice],
        prefix="cat_",
    ) -> None:
        super(CatNet, self).__init__(inputSize, outputSize, prefix=prefix)


class ConNet(MLPBaseSpace):

    def __init__(
        self,
        inputSize: int,
        outputSize: Union[int, nninn.InputChoice],
        prefix="con_",
    ) -> None:
        super(ConNet, self).__init__(inputSize, outputSize, prefix=prefix)
