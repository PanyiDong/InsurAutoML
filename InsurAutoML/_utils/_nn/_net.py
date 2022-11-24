"""
File Name: _net.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_utils/_nn/_net.py
File Created: Saturday, 19th November 2022 10:03:31 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Thursday, 24th November 2022 3:54:09 pm
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

from typing import List, Dict, Type, Union, Tuple
import torch
import nni.retiarii.nn.pytorch as nninn
from nni.retiarii import model_wrapper
from nni.nas.utils import basic_unit

from ..._experimental._nn._nni._nas._baseSpace import MLPBaseSpace, MLPLintSpace, MLPHead, RNNBaseSpace, RNNLintSpace, RNNHead
from ._components import TxtHead, CatHead, ConHead, TxtNet, CatNet, ConNet

# MLPHead = typing.cast(Type[MLPHead], basic_unit(MLPHead))
# MLPBaseSpace = typing.cast(Type[MLPBaseSpace], basic_unit(MLPBaseSpace))

############################################################################
# Non-Fusion Models


@model_wrapper
class MLPNet(MLPBaseSpace):

    def __init__(
        self,
        inputSize: int,
        outputSize: int,
    ) -> None:
        super().__init__(inputSize, outputSize)


@model_wrapper
class LiteMLPNet(MLPLintSpace):

    def __init__(
        self,
        inputSize: int,
        outputSize: int,
    ) -> None:
        super().__init__(inputSize, outputSize)


@model_wrapper
class RNNet(RNNBaseSpace):

    def __init__(
        self,
        inputSize: int,
        outputSize: int,
    ) -> None:
        super().__init__(inputSize, outputSize)


@model_wrapper
class LiteRNNet(RNNLintSpace):

    def __init__(
        self,
        inputSize: int,
        outputSize: int,
    ) -> None:
        super().__init__(inputSize, outputSize)

############################################################################
# Fusion Models


@model_wrapper
class FusTokenNet(nninn.Module):

    def __init__(
        self,
        inputSize: int,
        outputSize: int,
    ) -> None:
        super().__init__()
        self.net = nninn.LayerChoice(
            [MLPBaseSpace(inputSize, outputSize, prefix="MLP_base_"), MLPLintSpace(
                inputSize, outputSize, prefix="MLP_lint_")],
            label="FusTokenNet"
        )

    def forward(self, input: Union[torch.Tensor, List[torch.Tensor]], hidden: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:

        return self.net(input)


@model_wrapper
class FusEmbedNet(nninn.Module):

    def __init__(
        self,
        inputSize: Dict[str, int],
        outputSize: int,
    ) -> None:
        super().__init__()
        self.embeds = []
        # embedSize = [nninn.ValueChoice(
        #     [4, 8, 16, 32, 64], label=f"embedSize_{idx}") for idx in range(len(inputSize))]

        # for idx, _inputSize in enumerate(list(inputSize.values())):
        #     self.embeds.append(MLPHead(_inputSize, embedSize[idx], prefix=f"head_{idx}_"))
        # self.model = MLPHead(
        #     sum(embedSize), outputSize, prefix="model_")
        # Update: Nov.22, 2022
        # nni can't differentiate same module initialized with different parameters
        # so, need to wrap with different classes manually
        if "txt" in inputSize.keys():
            self.textnet = TxtHead(inputSize["txt"], nninn.ValueChoice(
                [4, 8, 16, 32, 64], label="txt_embedSize"))
            self.embeds.append(self.textnet)
        if "cat" in inputSize.keys():
            self.catnet = CatHead(inputSize["cat"], nninn.ValueChoice(
                [4, 8, 16, 32, 64], label="cat_embedSize"))
            self.embeds.append(self.catnet)
        if "con" in inputSize.keys():
            self.connet = ConHead(inputSize["con"], nninn.ValueChoice(
                [4, 8, 16, 32, 64], label="con_embedSize"))
            self.embeds.append(self.connet)
        self.model = MLPHead(
            sum([embed.outputSize for embed in self.embeds]), outputSize, prefix="model_")

    def forward(self, input: List[torch.Tensor]) -> torch.Tensor:

        # put everything through the embedding heads
        embed = [embed(input[index])
                 for index, embed in enumerate(self.embeds)]
        # concatenate the embeddings
        embed = torch.cat(embed, dim=-1)
        # put the concatenated embeddings through the model
        result = self.model(embed)

        return result

    # def build_model(self, inputSize, outputSize):
    #     return


@ model_wrapper
class FusModelNet(nninn.Module):

    def __init__(
        self,
        inputSize: Dict[str, int],
        outputSize: int,
    ) -> None:
        super().__init__()
        self.models = []
        if "txt" in inputSize.keys():
            self.textnet = TxtNet(inputSize["txt"], outputSize)
            self.models.append(self.textnet)
        if "cat" in inputSize.keys():
            self.catnet = CatNet(inputSize["cat"], outputSize)
            self.models.append(self.catnet)
        if "con" in inputSize.keys():
            self.connet = ConNet(inputSize["con"], outputSize)
            self.models.append(self.connet)

        # self.models = [MLPBaseSpace(_inputSize, outputSize, prefix="model{}_".format(index))
        #                for index, _inputSize in enumerate(inputSize)]

    def forward(self, input: List[torch.Tensor]) -> torch.Tensor:

        # put everything through the embedding heads
        self.textnet(input[0])
        self.catnet(input[1])
        self.connet(input[2])
        output = [model(input[index])
                  for index, model in enumerate(self.models)]
        # sum the outputs
        result = torch.sum(torch.stack(output), dim=0)

        return result
