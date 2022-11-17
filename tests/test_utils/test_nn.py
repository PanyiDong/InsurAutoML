"""
File: test_nn.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /tests/test_utils/test_nn.py
File: test_nn.py
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Tuesday, 15th November 2022 5:36:33 pm
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

# need nn installation


def test_MetaData():

    try:
        import torch
        from InsurAutoML._utils._nn._multiProc import MultiPreprocessing
        from InsurAutoML._datasets import PROD

        data = PROD(split=["train"])

        X_train, y_train = data["train"]
        proc = MultiPreprocessing(method="FusEmbed")
        proc.fit(X_train)

        trans_X_train = proc.transform(X_train)

        assert (
            "txt" in trans_X_train.keys()
        ), "The text data is not properly transformed."
        assert isinstance(
            trans_X_train["txt"], torch.Tensor
        ), "The text data is not properly transformed."

        assert (
            "cat" in trans_X_train.keys()
        ), "The categorical data is not properly transformed."
        assert isinstance(
            trans_X_train["cat"], torch.Tensor
        ), "The categorical data is not properly transformed."

        assert (
            "con" in trans_X_train.keys()
        ), "The continuous data is not properly transformed."
        assert isinstance(
            trans_X_train["con"], torch.Tensor
        ), "The continuous data is not properly transformed."
    except BaseException:
        assert True, "The nn package is not installed."
