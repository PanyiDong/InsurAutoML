"""
File: test_metadata.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /tests/test_utils/test_metadata.py
File: test_metadata.py
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Wednesday, 16th November 2022 11:54:23 am
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


def test_MetaData():

    from InsurAutoML._utils._metadata import MetaData
    from InsurAutoML._datasets import PROD

    data = PROD(split=["train"])

    X_train, y_train = data["train"]

    metadata = MetaData(X_train)

    expected_return = {
        ("Int", "Numerical"): ["Unnamed: 0", "Text_ID"],
        ("Object", "Text"): ["Product_Description"],
        ("Int", "Categorical"): ["Product_Type"],
    }

    X_train["Product_Type"] = X_train["Product_Type"].astype(str)
    metadata.update(X_train)

    metadata.force_update(["Product_Type"], [("Int", "Categorical")])
    print(metadata)

    assert (metadata.metadata[("Int", "Numerical")] == expected_return[(
        "Int", "Numerical")]), "Numerical columns are not correct."
    assert (metadata.metadata[("Object", "Text")] == expected_return[(
        "Object", "Text")]), "Text columns are not correct."
    assert (
        metadata.metadata[("Int", "Categorical")]
        == expected_return[("Int", "Categorical")]
    ), "Categorical columns are not correct."
