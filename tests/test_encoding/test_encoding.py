"""
File: test_encoding.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Latest Version: 0.2.0
Relative Path: /tests/test_encoding/test_encoding.py
File Created: Friday, 15th April 2022 7:25:45 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Friday, 15th April 2022 7:40:41 pm
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

from My_AutoML import load_data
from My_AutoML._encoding import DataEncoding
from My_AutoML import train_test_split


def test_encoding_1():

    data = load_data().load("Appendix", "insurance")
    data = data["insurance"]

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    train_X, test_X, train_y, test_y = train_test_split(X, y)

    encoder = DataEncoding(dummy_coding=True, transform=False)
    encoder.fit(train_X)

    transformed_X = encoder.refit(test_X)

    assert encoder._fitted == True, "Encoder not correctly fitted"
    assert transformed_X.shape[1] == 11, "Transformed X has wrong shape"


def test_encoding_2():

    data = load_data().load("Appendix", "insurance")
    data = data["insurance"]

    encoder = DataEncoding(dummy_coding=False, transform="standardize")

    encoder.fit(data)

    assert encoder._fitted == True, "Encoder not correctly fitted"


def test_encoding_3():

    data = load_data().load("Appendix", "insurance")
    data = data["insurance"]

    encoder = DataEncoding(dummy_coding=False, transform="center")

    encoder.fit(data)

    assert encoder._fitted == True, "Encoder not correctly fitted"


def test_encoding_4():

    data = load_data().load("Appendix", "insurance")
    data = data["insurance"]

    encoder = DataEncoding(dummy_coding=False, transform="log")

    encoder.fit(data)

    assert encoder._fitted == True, "Encoder not correctly fitted"
