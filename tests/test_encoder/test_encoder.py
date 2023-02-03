"""
File Name: test_encoder.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /tests/test_encoder/test_encoder.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 28th November 2022 11:35:20 pm
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


def test_encoder_1():

    from InsurAutoML.encoding import DataEncoding
    from InsurAutoML import load_data

    database = load_data().load("Appendix", "Employee")

    encoder = DataEncoding(dummy_coding=True, transform=False)
    encoder.fit(database["Employee"])
    data = encoder.refit(database["Employee"])

    # check whether the method is fitted
    assert encoder._fitted, "The encoder is not correctly fitted."


def test_encoder_2():

    from InsurAutoML.encoding import DataEncoding
    from InsurAutoML import load_data

    database = load_data().load("Appendix", "Employee")

    encoder = DataEncoding(dummy_coding=False, transform="standardize")
    encoder.fit(database["Employee"])
    data = encoder.refit(database["Employee"])

    # check whether the method is fitted
    assert encoder._fitted, "The encoder is not correctly fitted."


def test_encoder_3():

    from InsurAutoML.encoding import DataEncoding
    from InsurAutoML import load_data

    database = load_data().load("Appendix", "Employee")

    encoder = DataEncoding(dummy_coding=False, transform="center")
    encoder.fit(database["Employee"])
    data = encoder.refit(database["Employee"])

    # check whether the method is fitted
    assert encoder._fitted, "The encoder is not correctly fitted."


def test_encoder_4():

    from InsurAutoML.encoding import DataEncoding
    from InsurAutoML import load_data

    database = load_data().load("Appendix", "Employee")

    encoder = DataEncoding(dummy_coding=False, transform="log")
    encoder.fit(database["Employee"])
    data = encoder.refit(database["Employee"])

    # check whether the method is fitted
    assert encoder._fitted, "The encoder is not correctly fitted."
