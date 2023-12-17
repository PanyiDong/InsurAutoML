"""
File Name: test_base.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /tests/test_base.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 28th November 2022 11:36:58 pm
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


def test_load_data():
    from InsurAutoML.base import load_data
    import importlib

    rpy2_spec = importlib.util.find_spec("rpy2")

    database = load_data().load("Appendix")

    database_names = [
        "heart_2020_cleaned",
        "Employee",
        "insurance",
        "Medicalpremium",
        "TravelInsurancePrediction",
        "healthcare-dataset-stroke-data",
        "heart",
        "hurricanehist",
        "HealthInsurance",
        "ImbalancedInsurance",
        # "credit",
    ]

    if rpy2_spec is not None:
        database_names.append("credit")
        assert set(database.keys()) == set(
            database_names
        ), "Not all databases are loaded."
    else:
        assert set(database.keys()) == set(
            database_names
        ), "Not all databases are loaded."
