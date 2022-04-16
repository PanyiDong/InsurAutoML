"""
File: test_feature_selection.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Latest Version: 0.2.0
Relative Path: /tests/test_feature_selection/test_feature_selection.py
File Created: Friday, 15th April 2022 12:27:07 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Friday, 15th April 2022 5:46:26 pm
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

import pandas as pd
from My_AutoML._feature_selection import feature_selections


def test_feature_selection():

    # loop through all feature selection methods
    for method_name, method in zip(
        feature_selections.keys(), feature_selections.values()
    ):
        data = pd.read_csv("Appendix/Medicalpremium.csv")
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        if method_name in ["FeatureFilter", "ASFFS", "GeneticAlgorithm", "RBFSampler"]:
            feature_selection = method(n_components=5)
        else:
            feature_selection = method()

        feature_selection.fit(X, y)
        _X = feature_selection.transform(X)
         
        assert feature_selection._fitted == True, "Fitted should be True"
        if method_name != "polynomial":
            assert (
                _X.shape[1] <= X.shape[1]
            ), "Feature selection method {} failed".format(method_name)