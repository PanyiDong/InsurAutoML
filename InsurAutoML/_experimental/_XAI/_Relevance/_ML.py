"""
File Name: _ML.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_experimental/_XAI/_Relevance/_ML.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 14th November 2022 8:27:20 pm
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

from sklearn.linear_model import LinearRegression
import pandas as pd

# linear regression with relevance attached
class LinearRegressor(LinearRegression):
    def __init__(
        self,
    ):
        super().__init__()

    def fit(self, X, y):

        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)

        super().fit(X, y)

        return self

    def predict(self, X):
        return super().predict(X)

    def format(self, relevance, how="absolute"):

        if hasattr(self, "_feature_names"):
            for _feature, _relevance in zip(self._feature_names, relevance):
                if how == "absolute":
                    print("{:40s}: {:14.6f}".format(_feature, _relevance))
                elif how == "relative":
                    print("{:40s}: {:10.4f}%".format(_feature, _relevance * 100))
        else:
            print(relevance)

    def Relevance(self, output, how="absolute", print_result=False):

        weight = self.coef_**2
        weight /= weight.sum()

        if how == "absolute":
            self.relevance = weight * output
        elif how == "relative":
            self.relevance = weight

        if print_result:
            self.format(self.relevance, how=how)
        else:
            return self.relevance

    @property
    def feature_names(self):

        if hasattr(self, "_feature_names"):
            return self._feature_names
        else:
            raise AttributeError("No feature names found")
