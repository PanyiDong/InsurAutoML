"""
File Name: _file.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_utils/_file.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 14th November 2022 9:27:47 pm
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

from typing import Dict, List
import os
import pickle

# save model
def save_model(
    encoder: str,
    encoder_hyperparameters: Dict,
    imputer: str,
    imputer_hyperparameters: Dict,
    balancing: str,
    balancing_hyperparameters: Dict,
    scaling: str,
    scaling_hyperparameters: Dict,
    feature_selection: str,
    feature_selection_hyperparameters: Dict,
    model: str,
    model_hyperparameters: Dict,
    model_name: str,
) -> None:
    with open(model_name, "w") as f:
        f.write("{}\n".format(encoder))
        print(encoder_hyperparameters, file=f, end="\n")
        f.write("{}\n".format(imputer))
        print(imputer_hyperparameters, file=f, end="\n")
        f.write("{}\n".format(balancing))
        print(balancing_hyperparameters, file=f, end="\n")
        f.write("{}\n".format(scaling))
        print(scaling_hyperparameters, file=f, end="\n")
        f.write("{}\n".format(feature_selection))
        print(feature_selection_hyperparameters, file=f, end="\n")
        f.write("{}\n".format(model))
        print(model_hyperparameters, file=f, end="\n")


# save list of methods
def save_methods(file_name: str, methods: List) -> None:

    """
    Parameters
    ----------
    file_name: path of the file to save

    methods: list of methods objects to save
    """

    with open(file_name, "wb") as out_f:
        for method in methods:
            pickle.dump(method, out_f)


# load methods
def load_methods(file_name: str) -> List:

    """
    Parameters
    ----------
    file_name: path of the file to load
    """

    with open(file_name, "rb") as in_f:
        results = []

        # load all methods
        while True:
            try:
                results.append(pickle.load(in_f))
            except EOFError:
                break

    return results


# find exact folder path
def find_exact_path(path: str, spec_str: str) -> str:

    for folder in os.listdir(path):

        if spec_str in os.path.join(path, folder):
            return os.path.join(path, folder)
