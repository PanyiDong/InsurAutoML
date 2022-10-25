"""
File: _file.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Latest Version: 0.2.0
Relative Path: /My_AutoML/_utils/_file.py
File Created: Wednesday, 6th April 2022 6:25:09 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Sunday, 17th April 2022 1:02:29 am
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

import os
import pickle

# save model
def save_model(
    encoder,
    encoder_hyperparameters,
    imputer,
    imputer_hyperparameters,
    balancing,
    balancing_hyperparameters,
    scaling,
    scaling_hyperparameters,
    feature_selection,
    feature_selection_hyperparameters,
    model,
    model_hyperparameters,
    model_name,
):
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
def save_methods(file_name, methods):

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
def load_methods(file_name):

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
def find_exact_path(path, spec_str) :
    
    for folder in os.listdir(path):
        
        if spec_str in os.path.join(path, folder):
            return os.path.join(path, folder)