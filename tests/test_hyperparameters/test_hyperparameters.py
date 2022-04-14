"""
File: test_hyperparameters.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Latest Version: 0.2.0
Relative Path: /tests/test_hyperparameters/test_hyperparameters.py
File Created: Thursday, 14th April 2022 12:25:53 am
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Thursday, 14th April 2022 12:31:22 am
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


def test_encoder_hyperparameters():
    
    from My_AutoML._hyperparameters import (
        encoder_hyperparameter,
    )
    
    assert (
        isinstance(encoder_hyperparameter, list)
    ), "Encoder hyperparameters correctly imported."
    
def test_imputer_hyperparameters():
    
    from My_AutoML._hyperparameters import (
        imputer_hyperparameter,
    )
    
    assert (
        isinstance(imputer_hyperparameter, list)
    ), "Imputer hyperparameters correctly imported."

def test_balancing_hyperparameters():
    
    from My_AutoML._hyperparameters import (
        balancing_hyperparameter,
    )
    
    assert (
        isinstance(balancing_hyperparameter, list)
    ), "Balancing hyperparameters correctly imported."
    
def test_scaling_hyperparameters():
    
    from My_AutoML._hyperparameters import (
        scaling_hyperparameter,
    )
    
    assert (
        isinstance(scaling_hyperparameter, list)
    ), "Scaling hyperparameters correctly imported."
    
def test_feature_selection_hyperparameters():
    
    from My_AutoML._hyperparameters import (
        feature_selection_hyperparameter,
    )
    
    assert (
        isinstance(feature_selection_hyperparameter, list)
    ), "Feature Selection hyperparameters correctly imported."
    
def test_regressor_hyperparameters():
    
    from My_AutoML._hyperparameters import (
        regressor_hyperparameter,
    )
    
    assert (
        isinstance(regressor_hyperparameter, list)
    ), "Regressor Selection hyperparameters correctly imported."
    
def test_classifier_hyperparameters():
    
    from My_AutoML._hyperparameters import (
        classifier_hyperparameter,
    )
    
    assert (
        isinstance(classifier_hyperparameter, list)
    ), "Classifier Selection hyperparameters correctly imported."