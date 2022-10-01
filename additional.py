"""
File: add.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Last Version: 0.2.1
Relative Path: /add.py
File Created: Saturday, 1st October 2022 9:15:59 am
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Saturday, 1st October 2022 1:28:20 pm
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

"""
Few common hyperparameter constructions:
1. Uniform distribution: tune.uniform(lower, upper)
2. Discrete uniform distribution with interval q: tune.quniform(lower, upper, q)
3. Radom integer between [low, upper): tune.randint(lower, upper)
4. Choice from a list: tune.choice([a, b, c])

For full guides of the hyperparameters, please refer to the following links:
https://docs.ray.io/en/latest/tune/api_docs/search_space.html
"""

"""
Allowed components:
1. add_encoders           and   add_encoder_hyperparameter
2. add_imputers           and   add_imputer_hyperparameter
3. add_balancings         and   add_balancing_hyperparameter
4. add_scalings           and   add_scaling_hyperparameter
5. add_feature_selections and   add_feature_selection_hyperparameter
6. add_regressors         and   add_regressor_hyperparameter
7. add_classifiers        and   add_classifier_hyperparameter

Each of the methods is stored in corresponding dictionaries, 
hyperparameters stored in corresponding lists.

For methods, the keys are the name of the methods, which should be the same 
as the method name in the corresponding class.

For hyperparameters, the identification key should be procedure + "_add_" + order,
e.g. "encoder_add_1" for the first encoder added to the pipeline while the value is the
name of the method, corresponding the ones stored in dictionaries. For other key/values,
keys should be method name + "_" + hyperparameter name, e.g. "DataEncoding_dummy_coding" 
for the hyperparameter "dummy_coding" of the method "DataEncoding". The value should be
the range of the hyperparameter.

Below is a workable example for you to follow. You just need to add your own methods, 
if not, leave it as it is.
"""

####################################################################################################################
# Template of self-defined models and hyperparameters

# from ray import tune
# from My_AutoML._encoding import DataEncoding
# from My_AutoML._imputation import SimpleImputer
# from My_AutoML._balancing import SimpleRandomOverSampling
# from My_AutoML._scaling import Standardize
# from My_AutoML._feature_selection import mRMR
# from My_AutoML._model import LinearRegression, LogisticRegression

# add_encoders = {"DataEncoding": DataEncoding}

# add_encoder_hyperparameter = [
#     {
#         "encoder_add_1": "DataEncoding",
#         "DataEncoding_dummy_coding": tune.choice([True, False]),
#     },
# ]

# add_imputers = {
#     "SimpleImputer": SimpleImputer,
# }

# add_imputer_hyperparameter = [
#     {
#         "imputer_add_1": "SimpleImputer",
#         "SimpleImputer_method": tune.choice(
#             ["mean", "zero", "median", "most frequent"]
#         ),
#     },
# ]

# add_balancings = {
#     "SimpleRandomOverSampling": SimpleRandomOverSampling,
# }

# add_balancing_hyperparameter = [
#     {
#         "balancing_add_1": "SimpleRandomOverSampling",
#         "SimpleRandomOverSampling_imbalance_threshold": tune.uniform(0.8, 1),
#     },
# ]

# add_scalings = {
#     "Standardize": Standardize,
# }

# add_scaling_hyperparameter = [
#     {"scaling_add_1": "Standardize"},
# ]

# add_feature_selections = {
#     "mRMR": mRMR,
# }

# add_feature_selection_hyperparameter = [
#     {
#         "feature_selection_add_1": "mRMR",
#         "mRMR_n_prop": tune.uniform(0, 1),
#     },
# ]

# add_regressors = {
#     "LinearRegression": LinearRegression,
# }

# add_regressor_hyperparameter = [
#     {
#         "model_add_1": "LinearRegression",
#     },
# ]

# add_classifiers = {
#     "LogisticRegression": LogisticRegression,
# }

# add_classifier_hyperparameter = [
#     {
#         "model_add_1": "LogisticRegression",
#         "LogisticRegression_penalty": tune.choice(["l2", "none"]),
#         "LogisticRegression_tol": tune.loguniform(1e-5, 1e-1),
#         "LogisticRegression_C": tune.loguniform(1e-5, 10),
#     },
# ]

####################################################################################################################
# Format of additional models and hyperparameters

add_encoders = {}

add_encoder_hyperparameter = []

add_imputers = {}

add_imputer_hyperparameter = []

add_balancings = {}

add_balancing_hyperparameter = []

add_scalings = {}

add_scaling_hyperparameter = []

add_feature_selections = {}

add_feature_selection_hyperparameter = []

add_regressors = {}

add_regressor_hyperparameter = []

add_classifiers = {}

add_classifier_hyperparameter = []
