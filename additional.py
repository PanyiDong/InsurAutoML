"""
File: additional.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Last Version: 0.2.1
Relative Path: /additional.py
File: additional.py
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 28th November 2022 11:32:31 pm
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

##########################################################################
# Template of self-defined models and hyperparameters

# from ray import tune
# from InsurAutoML.encoding import DataEncoding
# from InsurAutoML.imputation import SimpleImputer
# from InsurAutoML.balancing import SimpleRandomOverSampling
# from InsurAutoML.scaling import Standardize
# from InsurAutoML.feature_selection import mRMR
# from InsurAutoML.model import LinearRegression, LogisticRegression


# # additional encoders
# add_encoders = {"DataEncoding": DataEncoding}

# DATAENCODING = {
#     "encoder": "DataEncoding",
#     "dummy_coding": tune.choice([True, False]),
# }

# add_encoder_hyperparameter = [
#     DATAENCODING,
# ]

# # additional imputers
# add_imputers = {
#     "SimpleImputer": SimpleImputer,
# }

# SIMPLEIMPUTER = {
#     "imputer": "SimpleImputer",
#     "method": tune.choice(
#         ["mean", "zero", "median", "most frequent"]
#     ),
# }

# add_imputer_hyperparameter = [
#     SIMPLEIMPUTER,
# ]

# # additional balancings
# add_balancings = {
#     "SimpleRandomOverSampling": SimpleRandomOverSampling,
# }

# SIMPLERANDOMOVERSAMPLING = {
#     "balancing": "SimpleRandomOverSampling",
#     "imbalance_threshold": tune.uniform(0.8, 1),
# }

# add_balancing_hyperparameter = [
#     SIMPLERANDOMOVERSAMPLING,
# ]


# # additional scalings
# add_scalings = {
#     "Standardize": Standardize,
# }

# STANDARDIZE = {"scaling": "Standardize"}

# add_scaling_hyperparameter = [
#     STANDARDIZE,
# ]


# # additional feature selections
# add_feature_selections = {
#     "mRMR": mRMR,
# }

# MRMR = {
#     "feature_selection": "mRMR",
#     "n_prop": tune.uniform(0, 1),
# }

# add_feature_selection_hyperparameter = [
#     MRMR,
# ]


# # additional regressors
# add_regressors = {
#     "LinearRegression": LinearRegression,
# }

# LINEARREGRESSION = {
#     "model": "LinearRegression",
# }

# add_regressor_hyperparameter = [
#     LINEARREGRESSION,
# ]


# # additional classifiers
# add_classifiers = {
#     "LogisticRegression": LogisticRegression,
# }

# LOGISTICREGRESSION = {
#     "model": "LogisticRegression",
#     "penalty": tune.choice(["l2", "none"]),
#     "tol": tune.loguniform(1e-5, 1e-1),
#     "C": tune.loguniform(1e-5, 10),
# }

# add_classifier_hyperparameter = [
#     LOGISTICREGRESSION,
# ]

##########################################################################
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
