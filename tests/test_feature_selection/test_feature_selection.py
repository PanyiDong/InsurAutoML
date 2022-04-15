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
Last Modified: Friday, 15th April 2022 2:16:50 pm
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

from distutils.ccompiler import new_compiler
from numpy import percentile
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
        elif method_name == "extra_trees_preproc_for_classification":
            feature_selection = method(
                n_estimators=5,
                criterion="entropy",
                min_samples_leaf=5,
                min_samples_split=5,
                max_features=0.5,
                bootstrap=False,
                max_leaf_nodes=None,
                max_depth=None,
                min_weight_fraction_leaf=0.0,
                min_impurity_decrease=0.0,
            )
        elif method_name == "extra_trees_preproc_for_regression":
            feature_selection = method(
                n_estimators=5,
                criterion="mse",
                min_samples_leaf=5,
                min_samples_split=5,
                max_features=0.5,
                bootstrap=False,
                max_leaf_nodes=None,
                max_depth=None,
                min_weight_fraction_leaf=0.0,
            )
        elif method_name == "fast_ica":
            feature_selection = method(
                algorithm="parallel",
                whiten=False,
                fun="logcosh",
                n_component=5,
            )
        elif method_name == "feature_agglomeration":
            feature_selection = method(
                n_clusters=5,
                affinity="euclidean",
                linkage="ward",
                pooling_func="mean",
            )
        elif method_name == "kernel_pca":
            feature_selection = method(
                n_components=5,
                kernel="rbf",
                gamma=0.1,
                degree=3,
                coef0=0.5,
            )
        elif method_name == "kitchen_sinks":
            feature_selection = method(
                gamma=0.1,
                new_comonent=50,
            )
        elif method_name == "liblinear_svc_preprocessor":
            feature_selection = method(
                penalty="l1",
                loss="squared_hinge",
                dual=False,
                tol=0.0001,
                C=1.0,
                multi_class="ovr",
                fit_intercept=True,
                intercept_scaling=1,
            )
        elif method_name == "nystroem_sampler":
            feature_selection = method(
                kernel="rbf",
                n_components=50,
                gamma=0.1,
                degree=3,
                coef0=0.5,
            )
        elif method_name == "pca":
            feature_selection = method(
                keep_variance=0.5,
                whiten=True,
            )
        elif method_name == "polynomial":
            feature_selection = method(
                degree=3,
                interaction_only=False,
                include_bias=True,
            )
        elif method_name == "random_trees_embedding":
            feature_selection = method(
                n_estimators=5,
                max_depth=3,
                min_samples_split=5,
                min_samples_leaf=5,
                min_weight_fraction_leaf=1.0,
                max_leaf_nodes=None,
                bootstrap=True,
            )
        elif method_name == "select_percentile_classification":
            feature_selection = method(
                percentile=90,
                score_func="chi2",
            )
        elif method_name == "select_percentile_regression":
            feature_selection = method(
                percentile=90,
                score_func="f_regression",
            )
        elif method_name == "select_rates_classification":
            feature_selection = method(
                alpha=0.3,
                score_func="chi2",
                mode="fpr",
            )
        elif method_name == "select_rates_regression":
            feature_selection = method(
                alpha=0.3,
                score_func="f_regression",
                mode="fpr",
            )
        elif method_name == "truncatedSVD":
            feature_selection = method(
                target_dim=5,
            )
        elif method_name == "no_processing":
            feature_selection = method()
        else:
            raise ValueError("Unknown method name : {}.".format(method_name))

        feature_selection.fit(X, y)
        _X = feature_selection.transform(X)

        if method_name != "polynomial":
            assert (
                _X.shape[1] <= X.shape[1]
            ), "Feature selection method {} failed".format(method_name)
