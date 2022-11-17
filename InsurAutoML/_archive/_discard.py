"""
File: _discard.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Relative Path: /archive/_discard.py
File Created: Friday, 25th February 2022 6:13:42 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 24th October 2022 10:52:37 pm
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
import shutil
import glob
import numpy as np
import pandas as pd
import json
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
import autosklearn
from autosklearn.pipeline.components.feature_preprocessing.no_preprocessing import (
    NoPreprocessing, )
from autosklearn.pipeline.components.feature_preprocessing.densifier import Densifier
from autosklearn.pipeline.components.feature_preprocessing.extra_trees_preproc_for_classification import (
    ExtraTreesPreprocessorClassification, )
from autosklearn.pipeline.components.feature_preprocessing.extra_trees_preproc_for_regression import (
    ExtraTreesPreprocessorRegression, )
from autosklearn.pipeline.components.feature_preprocessing.fast_ica import FastICA
from autosklearn.pipeline.components.feature_preprocessing.feature_agglomeration import (
    FeatureAgglomeration, )
from autosklearn.pipeline.components.feature_preprocessing.kernel_pca import KernelPCA
from autosklearn.pipeline.components.feature_preprocessing.kitchen_sinks import (
    RandomKitchenSinks, )
from autosklearn.pipeline.components.feature_preprocessing.liblinear_svc_preprocessor import (
    LibLinear_Preprocessor, )
from autosklearn.pipeline.components.feature_preprocessing.nystroem_sampler import (
    Nystroem, )
from autosklearn.pipeline.components.feature_preprocessing.pca import PCA
from autosklearn.pipeline.components.feature_preprocessing.polynomial import (
    PolynomialFeatures,
)
from autosklearn.pipeline.components.feature_preprocessing.random_trees_embedding import (
    RandomTreesEmbedding, )
from autosklearn.pipeline.components.feature_preprocessing.select_percentile import (
    SelectPercentileBase, )
from autosklearn.pipeline.components.feature_preprocessing.select_percentile_classification import (
    SelectPercentileClassification, )
from autosklearn.pipeline.components.feature_preprocessing.select_percentile_regression import (
    SelectPercentileRegression, )
from autosklearn.pipeline.components.feature_preprocessing.select_rates_classification import (
    SelectClassificationRates, )
from autosklearn.pipeline.components.feature_preprocessing.select_rates_regression import (
    SelectRegressionRates, )
from autosklearn.pipeline.components.feature_preprocessing.truncatedSVD import (
    TruncatedSVD, )

from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier
from sklearn.neural_network import MLPClassifier

# R environment
import rpy2
import rpy2.robjects as ro
from rpy2.robjects import Formula, pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from sklearn.tree import ExtraTreeClassifier

from InsurAutoML._encoding import DataEncoding

_feature_mol: dict = {
    "no_preprocessing": NoPreprocessing,
    "densifier": Densifier,
    "extra_trees_preproc_for_classification": ExtraTreesPreprocessorClassification,
    "extra_trees_preproc_for_regression": ExtraTreesPreprocessorRegression,
    "fast_ica": FastICA,
    "feature_agglomeration": FeatureAgglomeration,
    "kernel_pca": KernelPCA,
    "kitchen_sinks": RandomKitchenSinks,
    "liblinear_svc_preprocessor": LibLinear_Preprocessor,
    "nystroem_sampler": Nystroem,
    "pca": PCA,
    "polynomial": PolynomialFeatures,
    "random_trees_embedding": RandomTreesEmbedding,
    "select_percentile": SelectPercentileBase,
    "select_percentile_classification": SelectPercentileClassification,
    "select_percentile_regression": SelectPercentileRegression,
    "select_rates_classification": SelectClassificationRates,
    "select_rates_regression": SelectRegressionRates,
    "truncatedSVD": TruncatedSVD,
}


class feature_selection:

    """
    Restrict the feature selection methods to test model performance

    Not yet ready
    """

    def __init__(
        self,
        test_perc=0.15,
        task_type="classification",
        seed=1,
        _base_info="My_AutoML/_database_info.json",
        feature_mol="all",
        temp_loc="tmp/",
        time_left_for_this_task=120,
        per_run_time_limit=5,
        ensemble_size=50,
        skip=False,
    ):
        self.test_perc = test_perc
        self.task_type = task_type
        self.seed = seed
        self._base_info = _base_info
        self.feature_mol = feature_mol
        self.temp_loc = temp_loc if temp_loc[-1] == "/" else temp_loc + "/"
        self.time_left_for_this_task = (
            time_left_for_this_task  # total time in seconds to find and tune the models
        )
        # time in seconds to fit machine learning models per call
        self.per_run_time_limit = per_run_time_limit
        self.ensemble_size = ensemble_size
        self.skip = skip

    def database_test(self, database):

        # set feature selection models
        if self.feature_mol == "all":
            self.feature_mol = [*_feature_mol]
        else:
            self.feature_mol = self.feature_mol

        # check if any models unknown
        for _mol in self.feature_mol:
            if _mol not in [*_feature_mol]:
                raise ValueError(
                    "{0} not avaiable! All possible models are: {1}.".format(
                        _mol, [*_feature_mol]
                    )
                )

        # get the database infomation (from json file)
        _base_info = json.load(open(self._base_info))

        # Loop through database and test performance among feature selection
        # methods
        database_names = [*database]
        for _name in database_names:
            if self.skip:
                self._skip_data_test(database[_name], _name, _base_info)
            else:
                self._data_test(database[_name], _name, _base_info)

    def _skip_data_test(self, data, data_name, base_info):

        tmp_folder = "{0}{1}_temp".format(self.temp_loc, data_name)

        # extract data information from database information and get response
        # name
        data_info = next(
            item for item in base_info if item["filename"] == data_name)
        response = next(
            item
            for item in data_info["property"]
            if item["task_type"] == self.task_type
        )["response"]

        # create empty temp folder to store feature_selection performance
        if os.path.isdir(tmp_folder):
            shutil.rmtree(tmp_folder)
        os.makedirs(tmp_folder)

        if response == "None":
            warnings.warn(
                "{0} not avaiable for {1}.".format(
                    data_name, self.task_type))
        else:
            # feature list
            features = list(data.columns)
            if isinstance(response, list):  # deal with multiple response data
                for _response in response:
                    features.remove(_response)
            else:
                features.remove(response)
            if isinstance(response, list):
                response = response[0]

            # write basic information to data_info file
            data_info_file = open(tmp_folder + "/data_info.txt", "a")
            data_info_file.write("Data name: {}\n".format(data_name))
            data_info_file.write("Task type: {}\n".format(self.task_type))
            data_info_file.write(
                "Number of samples: {},\nNumber of features: {}\n".format(
                    data[features].shape[0], data[features].shape[1]
                )
            )
            data_info_file.write("Features: {}\n".format(features))
            data_info_file.write("Response: {}\n".format(response))
            data_info_file.close()

            # preprocess string type
            preprocessor = DataEncoding(dummy_coding=False, transform=False)
            new_data = preprocessor.fit(data[features])

            # train_test_divide
            # need to split after feature selection, since fix seed, not
            # impacted by randomness
            X_train, X_test, y_train, y_test = train_test_split(
                new_data,
                data[[response]],
                test_size=self.test_perc,
                random_state=self.seed,
            )

            if self.task_type == "classification":
                automl = autosklearn.classification.AutoSklearnClassifier(
                    seed=self.seed,
                    time_left_for_this_task=self.time_left_for_this_task,
                    per_run_time_limit=self.per_run_time_limit,
                    ensemble_size=self.ensemble_size,
                    tmp_folder=tmp_folder + "/Auto-sklearn temp",
                    delete_tmp_folder_after_terminate=False,
                )
            elif self.task_type == "regression":
                automl = autosklearn.regression.AutoSklearnRegressor(
                    seed=self.seed,
                    time_left_for_this_task=self.time_left_for_this_task,
                    per_run_time_limit=self.per_run_time_limit,
                    ensemble_size=self.ensemble_size,
                    tmp_folder=tmp_folder + "/Auto-sklearn temp",
                    delete_tmp_folder_after_terminate=False,
                )

            automl.fit(X_train, y_train, dataset_name=data_name)
            fitting = automl.predict(X_train)
            predictions = automl.predict(X_test)

            result_file = open("{}/result.txt".format(tmp_folder), "a")
            result_file.write(
                "Train accuracy: {}\n".format(accuracy_score(y_train, fitting))
            )
            result_file.write(
                "Train MAE: {}\n".format(mean_absolute_error(y_train, fitting))
            )
            result_file.write(
                "Train MSE: {}\n".format(mean_squared_error(y_train, fitting))
            )
            result_file.write(
                "Test accuracy: {}\n".format(
                    accuracy_score(
                        y_test, predictions)))
            result_file.write(
                "Test MAE: {}\n".format(
                    mean_absolute_error(
                        y_test, predictions)))
            result_file.write(
                "Test MSE: {}\n".format(
                    mean_squared_error(
                        y_test, predictions)))
            result_file.close()

    def _data_test(self, data, data_name, base_info):

        tmp_folder = "{0}{1}_temp/feature_selection".format(
            self.temp_loc, data_name)

        # extract data information from database information and get response
        # name
        data_info = next(
            item for item in base_info if item["filename"] == data_name)
        response = next(
            item
            for item in data_info["property"]
            if item["task_type"] == self.task_type
        )["response"]

        # create empty temp folder to store feature_selection performance
        if os.path.isdir(tmp_folder):
            shutil.rmtree(tmp_folder)
        os.makedirs(tmp_folder)

        if response == "None":
            warnings.warn(
                "{0} not avaiable for {1}.".format(
                    data_name, self.task_type))
        else:
            # feature list
            features = list(data.columns)
            if isinstance(response, list):  # deal with multiple response data
                for _response in response:
                    features.remove(_response)
            else:
                features.remove(response)
            if isinstance(response, list):
                response = response[0]

            # write basic information to data_info file
            data_info_file = open(tmp_folder + "/data_info.txt", "a")
            data_info_file.write("Data name: {}\n".format(data_name))
            data_info_file.write("Task type: {}\n".format(self.task_type))
            data_info_file.write(
                "Number of samples: {},\nNumber of features: {}\n".format(
                    data[features].shape[0], data[features].shape[1]
                )
            )
            data_info_file.write("Features: {}\n".format(features))
            data_info_file.write("Response: {}\n".format(response))
            data_info_file.close()

            # after processing folder
            new_info_folder = tmp_folder + "/selected_data_info"
            if os.path.isdir(new_info_folder):
                shutil.rmtree(new_info_folder)
            os.makedirs(new_info_folder)

            for _mol in self.feature_mol:

                # preprocess string type
                preprocessor = DataEncoding(
                    dummy_coding=False, transform=False)
                new_data = preprocessor.fit(data[features])

                # train_test_divide
                # need to split after feature selection, since fix seed, not
                # impacted by randomness
                X_train, X_test, y_train, y_test = train_test_split(
                    new_data,
                    data[[response]],
                    test_size=self.test_perc,
                    random_state=self.seed,
                )

                # write basic information to data_info file
                new_info_file = open(
                    "{}/{}_info.txt".format(new_info_folder, _mol), "a"
                )
                new_info_file.write("Data name: {}\n".format(data_name))
                new_info_file.write("Task type: {}\n".format(self.task_type))
                new_info_file.write(
                    "Number of train samples: {},\nNumber of test samples: {},\nNumber of features: {}\n".format(
                        X_train.shape[0], X_test.shape[0], X_train.shape[1]))
                new_info_file.write("Features: {}\n".format(features))
                new_info_file.write("Response: {}\n".format(response))
                new_info_file.close()

                if self.task_type == "classification":
                    automl = autosklearn.classification.AutoSklearnClassifier(
                        include={"feature_preprocessor": [_mol]},
                        seed=self.seed,
                        time_left_for_this_task=self.time_left_for_this_task,
                        per_run_time_limit=self.per_run_time_limit,
                        ensemble_size=self.ensemble_size,
                        tmp_folder="{}/{}".format(new_info_folder, _mol),
                        delete_tmp_folder_after_terminate=False,
                    )
                elif self.task_type == "regression":
                    automl = autosklearn.regression.AutoSklearnRegressor(
                        include={"feature_preprocessor": [_mol]},
                        seed=self.seed,
                        time_left_for_this_task=self.time_left_for_this_task,
                        per_run_time_limit=self.per_run_time_limit,
                        ensemble_size=self.ensemble_size,
                        tmp_folder="{}/{}".format(new_info_folder, _mol),
                        delete_tmp_folder_after_terminate=False,
                    )

                automl.fit(X_train, y_train, dataset_name=data_name)
                fitting = automl.predict(X_train)
                predictions = automl.predict(X_test)

                result_file = open(
                    "{}/{}_result.txt".format(new_info_folder, _mol), "a"
                )
                result_file.write(
                    "Train accuracy: {}\n".format(
                        accuracy_score(
                            y_train, fitting)))
                result_file.write(
                    "Train MAE: {}\n".format(
                        mean_absolute_error(
                            y_train, fitting)))
                result_file.write(
                    "Train MSE: {}\n".format(
                        mean_squared_error(
                            y_train, fitting)))
                result_file.write(
                    "Test accuracy: {}\n".format(
                        accuracy_score(
                            y_test, predictions)))
                result_file.write(
                    "Test MAE: {}\n".format(
                        mean_absolute_error(
                            y_test, predictions)))
                result_file.write(
                    "Test MSE: {}\n".format(
                        mean_squared_error(
                            y_test, predictions)))
                result_file.close()

                # # perform feature selection on the data
                # if _feature_mol[_mol] == ExtraTreesPreprocessorClassification :
                #     _fit_mol = _feature_mol[_mol](
                #         n_estimators = 100,
                #         criterion = "gini",
                #         min_samples_leaf = 1,
                #         min_samples_split = 2,
                #         max_features = "auto",
                #         bootstrap = False,
                #         max_leaf_nodes = None,
                #         max_depth = None,
                #         min_weight_fraction_leaf = 0.0,
                #         min_impurity_decrease = 0.0,
                #         oob_score=False,
                #         n_jobs=1,
                #         random_state=self.seed,
                #         verbose=0,
                #         class_weight=None
                #     )
                # elif _feature_mol[_mol] == ExtraTreesPreprocessorRegression :
                #     _fit_mol = _feature_mol[_mol](
                #         n_estimators = 100,
                #         criterion = "squared_error",
                #         min_samples_leaf = 1,
                #         min_samples_split = 2,
                #         max_features = "auto",
                #         bootstrap=False,
                #         max_leaf_nodes=None,
                #         max_depth="None",
                #         min_weight_fraction_leaf=0.0,
                #         oob_score=False,
                #         n_jobs=1,
                #         random_state=None,
                #         verbose=0
                #     )
                # else :
                #     _fit_mol = _feature_mol[_mol](random_state = self.seed)
                # if _feature_mol[_mol] == ExtraTreesPreprocessorClassification or _feature_mol[_mol] == ExtraTreesPreprocessorRegression :
                #     _fit_mol.fit(data[features], data[[response]])
                # else :
                #     _fit_mol.fit(data[features])
                # _data = _fit_mol.transform(data[features])
