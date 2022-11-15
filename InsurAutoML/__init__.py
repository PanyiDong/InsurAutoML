"""
File: __init__.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/__init__.py
File: __init__.py
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Tuesday, 15th November 2022 4:04:57 pm
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

# read version from file
from .version import _get_version

__version__ = _get_version()

from ._base import no_processing, load_data
from ._utils import (
    # random_guess,
    # random_index,
    # random_list,
    # is_date,
    # feature_rounding,
    train_test_split,
    # minloc,
    # maxloc,
    # True_index,
    # nan_cov,
    # class_means,
    # empirical_covariance,
    # class_cov,
    # Pearson_Corr,
    # MI,
    # t_score,
    # ANOVA,
    # as_dataframe,
    type_of_task,
    # formatting,
    # Timer,
    # unify_nan,
    # remove_index_columns,
    # get_missing_matrix,
    EDA,
)

# from ._encoding import DataEncoding

# from ._imputation import (
#     SimpleImputer,
#     DummyImputer,
#     JointImputer,
#     ExpectationMaximization,
#     KNNImputer,
#     MissForestImputer,
#     MICE,
#     GAIN,
#     AAI_kNN,
#     KMI,
#     CMI,
#     k_Prototype_NN,
# )

# from ._balancing import (
#     SimpleRandomOverSampling,
#     SimpleRandomUnderSampling,
#     TomekLink,
#     EditedNearestNeighbor,
#     CondensedNearestNeighbor,
#     OneSidedSelection,
#     CNN_TomekLink,
#     Smote,
#     Smote_TomekLink,
#     Smote_ENN,
# )

# from ._scaling import (
#     MinMaxScale,
#     Standardize,
#     Normalize,
#     RobustScale,
#     PowerTransformer,
#     QuantileTransformer,
#     Winsorization,
#     Feature_Manipulation,
#     Feature_Truncation,
# )

# from ._feature_selection import (
#     PCA_FeatureSelection,
#     LDASelection,
#     FeatureFilter,
#     RBFSampler,
#     ASFFS,
#     GeneticAlgorithm,
# )

# extracted from sklearn
# not all used in the pipeline
# from ._feature_selection import (
#     Densifier,
#     ExtraTreesPreprocessorClassification,
#     ExtraTreesPreprocessorRegression,
#     FastICA,
#     FeatureAgglomeration,
#     KernelPCA,
#     RandomKitchenSinks,
#     LibLinear_Preprocessor,
#     Nystroem,
#     PCA,
#     PolynomialFeatures,
#     RandomTreesEmbedding,
#     SelectPercentileClassification,
#     SelectPercentileRegression,
#     SelectClassificationRates,
#     SelectRegressionRates,
#     TruncatedSVD,
# )

from ._hpo import (
    AutoTabular,
    AutoTabularClassifier,
    AutoTabularRegressor,
    AutoTextClassifier,
    AutoNextWordPrediction,
)

# from ._model import classifiers, regressors

# base = {"load_data": load_data}

# encoders = {"DataEncoding": DataEncoding}

# model_selection = {
#     "AutoTabular": AutoTabular,
#     "AutoTabularClassifier": AutoTabularClassifier,
#     "AutoTabularRegressor": AutoTabularRegressor,
#     "AutoTextClassifier": AutoTextClassifier,
#     "AutoNextWordPrediction": AutoNextWordPrediction,
# }

__all__ = [
    "load_data",  # _base
    "no_processing",
    "random_guess",  # _utils
    "random_index",
    "random_list",
    "is_date",
    "feature_rounding",
    "train_test_split",
    "minloc",
    "maxloc",
    "True_index",
    "nan_cov",
    "class_means",
    "empirical_covariance",
    "class_cov",
    "Pearson_Corr",
    "MI",
    "t_score",
    "ANOVA",
    "as_dataframe",
    "type_of_task",
    "formatting",
    "Timer",
    "unify_nan",
    "remove_index_columns",
    "get_missing_matrix",
    "SimpleImputer",  # _missing
    "DummyImputer",
    "JointImputer",
    "ExpectationMaximization",
    "KNNImputer",
    "MissForestImputer",
    "MICE",
    "GAIN",
    "AAI_kNN",
    "KMI",
    "CMI",
    "k_Prototype_NN",
    "DataEncoding",  # _encoding
    "MinMaxScale",  # _scaling
    "Standardize",
    "Normalize",
    "RobustScale",
    "PowerTransformer",
    "QuantileTransformer",
    "Winsorization",
    "Feature_Manipulation",
    "Feature_Truncation",
    "SimpleRandomOverSampling",  # _imbalance
    "SimpleRandomUnderSampling",
    "TomekLink",
    "EditedNearestNeighbor",
    "CondensedNearestNeighbor",
    "OneSidedSelection",
    "CNN_TomekLink",
    "Smote",
    "Smote_TomekLink",
    "Smote_ENN",
    "LDASelection",  # _feature_selection
    "PCA_FeatureSelection",
    "RBFSampler",
    "FeatureFilter",
    "ASFFS",
    "GeneticAlgorithm",
    "densifier",  # from sklearn
    "extra_trees_preproc_for_classification",
    "extra_trees_preproc_for_regression",
    "fast_ica",
    "feature_agglomeration",
    "kernel_pca",
    "kitchen_sinks",
    "liblinear_svc_preprocessor",
    "nystroem_sampler",
    "pca",
    "polynomial",
    "random_trees_embedding",
    #    'select_percentile',
    "select_percentile_classification",
    "select_percentile_regression",
    "select_rates_classification",
    "select_rates_regression",
    "truncatedSVD",
    "classifiers",  # _model
    "regressors",
    "AutoTabular",  # _model_selection
    "AutoTabularClassifier",
    "AutoTabularRegressor",
    "AutoTextClassifier",
    "AutoNextWordPrediction",
]
