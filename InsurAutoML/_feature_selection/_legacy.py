"""
File Name: _legacy.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_feature_selection/_legacy.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 14th November 2022 7:05:30 pm
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

from ._embed import RBFSampler
from ._filter import FeatureFilter
from ._wrapper import ASFFS
from ._hybrid import GeneticAlgorithm
from InsurAutoML._base import no_processing

import autosklearn
import autosklearn.pipeline.components.feature_preprocessing

feature_selections = {
    "no_processing": no_processing,
    # "LDASelection": LDASelection,
    # "PCA_FeatureSelection": PCA_FeatureSelection,
    "RBFSampler": RBFSampler,
    "FeatureFilter": FeatureFilter,
    "ASFFS": ASFFS,
    "GeneticAlgorithm": GeneticAlgorithm,
    # 'densifier' : autosklearn.pipeline.components.feature_preprocessing.densifier.Densifier,  # from autosklearn
    "extra_trees_preproc_for_classification": autosklearn.pipeline.components.feature_preprocessing.extra_trees_preproc_for_classification.ExtraTreesPreprocessorClassification,
    "extra_trees_preproc_for_regression": autosklearn.pipeline.components.feature_preprocessing.extra_trees_preproc_for_regression.ExtraTreesPreprocessorRegression,
    # "fast_ica": autosklearn.pipeline.components.feature_preprocessing.fast_ica.FastICA,
    # "feature_agglomeration": autosklearn.pipeline.components.feature_preprocessing.feature_agglomeration.FeatureAgglomeration,
    # "kernel_pca": autosklearn.pipeline.components.feature_preprocessing.kernel_pca.KernelPCA,
    # "kitchen_sinks": autosklearn.pipeline.components.feature_preprocessing.kitchen_sinks.RandomKitchenSinks,
    "liblinear_svc_preprocessor": autosklearn.pipeline.components.feature_preprocessing.liblinear_svc_preprocessor.LibLinear_Preprocessor,
    # "nystroem_sampler": autosklearn.pipeline.components.feature_preprocessing.nystroem_sampler.Nystroem,
    # "pca": autosklearn.pipeline.components.feature_preprocessing.pca.PCA,
    "polynomial": autosklearn.pipeline.components.feature_preprocessing.polynomial.PolynomialFeatures,
    # "random_trees_embedding": autosklearn.pipeline.components.feature_preprocessing.random_trees_embedding.RandomTreesEmbedding,
    # 'select_percentile' : autosklearn.pipeline.components.feature_preprocessing.select_percentile.SelectPercentileBase,
    "select_percentile_classification": autosklearn.pipeline.components.feature_preprocessing.select_percentile_classification.SelectPercentileClassification,
    "select_percentile_regression": autosklearn.pipeline.components.feature_preprocessing.select_percentile_regression.SelectPercentileRegression,
    "select_rates_classification": autosklearn.pipeline.components.feature_preprocessing.select_rates_classification.SelectClassificationRates,
    "select_rates_regression": autosklearn.pipeline.components.feature_preprocessing.select_rates_regression.SelectRegressionRates,
    "truncatedSVD": autosklearn.pipeline.components.feature_preprocessing.truncatedSVD.TruncatedSVD,
}
