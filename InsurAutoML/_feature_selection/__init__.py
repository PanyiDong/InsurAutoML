"""
File: __init__.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.3
Relative Path: /InsurAutoML/_feature_selection/__init__.py
File: __init__.py
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Tuesday, 15th November 2022 4:06:09 pm
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
Note: Aug. 8, 2022
The feature selection methods are arranged more systematically, which uses the categories defined
by Chandrashekar (2014):
1. Filter: use some statistical measures to score and select features.
2. Wrapper: use a machine learning model to score and select features.
3. Embedded: embed feature importance in the model training to select features.
4. Hybrid: combine the filter and wrapper method.

Other methods are imported and only called here.

[1] Chandrashekar, G., & Sahin, F. (2014). A survey on feature selection methods. Computers & Electrical Engineering, 40(1), 16-28.
"""

from ._filter import (
    FeatureFilter,
    mRMR,
    FOCI,
)
from ._wrapper import (
    # ExhaustiveFS,
    SFS,
    ASFFS,
)
from ._embed import (
    PCA_FeatureSelection,
    # LDASelection,
    RBFSampler,
)
from ._hybrid import (
    CBFS,
    GeneticAlgorithm,
)

from InsurAutoML._base import no_processing

feature_selections = {
    "no_processing": no_processing,
    # "LDASelection": LDASelection,
    # "PCA_FeatureSelection": PCA_FeatureSelection,
    "RBFSampler": RBFSampler,
    "FeatureFilter": FeatureFilter,
    "ASFFS": ASFFS,
    "GeneticAlgorithm": GeneticAlgorithm,
    # "ExhaustiveFS": ExhaustiveFS, # exhaustive search is not practical, takes too long
    "SFS": SFS,
    "mRMR": mRMR,
    "CBFS": CBFS,
    # "FOCI": FOCI,
}


# Update: Nov. 15, 2022
# autosklearn decrypted, no need to consider
# import importlib
# # check if autosklearn is installed, if not, use sklearn replacement
# autosklearn_spec = importlib.util.find_spec("autosklearn")
# sklearn_spec = importlib.util.find_spec("sklearn")

# if autosklearn_spec is not None:
#     from ._autosklearn import (
#         extra_trees_preproc_for_classification,
#         extra_trees_preproc_for_regression,
#         liblinear_svc_preprocessor,
#         polynomial,
#         select_percentile_classification,
#         select_percentile_regression,
#         select_rates_classification,
#         select_rates_regression,
#         truncatedSVD,
#     )

#     # from autosklearn
#     feature_selections[
#         "extra_trees_preproc_for_classification"
#     ] = extra_trees_preproc_for_classification
#     feature_selections[
#         "extra_trees_preproc_for_regression"
#     ] = extra_trees_preproc_for_regression
#     feature_selections["liblinear_svc_preprocessor"] = liblinear_svc_preprocessor
#     feature_selections["polynomial"] = polynomial
#     feature_selections[
#         "select_percentile_classification"
#     ] = select_percentile_classification
#     feature_selections["select_percentile_regression"] = select_percentile_regression
#     feature_selections["select_rates_classification"] = select_rates_classification
#     feature_selections["select_rates_regression"] = select_rates_regression
#     feature_selections["truncatedSVD"] = truncatedSVD
# # elif sklearn not installed, raise error
# elif sklearn_spec is None:
#     raise ImportError(
#         "None of autosklearn or sklearn is installed. Please install at least one of them to use feature selection."
#     )
# else:
from ._sklearn import (
    extra_trees_preproc_for_classification,
    extra_trees_preproc_for_regression,
    liblinear_svc_preprocessor,
    polynomial,
    select_percentile_classification,
    select_percentile_regression,
    select_rates_classification,
    select_rates_regression,
    truncatedSVD,
)

# from autosklearn
feature_selections[
    "extra_trees_preproc_for_classification"
] = extra_trees_preproc_for_classification
feature_selections[
    "extra_trees_preproc_for_regression"
] = extra_trees_preproc_for_regression
feature_selections["liblinear_svc_preprocessor"] = liblinear_svc_preprocessor
feature_selections["polynomial"] = polynomial
feature_selections[
    "select_percentile_classification"
] = select_percentile_classification
feature_selections["select_percentile_regression"] = select_percentile_regression
feature_selections["select_rates_classification"] = select_rates_classification
feature_selections["select_rates_regression"] = select_rates_regression
feature_selections["truncatedSVD"] = truncatedSVD
