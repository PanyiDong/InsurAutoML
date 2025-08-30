"""
File Name: __init__.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.6
Relative Path: /InsurAutoML/utils/__init__.py
File Created: Monday, 24th October 2022 11:56:57 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Friday, 29th August 2025 1:49:10 pm
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


from .base import (
    random_guess,
    random_index,
    random_list,
    minloc,
    maxloc,
    is_date,
    feature_rounding,
    True_index,
    type_of_task,
    Timer,
)
from .data import (
    train_test_split,
    as_dataframe,
    formatting,
    unify_nan,
    remove_index_columns,
    get_missing_matrix,
)
from .metadata import MetaData
from .file import save_model
from .stats import (
    nan_cov,
    class_means,
    empirical_covariance,
    class_cov,
    Pearson_Corr,
    MI,
    t_score,
    ANOVA,
    CCC,
    ACCC,
)

# from ._preprocessing import (
#     text_preprocessing_torchtext,
#     text_preprocessing_transformers,
# )

from .eda import EDA
