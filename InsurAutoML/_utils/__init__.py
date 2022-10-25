"""
File: __init__.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: My_AutoML
Latest Version: 0.2.0
Relative Path: /My_AutoML/_utils/__init__.py
File Created: Wednesday, 6th April 2022 12:00:12 am
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Sunday, 25th September 2022 11:11:35 pm
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

try:
    from ._c import (
        random_guess,
        random_index,
        random_list,
        minloc,
        maxloc,
    )
except ImportError:
    from ._base import (
        random_guess,
        random_index,
        random_list,
        minloc,
        maxloc,
    )
# else:
#     raise ImportError("Cannot import the C++ and Python extension.")

from ._base import (
    # random_guess,
    # random_index,
    # random_list,
    # minloc,
    # maxloc,
    is_date,
    feature_rounding,
    True_index,
    type_of_task,
    Timer,
)
from ._data import (
    train_test_split,
    as_dataframe,
    formatting,
    unify_nan,
    remove_index_columns,
    get_missing_matrix,
)
from ._file import save_model
from ._stat import (
    nan_cov,
    class_means,
    empirical_covariance,
    class_cov,
    Pearson_Corr,
    MI,
    t_score,
    ANOVA,
    ACCC,
)

# from ._preprocessing import (
#     text_preprocessing_torchtext,
#     text_preprocessing_transformers,
# )

from ._eda import EDA
