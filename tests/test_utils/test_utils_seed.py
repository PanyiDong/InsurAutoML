"""
File Name: test_utils_seed.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Actuarial and Risk Management Sciences, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.9
Relative Path: /tests/test_utils/test_utils_seed.py
File Created: Monday, 16th November 2026 10:21:28 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 16th November 2026 11:36:59 pm
Modified By: Panyi Dong (panyid2@illinois.edu)

-----
MIT License

Copyright (c) 2026, Panyi Dong

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
Random tie breaking of CCC/ACCC and the CCC feature filter: reproducible with a
seed, and the global numpy random state is not used.
"""

import numpy as np
import pandas as pd

from InsurAutoML.feature_selection import FeatureFilter
from InsurAutoML.utils import stats


def _global_state():
    return np.random.get_state()[1].copy()


def test_ccc_accc_seed():
    rng = np.random.default_rng(0)
    # discrete x: many ties
    x = rng.integers(0, 3, 300).astype(float)
    y = x + rng.normal(size=300)

    np.random.seed(0)
    state = _global_state()
    assert stats.CCC(x, y, seed=1) == stats.CCC(x, y, seed=1)
    assert stats.CCC(x, y, seed=1, fold=3) == stats.CCC(x, y, seed=1, fold=3)
    assert stats.CCC(x, y, seed=1) != stats.CCC(x, y, seed=2)
    Z = rng.integers(0, 3, (200, 2)).astype(float)
    target = Z.sum(axis=1) + rng.normal(size=200)
    assert stats.ACCC(Z, target, seed=3) == stats.ACCC(Z, target, seed=3)
    assert stats.ACCC(Z, target, X=Z[:, 0], seed=3) == stats.ACCC(
        Z, target, X=Z[:, 0], seed=3
    )
    # the global random state is not changed
    np.testing.assert_array_equal(_global_state(), state)

    # a generator is used as is (shared by successive calls)
    generator = np.random.default_rng(5)
    ranks = stats.rankdata_random([1, 1, 1, 1], seed=generator)
    assert sorted(ranks) == [1, 2, 3, 4]
    assert len(stats.randomNN(np.array([4, 5, 6]), seed=generator)) == 3


def test_feature_filter_seed():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.integers(0, 3, (300, 12)), columns=list("abcdefghijkl"))
    y = X["a"] + rng.normal(size=300)

    def selected(seed):
        return tuple(
            FeatureFilter("CCC", n_prop=0.5, seed=seed).fit(X, y).select_features
        )

    # default seed: deterministic
    assert (
        len(
            {
                tuple(FeatureFilter("CCC", n_prop=0.5).fit(X, y).select_features)
                for _ in range(3)
            }
        )
        == 1
    )
    assert selected(7) == selected(7)
    # no seed: random tie breaking
    assert len({selected(None) for _ in range(10)}) > 1
