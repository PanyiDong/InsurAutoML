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
