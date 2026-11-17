"""
File Name: test_imputer_nn.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Actuarial and Risk Management Sciences, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.9
Relative Path: /tests/test_imputer/test_imputer_nn.py
File Created: Monday, 16th November 2026 7:12:56 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 16th November 2026 11:35:50 pm
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

import warnings

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

import InsurAutoML.imputation.nn as nn_mod  # noqa: E402
from InsurAutoML.imputation.nn import GAIN, GAIN_torch, GAIN_tf  # noqa: E402

NUM_COLS = ["a", "b", "c", "d"]


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)
    warnings.simplefilter("ignore")


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    full = pd.DataFrame(rng.normal(size=(150, 4)), columns=NUM_COLS)
    mask = rng.random(full.shape) < 0.1
    missing = full.mask(mask)
    missing["cat"] = rng.choice(["x", "y", "z"], 150)
    return full, missing, mask


def _check_imputed(out, full, missing, mask):
    assert out.shape == missing.shape
    assert list(out.columns) == list(missing.columns)
    assert not out.isnull().values.any()
    num = out[NUM_COLS].to_numpy(dtype=float)
    # observed values unchanged (up to MinMax scaling round-trip)
    np.testing.assert_allclose(num[~mask], full.values[~mask], rtol=1e-8, atol=1e-8)
    # imputed values lie within the observed range of each column
    lo = missing[NUM_COLS].min().values
    hi = missing[NUM_COLS].max().values
    for j in range(len(NUM_COLS)):
        vals = num[mask[:, j], j]
        assert ((vals >= lo[j] - 1e-8) & (vals <= hi[j] + 1e-8)).all()
    # categorical column restored
    assert list(out["cat"]) == list(missing["cat"])


@pytest.mark.parametrize("optim", ["Adam", "SGD", "Adagrad", "RMSprop"])
def test_gain_torch_optimizers(data, optim):
    full, missing, mask = data
    imputer = GAIN_torch(optim=optim, max_iter=15, batch_size=64, seed=1)
    out = imputer.fill(missing)
    assert imputer._fitted
    assert 1 <= len(imputer.losses) <= 15
    assert all(np.isfinite(imputer.losses))
    _check_imputed(out, full, missing, mask)
    # input left untouched
    assert missing[NUM_COLS].isnull().values.sum() == mask.sum()


def test_gain_torch_default_lr():
    assert GAIN_torch(optim="LBFGS").lr == 1
    assert GAIN_torch(optim="Adam").lr == 0.001
    assert GAIN_torch(optim="SGD").lr == 0.01
    assert GAIN_torch(optim="SGD", lr=0.5).lr == 0.5


def test_gain_torch_unknown_optimizer(data):
    _, missing, _ = data
    with pytest.raises(KeyError, match="unknown optimizer"):
        GAIN_torch(optim="Foo", max_iter=2).fill(missing)


def test_gain_torch_progressbar_early_stop_and_refill(data):
    full, missing, mask = data
    # huge delta -> early stop after the second iteration
    imputer = GAIN_torch(
        max_iter=50, batch_size=500, delta=1e6, progressbar=True, scaling=True
    )
    out = imputer.fill(missing)
    assert imputer.batch_size == len(missing)  # capped at number of rows
    assert len(imputer.losses) == 2
    _check_imputed(out, full, missing, mask)

    # second call reuses the trained networks (no retraining)
    n_losses = len(imputer.losses)
    out2 = imputer.fill(missing)
    assert len(imputer.losses) == n_losses
    _check_imputed(out2, full, missing, mask)


def test_gain_torch_numpy_input_no_scaling():
    rng = np.random.default_rng(3)
    arr = rng.uniform(size=(80, 3))
    mask = rng.random(arr.shape) < 0.15
    arr_missing = np.where(mask, np.nan, arr)
    out = GAIN_torch(max_iter=5, batch_size=32, scaling=False).fill(arr_missing)
    assert isinstance(out, pd.DataFrame) and out.shape == (80, 3)
    assert not out.isnull().values.any()
    np.testing.assert_allclose(out.values[~mask], arr[~mask])


def test_gain_torch_no_missing_and_bad_input(data):
    full, _, _ = data
    with pytest.warns(UserWarning, match="No missing values"):
        warnings.simplefilter("always")
        out = GAIN_torch(max_iter=2).fill(full)
    pd.testing.assert_frame_equal(out, full)

    with pytest.raises(TypeError, match="Expect a dataframe"):
        GAIN_torch().fill(5)


def test_gain_torch_networks():
    imputer = GAIN_torch()
    imputer._initialization(p=3, h_dim=3)
    assert len(imputer.theta_G) == 6 and len(imputer.theta_D) == 6
    assert imputer.theta_G[0].shape == (6, 3)
    X = torch.rand(5, 3, dtype=torch.double)
    M = torch.ones(5, 3, dtype=torch.double)
    g = imputer.Generator(X, M)
    d = imputer.Discriminator(X, M)
    assert g.shape == d.shape == (5, 3)
    assert ((g >= 0) & (g <= 1)).all() and ((d >= 0) & (d <= 1)).all()
    G_loss, D_loss = imputer.network_loss(X, M, M)
    assert G_loss.ndim == 0 and D_loss.ndim == 0
    assert torch.isfinite(G_loss) and torch.isfinite(D_loss)

    idx = imputer.random_index(10, 4)
    assert len(idx) == 4 and len(set(idx)) == 4 and idx.max() < 10


def test_gain_wrapper_uses_torch(data):
    full, missing, mask = data
    imputer = GAIN(max_iter=5, batch_size=64, seed=2)
    assert isinstance(imputer.model, GAIN_torch)
    assert imputer.model.lr == 0.001  # default lr resolved by GAIN_torch
    out = imputer.fill(missing)
    assert imputer._fitted
    _check_imputed(out, full, missing, mask)


def test_gain_wrapper_backend_selection(monkeypatch):
    # only tensorflow "available" -> tensorflow implementation selected
    monkeypatch.setattr(nn_mod, "torch_spec", None)
    monkeypatch.setattr(nn_mod, "tensorflow_spec", object())
    assert isinstance(GAIN(max_iter=1).model, GAIN_tf)

    # neither backend available -> error
    monkeypatch.setattr(nn_mod, "tensorflow_spec", None)
    with pytest.raises(Exception, match="No tensorflow or torch"):
        GAIN()


def test_gain_tf_helpers(data):
    full, missing, mask = data
    imputer = GAIN_tf(batch_size=16, max_iter=1)

    m = imputer.mask_matrix(missing[NUM_COLS])
    np.testing.assert_array_equal(m.values, 1 - mask.astype(int))

    b = imputer.binary_sampler(p=0.5, size=(20, 3))
    assert b.shape == (20, 3) and set(np.unique(b)).issubset({0, 1})
    assert imputer.binary_sampler(p=1.0, size=(5,)).shape == (5, 1)
    assert imputer.binary_sampler(p=0.0, size=(6, 2)).sum() == 0

    u = imputer.uniform_sampler(low=0, high=0.01, size=(10, 2))
    assert u.shape == (10, 2) and (u >= 0).all() and (u < 0.01).all()
    assert imputer.uniform_sampler(size=(4,)).shape == (4, 1)

    # None dimensions default to 1
    assert imputer.binary_sampler(size=(None, 3)).shape == (1, 3)
    assert imputer.binary_sampler(size=(4, None)).shape == (4, 1)
    assert imputer.binary_sampler(size=[None, None]).shape == (1, 1)
    assert imputer.uniform_sampler(size=(None, 2)).shape == (1, 2)
    assert imputer.uniform_sampler(size=(3, None)).shape == (3, 1)

    # no missing values: returns an unchanged copy with a warning
    with pytest.warns(UserWarning, match="No nan values"):
        warnings.simplefilter("always")
        out = imputer.fill(full)
    pd.testing.assert_frame_equal(out, full)
    assert out is not full


def test_gain_torch_lbfgs(data):
    full, missing, mask = data
    imputer = GAIN_torch(optim="LBFGS", max_iter=3, batch_size=64, seed=1)
    out = imputer.fill(missing)
    assert 1 <= len(imputer.losses) <= 3
    assert all(np.isfinite(imputer.losses))
    _check_imputed(out, full, missing, mask)


def test_gain_torch_training_updates_parameters(data, monkeypatch):
    # parameters used in the forward pass must be the ones the optimizer updates
    # (also on GPU, where parameters are moved to the device)
    _, missing, _ = data
    imputer = GAIN_torch(max_iter=10, batch_size=64, lr=0.01, seed=1)
    snapshot = {}
    orig_init = imputer._initialization

    def init_and_snapshot(p, h_dim):
        orig_init(p, h_dim)
        snapshot["G"] = [t.detach().clone() for t in imputer.theta_G]
        snapshot["D"] = [t.detach().clone() for t in imputer.theta_D]

    monkeypatch.setattr(imputer, "_initialization", init_and_snapshot)
    imputer.fill(missing)

    for before, after in zip(snapshot["G"], imputer.theta_G):
        assert after.is_leaf and after.requires_grad
        assert after.device.type == nn_mod.device.type
    assert not torch.equal(snapshot["G"][0], imputer.theta_G[0].detach().cpu())
    assert not torch.equal(snapshot["D"][0], imputer.theta_D[0].detach().cpu())


def test_gain_torch_imputes_from_observed_values():
    # strongly related columns: b = a, c = -a; only b has missing entries
    rng = np.random.default_rng(5)
    a = rng.uniform(0, 1, size=300)
    full = pd.DataFrame({"a": a, "b": a.copy(), "c": -a})
    mask = np.zeros(full.shape, dtype=bool)
    mask[rng.random(300) < 0.2, 1] = True
    missing = full.mask(mask)

    out = GAIN_torch(max_iter=300, batch_size=64, lr=0.005, seed=0).fill(missing)

    assert not out.isnull().values.any()
    np.testing.assert_allclose(out.values[~mask], full.values[~mask], atol=1e-8)
    imputed = out["b"].values[mask[:, 1]]
    # imputed values are generator outputs, not a constant fill such as the column min
    assert np.unique(np.round(imputed, 8)).size > 1
    assert not np.allclose(imputed, missing["b"].min())
    # and they follow the observed related column
    assert np.corrcoef(imputed, full["a"].values[mask[:, 1]])[0, 1] > 0.5
