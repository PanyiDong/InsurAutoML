"""
File: test_model_nn.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.6
Relative Path: /tests/test_model/test_model_nn.py
File: test_model_nn.py
Author: Panyi Dong (panyid2@illinois.edu)

-----
MIT License

Copyright (c) 2022 - 2026, Panyi Dong

Tests for the torch-based neural network models (MLP / RNN).
"""

import warnings

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from InsurAutoML.model.FNN import (  # noqa: E402
    MLP_Model,
    MLP_Base,
    MLP_Classifier,
    MLP_Regressor,
)
from InsurAutoML.model.RNN import (  # noqa: E402
    RNN_Net,
    RNN_Base,
    RNN_Classifier,
    RNN_Regressor,
)

N_ROWS, N_FEAT = 128, 4


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)
    warnings.simplefilter("ignore")


def _clf_data(n_classes=3):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.normal(size=(N_ROWS, N_FEAT)), columns=[f"x{i}" for i in range(N_FEAT)]
    )
    y = pd.Series(np.arange(N_ROWS) % n_classes, name="y")
    return X, y


def _reg_data():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(N_ROWS, N_FEAT)).astype(np.float32)
    y = (X @ np.array([1.0, -2.0, 0.5, 0.0], dtype=np.float32)).astype(np.float32)
    return X, y


# ---------------------------------------------------------------------------
# MLP


@pytest.mark.parametrize("activation", ["ReLU", "Tanh", "Sigmoid"])
@pytest.mark.parametrize("softmax", [True, False])
def test_mlp_model_forward(activation, softmax):
    model = MLP_Model(
        input_size=N_FEAT,
        hidden_layer=2,
        hidden_size=8,
        output_size=3,
        softmax=softmax,
        activation=activation,
    )
    # input layer + 2 hidden layers (each Linear + activation) + output layer
    n_linear = sum(isinstance(m, torch.nn.Linear) for m in model.forward_model)
    assert n_linear == 4
    out = model(torch.randn(10, N_FEAT))
    assert out.shape == (10, 3)
    if softmax:
        assert torch.allclose(out.sum(dim=1), torch.ones(10), atol=1e-5)


def test_mlp_model_bad_activation():
    with pytest.raises(ValueError):
        MLP_Model(N_FEAT, 1, 4, 2, activation="Swish")


@pytest.mark.parametrize("optimizer", ["Adam", "SGD"])
def test_mlp_classifier(optimizer):
    X, y = _clf_data(3)
    clf = MLP_Classifier(
        hidden_layer=1,
        hidden_size=16,
        optimizer=optimizer,
        batch_size=32,
        num_epochs=3,
        is_cuda=False,
    )
    assert clf._fitted is False
    clf.fit(X, y)
    assert clf._fitted is True
    assert clf.input_size == N_FEAT and clf.output_size == 3

    pred = clf.predict(X)
    assert pred.shape == (N_ROWS,)
    assert set(np.unique(pred)).issubset({0, 1, 2})

    proba = clf.predict_proba(X.values)  # ndarray input path
    assert proba.shape == (N_ROWS, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(np.argmax(proba, axis=1), pred)


def test_mlp_classifier_nll_and_numpy_input():
    X, y = _clf_data(2)
    clf = MLP_Classifier(
        hidden_layer=1,
        hidden_size=8,
        softmax=False,
        criteria="NegativeLogLikelihood",
        learning_rate=0.01,
        num_epochs=2,
        is_cuda=False,
    )
    clf.fit(X.values, y.values)
    pred = clf.predict(X)
    assert set(np.unique(pred)).issubset({0, 1})


def test_mlp_classifier_rejects_regression_loss():
    X, y = _clf_data()
    with pytest.raises(ValueError, match="CrossEntropy"):
        MLP_Classifier(hidden_layer=1, hidden_size=4, criteria="MSE").fit(X, y)


@pytest.mark.parametrize("criteria", ["MSE", "MAE"])
def test_mlp_regressor(criteria):
    X, y = _reg_data()
    reg = MLP_Regressor(
        hidden_layer=1,
        hidden_size=16,
        criteria=criteria,
        optimizer="SGD",
        learning_rate=0.01,
        num_epochs=3,
        is_cuda=False,
    )
    reg.fit(pd.DataFrame(X), pd.Series(y))
    assert reg._fitted
    pred = reg.predict(X)
    assert pred.shape == (N_ROWS, 1)
    assert np.isfinite(pred).all()
    # predict_proba is not supported for regression
    with pytest.raises(NotImplementedError):
        reg.predict_proba(X)


def test_mlp_regressor_rejects_classification_loss():
    X, y = _reg_data()
    with pytest.raises(ValueError, match="MSE or MAE"):
        MLP_Regressor(hidden_layer=1, hidden_size=4, criteria="CrossEntropy").fit(X, y)


def test_mlp_base_unknown_criteria_and_cuda_default():
    X, y = _reg_data()
    base = MLP_Base(
        input_size=N_FEAT, hidden_layer=1, hidden_size=4, output_size=1, criteria="Foo"
    )
    with pytest.raises(ValueError, match="Not recognized criteria"):
        base.fit(torch.as_tensor(X), torch.as_tensor(y).unsqueeze(-1))


def test_mlp_base_regression_runs_with_default_device():
    # is_cuda=True: uses GPU if present, otherwise warns and falls back to CPU
    X, y = _reg_data()
    base = MLP_Base(
        input_size=N_FEAT, hidden_layer=0, hidden_size=4, output_size=1, num_epochs=2
    )
    base.fit(torch.as_tensor(X), torch.as_tensor(y).unsqueeze(-1))
    assert base._fitted
    assert base.predict(pd.DataFrame(X)).shape == (N_ROWS, 1)


def test_mlp_predict_before_fit():
    X, _ = _reg_data()
    reg = MLP_Regressor(hidden_layer=1, hidden_size=4)
    with pytest.raises(AttributeError):
        reg.predict(X)


# ---------------------------------------------------------------------------
# RNN


@pytest.mark.parametrize("unit", ["RNN", "LSTM", "GRU"])
@pytest.mark.parametrize("use_softmax", [True, False])
def test_rnn_net_forward(unit, use_softmax):
    net = RNN_Net(
        input_size=1,
        hidden_size=8,
        output_size=3,
        n_layers=2,
        RNN_unit=unit,
        activation="Tanh",
        use_softmax=use_softmax,
        device="cpu",
    )
    h = net.init_hidden(5)
    if unit == "LSTM":
        assert isinstance(h, tuple) and len(h) == 2
        assert h[0].shape == (2, 5, 8)
    else:
        assert h.shape == (2, 5, 8)
    net.eval()
    out, h2 = net(torch.randn(5, 6, 1), h)
    assert out.shape == (5, 6, 3)
    if unit == "LSTM":
        assert isinstance(h2, tuple) and h2[0].shape == (2, 5, 8)
    else:
        assert h2.shape == (2, 5, 8)


def test_rnn_net_bad_args():
    with pytest.raises(TypeError, match="RNN unit"):
        RNN_Net(1, 4, 2, RNN_unit="Transformer", device="cpu")
    with pytest.raises(TypeError, match="activation"):
        RNN_Net(1, 4, 2, activation="Swish", device="cpu")


@pytest.mark.parametrize(
    "unit,optimizer,criteria",
    [
        ("RNN", "Adam", "CrossEntropy"),
        ("LSTM", "SGD", "NegativeLogLikelihood"),
        ("GRU", "Adam", "CrossEntropy"),
    ],
)
def test_rnn_classifier(unit, optimizer, criteria):
    X, y = _clf_data(3)
    clf = RNN_Classifier(
        hidden_size=8,
        RNN_unit=unit,
        activation="ReLU",
        optimizer=optimizer,
        criteria=criteria,
        batch_size=32,
        num_epochs=2,
        is_cuda=False,
    )
    clf.fit(X, y)
    assert clf._fitted and clf.output_size == 3
    pred = clf.predict(X)
    assert pred.shape == (N_ROWS,)
    assert set(np.unique(pred)).issubset({0, 1, 2})
    proba = clf.predict_proba(X.values)
    assert proba.shape == (N_ROWS, 3)
    assert np.isfinite(proba).all() and (proba >= 0).all()
    # probabilities normalised over classes, consistent with predict
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(np.argmax(proba, axis=1), pred)
    # evaluation is deterministic (dropout disabled at prediction time)
    np.testing.assert_array_equal(clf.predict(X), pred)


def test_rnn_classifier_tensor_input():
    X, y = _clf_data(2)
    Xt = torch.as_tensor(X.values, dtype=torch.float).unsqueeze(-1)
    yt = torch.as_tensor(y.values, dtype=torch.long)
    clf = RNN_Classifier(hidden_size=4, num_epochs=1, is_cuda=False)
    clf.fit(Xt, yt)
    assert set(np.unique(clf.predict(X))).issubset({0, 1})


@pytest.mark.parametrize("criteria", ["MSE", "MAE"])
def test_rnn_regressor(criteria):
    X, y = _reg_data()
    reg = RNN_Regressor(
        hidden_size=8,
        RNN_unit="GRU",
        criteria=criteria,
        num_epochs=2,
        is_cuda=False,
    )
    reg.fit(pd.DataFrame(X), pd.Series(y))
    assert reg._fitted and reg.output_size == 1
    pred = reg.predict(pd.DataFrame(X))
    assert pred.shape == (N_ROWS, 1)
    assert np.isfinite(pred).all()
    with pytest.raises(NotImplementedError):
        reg.predict_proba(X)


def test_rnn_regressor_tensor_input_default_device():
    X, y = _reg_data()
    reg = RNN_Regressor(hidden_size=4, num_epochs=1)
    reg.fit(torch.as_tensor(X).unsqueeze(-1), torch.as_tensor(y))
    assert reg.predict(X).shape == (N_ROWS, 1)


def test_rnn_base_errors():
    X, y = _clf_data()
    Xt = torch.as_tensor(X.values, dtype=torch.float).unsqueeze(-1)
    yt = torch.as_tensor(y.values, dtype=torch.long)
    with pytest.raises(ValueError, match="RNN unit"):
        RNN_Base(RNN_unit="Foo", is_cuda=False).fit(Xt, yt)
    with pytest.raises(ValueError, match="Not recognized criteria"):
        RNN_Base(hidden_size=4, criteria="Foo", is_cuda=False).fit(Xt, yt)


def test_rnn_predict_before_fit():
    X, _ = _clf_data()
    with pytest.raises(AttributeError):
        RNN_Classifier(hidden_size=4).predict(X)


def test_no_gpu_fallback_warns(monkeypatch):
    # simulate a machine without GPU: is_cuda=True should warn and use CPU
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    warnings.simplefilter("always")
    X, y = _clf_data(2)
    with pytest.warns(UserWarning, match="No GPU detected"):
        mlp = MLP_Classifier(hidden_layer=1, hidden_size=4, num_epochs=1).fit(X, y)
    assert str(mlp.device) == "cpu"
    with pytest.warns(UserWarning, match="No GPU detected"):
        rnn = RNN_Classifier(hidden_size=4, num_epochs=1).fit(X, y)
    assert str(rnn.device) == "cpu"


def test_rnn_net_softmax_over_classes():
    net = RNN_Net(1, 8, 4, use_softmax=True, dropout=0.0, device="cpu")
    net.eval()
    out, _ = net(torch.randn(3, 5, 1), net.init_hidden(3))
    # log-softmax over the class (last) dimension at every time step
    torch.testing.assert_close(out.exp().sum(dim=-1), torch.ones(3, 5))
