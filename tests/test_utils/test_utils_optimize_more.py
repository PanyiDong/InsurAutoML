"""
File Name: test_utils_optimize_more.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Actuarial and Risk Management Sciences, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.9
Relative Path: /tests/test_utils/test_utils_optimize_more.py
File Created: Monday, 16th November 2026 7:12:56 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 16th November 2026 11:32:34 pm
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

import copy
import os
import pickle
import signal
import time

import numpy as np
import pytest

from InsurAutoML.constant import MAX_ITER
from InsurAutoML.utils import optimize
from InsurAutoML.utils.optimize import (
    InformedStopper,
    TimePlateauStopper,
    TrialDirnameCreator,
    check_func,
    check_status,
    get_algo,
    get_estimator,
    get_logger,
    get_metrics,
    get_progress_reporter,
    get_scheduler,
    legacy_progress_output,
    time_limit,
)

# stoppers


def test_time_plateau_stopper_trial_rules():
    stopper = TimePlateauStopper(
        timeout=60, std_ratio=0.1, num_results=4, grace_period=4
    )

    # a timed out trial (infinite loss) is stopped immediately
    assert stopper("timeout", {"loss": np.inf})

    # within the grace period, trials continue
    for _ in range(3):
        assert not stopper("trial", {"loss": 1.0})
    # plateau reached (no variation in the last results): stop
    assert stopper("trial", {"loss": 1.0})

    # still improving with large variation: continue
    improving = TimePlateauStopper(
        timeout=60, std_ratio=0.01, num_results=4, grace_period=4
    )
    for loss in [10.0, 5.0, 2.0]:
        assert not improving("trial", {"loss": loss})
    assert not improving("trial", {"loss": 1.0})


def test_time_plateau_stopper_threshold_and_max_iter():
    stopper = TimePlateauStopper(
        timeout=60, num_results=1, grace_period=1, metric_threshold=0.5, mode="min"
    )
    # threshold not reached yet: continue
    assert not stopper("trial", {"loss": 1.0})

    maximize = TimePlateauStopper(
        timeout=60, num_results=1, grace_period=1, metric_threshold=0.5, mode="max"
    )
    assert not maximize("trial", {"loss": 0.1})

    # maximum number of iterations per trial
    capped = TimePlateauStopper(timeout=60, grace_period=10**6, num_results=10**6)
    stops = [capped("trial", {"loss": float(i)}) for i in range(int(MAX_ITER / 32))]
    assert not any(stops[:-1]) and stops[-1]

    # a zero mean loss stops the trial
    zero = TimePlateauStopper(timeout=60, num_results=2, grace_period=2)
    zero("trial", {"loss": 0.0})
    assert zero("trial", {"loss": 0.0})


@pytest.mark.parametrize("stopper_class", [TimePlateauStopper, InformedStopper])
def test_stoppers_time_budget(stopper_class, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(optimize.time, "time", lambda: clock[0])
    stopper = stopper_class(timeout=10)

    assert not stopper.stop_all()  # starts the clock
    clock[0] += 4
    assert not stopper.stop_all()  # 6 seconds left
    clock[0] += 7
    assert stopper.stop_all()  # budget used up


def test_informed_stopper_trial_rules():
    stopper = InformedStopper(timeout=60, num_results=2, grace_period=2)
    assert stopper("timeout", {"loss": np.inf})
    assert not stopper("trial", {"loss": 1.0})
    # after the grace period the informed trial stops (one evaluation is enough)
    assert stopper("trial", {"loss": 1.0})

    capped = InformedStopper(timeout=60, grace_period=10**6, num_results=10**6)
    stops = [capped("trial", {"loss": 1.0}) for _ in range(int(MAX_ITER / 4))]
    assert not any(stops[:-1]) and stops[-1]


# trial names and output


class _Trial:
    def __init__(self, trial_id):
        self.trial_id = trial_id


def test_trial_dirname_creator():
    creator = TrialDirnameCreator(start=2)
    assert creator(_Trial("a")) == "iter_3_id_a"

    # Ray Tune deep-copies the config: the counter must stay shared
    assert copy.deepcopy(creator) is creator and copy.copy(creator) is creator
    assert creator(_Trial("b")) == "iter_4_id_b"

    # picklable for tuner.pkl, with its state
    restored = pickle.loads(pickle.dumps(creator))
    assert restored.count == 4 and restored(_Trial("c")) == "iter_5_id_c"


def test_legacy_progress_output(monkeypatch):
    monkeypatch.delenv("RAY_AIR_NEW_OUTPUT", raising=False)
    with legacy_progress_output():
        assert os.environ["RAY_AIR_NEW_OUTPUT"] == "0"
    assert "RAY_AIR_NEW_OUTPUT" not in os.environ

    monkeypatch.setenv("RAY_AIR_NEW_OUTPUT", "1")
    with legacy_progress_output():
        assert os.environ["RAY_AIR_NEW_OUTPUT"] == "0"
    assert os.environ["RAY_AIR_NEW_OUTPUT"] == "1"

    with legacy_progress_output(enable=False):
        assert os.environ["RAY_AIR_NEW_OUTPUT"] == "1"


def test_progress_reporter_error_rows():
    # max_error=0 means no limit on the error table (0 rows would break the table)
    reporter = get_progress_reporter("CLIReporter", max_evals=8, max_error=0)
    assert reporter._max_error_rows is None
    reporter = get_progress_reporter("CLIReporter", max_evals=8, max_error=3)
    assert reporter._max_error_rows == 3


@pytest.mark.skipif(
    not hasattr(signal, "SIGALRM"), reason="SIGALRM is not available on Windows"
)
def test_time_limit():
    with time_limit(5):
        pass
    with pytest.raises(TimeoutError):
        with time_limit(1):
            time.sleep(3)


@pytest.mark.skipif(hasattr(signal, "SIGALRM"), reason="SIGALRM is available")
def test_time_limit_unsupported_platform():
    with pytest.raises(NotImplementedError):
        with time_limit(1):
            pass


# factories


def test_get_algo_and_scheduler_custom_and_invalid():
    custom = lambda **kwargs: None  # noqa: E731
    assert get_algo(custom) is custom
    assert get_scheduler(custom) is custom
    with pytest.raises(TypeError):
        get_algo("NotAnAlgorithm")
    with pytest.raises(TypeError):
        get_scheduler("NotAScheduler")


def test_get_logger():
    from ray.tune.logger import CSVLoggerCallback, JsonLoggerCallback, LoggerCallback

    default = get_logger(None)
    assert len(default) == 1 and isinstance(default[0], LoggerCallback)

    loggers = get_logger(["Logger", "JSON", "CSV"])
    assert [type(item) for item in loggers][1:] == [
        JsonLoggerCallback,
        CSVLoggerCallback,
    ]

    with pytest.raises(TypeError):
        get_logger("CSV")


def test_get_estimator():
    from sklearn.linear_model import LinearRegression

    for name in [
        "Lasso",
        "Ridge",
        "ExtraTreeRegressor",
        "RandomForestRegressor",
        "LogisticRegression",
        "ExtraTreeClassifier",
        "RandomForestClassifier",
    ]:
        estimator = get_estimator(name)
        assert type(estimator).__name__ == name

    # a class with fit/predict is instantiated
    assert isinstance(get_estimator(LinearRegression), LinearRegression)

    class NoPredict:
        def fit(self):
            pass

    with pytest.raises(ValueError):
        get_estimator(NoPredict)

    # an instance: an unfitted copy (the given estimator is not modified)
    given = LinearRegression(fit_intercept=False)
    estimator = get_estimator(given)
    assert isinstance(estimator, LinearRegression) and estimator is not given
    assert estimator.fit_intercept is False

    # unknown names and objects without fit/predict
    with pytest.raises(ValueError, match="Unrecognized estimator"):
        get_estimator("NoSuchModel")
    with pytest.raises(ValueError, match="fit/predict"):
        get_estimator(object())


def test_get_metrics():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    # maximized metrics are negated for minimization
    assert get_metrics("neg_accuracy")(y_true, y_pred) == pytest.approx(-0.75)
    assert get_metrics("accuracy")(y_true, y_pred) == pytest.approx(0.75)
    assert get_metrics("precision")(y_true, y_pred) == pytest.approx(1.0)
    assert get_metrics("f1")(y_true, y_pred) == pytest.approx(2 / 3)
    assert get_metrics("MAE")(np.array([1.0, 2.0]), np.array([2.0, 2.0])) == 0.5
    assert get_metrics("MAX")(np.array([1.0, 2.0]), np.array([4.0, 2.0])) == 3.0
    assert (
        get_metrics("R2")(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) == 1.0
    )
    assert callable(get_metrics("hinge")) and callable(get_metrics("auc"))

    custom = lambda y, p: 0.0  # noqa: E731
    assert get_metrics(custom) is custom
    with pytest.raises(ValueError):
        get_metrics("not_a_metric")


def test_check_func_and_status_errors():
    from InsurAutoML.encoding import DataEncoding
    from InsurAutoML.hyperparameters import encoder_hyperparameter

    class NoFit:
        pass

    with pytest.raises(AttributeError):
        check_func(NoFit, ref="model")

    # method missing from the hyperparameter space
    with pytest.raises(ValueError):
        check_status({"Unknown": DataEncoding}, encoder_hyperparameter, ref="encoder")

    # hyperparameter not accepted by the method
    space = [{"encoder": "DataEncoding", "not_an_argument": [1]}]
    with pytest.raises(ValueError):
        check_status({"DataEncoding": DataEncoding}, space, ref="encoder")
