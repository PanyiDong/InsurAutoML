"""
File Name: test_dashboard_events.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Actuarial and Risk Management Sciences, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.9
Relative Path: /tests/test_dashboard/test_dashboard_events.py
File Created: Monday, 16th November 2026 7:49:55 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 16th November 2026 11:34:30 pm
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
Tests of the event log of the dashboard.
"""

import json
import logging
import math
import os
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from InsurAutoML.dashboard import (
    DashboardCallback,
    DashboardState,
    EventRecorder,
    load_state,
    read_events,
    standardize_config,
)
from InsurAutoML.dashboard.events import to_json


class FakeClock:
    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        return self.now


def test_to_json():
    value = {
        1: np.int64(3),
        "float": np.float32(0.5),
        "inf": float("inf"),
        "nan": np.nan,
        "bool": np.bool_(True),
        "array": np.array([1.0, np.inf]),
        "tuple": (1, "a"),
        "object": object,
    }
    converted = to_json(value)

    assert converted["1"] == 3 and isinstance(converted["1"], int)
    assert converted["float"] == 0.5
    assert converted["inf"] is None and converted["nan"] is None
    assert converted["bool"] is True
    assert converted["array"] == [1.0, None]
    assert converted["tuple"] == [1, "a"]
    assert isinstance(converted["object"], str)
    # valid (strict) JSON
    json.dumps(converted, allow_nan=False)


def test_recorder_writes_and_pickles(tmp_path):
    clock = FakeClock()
    recorder = EventRecorder(str(tmp_path / "dashboard"), clock=clock)
    event = recorder.emit("run_start", timeout=10, loss=float("inf"))

    assert event == {"type": "run_start", "time": 1000.0, "timeout": 10, "loss": None}
    assert os.path.isfile(recorder.path)

    # stored with the Ray Tune experiment: picklable, keeps writing to the log
    restored = pickle.loads(pickle.dumps(recorder))
    restored.emit("run_end")

    events, offset = read_events(recorder.path)
    assert [item["type"] for item in events] == ["run_start", "run_end"]
    assert offset == os.path.getsize(recorder.path)


def test_recorder_write_failure_does_not_raise(tmp_path):
    recorder = EventRecorder(str(tmp_path / "dashboard"))
    # the log path is a directory: writing fails
    os.makedirs(recorder.path)

    with pytest.warns(UserWarning, match="Unable to write dashboard events"):
        recorder.emit("run_start")
    # warned once
    recorder.emit("run_end")


def test_read_events_partial_and_corrupted_lines(tmp_path):
    path = tmp_path / "events.jsonl"
    assert read_events(str(path)) == ([], 0)

    path.write_bytes(b'{"type": "a"}\nnot json\n\n{"type": "b"}\n{"type": "c"')
    events, offset = read_events(str(path))
    assert [item["type"] for item in events] == ["a", "b"]

    # the partial line is read once complete
    with open(path, "ab") as out_f:
        out_f.write(b"}\n")
    events, _ = read_events(str(path), offset)
    assert [item["type"] for item in events] == ["c"]


def _events(clock, recorder):
    """A run of 2 rounds (3 trials planned per round)."""
    recorder.emit("run_start", timeout=100, max_evals=6, rounds=2)
    recorder.emit("round_start", round=0, name="model", num_samples=3)
    recorder.emit(
        "trial_start", round=0, trial_id="a", config={"model": {"method": "A"}}
    )
    recorder.emit(
        "trial_start", round=0, trial_id="b", config={"model": {"method": "B"}}
    )
    clock.now += 10
    recorder.emit(
        "trial_result",
        round=0,
        trial_id="a",
        loss=0.5,
        training_status="FITTED",
        iteration=1,
    )
    recorder.emit(
        "trial_result",
        round=0,
        trial_id="b",
        loss=float("inf"),
        training_status="TIMEOUT",
        iteration=1,
    )
    recorder.emit("trial_end", round=0, trial_id="a", status="COMPLETED")
    recorder.emit("trial_end", round=0, trial_id="b", status="COMPLETED")
    recorder.emit("trial_start", round=0, trial_id="c", config={})
    clock.now += 10
    recorder.emit("trial_end", round=0, trial_id="c", status="ERROR", error="boom")
    recorder.emit("trial_start", round=0, trial_id="d", config={})
    recorder.emit(
        "trial_result",
        round=0,
        trial_id="d",
        loss=0.2,
        training_status="FITTED",
        iteration=2,
    )
    clock.now += 5


def test_state_fold_and_progress(tmp_path):
    clock = FakeClock()
    recorder = EventRecorder(str(tmp_path), clock=clock)
    _events(clock, recorder)

    state = load_state(str(tmp_path))
    progress = state.progress(now=clock.now)
    trials = {trial["trial_id"]: trial for trial in state.trials.values()}

    assert [trials[key]["status"] for key in "abcd"] == [
        "FITTED",
        "TIMEOUT",
        "ERROR",
        "RUNNING",
    ]
    assert trials["c"]["error"] == "boom"
    assert trials["a"]["duration"] == 10 and trials["d"]["iterations"] == 2
    assert state.peak_running == 2
    assert progress["counts"]["RUNNING"] == 1
    assert progress["trials_finished"] == 3
    # round 2 not started: assumed to have the same budget
    assert progress["trials_total"] == 6
    assert progress["round"] == 1 and progress["rounds"] == 2
    assert progress["elapsed"] == 25 and progress["time_left"] == 75
    # 3 trials left, median trial time 10s, 2 trials in parallel
    assert progress["eta"] == pytest.approx(15)

    # running trial: best among the fitted ones
    assert state.best_trial()["trial_id"] == "d"

    # end of the round closes the running trial (stopped with the experiment)
    recorder.emit("round_end", round=0)
    recorder.emit("round_start", round=1, name="model_2", num_samples=3)
    clock.now += 5
    state = load_state(recorder.path)
    trial_d = next(t for t in state.trials.values() if t["trial_id"] == "d")
    assert trial_d["status"] == "FITTED" and trial_d["duration"] == 5
    progress = state.progress(now=clock.now)
    assert progress["round"] == 2 and progress["elapsed"] == 30

    recorder.emit("run_end")
    state = load_state(recorder.path)
    progress = state.progress(now=clock.now + 100)
    assert progress["finished"] and progress["eta"] == 0
    # wall time stops at the end of the run
    assert progress["wall_time"] == 30

    snapshot = state.snapshot()
    json.dumps(snapshot, allow_nan=False)
    assert snapshot["best"]["trial_id"] == "d"
    assert len(snapshot["trials"]) == 4 and snapshot["running"] == []


def test_state_without_results_and_resume():
    state = DashboardState()
    state.apply({"type": "unknown", "time": 1})
    assert state.best_trial() is None

    state.apply({"type": "run_start", "time": 10, "timeout": None, "max_evals": -1})
    state.apply({"type": "round_start", "time": 10, "round": 0, "num_samples": -1})
    state.apply({"type": "trial_start", "time": 10, "round": 0, "trial_id": "a"})
    # unknown trials are ignored
    state.apply({"type": "trial_result", "time": 11, "round": 0, "trial_id": "x"})
    state.apply({"type": "trial_end", "time": 11, "round": 0, "trial_id": "x"})
    state.apply(
        {
            "type": "trial_end",
            "time": 12,
            "round": 0,
            "trial_id": "a",
            "status": "COMPLETED",
        }
    )
    progress = state.progress(now=20)
    # no result reported
    assert progress["counts"]["TERMINATED"] == 1
    # no budget: no estimate
    assert progress["trials_total"] is None and progress["time_left"] is None
    assert progress["eta"] is None

    # resumed run: start time and the time spent in the round are kept
    state.apply({"type": "round_end", "time": 20, "round": 0})
    state.apply({"type": "run_start", "time": 50, "timeout": 100})
    state.apply({"type": "round_start", "time": 50, "round": 0, "num_samples": 4})
    state.apply({"type": "trial_start", "time": 50, "round": 0, "trial_id": "a"})
    assert state.run["start"] == 10
    assert state._round(0)["duration"] == 10
    assert state.progress(now=55)["elapsed"] == 15
    # restored trial is running again (not duplicated)
    assert len(state.trials) == 1 and state.progress(now=55)["counts"]["RUNNING"] == 1


def test_standardize_config():
    # default format
    config = {
        "task_type": "classification",
        "encoder": {"encoder": "DataEncoding", "dummy_coding": True},
        "model": {"model": "LightGBM", "n_estimators": np.int64(10)},
    }
    assert standardize_config(config) == {
        "encoder": {
            "method": "DataEncoding",
            "hyperparameters": {"dummy_coding": True},
        },
        "model": {"method": "LightGBM", "hyperparameters": {"n_estimators": 10}},
    }
    # the trial config is not changed
    assert config["model"]["model"] == "LightGBM"

    # HyperOpt format (method name prefixed)
    config = {"model": {"model_3": "LightGBM", "LightGBM_n_estimators": 10}}
    assert standardize_config(config) == {
        "model": {"method": "LightGBM", "hyperparameters": {"n_estimators": 10}}
    }

    # unknown format
    config = {"model": {"something": 1}}
    assert standardize_config(config) == {
        "model": {"method": None, "hyperparameters": {"something": 1}}
    }


def test_callback_events(tmp_path):
    recorder = EventRecorder(str(tmp_path))
    callback = DashboardCallback(recorder, round_index=2)

    error_file = tmp_path / "error.txt"
    error_file.write_text("x" * 5000 + "Traceback: boom")
    trial = SimpleNamespace(
        trial_id="t1",
        config={"model": {"model": "A", "alpha": 0.1}, "task_type": "regression"},
        error_file=str(error_file),
    )
    callback.on_trial_start(0, [trial], trial)
    callback.on_trial_result(
        0,
        [trial],
        trial,
        {"loss": math.inf, "training_status": "TIMEOUT", "training_iteration": 1},
    )
    callback.on_trial_complete(0, [trial], trial)
    callback.on_trial_error(0, [trial], trial)

    trial_2 = SimpleNamespace(trial_id="t2", config=None, error_msg="failed")
    callback.on_trial_start(0, [trial_2], trial_2)
    callback.on_trial_error(0, [trial_2], trial_2)

    # config that cannot be read
    trial_3 = SimpleNamespace(trial_id="t3", config=42)
    callback.on_trial_start(0, [trial_3], trial_3)

    events, _ = read_events(recorder.path)
    assert [event["type"] for event in events] == [
        "trial_start",
        "trial_result",
        "trial_end",
        "trial_end",
        "trial_start",
        "trial_end",
        "trial_start",
    ]
    assert all(event["round"] == 2 for event in events)
    assert events[0]["config"] == {
        "model": {"method": "A", "hyperparameters": {"alpha": 0.1}}
    }
    assert events[1]["loss"] is None and events[1]["training_status"] == "TIMEOUT"
    assert events[3]["status"] == "ERROR"
    assert events[3]["error"].endswith("Traceback: boom")
    assert len(events[3]["error"]) == 4000
    assert events[4]["config"] == {} and events[5]["error"] == "failed"
    assert events[6]["config"] == {}


def test_progress_reporter_dashboard():
    from InsurAutoML.utils.optimize import SilentReporter, get_progress_reporter

    # explicit reporter: Ray Tune's default one clears the notebook output
    for name in ("Dashboard", None):
        reporter = get_progress_reporter(name, max_evals=8, max_error=1)
        assert isinstance(reporter, SilentReporter)
        assert reporter.should_report([], done=True) is False
        assert reporter.report([], True) is None


def test_dashboard_announce(monkeypatch, capsys):
    import sys
    import types

    from InsurAutoML.hpo import base

    announce = base.AutoTabularBase._dashboard_announce
    announce("Track <run> at", "http://127.0.0.1:1")
    assert capsys.readouterr().out == "[Dashboard] Track <run> at http://127.0.0.1:1\n"
    announce("Recorded")
    assert capsys.readouterr().out == "[Dashboard] Recorded\n"

    # notebooks: clickable link (IPython is not a dependency: fake module)
    shown = []
    display = types.ModuleType("IPython.display")
    display.HTML = lambda data: types.SimpleNamespace(data=data)
    display.display = shown.append
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.display", display)
    monkeypatch.setattr(base, "type_of_script", lambda: "jupyter")
    announce("Track <run> at", "http://127.0.0.1:1")
    assert capsys.readouterr().out == ""
    assert 'href="http://127.0.0.1:1"' in shown[0].data
    assert "Track &lt;run&gt; at" in shown[0].data


def test_quiet_ray_output(monkeypatch):
    from InsurAutoML.utils.optimize import RAY_TUNE_LOGGERS, quiet_ray_output

    monkeypatch.delenv("RAY_AIR_NEW_OUTPUT", raising=False)
    tune_logger = logging.getLogger(RAY_TUNE_LOGGERS[0])
    level = tune_logger.level

    with quiet_ray_output():
        assert tune_logger.level == logging.CRITICAL
        assert os.environ["RAY_AIR_NEW_OUTPUT"] == "0"
    assert tune_logger.level == level
    assert "RAY_AIR_NEW_OUTPUT" not in os.environ

    with quiet_ray_output(enable=False):
        assert tune_logger.level == level


def test_state_of_stopped_run():
    # killed process: no round_end/run_end, the clock stops at the last event
    from InsurAutoML.dashboard.events import STALE_AFTER

    state = DashboardState()
    state.apply({"type": "run_start", "time": 0, "timeout": 10000})
    state.apply({"type": "round_start", "time": 0, "round": 0, "num_samples": 4})
    state.apply({"type": "trial_start", "time": 10, "round": 0, "trial_id": "a"})
    state.apply(
        {
            "type": "trial_result",
            "time": 50,
            "round": 0,
            "trial_id": "a",
            "loss": 1.0,
            "training_status": "FITTED",
        }
    )

    running = state.progress(now=100)
    assert running["elapsed"] == 100 and not running["stale"]
    stopped = state.progress(now=50 + STALE_AFTER + 1000)
    assert stopped["stale"] and stopped["now"] == 50 and stopped["elapsed"] == 50

    # a restored trial is timed from its restart
    state.apply(
        {
            "type": "trial_end",
            "time": 60,
            "round": 0,
            "trial_id": "a",
            "status": "COMPLETED",
        }
    )
    state.apply({"type": "trial_start", "time": 500, "round": 0, "trial_id": "a"})
    state.apply(
        {
            "type": "trial_end",
            "time": 520,
            "round": 0,
            "trial_id": "a",
            "status": "COMPLETED",
        }
    )
    assert next(iter(state.trials.values()))["duration"] == 20
