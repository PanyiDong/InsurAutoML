"""
File Name: test_dashboard_fit.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Actuarial and Risk Management Sciences, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.9
Relative Path: /tests/test_dashboard/test_dashboard_fit.py
File Created: Monday, 16th November 2026 9:02:52 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 16th November 2026 11:34:36 pm
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
The dashboard (default progress reporter) records the optimization of a fit.
"""

import os

import pytest

import InsurAutoML
from InsurAutoML import load_data
from InsurAutoML.dashboard import EVENTS_DIR, EVENTS_FILE, load_state
from InsurAutoML.dashboard.server import ENV_SERVE, active_dashboard

DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "example",
    "example_data",
)


@pytest.fixture(autouse=True)
def _in_tmp_dir(tmp_path, monkeypatch):
    # the models write files in the working directory (e.g. catboost_info)
    monkeypatch.chdir(tmp_path)


def _heart():
    data = load_data().load(DATA, "heart")["heart"]
    return data.drop(columns=["HeartDisease"]), data[["HeartDisease"]]


def test_fit_records_dashboard_events(tmp_path, capsys):
    X, y = _heart()
    model_name = "dashboard_heart"
    mol = InsurAutoML.AutoTabular(
        model_name=model_name,
        temp_directory=str(tmp_path / "tmp"),
        n_estimators=2,
        ensemble_strategy="bagging",
        timeout=30,
        max_evals=4,
        validation=True,
        objective="accuracy",
        save=False,
        seed=1,
    )
    mol.fit(X, y)

    path = os.path.join(str(tmp_path / "tmp"), model_name, EVENTS_DIR, EVENTS_FILE)
    assert os.path.isfile(path)
    # the URL of the dashboard is printed (instead of the Ray Tune output)
    assert mol.dashboard_url is not None
    assert (
        "Track the optimization of {} at {}".format(model_name, mol.dashboard_url)
        in capsys.readouterr().out
    )
    assert active_dashboard().path == path

    state = load_state(path)
    # search space of the map
    assert state.schema["steps"][-1] == "model"
    assert len(state.schema["methods"]["model"]) > 1
    progress = state.progress()
    assert state.finished
    assert state.run["model_name"] == model_name and state.run["rounds"] == 2
    assert [item["name"] for item in state.rounds] == [
        "{}_bagging_{}".format(model_name, index) for index in (1, 2)
    ]
    assert progress["trials_started"] >= 2
    assert progress["counts"]["RUNNING"] == 0
    best = state.best_trial()
    assert best is not None
    # standardized settings of the trial
    assert best["config"]["model"]["method"] is not None
    assert set(best["config"]) >= {"encoder", "model"}


def test_fit_without_web_server(tmp_path, capsys, monkeypatch):
    # events recorded, no web server
    monkeypatch.setenv(ENV_SERVE, "0")
    X, y = _heart()
    model_name = "recorded_heart"
    mol = InsurAutoML.AutoTabular(
        model_name=model_name,
        temp_directory=str(tmp_path / "tmp"),
        n_estimators=1,
        timeout=20,
        max_evals=2,
        validation=True,
        objective="accuracy",
        search_algo="HyperOpt",
        save=False,
        seed=1,
    )
    mol.fit(X, y)

    assert mol.dashboard_url is None
    out = capsys.readouterr().out
    assert "python -m InsurAutoML.dashboard" in out
    state = load_state(os.path.join(str(tmp_path / "tmp"), model_name, EVENTS_DIR))
    assert state.finished
    # the space of the map is described whatever the search algorithm
    assert state.schema is not None
    for trial in state.trials.values():
        assert trial["config"]["model"]["method"] in state.schema["methods"]["model"]


def test_dashboard_space_failure(tmp_path, monkeypatch):
    mol = InsurAutoML.AutoTabular(model_name="space", temp_directory=str(tmp_path))
    mol._logger = __import__("logging").getLogger("test")

    def fail(X, y):
        raise RuntimeError("no space")

    monkeypatch.setattr(mol, "get_hyperparameter_space", fail)
    assert mol._dashboard_space(None, None) is None
    # search algorithm restored
    assert mol.search_algo == "RandomSearch"


def test_fit_with_ray_reporter_has_no_dashboard(tmp_path):
    X, y = _heart()
    model_name = "cli_heart"
    mol = InsurAutoML.AutoTabular(
        model_name=model_name,
        temp_directory=str(tmp_path / "tmp"),
        n_estimators=1,
        timeout=20,
        max_evals=2,
        validation=True,
        objective="accuracy",
        progress_reporter="CLIReporter",
        save=False,
        seed=1,
    )
    mol.fit(X, y)

    assert mol._dashboard is None
    assert not os.path.exists(
        os.path.join(str(tmp_path / "tmp"), model_name, EVENTS_DIR)
    )


def test_invalid_progress_reporter(tmp_path):
    X, y = _heart()
    mol = InsurAutoML.AutoTabular(
        model_name="invalid_reporter",
        temp_directory=str(tmp_path / "tmp"),
        progress_reporter="TensorBoard",
    )
    with pytest.raises(ValueError, match="Progress reporter must be one of"):
        mol.fit(X, y)
