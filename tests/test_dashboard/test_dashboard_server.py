"""
Tests of the web server and the replay command of the dashboard.
"""

import json
import os
import socket
import time
import urllib.error
import urllib.request

import pytest

from InsurAutoML.dashboard import server as server_module
from InsurAutoML.dashboard.__main__ import main, resolve_events
from InsurAutoML.dashboard.events import EVENTS_DIR, EVENTS_FILE, EventRecorder, EventTail
from InsurAutoML.dashboard.server import (
    DashboardServer,
    plotly_js_path,
    start_dashboard,
    stop_dashboard,
)

SCHEMA = {
    "steps": ["model"],
    "methods": {
        "model": {
            "A": {"rate": {"type": "float", "low": 1e-3, "high": 1.0, "log": True}},
            "B": {},
        }
    },
}


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _record(directory, model_name="run", schema=SCHEMA):
    recorder = EventRecorder(str(directory))
    recorder.emit("run_start", model_name=model_name, timeout=60, max_evals=4, rounds=1)
    if schema is not None:
        recorder.emit("space", schema=schema)
    recorder.emit("round_start", round=0, name=model_name, num_samples=4)
    for key, model, loss in (("a", "A", 0.2), ("b", "B", 0.5)):
        recorder.emit(
            "trial_start", round=0, trial_id=key,
            config={"model": {"method": model, "hyperparameters": {"rate": 0.01} if model == "A" else {}}},
        )
        recorder.emit("trial_result", round=0, trial_id=key, loss=loss,
                      training_status="FITTED", iteration=1)
        recorder.emit("trial_end", round=0, trial_id=key, status="COMPLETED")
    return recorder


def _get(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.status, response.headers.get("Content-Type"), response.read()


def _get_json(url):
    return json.loads(_get(url)[2])


def _wait_map(url, emphasis="model"):
    end = time.time() + 60
    while time.time() < end:
        result = _get_json(url + "/api/map?emphasis=" + emphasis)
        if result["status"] != "building":
            return result
        time.sleep(0.1)
    raise TimeoutError


@pytest.fixture
def served(tmp_path, monkeypatch):
    monkeypatch.setattr(server_module, "DEFAULT_PORT", _free_port())
    recorder = _record(tmp_path / "run" / EVENTS_DIR)
    dashboard = DashboardServer(recorder.path)
    url = dashboard.start()
    yield dashboard, url, recorder
    dashboard.stop()


def test_server_pages(served, monkeypatch):
    dashboard, url, _ = served
    assert dashboard.running and dashboard.start() == url

    status, content_type, body = _get(url + "/")
    assert status == 200 and content_type.startswith("text/html")
    assert b"/static/app.js" in body
    for name, kind in (("app.js", "javascript"), ("style.css", "css")):
        status, content_type, _ = _get(url + "/static/" + name)
        assert status == 200 and kind in content_type

    # plotly.js of the plotly package (offline)
    if plotly_js_path() is not None:
        assert _get(url + "/static/plotly.min.js")[0] == 200
    monkeypatch.setattr(server_module, "plotly_js_path", lambda: None)
    with pytest.raises(urllib.error.HTTPError) as err:
        _get(url + "/static/plotly.min.js")
    assert err.value.code == 404

    # only the files of the dashboard
    for path in ("/static/../server.py", "/nothing", "/static/events.py"):
        with pytest.raises(urllib.error.HTTPError) as err:
            _get(url + path)
        assert err.value.code == 404


def test_server_api(served):
    dashboard, url, recorder = served

    state = _get_json(url + "/api/state")
    assert state["run"]["model_name"] == "run"
    assert state["map_steps"] == ["model"]
    assert state["best"]["trial_id"] == "a"
    assert len(state["trials"]) == 2
    assert state["path"] == recorder.path
    assert state["server_time"] >= state["last_time"]

    result = _wait_map(url)
    assert result["status"] == "ready"
    assert len(result["layout"]["x"]) == 1500

    scores = _get_json(url + "/api/map/scores?emphasis=model")
    assert scores["status"] == "ready"
    assert scores["scores"] == {"0:a": 1.0, "0:b": 0.0}
    assert set(scores["positions"]) == {"0:a", "0:b"}
    # unchanged trials: cached
    assert _get_json(url + "/api/map/scores?emphasis=model") == scores

    # new events are followed
    recorder.emit("trial_start", round=0, trial_id="c", config={})
    assert len(_get_json(url + "/api/state")["trials"]) == 3

    # another run
    other = _record(os.path.join(os.path.dirname(os.path.dirname(recorder.path)), "other", EVENTS_DIR),
                    model_name="other", schema=None)
    dashboard.set_path(other.path)
    state = _get_json(url + "/api/state")
    assert state["run"]["model_name"] == "other" and state["map_steps"] == []
    assert _get_json(url + "/api/map")["status"] == "waiting"
    assert _get_json(url + "/api/map/scores")["status"] == "waiting"


def test_server_ports_and_url(tmp_path, monkeypatch):
    port = _free_port()
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", port))
        busy.listen()
        dashboard = DashboardServer(str(tmp_path), port=port)
        url = dashboard.start()
        try:
            # port in use: the next free one
            assert dashboard.port != port and url.endswith(":{}".format(dashboard.port))
        finally:
            dashboard.stop()
    assert not dashboard.running

    monkeypatch.setattr(server_module, "PORT_ATTEMPTS", 0)
    with pytest.raises(OSError, match="No free port"):
        DashboardServer(str(tmp_path), port=port)._bind()

    assert DashboardServer(str(tmp_path), host="0.0.0.0").url.startswith("http://127.0.0.1:")
    assert DashboardServer(str(tmp_path), host="::1").url.startswith("http://[::1]:")
    monkeypatch.setenv(server_module.ENV_HOST, "localhost")
    monkeypatch.setenv(server_module.ENV_PORT, "9999")
    assert DashboardServer(str(tmp_path)).url == "http://localhost:9999"


def test_start_dashboard(tmp_path, monkeypatch):
    monkeypatch.setattr(server_module, "DEFAULT_PORT", _free_port())
    first = _record(tmp_path / "first")
    second = _record(tmp_path / "second", model_name="second")
    stop_dashboard()
    try:
        url = start_dashboard(first.path)
        assert url is not None
        assert server_module.active_dashboard().path == first.path
        # one server per process: switched to the new run
        assert start_dashboard(second.path) == url
        assert _get_json(url + "/api/state")["run"]["model_name"] == "second"
    finally:
        stop_dashboard()
    assert server_module.active_dashboard() is None

    # disabled
    monkeypatch.setenv(server_module.ENV_SERVE, "0")
    assert start_dashboard(first.path) is None
    monkeypatch.delenv(server_module.ENV_SERVE)

    # no port: not started
    def no_port(self):
        raise OSError("no port")

    monkeypatch.setattr(DashboardServer, "_bind", no_port)
    assert start_dashboard(first.path) is None


def test_event_tail(tmp_path):
    recorder = _record(tmp_path)
    tail = EventTail(str(tmp_path))
    assert len(tail.refresh().trials) == 2

    # a new run replaces the log: rebuilt
    os.remove(recorder.path)
    # deleted log: state kept
    assert len(tail.refresh().trials) == 2
    recorder = EventRecorder(str(tmp_path))
    recorder.emit("run_start", model_name="new")
    recorder.emit("round_start", round=0, num_samples=1)
    for key in "xyz":
        recorder.emit("trial_start", round=0, trial_id=key, config={})
    state = tail.refresh()
    assert state.run["model_name"] == "new" and len(state.trials) == 3

    # smaller log (truncated)
    with open(recorder.path, "w") as out_f:
        out_f.write(json.dumps({"type": "run_start", "time": 1, "model_name": "new"}) + "\n")
    state = tail.refresh()
    assert len(state.trials) == 0


def test_replay_command(tmp_path, monkeypatch, capsys):
    recorder = _record(tmp_path / "model" / EVENTS_DIR)
    run_dir = str(tmp_path / "model")
    for path in (run_dir, os.path.dirname(recorder.path), recorder.path):
        assert resolve_events(path) == recorder.path
    with pytest.raises(FileNotFoundError):
        resolve_events(str(tmp_path))
    assert main([str(tmp_path / "missing")]) == 1

    # serves until stopped (here: right away)
    started = {}
    opened = []

    def start(self):
        started["path"] = self.path
        return "http://127.0.0.1:1"

    monkeypatch.setattr(DashboardServer, "start", start)
    monkeypatch.setattr(DashboardServer, "running", property(lambda self: False))
    monkeypatch.setattr("webbrowser.open", opened.append)
    assert main([run_dir, "--port", "1", "--open"]) == 0
    assert started["path"] == recorder.path and opened == ["http://127.0.0.1:1"]
    assert "http://127.0.0.1:1" in capsys.readouterr().out
    assert os.path.basename(recorder.path) == EVENTS_FILE
