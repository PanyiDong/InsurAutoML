"""
Tests of the map of the search space.
"""

import os
import time

import numpy as np
import pytest

from InsurAutoML.dashboard import layout as layout_module
from InsurAutoML.dashboard.layout import MapBuilder, SpaceMap, trial_scores

SCHEMA = {
    "steps": ["scaling", "model"],
    "methods": {
        "scaling": {"Standardize": {}, "MinMax": {}},
        "model": {
            "A": {"rate": {"type": "float", "low": 1e-3, "high": 1.0, "log": True}},
            "B": {"depth": {"type": "int", "low": 1.0, "high": 10.0, "log": False}},
            "C": {},
        },
    },
}


def _trial(key, model, loss=None, status="FITTED", scaling="Standardize", **params):
    return {
        "key": key,
        "loss": loss,
        "training_status": status,
        "status": status,
        "config": {
            "scaling": {"method": scaling, "hyperparameters": {}},
            "model": {"method": model, "hyperparameters": params},
        },
    }


def test_trial_scores():
    assert trial_scores([]) == {}
    assert trial_scores([_trial("a", "A", loss=1.0)]) == {"a": 1.0}
    scores = trial_scores(
        [
            _trial("a", "A", loss=3.0),
            _trial("b", "A", loss=1.0),
            _trial("c", "A", loss=2.0),
            _trial("d", "A", loss=2.0),
            _trial("e", "A", loss=None),
            _trial("f", "A", loss=0.5, status="TIMEOUT"),
        ]
    )
    # best 1, worst 0, ties share the rank, unfitted trials not scored
    assert scores == {"b": 1.0, "c": pytest.approx(0.5), "d": pytest.approx(0.5), "a": 0.0}


def test_space_map(tmp_path):
    space_map = SpaceMap(SCHEMA, n_references=90, cache_dir=str(tmp_path)).build()

    assert space_map.coords.shape == (90, 2)
    assert space_map.coords.min() >= 0 and space_map.coords.max() <= 1
    assert space_map.group_step == "model"
    layout = space_map.layout()
    assert len(layout["x"]) == 90 and layout["emphasis"] is None
    assert {label["text"] for label in layout["labels"]} == {"A", "B", "C"}

    # cached: the same layout without t-SNE
    assert os.path.isfile(space_map.cache_path)
    cached = SpaceMap(SCHEMA, n_references=90, cache_dir=str(tmp_path))
    assert cached._load()
    np.testing.assert_allclose(cached.build().coords, space_map.coords, atol=1e-4)
    # corrupted cache: rebuilt
    with open(space_map.cache_path, "w") as out_f:
        out_f.write("not json")
    assert not SpaceMap(SCHEMA, n_references=90, cache_dir=str(tmp_path))._load()

    # trials are placed in the area of their model
    trials = [
        _trial("a", "A", loss=1.0, rate=0.01),
        _trial("b", "B", loss=2.0, depth=3),
        _trial("x", "Unknown", loss=3.0),
    ]
    positions = space_map.place([t["config"] for t in trials], [t["key"] for t in trials])
    assert set(positions) == {"a", "b", "x"}
    model = space_map.encoded.methods["model"]
    for key, order in (("a", 0), ("b", 1)):
        position = np.asarray(positions[key])
        nearest = np.argmin(((space_map.coords - position) ** 2).sum(axis=1))
        assert model[nearest] == order
    assert space_map.place([], []) == {}

    # the performance spreads to the settings around a trial, of its model only
    field = space_map.field(
        [trials[0]["config"]], [positions["a"]], [1.0]
    )
    score = np.array([np.nan if s is None else s for s in field["score"]])
    confidence = np.asarray(field["confidence"])
    assert np.all(confidence[model != 0] == 0)
    assert confidence[model == 0].max() > 0.5
    assert np.nanmax(score) == pytest.approx(1.0)
    empty = space_map.field([], [], [])
    assert empty["score"] == [None] * 90 and empty["confidence"] == [0.0] * 90


def test_space_map_emphasis_and_trivial_space():
    space_map = SpaceMap(SCHEMA, emphasis="scaling", n_references=40).build()
    assert space_map.group_step == "scaling"
    assert {label["text"] for label in space_map.labels()} == {"Standardize", "MinMax"}

    # nothing to separate: a single point
    trivial = {"steps": ["model"], "methods": {"model": {"A": {}}}}
    space_map = SpaceMap(trivial, n_references=10).build()
    assert not space_map.coords.any()
    assert SpaceMap(trivial).labels() == []


def _wait(builder, emphasis=None, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        status = builder.status(emphasis)["status"]
        if status != "building":
            return status
        time.sleep(0.05)
    raise TimeoutError


def test_map_builder(tmp_path, monkeypatch):
    builder = MapBuilder(cache_dir=str(tmp_path), n_references=60)
    assert builder.status() == {"status": "waiting"}
    assert builder.build() is None and builder.scores([]) is None

    builder.set_schema(SCHEMA)
    assert builder.status()["status"] == "building"
    assert _wait(builder) == "ready"
    # "model" is the default grouping
    assert builder.get("model") is builder.get(None) is not None

    trials = [
        _trial("a", "A", loss=1.0, rate=0.01),
        _trial("b", "B", loss=2.0, depth=3),
        _trial("r", "C", status="RUNNING"),
        {"key": "n", "config": {}},
    ]
    scores = builder.scores(trials)
    assert set(scores["positions"]) == {"a", "b", "r"}
    assert scores["scores"] == {"a": 1.0, "b": 0.0}
    assert len(scores["score"]) == len(scores["confidence"]) == 60

    # unknown step: default grouping; other step built in the calling thread
    assert builder.status("nope")["status"] == "ready"
    assert builder.build("scaling").group_step == "scaling"

    # same schema: kept; another schema: maps rebuilt
    builder.set_schema(dict(SCHEMA))
    assert builder.get() is not None
    builder.set_schema({"steps": ["model"], "methods": {"model": {"A": {}, "B": {}}}})
    assert builder.get() is None

    # failure: map unavailable
    def fail(self):
        raise RuntimeError("no layout")

    monkeypatch.setattr(layout_module.SpaceMap, "build", fail)
    builder.set_schema(SCHEMA)
    builder.status()
    assert _wait(builder) == "unavailable"
    assert builder.status()["error"] == "no layout"
