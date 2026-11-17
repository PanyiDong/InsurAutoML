"""
Tests of the search space description, sampling and distances of the dashboard map.
"""

import math

import numpy as np
import pytest
from ray import tune

from InsurAutoML.dashboard.space import (
    EMPHASIS_WEIGHT,
    MODEL_WEIGHT,
    SAME_METHOD_SCALE,
    distances,
    encode,
    sample_configs,
    space_schema,
    step_weights,
)


def _space():
    return {
        "task_type": "classification",
        "encoder": tune.choice(
            [{"encoder": "DataEncoding", "dummy_coding": tune.choice([True, False])}]
        ),
        "imputer": tune.choice([{"imputer": "no_processing"}]),
        "scaling": {"scaling": "Standardize", "with_mean": True},
        "model": tune.choice(
            [
                {
                    "model": "LightGBM",
                    "learning_rate": tune.qloguniform(1e-3, 1.0, 1e-3),
                    "num_leaves": tune.qrandint(2, 64),
                    "boosting": tune.choice(["gbdt", "dart"]),
                    "fixed": 3,
                    "single": tune.choice(["only"]),
                    "custom": tune.sample_from(lambda _: 1),
                },
                {"model": "Logistic", "C": tune.uniform(0.0, 10.0)},
                {"model": "Dummy"},
            ]
        ),
        # not a step: options are not settings of methods
        "other": tune.choice([1, 2]),
    }


def test_space_schema():
    schema = space_schema(_space())

    assert schema["steps"] == ["encoder", "imputer", "scaling", "model"]
    assert schema["methods"]["encoder"] == {
        "DataEncoding": {
            "dummy_coding": {"type": "categorical", "categories": [True, False]}
        }
    }
    assert schema["methods"]["imputer"] == {"no_processing": {}}
    # constant hyperparameters are left out
    assert schema["methods"]["scaling"] == {"Standardize": {}}
    lightgbm = schema["methods"]["model"]["LightGBM"]
    assert lightgbm["learning_rate"] == {
        "type": "float", "low": 1e-3, "high": 1.0, "log": True,
    }
    assert lightgbm["num_leaves"] == {
        "type": "int", "low": 2.0, "high": 64.0, "log": False,
    }
    assert lightgbm["boosting"]["categories"] == ["gbdt", "dart"]
    # constants, single categories, functions are not supported
    assert set(lightgbm) == {"learning_rate", "num_leaves", "boosting"}
    assert schema["methods"]["model"]["Dummy"] == {}

    # HyperOpt-like method key, non-JSON categories
    schema = space_schema(
        {"model": tune.choice([{"model_1": "A", "A_k": tune.choice([None, int])}])}
    )
    assert schema["methods"]["model"]["A"] == {
        "k": {"type": "categorical", "categories": [None, str(int)]}
    }

    assert space_schema({"task_type": "regression"}) is None
    assert space_schema({"model": tune.choice([{"other": 1}])}) is None


def test_sample_configs():
    schema = space_schema(_space())
    configs = sample_configs(schema, 200, seed=1)

    assert configs == sample_configs(schema, 200, seed=1)
    assert len(configs) == 200
    assert {config["model"]["method"] for config in configs} == {
        "LightGBM", "Logistic", "Dummy",
    }
    for config in configs:
        assert set(config) == {"encoder", "imputer", "scaling", "model"}
        params = config["model"]["hyperparameters"]
        if config["model"]["method"] == "LightGBM":
            assert 1e-3 <= params["learning_rate"] <= 1.0
            assert isinstance(params["num_leaves"], int)
            assert 2 <= params["num_leaves"] <= 64
            assert params["boosting"] in ("gbdt", "dart")
        elif config["model"]["method"] == "Logistic":
            assert 0 <= params["C"] <= 10
        else:
            assert params == {}
    # log-uniform: small values are as likely as large ones
    rates = [
        c["model"]["hyperparameters"]["learning_rate"]
        for c in configs
        if c["model"]["method"] == "LightGBM"
    ]
    assert np.median(np.log10(rates)) < -1


def test_step_weights():
    schema = space_schema(_space())
    weights = step_weights(schema)
    # a step with a single method without hyperparameters does not count
    assert weights == {"encoder": 1.0, "imputer": 0.0, "scaling": 0.0, "model": MODEL_WEIGHT}
    assert step_weights(schema, "encoder")["encoder"] == EMPHASIS_WEIGHT


def _config(model, **params):
    return {"model": {"method": model, "hyperparameters": params}}


def test_encode_and_distances():
    schema = {
        "steps": ["model"],
        "methods": {
            "model": {
                "A": {
                    "rate": {"type": "float", "low": 1e-4, "high": 1.0, "log": True},
                    "depth": {"type": "int", "low": 0.0, "high": 10.0, "log": False},
                    "kind": {"type": "categorical", "categories": ["x", "y"]},
                },
                "B": {},
            }
        },
    }
    configs = [
        _config("A", rate=1e-2, depth=5, kind="x"),
        _config("A", rate=1e-2, depth=5, kind="y"),
        _config("A", rate=1.0, depth=10, kind="x"),
        _config("B"),
        _config("Unknown"),
        # missing, non-numeric, non-finite, out of range, unknown category
        _config("A", depth="deep", rate=math.inf, kind="z"),
        _config("A", rate=-1.0, depth=100),
        {},
    ]
    encoded = encode(schema, configs)
    assert len(encoded) == len(configs)
    assert encoded.methods["model"].tolist() == [0, 0, 0, 1, -1, 0, 0, -1]
    values = encoded.values[("model", "A")]
    # log scale: 1e-2 is half way between 1e-4 and 1
    assert values[0].tolist() == pytest.approx([0.5, 0.5, 0.0])
    assert np.isnan(values[5, 0]) and np.isnan(values[5, 1]) and values[5, 2] == -1
    # clipped to the range, non positive values of log scales at the lower bound
    assert values[6, :2].tolist() == [0.0, 1.0]

    weights = {"model": 1.0}
    matrix = distances(schema, encoded, encoded, weights)
    assert matrix.shape == (8, 8)
    assert matrix[0, 0] == 0
    # categorical differs: 1 of 3 hyperparameters
    assert matrix[0, 1] == pytest.approx(SAME_METHOD_SCALE / 3)
    assert matrix[0, 2] == pytest.approx(SAME_METHOD_SCALE * (0.5 + 0.5 + 0) / 3)
    # other method, unknown method: 1
    assert matrix[0, 3] == 1 and matrix[0, 4] == 1 and matrix[4, 4] == 1
    # same method, settings closer than any other method
    assert matrix[0, :3].max() < 1
    # missing hyperparameters left out (only the category differs)
    assert matrix[0, 5] == pytest.approx(SAME_METHOD_SCALE)
    assert matrix[3, 3] == 0

    # no weight / empty sets
    assert not distances(schema, encoded, encoded, {"model": 0.0}).any()
    assert distances(schema, encode(schema, []), encoded, weights).shape == (0, 8)
