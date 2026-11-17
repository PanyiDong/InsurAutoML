"""
File Name: space.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Actuarial and Risk Management Sciences, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.9
Relative Path: /InsurAutoML/dashboard/space.py
File Created: Monday, 16th November 2026 8:30:56 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 16th November 2026 11:33:55 pm
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

Search space of the dashboard map.

The search space of InsurAutoML is a tree: every step of the pipeline (encoder,
imputer, ..., model) chooses one method, and every method has its own
hyperparameters. Here the space is described as plain JSON (schema), settings are
sampled from it, and the distance between two settings is defined:

    distance = sum over steps of weight(step) * d(step) / sum of weights
    d(step)  = 1 if the methods differ, otherwise
               SAME_METHOD_SCALE * mean difference of the hyperparameters

where numeric hyperparameters are scaled to [0, 1] by their range (log scale for
log-uniform ones) and categorical hyperparameters differ by 0 or 1.
"""

from __future__ import annotations

import math
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# within the same method, settings are always closer than to other methods
SAME_METHOD_SCALE = 0.8

# default weights of the steps (the model dominates the map)
DEFAULT_WEIGHT = 1.0
MODEL_WEIGHT = 2.0
EMPHASIS_WEIGHT = 4.0

Schema = Dict[str, Any]
Config = Dict[str, Dict[str, Any]]


# ------------------------------------------------------------------ schema


def _domain_spec(domain: Any) -> Optional[Dict[str, Any]]:
    """JSON description of a Ray Tune search domain (None: not supported)."""
    from ray.tune.search.sample import Categorical, Float, Integer, Quantized

    if isinstance(domain, Categorical):
        categories = list(domain.categories)
        if len(categories) < 2:
            return None
        return {"type": "categorical", "categories": [_label(c) for c in categories]}

    if isinstance(domain, (Float, Integer)):
        sampler = domain.get_sampler()
        if isinstance(sampler, Quantized):
            sampler = sampler.get_sampler()
        low, high = domain.lower, domain.upper
        if low is None or high is None or not high > low:
            return None
        log = "LogUniform" in type(sampler).__name__ and low > 0
        return {
            "type": "float" if isinstance(domain, Float) else "int",
            "low": float(low),
            "high": float(high),
            "log": bool(log),
        }

    return None


def _label(value: Any) -> Any:
    """Category as a JSON value."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _method_name(step: str, method: Dict[str, Any]) -> Optional[str]:
    if step in method:
        return str(method[step])
    key = next((key for key in method if key.startswith(step + "_")), None)
    return str(method[key]) if key is not None else None


def space_schema(space: Dict[str, Any]) -> Optional[Schema]:
    """
    Describe a search space in Ray Tune format (RandomSearch), e.g.
    {"encoder": tune.choice([{"encoder": "DataEncoding", "dummy_coding":
    tune.choice([True, False])}, ...]), ..., "task_type": "classification"}, as

    {"steps": [...], "methods": {step: {method: {param: spec}}}}

    with spec {"type": "float"/"int", "low", "high", "log"} or
    {"type": "categorical", "categories"}; constant hyperparameters are left out.
    """
    from ray.tune.search.sample import Categorical

    steps: List[str] = []
    methods: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for step, value in space.items():
        if isinstance(value, Categorical):
            options = list(value.categories)
        elif isinstance(value, dict):
            options = [value]
        else:
            continue
        if not options or not all(isinstance(item, dict) for item in options):
            continue

        step_methods: Dict[str, Dict[str, Any]] = {}
        for option in options:
            name = _method_name(step, option)
            if name is None:
                continue
            # HyperOpt format: hyperparameters prefixed by the method name (removed
            # in the settings of the trials, see callback.standardize_config)
            prefixed = step not in option
            params = {}
            for param, domain in option.items():
                if param == step or param.startswith(step + "_"):
                    continue
                spec = _domain_spec(domain)
                if spec is not None:
                    param = str(param)
                    if prefixed and param.startswith(name + "_"):
                        param = param[len(name) + 1 :]
                    params[param] = spec
            step_methods[name] = params

        if step_methods:
            steps.append(str(step))
            methods[str(step)] = step_methods

    if not steps:
        return None

    return {"steps": steps, "methods": methods}


def sample_configs(schema: Schema, n: int, seed: int = 0) -> List[Config]:
    """Sample settings from the space (methods uniformly, as tune.choice)."""
    rng = np.random.default_rng(seed)
    configs: List[Config] = []
    for _ in range(n):
        config: Config = {}
        for step in schema["steps"]:
            names = list(schema["methods"][step])
            name = names[int(rng.integers(len(names)))]
            hyperparameters = {}
            for param, spec in schema["methods"][step][name].items():
                hyperparameters[param] = _sample(spec, rng)
            config[step] = {"method": name, "hyperparameters": hyperparameters}
        configs.append(config)

    return configs


def _sample(spec: Dict[str, Any], rng: np.random.Generator) -> Any:
    if spec["type"] == "categorical":
        return spec["categories"][int(rng.integers(len(spec["categories"])))]
    low, high = spec["low"], spec["high"]
    if spec["log"]:
        value = math.exp(rng.uniform(math.log(low), math.log(high)))
    else:
        value = rng.uniform(low, high)
    return int(round(value)) if spec["type"] == "int" else float(value)


# ---------------------------------------------------------------- encoding


def step_weights(schema: Schema, emphasis: Optional[str] = None) -> Dict[str, float]:
    """Weights of the steps in the distance (emphasis: the step grouping the map)."""
    weights = {}
    for step in schema["steps"]:
        weight = MODEL_WEIGHT if step == "model" else DEFAULT_WEIGHT
        if emphasis is not None and step == emphasis:
            weight = EMPHASIS_WEIGHT
        # a step with a single method does not separate settings
        if len(schema["methods"][step]) <= 1 and not any(
            schema["methods"][step].values()
        ):
            weight = 0.0
        weights[step] = weight
    return weights


class EncodedConfigs:
    """Settings as arrays: method index per step, scaled hyperparameters per method."""

    def __init__(
        self,
        methods: Dict[str, np.ndarray],
        values: Dict[Tuple[str, str], np.ndarray],
    ) -> None:
        self.methods = methods
        self.values = values

    def __len__(self) -> int:
        return len(next(iter(self.methods.values()))) if self.methods else 0


def encode(schema: Schema, configs: Sequence[Config]) -> EncodedConfigs:
    """
    Encode settings: unknown methods get index -1 (differ from every method),
    missing hyperparameters NaN (left out of the difference).
    """
    n = len(configs)
    methods: Dict[str, np.ndarray] = {}
    values: Dict[Tuple[str, str], np.ndarray] = {}

    for step in schema["steps"]:
        names = list(schema["methods"][step])
        index = {name: order for order, name in enumerate(names)}
        step_methods = np.full(n, -1, dtype=int)
        for name in names:
            params = schema["methods"][step][name]
            if params:
                values[(step, name)] = np.full((n, len(params)), np.nan)

        for row, config in enumerate(configs):
            setting = (config or {}).get(step) or {}
            name = setting.get("method")
            if name is None or str(name) not in index:
                continue
            name = str(name)
            step_methods[row] = index[name]
            params = schema["methods"][step][name]
            hyperparameters = setting.get("hyperparameters") or {}
            for col, (param, spec) in enumerate(params.items()):
                if param in hyperparameters:
                    values[(step, name)][row, col] = _scale(
                        spec, hyperparameters[param]
                    )

        methods[step] = step_methods

    return EncodedConfigs(methods, values)


def _scale(spec: Dict[str, Any], value: Any) -> float:
    """Hyperparameter value in [0, 1] (category index for categorical)."""
    if spec["type"] == "categorical":
        labels = [str(item) for item in spec["categories"]]
        try:
            return float(labels.index(str(_label(value))))
        except ValueError:
            # unknown category: differs from the known ones and from other unknown
            # values (stable code from the value)
            return -1.0 - float(zlib.crc32(str(value).encode("utf-8")) % 1_000_000)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    if not math.isfinite(value):
        return np.nan
    low, high = spec["low"], spec["high"]
    if spec["log"]:
        if value <= 0:
            return 0.0
        scaled = (math.log(value) - math.log(low)) / (math.log(high) - math.log(low))
    else:
        scaled = (value - low) / (high - low)
    return float(min(1.0, max(0.0, scaled)))


def distances(
    schema: Schema,
    left: EncodedConfigs,
    right: EncodedConfigs,
    weights: Dict[str, float],
) -> np.ndarray:
    """Distance matrix (len(left), len(right)) between two sets of settings."""
    n_left, n_right = len(left), len(right)
    total = np.zeros((n_left, n_right))
    weight_sum = sum(weights.values())
    if weight_sum <= 0 or n_left == 0 or n_right == 0:
        return total

    for step in schema["steps"]:
        weight = weights.get(step, 0.0)
        if weight <= 0:
            continue
        m_left, m_right = left.methods[step], right.methods[step]
        step_distance = np.ones((n_left, n_right))

        for order, name in enumerate(schema["methods"][step]):
            rows = np.flatnonzero(m_left == order)
            cols = np.flatnonzero(m_right == order)
            if len(rows) == 0 or len(cols) == 0:
                continue
            within = np.zeros((len(rows), len(cols)))
            key = (step, name)
            if key in left.values:
                within = _within_distance(
                    schema["methods"][step][name],
                    left.values[key][rows],
                    right.values[key][cols],
                )
            step_distance[np.ix_(rows, cols)] = SAME_METHOD_SCALE * within

        total += weight * step_distance

    return total / weight_sum


def _within_distance(
    params: Dict[str, Dict[str, Any]], left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    """Mean difference of the hyperparameters of the same method."""
    summed = np.zeros((left.shape[0], right.shape[0]))
    count = np.zeros_like(summed)
    for col, spec in enumerate(params.values()):
        a, b = left[:, col][:, None], right[:, col][None, :]
        if spec["type"] == "categorical":
            diff = (a != b).astype(float)
        else:
            diff = np.abs(a - b)
        valid = ~(np.isnan(a) | np.isnan(b))
        summed += np.where(valid, diff, 0.0)
        count += valid
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, summed / np.maximum(count, 1), 0.0)
