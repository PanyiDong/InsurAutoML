"""
Map of the search space.

Settings sampled from the whole space (reference settings) are laid out once in
2D by t-SNE on the distance of the settings. A trial is placed at the weighted
position of its nearest reference settings, so the map does not move while the
optimization runs. The performance of the trials is spread to the reference
settings around them on the map (within the area of the same method): the map
shows which areas of the space perform well, and fades where no trial has been
run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .space import (
    Config,
    EncodedConfigs,
    Schema,
    distances,
    encode,
    sample_configs,
    step_weights,
)

logger = logging.getLogger(__name__)

N_REFERENCES = 1500
SEED = 0
# number of nearest reference settings placing a trial
N_NEIGHBORS = 10
# bandwidth (on the map, scaled to [0, 1]) spreading the performance of a trial
# to the settings around it (of the same group)
BANDWIDTH = 0.035


def trial_scores(trials: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """
    Performance of the fitted trials as rank percentile of the loss in [0, 1],
    1 for the best (lowest loss), 0 for the worst.
    """
    fitted = [
        trial
        for trial in trials
        if trial.get("training_status") == "FITTED" and trial.get("loss") is not None
    ]
    if not fitted:
        return {}
    losses = np.array([trial["loss"] for trial in fitted], dtype=float)
    # average rank of ties
    order = losses.argsort(kind="stable")
    ranks = np.empty(len(losses))
    ranks[order] = np.arange(len(losses))
    for value in np.unique(losses):
        tied = losses == value
        ranks[tied] = ranks[tied].mean()
    if len(losses) == 1:
        return {fitted[0]["key"]: 1.0}
    scores = 1.0 - ranks / (len(losses) - 1)
    return {trial["key"]: float(score) for trial, score in zip(fitted, scores)}


class SpaceMap:
    """Layout of the space for one emphasis (step weighting)."""

    def __init__(
        self,
        schema: Schema,
        emphasis: Optional[str] = None,
        n_references: int = N_REFERENCES,
        seed: int = SEED,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.schema = schema
        self.emphasis = emphasis
        self.weights = step_weights(schema, emphasis)
        self.references: List[Config] = sample_configs(schema, n_references, seed)
        self.encoded: EncodedConfigs = encode(schema, self.references)
        self.coords: Optional[np.ndarray] = None
        self.cache_path = (
            os.path.join(cache_dir, "map_{}.json".format(self._key(n_references, seed)))
            if cache_dir
            else None
        )

    def _key(self, n_references: int, seed: int) -> str:
        content = json.dumps(
            [self.schema, self.emphasis, n_references, seed], sort_keys=True
        )
        return hashlib.sha1(content.encode("utf-8")).hexdigest()[:16]

    # ---------------------------------------------------------------- layout

    def build(self) -> "SpaceMap":
        """Lay out the reference settings (t-SNE), cached on disk."""
        if self._load():
            return self

        n = len(self.references)
        if sum(self.weights.values()) <= 0 or n < 5:
            # nothing to separate: a single point
            self.coords = np.zeros((n, 2))
        else:
            from sklearn.manifold import TSNE

            matrix = distances(self.schema, self.encoded, self.encoded, self.weights)
            np.fill_diagonal(matrix, 0.0)
            tsne = TSNE(
                n_components=2,
                metric="precomputed",
                init="random",
                perplexity=float(min(30.0, (n - 1) / 3.0)),
                random_state=SEED,
                n_jobs=1,
            )
            coords = tsne.fit_transform(matrix)
            # scale to [0, 1]
            span = coords.max(axis=0) - coords.min(axis=0)
            span[span == 0] = 1.0
            self.coords = (coords - coords.min(axis=0)) / span
        self._save()

        return self

    def _load(self) -> bool:
        if not self.cache_path or not os.path.isfile(self.cache_path):
            return False
        try:
            with open(self.cache_path, "r", encoding="utf-8") as inp_f:
                coords = np.asarray(json.load(inp_f)["coords"], dtype=float)
        except (OSError, ValueError, KeyError):
            return False
        if coords.shape != (len(self.references), 2):
            return False
        self.coords = coords
        return True

    def _save(self) -> None:
        if not self.cache_path or self.coords is None:
            return
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as out_f:
                json.dump({"coords": np.round(self.coords, 5).tolist()}, out_f)
        except OSError as err:
            logger.warning("Dashboard: unable to cache the map: {}".format(err))

    @property
    def group_step(self) -> str:
        """Step grouping the map (its methods form the areas of the map)."""
        if self.emphasis in self.schema["steps"]:
            return str(self.emphasis)
        return "model" if "model" in self.schema["steps"] else self.schema["steps"][-1]

    def labels(self) -> List[Dict[str, Any]]:
        """Method names of the grouping step, above their area."""
        if self.coords is None:
            return []
        step = self.group_step
        names = list(self.schema["methods"][step])
        result = []
        for order, name in enumerate(names):
            mask = self.encoded.methods[step] == order
            if not mask.any():
                continue
            # above the area: the trials in it stay visible
            points = self.coords[mask]
            result.append(
                {
                    "text": name,
                    "x": float(np.median(points[:, 0])),
                    "y": float(np.quantile(points[:, 1], 0.95) + 0.025),
                }
            )
        return result

    def layout(self) -> Dict[str, Any]:
        """Positions of the reference settings (static)."""
        assert self.coords is not None
        return {
            "emphasis": self.emphasis,
            "steps": self.schema["steps"],
            "x": np.round(self.coords[:, 0], 4).tolist(),
            "y": np.round(self.coords[:, 1], 4).tolist(),
            "labels": self.labels(),
        }

    # ---------------------------------------------------------------- trials

    def place(
        self, configs: Sequence[Config], keys: Sequence[str]
    ) -> Dict[str, List[float]]:
        """
        Positions of trials on the map: weighted position of the nearest
        reference settings with the same method of the grouping step (the
        trial is in the area of its method).
        """
        assert self.coords is not None
        if not configs:
            return {}
        encoded = encode(self.schema, configs)
        matrix = distances(self.schema, encoded, self.encoded, self.weights)
        step = self.group_step
        group, ref_group = encoded.methods[step], self.encoded.methods[step]
        positions = {}
        for row, key in enumerate(keys):
            candidates = np.flatnonzero(ref_group == group[row])
            if group[row] < 0 or len(candidates) == 0:
                candidates = np.arange(len(ref_group))
            row_distance = matrix[row, candidates]
            k = min(N_NEIGHBORS, len(candidates))
            nearest = np.argpartition(row_distance, k - 1)[:k]
            weight = 1.0 / (row_distance[nearest] + 0.02)
            position = (self.coords[candidates[nearest]] * weight[:, None]).sum(
                axis=0
            ) / weight.sum()
            # small fixed offset, trials with the same settings stay visible
            seed = int(hashlib.md5(str(key).encode("utf-8")).hexdigest()[:8], 16)
            position = position + np.random.default_rng(seed).normal(0, 0.004, 2)
            positions[key] = [
                round(float(position[0]), 4),
                round(float(position[1]), 4),
            ]
        return positions

    def field(
        self,
        configs: Sequence[Config],
        positions: Sequence[Sequence[float]],
        scores: Sequence[float],
    ) -> Dict[str, List[Optional[float]]]:
        """
        Estimated performance of the reference settings from the scored trials
        around them on the map (Gaussian kernel, trials of the same group only),
        and the confidence (closeness of the nearest trial, in [0, 1]).
        """
        assert self.coords is not None
        n = len(self.references)
        if not configs:
            return {"score": [None] * n, "confidence": [0.0] * n}
        step = self.group_step
        group = encode(self.schema, configs).methods[step]
        ref_group = self.encoded.methods[step]
        points = np.asarray(positions, dtype=float)
        squared = ((self.coords[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)
        kernel = np.exp(-squared / (2 * BANDWIDTH**2))
        kernel *= ref_group[:, None] == group[None, :]
        total = kernel.sum(axis=1)
        values = np.asarray(scores, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            estimate = (kernel * values[None, :]).sum(axis=1) / total
        confidence = kernel.max(axis=1)
        return {
            "score": [
                round(float(value), 3) if conf > 1e-3 else None
                for value, conf in zip(estimate, confidence)
            ],
            "confidence": np.round(confidence, 3).tolist(),
        }


class MapBuilder:
    """
    Build the maps in a background thread (one per emphasis) and serve them.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        n_references: int = N_REFERENCES,
    ) -> None:
        self.cache_dir = cache_dir
        self.n_references = n_references
        self.schema: Optional[Schema] = None
        self._maps: Dict[Optional[str], SpaceMap] = {}
        self._status: Dict[Optional[str], str] = {}
        self._errors: Dict[Optional[str], str] = {}
        self._lock = threading.Lock()

    def set_schema(self, schema: Optional[Schema]) -> None:
        with self._lock:
            if schema == self.schema:
                return
            self.schema = schema
            self._maps.clear()
            self._status.clear()
            self._errors.clear()

    def _emphasis(self, emphasis: Optional[str]) -> Optional[str]:
        if self.schema is None or emphasis not in self.schema["steps"]:
            return None
        return None if emphasis == "model" else emphasis

    def status(self, emphasis: Optional[str] = None) -> Dict[str, Any]:
        """Status of the map, starts building it if needed."""
        emphasis = self._emphasis(emphasis)
        with self._lock:
            if self.schema is None:
                return {"status": "waiting"}
            status = self._status.get(emphasis)
            if status is None:
                self._status[emphasis] = "building"
                schema = self.schema
                threading.Thread(
                    target=self._build, args=(schema, emphasis), daemon=True
                ).start()
                status = "building"
            return {"status": status, "error": self._errors.get(emphasis)}

    def build(self, emphasis: Optional[str] = None) -> Optional[SpaceMap]:
        """Build the map in the calling thread (replay, tests)."""
        emphasis = self._emphasis(emphasis)
        if self.schema is None:
            return None
        with self._lock:
            self._status[emphasis] = "building"
        self._build(self.schema, emphasis)
        return self.get(emphasis)

    def _build(self, schema: Schema, emphasis: Optional[str]) -> None:
        try:
            space_map = SpaceMap(
                schema,
                emphasis,
                n_references=self.n_references,
                cache_dir=self.cache_dir,
            ).build()
        except Exception as err:
            logger.warning("Dashboard: unable to build the map: {}".format(err))
            with self._lock:
                if schema == self.schema:
                    self._status[emphasis] = "unavailable"
                    self._errors[emphasis] = str(err)
            return
        with self._lock:
            if schema == self.schema:
                self._maps[emphasis] = space_map
                self._status[emphasis] = "ready"

    def get(self, emphasis: Optional[str] = None) -> Optional[SpaceMap]:
        with self._lock:
            return self._maps.get(self._emphasis(emphasis))

    def scores(
        self, trials: Sequence[Dict[str, Any]], emphasis: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Positions of the trials and the estimated performance of the space."""
        space_map = self.get(emphasis)
        if space_map is None:
            return None
        trials = [trial for trial in trials if trial.get("config")]
        scored = trial_scores(trials)
        positions = space_map.place(
            [trial["config"] for trial in trials], [trial["key"] for trial in trials]
        )
        scored_trials = [trial for trial in trials if trial["key"] in scored]
        field = space_map.field(
            [trial["config"] for trial in scored_trials],
            [positions[trial["key"]] for trial in scored_trials],
            [scored[trial["key"]] for trial in scored_trials],
        )
        return {"positions": positions, "scores": scored, **field}
