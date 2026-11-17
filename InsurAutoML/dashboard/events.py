"""
Event log of an optimization run.

The dashboard is driven by an append-only JSON lines file, one event per line:

    run_start     settings of the run (search algorithm, timeout, max_evals, ...)
    round_start   one Ray Tune experiment (ensemble round), with its trial budget
    trial_start   a trial and its (standardized) settings
    trial_result  a result reported by a trial (loss, training status)
    trial_end     a trial finished (completed or errored)
    round_end     the Ray Tune experiment finished
    run_end       the fit finished

The log is the source of truth: the live dashboard and the replay of a finished
(or crashed) run are both built by folding the events into a DashboardState.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import threading
import time
import warnings
from typing import Any, Callable, Dict, List, Optional

import numpy as np

EVENTS_DIR = "dashboard"
EVENTS_FILE = "events.jsonl"

# trial status in the dashboard
RUNNING = "RUNNING"
FITTED = "FITTED"
TIMEOUT = "TIMEOUT"
ERROR = "ERROR"
# finished without any reported result
TERMINATED = "TERMINATED"


def to_json(value: Any) -> Any:
    """
    Convert a value to plain JSON types. Non-finite floats become None (not valid
    JSON for the browser), unknown objects become their string representation.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [to_json(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json(item) for item in value]
    return str(value)


class EventRecorder:
    """
    Append events to the event log (thread safe, picklable).

    Ray Tune stores the callbacks (holding the recorder) with the experiment for
    restoring, so a resumed experiment keeps writing to the same log.
    """

    def __init__(self, directory: str, clock: Callable[[], float] = time.time) -> None:
        self.directory = os.path.abspath(directory)
        self.path = os.path.join(self.directory, EVENTS_FILE)
        self.clock = clock
        self._lock = threading.Lock()
        self._failed = False

        os.makedirs(self.directory, exist_ok=True)

    def emit(self, event_type: str, **fields: Any) -> Dict[str, Any]:
        event = to_json({"type": event_type, "time": self.clock(), **fields})
        line = json.dumps(event) + "\n"

        with self._lock:
            try:
                # open per event: no file handle is kept (Windows, pickling)
                with open(self.path, "a", encoding="utf-8") as out_f:
                    out_f.write(line)
            except OSError as err:
                # never fail the optimization because of the dashboard
                if not self._failed:
                    warnings.warn(
                        "Unable to write dashboard events to {}: {}".format(
                            self.path, err
                        )
                    )
                self._failed = True

        return event

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()


def read_events(path: str, offset: int = 0) -> tuple:
    """
    Read the events of the log from byte offset, returns (events, new offset).
    A partially written last line is left for the next read.
    """
    events: List[Dict[str, Any]] = []
    if not os.path.isfile(path):
        return events, offset

    with open(path, "rb") as inp_f:
        inp_f.seek(offset)
        data = inp_f.read()

    # only complete lines
    end = data.rfind(b"\n") + 1
    for line in data[:end].splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line.decode("utf-8")))
        except (ValueError, UnicodeDecodeError):
            # skip corrupted lines
            continue

    return events, offset + end


def load_state(path: str) -> "DashboardState":
    """
    Build the state of a run from its event log (file or the directory holding it).
    """
    if os.path.isdir(path):
        path = os.path.join(path, EVENTS_FILE)
    state = DashboardState()
    events, _ = read_events(path)
    for event in events:
        state.apply(event)

    return state


class DashboardState:
    """
    State of an optimization run, folded from its events (thread safe).
    """

    def __init__(self) -> None:
        self.run: Dict[str, Any] = {}
        # search space of the map (see space.space_schema)
        self.schema: Optional[Dict[str, Any]] = None
        self.rounds: List[Dict[str, Any]] = []
        self.trials: Dict[str, Dict[str, Any]] = {}
        self.finished = False
        self.end_time: Optional[float] = None
        self.last_time: Optional[float] = None
        self.peak_running = 0
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ events

    def apply(self, event: Dict[str, Any]) -> None:
        handler = getattr(self, "_on_" + str(event.get("type")), None)
        if handler is None:
            return
        with self._lock:
            self.last_time = event.get("time", self.last_time)
            handler(event)

    def _on_run_start(self, event: Dict[str, Any]) -> None:
        # a resumed run appends a new run_start: keep the original start time
        start = self.run.get("start", event.get("time"))
        self.run = {
            key: value for key, value in event.items() if key not in ("type", "time")
        }
        self.run["start"] = start
        self.finished = False
        self.end_time = None

    def _on_space(self, event: Dict[str, Any]) -> None:
        self.schema = event.get("schema")

    def _on_round_start(self, event: Dict[str, Any]) -> None:
        index = event.get("round", len(self.rounds))
        current = self._round(index)
        if current is None:
            self.rounds.append(
                {
                    "round": index,
                    "name": event.get("name"),
                    "num_samples": event.get("num_samples"),
                    "start": event.get("time"),
                    "end": None,
                    "duration": 0.0,
                }
            )
        else:
            # resumed round: time already spent is kept in duration
            current["start"] = event.get("time")
            current["end"] = None

    def _on_round_end(self, event: Dict[str, Any]) -> None:
        # trials stopped with the experiment (e.g. time budget reached) are not
        # reported as completed by Ray Tune
        for trial in self._running():
            if trial["round"] == event.get("round"):
                self._on_trial_end(
                    {
                        "time": event.get("time"),
                        "round": trial["round"],
                        "trial_id": trial["trial_id"],
                        "status": "STOPPED",
                    }
                )

        current = self._round(event.get("round"))
        if current is None or current["end"] is not None:
            return
        current["end"] = event.get("time")
        if current["start"] is not None and current["end"] is not None:
            current["duration"] += current["end"] - current["start"]

    def _on_trial_start(self, event: Dict[str, Any]) -> None:
        key = self._key(event)
        trial = self.trials.get(key)
        if trial is None:
            trial = {
                "key": key,
                "number": len(self.trials) + 1,
                "trial_id": event.get("trial_id"),
                "round": event.get("round"),
                "config": event.get("config") or {},
                "status": RUNNING,
                "training_status": None,
                "loss": None,
                "iterations": 0,
                "start": event.get("time"),
                "end": None,
                "duration": None,
                "error": None,
            }
            self.trials[key] = trial
        else:
            # restored trial
            trial["status"] = RUNNING
            trial["end"] = None
        self.peak_running = max(self.peak_running, len(self._running()))

    def _on_trial_result(self, event: Dict[str, Any]) -> None:
        trial = self.trials.get(self._key(event))
        if trial is None:
            return
        trial["training_status"] = event.get("training_status")
        trial["loss"] = event.get("loss")
        trial["iterations"] = event.get("iteration") or trial["iterations"] + 1

    def _on_trial_end(self, event: Dict[str, Any]) -> None:
        trial = self.trials.get(self._key(event))
        if trial is None:
            return
        if event.get("status") == ERROR:
            trial["status"] = ERROR
            trial["error"] = event.get("error")
        elif trial["training_status"] in (FITTED, TIMEOUT):
            trial["status"] = trial["training_status"]
        else:
            trial["status"] = TERMINATED
        trial["end"] = event.get("time")
        if trial["start"] is not None and trial["end"] is not None:
            trial["duration"] = trial["end"] - trial["start"]

    def _on_run_end(self, event: Dict[str, Any]) -> None:
        self.finished = True
        self.end_time = event.get("time")

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _key(event: Dict[str, Any]) -> str:
        return "{}:{}".format(event.get("round"), event.get("trial_id"))

    def _round(self, index: Any) -> Optional[Dict[str, Any]]:
        return next((item for item in self.rounds if item["round"] == index), None)

    def _running(self) -> List[Dict[str, Any]]:
        return [trial for trial in self.trials.values() if trial["status"] == RUNNING]

    def _now(self, now: Optional[float]) -> float:
        if self.finished and self.end_time is not None:
            return self.end_time
        return time.time() if now is None else now

    # --------------------------------------------------------------- summaries

    def best_trial(self) -> Optional[Dict[str, Any]]:
        """Fitted trial with the lowest loss."""
        with self._lock:
            fitted = [
                trial
                for trial in self.trials.values()
                if trial["training_status"] == FITTED and trial["loss"] is not None
            ]
            if not fitted:
                return None
            return dict(min(fitted, key=lambda trial: trial["loss"]))

    def progress(self, now: Optional[float] = None) -> Dict[str, Any]:
        """
        Progress of the run and the estimated remaining time.

        The time budget (timeout) is shared by all rounds (the stopper is reused
        across the ensemble rounds); the remaining time is the smaller of the
        remaining time budget and the time to finish the remaining trials at the
        observed pace.
        """
        with self._lock:
            now = self._now(now)
            counts = {status: 0 for status in (RUNNING, FITTED, TIMEOUT, ERROR)}
            counts[TERMINATED] = 0
            for trial in self.trials.values():
                counts[trial["status"]] = counts.get(trial["status"], 0) + 1
            finished = len(self.trials) - counts[RUNNING]

            # time spent in the tuning rounds
            elapsed = 0.0
            for item in self.rounds:
                elapsed += item["duration"]
                if item["end"] is None and item["start"] is not None:
                    elapsed += max(0.0, now - item["start"])
            wall = max(0.0, now - self.run["start"]) if "start" in self.run else 0.0

            planned_rounds = max(int(self.run.get("rounds") or 1), len(self.rounds), 1)
            current_round = len(self.rounds)
            num_samples = [
                item["num_samples"]
                for item in self.rounds
                if item["num_samples"] is not None and item["num_samples"] > 0
            ]
            if num_samples:
                # rounds not started yet are assumed to have the same budget
                total_trials: Optional[int] = sum(num_samples) + num_samples[-1] * (
                    planned_rounds - len(num_samples)
                )
            else:
                max_evals = self.run.get("max_evals")
                total_trials = max_evals if max_evals and max_evals > 0 else None

            timeout = self.run.get("timeout")
            time_left = (
                max(0.0, float(timeout) - elapsed)
                if timeout is not None and timeout > 0
                else None
            )

            durations = [
                trial["duration"]
                for trial in self.trials.values()
                if trial["duration"] is not None
            ]
            trial_time_left = None
            if total_trials is not None and len(durations) >= 3:
                parallel = max(1, self.peak_running)
                median = statistics.median(durations)
                # trials not started yet, and the expected rest of the running ones
                queued = max(0, total_trials - len(self.trials))
                running_left = max(
                    [
                        max(0.0, median - (now - trial["start"]))
                        for trial in self._running()
                        if trial["start"] is not None
                    ]
                    or [0.0]
                )
                trial_time_left = queued * median / parallel + running_left

            if self.finished:
                eta: Optional[float] = 0.0
            else:
                candidates = [
                    value for value in (time_left, trial_time_left) if value is not None
                ]
                # the time budget alone is not an estimate before any trial ended
                eta = min(candidates) if trial_time_left is not None else None

            return {
                "finished": self.finished,
                "elapsed": elapsed,
                "wall_time": wall,
                "timeout": timeout,
                "time_left": time_left,
                "eta": eta,
                "round": current_round,
                "rounds": planned_rounds,
                "trials_total": total_trials,
                "trials_started": len(self.trials),
                "trials_finished": finished,
                "counts": counts,
            }

    def snapshot(self, now: Optional[float] = None) -> Dict[str, Any]:
        """Everything the dashboard page shows (plain JSON)."""
        with self._lock:
            return to_json(
                {
                    "run": self.run,
                    "progress": self.progress(now),
                    "best": self.best_trial(),
                    "running": [dict(trial) for trial in self._running()],
                    "rounds": [dict(item) for item in self.rounds],
                    "trials": [dict(trial) for trial in self.trials.values()],
                }
            )


class EventTail:
    """
    Follow an event log: new events are folded into the state on refresh.

    The log of a new run replaces the previous one (other first line, or smaller
    file): the state is rebuilt. A deleted log (e.g. delete_temp_after_terminate)
    keeps the state.
    """

    def __init__(self, path: str) -> None:
        if os.path.isdir(path):
            path = os.path.join(path, EVENTS_FILE)
        self.path = os.path.abspath(path)
        self.state = DashboardState()
        self._offset = 0
        self._head: Optional[bytes] = None
        self._lock = threading.Lock()

    def _first_line(self) -> Optional[bytes]:
        try:
            with open(self.path, "rb") as inp_f:
                line = inp_f.readline()
        except OSError:
            return None
        return line if line.endswith(b"\n") else None

    def refresh(self) -> DashboardState:
        with self._lock:
            try:
                size = os.path.getsize(self.path)
            except OSError:
                return self.state
            head = self._first_line()
            if size < self._offset or (
                self._head is not None and head is not None and head != self._head
            ):
                self.state = DashboardState()
                self._offset = 0
            if head is not None:
                self._head = head
            events, self._offset = read_events(self.path, self._offset)
            for event in events:
                self.state.apply(event)
            return self.state
