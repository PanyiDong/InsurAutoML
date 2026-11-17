"""
Ray Tune callback recording the trials of an optimization run for the dashboard.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from ray.tune import Callback

from .events import ERROR, EventRecorder

logger = logging.getLogger(__name__)

# keep the error message short in the event log
MAX_ERROR_LENGTH = 4000


def standardize_config(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Settings of a trial as {step: {"method": ..., "hyperparameters": {...}}}.

    The search algorithms store the settings differently (e.g. HyperOpt prefixes
    the hyperparameters with the method name); they are read the same way the
    objective reads them before fitting.
    """
    from ..hpo.utils import TabularObjective

    standardized: Dict[str, Dict[str, Any]] = {}
    for step, value in config.items():
        if not isinstance(value, dict):
            # constants, e.g. task_type
            continue
        try:
            method, hyperparameters = TabularObjective._extract_hyper(config, step)
        except Exception:
            method, hyperparameters = None, dict(value)
        standardized[step] = {
            "method": method,
            "hyperparameters": dict(hyperparameters),
        }

    return standardized


def _trial_error(trial: Any) -> Optional[str]:
    """Error message of a failed trial (end of the error file)."""
    error_file = getattr(trial, "error_file", None)
    if error_file and os.path.isfile(error_file):
        try:
            with open(error_file, "r", encoding="utf-8", errors="replace") as inp_f:
                return inp_f.read()[-MAX_ERROR_LENGTH:]
        except OSError:
            pass
    error_msg = getattr(trial, "error_msg", None)
    return str(error_msg)[-MAX_ERROR_LENGTH:] if error_msg else None


class DashboardCallback(Callback):
    """
    Record the trials of one Ray Tune experiment (one ensemble round).

    Runs in the driver; failures are logged and never interrupt the optimization.
    """

    def __init__(self, recorder: EventRecorder, round_index: int) -> None:
        self.recorder = recorder
        self.round_index = round_index

    def _emit(self, event_type: str, trial: Any, **fields: Any) -> None:
        try:
            self.recorder.emit(
                event_type,
                round=self.round_index,
                trial_id=str(trial.trial_id),
                **fields,
            )
        except Exception as err:  # pragma: no cover - defensive
            logger.warning("Dashboard: unable to record {}: {}".format(event_type, err))

    def on_trial_start(
        self, iteration: int, trials: List[Any], trial: Any, **info: Any
    ) -> None:
        try:
            config = standardize_config(dict(trial.config or {}))
        except Exception:
            config = {}
        self._emit("trial_start", trial, config=config)

    def on_trial_result(
        self,
        iteration: int,
        trials: List[Any],
        trial: Any,
        result: Dict,
        **info: Any,
    ) -> None:
        self._emit(
            "trial_result",
            trial,
            loss=result.get("loss"),
            training_status=result.get("training_status"),
            iteration=result.get("training_iteration"),
        )

    def on_trial_complete(
        self, iteration: int, trials: List[Any], trial: Any, **info: Any
    ) -> None:
        self._emit("trial_end", trial, status="COMPLETED")

    def on_trial_error(
        self, iteration: int, trials: List[Any], trial: Any, **info: Any
    ) -> None:
        self._emit("trial_end", trial, status=ERROR, error=_trial_error(trial))
