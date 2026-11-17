"""
Dashboard of the hyperparameter optimization.

The trials of a run are recorded in an event log (events.jsonl, under
<temp_directory>/<model_name>/dashboard), which drives the dashboard.
"""

from .events import (
    EVENTS_DIR,
    EVENTS_FILE,
    DashboardState,
    EventRecorder,
    load_state,
    read_events,
)
from .callback import DashboardCallback, standardize_config
from .server import DashboardServer, start_dashboard, stop_dashboard

__all__ = [
    "EVENTS_DIR",
    "EVENTS_FILE",
    "DashboardState",
    "EventRecorder",
    "load_state",
    "read_events",
    "DashboardCallback",
    "standardize_config",
    "DashboardServer",
    "start_dashboard",
    "stop_dashboard",
]
