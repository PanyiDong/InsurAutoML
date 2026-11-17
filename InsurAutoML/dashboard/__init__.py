"""
File Name: __init__.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Actuarial and Risk Management Sciences, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.9
Relative Path: /InsurAutoML/dashboard/__init__.py
File Created: Monday, 16th November 2026 7:42:03 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 16th November 2026 11:33:08 pm
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
