"""
File Name: __main__.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Actuarial and Risk Management Sciences, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.9
Relative Path: /InsurAutoML/dashboard/__main__.py
File Created: Monday, 16th November 2026 8:33:06 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 16th November 2026 11:33:17 pm
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

Dashboard of a recorded run (finished, or running in another process):

    python -m InsurAutoML.dashboard tmp/<model_name> [--host HOST] [--port PORT] [--open]

The path is the run directory (<temp_directory>/<model_name>), its dashboard
directory, or the event log (events.jsonl).
"""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from typing import List, Optional

from .events import EVENTS_DIR, EVENTS_FILE
from .server import DashboardServer


def resolve_events(path: str) -> str:
    """Event log of a run directory, dashboard directory or event log."""
    candidates = [
        path,
        os.path.join(path, EVENTS_FILE),
        os.path.join(path, EVENTS_DIR, EVENTS_FILE),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(
        "No dashboard event log ({}) found in {}".format(EVENTS_FILE, path)
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m InsurAutoML.dashboard",
        description="Dashboard of a recorded InsurAutoML optimization run.",
    )
    parser.add_argument(
        "path", help="run directory (<temp_directory>/<model_name>) or event log"
    )
    parser.add_argument(
        "--host", default=None, help="address to listen on (default 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="port (default 8765, or the next free one)",
    )
    parser.add_argument(
        "--open", action="store_true", help="open the dashboard in the browser"
    )
    args = parser.parse_args(argv)

    try:
        events = resolve_events(args.path)
    except FileNotFoundError as err:
        print(err, file=sys.stderr)
        return 1

    server = DashboardServer(events, host=args.host, port=args.port)
    url = server.start()
    print("[Dashboard] {} at {} (Ctrl+C to stop)".format(events, url), flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        # short joins: Ctrl+C is handled on all platforms
        while server.running:
            server._thread.join(timeout=0.5)  # type: ignore[union-attr]
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
