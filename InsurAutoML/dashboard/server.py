"""
File Name: server.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Actuarial and Risk Management Sciences, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.9
Relative Path: /InsurAutoML/dashboard/server.py
File Created: Monday, 16th November 2026 8:33:05 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 16th November 2026 11:33:46 pm
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

Local web server of the dashboard (Python standard library only).

The page (static/) polls a small JSON API:

    GET /api/state                  progress, best/running trials, all trials
    GET /api/map?emphasis=<step>    status of the map, and its layout when ready
    GET /api/map/scores?emphasis=   positions of the trials and the estimated
                                    performance of the space

One server runs per Python process; a new fit switches it to the new run.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple, cast
from urllib.parse import parse_qs, urlparse

from .events import EventTail
from .layout import MapBuilder

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
# ports tried after the requested one when it is in use
PORT_ATTEMPTS = 20

# settings through environment variables
ENV_HOST = "INSURAUTOML_DASHBOARD_HOST"
ENV_PORT = "INSURAUTOML_DASHBOARD_PORT"
# "0": record the events only, no web server
ENV_SERVE = "INSURAUTOML_DASHBOARD_SERVER"

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
STATIC_FILES = {
    "index.html": "text/html; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "style.css": "text/css; charset=utf-8",
}


def plotly_js_path() -> Optional[str]:
    """plotly.js bundled with the plotly package (served locally, works offline)."""
    spec = importlib.util.find_spec("plotly")
    if spec is None or not spec.submodule_search_locations:
        return None
    for location in spec.submodule_search_locations:
        path = os.path.join(location, "package_data", "plotly.min.js")
        if os.path.isfile(path):
            return path
    return None


class _HTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    # on Windows, SO_REUSEADDR allows binding a port in use by another process
    allow_reuse_address = sys.platform != "win32"
    dashboard: "DashboardServer"


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # no request log on the console
        pass

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        query = {key: values[-1] for key, values in parse_qs(url.query).items()}
        route = url.path.rstrip("/") or "/"
        dashboard = cast(_HTTPServer, self.server).dashboard
        try:
            if route in ("/", "/index.html"):
                self._file(
                    os.path.join(STATIC_DIR, "index.html"), STATIC_FILES["index.html"]
                )
            elif route == "/static/plotly.min.js":
                path = plotly_js_path()
                if path is None:
                    self._send(404, b"plotly is not installed", "text/plain")
                else:
                    self._file(path, "text/javascript; charset=utf-8", cache=True)
            elif route.startswith("/static/") and route[8:] in STATIC_FILES:
                name = route[8:]
                self._file(os.path.join(STATIC_DIR, name), STATIC_FILES[name])
            elif route == "/api/state":
                self._json(dashboard.state())
            elif route == "/api/map":
                self._json(dashboard.map(query.get("emphasis")))
            elif route == "/api/map/scores":
                self._json(dashboard.map_scores(query.get("emphasis")))
            else:
                self._send(404, b"not found", "text/plain")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as err:  # pragma: no cover - defensive
            logger.warning("Dashboard: error serving {}: {}".format(self.path, err))
            try:
                self._json({"error": str(err)}, status=500)
            except OSError:
                pass

    def _send(
        self, status: int, body: bytes, content_type: str, cache: bool = False
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: str, content_type: str, cache: bool = False) -> None:
        with open(path, "rb") as inp_f:
            self._send(200, inp_f.read(), content_type, cache)

    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self._send(status, body, "application/json")


class DashboardServer:
    """
    Dashboard of the run recorded in an event log (file, or directory holding it).
    """

    def __init__(
        self,
        path: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ) -> None:
        self.host = host or os.environ.get(ENV_HOST) or DEFAULT_HOST
        self.port = int(port or os.environ.get(ENV_PORT) or DEFAULT_PORT)
        self._lock = threading.Lock()
        self._httpd: Optional[_HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self.set_path(path)

    # ------------------------------------------------------------------ data

    def set_path(self, path: str) -> None:
        """Show the run recorded in another event log."""
        tail = EventTail(path)
        with self._lock:
            self._tail = tail
            self._maps = MapBuilder(cache_dir=os.path.dirname(tail.path))
            self._scores_cache: Tuple[Any, Any] = (None, None)

    @property
    def path(self) -> str:
        return self._tail.path

    def _refresh(self):
        with self._lock:
            tail, maps = self._tail, self._maps
        state = tail.refresh()
        maps.set_schema(state.schema)
        return state, maps

    def state(self) -> Dict[str, Any]:
        state, maps = self._refresh()
        snapshot = state.snapshot()
        snapshot["path"] = self.path
        # times of the server: elapsed times of running trials, stale runs
        snapshot["server_time"] = time.time()
        snapshot["last_time"] = state.last_time
        snapshot["map_steps"] = state.schema["steps"] if state.schema else []
        return snapshot

    def map(self, emphasis: Optional[str] = None) -> Dict[str, Any]:
        _, maps = self._refresh()
        status = maps.status(emphasis)
        space_map = maps.get(emphasis) if status["status"] == "ready" else None
        if space_map is not None:
            status["layout"] = space_map.layout()
        return status

    def map_scores(self, emphasis: Optional[str] = None) -> Dict[str, Any]:
        state, maps = self._refresh()
        trials = state.snapshot()["trials"]
        # unchanged trials: same scores
        key = (
            id(maps),
            emphasis,
            tuple((t["key"], t["status"], t["loss"]) for t in trials),
        )
        with self._lock:
            if self._scores_cache[0] == key:
                return self._scores_cache[1]
        scores = maps.scores(trials, emphasis)
        payload = {"status": "ready", **scores} if scores else maps.status(emphasis)
        if scores:
            with self._lock:
                self._scores_cache = (key, payload)
        return payload

    # ---------------------------------------------------------------- server

    @property
    def url(self) -> str:
        host = self.host
        if host in ("0.0.0.0", "::", ""):
            host = "127.0.0.1"
        if ":" in host:
            host = "[{}]".format(host)
        return "http://{}:{}".format(host, self.port)

    def _bind(self) -> _HTTPServer:
        last_error: Optional[OSError] = None
        for port in range(self.port, self.port + PORT_ATTEMPTS):
            try:
                httpd = _HTTPServer((self.host, port), _Handler)
            except OSError as err:
                last_error = err
                continue
            httpd.dashboard = self
            self.port = httpd.server_address[1]
            return httpd
        raise OSError(
            "No free port in {}-{}: {}".format(
                self.port, self.port + PORT_ATTEMPTS - 1, last_error
            )
        )

    def start(self) -> str:
        """Serve in a background (daemon) thread, returns the URL."""
        if self.running:
            return self.url
        self._httpd = self._bind()
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="InsurAutoML-dashboard",
            daemon=True,
        )
        self._thread.start()
        return self.url

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._httpd = None
        self._thread = None


# ------------------------------------------------------- server of the process

_ACTIVE: Optional[DashboardServer] = None
_ACTIVE_LOCK = threading.Lock()


def start_dashboard(path: str) -> Optional[str]:
    """
    Show the run recorded in the event log in the dashboard of this process
    (started if needed), returns its URL (None: disabled or not available).
    """
    global _ACTIVE
    if os.environ.get(ENV_SERVE, "1") == "0":
        return None
    with _ACTIVE_LOCK:
        if _ACTIVE is not None and _ACTIVE.running:
            _ACTIVE.set_path(path)
            return _ACTIVE.url
        try:
            server = DashboardServer(path)
            url = server.start()
        except (OSError, ValueError) as err:
            logger.warning("Dashboard: unable to start the web server: {}".format(err))
            return None
        _ACTIVE = server
        return url


def stop_dashboard() -> None:
    """Stop the dashboard of this process."""
    global _ACTIVE
    with _ACTIVE_LOCK:
        if _ACTIVE is not None:
            _ACTIVE.stop()
        _ACTIVE = None


def active_dashboard() -> Optional[DashboardServer]:
    return _ACTIVE
