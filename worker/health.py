"""Tiny HTTP server for Kubernetes probes and Prometheus scraping.

``/healthz``  200 while the process is alive
``/readyz``   200 once the model is loaded and the worker is accepting tasks
``/metrics``  Prometheus exposition
``/status``   JSON: model, busy, current task, counters
"""
from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

logger = logging.getLogger(__name__)


def serve(port: int, status_fn: Callable[[], dict], ready_fn: Callable[[], bool]) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path.startswith("/metrics"):
                body = generate_latest()
                self._send(200, body, CONTENT_TYPE_LATEST)
            elif self.path.startswith("/readyz"):
                ready = ready_fn()
                self._send(200 if ready else 503, json.dumps({"ready": ready}).encode(), "application/json")
            elif self.path.startswith("/healthz"):
                self._send(200, b'{"status":"ok"}', "application/json")
            elif self.path.startswith("/status"):
                self._send(200, json.dumps(status_fn()).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # silence per-probe access logs
            return

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    thread = threading.Thread(target=server.serve_forever, name="worker-http", daemon=True)
    thread.start()
    logger.info("worker http endpoints on :%d (/healthz /readyz /metrics /status)", port)
    return server
