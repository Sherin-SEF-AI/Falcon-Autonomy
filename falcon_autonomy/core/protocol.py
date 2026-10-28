"""The local protocol between the GUI (or CLI) and a worker process (standard library only; docs/protocol.md).

A worker is `python -m falcon_autonomy.workers <name> [args]` in CARLA's Python. It speaks JSON lines:

worker -> caller, on stdout, one JSON object per line, each with a type "t":

    {"t": "hello",    "worker": "drive", "protocol": 1, "app_version": "0.1.0", "pid": 1234}
    {"t": "progress", "f": 0.42, "text": "frame 120 of 300", "eta_s": 31.5}
    {"t": "log",      "level": "info" | "warning" | "error", "text": "..."}
    {"t": "metric",   "name": "frames_per_s", "value": 18.2, "unit": "1/s", "source": "measured"}
    {"t": "frame",    "stream": "rgb", "frame": 1203, "sim_t": 12.03, "path": "/.../000120.jpg"}
    {"t": "result",   "status": "ok", "run_json": "/.../run.json", ...}

  Anything on stdout that is not such a line is shown as log text; stderr is kept as the worker's log.

caller -> worker, on stdin, one JSON object per line with a command "cmd":

    {"cmd": "cancel"}                           tidy up and exit with CANCELLED (as SIGTERM does)
    {"cmd": "control", "throttle": .., ...}     live sessions (falcon_autonomy.workers.live)
    {"cmd": "event", "kind": "...", ...}        live events

Live video does not go through stdout: a live worker serves frames on a local socket (docs/protocol.md, "frames").

Exit codes (every worker; run.json "status" and "exit_code" say the same):
"""
from __future__ import annotations

import json
import sys
import threading
import time

PROTOCOL_VERSION = 1

OK = 0
CRASHED = 1          # an unexpected error (a bug): the traceback is in the worker's log and run.json
FAILED = 2           # bad input, or the run failed for a reason stated in run.json "error"
CANCELLED = 3        # a cancel (stdin "cancel", SIGINT or SIGTERM); what was written is kept or removed as documented
SERVER_LOST = 4      # the CARLA server stopped answering during the run (the watchdog ended the worker)
NO_SERVER = 5        # no CARLA server answered at the start
REFUSED = 6          # refused before starting: over the render budget, not enough memory, a port not ours
NO_CARLA = 7         # the carla module (or numpy) could not be imported, or its version is not the server's

EXIT_CODES = {
    OK: ('ok', 'Finished.'),
    CRASHED: ('crashed', 'An unexpected error (a bug); the traceback is in the log and run.json.'),
    FAILED: ('failed', 'The run failed; run.json "error" says why.'),
    CANCELLED: ('cancelled', 'Cancelled by the user.'),
    SERVER_LOST: ('server_lost', 'The CARLA server stopped answering; the watchdog ended the run.'),
    NO_SERVER: ('no_server', 'No CARLA server answered at the start.'),
    REFUSED: ('refused', 'Refused before starting (render budget, memory or port).'),
    NO_CARLA: ('no_carla', 'CARLA\'s Python module could not be imported, or does not match the server.'),
}

__doc__ += '\n'.join(f'    {code}  {name:12s} {text}' for code, (name, text) in EXIT_CODES.items()) + '\n'


def status_of(code: int) -> str:
    return EXIT_CODES.get(code, ('crashed', ''))[0]


def describe_exit(code) -> str:
    if code is None:
        return 'still running'
    if code < 0:
        return f'killed by signal {-code}'
    name, text = EXIT_CODES.get(code, ('crashed', f'Unknown exit code {code}.'))
    return text


def parse_line(line: str) -> dict:
    """A worker's stdout line as a message: a JSON object with "t", else {"t": "log", "level": "info", "text": line}."""
    s = line.rstrip('\r\n')
    if s.startswith('{'):
        try:
            d = json.loads(s)
        except ValueError:
            d = None
        if isinstance(d, dict) and isinstance(d.get('t'), str):
            return d
    return {'t': 'log', 'level': 'info', 'text': s}


class Emitter:
    """The worker's side: writes protocol lines to stdout (thread-safe; progress rate-limited)."""

    def __init__(self, stream=None, min_interval=0.25):
        self.stream = stream or sys.stdout
        self.min_interval = float(min_interval)
        self._lock = threading.Lock()
        self._last_progress = 0.0
        self._t0 = time.monotonic()
        self._first_f = None

    def send(self, msg: dict):
        line = json.dumps(msg, separators=(',', ':'), default=_jsonable)
        with self._lock:
            self.stream.write(line + '\n')
            self.stream.flush()

    def hello(self, worker: str, **extra):
        from .. import __version__
        import os
        self.send({'t': 'hello', 'worker': worker, 'protocol': PROTOCOL_VERSION, 'app_version': __version__,
                   'pid': os.getpid(), **extra})

    def progress(self, f: float, text: str = '', force=False):
        """f in [0, 1]. The ETA is extrapolated from the progress so far (wall clock), once there is some."""
        now = time.monotonic()
        if not force and f < 1.0 and now - self._last_progress < self.min_interval:
            return
        self._last_progress = now
        f = min(max(float(f), 0.0), 1.0)
        if self._first_f is None and f > 0:
            self._first_f = (f, now)
        eta = None
        if self._first_f is not None and f > self._first_f[0] + 1e-6:
            f0, t0 = self._first_f
            eta = (now - t0) / (f - f0) * (1.0 - f)
        self.send({'t': 'progress', 'f': round(f, 4), 'text': text,
                   'eta_s': None if eta is None else round(eta, 1)})

    def log(self, text: str, level='info'):
        self.send({'t': 'log', 'level': level, 'text': str(text)})

    def metric(self, name: str, value, unit: str = '', source='measured', **extra):
        self.send({'t': 'metric', 'name': name, 'value': value, 'unit': unit, 'source': source, **extra})

    def result(self, status: str, **extra):
        self.send({'t': 'result', 'status': status, **extra})


class CommandReader(threading.Thread):
    """The worker's side of stdin: a daemon thread calling on_command(dict) per command line; EOF is not a cancel
    (a CLI caller may close stdin), a malformed line is ignored."""

    def __init__(self, on_command, stream=None):
        super().__init__(daemon=True, name='falcon-commands')
        self.on_command = on_command
        self.stream = stream or sys.stdin

    def run(self):
        try:
            for line in self.stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if isinstance(d, dict) and 'cmd' in d:
                    self.on_command(d)
        except (OSError, ValueError):
            return


def _jsonable(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)
