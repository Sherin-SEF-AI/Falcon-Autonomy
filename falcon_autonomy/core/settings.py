"""User settings and secrets ($XDG_CONFIG_HOME/FalconAutonomy; standard library only).

settings.json holds machine-specific choices — where CARLA is, which ports Falcon's own server uses, the server's
memory cap, the theme, recent projects, measured throughput — and never a secret. secrets.json (mode 0600) holds
credentials (a Jetson's password, an API key the user configured); nothing reads it but the code that needs that one
secret, it is never shown on screen and never copied into a project or an output.

Ports: Falcon starts its own server on its own RPC port (DEFAULT_RPC_PORT; CARLA also takes the next two for its
streams) and traffic-manager port. It never touches a server it did not start, nor RESERVED_PORTS (CARLA's defaults,
where a user's own server usually listens), unless the user chose such a port in the app (allowed_ports).
"""
from __future__ import annotations

import copy
import json
import os

from .paths import atomic_write, config_dir

DEFAULT_RPC_PORT = 2310
DEFAULT_TM_PORT = 8310
RESERVED_PORTS = frozenset(list(range(2000, 2011)) + [8100])
SERVER_MEMORY_GB = 12.0
MEMORY_RESERVE_GB = 2.0          # left free for the desktop and the GUI whenever Falcon starts something heavy

DEFAULTS = {
    'carla_root': '',
    'client_python': '',
    'rpc_port': DEFAULT_RPC_PORT,
    'tm_port': DEFAULT_TM_PORT,
    'allowed_ports': [],                 # reserved ports the user explicitly chose in the app
    'server_memory_gb': SERVER_MEMORY_GB,
    'server_quality': 'Epic',
    'server_ros2': False,                # start Falcon's server with --ros2 (CARLA's native ROS 2; docs/stack-testing.md)
    'render_budget_mpx': 200.0,
    'theme': 'dark',
    'recent_projects': [],
    'first_run_done': False,
    'measured': {},                      # throughput measured on this machine (budget.Throughput.to_dict())
    'geocoding_allowed': False,          # network place-name search only when the user turns it on
    'cloud_llm': None,                   # never set unless the user configures and approves one
    'hil_device': {},                    # the HIL device: address, ports, SSH user and key path (core.hil); no password
}


class Settings:
    def __init__(self, path: str | None = None):
        self.path = path or os.path.join(config_dir(), 'settings.json')
        self.data = copy.deepcopy(DEFAULTS)
        self.load()

    def load(self):
        try:
            with open(self.path, encoding='utf-8') as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            return
        if isinstance(d, dict):
            for k, v in d.items():
                if k in DEFAULTS:
                    self.data[k] = v

    def save(self):
        atomic_write(self.path, json.dumps(self.data, indent=2) + '\n')

    def get(self, key, default=None):
        return self.data.get(key, DEFAULTS.get(key, default))

    def set(self, key, value, save=True):
        if key not in DEFAULTS:
            raise KeyError(f'Unknown setting {key!r}')
        self.data[key] = value
        if save:
            self.save()

    def add_recent(self, path: str, keep=10):
        r = [p for p in self.data.get('recent_projects', []) if p != path]
        self.data['recent_projects'] = [path] + r[:keep - 1]
        self.save()


def port_allowed(port: int, settings: 'Settings | None' = None) -> bool:
    """Whether Falcon may start a server on (or connect its traffic manager to) port: never a reserved one unless
    the user chose it in the app."""
    port = int(port)
    if not 1024 <= port <= 65533:
        return False
    if port in RESERVED_PORTS:
        return bool(settings) and port in set(settings.get('allowed_ports') or [])
    return True


def rpc_ports(port: int) -> tuple:
    """The ports a CARLA server on RPC port p listens on: p (RPC), p + 1 (streaming), p + 2 (secondary server)."""
    return int(port), int(port) + 1, int(port) + 2


class Secrets:
    """Credentials, kept apart from settings in a file only the user can read. Values are never logged or shown."""

    def __init__(self, path: str | None = None):
        self.path = path or os.path.join(config_dir(), 'secrets.json')

    def _read(self) -> dict:
        try:
            with open(self.path, encoding='utf-8') as fh:
                d = json.load(fh)
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def get(self, key: str) -> str | None:
        return self._read().get(key)

    def has(self, key: str) -> bool:
        return key in self._read()

    def set(self, key: str, value: str | None):
        d = self._read()
        if value is None:
            d.pop(key, None)
        else:
            d[key] = str(value)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + '.tmp'
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            json.dump(d, fh)
        os.replace(tmp, self.path)
        os.chmod(self.path, 0o600)
