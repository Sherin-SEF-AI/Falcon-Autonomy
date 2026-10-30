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
# the assistant (docs/assistant.md): off and without consent until the user turns it on and accepts its notice; the
# key is never here (assistant.keystore). Prices: $ per million tokens (input, output), the user's own table — the
# defaults are what console.groq.com/docs/models listed on 2026-10-30.
ASSISTANT_DEFAULTS = {
    'enabled': False,
    'consent': 0,                        # the version of the notice the user accepted (0: none)
    'consent_time': '',
    'base_url': 'https://api.groq.com/openai/v1',
    'main_model': 'openai/gpt-oss-120b',
    'fast_model': 'qwen/qwen3.8-27b',    # Preview on Groq: falls back to openai/gpt-oss-20b when it is gone
    'tpm_limit': 8000,                   # tokens per minute the account allows (its limits page is authoritative)
    'temperature': 0.3,
    'reasoning_effort': 'medium',
    'mode': 'ask',
    'preview': False,                    # show what will be sent before each message goes
    'suggestions': True,                 # next-step chips on the pages and Home (deterministic offline; worded by the
                                         # fast model when the assistant is on)
    'show_me': False,                    # the agent points at what it explains (opens pages, outlines controls)
    'prices': {'openai/gpt-oss-120b': [0.15, 0.60], 'openai/gpt-oss-20b': [0.075, 0.30],
               'qwen/qwen3.8-27b': [0.80, 4.00]},
    'models': [],                        # the server's last model list ({id, context_window}), from Refresh
    'models_checked': '',
}

DEFAULTS = {
    'carla_root': '',
    'client_python': '',
    'rpc_port': DEFAULT_RPC_PORT,
    'tm_port': DEFAULT_TM_PORT,
    'allowed_ports': [],                 # reserved ports the user explicitly chose in the app
    'server_memory_gb': SERVER_MEMORY_GB,
    'server_quality': 'Epic',
    'server_ros2': False,                # start Falcon's server with --ros2 (CARLA's native ROS 2; docs/stack-testing.md)
    'free_gpu_for_jobs': True,           # stop Falcon's idle server before a GPU job (measured: it holds 5.9 GB idle)
    'render_budget_mpx': 200.0,
    'theme': 'dark',
    'recent_projects': [],
    'first_run_done': False,
    'measured': {},                      # throughput measured on this machine (budget.Throughput.to_dict())
    'geocoding_allowed': False,          # network place-name search only when the user turns it on
    'cloud_llm': None,                   # never set unless the user configures and approves one
    'notify_jobs': True,                 # a desktop notification when a long job ends while the window is in the back
    'hil_device': {},                    # the HIL device: address, ports, SSH user and key path (core.hil); no password
    'assistant': copy.deepcopy(ASSISTANT_DEFAULTS),
}


# What each setting is, in the Settings page's words (the assistant's knowledge_setting and the manual read these):
# key -> (where it is set, what it does). Settings marked private are explained but their values never shown.
EXPLAIN = {
    'carla_root': ('Settings ▸ CARLA ▸ CARLA folder', 'the CARLA 0.9.16 folder (it holds CarlaUE4/Binaries/Linux/'
                   'CarlaUE4-Linux-Shipping); empty: Falcon uses the first one it finds'),
    'client_python': ('Settings ▸ CARLA ▸ CARLA\'s Python', 'the venv with carla==0.9.16, numpy and OpenCV that the '
                      'workers run in; empty: the venv beside the CARLA folder'),
    'rpc_port': ('Settings ▸ Falcon\'s server ▸ RPC port', 'the port of Falcon\'s own server (CARLA also listens on '
                 'the next two); CARLA\'s defaults 2000–2010 and 8100 only when chosen there, and then Falcon connects '
                 'to the server it finds and never starts or stops it'),
    'tm_port': ('Settings ▸ Falcon\'s server ▸ Traffic manager port', 'the traffic manager\'s port of Falcon\'s server'),
    'allowed_ports': ('Settings ▸ Falcon\'s server (asked when a default port is chosen)', 'CARLA default ports the '
                      'user chose explicitly: a server there is the user\'s own (external)'),
    'server_memory_gb': ('Settings ▸ Falcon\'s server ▸ Memory cap', 'the server\'s systemd memory cap (no swap); a '
                         '0.9.16 server capped at 12 GB ran 77 Mpx per step and crashed at 306'),
    'server_quality': ('Settings ▸ Falcon\'s server ▸ Quality', 'the server\'s rendering quality: Epic or Low'),
    'server_ros2': ('Settings (stack testing)', 'start Falcon\'s server with --ros2, CARLA\'s native ROS 2 '
                    '(docs/stack-testing.md)'),
    'free_gpu_for_jobs': ('Settings ▸ Falcon\'s server ▸ GPU jobs', 'stop Falcon\'s idle server before a training or '
                          'other GPU job: an idle 0.9.16 server holds about 6 GB of video memory'),
    'render_budget_mpx': ('Settings ▸ Falcon\'s server ▸ Render budget', 'runs rendering more megapixels per step are '
                          'refused before they start'),
    'theme': ('Settings ▸ Appearance ▸ Theme', 'dark or light'),
    'recent_projects': ('File ▸ Recent projects', 'the projects opened last (private: paths)'),
    'first_run_done': ('Tools ▸ First-run wizard', 'whether the first-run wizard has finished'),
    'measured': ('the first-run wizard\'s smoke drive and every drive after', 'this machine\'s measured rendering '
                 'throughput (megapixels per second and per-step overhead) behind every cost estimate'),
    'geocoding_allowed': ('Settings ▸ Appearance, privacy and notifications ▸ Geocoding', 'place-name search over the '
                          'network (OpenStreetMap Nominatim); off: imports work from files and bounding boxes only'),
    'cloud_llm': ('not set from the app', 'a cloud language model the user configured and approved; never set '
                  'otherwise'),
    'notify_jobs': ('Settings ▸ Appearance, privacy and notifications ▸ Job notifications', 'a desktop notification '
                    'when a job of 20 s or more ends while Falcon is in the background; nothing leaves the machine'),
    'hil_device': ('Hardware page', 'the hardware-in-the-loop device: address, ports, SSH user and key path (private: '
                   'never in a project; no password here)'),
    'assistant': ('Settings ▸ Assistant (and the Assistant panel)', 'the assistant: on or off, the notice accepted, '
                  'the server, the main and fast models, the token limit, temperature, reasoning effort, permission '
                  'mode, the price table, next-step suggestions on or off, show-me mode; never the key (keyring or '
                  'secrets file)'),
}
PRIVATE_SETTINGS = ('recent_projects', 'hil_device', 'cloud_llm')

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
