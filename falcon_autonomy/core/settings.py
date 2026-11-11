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
    'provider': 'groq',                  # assistant.provider.PROFILES (4.0: groq; local vllm/ollama/sglang to come)
}
# a cloud burst (4.0, cluster/cloud): never without an approved estimate, and refused over these caps even when approved
CLOUD_DEFAULTS = {
    'provider': '',                      # '' (none), 'runpod' or 'lambda'; the API key is in the keyring (keystore)
    'max_usd_per_burst': 10.0,
    'max_usd_month': 50.0,
    'ttl_factor': 1.5,                   # a burst's instances are terminated past its estimate × this
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
    'lineage_auto_index': False,         # index every finished run's frames into the content store (core.lineage)
    'render_budget_mpx': 200.0,
    'theme': 'dark',
    'reduce_motion': False,              # no page crossfades, sliding selection or pulsing dots (ui.motion)
    'recent_projects': [],
    'first_run_done': False,
    'measured': {},                      # throughput measured on this machine (budget.Throughput.to_dict())
    'geocoding_allowed': False,          # network place-name search only when the user turns it on
    'cloud_llm': None,                   # never set unless the user configures and approves one
    'notify_jobs': True,                 # a desktop notification when a long job ends while the window is in the back
    'user_name': '',                     # who you are in a team workspace (core.team): a label, not an account
    'hil_device': {},                    # the HIL device: address, ports, SSH user and key path (core.hil); no password
    'assistant': copy.deepcopy(ASSISTANT_DEFAULTS),
    # 4.0 (in progress): a second CARLA backend, the low-VRAM profile, cluster and cloud execution, CAN and PTP HIL
    'carla_roots': {},                   # backend id -> its CARLA folder (empty: carla_root / the first found)
    'client_pythons': {},                # backend id -> the venv its workers run in (empty: client_python / beside it)
    'default_backend': 'carla-0.9.16',   # 'carla-0.9.16' or 'carla-ue5' (a project's simulator.backend overrides it)
    'profile': 'standard',               # 'standard' or 'low_vram' (core.profiles)
    'inference_precision': 'fp32',       # 'fp32', 'fp16' or 'auto' (fp16 kept only when accuracy holds)
    'texture_pool_mb': 0,                # the server's texture streaming pool; 0: CARLA's own
    'cluster': {'nodes': []},            # compute nodes: {name, host, ssh_user, ssh_key (a path), slots, pool}; no secret
    'cloud': copy.deepcopy(CLOUD_DEFAULTS),
    'hil_can': {'interfaces_allowed': ['vcan*']},  # CAN interfaces Falcon may open (patterns); real ones added by hand
    'hil_ptp': {'interface': '', 'clock_source': 'auto'},   # PTP on this NIC, or an NTP-style fallback (recorded)
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
    'lineage_auto_index': ('settings.json (lineage_auto_index)', 'after every finished run, queue the index of its '
                           'frames into the workspace\'s content store; off by default because the store hard-links '
                           'the frames, which makes the run\'s own files read-only (falcon lineage index does it on '
                           'demand)'),
    'render_budget_mpx': ('Settings ▸ Falcon\'s server ▸ Render budget', 'runs rendering more megapixels per step are '
                          'refused before they start'),
    'theme': ('Settings ▸ Appearance ▸ Theme', 'dark or light'),
    'reduce_motion': ('Settings ▸ Appearance ▸ Motion', 'turn off the interface\'s animations (page crossfades, '
                      'the navigator\'s sliding selection, pulsing status dots)'),
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
    'user_name': ('Settings ▸ Team ▸ Your name', 'who you are in a shared team workspace: the owner of the runs '
                  'you record, the author of your comments and triage changes (a label, not an account; empty: '
                  '$FALCON_USER or the login name)'),
    'hil_device': ('Hardware page', 'the hardware-in-the-loop device: address, ports, SSH user and key path (private: '
                   'never in a project; no password here)'),
    'assistant': ('Settings ▸ Assistant (and the Assistant panel)', 'the assistant: on or off, the notice accepted, '
                  'the server, the main and fast models, the token limit, temperature, reasoning effort, permission '
                  'mode, the price table, next-step suggestions on or off, show-me mode; never the key (keyring or '
                  'secrets file)'),
    'carla_roots': ('Settings ▸ Simulator (4.0, in progress)', 'the CARLA folder of each backend (carla-0.9.16, '
                    'carla-ue5); empty: the CARLA folder setting or the first one found'),
    'client_pythons': ('Settings ▸ Simulator (4.0, in progress)', 'the venv each backend\'s workers run in; empty: '
                       'the CARLA Python setting or the venv beside that backend\'s folder'),
    'default_backend': ('Settings ▸ Simulator (4.0, in progress)', 'the CARLA backend runs use when the project does '
                        'not name one: carla-0.9.16 (UE4) or carla-ue5 (0.10.x)'),
    'profile': ('Settings ▸ Performance (4.0, in progress)', 'standard, or low_vram: Low quality, a smaller texture '
                'pool and render budget, FP16 inference, for cards of about 8 GB'),
    'inference_precision': ('Settings ▸ Performance (4.0, in progress)', 'fp32, fp16 or auto (fp16 kept only when '
                            'the accuracy drop stays within half a point)'),
    'texture_pool_mb': ('Settings ▸ Performance (4.0, in progress)', 'the server\'s texture streaming pool in MB; 0: '
                        'CARLA\'s own'),
    'cluster': ('Settings ▸ Compute (4.0, in progress)', 'compute nodes jobs may be placed on: address, SSH user, key '
                'path, slots and pool (private: addresses; never a password)'),
    'cloud': ('Settings ▸ Compute (4.0, in progress)', 'cloud bursts: the provider, the caps per burst ($10) and per '
              'month ($50) that refuse a burst even when approved, and the factor over the estimate after which its '
              'machines are terminated; the API key is kept in the keyring'),
    'hil_can': ('Settings ▸ Hardware (4.0, in progress)', 'the CAN interfaces Falcon may open (patterns; vcan* by '
                'default, a real adapter only once confirmed there)'),
    'hil_ptp': ('Settings ▸ Hardware (4.0, in progress)', 'the network interface for PTP time sync, and the clock '
                'source (auto: PTP when available, else an NTP-style fallback; both recorded)'),
}
PRIVATE_SETTINGS = ('recent_projects', 'hil_device', 'cloud_llm', 'cluster')

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
                    dv = DEFAULTS[k]
                    # a settings file from an older build: a group's keys added since keep their defaults
                    self.data[k] = ({**copy.deepcopy(dv), **v} if isinstance(dv, dict) and dv and isinstance(v, dict)
                                    else v)

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
