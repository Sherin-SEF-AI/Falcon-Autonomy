"""CARLA backends: the two simulators Falcon can run on, what each one is, where it is and what it can do (standard
library only; imported by the GUI, the CLI and — for detection and the capability table — by the workers in CARLA's
own Python).

    carla-0.9.16   CARLA 0.9.16 on Unreal Engine 4.26: everything Falcon measured was measured on it (the default)
    carla-ue5      CARLA's Unreal Engine 5 line (0.10.x, on UE 5.5): an adapter, tested against a fake; not yet run
                   against a real 0.10 server

A backend is a Backend record: the server binary inside its CARLA folder and the project argument it takes, the
quality arguments, the client wheel, the inventory and city catalogue Falcon measured on it, the maps it ships and its
capabilities. Capabilities (CAPS) are what Falcon relies on per backend:

    ros2             the server's native ROS 2 interface (--ros2; 0.9.16 only — the UE5 line has none)
    osm2odr          carla.Osm2Odr, which turns OpenStreetMap into OpenDRIVE (importing a place)
    tm_osm           TrafficManager.set_osm_mode
    lane_offset      TrafficManager.vehicle_lane_offset (lateral filtering of two-wheelers)
    opendrive_world  Client.generate_opendrive_world (an imported world)

The UE5 figures are the adapter's assumptions until `falcon backend probe` reads them from a server: the probe records
the maps, blueprints, semantic tags, weather fields and camera attributes the server and its client module have, and
the API capabilities (osm2odr, tm_osm, lane_offset, ros2 as the client module has them) replace the declared ones.
opendrive_world is never inferred from a probe (the method exists in both client modules; whether a UE5 server builds
a world from it was not checked), so it stays as declared.

check_needs() refuses, before anything is queued, what a backend cannot do: a map it does not have, an imported world,
native ROS 2, an OSM import. The worker refuses the same at run time (workers.compat) when it gets there anyway.

`falcon backend list|check|probe` (main_backend) shows the table, checks a backend's folder and Python, and probes it.
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field

DEFAULT = 'carla-0.9.16'
CAPS = ('ros2', 'osm2odr', 'tm_osm', 'lane_offset', 'opendrive_world')
CAP_TEXT = {'ros2': 'native ROS 2 (--ros2)', 'osm2odr': 'OSM import (carla.Osm2Odr)',
            'tm_osm': 'traffic manager OSM mode', 'lane_offset': 'lane offset (lateral filtering)',
            'opendrive_world': 'imported OpenDRIVE worlds'}
# what each capability is needed for, in a refusal's words
CAP_NEED = {'ros2': 'native ROS 2 (the server\'s --ros2)', 'osm2odr': 'importing a place (carla.Osm2Odr)',
            'tm_osm': 'the traffic manager\'s OSM mode', 'lane_offset': 'lateral filtering (vehicle_lane_offset)',
            'opendrive_world': 'an imported (OpenDRIVE) world'}
# the texture streaming pool: both forms name the same value; which one CARLA 0.9.16's packaged server honours has
# not been measured (docs/profiles.md), so Falcon passes both
POOL_INI = '-ini:Engine:[/Script/Engine.RendererSettings]:r.Streaming.PoolSize={mb}'
POOL_EXEC = '-ExecCmds=r.Streaming.PoolSize {mb}'


class BackendError(ValueError):
    pass


@dataclass(frozen=True)
class Backend:
    id: str
    title: str
    carla: str                    # the CARLA release
    engine: str
    versions: tuple               # client / server version prefixes that are this backend
    binary: str                   # the server, relative to the CARLA folder
    project: str                  # the project argument the shipping binary takes
    launcher: str                 # the release's start script (what a user runs by hand)
    wheel: str                    # how its client module is installed
    maps: tuple = ()              # the maps it ships (empty: whatever the server lists; not limited)
    inventory: str = ''           # resources/carla file measured on it ('' none)
    city: str = ''
    caps: dict = field(default_factory=dict)
    measured: bool = False        # Falcon's numbers (timings, VRAM, tone curve, tags) were measured on it
    notes: tuple = ()

    def cap(self, name: str) -> bool:
        return bool(self.caps.get(name, False))


BACKENDS = {
    'carla-0.9.16': Backend(
        id='carla-0.9.16', title='CARLA 0.9.16', carla='0.9.16', engine='Unreal Engine 4.26', versions=('0.9.16',),
        binary=os.path.join('CarlaUE4', 'Binaries', 'Linux', 'CarlaUE4-Linux-Shipping'), project='CarlaUE4',
        launcher='CarlaUE4.sh', wheel='pip install carla==0.9.16 numpy opencv-python-headless',
        inventory='inventory-0.9.16.json', city='city-0.9.16.json',
        caps={'ros2': True, 'osm2odr': True, 'tm_osm': True, 'lane_offset': True, 'opendrive_world': True},
        measured=True,
        notes=('Every figure in Falcon\'s documentation was measured on it.',)),
    'carla-ue5': Backend(
        id='carla-ue5', title='CARLA 0.10 (Unreal Engine 5)', carla='0.10.x', engine='Unreal Engine 5.5',
        versions=('0.10.',),
        binary=os.path.join('CarlaUnreal', 'Binaries', 'Linux', 'CarlaUnreal-Linux-Shipping'), project='CarlaUnreal',
        launcher='CarlaUnreal.sh',
        wheel='pip install the carla wheel from the release\'s PythonAPI/carla/dist (0.10.x) numpy '
              'opencv-python-headless',
        maps=('Town10HD_Opt', 'Mine_01'),
        caps={'ros2': False, 'osm2odr': False, 'tm_osm': False, 'lane_offset': True, 'opendrive_world': False},
        measured=False,
        notes=('An adapter tested against a fake (tests/fake_carla_ue5); not yet run against a real 0.10 server.',
               'Ships two maps; no native ROS 2; scenario code is the same, a world it lacks is refused.',
               'Blueprint names, semantic tag values and camera attributes are read at run time; the tone curve '
               'and vignette correction, the traffic keeper\'s constants and every timing are 0.9.16\'s.')),
}


def get(backend_id: str) -> Backend:
    b = BACKENDS.get(backend_id or DEFAULT)
    if b is None:
        raise BackendError(f'unknown backend {backend_id!r} (one of {", ".join(BACKENDS)})')
    return b


# ================================================================================================ detection
def detect_root(path) -> str:
    """The backend whose server binary this CARLA folder holds, or ''."""
    if not path:
        return ''
    for b in BACKENDS.values():
        if os.path.isfile(os.path.join(path, b.binary)):
            return b.id
    return ''


def detect_version(version) -> str:
    """The backend of a carla client or server version ('0.9.16', '0.10.0-…'), or ''."""
    v = str(version or '').strip()
    for b in BACKENDS.values():
        if any(v.startswith(p) for p in b.versions):
            return b.id
    return ''


def fake_backend(folder: str) -> str:
    """The backend of a fake `carla` package (tests): its module's __version__ (the last assignment in
    carla/__init__.py), DEFAULT when it names none Falcon knows."""
    try:
        with open(os.path.join(folder, 'carla', '__init__.py'), encoding='utf-8') as fh:
            found = re.findall(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", fh.read(), re.M)
    except OSError:
        return DEFAULT
    return (detect_version(found[-1]) if found else '') or DEFAULT


def detect(root: str = '', version: str = '') -> str:
    """A backend from a CARLA folder or a version, '' when neither says."""
    return detect_root(root) or detect_version(version)


# ================================================================================================ settings
def default_backend(settings) -> str:
    b = (settings.get('default_backend', DEFAULT) if settings is not None else DEFAULT) or DEFAULT
    return b if b in BACKENDS else DEFAULT


def run_backend(project=None, settings=None) -> str:
    """The backend a project's runs use: its simulator.backend when it names one, else the setting default_backend
    (the user's settings when none are given)."""
    doc = getattr(project, 'doc', None) or {}
    named = (doc.get('simulator') or {}).get('backend')
    if named:
        return get(named).id
    if settings is None:
        from ..core.settings import Settings
        settings = Settings()
    return default_backend(settings)


def root_for(settings, backend_id: str) -> str:
    """The CARLA folder of a backend: settings carla_roots[id]; for 0.9.16 also carla_root (as before); else the
    first folder of that backend found in the usual places."""
    roots = (settings.get('carla_roots', {}) if settings is not None else {}) or {}
    r = roots.get(backend_id) or ''
    if not r and backend_id == DEFAULT and settings is not None:
        r = settings.get('carla_root') or ''
    if not r:
        from .server import candidate_roots
        found = candidate_roots(backend=backend_id)
        r = found[0] if found else ''
    return r


def python_for(settings, backend_id: str, root: str = '') -> str:
    """The Python a backend's workers run in: client_pythons[id]; for 0.9.16 also client_python; else the venv
    beside its CARLA folder."""
    pys = (settings.get('client_pythons', {}) if settings is not None else {}) or {}
    py = pys.get(backend_id) or ''
    if not py and backend_id == DEFAULT and settings is not None:
        py = settings.get('client_python') or ''
    if not py:
        from .server import client_python
        py = client_python(root if root else root_for(settings, backend_id))
    return py


# ================================================================================================ the server command
def server_args(backend_id: str, port: int, quality: str = 'Epic', no_render: bool = False, ros2: bool = False,
                texture_pool_mb: int = 0) -> list:
    """The server's arguments after its binary. 0.9.16's are exactly what Falcon always passed; the UE5 line takes
    the same switches (unverified on a 0.10 server). BackendError when the backend cannot do what is asked."""
    b = get(backend_id)
    args = [b.project, f'-carla-rpc-port={int(port)}', '-nosound']
    if no_render:
        args.append('-nullrhi')                       # (-quality-level=Low crashes UE4 with -nullrhi)
    else:
        args.append('-RenderOffScreen')
        if quality == 'Low':
            args.append('-quality-level=Low')
    if ros2:
        if not b.cap('ros2'):
            raise BackendError(f'{b.title} has no native ROS 2 interface: turn off the server\'s ROS 2 in Settings '
                               '(the ROS 2 bridge container still works)')
        args.append('--ros2')                         # CARLA 0.9.16's native ROS 2 (its own Fast DDS)
    mb = int(texture_pool_mb or 0)
    if mb > 0 and not no_render:
        args += [POOL_INI.format(mb=mb), POOL_EXEC.format(mb=mb)]
    return args


# ================================================================================================ probes and needs
def probe_path(backend_id: str) -> str:
    from ..core.paths import config_dir
    return os.path.join(config_dir(), 'backends', f'{backend_id}.json')


def load_probe(backend_id: str) -> dict:
    """What `falcon backend probe` last read from a server of this backend ({} when never probed)."""
    try:
        with open(probe_path(backend_id), encoding='utf-8') as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save_probe(backend_id: str, info: dict) -> str:
    from ..core.paths import atomic_write
    from ..core.provenance import utc_now
    path = probe_path(backend_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    atomic_write(path, json.dumps(dict(info, backend=backend_id, probed=utc_now()), indent=1) + '\n')
    return path


def capabilities(backend_id: str, probe: dict | None = None) -> dict:
    """{cap: bool} — declared, with the API capabilities a probe read in their place (never opendrive_world)."""
    b = get(backend_id)
    out = {c: b.cap(c) for c in CAPS}
    api = ((probe if probe is not None else load_probe(backend_id)).get('caps') or {})
    for c in ('ros2', 'osm2odr', 'tm_osm', 'lane_offset'):
        if c in api:
            out[c] = bool(api[c])
    return out


def maps(backend_id: str, probe: dict | None = None) -> tuple:
    """The maps a backend has: a probe's list, else the declared one (empty: not limited)."""
    p = probe if probe is not None else load_probe(backend_id)
    return tuple(p.get('maps') or get(backend_id).maps)


def advertised(settings, fake_carla: str = '') -> dict:
    """What a cluster node says it can run (its daemon's hello; docs/cluster.md): {backend id: {'caps': {cap: bool},
    'maps': [...], 'probed': when `falcon backend probe` last read them ('' never), 'measured': bool, 'title'}} for
    every backend this machine has — its CARLA folder holds that backend's server and its client Python is there
    (root_for, python_for) — with the capabilities and maps as capabilities()/maps() give them (declared, replaced by
    the probe cache where one was read). A fake carla (tests) is the one backend it is."""
    if fake_carla:
        ids = [fake_backend(fake_carla)]
    else:
        ids = []
        for b in BACKENDS.values():
            root = root_for(settings, b.id)
            py = python_for(settings, b.id, root) if root else ''
            if detect_root(root) == b.id and py and os.access(py, os.X_OK):
                ids.append(b.id)
    out = {}
    for bid in ids:
        p = load_probe(bid)
        out[bid] = {'title': get(bid).title, 'caps': capabilities(bid, p), 'maps': list(maps(bid, p)),
                    'probed': p.get('probed', ''), 'measured': get(bid).measured}
    return out


def needs_of(cfg: dict) -> dict:
    """What a run config (drive, scenario, live) needs of its backend: {'town', 'opendrive_world', 'lane_offset'}."""
    out = {}
    if cfg.get('xodr'):
        out['opendrive_world'] = True
    elif cfg.get('town'):
        out['town'] = cfg['town']
    tr = cfg.get('traffic') or {}
    if tr.get('lateral_filtering'):
        out['lane_offset'] = True
    if cfg.get('schema') == 'falcon.osm_import/1' and not cfg.get('city_only'):
        out['osm2odr'] = True
    return out


def check_needs(backend_id: str, needs: dict, probe: dict | None = None) -> list:
    """Refusals (sentences) for what the backend cannot do: needs {'town': name, cap: True, …}. [] when it can."""
    b = get(backend_id)
    p = probe if probe is not None else load_probe(backend_id)
    caps = capabilities(backend_id, p)
    out = []
    town = needs.get('town')
    have = maps(backend_id, p)
    if town and have and town not in have:
        out.append(f'{b.title} has no map {town} (it has {", ".join(have)}): choose one of those, or run the '
                   f'project on {get(DEFAULT).title}')
    for c in CAPS:
        if needs.get(c) and not caps[c]:
            out.append(f'{b.title} cannot run {CAP_NEED[c]}' + ('' if b.measured else ' (as far as Falcon knows: '
                                                                 '`falcon backend probe` reads it from a server)'))
    return out


# ================================================================================================ `falcon backend`
def _table(settings) -> list:
    from .server import carla_version
    rows = []
    for b in BACKENDS.values():
        root = root_for(settings, b.id)
        p = load_probe(b.id)
        rows.append({'id': b.id, 'title': b.title, 'engine': b.engine, 'default': b.id == default_backend(settings),
                     'root': root, 'root_ok': detect_root(root) == b.id, 'version': carla_version(root) if root else '',
                     'python': python_for(settings, b.id, root), 'maps': list(maps(b.id, p)),
                     'caps': capabilities(b.id, p), 'probed': p.get('probed', ''), 'measured': b.measured,
                     'notes': list(b.notes)})
    return rows


def _print_table(rows):
    from ..core.paths import tilde
    for r in rows:
        mark = ' (default)' if r['default'] else ''
        print(f'{r["id"]}{mark}: {r["title"]}, {r["engine"]}')
        ver = f' ({r["version"]})' if r['version'] else ''
        print(f'  folder   {tilde(r["root"]) + ver if r["root_ok"] else "not found"}')
        print(f'  python   {tilde(r["python"]) or "not found"}')
        print(f'  maps     {", ".join(r["maps"]) or "whatever the server lists"}'
              + (f' (probed {r["probed"]})' if r['probed'] else ''))
        print('  can      ' + ', '.join(f'{CAP_TEXT[c]} {"yes" if v else "no"}' for c, v in r['caps'].items()))
        for n in r['notes']:
            print(f'  note     {n}')


def check(settings, backend_id: str, root: str = '', python: str = '') -> dict:
    """{'ok', 'rows': [(name, ok, detail)]}: the backend's folder holds its server, its Python imports its client."""
    from ..core.paths import tilde
    from .server import check_client
    b = get(backend_id)
    root = root or root_for(settings, backend_id)
    rows = []
    found = detect_root(root)
    if found == backend_id:
        rows.append(('CARLA folder', True, f'{tilde(root)} holds {b.binary}'))
    elif found:
        rows.append(('CARLA folder', False, f'{tilde(root)} is a {get(found).title} folder, not {b.title}'))
    else:
        rows.append(('CARLA folder', False, f'{tilde(root) or "(none)"}: no {b.binary} — set it in Settings ▸ '
                                            'Simulator' if root else f'no {b.title} folder found: set it in Settings ▸ '
                                                                     'Simulator'))
    py = python or python_for(settings, backend_id, root)
    chk = check_client(py, backend=backend_id) if py else {'ok': False, 'text': 'no venv beside the CARLA folder'}
    rows.append(('CARLA\'s Python', chk['ok'], f'{tilde(py)}: {chk["text"]}' if py else chk['text']))
    p = load_probe(backend_id)
    rows.append(('Probe', True, f'probed {p["probed"]}: {len(p.get("maps") or [])} maps, '
                                f'{p.get("blueprint_count", "?")} blueprints' if p else
                 'never probed (falcon backend probe reads maps and capabilities from a server)'))
    return {'ok': all(r[1] for r in rows), 'rows': rows, 'backend': backend_id}


def main_backend(argv) -> int:
    """`falcon backend list [--json]`, `falcon backend check [ID] [--root DIR] [--python PY] [--json]`,
    `falcon backend probe [ID] [--fake-carla DIR] [--wait S] [--keep-server]`: 0 ok, 2 failed (or the probe's code)."""
    import argparse
    from ..core import protocol as P
    from ..core.settings import Settings
    ap = argparse.ArgumentParser(prog='falcon backend', description='the CARLA backends Falcon can run on')
    sub = ap.add_subparsers(dest='action', required=True)
    lp = sub.add_parser('list', help='each backend: its folder, Python, maps and capabilities')
    lp.add_argument('--json', action='store_true')
    cp = sub.add_parser('check', help='a backend\'s folder holds its server and its Python imports its client')
    cp.add_argument('backend', nargs='?', help=f'one of {", ".join(BACKENDS)} (default: the setting)')
    cp.add_argument('--root', default='', help='the CARLA folder to check (default: the setting)')
    cp.add_argument('--python', default='', help='the client Python to check (default: the setting)')
    cp.add_argument('--json', action='store_true')
    pp = sub.add_parser('probe', help='read maps, blueprints, tags and capabilities from a server of the backend '
                                      '(starts Falcon\'s own server when none runs)')
    pp.add_argument('backend', nargs='?', help=f'one of {", ".join(BACKENDS)} (default: the setting)')
    pp.add_argument('--fake-carla', default='', help='a fake carla package (tests): no server is started')
    pp.add_argument('--wait', type=float, default=60.0, help='wait this long for the server, s')
    pp.add_argument('--keep-server', action='store_true', help='leave Falcon\'s server running afterwards')
    pp.add_argument('--json', action='store_true')
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return P.OK if e.code == 0 else P.FAILED
    st = Settings()
    if a.action == 'list':
        rows = _table(st)
        if a.json:
            print(json.dumps(rows, indent=1))
        else:
            _print_table(rows)
        return P.OK
    bid = a.backend or default_backend(st)
    if bid not in BACKENDS:
        print(f'failed: unknown backend {bid!r} (one of {", ".join(BACKENDS)})', file=sys.stderr)
        return P.FAILED
    if a.action == 'check':
        res = check(st, bid, a.root, a.python)
        if a.json:
            print(json.dumps(res, indent=1))
        else:
            print(f'{get(bid).title}:')
            for name, ok, detail in res['rows']:
                print(f'  {"ok " if ok else "-- "} {name:16s} {detail}')
        return P.OK if res['ok'] else P.FAILED
    return _probe(st, bid, a)


def _probe(st, bid: str, a) -> int:
    from ..cli import _run_job
    from ..core import protocol as P
    from ..core.jobs import JobQueue, SubprocessLauncher
    from .service import CarlaService
    svc = CarlaService(st, JobQueue(SubprocessLauncher()), '', fake_carla=a.fake_carla)
    try:
        if a.fake_carla and svc.backend != bid:
            print(f'failed: the fake carla at {a.fake_carla} is {svc.backend}, not {bid}', file=sys.stderr)
            return P.FAILED
        if not a.fake_carla:
            svc.use_backend(bid)
        job = svc.submit_probe(a.wait)
        code = _run_job(svc, job, quiet=True)
        info = (job.result or {}).get('summary') or {}
        if code != P.OK or not info:
            print(f'failed: the probe ended {P.status_of(code)}', file=sys.stderr)
            return code if code != P.OK else P.FAILED
        got = detect_version(info.get('server_version') or info.get('client_version'))
        if got and got != bid:
            print(f'failed: the server answered as {got} ({info.get("server_version")}), not {bid}',
                  file=sys.stderr)
            return P.FAILED
        path = save_probe(bid, info)
        if a.json:
            print(json.dumps(dict(info, saved=path), indent=1))
        else:
            caps = capabilities(bid, info)
            print(f'{get(bid).title}: server {info.get("server_version")}, client {info.get("client_version")}')
            print(f'  maps        {", ".join(info.get("maps") or [])}')
            print(f'  blueprints  {info.get("blueprint_count")}')
            print(f'  tags        {len(info.get("labels") or {})} semantic labels')
            print('  can         ' + ', '.join(f'{CAP_TEXT[c]} {"yes" if v else "no"}' for c, v in caps.items()))
            print(f'  saved       {path}')
        return P.OK
    finally:
        if not a.keep_server:
            svc.shutdown()
