"""What differs between CARLA backends, inside a worker (runs in CARLA's Python: standard library and numpy).

Worker.import_carla() attaches a Compat to the worker (w.compat) and makes it the process's active one (active()).
On CARLA 0.9.16 — the default backend, on which everything Falcon does was measured — it changes nothing: every
method is the identity, and run.json only records the backend in its provenance. On another backend (CARLA's UE5
line, sim.backends) it adapts what Falcon knows to differ, and says so in run.json (`compat`):

* the backend: from the client module's version (carla.__version__, else the installed wheel's); a job started for
  another backend (FALCON_BACKEND, set by sim.server.worker_command) is refused (NO_CARLA), as is a run config made
  for another one (cfg['backend'], a project's simulator.backend);
* blueprints: BlueprintLibrary.find / filter fall back to the backend's aliases (resources/carla/
  blueprints-<backend>.json) and then to a blueprint of the same base type (0.9.16's inventory says which), each
  substitution recorded once;
* semantic tags: carla.CityObjectLabel is read at run time; the tags the instance camera and the semantic lidar
  write are mapped to Falcon's classes (0.9.16's numbering) by label name (core.groundtruth.tags_for, tag_lut);
* worlds and capabilities: a map the server does not list, an imported (OpenDRIVE) world, an OSM import, lateral
  filtering are refused or turned off with a reason (sim.backends.check_needs; require(), has());
  set_osm_mode is called only where the traffic manager has it;
* weather: fields the backend's WeatherParameters lacks are noted when a run asks for them (non-zero);
* cameras: CARLA lens and post-process attributes a view asks for (core.lens.VIEW_ATTRIBUTES) that the backend's
  blueprint lacks are left out, and noted;
* tone mapping: the vignette correction (sim.tonemap) is 0.9.16's measured camera response; on a backend it was not
  measured for it is off, and noted.
"""
from __future__ import annotations

import json
import os

from ..core import protocol as P
from ..sim import backends as B

_ACTIVE = None
RESOURCES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'resources', 'carla')


def active() -> 'Compat':
    """The worker's Compat (the identity before a worker imported carla)."""
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = Compat()
    return _ACTIVE


def of(w) -> 'Compat':
    """A worker's Compat; the active one for an object without its own (a test's stand-in worker)."""
    c = getattr(w, 'compat', None)
    return c if isinstance(c, Compat) else active()


def client_version(carla) -> str:
    """The carla client module's version: its __version__, else the installed wheel's metadata ('' unknown)."""
    v = str(getattr(carla, '__version__', '') or '')
    if v:
        return v
    try:
        from importlib.metadata import version
        return version('carla')
    except Exception:                                      # noqa: BLE001 (not installed as a distribution)
        return ''


def load_aliases(backend_id: str) -> dict:
    """{0.9.16 blueprint id: this backend's} from resources/carla/blueprints-<backend>.json ({} without one)."""
    try:
        with open(os.path.join(RESOURCES, f'blueprints-{backend_id}.json'), encoding='utf-8') as fh:
            return dict(json.load(fh).get('aliases') or {})
    except (OSError, ValueError):
        return {}


def base_type_0916(bp_id: str) -> str:
    """A 0.9.16 blueprint's base type from Falcon's 0.9.16 inventory ('' unknown)."""
    try:
        with open(os.path.join(RESOURCES, B.get(B.DEFAULT).inventory), encoding='utf-8') as fh:
            inv = json.load(fh)
    except (OSError, ValueError):
        return ''
    for v in inv.get('vehicles') or []:
        if v.get('id') == bp_id:
            return str(v.get('base_type') or '').lower()
    return ''


class Compat:
    def __init__(self, carla=None, expected: str = '', emit_note=None):
        self.carla = carla
        self.version = client_version(carla) if carla is not None else ''
        found = B.detect_version(self.version)
        self.expected = expected or ''
        self.mismatch = bool(found and expected and found != expected)
        self.backend_id = found or expected or B.DEFAULT
        self.backend = B.get(self.backend_id)
        self.default = self.backend_id == B.DEFAULT
        self.notes: list = []
        self.substitutions: dict = {}
        self._emit_note = emit_note
        self._noted: set = set()
        self.lut = None                                    # backend tag -> Falcon class id (None: the identity)
        self.labels: dict = {}

    # ------------------------------------------------------------------ attaching
    def attach(self):
        """Adapt the client module (nothing on 0.9.16). Raises common.Failed on a backend mismatch."""
        from .common import Failed
        if self.mismatch:
            raise Failed(f'this job was started for {B.get(self.expected).title}, but CARLA\'s Python has the carla '
                         f'{self.version} client ({self.backend.title}): set each backend\'s Python in Settings ▸ '
                         'Simulator', P.NO_CARLA)
        if self.default or self.carla is None:
            return self
        from ..core import groundtruth as GT
        self.labels = GT.labels_of(getattr(self.carla, 'CityObjectLabel', None))
        if self.labels:
            self.lut, unknown = GT.tag_lut(self.labels)
            if unknown:
                self.note(f'{self.backend.title}\'s semantic labels {", ".join(sorted(unknown))} have no Falcon class: '
                          'labelled unlabelled')
        else:
            self.note(f'{self.backend.title}\'s client has no CityObjectLabel: its tags are taken as 0.9.16\'s')
        self._patch_library()
        return self

    def note(self, text: str):
        if text in self._noted:
            return
        self._noted.add(text)
        self.notes.append(text)
        if self._emit_note is not None:
            self._emit_note(text)

    def doc(self) -> dict:
        """What run.json's provenance records: the backend and the client's version."""
        return {'id': self.backend_id, 'carla_client': self.version or None}

    def report(self) -> dict | None:
        """run.json `compat` on a backend other than 0.9.16 (None on 0.9.16)."""
        if self.default:
            return None
        return {'backend': self.backend_id, 'substitutions': dict(self.substitutions), 'notes': list(self.notes),
                'labels_read': bool(self.labels)}

    # ------------------------------------------------------------------ blueprints
    def _patch_library(self):
        lib_cls = getattr(self.carla, 'BlueprintLibrary', None)
        if lib_cls is None or getattr(lib_cls, '_falcon_compat', False):
            return
        orig_find, orig_filter = lib_cls.find, lib_cls.filter
        aliases = load_aliases(self.backend_id)
        me = self

        def find(lib, bp_id):
            try:
                return orig_find(lib, bp_id)
            except (IndexError, KeyError, RuntimeError):
                pass
            alt = me.substitute(lib, bp_id, aliases, orig_find, orig_filter)
            if alt is None:
                raise IndexError(f'blueprint {bp_id!r} not found on {me.backend.title} (no alias, no blueprint of its '
                                 'base type)')
            return orig_find(lib, alt)

        def filter_(lib, pattern):
            got = orig_filter(lib, pattern)
            if len(got) or any(c in pattern for c in '*?[') or pattern not in aliases:
                return got
            return orig_filter(lib, aliases[pattern])

        find.__doc__ = orig_find.__doc__
        lib_cls.find, lib_cls.filter = find, filter_
        try:
            lib_cls._falcon_compat = True
        except (AttributeError, TypeError):
            pass

    def substitute(self, lib, bp_id: str, aliases: dict, find, filter_) -> str | None:
        """This backend's blueprint for a 0.9.16 id: its alias when the server has it, else the first (by id) of the
        same base type; recorded once."""
        if bp_id in self.substitutions:
            return self.substitutions[bp_id]
        alt = aliases.get(bp_id)
        how = 'alias'
        if alt is not None:
            try:
                find(lib, alt)
            except (IndexError, KeyError, RuntimeError):
                alt = None
        if alt is None and bp_id.startswith('vehicle.'):
            base = base_type_0916(bp_id)
            same = sorted(b.id for b in filter_(lib, 'vehicle.*')
                          if base and b.has_attribute('base_type') and
                          b.get_attribute('base_type').as_str().lower() == base)
            alt, how = (same[0], f'same base type, {base}') if same else (None, '')
        if alt is None:
            return None
        self.substitutions[bp_id] = alt
        self.note(f'{bp_id} is {alt} on {self.backend.title} ({how})')
        return alt

    # ------------------------------------------------------------------ semantic tags
    def canon_tags(self, tags):
        """A backend's semantic tags (any shape, integer) as Falcon's classes (0.9.16's numbering)."""
        if self.lut is None:
            return tags
        import numpy as np
        return self.lut[np.asarray(tags).astype(np.int64) & 255]

    # ------------------------------------------------------------------ worlds and capabilities
    def has(self, cap: str, obj=None) -> bool:
        """Whether the client has a capability (sim.backends.CAPS) — read from the module itself where it can be."""
        c = self.carla
        if c is None:
            return self.backend.cap(cap)
        if cap == 'osm2odr':
            return hasattr(c, 'Osm2Odr')
        if cap in ('tm_osm', 'lane_offset'):
            tm = obj if obj is not None else getattr(c, 'TrafficManager', None)
            return tm is not None and hasattr(tm, 'set_osm_mode' if cap == 'tm_osm' else 'vehicle_lane_offset')
        if cap == 'ros2':
            return hasattr(getattr(c, 'Sensor', None), 'enable_for_ros')
        return self.backend.cap(cap)

    def require(self, cap: str, what: str = ''):
        from .common import Failed
        if not self.has(cap):
            raise Failed(f'{self.backend.title} cannot do {what or B.CAP_NEED[cap]}: its client has no '
                         f'{B.CAP_NEED[cap]}' + ('' if self.default else f' (run it on {B.get(B.DEFAULT).title})'),
                         P.REFUSED)

    def check_config(self, cfg: dict, client=None):
        """Refuse (common.Failed, REFUSED) a run config this backend cannot run: one made for another backend
        (cfg['backend']; a config naming none runs on any), an imported world without OpenDRIVE worlds, a map the
        server does not list (non-default backends: 0.9.16 is never asked, as before)."""
        from .common import Failed
        want = cfg.get('backend')                # absent: made for whatever backend runs it (the setting's)
        if want and want != self.backend_id:
            raise Failed(f'this run is for {B.get(want).title} (the project\'s simulator) but runs on '
                         f'{self.backend.title}: switch the backend in Settings ▸ Simulator', P.REFUSED)
        if self.default:
            return
        if cfg.get('xodr') and not self.backend.cap('opendrive_world'):
            raise Failed(f'{self.backend.title} cannot run an imported (OpenDRIVE) world as far as Falcon knows: run '
                         f'it on {B.get(B.DEFAULT).title}', P.REFUSED)
        town = cfg.get('town')
        if town and client is not None:
            try:
                have = sorted({m.rpartition('/')[2] for m in client.get_available_maps()})
            except RuntimeError:
                have = []
            if have and town not in have:
                raise Failed(f'{self.backend.title} has no map {town} (it has {", ".join(have)}): choose one of '
                             f'those, or run the project on {B.get(B.DEFAULT).title}', P.REFUSED)

    def set_osm_mode(self, tm, on: bool) -> bool:
        """TrafficManager.set_osm_mode where the traffic manager has it; False (noted) where not."""
        if tm is not None and hasattr(tm, 'set_osm_mode'):
            tm.set_osm_mode(bool(on))
            return True
        self.note(f'{self.backend.title}\'s traffic manager has no OSM mode: left as it is')
        return False

    # ------------------------------------------------------------------ weather, cameras, tone mapping
    def weather_missing(self, wp, fields: dict) -> list:
        """The fields a run asks for (non-zero) that this backend's WeatherParameters lacks; noted."""
        from ..core.weather import unsupported
        miss = unsupported(fields, [k for k in (fields or {}) if hasattr(wp, k)])
        if miss and not self.default:
            self.note(f'{self.backend.title}\'s weather has no {", ".join(miss)}: not applied')
        return miss

    def missing_attribute(self, bp_id: str, key: str):
        """A camera view's lens or post-process attribute (core.lens.VIEW_ATTRIBUTES) the backend's blueprint lacks."""
        if self.default:
            return
        from ..core.lens import LABEL_VIEW_ATTRIBUTES, VIEW_ATTRIBUTES
        if key in VIEW_ATTRIBUTES or key in LABEL_VIEW_ATTRIBUTES:
            self.note(f'{self.backend.title}\'s {bp_id} has no {key}: left at the engine\'s own (a remapped camera\'s '
                      'views may differ at their seams)')

    def tonemap_measured(self) -> bool:
        """Whether sim.tonemap's camera response was measured on this backend (the vignette correction needs it)."""
        from ..sim import tonemap as TM
        ok = TM.params_for(self.backend_id) is not None
        if not ok:
            self.note(f'the vignette correction is off: {self.backend.title}\'s camera response and vignette were not '
                      'measured (sim.tonemap)')
        return ok


def attach(worker, carla) -> Compat:
    """Make the worker's Compat for this client module, attach it and make it the active one."""
    global _ACTIVE
    c = Compat(carla, os.environ.get('FALCON_BACKEND', ''), worker.note)
    _ACTIVE = c
    worker.compat = c
    c.attach()
    return c
