"""The log: an ego trajectory and the actors' trajectories of a real (or simulated) drive, in CARLA's frame (numpy).

A log (schema falcon.log/1, `log.json`) is what every importer produces and what a replay is planned from:

* ``ego`` and ``actors``: tracks — an id, a Falcon class (CLASSES), the source's own class, a box size (length, width,
  height in metres, and where it came from), and samples of time (s from the log's start) and pose: x, y, z (m) and
  yaw (degrees), optionally pitch and roll. A track is observed between consecutive samples no further apart than
  its ``max_gap_s``; outside its samples it is absent (or held at its first / last pose, ``hold``).
* ``map``: what the coordinates refer to — a CARLA world (``kind`` 'carla': a town name, or an OpenDRIVE file's
  hash), a dataset map without a CARLA counterpart ('nuscenes', 'waymo'), a geographic origin ('geo': x east and −y
  north in metres of a local projection about lat0, lon0) or nothing known ('unknown').
* ``source``: the format, the files read (with their SHA-256), the scene or segment; ``notes``, ``left_out`` (what the
  source had that the log does not carry, and why) and ``heuristic`` (every value that is not read from the source).

Frames. Every source in this package is right-handed with z up (nuScenes global, Waymo's world and vehicle frames,
ROS REP 103/105 map and odom, KITTI's OXTS local east-north-up); CARLA is left-handed (x forward, y right, z up).
A right-handed pose (R, t) becomes CARLA's by the reflection S = diag(1, −1, 1): t → S·t, R → S·R·S (never by
flipping Euler angles one by one); for a heading about z alone this is yaw → −yaw. CARLA's Euler angles of S·R·S
are core.rig.carla_euler's (Transform::GetMatrix's convention). tests/test_logreplay.py checks this on known poses.
"""
from __future__ import annotations

import math
import os

import numpy as np

from ..rig import carla_euler

LOG_SCHEMA = 'falcon.log/1'
CLASSES = ('car', 'truck', 'bus', 'motorcycle', 'bicycle', 'pedestrian')
# typical boxes (length, width, height, m) for a track whose source gives none — heuristic, labelled where used
TYPICAL_SIZE = {'car': (4.6, 2.0, 1.5), 'truck': (8.0, 2.5, 3.2), 'bus': (11.0, 2.6, 3.2),
                'motorcycle': (2.2, 0.8, 1.5), 'bicycle': (1.8, 0.6, 1.7), 'pedestrian': (0.6, 0.6, 1.75)}
S3 = np.diag([1.0, -1.0, 1.0])


class LogError(ValueError):
    pass


# ------------------------------------------------------------------------------------------------ frames
def quat_matrix(w, x, y, z) -> np.ndarray:
    """The rotation matrix of a (normalised) quaternion w + xi + yj + zk."""
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rh_to_carla(R, t) -> tuple:
    """A right-handed pose (3 × 3 rotation, translation) as CARLA's (x, y, z, roll, pitch, yaw), degrees."""
    Rc = S3 @ np.asarray(R, float) @ S3
    roll, pitch, yaw = carla_euler(Rc)
    t = np.asarray(t, float)
    return float(t[0]), float(-t[1]), float(t[2]), roll, pitch, yaw


def rh_yaw_to_carla(yaw_rad) -> np.ndarray:
    """A right-handed heading about z (radians, counter-clockwise from +x) as CARLA's yaw (degrees)."""
    return -np.degrees(np.asarray(yaw_rad, float))


def carla_to_rh(x, y, z, roll, pitch, yaw) -> np.ndarray:
    """The 4 × 4 right-handed matrix of a CARLA pose (the inverse of rh_to_carla)."""
    from ..rig import carla_rotation
    T = np.eye(4)
    T[:3, :3] = S3 @ carla_rotation(roll, pitch, yaw) @ S3
    T[:3, 3] = [x, -y, z]
    return T


def wrap_deg(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


# ------------------------------------------------------------------------------------------------ geography
NUSCENES_ORIGINS = {
    # the nuScenes devkit's reference coordinates (nuscenes/scripts/export_poses.py): the south-western corner of
    # each map, where its global frame's origin is; approximate (the devkit's own words: assumptions, no accuracy)
    'boston-seaport': (42.336849169438615, -71.05785369873047),
    'singapore-onenorth': (1.2882100868743724, 103.78475189208984),
    'singapore-hollandvillage': (1.2993652317780957, 103.78217697143555),
    'singapore-queenstown': (1.2782562240223188, 103.76741409301758),
}
EARTH_RADIUS_M = 6.378137e6


def nuscenes_latlon(location: str, x, y):
    """(lat, lon) of nuScenes global (x east, y north) metres: the devkit's method exactly (export_poses.py: a great
    circle from the map's reference corner, bearing atan(x / y), distance hypot(x, y))."""
    if location not in NUSCENES_ORIGINS:
        raise LogError(f'no geographic reference for the nuScenes map {location!r}')
    lat0, lon0 = (math.radians(v) for v in NUSCENES_ORIGINS[location])
    x, y = np.asarray(x, float), np.asarray(y, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        bearing = np.arctan(x / y)
    bearing = np.where(y == 0, np.sign(x) * math.pi / 2, bearing)
    ad = np.hypot(x, y) / EARTH_RADIUS_M
    lat = np.arcsin(math.sin(lat0) * np.cos(ad) + math.cos(lat0) * np.sin(ad) * np.cos(bearing))
    lon = lon0 + np.arctan2(np.sin(bearing) * np.sin(ad) * math.cos(lat0), np.cos(ad) - math.sin(lat0) * np.sin(lat))
    return np.degrees(lat), np.degrees(lon)


def latlon_of(geo: dict, x, y):
    """(lat, lon) of a log's CARLA-axes coordinates (x east, −y north) under its geo block."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if geo.get('method') == 'nuscenes_devkit':
        return nuscenes_latlon(geo['location'], x, -y)
    from ..osm import local_projection
    return local_projection(geo['lat0'], geo['lon0']).to_geo(x, -y)


# ------------------------------------------------------------------------------------------------ tracks
class Track:
    """One object's samples in CARLA's frame. t (s), x, y, z (m), yaw (degrees) are numpy arrays of equal length,
    time increasing; pitch and roll optional."""

    def __init__(self, ident, cls, t, x, y, z, yaw, size=None, source_class='', size_source='source',
                 max_gap_s=None, pitch=None, roll=None, hold=(False, False), blueprint=None):
        self.id = str(ident)
        if cls not in CLASSES:
            raise LogError(f'track {ident}: class {cls!r} is not one of {", ".join(CLASSES)}')
        self.cls, self.source_class = cls, str(source_class or cls)
        order = np.argsort(np.asarray(t, float), kind='stable')
        self.t = np.asarray(t, float)[order]
        self.x, self.y, self.z = (np.asarray(v, float)[order] for v in (x, y, z))
        self.yaw = np.asarray(yaw, float)[order]
        self.pitch = None if pitch is None else np.asarray(pitch, float)[order]
        self.roll = None if roll is None else np.asarray(roll, float)[order]
        if len({len(self.t), len(self.x), len(self.y), len(self.z), len(self.yaw)}) != 1 or not len(self.t):
            raise LogError(f'track {ident}: its samples are empty or of unequal lengths')
        if size is None:
            size, size_source = TYPICAL_SIZE[cls], 'typical'
        self.size = tuple(float(v) for v in size)
        self.size_source = size_source
        dt = np.diff(self.t)
        med = float(np.median(dt)) if len(dt) else 0.1
        self.max_gap_s = float(max_gap_s) if max_gap_s is not None else max(3.0 * med, 0.15)
        self.hold = (bool(hold[0]), bool(hold[1]))
        self.blueprint = blueprint

    def __len__(self):
        return len(self.t)

    @property
    def span(self) -> tuple:
        return float(self.t[0]), float(self.t[-1])

    def copy(self) -> 'Track':
        return Track(self.id, self.cls, self.t.copy(), self.x.copy(), self.y.copy(), self.z.copy(), self.yaw.copy(),
                     self.size, self.source_class, self.size_source, self.max_gap_s,
                     None if self.pitch is None else self.pitch.copy(), None if self.roll is None else self.roll.copy(),
                     self.hold, self.blueprint)

    def sample(self, tq) -> tuple:
        """Poses at times tq: (valid mask, x, y, z, yaw). Linear between consecutive samples at most max_gap_s apart,
        yaw along the shorter arc; outside the samples absent unless held."""
        tq = np.atleast_1d(np.asarray(tq, float))
        n = len(self.t)
        x, y, z, yaw = (np.full(len(tq), np.nan) for _ in range(4))
        valid = np.zeros(len(tq), bool)
        if n == 1:
            hit = np.abs(tq - self.t[0]) <= 1e-9
            valid |= hit
            k0 = np.zeros(len(tq), int)
            f = np.zeros(len(tq))
            k1 = k0
        else:
            k0 = np.clip(np.searchsorted(self.t, tq, side='right') - 1, 0, n - 2)
            k1 = k0 + 1
            gap = self.t[k1] - self.t[k0]
            inside = (tq >= self.t[k0] - 1e-9) & (tq <= self.t[k1] + 1e-9)
            valid |= inside & (gap <= self.max_gap_s + 1e-9)
            f = np.clip((tq - self.t[k0]) / np.maximum(gap, 1e-12), 0.0, 1.0)
        dyaw = wrap_deg(self.yaw[k1] - self.yaw[k0])
        x = self.x[k0] + (self.x[k1] - self.x[k0]) * f
        y = self.y[k0] + (self.y[k1] - self.y[k0]) * f
        z = self.z[k0] + (self.z[k1] - self.z[k0]) * f
        yaw = wrap_deg(self.yaw[k0] + dyaw * f)
        before, after = tq < self.t[0] - 1e-9, tq > self.t[-1] + 1e-9
        if self.hold[0]:
            x, y, z, yaw = (np.where(before, v[0], a) for v, a in ((self.x, x), (self.y, y), (self.z, z),
                                                                     (self.yaw, yaw)))
            valid |= before
        if self.hold[1]:
            x, y, z, yaw = (np.where(after, v[-1], a) for v, a in ((self.x, x), (self.y, y), (self.z, z),
                                                                     (self.yaw, yaw)))
            valid |= after
        if not self.hold[0]:
            valid &= ~before
        if not self.hold[1]:
            valid &= ~after
        return valid, x, y, z, yaw

    def path_length(self) -> float:
        return float(np.sum(np.hypot(np.diff(self.x), np.diff(self.y))))

    def to_doc(self) -> dict:
        d = {'id': self.id, 'class': self.cls, 'source_class': self.source_class,
             'size': [round(v, 4) for v in self.size], 'size_source': self.size_source,
             'max_gap_s': round(self.max_gap_s, 6), 'hold': list(self.hold),
             't': _r(self.t, 6), 'x': _r(self.x, 6), 'y': _r(self.y, 6), 'z': _r(self.z, 6), 'yaw': _r(self.yaw, 6)}
        if self.pitch is not None:
            d['pitch'] = _r(self.pitch, 6)
        if self.roll is not None:
            d['roll'] = _r(self.roll, 6)
        if self.blueprint:
            d['blueprint'] = self.blueprint
        return d

    @classmethod
    def from_doc(cls, d: dict) -> 'Track':
        return cls(d['id'], d['class'], d['t'], d['x'], d['y'], d['z'], d['yaw'], d.get('size'),
                   d.get('source_class', ''), d.get('size_source', 'source'), d.get('max_gap_s'), d.get('pitch'),
                   d.get('roll'), tuple(d.get('hold') or (False, False)), d.get('blueprint'))


def _r(a, n) -> list:
    return [round(float(v), n) + 0.0 for v in a]


class Log:
    """An imported log. meta: name, source, map, notes, left_out, heuristic, provenance."""

    def __init__(self, ego: Track, actors: list, meta: dict | None = None):
        self.ego, self.actors = ego, list(actors)
        self.meta = dict(meta or {})
        ids = [a.id for a in self.actors]
        if len(set(ids)) != len(ids) or 'ego' in ids:
            raise LogError('actor ids must be unique and not "ego"')

    @property
    def map(self) -> dict:
        return self.meta.get('map') or {'kind': 'unknown'}

    @property
    def span(self) -> tuple:
        return self.ego.span

    def actor(self, ident: str) -> Track:
        for a in self.actors:
            if a.id == str(ident):
                return a
        if str(ident) == 'ego':
            return self.ego
        raise LogError(f'the log has no actor {ident!r}')

    def counts(self) -> dict:
        c = {}
        for a in self.actors:
            c[a.cls] = c.get(a.cls, 0) + 1
        return {'actors': len(self.actors), 'by_class': dict(sorted(c.items())),
                'ego_samples': len(self.ego), 'actor_samples': int(sum(len(a) for a in self.actors)),
                'duration_s': round(self.ego.span[1] - self.ego.span[0], 6),
                'ego_path_m': round(self.ego.path_length(), 3)}

    def copy(self) -> 'Log':
        import copy
        return Log(self.ego.copy(), [a.copy() for a in self.actors], copy.deepcopy(self.meta))

    def to_doc(self) -> dict:
        doc = {'schema': LOG_SCHEMA,
               'frame': 'CARLA (x forward/east, y right/south, z up; left-handed), metres, degrees; t in seconds '
                        'from the log\'s first ego sample'}
        doc.update({k: v for k, v in self.meta.items() if k not in ('ego', 'actors')})
        doc['counts'] = self.counts()
        doc['ego'] = self.ego.to_doc()
        doc['actors'] = [a.to_doc() for a in self.actors]
        return doc

    @classmethod
    def from_doc(cls, doc: dict) -> 'Log':
        if doc.get('schema') != LOG_SCHEMA:
            raise LogError(f'not a log (schema {doc.get("schema")!r}, expected {LOG_SCHEMA})')
        meta = {k: v for k, v in doc.items() if k not in ('ego', 'actors', 'schema', 'counts', 'frame')}
        return cls(Track.from_doc(doc['ego']), [Track.from_doc(a) for a in doc['actors']], meta)

    def save(self, path: str) -> str:
        """Write log.json (path: the file, or a folder to hold it). Returns the file's path."""
        import json
        from ..paths import atomic_write
        if not path.endswith('.json'):
            os.makedirs(path, exist_ok=True)
            path = os.path.join(path, 'log.json')
        atomic_write(path, json.dumps(self.to_doc(), separators=(',', ':'), ensure_ascii=False) + '\n')
        return path

    @classmethod
    def load(cls, path: str) -> 'Log':
        import json
        if os.path.isdir(path):
            path = os.path.join(path, 'log.json')
        with open(path, encoding='utf-8') as fh:
            return cls.from_doc(json.load(fh))


def rebase_time(ego: Track, actors: list, t0: float | None = None):
    """Shift every track's time so the ego's first sample is t = 0 (or t0 is); returns the shift (s)."""
    t0 = float(ego.t[0]) if t0 is None else float(t0)
    for tr in [ego] + list(actors):
        tr.t = tr.t - t0
    return t0


def finite_velocity(t, x, y) -> tuple:
    """(vx, vy) by central differences (one-sided at the ends); zeros for a single sample."""
    t, x, y = (np.asarray(v, float) for v in (t, x, y))
    if len(t) < 2:
        return np.zeros(len(t)), np.zeros(len(t))
    return np.gradient(x, t), np.gradient(y, t)
