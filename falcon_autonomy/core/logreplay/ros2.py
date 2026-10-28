"""ROS 2 bags as logs: rosbag2 folders, MCAP files and sqlite3 (.db3) files, decoded by this package's own CDR decoder
(cdr.py; the mcap package only reads the MCAP container).

What is read:

* **The ego's trajectory**, from the first source found of: tf2_msgs/TFMessage on /tf and /tf_static (the chain from
  the world frame — 'map', else 'odom', else the tree's root, or the one given — to the ego frame — 'base_link',
  'base_footprint', 'ego_vehicle' or 'hero', or the one given; sampled at the times of the chain's dynamic transform
  nearest the ego, the other transforms interpolated: translation linearly, rotation by slerp); nav_msgs/Odometry (the
  pose of child_frame_id in header.frame_id); geometry_msgs/PoseStamped or PoseWithCovarianceStamped; or, when none
  of those is there, sensor_msgs/NavSatFix (positions in a transverse Mercator about the first fix, the heading from
  the direction of motion over at least 0.5 m — heuristic, and labelled so).
* **The actors**, from vision_msgs/Detection3DArray (id: Detection3D.id, or tracking_id in older versions; class:
  the best hypothesis' class_id, or id in vision_msgs 3; the box: bbox.center and bbox.size, x along the box's
  heading) and autoware_perception_msgs/TrackedObjects or autoware_auto_perception_msgs/TrackedObjects (id: the
  object_id UUID, or the integer id; class: the most probable ObjectClassification label; the box: the kinematics'
  pose and shape.dimensions, a polygon's footprint bounding box). Each box is taken from its header's frame to the
  world frame through the TF tree at its stamp (or through the ego's own trajectory when its frame is the ego frame).
  Detections without an id cannot be followed in time: they are counted and left out. Classes that name none of
  Falcon's (models.CLASSES) are left out and counted, unless the caller asks to classify them by size (heuristic).
* **Times**: header stamps (the sensor's time), not the bag's log times; t = 0 at the ego's first sample.

Frames: ROS's map and odom frames are right-handed (REP 103 / 105); poses go to CARLA's frame through S = diag(1, −1,
1) (model.rh_to_carla). A bag exported by Falcon (core.exporters 'mcap') round-trips exactly: its map frame is CARLA's
world mirrored, so the import gives CARLA's world coordinates back; with the export's `objects` option its
/falcon/objects topic carries every actor's box (Detection3D id = the CARLA actor id; the ego's own box as id 'ego').

Definitions: an MCAP file's ros2msg schemas; a .db3's `message_definitions` table (rosbag2 Iron and later), else
cdr.BUILTIN for the standard types (a topic of another type without a stored definition is refused, saying so).
Encodings other than CDR, and ros2idl schemas, are refused.
"""
from __future__ import annotations

import glob
import math
import os

import numpy as np

from . import cdr
from .model import CLASSES, Log, LogError, Track, quat_matrix, rebase_time, rh_to_carla

EGO_FRAMES = ('base_link', 'base_footprint', 'ego_vehicle', 'hero')
WORLD_FRAMES = ('map', 'world', 'odom')
TF_TYPE = 'tf2_msgs/msg/TFMessage'
DET_TYPES = ('vision_msgs/msg/Detection3DArray',)
TRACK_TYPES = ('autoware_perception_msgs/msg/TrackedObjects', 'autoware_auto_perception_msgs/msg/TrackedObjects')
ODOM_TYPE = 'nav_msgs/msg/Odometry'
POSE_TYPES = ('geometry_msgs/msg/PoseStamped', 'geometry_msgs/msg/PoseWithCovarianceStamped')
FIX_TYPE = 'sensor_msgs/msg/NavSatFix'
AUTOWARE_LABELS = {0: None, 1: 'car', 2: 'truck', 3: 'bus', 4: 'truck', 5: 'motorcycle', 6: 'bicycle', 7: 'pedestrian'}


# ------------------------------------------------------------------------------------------------ the container
class Bag:
    """The topics of a bag and its messages, raw. path: a rosbag2 folder, an .mcap or a .db3."""

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        if os.path.isdir(self.path):
            files = sorted(glob.glob(os.path.join(self.path, '*.mcap'))) or \
                sorted(glob.glob(os.path.join(self.path, '*.db3')))
            if not files:
                raise LogError(f'{path}: no .mcap or .db3 file in the folder')
        elif os.path.isfile(self.path):
            files = [self.path]
        else:
            raise LogError(f'{path}: no such file or folder')
        self.files = files
        self.storage = 'mcap' if files[0].endswith('.mcap') else 'sqlite3'
        self.decoder = cdr.Decoder()
        self.topics: dict[str, str] = {}
        self.counts: dict[str, int] = {}
        self.refused: dict[str, str] = {}
        (self._scan_mcap if self.storage == 'mcap' else self._scan_db3)()

    # ---------------------------------------------------------------- MCAP
    def _scan_mcap(self):
        from mcap.reader import make_reader
        for f in self.files:
            with open(f, 'rb') as fh:
                r = make_reader(fh)
                summ = r.get_summary()
                if summ is None:
                    chans = {}
                    for sch, ch, _ in r.iter_messages():
                        chans[ch.id] = (sch, ch)
                    items = list(chans.values())
                    stats = {}
                else:
                    items = [(summ.schemas.get(c.schema_id), c) for c in summ.channels.values()]
                    stats = summ.statistics.channel_message_counts if summ.statistics else {}
                for sch, ch in items:
                    if ch.message_encoding != 'cdr':
                        self.refused[ch.topic] = f'message encoding {ch.message_encoding!r}, not cdr'
                        continue
                    if sch is None or sch.encoding != 'ros2msg':
                        self.refused[ch.topic] = f'schema encoding {getattr(sch, "encoding", None)!r}, not ros2msg'
                        continue
                    try:
                        self.decoder.add_schema(sch.name, sch.data.decode('utf-8'))
                    except (cdr.CdrError, UnicodeDecodeError) as e:
                        self.refused[ch.topic] = f'its schema cannot be read: {e}'
                        continue
                    self.topics[ch.topic] = cdr.norm_type(sch.name)
                    self.counts[ch.topic] = self.counts.get(ch.topic, 0) + int(stats.get(ch.id, 0))

    def _mcap_messages(self, topics):
        from mcap.reader import make_reader
        for f in self.files:
            with open(f, 'rb') as fh:
                for _sch, ch, msg in make_reader(fh).iter_messages(topics=list(topics)):
                    yield ch.topic, msg.log_time, msg.data

    # ---------------------------------------------------------------- sqlite3
    def _scan_db3(self):
        import sqlite3
        for f in self.files:
            con = sqlite3.connect(f'file:{f}?mode=ro', uri=True)
            try:
                tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if not {'topics', 'messages'} <= tables:
                    raise LogError(f'{f}: not a rosbag2 sqlite3 file (no topics / messages tables)')
                defs = {}
                if 'message_definitions' in tables:
                    for t, enc, text in con.execute('SELECT topic_type, encoding, encoded_message_definition '
                                                    'FROM message_definitions'):
                        if enc == 'ros2msg' and text:
                            defs[cdr.norm_type(t)] = text
                for _tid, name, typ, fmt in con.execute('SELECT id, name, type, serialization_format FROM topics'):
                    t = cdr.norm_type(typ)
                    if fmt != 'cdr':
                        self.refused[name] = f'serialization format {fmt!r}, not cdr'
                        continue
                    if t in defs:
                        self.decoder.add_schema(t, defs[t])
                    if not self.decoder.has(t):
                        self.refused[name] = (f'the bag stores no definition of {t} (rosbag2 before Iron) and it '
                                              'is not one of the built-in types')
                        continue
                    self.topics[name] = t
                for name, n in con.execute('SELECT t.name, COUNT(m.id) FROM topics t JOIN messages m '
                                           'ON m.topic_id = t.id GROUP BY t.name'):
                    if name in self.topics:
                        self.counts[name] = self.counts.get(name, 0) + int(n)
            finally:
                con.close()

    def _db3_messages(self, topics):
        import sqlite3
        want = set(topics)
        for f in self.files:
            con = sqlite3.connect(f'file:{f}?mode=ro', uri=True)
            try:
                for name, ts, data in con.execute('SELECT t.name, m.timestamp, m.data FROM messages m JOIN topics t '
                                                  'ON m.topic_id = t.id ORDER BY m.timestamp'):
                    if name in want:
                        yield name, int(ts), bytes(data)
            finally:
                con.close()

    # ---------------------------------------------------------------- reading
    def of_type(self, *types) -> list:
        return sorted(tp for tp, t in self.topics.items() if t in types)

    def messages(self, topics):
        """(topic, decoded message, log time ns) of the given topics, in file order."""
        gen = self._mcap_messages(topics) if self.storage == 'mcap' else self._db3_messages(topics)
        for topic, t_ns, data in gen:
            yield topic, self.decoder.decode(self.topics[topic], data), int(t_ns)


class Clock:
    """Header stamps (or, without one, the bag's log time) as seconds from a whole second at or before the first
    one: float seconds since 1970 would keep only about 0.2 µs."""

    def __init__(self):
        self.base = None

    def __call__(self, header: dict, fallback_ns: int) -> float:
        st = (header or {}).get('stamp') or {}
        ns = int(st['sec']) * 1_000_000_000 + int(st['nanosec']) if st.get('sec') or st.get('nanosec') \
            else int(fallback_ns)
        if self.base is None:
            self.base = (ns // 1_000_000_000) * 1_000_000_000
        return (ns - self.base) * 1e-9


def pose_matrix(p: dict) -> np.ndarray:
    """4 × 4 of a geometry_msgs Pose (position + orientation) or Transform (translation + rotation)."""
    pos = p.get('position') or p.get('translation') or {}
    q = p.get('orientation') or p.get('rotation') or {}
    T = np.eye(4)
    T[:3, :3] = quat_matrix(q.get('w', 1.0), q.get('x', 0.0), q.get('y', 0.0), q.get('z', 0.0))
    T[:3, 3] = [pos.get('x', 0.0), pos.get('y', 0.0), pos.get('z', 0.0)]
    return T


# ------------------------------------------------------------------------------------------------ TF
def _quat_of(R):
    from ..rig import quat_of
    return np.array(quat_of(R), float)          # w, x, y, z


def _slerp(q0, q1, f):
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + f * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(min(d, 1.0))
    return (math.sin((1 - f) * th) * q0 + math.sin(f * th) * q1) / math.sin(th)


class TfBuffer:
    """Every transform of a bag: static edges, and dynamic ones as time series. lookup() composes the chain."""

    def __init__(self):
        self.static: dict = {}
        self.dyn: dict = {}
        self.parent: dict = {}

    def add(self, tf: dict, t: float, static: bool):
        parent = tf['header']['frame_id'].lstrip('/')
        child = tf['child_frame_id'].lstrip('/')
        T = pose_matrix(tf['transform'])
        self.parent[child] = parent
        if static:
            self.static[(parent, child)] = T
        else:
            self.dyn.setdefault((parent, child), []).append((t, T))

    def finish(self):
        out = {}
        for k, rows in self.dyn.items():
            rows.sort(key=lambda r: r[0])
            t = np.array([r[0] for r in rows])
            keep = np.r_[True, np.diff(t) > 0]
            out[k] = (t[keep], np.array([r[1] for r in rows])[keep])
        self.dyn = out

    def frames(self) -> set:
        return set(self.parent) | set(self.parent.values())

    def chain(self, root: str, frame: str):
        """The edges (parent, child) from root down to frame, or None."""
        edges, cur = [], frame
        for _ in range(64):
            if cur == root:
                return list(reversed(edges))
            p = self.parent.get(cur)
            if p is None:
                return None
            edges.append((p, cur))
            cur = p
        return None

    def root_of(self, frame: str) -> str:
        cur = frame
        for _ in range(64):
            p = self.parent.get(cur)
            if p is None:
                return cur
            cur = p
        return cur

    def edge(self, e, t: float, tol: float):
        if e in self.static:
            return self.static[e]
        ts, Ts = self.dyn[e]
        if len(ts) == 1:
            return Ts[0] if abs(t - ts[0]) <= tol else None
        k = int(np.clip(np.searchsorted(ts, t, side='right') - 1, 0, len(ts) - 2))
        t0, t1 = ts[k], ts[k + 1]
        if t < t0 - tol or t > t1 + tol:
            return None
        if t <= t0:
            return Ts[k]
        if t >= t1:
            return Ts[k + 1]
        gap = t1 - t0
        if gap > max(10 * tol, 1.0):
            return None
        f = (t - t0) / gap
        T = np.eye(4)
        T[:3, 3] = Ts[k][:3, 3] + f * (Ts[k + 1][:3, 3] - Ts[k][:3, 3])
        q = _slerp(_quat_of(Ts[k][:3, :3]), _quat_of(Ts[k + 1][:3, :3]), f)
        T[:3, :3] = quat_matrix(*q)
        return T

    def lookup(self, root: str, frame: str, t: float, tol: float = 0.05):
        """The pose of frame in root at time t (4 × 4), or None."""
        if frame == root:
            return np.eye(4)
        ch = self.chain(root, frame)
        if ch is None:
            return None
        T = np.eye(4)
        for e in ch:
            M = self.edge(e, t, tol)
            if M is None:
                return None
            T = T @ M
        return T


# ------------------------------------------------------------------------------------------------ classes
def falcon_class(name) -> str | None:
    """A Falcon class (model.CLASSES) for a source's class name (nuScenes categories, vision_msgs class ids, common
    names), or None when it names none of them."""
    s = str(name or '').lower()
    if not s or s.startswith(('static', 'movable_object', 'animal', 'flat', 'noise')):
        return None                 # nuScenes' static objects, cones, barriers, debris, racks and animals: no actor
    if s in CLASSES:
        return s
    if any(k in s for k in ('pedestrian', 'person', 'walker', 'human', 'child', 'adult')):
        return 'pedestrian'
    if 'motor' in s or 'scooter' in s or 'moped' in s:
        return 'motorcycle'
    if any(k in s for k in ('bicycle', 'cyclist', 'bike')):
        return 'bicycle'
    if 'bus' in s:
        return 'bus'
    if any(k in s for k in ('truck', 'trailer', 'construction', 'lorry', 'hgv')):
        return 'truck'
    if any(k in s for k in ('car', 'vehicle', 'van', 'suv', 'taxi', 'police', 'ambulance', 'emergency', 'sedan')):
        return 'car'
    return None


def class_by_size(length: float) -> str:
    """Heuristic: a class from a box's length alone (only when the caller asks for it)."""
    if length < 1.0:
        return 'pedestrian'
    if length < 2.6:
        return 'motorcycle'
    if length < 6.5:
        return 'car'
    return 'truck'


# ------------------------------------------------------------------------------------------------ the import
def _box_of_detection(d: dict, parent_header: dict):
    hdr = d.get('header') or {}
    if not hdr.get('frame_id'):
        hdr = parent_header
    ident = d.get('id') or d.get('tracking_id') or ''
    res = d.get('results') or []
    cls, score = '', -1.0
    for r in res:
        h = r.get('hypothesis') or r
        s = float(h.get('score', 0.0))
        if s > score:
            cls, score = str(h.get('class_id', h.get('id', ''))), s
    bb = d.get('bbox') or {}
    size = bb.get('size') or {}
    return str(ident), cls, hdr, pose_matrix(bb.get('center') or {}), (size.get('x', 0.0), size.get('y', 0.0),
                                                                      size.get('z', 0.0))


def _box_of_tracked(o: dict):
    oid = o.get('object_id')
    if isinstance(oid, dict):
        u = oid.get('uuid', b'')
        ident = bytes(u).hex() if isinstance(u, (bytes, bytearray)) else ''.join(f'{int(v):02x}' for v in u)
    else:
        ident = str(oid)
    best, p = None, -1.0
    for c in o.get('classification') or []:
        if float(c.get('probability', 0.0)) > p:
            best, p = int(c.get('label', 0)), float(c.get('probability', 0.0))
    kin = o.get('kinematics') or {}
    pose = (kin.get('pose_with_covariance') or {}).get('pose') or kin.get('pose') or {}
    shape = o.get('shape')
    if isinstance(shape, list):
        shape = shape[0] if shape else {}
    shape = shape or {}
    dim = shape.get('dimensions') or {}
    L, W, H = float(dim.get('x', 0.0)), float(dim.get('y', 0.0)), float(dim.get('z', 0.0))
    pts = (shape.get('footprint') or {}).get('points') or []
    if int(shape.get('type', 0)) == 2 and pts:
        xs, ys = [q['x'] for q in pts], [q['y'] for q in pts]
        L, W = max(xs) - min(xs), max(ys) - min(ys)
    return ident, best, pose_matrix(pose), (L, W, H)


def import_bag(path: str, world_frame: str | None = None, ego_frame: str | None = None, topics: list | None = None,
               unknown_by_size: bool = False, map_hint: dict | None = None) -> Log:
    """A ROS 2 bag as a Log (see the module docstring). topics: restrict the object topics read; map_hint: the
    log's map block when the caller knows it (a Falcon export's town)."""
    bag = Bag(path)
    stamp_s = Clock()
    notes, left, heur = [], [], []
    for tp, why in sorted(bag.refused.items()):
        left.append({'what': f'topic {tp}', 'why': why})
    tf = TfBuffer()
    tf_topics = [tp for tp in bag.of_type(TF_TYPE)]
    for tp, m, t_ns in bag.messages(tf_topics):
        static = tp.endswith('tf_static')
        for x in m.get('transforms') or []:
            tf.add(x, stamp_s(x.get('header'), t_ns), static)
    tf.finish()
    frames = tf.frames()
    ego_f = ego_frame or next((f for f in EGO_FRAMES if f in frames), None)
    world_f = world_frame
    ego_samples = None                                  # (t, 4 × 4 in world_f)
    source = None
    if ego_f is not None and ego_f in frames:
        root = tf.root_of(ego_f)
        if world_f is None:
            world_f = next((w for w in WORLD_FRAMES if tf.chain(w, ego_f) is not None), root)
        ch = tf.chain(world_f, ego_f)
        dyn = [e for e in (ch or []) if e in tf.dyn]
        if ch and dyn:
            ts = tf.dyn[dyn[-1]][0]
            rows = [(t, tf.lookup(world_f, ego_f, t)) for t in ts]
            rows = [(t, T) for t, T in rows if T is not None]
            if rows:
                ego_samples = rows
                source = f'/tf: {" -> ".join([world_f] + [c for _, c in ch])}'
    if ego_samples is None:
        for typ in (ODOM_TYPE,) + POSE_TYPES:
            tps = bag.of_type(typ)
            if not tps:
                continue
            rows = []
            for _tp, m, t_ns in bag.messages(tps[:1]):
                p = m['pose']['pose'] if typ in (ODOM_TYPE, POSE_TYPES[1]) else m['pose']
                world_f = world_f or m['header']['frame_id'].lstrip('/')
                rows.append((stamp_s(m['header'], t_ns), pose_matrix(p)))
            if rows:
                ego_samples, source = rows, f'{tps[0]} ({typ})'
                if typ == ODOM_TYPE:
                    notes.append('ego from nav_msgs/Odometry: the pose of its child frame in its header frame')
                break
    geo = None
    if ego_samples is None and bag.of_type(FIX_TYPE):
        tp = bag.of_type(FIX_TYPE)[0]
        fixes = [(stamp_s(m['header'], t_ns), m['latitude'], m['longitude'], m['altitude'])
                 for _, m, t_ns in bag.messages([tp]) if m['status']['status'] >= 0 and math.isfinite(m['latitude'])]
        if len(fixes) >= 2:
            from ..osm import local_projection
            f = np.array(fixes)
            lat0, lon0 = float(f[0, 1]), float(f[0, 2])
            e, n = local_projection(lat0, lon0).to_local(f[:, 1], f[:, 2])
            yaw = _heading_from_motion(e, n)
            ego_samples = []
            for k in range(len(f)):
                T = np.eye(4)
                c, s = math.cos(yaw[k]), math.sin(yaw[k])
                T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
                T[:3, 3] = [e[k], n[k], f[k, 3] - f[0, 3]]
                ego_samples.append((float(f[k, 0]), T))
            geo = {'lat0': lat0, 'lon0': lon0, 'method': 'tmerc', 'source': f'{tp} (the first fix)'}
            source = f'{tp} (sensor_msgs/NavSatFix)'
            world_f = 'enu'
            heur.append({'what': 'ego heading', 'how': 'from the direction of motion over at least 0.5 m (NavSatFix '
                                                       'carries no orientation); held while slower'})
            heur.append({'what': 'ego height', 'how': 'NavSatFix altitude relative to the first fix'})
    if ego_samples is None:
        raise LogError(f'{path}: no ego trajectory — no TF chain to {", ".join(EGO_FRAMES)} (or the frame given), no '
                       'nav_msgs/Odometry, no PoseStamped and no NavSatFix topic '
                       f'(topics: {", ".join(f"{k} [{v}]" for k, v in sorted(bag.topics.items())) or "none"})')
    ego_samples.sort(key=lambda r: r[0])
    et = np.array([r[0] for r in ego_samples])
    epose = np.array([rh_to_carla(T[:3, :3], T[:3, 3]) for _, T in ego_samples])

    def ego_at(t):
        """The ego's 4 × 4 (right-handed, world frame) at t, interpolated; None outside its samples."""
        if t < et[0] - 0.05 or t > et[-1] + 0.05:
            return None
        k = int(np.clip(np.searchsorted(et, t, side='right') - 1, 0, max(len(et) - 2, 0)))
        if len(et) == 1:
            return ego_samples[0][1]
        t0, t1 = et[k], et[k + 1]
        f = float(np.clip((t - t0) / max(t1 - t0, 1e-12), 0.0, 1.0))
        A, B = ego_samples[k][1], ego_samples[k + 1][1]
        T = np.eye(4)
        T[:3, 3] = A[:3, 3] + f * (B[:3, 3] - A[:3, 3])
        T[:3, :3] = quat_matrix(*_slerp(_quat_of(A[:3, :3]), _quat_of(B[:3, :3]), f))
        return T

    def to_world(frame: str, t: float):
        frame = (frame or '').lstrip('/')
        if not frame or frame == world_f:
            return np.eye(4)
        if frame in frames and world_f in frames:
            T = tf.lookup(world_f, frame, t)
            if T is not None:
                return T
        if frame == ego_f:
            return ego_at(t)
        return None

    obj_topics = bag.of_type(*DET_TYPES) + bag.of_type(*TRACK_TYPES)
    if topics:
        obj_topics = [tp for tp in obj_topics if tp in topics]
    tracks: dict = {}
    counts = {'no_id': 0, 'no_frame': 0, 'unknown_class': {}, 'by_size': 0}
    ego_size = None
    for tp, m, t_ns in bag.messages(obj_topics):
        typ = bag.topics[tp]
        hdr = m.get('header') or {}
        boxes = []
        if typ in DET_TYPES:
            for d in m.get('detections') or []:
                ident, cls, h, P, size = _box_of_detection(d, hdr)
                boxes.append((ident, cls, h, P, size, cls))
        else:
            for o in m.get('objects') or []:
                ident, label, P, size = _box_of_tracked(o)
                boxes.append((ident, AUTOWARE_LABELS.get(label), hdr, P, size, f'autoware label {label}'))
        for ident, cls, h, P, size, src_cls in boxes:
            t = stamp_s(h, t_ns)
            W = to_world(h.get('frame_id', ''), t)
            if W is None:
                counts['no_frame'] += 1
                continue
            if not ident:
                counts['no_id'] += 1
                continue
            if ident == 'ego':
                ego_size = size
                continue
            fc = falcon_class(cls)
            if fc is None:
                if unknown_by_size and size[0] > 0:
                    fc = class_by_size(size[0])
                    counts['by_size'] += 1
                else:
                    counts['unknown_class'][str(src_cls)] = counts['unknown_class'].get(str(src_cls), 0) + 1
                    continue
            Tw = W @ P
            x, y, z, roll, pitch, yaw = rh_to_carla(Tw[:3, :3], Tw[:3, 3])
            key = f'{tp}:{ident}' if len(obj_topics) > 1 else ident
            tracks.setdefault(key, {'cls': fc, 'src': str(src_cls), 'rows': [], 'size': []})
            tracks[key]['rows'].append((t, x, y, z, yaw, pitch, roll))
            if all(v > 0 for v in size):
                tracks[key]['size'].append(size)
    if counts['no_id']:
        left.append({'what': f'{counts["no_id"]} detections without an id', 'why': 'they cannot be followed in time'})
    if counts['no_frame']:
        left.append({'what': f'{counts["no_frame"]} boxes in frames not connected to {world_f}',
                     'why': 'no transform to the world frame at their stamp'})
    for c, n in sorted(counts['unknown_class'].items()):
        left.append({'what': f'{n} boxes of class {c!r}', 'why': 'it names none of Falcon\'s classes (ask to classify '
                                                                 'unknown classes by size)'})
    if counts['by_size']:
        heur.append({'what': f'the class of {counts["by_size"]} boxes', 'how': 'from the box length (pedestrian '
                                                                                '< 1 m < motorcycle < 2.6 m < car '
                                                                                '< 6.5 m < truck)'})
    ego = Track('ego', 'car', et, epose[:, 0], epose[:, 1], epose[:, 2], epose[:, 5],
                size=None if ego_size is None else (ego_size[0], ego_size[1], ego_size[2]),
                source_class='ego', pitch=epose[:, 4], roll=epose[:, 3])
    if ego_size is None:
        heur.append({'what': 'ego box', 'how': f'a typical car, {ego.size[0]} × {ego.size[1]} × {ego.size[2]} m '
                                               '(the bag does not say)'})
    actors = []
    for key in sorted(tracks):
        d = tracks[key]
        rows = np.array(sorted(d['rows']))
        _, uniq = np.unique(rows[:, 0], return_index=True)
        rows = rows[uniq]
        size = tuple(np.median(np.array(d['size']), axis=0)) if d['size'] else None
        actors.append(Track(key, d['cls'], rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3], rows[:, 4], size,
                            d['src'], 'source' if size else 'typical', pitch=rows[:, 5], roll=rows[:, 6]))
    t0 = rebase_time(ego, actors) + (stamp_s.base or 0) * 1e-9
    from ..provenance import sha256_file
    mp = dict(map_hint) if map_hint else ({'kind': 'geo', 'geo': geo} if geo else
                                          {'kind': 'unknown', 'note': 'a ROS bag does not say which map its frames '
                                                                      'are in'})
    if geo and not map_hint:
        mp['name'] = None
    meta = {'name': os.path.basename(os.path.normpath(path)),
            'source': {'format': 'ros2', 'storage': bag.storage, 'path': bag.path,
                       'files': [{'path': f, 'sha256': sha256_file(f)} for f in bag.files],
                       'ego': source, 'world_frame': world_f, 'ego_frame': ego_f, 'object_topics': obj_topics,
                       'topics': bag.topics, 'messages': bag.counts, 't0_s': t0},
            'ego_origin': f'the origin of {ego_f}' if ego_f and source and source.startswith('/tf') else
                          'the pose\'s origin as the bag gives it',
            'map': mp, 'notes': notes, 'left_out': left, 'heuristic': heur}
    return Log(ego, actors, meta)


def _heading_from_motion(e, n, min_m: float = 0.5) -> np.ndarray:
    """Heading (radians, counter-clockwise from east) of each sample from the displacement to the first sample at
    least min_m away (forward, else backward); held from the last known one in between."""
    e, n = np.asarray(e, float), np.asarray(n, float)
    out = np.full(len(e), np.nan)
    for i in range(len(e)):
        for j in list(range(i + 1, len(e))) + list(range(i - 1, -1, -1)):
            d = math.hypot(e[j] - e[i], n[j] - n[i])
            if d >= min_m:
                a, b = (i, j) if j > i else (j, i)
                out[i] = math.atan2(n[b] - n[a], e[b] - e[a])
                break
    if np.all(np.isnan(out)):
        return np.zeros(len(e))
    last = out[~np.isnan(out)][0]
    for i in range(len(out)):
        if np.isnan(out[i]):
            out[i] = last
        last = out[i]
    return out


# ------------------------------------------------------------------------------------------------ writing (the export)
VISION_DEFS = ('geometry_msgs/msg/Point', 'geometry_msgs/msg/Pose', 'geometry_msgs/msg/PoseWithCovariance',
               'vision_msgs/msg/ObjectHypothesis', 'vision_msgs/msg/ObjectHypothesisWithPose',
               'vision_msgs/msg/BoundingBox3D', 'vision_msgs/msg/Detection3D', 'vision_msgs/msg/Detection3DArray')
OBJECTS_TOPIC = '/falcon/objects'


def register_vision_msgs():
    """Add vision_msgs 4.x (Humble) Detection3DArray and what it needs to core.mcap_bag's definitions, so the MCAP
    export can write the actors' boxes (definitions already there are kept)."""
    from .. import mcap_bag as B
    for t in VISION_DEFS:
        B.MSG_DEFS.setdefault(t, cdr.BUILTIN[t])


def actor_rows_by_step(run_dir: str) -> dict:
    """{step: actors.jsonl rows} of a run (empty without actors.jsonl)."""
    import json
    p = os.path.join(run_dir, 'actors.jsonl')
    out = {}
    if os.path.isfile(p):
        with open(p, encoding='utf-8') as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    out[int(r['step'])] = r['actors']
    return out


def detection3d(a: dict) -> dict:
    """A vision_msgs/Detection3D of an actors.jsonl row, in the 'map' frame of Falcon's bags (CARLA's world
    mirrored): the box's centre and orientation, its full size (x along the actor's heading), id the CARLA actor id
    ('ego' for the hero), class_id its nuScenes category."""
    from ..rig import carla_rotation, quat_of
    R = carla_rotation(a['roll'], a['pitch'], a['yaw'])
    c = R @ np.asarray(a.get('bbox_center') or (0.0, 0.0, 0.0), float) + [a['x'], a['y'], a['z']]
    S = np.diag([1.0, -1.0, 1.0])
    w, x, y, z = quat_of(S @ R @ S)
    pose = {'position': {'x': float(c[0]), 'y': float(-c[1]), 'z': float(c[2])},
            'orientation': {'x': x, 'y': y, 'z': z, 'w': w}}
    ex = a.get('extent') or (0.0, 0.0, 0.0)
    return {'header': {}, 'results': [{'hypothesis': {'class_id': str(a.get('category') or ''), 'score': 1.0},
                                       'pose': {'pose': pose}}],
            'bbox': {'center': pose, 'size': {'x': 2 * ex[0], 'y': 2 * ex[1], 'z': 2 * ex[2]}},
            'id': 'ego' if a.get('role') == 'hero' else str(a['id'])}
