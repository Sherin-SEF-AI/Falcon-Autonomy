"""Log import, frames, changes, plans and map matching (core.logreplay), without CARLA.

Importers are tested on small synthetic files written in each format's documented layout (no dataset is downloaded,
and no Waymo data is copied anywhere): nuScenes JSON tables (and the nuscenes-devkit's own interpolation of the same
tables, run in .venv-devkits), Waymo v2 Parquet components, ROS 2 bags written by an independent encoder (mcap_ros2's,
in .venv-devkits) and by core.mcap_bag, in MCAP and sqlite3; KITTI raw OXTS on the real drives when they are on this
machine. Frames on known poses; the CDR decoder field by field against mcap_ros2's decoder.
"""
import json
import math
import os
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from conftest import ROOT
from falcon_autonomy.core import mcap_bag as B
from falcon_autonomy.core.logreplay import cdr, import_log
from falcon_autonomy.core.logreplay import changes as CH
from falcon_autonomy.core.logreplay import mapmatch as MM
from falcon_autonomy.core.logreplay.model import (Log, LogError, Track, carla_to_rh, latlon_of, nuscenes_latlon,
                                                  quat_matrix, rh_to_carla)
from falcon_autonomy.core.logreplay.plan import make_plan, resolve_placement
from falcon_autonomy.core.rig import carla_rotation, quat_of

VENV_PY = os.environ.get('FALCON_DEVKITS_PYTHON') or os.path.join(ROOT, '.venv-devkits', 'bin', 'python')
KITTI = os.path.expanduser('~/Desktop/LabeloxAV/.scratch/demo/kitti/2011_09_26')
S = np.diag([1.0, -1.0, 1.0])


def devkit(modules, script, *args):
    """Run a script in a Python that imports the modules (this one, else the devkit venv); skip when none does."""
    for py in (sys.executable, VENV_PY):
        if py and os.path.isfile(py) and subprocess.run([py, '-P', '-c', 'import ' + ', '.join(modules)],
                                                        capture_output=True, timeout=60).returncode == 0:
            env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHON')}
            r = subprocess.run([py, '-P', '-c', script, *map(str, args)], capture_output=True, text=True,
                               timeout=180, env=env)
            assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
            return json.loads(r.stdout.strip().splitlines()[-1])
    pytest.skip(f'{", ".join(modules)} not importable here nor in {VENV_PY}')


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


# --------------------------------------------------------------------------------------------------------- frames
def test_frames_on_known_poses():
    assert rh_to_carla(np.eye(3), (1, 2, 3)) == pytest.approx((1, -2, 3, 0, 0, 0))
    # turning left 30° (counter-clockwise, right-handed) is CARLA yaw −30° (CARLA's yaw turns towards +y, its right)
    assert rh_to_carla(rz(math.radians(30)), (0, 0, 0))[5] == pytest.approx(-30.0)
    # nose up 10° is a negative rotation about ROS's y (left) axis; CARLA's pitch is positive nose up
    assert rh_to_carla(ry(math.radians(-10)), (0, 0, 0))[4] == pytest.approx(10.0)
    # left side up (positive ROS roll) stays positive in CARLA's roll
    assert rh_to_carla(rx(math.radians(5)), (0, 0, 0))[3] == pytest.approx(5.0)


def test_frames_carry_points_consistently():
    """The defining property: a body point p of a right-handed pose (R, t) lands at S·(R p + t), and CARLA's pose of
    S·R·S puts the mirrored body point S·p exactly there — for random poses, roll and pitch included."""
    rng = np.random.default_rng(1)
    for _ in range(50):
        R = rz(rng.uniform(-3, 3)) @ ry(rng.uniform(-1.2, 1.2)) @ rx(rng.uniform(-1, 1))
        t = rng.uniform(-100, 100, 3)
        p = rng.uniform(-5, 5, 3)
        x, y, z, roll, pitch, yaw = rh_to_carla(R, t)
        got = carla_rotation(roll, pitch, yaw) @ (S @ p) + [x, y, z]
        assert got == pytest.approx(S @ (R @ p + t), abs=1e-9)
        assert carla_to_rh(x, y, z, roll, pitch, yaw)[:3, :3] == pytest.approx(R, abs=1e-9)
        w, qx, qy, qz = quat_of(R)
        assert quat_matrix(w, qx, qy, qz) == pytest.approx(R, abs=1e-12)


def test_nuscenes_geo_reference_is_the_devkits():
    xy = [(400.0, 1100.0), (1800.5, 600.25), (10.0, 2500.0)]
    lat, lon = nuscenes_latlon('boston-seaport', [p[0] for p in xy], [p[1] for p in xy])
    r = devkit(('nuscenes.scripts.export_poses',), r'''
import json, sys
from nuscenes.scripts.export_poses import derive_latlon
pts = json.loads(sys.argv[1])
out = derive_latlon('boston-seaport', [{'timestamp': 0, 'translation': [x, y, 0]} for x, y in pts])
print(json.dumps([[o['latitude'], o['longitude']] for o in out]))''', json.dumps(xy))
    assert np.column_stack([lat, lon]) == pytest.approx(np.array(r), abs=1e-12)
    with pytest.raises(LogError):
        nuscenes_latlon('nowhere', [0.0], [1.0])


# --------------------------------------------------------------------------------------------------------- CDR
AUTOWARE = {
    'autoware_perception_msgs/TrackedObjects': ('std_msgs/Header header\n'
                                                'autoware_perception_msgs/TrackedObject[] objects\n'),
    'autoware_perception_msgs/TrackedObject': ('unique_identifier_msgs/UUID object_id\nfloat32 existence_probability\n'
                                               'autoware_perception_msgs/ObjectClassification[] classification\n'
                                               'autoware_perception_msgs/TrackedObjectKinematics kinematics\n'
                                               'autoware_perception_msgs/Shape shape\n'),
    'unique_identifier_msgs/UUID': 'uint8[16] uuid\n',
    'autoware_perception_msgs/ObjectClassification': ('uint8 UNKNOWN = 0\nuint8 CAR = 1\nuint8 PEDESTRIAN = 7\n'
                                                      'uint8 label\nfloat32 probability\n'),
    'autoware_perception_msgs/TrackedObjectKinematics': (
        'uint8 UNAVAILABLE=0\ngeometry_msgs/PoseWithCovariance pose_with_covariance\n'
        'geometry_msgs/TwistWithCovariance twist_with_covariance\n'
        'geometry_msgs/AccelWithCovariance acceleration_with_covariance\nuint8 orientation_availability\n'
        'bool is_stationary\n'),
    'autoware_perception_msgs/Shape': ('uint8 BOUNDING_BOX=0\nuint8 type\ngeometry_msgs/Polygon footprint\n'
                                       'geometry_msgs/Vector3 dimensions\n'),
}


def schema_of(t: str, defs: dict) -> str:
    """The ros2msg schema text of t with every dependency (as rosbag2 writes it)."""
    seen, order = set(), []

    def walk(name):
        n = cdr.norm_type(name)
        if n in seen:
            return
        seen.add(n)
        order.append(n)
        for _, base, _ in cdr.parse_fields(n, defs[n]):
            if base not in cdr._PRIM and base != 'string':
                walk(base)
    walk(t)
    out = defs[order[0]]
    for n in order[1:]:
        p = n.split('/')
        out += '\n' + '=' * 80 + f'\nMSG: {p[0]}/{p[-1]}\n' + defs[n]
    return out


def all_defs():
    d = dict(cdr.BUILTIN)
    d.update({cdr.norm_type(k): v for k, v in AUTOWARE.items()})
    return d


def tf(parent, child, t, x, y, yaw):
    return {'header': {'stamp': {'sec': int(t), 'nanosec': int(round((t % 1) * 1e9))}, 'frame_id': parent},
            'child_frame_id': child,
            'transform': {'translation': {'x': x, 'y': y, 'z': 0.0},
                          'rotation': {'x': 0.0, 'y': 0.0, 'z': math.sin(yaw / 2), 'w': math.cos(yaw / 2)}}}


def hdr(t, frame):
    return {'stamp': {'sec': int(t), 'nanosec': int(round((t % 1) * 1e9))}, 'frame_id': frame}


T0 = 1_700_000_000.0


def scene_messages():
    """A drive in ROS terms: map → odom static (10, 5), turned 90°; odom → base_link moving 5 m/s along odom x;
    a car detected 10 m ahead in base_link (vision_msgs, id 'a'), an unknown 'sign' and one without an id; a
    pedestrian walking +x at 1 m/s in the map frame (autoware TrackedObjects); odometry and pose topics too."""
    msgs = []
    msgs.append(('/tf_static', 'tf2_msgs/msg/TFMessage', T0, {'transforms': [tf('map', 'odom', T0, 10, 5,
                                                                                math.pi / 2)]}))
    for k in range(31):
        t = T0 + 0.1 * k
        msgs.append(('/tf', 'tf2_msgs/msg/TFMessage', t, {'transforms': [tf('odom', 'base_link', t, 0.5 * k, 0, 0)]}))
        pose = {'position': {'x': 0.5 * k, 'y': 0.0, 'z': 0.0}, 'orientation': {'x': 0, 'y': 0, 'z': 0, 'w': 1.0}}
        msgs.append(('/odom', 'nav_msgs/msg/Odometry', t, {'header': hdr(t, 'odom'), 'child_frame_id': 'base_link',
                                                           'pose': {'pose': pose, 'covariance': [0.0] * 36},
                                                           'twist': {'twist': {'linear': {'x': 5.0, 'y': 0, 'z': 0},
                                                                               'angular': {'x': 0, 'y': 0, 'z': 0}},
                                                                     'covariance': [0.0] * 36}}))
        msgs.append(('/pose', 'geometry_msgs/msg/PoseStamped', t, {'header': hdr(t, 'odom'), 'pose': pose}))
        det = {'header': hdr(t, 'base_link'),
               'results': [{'hypothesis': {'class_id': 'car', 'score': 0.9},
                            'pose': {'pose': {'position': {'x': 10.0, 'y': 0, 'z': 0.75},
                                              'orientation': {'x': 0, 'y': 0, 'z': 0, 'w': 1.0}},
                                     'covariance': [0.0] * 36}}],
               'bbox': {'center': {'position': {'x': 10.0, 'y': 0.0, 'z': 0.75},
                                   'orientation': {'x': 0, 'y': 0, 'z': 0, 'w': 1.0}},
                        'size': {'x': 4.5, 'y': 1.9, 'z': 1.5}}, 'id': 'a'}
        sign = dict(det, id='s', results=[{'hypothesis': {'class_id': 'traffic_sign', 'score': 1.0},
                                           'pose': det['results'][0]['pose']}])
        anon = dict(det, id='')
        msgs.append(('/detections', 'vision_msgs/msg/Detection3DArray', t,
                     {'header': hdr(t, 'base_link'), 'detections': [det, sign, anon]}))
        ped = {'object_id': {'uuid': list(range(16))}, 'existence_probability': 1.0,
               'classification': [{'label': 1, 'probability': 0.2}, {'label': 7, 'probability': 0.8}],
               'kinematics': {'pose_with_covariance': {'pose': {'position': {'x': 0.1 * k, 'y': 0.0, 'z': 0.9},
                                                                'orientation': {'x': 0, 'y': 0, 'z': 0, 'w': 1.0}},
                                                       'covariance': [0.0] * 36},
                              'twist_with_covariance': {'twist': {'linear': {'x': 1.0, 'y': 0, 'z': 0},
                                                                  'angular': {'x': 0, 'y': 0, 'z': 0}},
                                                        'covariance': [0.0] * 36},
                              'acceleration_with_covariance': {'accel': {'linear': {'x': 0, 'y': 0, 'z': 0},
                                                                         'angular': {'x': 0, 'y': 0, 'z': 0}},
                                                               'covariance': [0.0] * 36},
                              'orientation_availability': 2, 'is_stationary': False},
               'shape': {'type': 0, 'footprint': {'points': [{'x': 0.0, 'y': 0.0, 'z': 0.0}]},
                         'dimensions': {'x': 0.6, 'y': 0.5, 'z': 1.8}}}
        msgs.append(('/tracked', 'autoware_perception_msgs/msg/TrackedObjects', t,
                     {'header': hdr(t, 'map'), 'objects': [ped]}))
    return msgs


WRITE_BAG = r'''
import json, sys
from mcap_ros2.writer import Writer
from mcap_ros2.decoder import DecoderFactory
from mcap.reader import make_reader
spec = json.load(open(sys.argv[1]))
with open(sys.argv[2], 'wb') as fh:
    w = Writer(fh)
    sch = {t: w.register_msgdef(t.replace('/msg/', '/'), text) for t, text in spec['schemas'].items()}
    for topic, t, stamp, msg in spec['messages']:
        w.write_message(topic, sch[t], msg, int(round(stamp * 1e9)), int(round(stamp * 1e9)))
    w.finish()

def plain(o):
    if isinstance(o, (bytes, bytearray)):
        return list(o)
    if isinstance(o, (list, tuple)):
        return [plain(v) for v in o]
    if hasattr(o, '__slots__'):
        return {k: plain(getattr(o, k)) for k in o.__slots__}
    if hasattr(o, '__dict__'):
        return {k: plain(v) for k, v in vars(o).items() if not k.startswith('_')}
    return o
out = []
with open(sys.argv[2], 'rb') as fh:
    for s, c, m, d in make_reader(fh, decoder_factories=[DecoderFactory()]).iter_decoded_messages():
        out.append([c.topic, plain(d)])
print(json.dumps(out))
'''


def _plain(o):
    if isinstance(o, (bytes, bytearray)):
        return list(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, list):
        return [_plain(v) for v in o]
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    return o


@pytest.fixture(scope='module')
def ros_bag(tmp_path_factory):
    """The scene written by mcap_ros2 (an independent encoder) and what mcap_ros2 decodes from it."""
    d = tmp_path_factory.mktemp('bag')
    msgs = scene_messages()
    defs = all_defs()
    types = sorted({m[1] for m in msgs})
    spec = {'schemas': {t: schema_of(t, defs) for t in types}, 'messages': msgs}
    (d / 'spec.json').write_text(json.dumps(spec))
    decoded = devkit(('mcap_ros2.writer', 'mcap_ros2.decoder'), WRITE_BAG, d / 'spec.json', d / 'scene.mcap')
    return str(d / 'scene.mcap'), msgs, decoded


def test_cdr_decoder_agrees_with_mcap_ros2(ros_bag):
    from falcon_autonomy.core.logreplay.ros2 import Bag
    path, msgs, ref = ros_bag
    bag = Bag(path)
    assert set(bag.topics) == {'/tf_static', '/tf', '/odom', '/pose', '/detections', '/tracked'}
    mine = [[tp, _plain(m)] for tp, m, _ in bag.messages(sorted(bag.topics))]
    assert len(mine) == len(ref)
    by_topic = {}
    for tp, m in ref:
        by_topic.setdefault(tp, []).append(m)
    got = {}
    for tp, m in mine:
        got.setdefault(tp, []).append(m)
    for tp in by_topic:
        assert _close(got[tp], by_topic[tp]), tp


def _close(a, b, tol=1e-6):
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_close(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))
    return a == b


def test_cdr_decodes_mcap_bag_writes_and_refuses_other_encodings():
    m = {'header': {'stamp': {'sec': 3, 'nanosec': 7}, 'frame_id': 'map'}, 'child_frame_id': 'base_link',
         'transform': {'translation': {'x': 1.5, 'y': -2.0, 'z': 0.25},
                       'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.6, 'w': 0.8}}}
    dec = cdr.Decoder()
    got = dec.decode('geometry_msgs/msg/TransformStamped', B.encode('geometry_msgs/msg/TransformStamped', m))
    assert got == m
    # big-endian plain CDR reads the same
    be = bytearray(b'\x00\x00\x00\x00')
    import struct
    be += struct.pack('>iI', 3, 7) + struct.pack('>I', 4) + b'map\x00'
    assert dec.decode('std_msgs/msg/Header', bytes(be)) == {'stamp': {'sec': 3, 'nanosec': 7}, 'frame_id': 'map'}
    with pytest.raises(cdr.CdrError, match='not plain CDR'):
        dec.decode('std_msgs/msg/Header', b'\x00\x03\x00\x00' + bytes(12))
    with pytest.raises(cdr.CdrError, match='ends early'):
        dec.decode('std_msgs/msg/Header', b'\x00\x01\x00\x00' + bytes(6))
    with pytest.raises(cdr.CdrError, match='wstring'):
        cdr.parse_fields('a/msg/B', 'wstring w\n')
    f = cdr.parse_fields('a/msg/B', 'int32 X=3  # a constant\nstring<=8 s "x"\nfloat64[<=4] v\nB[2] two # c\n')
    assert f == [('s', 'string', None), ('v', 'float64', -1), ('two', 'a/msg/B', 2)]


# --------------------------------------------------------------------------------------------------------- ROS bags
def test_ros_bag_import_tf_chain_detections_and_tracked_objects(ros_bag):
    path = ros_bag[0]
    log = import_log(None, path)
    src = log.meta['source']
    assert src['ego'] == '/tf: map -> odom -> base_link' and src['world_frame'] == 'map'
    e = log.ego
    assert len(e) == 31 and e.t[0] == 0.0 and e.t[-1] == pytest.approx(3.0)
    # map: (10, 5) + Rz(90°)(5t, 0) = (10, 5 + 5t), heading 90° (north); CARLA: (10, −5 − 5t), yaw −90°
    assert e.x == pytest.approx(np.full(31, 10.0), abs=1e-9)
    assert e.y == pytest.approx(-5 - 0.5 * np.arange(31), abs=1e-9)
    assert e.yaw == pytest.approx(np.full(31, -90.0), abs=1e-9)
    ids = {a.id: a for a in log.actors}
    car = ids['/detections:a']
    assert car.cls == 'car' and car.size == pytest.approx((4.5, 1.9, 1.5))
    assert car.x == pytest.approx(np.full(31, 10.0), abs=1e-9)
    assert car.y == pytest.approx(-15 - 0.5 * np.arange(31), abs=1e-9)
    assert car.z == pytest.approx(np.full(31, 0.75)) and car.yaw == pytest.approx(np.full(31, -90.0))
    ped = ids['/tracked:' + bytes(range(16)).hex()]
    assert ped.cls == 'pedestrian' and ped.x == pytest.approx(0.1 * np.arange(31)) and ped.y == pytest.approx(0.0)
    left = ' '.join(x['what'] for x in log.meta['left_out'])
    assert "class 'traffic_sign'" in left and '31 detections without an id' in left
    assert any(h['what'] == 'ego box' for h in log.meta['heuristic'])
    assert log.map['kind'] == 'unknown'
    # the same bag through odometry when the TF is not asked for: a frame that does not exist falls back
    log2 = import_log('ros2', path, ego_frame='no_such_frame')
    assert log2.meta['source']['ego'].startswith('/odom') and log2.ego.x == pytest.approx(0.5 * np.arange(31))


def test_ros_bag_navsatfix_only_is_geo_located(tmp_path):
    from falcon_autonomy.core.osm import local_projection
    lat0, lon0 = 12.9716, 77.5946
    proj = local_projection(lat0, lon0)
    bag = B.BagWriter(str(tmp_path / 'fix'))
    bag.add_topic('/gnss/fix', 'sensor_msgs/msg/NavSatFix')
    for k in range(40):                                   # 2 m/s north-east
        e = n = 2.0 * k * 0.1 / math.sqrt(2)
        lat, lon = proj.to_geo(e, n)
        t = int((T0 + 0.1 * k) * 1e9)
        bag.write('/gnss/fix', t, {'header': B.header(t, 'gnss'), 'status': {'status': 0, 'service': 1},
                                   'latitude': lat, 'longitude': lon, 'altitude': 900.0})
    bag.close()
    log = import_log(None, str(tmp_path / 'fix'))
    assert log.map['kind'] == 'geo' and log.map['geo']['lat0'] == pytest.approx(lat0)
    assert log.ego.yaw[5] == pytest.approx(-45.0, abs=0.01)           # north-east is CARLA yaw −45°
    assert log.ego.path_length() == pytest.approx(2 * 3.9, rel=1e-3)
    assert any('heading' in h['what'] for h in log.meta['heuristic'])


def _db3(path, topics, rows, defs=None):
    con = sqlite3.connect(path)
    con.execute('CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT NOT NULL, type TEXT NOT NULL, '
                'serialization_format TEXT NOT NULL, offered_qos_profiles TEXT NOT NULL)')
    con.execute('CREATE TABLE messages(id INTEGER PRIMARY KEY, topic_id INTEGER NOT NULL, timestamp INTEGER NOT NULL, '
                'data BLOB NOT NULL)')
    if defs is not None:
        con.execute('CREATE TABLE message_definitions(id INTEGER PRIMARY KEY, topic_type TEXT NOT NULL, '
                    'encoding TEXT NOT NULL, encoded_message_definition TEXT NOT NULL, '
                    'type_description_hash TEXT NOT NULL)')
        for t, text in defs.items():
            con.execute('INSERT INTO message_definitions(topic_type, encoding, encoded_message_definition, '
                        'type_description_hash) VALUES (?, ?, ?, ?)', (t, 'ros2msg', text, ''))
    for i, (name, typ) in enumerate(topics, 1):
        con.execute('INSERT INTO topics VALUES (?, ?, ?, ?, ?)', (i, name, typ, 'cdr', ''))
    for tid, ts, data in rows:
        con.execute('INSERT INTO messages(topic_id, timestamp, data) VALUES (?, ?, ?)', (tid, ts, data))
    con.commit()
    con.close()


def test_ros_bag_sqlite3_with_and_without_stored_definitions(tmp_path):
    from falcon_autonomy.core.logreplay.ros2 import register_vision_msgs
    register_vision_msgs()
    rows = []
    for k in range(20):
        t = int((T0 + 0.05 * k) * 1e9)
        rows.append((1, t, B.encode('tf2_msgs/msg/TFMessage', {'transforms': [tf('map', 'base_link', T0 + 0.05 * k,
                                                                                  k * 0.25, 2.0, 0.0)]})))
        det = {'header': B.header(t, 'map'), 'detections': [
            {'header': B.header(t, 'map'), 'results': [{'hypothesis': {'class_id': 'vehicle.bus.rigid',
                                                                       'score': 1.0}}],
             'bbox': {'center': {'position': {'x': 30.0, 'y': 2.0, 'z': 1.6}}, 'size': {'x': 11, 'y': 2.6, 'z': 3.2}},
             'id': '7'}]}
        rows.append((2, t, B.encode('vision_msgs/msg/Detection3DArray', det)))
    topics = [('/tf', 'tf2_msgs/msg/TFMessage'), ('/objects', 'vision_msgs/msg/Detection3DArray'),
              ('/custom', 'my_msgs/msg/Thing')]
    os.makedirs(tmp_path / 'a')
    _db3(str(tmp_path / 'a' / 'a_0.db3'), topics, rows)               # before Iron: no definitions stored
    log = import_log(None, str(tmp_path / 'a'))
    assert log.meta['source']['storage'] == 'sqlite3'
    assert log.ego.x == pytest.approx(0.25 * np.arange(20)) and log.ego.y == pytest.approx(np.full(20, -2.0))
    bus = log.actors[0]
    assert bus.cls == 'bus' and bus.x[0] == pytest.approx(30.0) and bus.y[0] == pytest.approx(-2.0)
    assert any('my_msgs/msg/Thing' in x['why'] for x in log.meta['left_out'])
    # Iron and later store the definitions: they are used
    os.makedirs(tmp_path / 'b')
    _db3(str(tmp_path / 'b' / 'b_0.db3'), topics[:2], rows,
         {'tf2_msgs/msg/TFMessage': B.schema_text('tf2_msgs/msg/TFMessage'),
          'vision_msgs/msg/Detection3DArray': B.schema_text('vision_msgs/msg/Detection3DArray')})
    log_b = import_log('ros2', str(tmp_path / 'b' / 'b_0.db3'))
    assert log_b.ego.x == pytest.approx(log.ego.x) and log_b.actors[0].x == pytest.approx(bus.x)


# --------------------------------------------------------------------------------------------------------- nuScenes
def nuscenes_tables(root, version='v1.0-mini'):
    """A two-instance scene in nuScenes' table layout: key frames at 2 Hz, LIDAR_TOP sweeps at 20 Hz between them,
    the ego driving north at 5 m/s from (400, 1100) in boston-seaport; a car ahead turning, a pedestrian, a barrier."""
    d = os.path.join(root, version)
    os.makedirs(d, exist_ok=True)
    tok = lambda *p: '_'.join(str(x) for x in p)                          # noqa: E731
    yaw_q = lambda a: [math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)]       # noqa: E731
    t0 = 1_533_151_603_547_590
    keys = [t0 + 500_000 * k for k in range(5)]
    samples = [{'token': tok('s', k), 'timestamp': ts, 'scene_token': 'sc', 'prev': tok('s', k - 1) if k else '',
                'next': tok('s', k + 1) if k < 4 else '', 'data': {}, 'anns': []} for k, ts in enumerate(keys)]
    sds, poses = [], []
    for k in range(4 * 10 + 1):
        ts = t0 + 50_000 * k
        sample = min(int(math.ceil(k / 10)), 4)
        is_key = k % 10 == 0
        poses.append({'token': tok('ep', k), 'timestamp': ts, 'translation': [400.0, 1100.0 + 5.0 * k * 0.05, 0.0],
                      'rotation': yaw_q(math.pi / 2)})
        sds.append({'token': tok('sd', k), 'sample_token': tok('s', sample), 'ego_pose_token': tok('ep', k),
                    'calibrated_sensor_token': 'cs_lidar', 'timestamp': ts, 'fileformat': 'pcd',
                    'is_key_frame': is_key, 'height': 0, 'width': 0,
                    'filename': f'{"samples" if is_key else "sweeps"}/LIDAR_TOP/x{k}.pcd.bin',
                    'prev': tok('sd', k - 1) if k else '', 'next': tok('sd', k + 1) if k < 40 else ''})
    anns = []
    for k, ts in enumerate(keys):
        a = 0.2 * k
        anns.append({'token': tok('ac', k), 'sample_token': tok('s', k), 'instance_token': 'car1',
                     'translation': [402.0 + 0.8 * k * k, 1120.0 + 4.0 * k, 0.9], 'size': [1.9, 4.6, 1.6],
                     'rotation': yaw_q(math.pi / 2 - a), 'prev': tok('ac', k - 1) if k else '',
                     'next': tok('ac', k + 1) if k < 4 else '', 'num_lidar_pts': 10, 'num_radar_pts': 0,
                     'visibility_token': '4', 'attribute_tokens': []})
        anns.append({'token': tok('ap', k), 'sample_token': tok('s', k), 'instance_token': 'ped1',
                     'translation': [395.0 + 0.7 * k, 1110.0, 0.9], 'size': [0.7, 0.6, 1.8],
                     'rotation': yaw_q(3.0 if k % 2 else -3.0), 'prev': tok('ap', k - 1) if k else '',
                     'next': tok('ap', k + 1) if k < 4 else '', 'num_lidar_pts': 5, 'num_radar_pts': 0,
                     'visibility_token': '4', 'attribute_tokens': []})
        anns.append({'token': tok('ab', k), 'sample_token': tok('s', k), 'instance_token': 'bar1',
                     'translation': [405.0, 1105.0, 0.5], 'size': [2.0, 0.5, 1.0], 'rotation': yaw_q(0.0),
                     'prev': tok('ab', k - 1) if k else '', 'next': tok('ab', k + 1) if k < 4 else '',
                     'num_lidar_pts': 3, 'num_radar_pts': 0, 'visibility_token': '4', 'attribute_tokens': []})
    tables = {
        'scene': [{'token': 'sc', 'name': 'scene-0001', 'description': 'synthetic', 'log_token': 'lg',
                   'nbr_samples': 5, 'first_sample_token': 's_0', 'last_sample_token': 's_4'}],
        'log': [{'token': 'lg', 'logfile': 'x', 'vehicle': 'n015', 'date_captured': '2018-08-01',
                 'location': 'boston-seaport'}],
        'map': [{'token': 'mp', 'log_tokens': ['lg'], 'category': 'semantic_prior', 'filename': 'maps/x.png'}],
        'sample': samples, 'sample_data': sds, 'ego_pose': poses, 'sample_annotation': anns,
        'sensor': [{'token': 'sn_lidar', 'channel': 'LIDAR_TOP', 'modality': 'lidar'}],
        'calibrated_sensor': [{'token': 'cs_lidar', 'sensor_token': 'sn_lidar', 'translation': [0.9, 0.0, 1.8],
                               'rotation': [1.0, 0.0, 0.0, 0.0], 'camera_intrinsic': []}],
        'instance': [{'token': 'car1', 'category_token': 'c_car', 'nbr_annotations': 5, 'first_annotation_token':
                      'ac_0', 'last_annotation_token': 'ac_4'},
                     {'token': 'ped1', 'category_token': 'c_ped', 'nbr_annotations': 5, 'first_annotation_token':
                      'ap_0', 'last_annotation_token': 'ap_4'},
                     {'token': 'bar1', 'category_token': 'c_bar', 'nbr_annotations': 5, 'first_annotation_token':
                      'ab_0', 'last_annotation_token': 'ab_4'}],
        'category': [{'token': 'c_car', 'name': 'vehicle.car', 'description': ''},
                     {'token': 'c_ped', 'name': 'human.pedestrian.adult', 'description': ''},
                     {'token': 'c_bar', 'name': 'movable_object.barrier', 'description': ''}],
        'attribute': [], 'visibility': [{'token': '4', 'level': 'v80-100', 'description': ''}]}
    for name, rows in tables.items():
        with open(os.path.join(d, name + '.json'), 'w') as fh:
            json.dump(rows, fh)
    os.makedirs(os.path.join(root, 'maps'), exist_ok=True)          # the devkit loads the map mask it lists
    import cv2
    cv2.imwrite(os.path.join(root, 'maps', 'x.png'), np.zeros((8, 8), np.uint8))
    return root


DEVKIT_BOXES = r'''
import json, sys
from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version='v1.0-mini', dataroot=sys.argv[1], verbose=False)
out = []
for sd in nusc.sample_data:
    if sd['is_key_frame']:
        continue
    for b in nusc.get_boxes(sd['token']):
        ann = nusc.get('sample_annotation', b.token)
        out.append([sd['timestamp'], ann['instance_token'], list(b.center), b.orientation.yaw_pitch_roll[0]])
print(json.dumps(out))
'''


def test_nuscenes_import_and_interpolation_match_the_devkit(tmp_path):
    root = nuscenes_tables(str(tmp_path / 'nusc'))
    log = import_log(None, root)
    assert log.meta['source']['scene'] == 'scene-0001' and log.map['kind'] == 'nuscenes'
    assert log.map['name'] == 'boston-seaport' and log.map['geo']['method'] == 'tmerc'
    e = log.ego
    assert len(e) == 41 and e.t[-1] == pytest.approx(2.0)
    assert e.x == pytest.approx(np.full(41, 400.0)) and e.y == pytest.approx(-(1100.0 + 5.0 * e.t))
    assert e.yaw == pytest.approx(np.full(41, -90.0))
    ids = {a.id: a for a in log.actors}
    assert set(ids) == {'car1', 'ped1'} and ids['car1'].size == pytest.approx((4.6, 1.9, 1.6))
    assert '5 annotations of movable_object.barrier' in json.dumps(log.meta['left_out'])
    ref = devkit(('nuscenes.nuscenes',), DEVKIT_BOXES, root)
    t0 = 1_533_151_603_547_590
    n = 0
    for ts, inst, c, yaw in ref:
        if inst not in ids:
            continue
        ok, x, y, z, yw = ids[inst].sample([(ts - t0) * 1e-6])
        if not ok[0]:
            continue                          # the devkit holds a new instance at its first pose; Falcon does not
        n += 1
        assert (x[0], -y[0], z[0]) == pytest.approx(tuple(c), abs=1e-6)
        assert ((-yw[0] - math.degrees(yaw) + 180) % 360 - 180) == pytest.approx(0.0, abs=1e-6)
    assert n >= 60                            # every sweep between key frames, for both instances


# --------------------------------------------------------------------------------------------------------- Waymo
def waymo_folder(root, seg='1234567890_0000_000_0020_000'):
    import pyarrow as pa
    import pyarrow.parquet as pq
    for comp in ('vehicle_pose', 'lidar_box', 'camera_box', 'stats'):
        os.makedirs(os.path.join(root, comp), exist_ok=True)
    ts = [1_550_000_000_000_000 + 100_000 * k for k in range(10)]
    poses = []
    for k in range(10):                     # the ego heading west (180°) at 10 m/s from (100, 50, 2)
        T = np.eye(4)
        T[:3, :3] = rz(math.pi)
        T[:3, 3] = [100.0 - 1.0 * k, 50.0, 2.0]
        poses.append(T.reshape(-1).tolist())
    pq.write_table(pa.table({'key.segment_context_name': [seg] * 10, 'key.frame_timestamp_micros': ts,
                             '[VehiclePoseComponent].world_from_vehicle.transform': poses}),
                   os.path.join(root, 'vehicle_pose', seg + '.parquet'))
    rows = {k: [] for k in ('seg', 'ts', 'id', 'cx', 'cy', 'cz', 'l', 'w', 'h', 'hd', 'type')}
    for k in range(10):
        for oid, cx, cy, L, typ, hd in (('veh', 20.0, 0.0, 4.5, 1, 0.0), ('bus', 30.0, -4.0, 12.0, 1, 0.5),
                                        ('ped', 8.0, 3.0, 0.8, 2, math.pi / 2), ('sign', 15.0, 6.0, 0.5, 3, 0.0),
                                        ('cyc', 12.0, -2.0, 1.8, 4, 0.0)):
            for key, v in (('seg', seg), ('ts', ts[k]), ('id', oid), ('cx', cx), ('cy', cy), ('cz', 1.0), ('l', L),
                           ('w', 2.0), ('h', 1.6), ('hd', hd), ('type', typ)):
                rows[key].append(v)
    names = {'seg': 'key.segment_context_name', 'ts': 'key.frame_timestamp_micros', 'id': 'key.laser_object_id',
             'cx': '[LiDARBoxComponent].box.center.x', 'cy': '[LiDARBoxComponent].box.center.y',
             'cz': '[LiDARBoxComponent].box.center.z', 'l': '[LiDARBoxComponent].box.size.x',
             'w': '[LiDARBoxComponent].box.size.y', 'h': '[LiDARBoxComponent].box.size.z',
             'hd': '[LiDARBoxComponent].box.heading', 'type': '[LiDARBoxComponent].type'}
    pq.write_table(pa.table({names[k]: v for k, v in rows.items()}), os.path.join(root, 'lidar_box', seg + '.parquet'))
    pq.write_table(pa.table({'key.segment_context_name': [seg], 'key.camera_name': [1],
                             '[CameraBoxComponent].box.center.x': [100.0]}),
                   os.path.join(root, 'camera_box', seg + '.parquet'))
    pq.write_table(pa.table({'key.segment_context_name': [seg], '[StatsComponent].location': ['location_sf'],
                             '[StatsComponent].time_of_day': ['Day'], '[StatsComponent].weather': ['sunny']}),
                   os.path.join(root, 'stats', seg + '.parquet'))
    return root


def test_waymo_v2_parquet_import(tmp_path):
    root = waymo_folder(str(tmp_path / 'waymo'))
    log = import_log(None, root)
    assert log.map == {'kind': 'waymo', 'name': 'location_sf', 'geo': None, 'note': log.map['note']}
    e = log.ego
    assert len(e) == 10 and e.x == pytest.approx(100.0 - np.arange(10)) and e.y == pytest.approx(np.full(10, -50.0))
    assert e.yaw == pytest.approx(np.full(10, 180.0)) or e.yaw == pytest.approx(np.full(10, -180.0))
    ids = {a.id: a for a in log.actors}
    assert set(ids) == {'veh', 'bus', 'ped', 'cyc'}
    assert (ids['veh'].cls, ids['bus'].cls, ids['ped'].cls, ids['cyc'].cls) == ('car', 'bus', 'pedestrian', 'bicycle')
    # 20 m ahead of a west-heading ego at x = 100 − k: world x = 80 − k; the ped 3 m to its left (south): y = 47
    assert ids['veh'].x == pytest.approx(80.0 - np.arange(10)) and ids['veh'].y == pytest.approx(np.full(10, -50.0))
    assert ids['ped'].x[0] == pytest.approx(92.0) and ids['ped'].y[0] == pytest.approx(-47.0)
    assert ids['ped'].yaw[0] == pytest.approx(90.0)          # heading south in the world: CARLA yaw +90°
    assert ids['veh'].z == pytest.approx(np.full(10, 3.0))
    left = json.dumps(log.meta['left_out'])
    assert '10 sign boxes' in left and '1 camera boxes' in left
    assert any('bus or truck' in h['what'] for h in log.meta['heuristic'])
    bad = tmp_path / 'bad'
    os.makedirs(bad / 'vehicle_pose')
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.table({'x': [1]}), str(bad / 'vehicle_pose' / 's.parquet'))
    with pytest.raises(LogError, match='it has x'):
        import_log('waymo', str(bad))


# --------------------------------------------------------------------------------------------------------- KITTI raw
@pytest.mark.skipif(not os.path.isdir(KITTI), reason='no KITTI raw drives on this machine')
def test_kitti_raw_oxts_on_real_drives():
    """Real data: the speed of the imported positions against the OXTS unit's own forward speed, and the heading of
    motion against the imported yaw (both measured by independent sensors of the unit)."""
    drives = sorted(d for d in os.listdir(KITTI) if d.endswith('_sync'))
    assert drives
    for d in drives:
        log = import_log(None, os.path.join(KITTI, d))
        assert log.map['kind'] == 'geo' and not log.actors
        chk = log.meta['source']['speed_check']
        assert chk['median_abs_speed_diff_mps'] < 0.2, (d, chk)
        e = log.ego
        vx, vy = np.gradient(e.x, e.t), np.gradient(e.y, e.t)
        m = np.hypot(vx, vy) > 2.0
        if m.sum() > 10:
            h = np.degrees(np.arctan2(vy[m], vx[m]))
            assert np.median(np.abs((h - e.yaw[m] + 180) % 360 - 180)) < 2.0, d


# --------------------------------------------------------------------------------------------------------- changes
def straight_log(n=101, v=10.0, map_block=None):
    t = np.arange(n) * 0.1
    ego = Track('ego', 'car', t, v * t, np.zeros(n), np.zeros(n), np.zeros(n), (4.6, 2.0, 1.5))
    lead = Track('lead', 'car', t, 30 + 5 * t, np.zeros(n), np.full(n, 0.75), np.zeros(n), (4.5, 1.9, 1.5))
    late = Track('late', 'car', t[50:], 60 + 0 * t[50:], np.full(51, 3.5), np.full(51, 0.75), np.zeros(51))
    return Log(ego, [lead, late], {'name': 'straight', 'map': map_block or {'kind': 'carla', 'name': 'Town10HD'}})


def test_changes_retime_remove_add_and_validate():
    log = straight_log()
    out, weather, done = CH.apply(log, [
        {'op': 'time_shift', 'actor': 'late', 'dt_s': -2.0},
        {'op': 'speed_scale', 'actor': 'lead', 'factor': 0.5, 'from_s': 2.0},
        {'op': 'add_pedestrian', 'id': 'p', 'at_ego_s': 6.0, 'from': 'left', 'start_m': 5.0, 'speed_mps': 1.25},
        {'op': 'weather', 'preset': 'WetCloudyNoon', 'params': {'fog_density': 30}}], seed=4)
    assert log.actor('late').t[0] == pytest.approx(5.0)                   # the original is untouched
    assert out.actor('late').t[0] == pytest.approx(3.0)
    lead, lead0 = out.actor('lead'), log.actor('lead')
    ok, x, *_ = lead.sample([4.0])                                        # at half speed after 2 s: 1 s of travel
    assert ok[0] and x[0] == pytest.approx(30 + 5 * 3.0)
    assert np.allclose(lead.x, lead0.x) and lead.t[-1] == pytest.approx(2.0 + 8.0 / 0.5)   # same path, re-timed
    p = out.actor('p')
    assert p.cls == 'pedestrian' and p.hold == (True, True)
    ok, x, y, *_ = p.sample([6.0])                                        # on the ego's path when the ego was there
    assert x[0] == pytest.approx(60.0) and y[0] == pytest.approx(0.0, abs=1e-9)
    ok, x, y, *_ = p.sample([0.0])                                        # waiting 5 m to the ego's left (−y)
    assert y[0] == pytest.approx(-5.0) and p.yaw[0] == pytest.approx(90.0)
    assert weather == {'preset': 'WetCloudyNoon', 'params': {'fog_density': 30}}
    assert [d['op'] for d in done] == ['time_shift', 'speed_scale', 'add_pedestrian', 'weather']
    stop, _, _ = CH.apply(log, [{'op': 'speed_scale', 'actor': 'lead', 'factor': 0.0, 'from_s': 3.0},
                                {'op': 'remove', 'actor': 'late'}])
    ok, x, *_ = stop.actor('lead').sample([3.0, 9.0])
    assert ok.all() and x == pytest.approx([45.0, 45.0]) and [a.id for a in stop.actors] == ['lead']
    held, _, _ = CH.apply(log, [{'op': 'time_shift', 'actor': 'late', 'dt_s': 1.0, 'outside': 'hold'}])
    assert held.actor('late').sample([0.0])[0][0]
    for bad, msg in (([{'op': 'nope'}], 'op'), ([{'op': 'remove'}], 'missing actor'),
                     ([{'op': 'remove', 'actor': 'ego'}], 'cannot be removed'),
                     ([{'op': 'remove', 'actor': 'x'}], 'no actor'),
                     ([{'op': 'add_pedestrian', 'at_ego_s': 99}], 'not in the log'),
                     ([{'op': 'speed_scale', 'actor': 'lead', 'factor': -1}], '≥ 0')):
        with pytest.raises(LogError, match=msg):
            CH.apply(log, bad)


def test_change_draws_are_seeded():
    log = straight_log()
    ch = [{'op': 'add_pedestrian', 'at_ego_s': 5.0, 'speed_mps': [0.8, 2.0]}]
    a = CH.apply(log, ch, seed=1)[2][0]['speed_mps']
    assert a == CH.apply(log, ch, seed=1)[2][0]['speed_mps'] and a != CH.apply(log, ch, seed=2)[2][0]['speed_mps']
    assert 0.8 <= a <= 2.0


# --------------------------------------------------------------------------------------------------------- plans
def test_plan_samples_places_and_repeats():
    log = straight_log()
    world = {'id': 'w', 'kind': 'town', 'town': 'Town10HD_Opt'}
    plan = make_plan(log, world, fps=20, changes=[{'op': 'add_pedestrian', 'at_ego_s': 6.0}], seed=3)
    assert plan['steps'] == 201 and plan['placement']['mode'] == 'matched' and plan['placement']['z'] == 'log'
    assert plan['ego']['x'][20] == pytest.approx(10.0) and plan['ego']['vx'][20] == pytest.approx(10.0)
    late = next(a for a in plan['actors'] if a['id'] == 'late')
    assert late['i0'] == 100 and len(late['x']) == 101                    # absent before it is observed
    ped = next(a for a in plan['actors'] if a['id'] == 'added_pedestrian_0')
    assert ped['i0'] == 0 and None not in ped['x']
    assert plan == make_plan(log, world, fps=20, changes=[{'op': 'add_pedestrian', 'at_ego_s': 6.0}], seed=3)
    with pytest.raises(LogError, match='driver'):
        make_plan(log, world, driver='ghost')
    gappy = straight_log()
    gappy.ego = Track('ego', 'car', np.r_[0.0, 0.1, 3.0], [0, 1, 30], [0, 0, 0], [0, 0, 0], [0, 0, 0], max_gap_s=0.5)
    with pytest.raises(LogError, match='gaps'):
        make_plan(gappy, world)


def test_placement_matched_geo_anchored():
    town = {'id': 'w', 'kind': 'town', 'town': 'Town10HD_Opt'}
    assert resolve_placement(straight_log(), town)['mode'] == 'matched'           # Town10HD and its _Opt variant
    other = straight_log(map_block={'kind': 'carla', 'name': 'Town03'})
    p = resolve_placement(other, town)
    assert p['mode'] == 'anchored' and 'no matching road' in p['why'] and p['anchor'] == {'spawn_index': 0}
    nusc = straight_log(map_block={'kind': 'nuscenes', 'name': 'boston-seaport',
                                   'geo': {'location': 'boston-seaport', 'method': 'tmerc', 'lat0': 42.336849169438615,
                                           'lon0': -71.05785369873047}})
    assert 'not a CARLA town' in resolve_placement(nusc, town)['why']
    # an OSM world around the log's place: geo placement; the same log far outside its area: anchored
    lat, lon = latlon_of(nusc.map['geo'], nusc.ego.x, nusc.ego.y)
    world = {'id': 'w', 'kind': 'osm', 'geo': {'lat0': float(lat[50]), 'lon0': float(lon[50]),
                                               'source': 'import centre'},
             'bbox': [float(lat.min()) - 0.01, float(lon.min()) - 0.01, float(lat.max()) + 0.01,
                      float(lon.max()) + 0.01]}
    assert resolve_placement(nusc, world)['mode'] == 'geo'
    plan = make_plan(nusc, world, fps=10)
    # the ego's path keeps its length and heading through the projections (to a millimetre over 100 m)
    x, y = np.array(plan['ego']['x']), np.array(plan['ego']['y'])
    assert float(np.sum(np.hypot(np.diff(x), np.diff(y)))) == pytest.approx(100.0, abs=1e-3)
    assert x[50] == pytest.approx(0.0, abs=0.05) and y[50] == pytest.approx(0.0, abs=0.05)
    far = dict(world, bbox=[0.0, 0.0, 0.1, 0.1])
    assert resolve_placement(nusc, far)['mode'] == 'anchored'


def test_rigid_anchor_moves_everything_together():
    from falcon_autonomy.core.logreplay.plan import rigid_anchor
    plan = make_plan(straight_log(), {'kind': 'town', 'town': 'Town01'}, fps=10)
    assert plan['placement']['mode'] == 'anchored'
    lead_before = (plan['actors'][0]['x'][0], plan['actors'][0]['y'][0])
    rigid_anchor(plan, 100.0, 50.0, 90.0)
    assert (plan['ego']['x'][0], plan['ego']['y'][0], plan['ego']['yaw'][0]) == pytest.approx((100, 50, 90))
    # the lead was 30 m ahead along +x; with the ego turned to +90° (CARLA: towards +y) it is 30 m along +y
    assert (plan['actors'][0]['x'][0], plan['actors'][0]['y'][0]) == pytest.approx((100.0, 80.0))
    assert lead_before == pytest.approx((30.0, 0.0))
    assert plan['actors'][0]['vy'][10] == pytest.approx(5.0) and plan['ego']['vy'][5] == pytest.approx(10.0)


def test_log_save_load_round_trip(tmp_path):
    log = straight_log()
    p = log.save(str(tmp_path / 'lg'))
    back = Log.load(str(tmp_path / 'lg'))
    assert back.to_doc() == json.loads(open(p).read())
    assert back.actor('late').t[0] == pytest.approx(5.0) and back.map == log.map
    with pytest.raises(LogError):
        Log.from_doc({'schema': 'x'})


# --------------------------------------------------------------------------------------------------------- matching
def test_lanes_from_xodr_and_the_match_verdicts():
    text = open(os.path.join(ROOT, 'tests', 'data', 'sumo_small.xodr')).read()
    lanes = MM.lanes_from_xodr(text)
    assert len(lanes) > 1000 and (lanes[:, 3] > 2.0).all()
    from falcon_autonomy.core import xodr as X
    od = X.parse(text)
    road = next(r for r in od.roads if not r.in_junction and X.driving_lanes(r)[1] >= 1)
    ref = X.sample_reference_line(road, 1.0)
    w = road.sections[0].right[0].widths[0][1]
    # a path along the first right lane's centre (w / 2 to the right of the reference line, along +s)
    px = ref[:, 0] + np.sin(ref[:, 2]) * w / 2
    py = ref[:, 1] - np.cos(ref[:, 2]) * w / 2
    good = MM.match(px, -py, -np.degrees(ref[:, 2]), lanes, 'test')
    assert good['verdict'] == 'match' and good['share_on_lane_in_direction'] > 0.95 and good['p95_lateral_m'] < 0.2
    wrong = MM.match(px[::-1], -py[::-1], -np.degrees(ref[::-1, 2]) + 180.0, lanes, 'test')
    if X.driving_lanes(road)[0] == 0:                   # a one-way road: driving it backwards is the wrong way
        assert wrong['share_wrong_way'] > 0.9 and wrong['verdict'] == 'partial'
    off = MM.match(px + 300.0, -py + 300.0, -np.degrees(ref[:, 2]), lanes, 'test')
    assert off['verdict'] == 'none' and off['share_no_lane_within_m'] > 0.5
    assert 'heuristic' in good['rule'] and good['thresholds'] == MM.THRESHOLDS


def test_match_on_a_straight_two_way_road():
    xs = np.arange(-100.0, 100.0, 2.0)
    lanes = np.r_[np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs), np.full_like(xs, 3.5)]),
                  np.column_stack([xs, np.full_like(xs, -3.5), np.full_like(xs, 180.0), np.full_like(xs, 3.5)])]
    x = np.linspace(-80, 80, 400)
    assert MM.match(x, np.full(400, 0.4), np.zeros(400), lanes)['verdict'] == 'match'
    on_wrong = MM.match(x, np.full(400, -3.5), np.zeros(400), lanes)
    assert on_wrong['share_wrong_way'] == 1.0 and on_wrong['verdict'] == 'partial'
    assert MM.match(x, np.full(400, 9.0), np.zeros(400), lanes)['verdict'] == 'none'
    stop = MM.resample_path(np.r_[np.zeros(500), np.arange(1, 11.0)], np.zeros(510), np.zeros(510))
    assert len(stop) == 11                                # a long stop counts once
