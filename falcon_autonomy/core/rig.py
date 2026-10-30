"""Rigs: a vehicle with its sensors — defaults, validation, cost, and import/export (standard library + numpy).

Frames. A sensor's pose is in CARLA's vehicle frame: metres from the vehicle's origin (CARLA's: the bounding box's
centre at ground height for CARLA's cars), x forward, y right, z up — LEFT-handed — and degrees, roll about x, pitch
about y (positive: nose up), yaw about z (positive: towards +y, i.e. to the right). ROS (REP 103) and nuScenes are
right-handed with y LEFT. Conversions go through rotation matrices: a CARLA rotation R becomes S·R·S with
S = diag(1, −1, 1), and a CARLA position (x, y, z) becomes (x, −y, z); Euler angles are never sign-flipped by rule.

Formats:
* Falcon rig JSON (schema falcon.rig/1): {"schema", "rig": <the project's rig object>} — lossless.
* ROS URDF: one fixed joint per sensor from base_link (the CARLA vehicle origin, ROS axes) to <sensor>_link (ROS body
  axes: x forward, y left, z up); cameras also get <sensor>_optical (z forward, x right, y down, REP 103). TF: the same
  transforms as a static-transform list (parent, child, translation, quaternion x y z w).
* nuScenes calibration: sensor.json and calibrated_sensor.json (translation in the ego frame, rotation as a quaternion
  w x y z of the sensor's frame — for cameras the optical frame, with camera_intrinsic). nuScenes' ego frame is the
  rear axle's centre; Falcon's is CARLA's vehicle origin — the export states which, it does not move it.
"""
from __future__ import annotations

import copy
import json
import math
import xml.etree.ElementTree as ET

import numpy as np

from . import budget

RIG_SCHEMA = 'falcon.rig/1'
SENSOR_TYPES = ('camera.pinhole', 'camera.fisheye', 'lidar', 'radar', 'gnss', 'imu', 'audio_array')
POSE0 = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}

# per-type defaults (CARLA 0.9.16 blueprint attributes where they exist; typical values labelled)
DEFAULTS = {
    'camera.pinhole': {'width': 1280, 'height': 720, 'fov': 90.0, 'rate_hz': 20.0, 'labels': ['semantic']},
    'camera.fisheye': {'width': 1280, 'height': 1024, 'rate_hz': 20.0, 'labels': ['semantic'],
                       'lens': {'model': 'equidistant', 'fov': 190.0, 'f_px': None, 'k': [0.0, 0.0, 0.0, 0.0]}},
    # a typical 32-channel spinning lidar (not a specific product)
    'lidar': {'channels': 32, 'range': 100.0, 'points_per_second': 600000, 'rotation_frequency': 10.0,
              'upper_fov': 15.0, 'lower_fov': -25.0, 'horizontal_fov': 360.0, 'noise_stddev': 0.015,
              'dropoff_general_rate': 0.0, 'labels': ['semantic', 'instance']},
    'radar': {'horizontal_fov': 30.0, 'vertical_fov': 10.0, 'range': 100.0, 'points_per_second': 1500,
              'rate_hz': 20.0},
    'gnss': {'rate_hz': 10.0, 'noise_lat_stddev': 0.0, 'noise_lon_stddev': 0.0, 'noise_alt_stddev': 0.0},
    'imu': {'rate_hz': 100.0, 'noise_accel_stddev': 0.0, 'noise_gyro_stddev': 0.0},
    # a SYNTHETIC microphone array (core.audio, docs/audio.md): not a CARLA sensor — nothing is spawned; its audio is
    # derived from the recorded actors after the drive. A uniform circle of `mics` microphones of `radius` m in the
    # array's x-y plane (mic 0 ahead, then clockwise seen from above, i.e. towards +y), or explicit `mic_positions`
    # ([[x, y, z], ...] m in the array's frame); a typical roof array, not a product
    'audio_array': {'mics': 8, 'radius': 0.2, 'mic_positions': None, 'sample_rate': 48000},
}
LIMITS = {'pose_m': 10.0, 'size_px': 8192, 'lidar_channels': 256, 'lidar_pps': 5_000_000}

# CARLA 0.9.16's vehicles commonly used as the ego, with their bounding-box extents (m, measured from the blueprints;
# the editor draws them): full length, width, height
VEHICLES = {
    'vehicle.lincoln.mkz_2020': (4.901, 2.106, 1.510),
    'vehicle.tesla.model3': (4.792, 2.163, 1.488),
    'vehicle.audi.a2': (3.705, 1.789, 1.549),
    'vehicle.toyota.prius': (4.513, 2.006, 1.522),
    'vehicle.nissan.patrol_2021': (5.565, 2.146, 2.045),
    'vehicle.mercedes.sprinter': (5.917, 1.991, 2.560),
    'vehicle.micro.microlino': (2.207, 1.481, 1.376),      # measured on the server (resources/carla inventory)
}


def new_sensor(stype: str, ident: str, name: str, pose: dict | None = None) -> dict:
    if stype not in SENSOR_TYPES:
        raise ValueError(f'unknown sensor type {stype!r}')
    s = {'id': ident, 'name': name, 'type': stype, 'pose': dict(POSE0, **(pose or {}))}
    s.update(copy.deepcopy(DEFAULTS[stype]))
    return s


def with_defaults(s: dict) -> dict:
    """A copy of a rig sensor with the settings it leaves out taken from its type's DEFAULTS (the schema lets a
    hand-written rig omit them; the workers read every one)."""
    return dict(copy.deepcopy(DEFAULTS.get(s.get('type'), {})), **copy.deepcopy(s))


def validate_rig(rig: dict) -> list:
    """Problems beyond the schema: duplicate ids, poses far from the car, sizes, lens and lidar settings."""
    out = []
    ids = [s['id'] for s in rig.get('sensors', [])]
    for d in sorted({i for i in ids if ids.count(i) > 1}):
        out.append(f'sensor id {d!r} is used more than once')
    for s in rig.get('sensors', []):
        p = s.get('pose', {})
        if max(abs(p.get(k, 0.0)) for k in ('x', 'y', 'z')) > LIMITS['pose_m']:
            out.append(f'{s["name"]}: more than {LIMITS["pose_m"]:g} m from the vehicle\'s origin')
        t = s.get('type')
        if t and t.startswith('camera.'):
            if not 16 <= int(s.get('width', 0)) <= LIMITS['size_px'] or not 16 <= int(s.get('height', 0)) <= \
                    LIMITS['size_px']:
                out.append(f'{s["name"]}: image size out of 16–{LIMITS["size_px"]} px')
            if t == 'camera.pinhole' and not 1.0 <= float(s.get('fov', 0)) < 180.0:
                out.append(f'{s["name"]}: a pinhole camera\'s field of view must be in 1–180°')
            if t == 'camera.fisheye' and not 60.0 <= float((s.get('lens') or {}).get('fov', 0)) <= 270.0:
                out.append(f'{s["name"]}: a fisheye\'s field of view must be in 60–270°')
        if t == 'audio_array':
            a = with_defaults(s)
            n = len(a['mic_positions']) if a.get('mic_positions') else int(a.get('mics') or 0)
            if not 2 <= n <= 64:
                out.append(f'{s["name"]}: an audio array needs 2–64 microphones')
            if not a.get('mic_positions') and not 0.01 <= float(a.get('radius') or 0) <= 2.0:
                out.append(f'{s["name"]}: the audio array\'s radius must be in 0.01–2 m')
        if t == 'lidar':
            if not 1 <= int(s.get('channels', 0)) <= LIMITS['lidar_channels']:
                out.append(f'{s["name"]}: lidar channels out of 1–{LIMITS["lidar_channels"]}')
            if float(s.get('upper_fov', 0)) <= float(s.get('lower_fov', 0)):
                out.append(f'{s["name"]}: the lidar\'s upper field of view must be above its lower')
    return out


def sensor_cost(s: dict) -> dict:
    """{'mpx', 'note'}: what the sensor renders per step (budget.sensor_mpx) and what that means."""
    mpx = budget.sensor_mpx(s)
    t = s.get('type', '')
    if t.startswith('camera.'):
        n = len(budget.label_cameras(s.get('labels')))
        if t == 'camera.fisheye':
            from . import lens
            views = len(lens.plan_camera(s, tables=False)['views'])
        else:
            views = 1
        return {'mpx': mpx, 'note': f'{views} view{"s" if views > 1 else ""} × (image + {n} label camera'
                                    f'{"s" if n != 1 else ""})'}
    if t == 'lidar':
        pts = int(s.get('points_per_second', 0)) / max(float(s.get('rotation_frequency', 10.0)), 1e-6)
        return {'mpx': 0.0, 'note': f'ray-cast, not rendered: about {pts / 1000:.0f} k points per rotation (+ the '
                                    'semantic lidar beside it for labels)'}
    if t == 'audio_array':
        return {'mpx': 0.0, 'note': 'synthetic audio derived after the drive (falcon derive audio): nothing rendered '
                                    'or spawned'}
    return {'mpx': 0.0, 'note': 'computed by the server, not rendered'}


# --------------------------------------------------------------------------------------------- frames and rotations
S_FLIP = np.diag([1.0, -1.0, 1.0])
CAM_OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])   # columns: optical x, y, z in ROS body


def carla_rotation(roll, pitch, yaw) -> np.ndarray:
    """CARLA's rotation matrix (Transform::GetMatrix's 3 × 3) for degrees."""
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cr, sr = math.cos(math.radians(roll)), math.sin(math.radians(roll))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    return np.array([[cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
                     [sp, -cp * sr, cp * cr]])


def carla_euler(R) -> tuple:
    """(roll, pitch, yaw) degrees of a CARLA rotation matrix (inverse of carla_rotation)."""
    pitch = math.degrees(math.asin(max(min(R[2, 0], 1.0), -1.0)))
    yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    roll = math.degrees(math.atan2(-R[2, 1], R[2, 2]))
    return roll, pitch, yaw


def to_ros(pose: dict) -> tuple[np.ndarray, np.ndarray]:
    """(translation, rotation matrix) of a CARLA vehicle-frame pose in ROS's right-handed body axes."""
    R = S_FLIP @ carla_rotation(pose['roll'], pose['pitch'], pose['yaw']) @ S_FLIP
    return np.array([pose['x'], -pose['y'], pose['z']]), R


def from_ros(t, R) -> dict:
    Rc = S_FLIP @ np.asarray(R, float) @ S_FLIP
    roll, pitch, yaw = carla_euler(Rc)
    return {'x': float(t[0]), 'y': float(-t[1]), 'z': float(t[2]), 'roll': roll, 'pitch': pitch, 'yaw': yaw}


def quat_of(R) -> tuple:
    """(w, x, y, z) of a rotation matrix (Shepperd's method), w ≥ 0."""
    R = np.asarray(R, float)
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * math.sqrt(tr + 1.0)
        q = (0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        v = [0.0, 0.0, 0.0]
        v[i] = 0.25 * s
        v[j] = (R[j, i] + R[i, j]) / s
        v[k] = (R[k, i] + R[i, k]) / s
        q = ((R[k, j] - R[j, k]) / s, *v)
    q = np.array(q)
    q = q / np.linalg.norm(q)
    return tuple(float(x) for x in (q if q[0] >= 0 else -q))


def matrix_of(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rpy_of(R) -> tuple:
    """ROS fixed-axis roll, pitch, yaw (radians) of a right-handed rotation (R = Rz(yaw) Ry(pitch) Rx(roll))."""
    pitch = math.asin(max(min(-R[2, 0], 1.0), -1.0))
    return math.atan2(R[2, 1], R[2, 2]), pitch, math.atan2(R[1, 0], R[0, 0])


def rpy_matrix(roll, pitch, yaw) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw),
                              math.sin(yaw))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def camera_K(s: dict) -> list:
    if s['type'] == 'camera.pinhole':
        f = s['width'] / (2.0 * math.tan(math.radians(s['fov']) / 2.0))
        return [[f, 0.0, s['width'] / 2.0], [0.0, f, s['height'] / 2.0], [0.0, 0.0, 1.0]]
    lens = s.get('lens') or {}
    f = lens.get('f_px') or budget.fisheye_focal_px(dict(s, fov=lens.get('fov', 190.0)))
    return [[f, 0.0, lens.get('cx', s['width'] / 2.0)], [0.0, f, lens.get('cy', s['height'] / 2.0)],
            [0.0, 0.0, 1.0]]


# --------------------------------------------------------------------------------------------- export / import
def export_json(rig: dict) -> str:
    return json.dumps({'schema': RIG_SCHEMA, 'rig': rig}, indent=2) + '\n'


def import_json(text: str) -> dict:
    d = json.loads(text)
    if d.get('schema') != RIG_SCHEMA or 'rig' not in d:
        raise ValueError(f'not a Falcon rig file (schema {d.get("schema")!r})')
    return d['rig']


def _link_name(s):
    return ''.join(c if c.isalnum() or c == '_' else '_' for c in s['name']) or s['id']


def tf_list(rig: dict) -> list:
    """Static transforms (ROS axes): [{'parent', 'child', 'translation': [x y z], 'rotation': [x y z w]}]."""
    out = []
    for s in rig['sensors']:
        t, R = to_ros(s['pose'])
        w, x, y, z = quat_of(R)
        link = f'{_link_name(s)}_link'
        out.append({'parent': 'base_link', 'child': link, 'translation': [round(float(v), 6) for v in t],
                    'rotation': [x, y, z, w]})
        if s['type'].startswith('camera.'):
            w, x, y, z = quat_of(CAM_OPTICAL)
            out.append({'parent': link, 'child': f'{_link_name(s)}_optical', 'translation': [0.0, 0.0, 0.0],
                        'rotation': [x, y, z, w]})
    return out


def export_urdf(rig: dict) -> str:
    robot = ET.Element('robot', name=_link_name({'name': rig['name'], 'id': rig['id']}))
    robot.append(ET.Comment(f' generated by Falcon Autonomy from rig {rig["id"]} ({rig["vehicle"]}); base_link is '
                            'CARLA\'s vehicle origin (bounding-box centre at ground height), ROS axes '))
    ET.SubElement(robot, 'link', name='base_link')
    for tf in tf_list(rig):
        ET.SubElement(robot, 'link', name=tf['child'])
        j = ET.SubElement(robot, 'joint', name=f'{tf["child"]}_joint', type='fixed')
        ET.SubElement(j, 'parent', link=tf['parent'])
        ET.SubElement(j, 'child', link=tf['child'])
        x, y, z, w = tf['rotation']
        r, p, yw = rpy_of(matrix_of((w, x, y, z)))
        ET.SubElement(j, 'origin', xyz=' '.join(f'{v:.6f}' for v in tf['translation']),
                      rpy=f'{r:.9f} {p:.9f} {yw:.9f}')
    ET.indent(robot)
    return '<?xml version="1.0"?>\n' + ET.tostring(robot, encoding='unicode') + '\n'


def import_urdf(text: str, base: dict | None = None) -> dict:
    """Sensors from a URDF: every fixed joint from base_link to a <name>_link, its pose converted to CARLA's frame.
    Types and settings are not in a URDF: they come from `base` (a rig, matched by name) or default to a pinhole
    camera for links with an _optical child, else a lidar named so, else an IMU."""
    root = ET.fromstring(text)
    joints = {j.find('child').get('link'): j for j in root.findall('joint')}
    optical = {j.find('parent').get('link') for j in root.findall('joint')
               if j.find('child').get('link').endswith('_optical')}
    known = {s['name']: s for s in (base or {}).get('sensors', [])}
    sensors = []
    for child, j in joints.items():
        if j.find('parent').get('link') != 'base_link' or not child.endswith('_link'):
            continue
        o = j.find('origin')
        xyz = [float(v) for v in (o.get('xyz', '0 0 0') if o is not None else '0 0 0').split()]
        rpy = [float(v) for v in (o.get('rpy', '0 0 0') if o is not None else '0 0 0').split()]
        name = child[:-5]
        pose = from_ros(xyz, rpy_matrix(*rpy))
        if name in known:
            s = copy.deepcopy(known[name])
            s['pose'] = pose
        else:
            stype = 'camera.pinhole' if child in optical else 'lidar' if 'lidar' in name.lower() else 'imu'
            s = new_sensor(stype, f'{name}', name, pose)
        sensors.append(s)
    rig = copy.deepcopy(base) if base else {'id': 'rig-urdf', 'name': root.get('name', 'URDF rig'),
                                            'vehicle': 'vehicle.lincoln.mkz_2020'}
    rig['sensors'] = sensors
    return rig


NUSCENES_MODALITY = {'camera.pinhole': 'camera', 'camera.fisheye': 'camera', 'lidar': 'lidar', 'radar': 'radar'}


def export_nuscenes(rig: dict, token=lambda s: s) -> dict:
    """{'sensor': [...], 'calibrated_sensor': [...]} as nuScenes v1.0 tables (cameras in their optical frame)."""
    sensors, cal = [], []
    for s in rig['sensors']:
        mod = NUSCENES_MODALITY.get(s['type'])
        if mod is None:
            continue                      # nuScenes has no GNSS or IMU calibration table entries
        t, R = to_ros(s['pose'])
        if mod == 'camera':
            R = R @ CAM_OPTICAL
        st = token(f'sensor-{s["id"]}')
        sensors.append({'token': st, 'channel': s['name'].upper().replace(' ', '_'), 'modality': mod})
        cal.append({'token': token(f'calibrated-{s["id"]}'), 'sensor_token': st,
                    'translation': [float(v) for v in t], 'rotation': list(quat_of(R)),
                    'camera_intrinsic': camera_K(s) if mod == 'camera' else []})
    return {'sensor': sensors, 'calibrated_sensor': cal,
            'ego_frame': 'CARLA vehicle origin (bounding-box centre at ground height), not nuScenes\' rear axle'}


def import_nuscenes(sensor_rows: list, calibrated_rows: list, base: dict | None = None) -> dict:
    by_token = {r['token']: r for r in sensor_rows}
    sensors = []
    for c in calibrated_rows:
        srow = by_token.get(c['sensor_token'])
        if srow is None:
            continue
        R = matrix_of(c['rotation'])
        cam = srow['modality'] == 'camera'
        if cam:
            R = R @ CAM_OPTICAL.T
        pose = from_ros(c['translation'], R)
        name = srow['channel']
        stype = {'camera': 'camera.pinhole', 'lidar': 'lidar', 'radar': 'radar'}[srow['modality']]
        s = new_sensor(stype, name.lower(), name, pose)
        K = c.get('camera_intrinsic') or []
        if cam and K:
            s['width'] = int(round(2 * K[0][2]))
            s['height'] = int(round(2 * K[1][2]))
            s['fov'] = math.degrees(2 * math.atan(s['width'] / (2 * K[0][0])))
        sensors.append(s)
    rig = copy.deepcopy(base) if base else {'id': 'rig-nuscenes', 'name': 'nuScenes rig',
                                            'vehicle': 'vehicle.lincoln.mkz_2020'}
    rig['sensors'] = sensors
    return rig
