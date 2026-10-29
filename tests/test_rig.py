"""Rigs: frame conversions (CARLA's left-handed vehicle frame to ROS / nuScenes), import/export round trips, cost."""
import math

import numpy as np
import pytest

from falcon_autonomy.core import rig as R
from falcon_autonomy.core.project import default_rig


def test_frames_by_physical_meaning():
    # a camera 0.5 m to the right (CARLA +y) is at ROS y = −0.5
    t, Rm = R.to_ros(dict(R.POSE0, y=0.5))
    assert t[1] == pytest.approx(-0.5)
    # facing right (CARLA yaw +90): its forward axis points to ROS −y
    _, Rm = R.to_ros(dict(R.POSE0, yaw=90.0))
    assert Rm[:, 0] == pytest.approx([0.0, -1.0, 0.0], abs=1e-12)
    # pitched up (CARLA pitch +10): its forward axis rises in both frames
    _, Rm = R.to_ros(dict(R.POSE0, pitch=10.0))
    assert Rm[2, 0] == pytest.approx(math.sin(math.radians(10)))
    assert R.carla_rotation(0, 10, 0)[2, 0] == pytest.approx(math.sin(math.radians(10)))
    # a right-handed result
    assert np.linalg.det(Rm) == pytest.approx(1.0)


@pytest.mark.parametrize('pose', [dict(x=1.2, y=-0.4, z=1.7, roll=3.0, pitch=-12.0, yaw=37.0),
                                  dict(x=-2.0, y=0.9, z=0.5, roll=-20.0, pitch=45.0, yaw=-170.0)])
def test_ros_round_trip(pose):
    t, Rm = R.to_ros(pose)
    back = R.from_ros(t, Rm)
    for k, v in pose.items():
        assert back[k] == pytest.approx(v, abs=1e-9)
    assert R.matrix_of(R.quat_of(Rm)) == pytest.approx(Rm, abs=1e-12)
    assert R.rpy_matrix(*R.rpy_of(Rm)) == pytest.approx(Rm, abs=1e-12)


def _rig():
    rig = default_rig()
    rig['sensors'].append(R.new_sensor('lidar', 'lidar-top', 'top_lidar', dict(x=0.0, y=0.0, z=2.0, yaw=0.5)))
    rig['sensors'].append(R.new_sensor('imu', 'imu', 'imu', dict(x=-1.0, z=0.4)))
    rig['sensors'][0]['pose'].update(y=0.3, pitch=-5.0, yaw=12.0)
    return rig


def test_urdf_round_trip():
    rig = _rig()
    text = R.export_urdf(rig)
    assert 'front_optical' in text and 'base_link' in text
    back = R.import_urdf(text, base=rig)
    for a, b in zip(rig['sensors'], back['sensors']):
        assert a['name'] == b['name'] and a['type'] == b['type']
        for k in a['pose']:
            assert b['pose'][k] == pytest.approx(a['pose'][k], abs=1e-6)
    plain = R.import_urdf(text)                         # without the rig: types guessed from the links
    assert [s['type'] for s in plain['sensors']] == ['camera.pinhole', 'lidar', 'imu']


def test_nuscenes_round_trip():
    rig = _rig()
    doc = R.export_nuscenes(rig)
    assert [s['modality'] for s in doc['sensor']] == ['camera', 'lidar']           # no IMU table in nuScenes
    cam = doc['calibrated_sensor'][0]
    # the optical axis (z) of the camera points mostly forward (+x in nuScenes' ego frame)
    assert R.matrix_of(cam['rotation'])[:, 2][0] > 0.9
    back = R.import_nuscenes(doc['sensor'], doc['calibrated_sensor'])
    for k in rig['sensors'][0]['pose']:
        assert back['sensors'][0]['pose'][k] == pytest.approx(rig['sensors'][0]['pose'][k], abs=1e-6)
    assert back['sensors'][0]['fov'] == pytest.approx(90.0)


def test_json_round_trip_and_validation():
    rig = _rig()
    assert R.import_json(R.export_json(rig)) == rig
    with pytest.raises(ValueError):
        R.import_json('{"schema": "other"}')
    bad = _rig()
    bad['sensors'][1]['id'] = bad['sensors'][0]['id']
    bad['sensors'][0]['pose']['x'] = 30.0
    bad['sensors'][1]['upper_fov'] = -40.0
    probs = R.validate_rig(bad)
    assert len(probs) == 3


def test_costs():
    s = R.new_sensor('camera.pinhole', 'c', 'c')
    assert R.sensor_cost(s)['mpx'] == pytest.approx(1280 * 720 * 3 / 1e6)          # image + instance + depth
    assert R.sensor_cost(R.new_sensor('lidar', 'l', 'l'))['mpx'] == 0.0
    assert '60 k points' in R.sensor_cost(R.new_sensor('lidar', 'l', 'l'))['note']


def test_with_defaults_fills_what_a_hand_written_sensor_leaves_out():
    # a rig lidar without rotation_frequency passes the schema; the drive and HIL workers read every setting
    s = {'id': 'l', 'name': 'L', 'type': 'lidar', 'pose': dict(R.POSE0), 'channels': 64}
    d = R.with_defaults(s)
    assert d['channels'] == 64 and d['rotation_frequency'] == R.DEFAULTS['lidar']['rotation_frequency']
    assert 'rotation_frequency' not in s                     # the rig itself is not changed
    d['labels'].append('x')
    assert 'x' not in R.DEFAULTS['lidar']['labels']
