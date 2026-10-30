"""Cooperative roadside units (docs/v2x.md): the CPM's UPER encoding (core.cpm) — round trips, ranges, frozen bytes,
and against asn1tools compiling ETSI's own ASN.1 (tests/data/etsi_cpm_asn1, BSD-3-Clause: cpm_ts103324 v2.1.1 and its
CDD submodule) when a Python with asn1tools is found (this one, or FALCON_DEVKITS_PYTHON); the RSU definitions, frames
and junction placement (core.v2x); drives recording roadside sensors and their CPMs against the fake carla (static
sensors), the measured benefit, the MCAP export, the stack bridge's CPM topics, the Worlds page's section; and
(FALCON_ROS2=1 with an image holding etsi_its_messages) the asn1c-generated decoder of etsi_its_messages' conversion
node reading Falcon's bytes."""
import base64
import json
import math
import os
import shutil
import subprocess
import sys
import threading

import numpy as np
import pytest

from conftest import ROOT, worker_env
from falcon_autonomy.core import cpm as C
from falcon_autonomy.core import protocol as P
from falcon_autonomy.core import v2x as V

ASN1_DIR = os.path.join(ROOT, 'tests', 'data', 'etsi_cpm_asn1')
SMALL_XODR = os.path.join(ROOT, 'tests', 'data', 'sumo_small.xodr')


def sample_cpm(sectors=20):
    objs = [C.perceived_object({'object_id': 70123, 'pos': (12.34, -5.6, 0.8), 'vel': (3.2, -0.4), 'yaw_enu_deg': 350.2,
                                'yaw_rate_dps': -4.0, 'size': (4.5, 1.9, 1.5), 'age_ms': 900, 'quality': 12,
                                'sensor_ids': [1, 2], 'category': 'vehicle.car', 'sigma_pos_m': 0.2,
                                'sigma_vel_mps': 0.1}),
            C.perceived_object({'object_id': 5, 'pos': (-40.0, 20.0, 0.9), 'vel': (0, 1.1), 'yaw_enu_deg': 90,
                                'size': (0.5, 0.5, 1.8), 'category': 'human.pedestrian.adult'}),
            C.perceived_object({'object_id': 6, 'pos': (1.0, 2.0, 0.6), 'vel': (8, 0), 'yaw_enu_deg': 0,
                                'size': (2.2, 0.8, 1.5), 'category': 'vehicle.motorcycle', 'sensor_ids': [2]}),
            C.perceived_object({'object_id': 7, 'pos': (1.0, 2.0, 1.6), 'vel': (8, 0), 'yaw_enu_deg': 0,
                                'size': (11.2, 2.5, 3.2), 'category': 'vehicle.bus.rigid', 'sensor_ids': [1]})]
    sens = [C.sensor_information(1, 'camera', 45.0, 90.0, 100.0, -20.0, 55.0, (0.3, 0.1, 6.0)),
            C.sensor_information(2, 'lidar', 0.0, 360.0, 80.0, -5.0, 40.0)]
    regs = [C.perception_region(1, (0.3, 0.1, 6.0), [(30.0 + k, 10.0 + 5 * k, 15.0 + 5 * k) for k in range(16)],
                                [70123, 7]),
            C.perception_region(2, (0.0, 0.0, 6.5), [(55.0, 360.0 / sectors * k, 360.0 / sectors * (k + 1))
                                                     for k in range(sectors)], [])]
    return C.cpm_message(123456, C.its_time_ms('2025-01-01T00:00:00Z') + 1234, 12.9716, 77.5946, 920.0, sens, regs,
                         objs, rate=10.0)


# ------------------------------------------------------------------------------------------------ the encoding
def test_cpm_round_trips_and_has_the_its_header():
    m = sample_cpm()
    data = C.encode(m)
    assert C.decode(data) == m
    # ItsPduHeader: protocolVersion 2, messageId 14 (cpm), stationId — whole octets (WITH COMPONENTS is not
    # PER-visible), then the payload's extension bit
    assert data[:6] == bytes([2, 14]) + (123456).to_bytes(4, 'big')
    cont = C.containers(m)
    assert set(cont) == {'originatingRsuContainer', 'sensorInformationContainer', 'perceptionRegionContainer',
                         'perceivedObjectContainer'}
    assert len(cont['perceptionRegionContainer'][1]['perceptionRegionShape']['radialShapes']['radialShapesList']) == 20
    # 20 sectors exceed RadialShapesList's root (1..16): its extension bit and a length — and it still round-trips
    assert C.decode(C.encode(sample_cpm(sectors=16))) == sample_cpm(sectors=16)
    assert C.btp_payload(data)[:4] == bytes([0x07, 0xD9, 0, 0]) and C.btp_payload(data)[4:] == data


def test_cpm_units_frames_and_classes():
    assert C.its_time_ms('2025-01-01T00:00:00Z') == (7671 * 86400 + 5) * 1000       # 21 years, 6 leap days, 5 leap s
    assert C.wgs84_heading(0.0) == 90.0 and C.wgs84_heading(90.0) == 180.0           # CARLA +x East, +y South
    assert C.enu_angle(90.0) == 270.0 and C.enu_angle(-90.0) == 90.0
    assert C.confidence_cm(0.0) == 1 and C.confidence_cm(0.1) == 20 and C.confidence_cm(100.0) == 4095
    assert C.object_class('vehicle.bus.rigid') == {'vehicleSubClass': 6}
    assert C.object_class('vehicle.motorcycle') == {'vruSubClass': {'motorcyclist': 'motorcycle'}}
    assert C.object_class('human.pedestrian.child') == {'vruSubClass': {'pedestrian': 'ordinary-pedestrian'}}
    po = C.perceived_object({'object_id': 1, 'pos': (2000.0, -2000.0, 0.0), 'vel': (500.0, 0), 'yaw_enu_deg': 359.99,
                             'size': (40.0, 0.01, 1.0)})
    assert po['position']['xCoordinate']['value'] == 131071 and po['position']['yCoordinate']['value'] == -131072
    assert po['velocity']['cartesianVelocity']['xVelocity']['value'] == 16382          # positiveOutOfRange
    assert po['angles']['zAngle']['value'] == 0 and po['objectDimensionX']['value'] == 255
    assert po['objectDimensionY']['value'] == 1
    assert C.rate_hz(10.0) == {'mantissa': 10, 'exponent': 0} and C.rate_hz(2.5) == {'mantissa': 25, 'exponent': -1}


def test_out_of_range_values_are_refused_naming_the_component():
    m = sample_cpm()
    C.containers(m)['perceivedObjectContainer']['perceivedObjects'][0]['objectAge'] = 5000
    with pytest.raises(C.EncodeError, match='objectAge'):
        C.encode(m)
    m = sample_cpm()
    C.containers(m)['perceivedObjectContainer']['perceivedObjects'][0]['acceleration'] = {}
    with pytest.raises(C.EncodeError, match='acceleration: not implemented'):
        C.encode(m)


# the encoding of sample_cpm(), frozen after asn1tools (with the effective-constraint patch below) and the asn1c
# decoder of etsi_its_messages 3.5.0 decoded it to the same values (docs/v2x.md); a change here is a change on the wire
GOLDEN_SHA256 = 'e4e2f37d6dced81073b16124cbaf721ff994c56df36fd2f3bb0a33820177e24b'


def test_cpm_bytes_are_frozen():
    import hashlib
    data = C.encode(sample_cpm())
    assert len(data) == 424 and hashlib.sha256(data).hexdigest() == GOLDEN_SHA256


ASN1TOOLS_CHECK = r'''
import glob, json, os, re, shutil, sys, tempfile
import asn1tools
src, data_hex, want = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
tmp = tempfile.mkdtemp()
for f in glob.glob(os.path.join(src, '*.asn')):
    t = open(f, encoding='utf-8').read()
    # asn1tools 0.169 does not apply the PER-visible union constraint of ObjectClass.vehicleSubClass; its effective
    # range (X.691: the smallest range containing the union), which asn1c applies (4 bits), is 0..14
    t = t.replace('TrafficParticipantType (unknown|passengerCar..tram|agricultural)', 'TrafficParticipantType (0..14)')
    # nor X.680's rule that serially applied constraints take the last one's extensibility: ConstraintWrappedCpmContainers
    # (WrappedCpmContainers, SIZE(1..8,...), constrained again without an extension marker) has no extension bit — as
    # asn1c encodes it
    t = re.sub(r'ConstraintWrappedCpmContainers ::= WrappedCpmContainers\s*\(\(WITH COMPONENT.*?ALL EXCEPT 2\)\}\)\)\)',
               'ConstraintWrappedCpmContainers ::= SEQUENCE SIZE(1..8) OF WrappedCpmContainer', t, flags=re.S)
    open(os.path.join(tmp, os.path.basename(f)), 'w', encoding='utf-8').write(t)
c = asn1tools.compile_files(sorted(glob.glob(os.path.join(tmp, '*.asn'))), 'uper')
dec = c.decode('CollectivePerceptionMessage', bytes.fromhex(data_hex))
names = {2: 'OriginatingRsuContainer', 3: 'SensorInformationContainer', 4: 'PerceptionRegionContainer',
         5: 'PerceivedObjectContainer'}
def norm(v):
    if isinstance(v, tuple):
        return {v[0]: norm(v[1])}
    if isinstance(v, dict):
        return {k: norm(x) for k, x in v.items()}
    if isinstance(v, list):
        return [norm(x) for x in v]
    return v
ok = dec['header'] == want['header'] and norm(dec['payload']['managementContainer']) == want['payload']['managementContainer']
for x, y in zip(dec['payload']['cpmContainers'], want['payload']['cpmContainers']):
    ok = ok and x['containerId'] == y['containerId'] and norm(c.decode(names[x['containerId']], x['containerData'])) == y['containerData']
print(json.dumps({'equal': ok, 'version': asn1tools.__version__}))
'''


def test_cpm_against_asn1tools_and_etsis_asn1():
    sys.path.insert(0, os.path.join(ROOT, 'tests'))
    from test_exporters import devkit_python
    py = devkit_python('asn1tools')
    if py is None:
        pytest.skip('asn1tools not installed (this Python, or FALCON_DEVKITS_PYTHON)')
    m = sample_cpm()
    r = subprocess.run([py, '-I', '-c', ASN1TOOLS_CHECK, ASN1_DIR, C.encode(m).hex(), json.dumps(m)],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    assert json.loads(r.stdout.strip().splitlines()[-1])['equal']


# ------------------------------------------------------------------------------------------------ definitions
def test_rsu_frames_presets_and_validation():
    r = V.new_rsu('w', 10.0, 20.0, 90.0, 'camera-lidar', 6.0, ident='rsu-a')
    cam, lid = r['sensors']
    assert cam['type'] == 'camera.pinhole' and lid['type'] == 'lidar' and lid['channels'] == 32
    assert V.mount_pose(r, cam)['z'] == 6.0 and V.mount_pose(r, lid)['z'] == pytest.approx(6.4)
    p = V.to_world(r, {'x': 1.0, 'y': 0.0, 'z': 6.0, 'roll': 0.0, 'pitch': -20.0, 'yaw': 10.0})
    assert (p['x'], p['y'], p['z'], p['yaw']) == (pytest.approx(10.0), pytest.approx(21.0), 6.0, 100.0)
    assert V.to_enu(r, [[11.0, 18.0, 1.0]]).tolist() == [[1.0, 2.0, 1.0]]             # y North = −CARLA y
    assert V.station_id(r) == V.station_id(dict(r)) and V.station_id(dict(r, station_id=7)) == 7
    assert V.validate_rsu(r, {'w'}) == []
    bad = dict(r, height_m=50.0, sensors=r['sensors'] + [{'id': 'g', 'name': 'gnss', 'type': 'gnss',
                                                          'pose': dict(cam['pose'])}])
    assert len(V.validate_rsu(bad, {'x'})) == 3
    s = V.sensor_with_defaults({'id': 'c', 'name': 'c', 'type': 'camera.pinhole', 'labels': [],
                                'pose': dict(cam['pose']), 'faults': [{'kind': 'drop'}]})
    assert s['labels'] == ['semantic'] and 'faults' not in s and s['width'] == 1280
    c = V.cpm_settings(dict(r, cpm={'rate_hz': 5, 'noise': {'position_m': 0.3}}))
    assert c['rate_hz'] == 5 and c['noise'] == {'position_m': 0.3, 'velocity_mps': 0.0} and c['min_pixels'] == 20


def test_junctions_and_placement_on_a_converted_world():
    text = open(SMALL_XODR).read()
    js = V.junctions(text)
    assert len(js) == 3 and js[0]['signalised']                      # signalised first
    from falcon_autonomy.core.cityprops import Lanes
    lanes = Lanes(text)
    for j in js:
        for c in V.corners(j, text, lanes):
            _, gap = lanes.nearest('driving', c['x'], -c['y'], radius=30.0)
            assert gap >= 0.5 and c['distance_m'] >= j['radius_m']
            to_centre = math.degrees(math.atan2(j['y'] - c['y'], j['x'] - c['x']))
            assert abs((to_centre - c['yaw'] + 180) % 360 - 180) < 0.1
    r = V.place_at_junction('w', js[1], text, 'pole-camera', 7.0)
    assert r['junction']['id'] == js[1]['id'] and len(r['sensors']) == 1 and r['height_m'] == 7.0
    assert -45.0 <= r['sensors'][0]['pose']['pitch'] <= -8.0


def test_project_schema_and_drive_config(project, tmp_path):
    from falcon_autonomy.core.drive import drive_config
    from falcon_autonomy.core.project import validate
    plain = drive_config(project, 'world-town10', 'rig-default', str(tmp_path / 'a'), 2310, 8310)
    assert 'infrastructure' not in plain                         # no RSUs: the config (and its hash) is unchanged
    r = V.new_rsu('world-town10', 20.0, 8.0, 200.0, ident='rsu-a')
    project.doc['infrastructure'] = {'rsus': [r]}
    assert validate(project.doc) == [] and V.validate_all(project.doc) == []
    cfg = drive_config(project, 'world-town10', 'rig-default', str(tmp_path / 'b'), 2310, 8310)
    got = cfg['infrastructure']['rsus'][0]
    assert got['station_id'] == V.station_id(r) and got['cpm']['rate_hz'] == 10.0
    assert got['sensors'][1]['dropoff_general_rate'] == 0.0      # the type's defaults filled in
    project.doc['infrastructure']['rsus'][0]['height_m'] = 0.2
    assert any('height_m' in e for e in validate(project.doc))


def test_spawn_order_starts_the_ego_heading_for_the_rsu():
    spawns = [(0.0, 0.0, 0.0), (100.0, 0.0, 180.0), (-80.0, 0.0, 0.0), (-80.0, 0.0, 180.0), (300.0, 0.0, 0.0),
              (10.0, 0.0, 90.0)]
    order, n = V.spawn_order(spawns, [4, 3, 2, 1, 0, 5], {'x': 0.0, 'y': 0.0, 'min_m': 50, 'max_m': 150,
                                                          'traffic_within_m': 200})
    assert n == 2 and order[:2] == [2, 1] and order[2:] == [3, 0, 5, 4]


# ------------------------------------------------------------------------------------------------ the fake carla
def rsu_project(project, rate=10.0):
    from falcon_autonomy.core.rig import new_sensor
    rig = project.doc['rigs'][0]
    rig['sensors'][0].update(width=192, height=108, labels=['semantic'])
    lid = new_sensor('lidar', 'lid', 'top', {'z': 1.9})
    lid.update(channels=16, points_per_second=40000, range=60.0)
    rig['sensors'].append(lid)
    project.doc['traffic_profiles'][0]['vehicles'] = 8
    r = V.new_rsu('world-town10', 20.0, 8.0, 200.0, 'camera-lidar', 6.0, 'test pole', ident='rsu-a')
    r['sensors'][0].update(width=192, height=108)
    r['sensors'][1].update(channels=16, points_per_second=40000, range=60.0)
    r['cpm'] = {'rate_hz': rate}
    project.doc['infrastructure'] = {'rsus': [r]}
    return r


def drive(project, out, seconds=1.0, seed=1, rsu=None):
    from falcon_autonomy.core.drive import drive_config
    cfg = drive_config(project, 'world-town10', 'rig-default', str(out), 2310, 8310, seconds=seconds, seed=seed)
    if rsu is not None:
        V.start_near(cfg, rsu, 10, 60)
    p = str(out) + '.json'
    with open(p, 'w') as fh:
        json.dump(cfg, fh)
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy.workers', 'drive', p], capture_output=True, text=True,
                       env=worker_env(), timeout=300)
    return r


@pytest.fixture(scope='module')
def rsu_run(tmp_path_factory):
    from falcon_autonomy.core.project import Project
    p = Project.new('T')
    p.doc['rigs'][0]['sensors'][0].update(width=160, height=90)
    r = rsu_project(p)
    out = tmp_path_factory.mktemp('v2x') / 'run'
    res = drive(p, out, rsu=r)
    assert res.returncode == P.OK, res.stderr[-3000:]
    return str(out)


def _jsonl(path):
    return [json.loads(x) for x in open(path)]


def test_drive_records_the_roadside_sensors_with_labels(rsu_run):
    import cv2
    out = rsu_run
    run = json.load(open(os.path.join(out, 'run.json')))
    assert run['status'] == 'ok' and run['infrastructure']['stations'][0]['cpms'] == 10      # 20 steps, 10 Hz at 20
    assert run['start_near']['ego_candidates'] > 0
    frames = _jsonl(os.path.join(out, 'frames.jsonl'))
    for kind in ('rgb', 'semantic', 'instance', 'range', 'objects'):
        assert f'rsu-a.cam/{kind}' in frames[3]['files'], kind
    assert any('rsu-a.lidar/lidar' in f['files'] for f in frames) and 'rsu-a.lidar' in frames[1]['lidar_pose']
    ds = json.load(open(os.path.join(out, 'dataset.json')))
    st = ds['infrastructure']['stations'][0]
    assert st['id'] == 'rsu-a' and st['sensors'][0]['pose_station']['z'] == 6.0 and ds['ego']['actor_id'] > 0
    cam = next(s for s in ds['sensors'] if s['id'] == 'rsu-a.cam')
    assert cam['station'] == 'rsu-a' and cam['pose']['z'] == 6.0
    assert next(s for s in ds['sensors'] if s['id'] == 'cam-front')['station'] is None
    sem = cv2.imread(os.path.join(out, frames[5]['files']['rsu-a.cam/semantic']), cv2.IMREAD_UNCHANGED)
    assert sem.shape == (108, 192) and 29 not in np.unique(sem)          # no "ego" class from a roadside camera
    objs = json.load(open(os.path.join(out, frames[5]['files']['rsu-a.cam/objects'])))['objects']
    ego_id = ds['ego']['actor_id']
    assert objs and all(o['box3d']['frame'] == 'vehicle' for o in objs)
    seen_ego = [o for o in objs if o['track_id'] == ego_id]
    assert not seen_ego or seen_ego[0]['class'] == 'car'                 # the ego is a car to the pole
    lc = {c['sensor']: c for c in run['lidar_checks']}
    assert lc['rsu-a.lidar']['match']['exact_slices'] == lc['rsu-a.lidar']['match']['slices'] == 20
    assert lc['rsu-a.lidar']['box_label_share'] is None or lc['rsu-a.lidar']['box_label_share'] >= 0.99


def test_cpms_list_what_the_sensors_saw_and_nothing_hidden(rsu_run):
    import cv2
    out = rsu_run
    lines = _jsonl(os.path.join(out, 'cpm.jsonl'))
    assert [x['step'] for x in lines] == list(range(0, 20, 2))
    frames = {f['step']: f for f in _jsonl(os.path.join(out, 'frames.jsonl'))}
    actors = {a['step']: {x['id']: x for x in a['actors']} for a in _jsonl(os.path.join(out, 'actors.jsonl'))}
    for line in lines:
        assert C.decode(base64.b64decode(line['uper'])) == line['cpm'] and line['bytes'] == len(
            base64.b64decode(line['uper']))
        cont = C.containers(line['cpm'])
        objs = cont['perceivedObjectContainer']['perceivedObjects']
        assert len(objs) == len(line['objects']) == cont['perceivedObjectContainer']['numberOfPerceivedObjects']
        # every reported object: at least 20 instance pixels (camera, sensor 1) or 3 lidar points (sensor 2)
        for o, x in zip(objs, line['objects']):
            ev = {int(k): n for k, n in x['evidence'].items()}
            assert ev.get(1, 0) >= 20 or ev.get(2, 0) >= 3
            assert sorted(ev) == o['sensorIdList']
            a = actors[line['step']][x['actor_id']]
            # the ground truth in the station's ENU frame: x East = CARLA x − 20, y North = −(CARLA y − 8)
            ctr = np.array([a['x'], a['y']])        # actor origin; the box centre is within its extent
            assert abs(o['position']['xCoordinate']['value'] / 100 - (ctr[0] - 20.0)) <= a['extent'][0] + 0.05
            assert abs(o['position']['yCoordinate']['value'] / 100 + (ctr[1] - 8.0)) <= a['extent'][0] + 0.05
            vel = o['velocity']['cartesianVelocity']                    # actors.jsonl rounds to mm/s
            assert abs(vel['xVelocity']['value'] - a['vx'] * 100) <= 1 and abs(vel['yVelocity']['value'] + a['vy'] * 100) <= 1
        # nothing hidden from the camera is reported by it: an actor whose id has no pixel in the recorded instance
        # image of that step carries no camera evidence
        f = frames[line['step']]
        inst = cv2.imread(os.path.join(out, f['files']['rsu-a.cam/instance']), cv2.IMREAD_UNCHANGED)
        ob = json.load(open(os.path.join(out, f['files']['rsu-a.cam/objects'])))['objects']
        px = {o['track_id']: o['pixels'] for o in ob if isinstance(o['track_id'], int)}
        assert inst is not None
        for x in line['objects']:
            if '1' in x['evidence']:
                assert px.get(x['actor_id'], 0) >= 20
        for aid, n in px.items():
            if n >= 20 and aid in actors[line['step']]:
                assert aid in {x['actor_id'] for x in line['objects']}
        # the sensors' fields of view and free space
        si = cont['sensorInformationContainer']
        assert [s['sensorType'] for s in si] == [3, 2]
        reg = cont['perceptionRegionContainer']
        assert [len(g['perceptionRegionShape']['radialShapes']['radialShapesList']) for g in reg] == [16, 16]
        for g in reg:
            assert all(0 <= d['range'] <= 1000 for d in g['perceptionRegionShape']['radialShapes']['radialShapesList'])


def test_benefit_and_decode_commands(rsu_run, tmp_path):
    from falcon_autonomy.cli import main
    b = V.benefit(rsu_run, radius_m=60.0)
    s = b['shares']['all']
    assert s['road_user_steps'] > 0 and 0.0 <= s['ego_share'] <= s['ego_or_rsu_share'] <= 1.0
    assert b['evaluated_steps'] == 10
    for only in ('camera', 'lidar'):                          # a rig with fewer sensors sees no more
        o = V.benefit(rsu_run, radius_m=60.0, ego_sensors=only)['shares']['all']
        assert o['ego_share'] <= s['ego_share'] and o['road_user_steps'] == s['road_user_steps']
    vis = V.ego_visibility(rsu_run)
    ds = json.load(open(os.path.join(rsu_run, 'dataset.json')))
    assert all(ds['ego']['actor_id'] not in v for v in vis.values())
    assert main(['v2x', 'decode', rsu_run]) == 0
    assert main(['v2x', 'benefit', rsu_run, rsu_run, '--out', str(tmp_path / 'b.json')]) == 0
    doc = json.load(open(tmp_path / 'b.json'))
    assert len(doc['runs']) == 2 and doc['pooled']['all']['ego_share'] == s['ego_share']


def test_mcap_export_carries_the_cpms(rsu_run, tmp_path):
    mcap_reader = pytest.importorskip('mcap.reader', reason='the mcap package is not installed')
    from falcon_autonomy.core import exporters as E
    from falcon_autonomy.core import ros2msg as R
    out = str(tmp_path / 'bag')
    E.export([rsu_run], 'mcap', out, {})
    bags = [os.path.join(d, f) for d, _, fs in os.walk(out) for f in fs if f.endswith('.mcap')]
    assert len(bags) == 1
    with open(bags[0], 'rb') as fh:
        r = mcap_reader.make_reader(fh)
        got = [(ch.topic, sc.name, m.data) for sc, ch, m in r.iter_messages(topics=['/falcon/v2x/rsu_a/cpm',
                                                                                     '/falcon/v2x/rsu_a/cpm_json'])]
    udp = [d for t, n, d in got if t.endswith('/cpm')]
    assert len(udp) == 10 and {n for _, n, _ in got} == {'udp_msgs/msg/UdpPacket', 'std_msgs/msg/String'}
    m = R.decode('udp_msgs/msg/UdpPacket', udp[0])
    assert m['data'][:4] == bytes([0x07, 0xD9, 0, 0])
    first = _jsonl(os.path.join(rsu_run, 'cpm.jsonl'))[0]
    assert C.decode(m['data'][4:]) == first['cpm']


# ------------------------------------------------------------------------------------------------ the stack bridge
def test_topic_plan_and_converter_command(project):
    from falcon_autonomy.core import stackbridge as SB
    plan = SB.topic_plan(project.doc['rigs'][0], SB.bridge_defaults(), ['rsu-a'])
    cpm = [e for e in plan if e.get('station') == 'rsu-a']
    assert [(e['topic'], e['type']) for e in cpm] == [('/falcon/v2x/rsu_a/cpm', 'udp_msgs/msg/UdpPacket'),
                                                      ('/falcon/v2x/rsu_a/cpm_json', 'std_msgs/msg/String')]
    assert not [e for e in SB.topic_plan(project.doc['rigs'][0], dict(SB.bridge_defaults(), v2x=False), ['rsu-a'])
                if e.get('station')]
    cmd = SB.cpm_converter_command('rsu-a')
    assert cmd[:3] == ['docker', 'run', '--rm'] and '~/udp/in:=/falcon/v2x/rsu_a/cpm' in cmd
    assert '~/cpm_ts/out:=/falcon/v2x/rsu_a/cpm_ts' in cmd


def test_a_stack_run_publishes_the_cpms(project, tmp_path):
    sys.path.insert(0, os.path.join(ROOT, 'tests'))
    import test_stack as TS
    from falcon_autonomy.core import ros2msg as R
    rsu_project(project, rate=20.0)
    cfg, port = TS.ros2_config(project, tmp_path, values={'speed': 30.0, 'distance': 60.0, 'kind': 'car'})
    assert cfg['infrastructure']['rsus'][0]['id'] == 'rsu-a'
    bus, res = TS.Bus(speed=30.0 / 3.6), {}
    seen = []
    deliver = bus.deliver

    def tap(topic, data):
        if topic.startswith('/falcon/v2x/'):
            seen.append((topic, R.decode(bus.types[topic], data)))
            return
        deliver(topic, data)
    bus.deliver = tap
    th = threading.Thread(target=TS.serve, args=(port, bus, res), daemon=True)
    th.start()
    r, run = TS.run_worker(cfg, tmp_path)
    th.join(30)
    assert r.returncode == P.OK, r.stderr[-3000:] + json.dumps(run.get('error'))
    n = run['performance']['steps']
    assert run['infrastructure']['stations'][0]['cpms'] == n and run['bridge']['missed'] == 0
    udp = [m for t, m in seen if t == '/falcon/v2x/rsu_a/cpm']
    js = [json.loads(m['data']) for t, m in seen if t == '/falcon/v2x/rsu_a/cpm_json']
    assert len(udp) == len(js) == n
    assert C.decode(udp[5]['data'][4:]) == js[5]['cpm']
    assert udp[5]['header']['stamp'] == bus.first['/clock']['clock'] or udp[5]['header']['stamp']['sec'] >= 0
    lines = _jsonl(os.path.join(tmp_path, 'run', 'cpm.jsonl'))
    assert len(lines) == n and lines[5]['cpm'] == js[5]['cpm']


def test_the_node_reports_message_types_it_lacks():
    sys.path.insert(0, os.path.join(ROOT, 'falcon_autonomy', 'ros2'))
    import falcon_bridge_node as NODE

    class Adapter:
        def publisher(self, topic, typ, qos):
            if typ.startswith('udp_msgs/'):
                raise ModuleNotFoundError("No module named 'udp_msgs'")
            return lambda data: None

        def subscribe(self, topic, typ, cb):
            return True, ''

        def info(self):
            return {}
    sent = []
    core = NODE.BridgeCore(Adapter(), lambda h, p=b'': sent.append(h))
    core.handle({'t': 'hello', 'protocol': 1, 'topics': [
        {'topic': '/clock', 'type': 'rosgraph_msgs/msg/Clock'},
        {'topic': '/falcon/v2x/rsu_a/cpm', 'type': 'udp_msgs/msg/UdpPacket'}], 'control': None}, b'')
    assert sent[0]['t'] == 'ready' and [m['topic'] for m in sent[0]['missing']] == ['/falcon/v2x/rsu_a/cpm']


# ------------------------------------------------------------------------------------------------ the GUI
from test_gui import pump, qapp, slot_exceptions, window  # noqa: E402,F401


@pytest.mark.gui
def test_worlds_page_places_and_edits_roadside_units(window, qapp):
    page = window.go('worlds')
    page.list.setCurrentRow(0)
    sec = page.infra
    assert sec.list.count() == 0 and not sec.add_j_btn.isEnabled()          # a town: no junctions offline
    r = sec.add_at_position()
    pump(qapp, timeout=0.2)
    doc = window.project.doc
    assert V.rsus(doc, 'world-town10')[0]['id'] == r['id'] and sec.list.count() == 1
    assert sec.map.rsus and sec.map.sel == 0
    sec.spins['height_m'].setValue(8.0)
    sec.rate.setValue(5.0)
    sec.n_pos.setValue(0.25)
    pump(qapp, timeout=0.1)
    got = V.rsus(window.project.doc)[0]
    assert got['height_m'] == 8.0 and got['cpm'] == {'rate_hz': 5.0, 'noise': {'position_m': 0.25}}
    from falcon_autonomy.core.project import validate
    assert validate(window.project.doc) == []
    window.undo.undo()
    pump(qapp, timeout=0.1)
    assert 'noise' not in V.rsus(window.project.doc)[0].get('cpm', {})
    sec.list.setCurrentRow(0)
    sec.remove()
    pump(qapp, timeout=0.1)
    assert V.rsus(window.project.doc) == [] and sec.list.count() == 0


@pytest.mark.gui
def test_worlds_page_places_at_a_junction_of_an_imported_world(window, qapp):
    """An OpenDRIVE world (a SUMO conversion with three junctions, tests/data/sumo_small.xodr) in the workspace: its
    junctions are offered, a pole goes to a corner of the chosen one and shows on its road map."""
    from falcon_autonomy.core import xodr as X
    from falcon_autonomy.core.project import Edit
    d = os.path.join(window.workspace(), 'worlds', 'w-sumo')
    os.makedirs(d, exist_ok=True)
    shutil.copy(SMALL_XODR, os.path.join(d, 'world.xodr'))
    with open(os.path.join(d, 'graph.json'), 'w') as fh:
        json.dump(X.road_graph(X.parse(open(SMALL_XODR).read())), fh)
    window.edit(Edit('insert', ['worlds'], {'id': 'w-sumo', 'name': 'SUMO sample', 'kind': 'xodr',
                                            'xodr': 'worlds/w-sumo/world.xodr', 'weather': {'preset': 'ClearNoon'},
                                            'traffic_profile': 'tp-default'}, text='add'))
    page = window.go('worlds')
    page.list.setCurrentRow(page.list.count() - 1)
    sec = page.infra
    pump(qapp, timeout=0.2)
    assert sec.junction.count() == 3 and sec.add_j_btn.isEnabled()
    sec.junction.setCurrentIndex(1)
    r = sec.add_at_junction()
    assert r is not None and r['junction']['id'] == sec.junction.currentData()
    pump(qapp, timeout=0.2)
    assert V.rsus(window.project.doc, 'w-sumo')[0]['junction']['id'] == r['junction']['id']
    assert sec.map.lines and sec.map.rsus and sec.list.count() == 1
    r2 = sec.add_at_junction()                             # a second pole at the same junction: the next corner
    assert r2['junction']['corner'] != r['junction']['corner'] or len(V.corners(V.junctions(
        open(SMALL_XODR).read())[1], open(SMALL_XODR).read())) == 1


# ------------------------------------------------------------------------------------------------ asn1c (ROS 2)
CONVERT_CHECK = r'''
import json, sys, threading, time
import rclpy
from rclpy.node import Node
from rosidl_runtime_py.convert import message_to_ordereddict
from udp_msgs.msg import UdpPacket
from etsi_its_cpm_ts_msgs.msg import CollectivePerceptionMessage
payload = bytes.fromhex(sys.argv[1])
rclpy.init()
n = Node('falcon_cpm_check')
got = []
n.create_subscription(CollectivePerceptionMessage, '/falcon/v2x/rsu_a/cpm_ts', lambda m: got.append(m), 10)
pub = n.create_publisher(UdpPacket, '/falcon/v2x/rsu_a/cpm', 10)
t0 = time.time()
while not got and time.time() - t0 < 40:
    m = UdpPacket()
    m.data = list(payload)
    pub.publish(m)
    rclpy.spin_once(n, timeout_sec=0.5)
print(json.dumps(message_to_ordereddict(got[0]) if got else None, default=str))
'''


@pytest.mark.ros2
def test_etsi_its_conversion_decodes_falcons_cpm(tmp_path):
    """etsi_its_messages' conversion node (its asn1c-generated UPER decoder) reads Falcon's BTP + UPER payload and
    publishes etsi_its_cpm_ts_msgs/CollectivePerceptionMessage with the same values."""
    from falcon_autonomy.core import stackbridge as SB
    image = os.environ.get('FALCON_ROS2_IMAGE', SB.IMAGE)
    probe = subprocess.run(['docker', 'run', '--rm', image, 'bash', '-lc', 'ros2 pkg prefix etsi_its_conversion'],
                           capture_output=True, text=True, timeout=120)
    if probe.returncode != 0:
        pytest.skip(f'{image} has no etsi_its_conversion (rebuild it: falcon stack image)')
    name = f'falcon-test-cpm-{os.getpid()}'
    dom = 77
    conv = SB.cpm_converter_command('rsu-a', name, dom)
    conv[conv.index(SB.IMAGE)] = image
    proc = subprocess.Popen(conv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        m = sample_cpm()
        payload = C.btp_payload(C.encode(m))
        chk = SB.docker_run(name + '-pub', ['python3', '-c', CONVERT_CHECK, payload.hex()], dom, image=image)
        r = subprocess.run(chk, capture_output=True, text=True, timeout=180)
        assert r.returncode == 0, r.stderr[-3000:]
        got = json.loads(r.stdout.strip().splitlines()[-1])
        assert got is not None, 'the conversion node published nothing'
    finally:
        subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=60)
        proc.wait(30)
    same_as_ros(m, got)                     # every value Falcon wrote, as the asn1c decoder read it


def _snake(k: str) -> str:
    import re
    return re.sub('([A-Z])', r'_\1', k).lower()


def _unwrap(r):
    while isinstance(r, dict) and set(r) == {'value'}:
        r = r['value']
    return r


def same_as_ros(f, r, path='cpm'):
    """A Falcon CPM dict against etsi_its_messages' ROS message (message_to_ordereddict): each named type wraps its
    value in 'value', SEQUENCE OF in 'array', a CHOICE has 'choice' and its alternative's field, an OPTIONAL component
    '<name>_is_present', a WrappedCpmContainer 'container_data_<container>'."""
    r = _unwrap(r)
    if isinstance(f, bool) or isinstance(f, int):
        assert r == f, f'{path}: {r} != {f}'
    elif isinstance(f, str):                                      # an ENUMERATED: its index
        assert isinstance(r, int), path
    elif isinstance(f, list):
        arr = r['array'] if isinstance(r, dict) else r
        assert len(arr) == len(f), f'{path}: {len(arr)} items != {len(f)}'
        for k, (a, b) in enumerate(zip(f, arr)):
            same_as_ros(a, b, f'{path}[{k}]')
    elif set(f) == {'containerId', 'containerData'}:
        names = {v: k for k, v in C.CONTAINER_IDS.items()}
        assert _unwrap(r['container_id']) == f['containerId'], path
        same_as_ros(f['containerData'], r['container_data_' + _snake(names[f['containerId']])], path + '.data')
    elif 'choice' in r:
        (k, v), = f.items()
        same_as_ros(v, r[_snake(k)], f'{path}.{k}')
    else:
        for k, v in f.items():
            rk = _snake(k)
            assert rk in r, f'{path}.{k} missing in {sorted(r)[:8]}'
            if rk + '_is_present' in r:
                assert r[rk + '_is_present'], f'{path}.{k} not present'
            same_as_ros(v, r[rk], f'{path}.{k}')
