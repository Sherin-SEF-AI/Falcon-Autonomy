"""Workers against the fake carla: the drive's outputs, determinism, exit codes, cancel, the watchdog, tidy-up."""
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np
import pytest

from conftest import worker_env
from falcon_autonomy.core import protocol as P
from falcon_autonomy.core.drive import drive_config
from falcon_autonomy.core.reproduce import compare_drives


def _cfg(project, out, **kw):
    cfg = drive_config(project, 'world-town10', 'rig-default', str(out), 2310, 8310, seconds=kw.pop('seconds', 1.0),
                       seed=kw.pop('seed', 1))
    cfg.update(kw)
    p = os.path.join(os.path.dirname(str(out)), os.path.basename(str(out)) + '.json')
    with open(p, 'w') as fh:
        json.dump(cfg, fh)
    return p


def _run(cfg_path, env=None, timeout=120):
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy.workers', 'drive', cfg_path], capture_output=True,
                       text=True, env=worker_env(env), timeout=timeout)
    return r.returncode, [P.parse_line(x) for x in r.stdout.splitlines()], r.stderr


def test_drive_records_frames_labels_and_run_json(project, tmp_path):
    code, msgs, err = _run(_cfg(project, tmp_path / 'run'))
    assert code == P.OK, err
    out = tmp_path / 'run'
    run = json.loads((out / 'run.json').read_text())
    assert run['status'] == 'ok' and run['exit_code'] == 0 and run['provenance']['seed'] == 1
    assert run['provenance']['host']['carla_server'] == '0.9.16'
    perf = run['performance']
    assert perf['steps'] == 20 and perf['source'] == 'measured' and perf['mpx_per_step'] == pytest.approx(160 * 90 * 3 / 1e6, abs=1e-3)
    frames = [json.loads(x) for x in (out / 'frames.jsonl').read_text().splitlines()]
    assert [f['step'] for f in frames] == list(range(20))
    assert all(b['frame'] == a['frame'] + 1 for a, b in zip(frames, frames[1:]))       # one tick per step
    assert frames[1]['sim_t'] == pytest.approx(0.05)
    import cv2
    rgb = cv2.imread(str(out / frames[5]['files']['cam-front/rgb']))
    sem = cv2.imread(str(out / frames[5]['files']['cam-front/semantic']), cv2.IMREAD_UNCHANGED)
    assert rgb.shape == (90, 160, 3) and sem.shape == (90, 160) and sem.dtype == np.uint8
    assert {1, 11} <= set(np.unique(sem).tolist())                 # the fake's road and sky
    ds = json.loads((out / 'dataset.json').read_text())
    assert ds['frames'] == 20 and ds['sensors'][0]['streams'] == ['instance', 'objects', 'range', 'rgb', 'semantic']
    assert ds['sensors'][0]['K'][0][0] == pytest.approx(80.0)    # f = W / (2 tan(fov / 2)) at 90°
    kinds = [m['t'] for m in msgs]
    assert kinds[0] == 'hello' and kinds[-1] == 'result' and 'progress' in kinds and 'metric' in kinds


def test_labels_are_the_same_tick_as_the_image(project, tmp_path):
    """The fake's RGB draws the sky blue (B 230) and everything else not: the semantic image's sky (tag 11) must be
    the RGB's sky pixel for pixel (JPEG aside), frame by frame: the label camera is at the image's pose and tick."""
    project.doc['rigs'][0]['sensors'][0]['pose'].update(pitch=8.0)        # horizon in view
    code, _, err = _run(_cfg(project, tmp_path / 'run'))
    assert code == P.OK, err
    import cv2
    out = tmp_path / 'run'
    frames = [json.loads(x) for x in (out / 'frames.jsonl').read_text().splitlines()]

    def sky_rgb(f):
        im = cv2.imread(str(out / f['files']['cam-front/rgb']))
        return (im[:, :, 0] > 210) & (im[:, :, 1] > 145)

    def sky_sem(f):
        return cv2.imread(str(out / f['files']['cam-front/semantic']), cv2.IMREAD_UNCHANGED) == 11
    for f in frames[2::6]:
        assert sky_sem(f).any() and np.mean(sky_rgb(f) == sky_sem(f)) > 0.98


def test_same_seed_reproduces_bit_for_bit(project, tmp_path):
    assert _run(_cfg(project, tmp_path / 'a', seed=4))[0] == P.OK
    assert _run(_cfg(project, tmp_path / 'b', seed=4))[0] == P.OK
    rep = compare_drives(str(tmp_path / 'a'), str(tmp_path / 'b'), {'pose_tol': 0.0, 'rgb_tol': 0.0,
                                                                     'label_agree': 1.0})
    assert rep['ok'], rep['checks']


def test_over_budget_is_refused_before_connecting(project, tmp_path):
    project.doc['rigs'][0]['sensors'][0].update(width=8192, height=8192, labels=['semantic', 'instance', 'depth'])
    code, msgs, err = _run(_cfg(project, tmp_path / 'run'))
    assert code == P.REFUSED and 'budget' in err
    run = json.loads((tmp_path / 'run' / 'run.json').read_text())
    assert run['status'] == 'refused' and os.listdir(tmp_path / 'run') == ['run.json']


def test_no_server_exits_no_server(project, tmp_path):
    code, _, err = _run(_cfg(project, tmp_path / 'run', connect_wait_s=2.0), {'FAKE_CARLA_NO_SERVER': '1'})
    assert code == P.NO_SERVER and 'no CARLA server answered' in err


def test_watchdog_ends_a_worker_whose_server_hangs(project, tmp_path):
    """A crashed server leaves the client hanging in its call: the watchdog writes run.json and exits SERVER_LOST."""
    cfg = _cfg(project, tmp_path / 'run', timeout=1.0, seconds=5.0)
    t0 = time.monotonic()
    code, _, err = _run(cfg, {'FAKE_CARLA_HANG_AT_TICK': '15'}, timeout=60)
    assert code == P.SERVER_LOST and time.monotonic() - t0 < 30
    run = json.loads((tmp_path / 'run' / 'run.json').read_text())
    assert run['status'] == 'server_lost' and 'stopped answering' in run['error']


def test_sigterm_cancels_tidies_and_leaves_only_run_json(project, tmp_path):
    cfg = _cfg(project, tmp_path / 'run', seconds=60.0)
    p = subprocess.Popen([sys.executable, '-m', 'falcon_autonomy.workers', 'drive', cfg], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, env=worker_env())
    for line in p.stdout:                                # wait until it is recording
        if '"step ' in line:
            break
    # numpy (and its BLAS threads) are loaded by now: the signal must still reach the watchdog, not kill the worker
    p.send_signal(signal.SIGTERM)
    p.wait(60)
    assert p.returncode == P.CANCELLED
    run = json.loads((tmp_path / 'run' / 'run.json').read_text())
    assert run['status'] == 'cancelled' and os.listdir(tmp_path / 'run') == ['run.json']


def test_stdin_cancel(project, tmp_path):
    cfg = _cfg(project, tmp_path / 'run', seconds=60.0)
    p = subprocess.Popen([sys.executable, '-m', 'falcon_autonomy.workers', 'drive', cfg], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=worker_env())
    for line in p.stdout:
        if '"step ' in line:
            break
    p.stdin.write('{"cmd": "cancel"}\n')
    p.stdin.flush()
    p.wait(60)
    assert p.returncode == P.CANCELLED


def test_tidy_destroys_in_a_safe_order(fake_module):
    """Children before parents, vehicles that were on the autopilot by id (Actor.destroy() on one aborts libcarla),
    and the world back in asynchronous mode."""
    from falcon_autonomy.workers.common import Worker
    carla = fake_module
    w = Worker(['x'])
    w.import_carla()
    w.client = carla.Client('127.0.0.1', 2310)
    w.world = world = w.client.get_world()
    st = world.get_settings()
    st.synchronous_mode, st.fixed_delta_seconds = True, 0.05
    world.apply_settings(st)
    w.tm = w.client.get_trafficmanager(8310)
    lib = world.get_blueprint_library()
    sp = world.get_map().get_spawn_points()
    ego = w.own(world.spawn_actor(lib.find('vehicle.lincoln.mkz_2020'), sp[0]))
    ego.set_autopilot(True, 8310)
    cam = w.own(world.spawn_actor(lib.find('sensor.camera.rgb'), carla.Transform(), attach_to=ego))
    cam.listen(lambda d: None)
    w.tidy()
    gone = [e[1] for e in carla.LOG if e[0] == 'destroy']
    assert gone == [cam.id, ego.id]
    assert world.get_settings().synchronous_mode is False and world.get_settings().fixed_delta_seconds is None
    assert ('tm_sync', False) in carla.LOG


def test_tidy_after_a_killed_worker(fake_module):
    from falcon_autonomy.workers.tidy import tidy_server
    carla = fake_module
    c = carla.Client('127.0.0.1', 2310)
    world = c.get_world()
    lib = world.get_blueprint_library()
    sp = world.get_map().get_spawn_points()
    bp = lib.find('vehicle.lincoln.mkz_2020')
    bp.set_attribute('role_name', 'hero')
    ego = world.spawn_actor(bp, sp[0])
    s = lib.find('sensor.camera.rgb')
    s.set_attribute('role_name', 'falcon_sensor')
    world.spawn_actor(s, carla.Transform(), attach_to=ego)
    other = lib.find('vehicle.audi.a2')
    other.set_attribute('role_name', 'someone_elses')
    keep = world.spawn_actor(other, sp[1])
    st = world.get_settings()
    st.synchronous_mode = True
    world.apply_settings(st)
    res = tidy_server(c, world)
    assert res['destroyed'] == 2 and res['was_synchronous'] and keep.is_alive
    assert not world.get_settings().synchronous_mode


def test_probe(tmp_path):
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy.workers', 'probe', '--port', '2310'],
                       capture_output=True, text=True, env=worker_env(), timeout=60)
    assert r.returncode == 0
    info = [P.parse_line(x) for x in r.stdout.splitlines() if '"server"' in x][0]
    assert info['server_version'] == '0.9.16' and 'Town10HD_Opt' in info['maps']


def test_unknown_worker_is_a_usage_error():
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy.workers', 'nope'], capture_output=True, text=True,
                       env=worker_env(), timeout=60)
    assert r.returncode == P.FAILED and 'usage' in r.stderr


def test_left_hand_profile_is_refused_on_a_right_hand_town(project):
    from falcon_autonomy.core.project import ProjectError
    from falcon_autonomy.core.traffic import preset
    project.doc['traffic_profiles'].append(preset('tp-india-mixed'))
    project.doc['worlds'][0]['traffic_profile'] = 'tp-india-mixed'
    with pytest.raises(ProjectError, match='left-hand traffic but the map'):
        drive_config(project, 'world-town10', 'rig-default', '/tmp/x', 2310, 8310)


def test_indian_fleet_on_an_imported_left_hand_world(project, tmp_path):
    """An imported (OpenDRIVE) world: generated from its file, the Indian fleet drawn by its weights (the autorickshaw
    substitute included), walkers on the navigation mesh."""
    from falcon_autonomy.core.traffic import preset
    ws = tmp_path / 'test.falcon'
    ws.mkdir(exist_ok=True)
    (ws / 'blr.xodr').write_text('<OpenDRIVE><header/></OpenDRIVE>')
    prof = preset('tp-india-mixed')
    prof.update(vehicles=12, walkers=4)
    project.doc['traffic_profiles'].append(prof)
    project.doc['worlds'].append({'id': 'w-blr', 'name': 'Bengaluru', 'kind': 'osm', 'xodr': 'blr.xodr',
                                  'osm': {'handedness': 'LHT'}, 'traffic_profile': 'tp-india-mixed',
                                  'weather': {'preset': 'ClearNoon'}})
    cfg = drive_config(project, 'w-blr', 'rig-default', str(tmp_path / 'run'), 2310, 8310, seconds=0.5, seed=5,
                       workspace=str(ws))
    assert cfg['xodr'].endswith('blr.xodr') and cfg['handedness'] == 'LHT' and not cfg['town']
    p = tmp_path / 'run.json.cfg'
    p.write_text(json.dumps(cfg))
    code, _, err = _run(str(p))
    assert code == P.OK, err
    run = json.loads((tmp_path / 'run' / 'run.json').read_text())
    placed = run['traffic']['placed']
    assert sum(placed.values()) == 12 and placed.get('motorcycle', 0) >= 2
    assert set(placed) <= {'motorcycle', 'bicycle', 'car', 'van', 'bus', 'truck', 'blueprint:vehicle.micro.microlino'}
    assert run['traffic']['walkers'] == 4
    assert run['provenance']['inputs']['xodr']['sha256']


def test_handedness_of_the_fakes_right_hand_road(project, tmp_path):
    cfg = drive_config(project, 'world-town10', 'rig-default', str(tmp_path / 'h'), 2310, 8310, seed=1)
    cfg.update(seconds=8.0, vehicles=6)
    p = tmp_path / 'h.json'
    p.write_text(json.dumps(cfg))
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy.workers', 'handedness', str(p)], capture_output=True,
                       text=True, env=worker_env(), timeout=120)
    assert r.returncode == P.OK, r.stderr
    rep = json.loads((tmp_path / 'h' / 'run.json').read_text())['handedness']
    assert rep['verdict'] == 'RHT' and rep['keep_left_share'] == 0.0 and rep['samples'] > 10
    assert rep['map_left_lanes_along_plus_s_share'] == 0.0


def test_osm_import_of_central_bengaluru_sample(tmp_path):
    """The fixture (300 m of central Bengaluru): cropped, converted, made left-hand, checked against OSM's lanes."""
    from falcon_autonomy.core import osm as O
    from falcon_autonomy.core.worlds import import_config
    src = os.path.join(os.path.dirname(__file__), 'data', 'blr_small.osm')
    d = O.read_osm(src)
    lat0, lon0 = O.summary(d)['centre']
    cfg = import_config(src, str(tmp_path / 'w'), bbox=O.bbox_around(lat0, lon0, 200.0), lht=True)
    p = tmp_path / 'imp.json'
    p.write_text(json.dumps(cfg))
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy.workers', 'osm_import', str(p)], capture_output=True,
                       text=True, env=worker_env(), timeout=120)
    assert r.returncode == P.OK, r.stderr
    rep = json.loads((tmp_path / 'w' / 'import.json').read_text())
    assert rep['handedness'] == 'LHT' and rep['proj'].startswith('+proj=tmerc +lat_0=')
    assert rep['lane_check']['comparable'] >= 3 and rep['lane_check']['agreement'] == 1.0
    assert rep['not_represented']['text'].startswith('an OpenDRIVE world is roads')
    for f in ('area.osm', 'raw.xodr', 'world.xodr', 'graph.json', 'run.json'):
        assert (tmp_path / 'w' / f).is_file()
    assert 'rule="LHT"' in (tmp_path / 'w' / 'world.xodr').read_text()


def test_osm_import_clamps_converter_objects_in_the_loaded_file(tmp_path, monkeypatch):
    """The file CARLA loads (world.xodr, after the left-hand conversion) has no object past its road's end."""
    import xml.etree.ElementTree as ET
    from falcon_autonomy.core import xodr
    from falcon_autonomy.core.worlds import import_config
    src = os.path.join(os.path.dirname(__file__), 'data', 'blr_small.osm')
    cfg = import_config(src, str(tmp_path / 'w'), lht=True)
    p = tmp_path / 'imp.json'
    p.write_text(json.dumps(cfg))
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy.workers', 'osm_import', str(p)], capture_output=True,
                       text=True, env=worker_env({'FAKE_CARLA_OBJECT_PAST_END_M': '0.0035'}), timeout=120)
    assert r.returncode == P.OK, r.stderr
    world = (tmp_path / 'w' / 'world.xodr').read_text()
    assert xodr.sanitize(world)[1] == [] and 'crosswalk' in world
    rep = json.loads((tmp_path / 'w' / 'import.json').read_text())
    assert any('moved from s' in n for n in rep['notes'])


def test_lidar_labels_proven_and_checked(project, tmp_path):
    from falcon_autonomy.core.rig import new_sensor
    rig = project.doc['rigs'][0]
    rig['sensors'][0].update(labels=['semantic', 'depth'], width=192, height=108)
    rig['sensors'].append(new_sensor('lidar', 'lid', 'top', {'x': 0.0, 'z': 1.9}))
    rig['sensors'][-1].update(channels=16, points_per_second=40000, range=60.0, rotation_frequency=10.0,
                              upper_fov=5.0, lower_fov=-20.0)
    rig['sensors'].append(new_sensor('imu', 'imu', 'imu'))
    rig['sensors'].append(new_sensor('gnss', 'gnss', 'gnss'))
    rig['sensors'].append(new_sensor('radar', 'rad', 'front_radar', {'x': 2.3, 'z': 0.6}))
    project.doc['traffic_profiles'][0]['vehicles'] = 8
    code, msgs, err = _run(_cfg(project, tmp_path / 'run', seconds=1.0, seed=3))
    assert code == P.OK, err
    out = tmp_path / 'run'
    run = json.loads((out / 'run.json').read_text())
    lc = run['lidar_checks'][0]
    assert lc['sweeps'] == 10                                  # 20 steps at 20 Hz, 10 Hz rotation
    m = lc['match']
    assert m['exact_slices'] == m['slices'] == 20 and m['max_dir_err_deg'] <= 0.01
    assert lc['box_points'] > 0 and lc['box_label_share'] >= 0.99
    rv = lc['range_vs_camera']
    assert rv['points'] > 50 and rv['median_abs_m'] <= 0.02
    b = np.fromfile(out / 'lidar' / 'lid' / '000003.bin', np.float32).reshape(-1, 4)
    lab = (out / 'lidar' / 'lid' / '000003.labels.bin').read_bytes()
    assert len(lab) == 5 * len(b) and len(b) > 0
    frames = [json.loads(x) for x in (out / 'frames.jsonl').read_text().splitlines()]
    assert any('rad/radar' in f['files'] for f in frames)
    assert len((out / 'imu' / 'imu.jsonl').read_text().splitlines()) >= 19
    assert (out / 'gnss' / 'gnss.jsonl').read_text().count('lat') >= 9


def test_lidar_without_every_setting_drives(project, tmp_path):
    # a hand-written rig lidar may leave settings out (the schema allows it): the run takes the type's defaults
    rig = project.doc['rigs'][0]
    rig['sensors'][0].update(width=192, height=108)
    rig['sensors'].append({'id': 'lid', 'name': 'top', 'type': 'lidar', 'channels': 16, 'points_per_second': 40000,
                           'pose': {'x': 0.0, 'y': 0.0, 'z': 1.9, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}})
    code, _, err = _run(_cfg(project, tmp_path / 'run', seconds=0.5, seed=1))
    assert code == P.OK, err
    run = json.loads((tmp_path / 'run' / 'run.json').read_text())
    assert run['lidar_checks'][0]['sweeps'] == 5                # 10 steps at 20 Hz, the default 10 Hz rotation


def test_faults_in_a_drive(project, tmp_path):
    cam = project.doc['rigs'][0]['sensors'][0]
    cam['faults'] = [{'kind': 'mount_rotation', 'deg': [0, 0, 5.0]}, {'kind': 'drop', 'rate': 0.3},
                     {'kind': 'freeze', 'from_s': 0.3, 'to_s': 0.5}, {'kind': 'noise', 'sigma': 4.0},
                     {'kind': 'clock_offset', 'ms': 40}]
    code, _, err = _run(_cfg(project, tmp_path / 'run', seconds=1.0, seed=2))
    assert code == P.OK, err
    out = tmp_path / 'run'
    run = json.loads((out / 'run.json').read_text())
    fa = run['faults_applied']['cam-front']
    assert fa['frames']['drop'] >= 2 and fa['frames']['noise'] >= 5 and fa['planned'] == cam['faults']
    frames = [json.loads(x) for x in (out / 'frames.jsonl').read_text().splitlines()]
    dropped = [f for f in frames if f.get('faults', {}).get('cam-front') == 'drop']
    assert dropped and all('cam-front/rgb' not in f['files'] for f in dropped)
    frozen = [f for f in frames if f.get('faults', {}).get('cam-front') == 'freeze']
    assert all(f['files']['cam-front/rgb'] == frozen[0]['files']['cam-front/rgb'] for f in frozen)
    assert all(f['stamps']['cam-front'] == pytest.approx(f['sim_t'] + 0.04) for f in frames)
    ds = json.loads((out / 'dataset.json').read_text())
    assert ds['sensors'][0]['pose']['yaw'] == 0.0                # the planned calibration, not the faulty mount


def test_lanes_and_camera_ground_truth(project, tmp_path):
    project.doc['rigs'][0]['sensors'][0]['labels'] = ['semantic', 'boxes2d', 'boxes3d']
    project.doc['traffic_profiles'][0]['vehicles'] = 6
    code, _, err = _run(_cfg(project, tmp_path / 'run', seconds=0.5, seed=4))
    assert code == P.OK, err
    out = tmp_path / 'run'
    lanes = json.loads((out / 'lanes.json').read_text())
    assert lanes['points'] and len(lanes['graph']) == 4
    assert {r[9] for r in lanes['points']} == {'SolidSolid', 'Broken'}
    f = json.loads((out / 'frames.jsonl').read_text().splitlines()[5])
    objs = json.loads((out / f['files']['cam-front/objects']).read_text())['objects']
    cars = [o for o in objs if o['class'] == 'car']
    # actors keep their actor id as their track id; the map's parked cars (level geometry) are 'level:<k>'
    assert cars and all(o['track_id'] == o['actor_id'] if o['actor_id'] is not None else
                        str(o['track_id']).startswith('level:') for o in cars)
    o = cars[0]
    x0, y0, x1, y1 = o['bbox']
    assert 0 <= x0 < x1 <= 160 and 0 <= y0 < y1 <= 90 and o['pixels'] > 0 and o['range_m'] > 0
    assert set(o['box3d']) >= {'center', 'size', 'yaw_deg'} and 0 <= o['occluded'] <= 1


def test_optical_flow_matches_the_cameras_own_motion(project, tmp_path):
    """The flow layer (decoded as measured on CARLA 0.9.16) against the flow predicted from the range image and the
    ego's motion between two frames, on the fake's static ground: each pixel of frame 1 came from where its 3D point
    projects in frame 0's camera."""
    cam = project.doc['rigs'][0]['sensors'][0]
    cam.update(labels=['semantic', 'depth', 'flow'], width=96, height=64, pose=dict(cam['pose'], pitch=-15.0))
    project.doc['traffic_profiles'][0]['vehicles'] = 0
    code, _, err = _run(_cfg(project, tmp_path / 'run', seconds=0.5, seed=1))
    assert code == P.OK, err
    out = tmp_path / 'run'
    frames = [json.loads(x) for x in (out / 'frames.jsonl').read_text().splitlines()]
    f0, f1 = frames[4], frames[5]
    flow = np.load(out / f1['files']['cam-front/flow']).astype(np.float32)
    assert flow.shape == (64, 96, 2)
    import cv2
    from falcon_autonomy.core import lens as L
    from falcon_autonomy.core.rig import carla_rotation
    rng = cv2.imread(str(out / f1['files']['cam-front/range']), cv2.IMREAD_UNCHANGED).astype(np.float64) / 256.0
    plan = L.plan_camera(cam, tables=False)
    rays, _ = L.output_rays(L.intrinsics(cam))
    o2v = L.optical_to_vehicle(cam['pose'])
    mount = np.array([cam['pose'][k] for k in 'xyz'])
    Pv = (rays * rng[..., None]) @ o2v.T + mount                  # vehicle frame at f1

    def T(e):
        return carla_rotation(e['roll'], e['pitch'], e['yaw']), np.array([e['x'], e['y'], e['z']])
    R0, p0 = T(f0['ego'])
    R1, p1 = T(f1['ego'])
    Pw = Pv @ R1.T + p1
    Pc = ((Pw - p0) @ R0 - mount) @ o2v                          # optical axes of the camera at f0
    K = np.asarray(plan['K'])
    u0 = K[0, 0] * Pc[..., 0] / Pc[..., 2] + K[0, 2]
    v0 = K[1, 1] * Pc[..., 1] / Pc[..., 2] + K[1, 2]
    vv, uu = np.mgrid[0:64, 0:96] + 0.5
    m = (rng > 0) & (rng < 200)
    assert m.sum() > 1000
    assert np.median(np.abs(flow[..., 0][m] - (uu - u0)[m])) < 0.1
    assert np.median(np.abs(flow[..., 1][m] - (vv - v0)[m])) < 0.1
