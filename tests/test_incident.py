"""A real incident made a scenario (core.incident) on synthetic data only: a reconstruction's posed frames written by
hand (a car driving along a straight imported road at the null-island origin), boxes projected from known positions,
the extraction, the two scenarios, the OpenSCENARIO round trip, the director's replay on the fake, the search's
scenario file, and the privacy markers."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import worker_env
from falcon_autonomy.core import incident as INC
from falcon_autonomy.core import openscenario as OS
from falcon_autonomy.core import protocol as P
from falcon_autonomy.core import scenarios as SC

LAT0 = LON0 = 0.0                          # synthetic: no real place
CAM_H = 1.4
K = [500.0, 500.0, 320.0, 180.0]
SPEED = 8.0
LANE_Y = -1.75                             # east-north-up: the right lane of a road along +x (right-hand traffic)
XODR = """<?xml version="1.0"?>
<OpenDRIVE><header revMajor="1" revMinor="4"><geoReference><![CDATA[+proj=tmerc +lat_0=0 +lon_0=0 +k=1 +x_0=0 +y_0=0
+ellps=WGS84 +units=m +no_defs]]></geoReference></header>
<road id="1" length="400" junction="-1"><planView><geometry s="0" x="-200" y="0" hdg="0" length="400"><line/></geometry>
</planView><lanes><laneSection s="0"><left><lane id="1" type="driving"><width sOffset="0" a="3.5" b="0" c="0" d="0"/>
</lane></left><center><lane id="0" type="none"/></center><right><lane id="-1" type="driving">
<width sOffset="0" a="3.5" b="0" c="0" d="0"/></lane></right></laneSection></lanes></road></OpenDRIVE>"""


def view_of(e, n):
    """world → camera (OpenCV: x right, y down, z forward) for a camera facing east at (e, n, CAM_H)."""
    R = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    V = np.eye(4)
    V[:3, :3] = R
    V[:3, 3] = -R @ np.array([e, n, CAM_H])
    return V


def pixel(V, p):
    q = V[:3, :3] @ np.asarray(p, float) + V[:3, 3]
    return K[0] * q[0] / q[2] + K[2], K[1] * q[1] / q[2] + K[3]


def lead(t):
    """A car ahead in the ego's lane: 20 m ahead at 8 m/s, braking at 4 m/s² from t = 2 s."""
    if t <= 2.0:
        return 20.0 + SPEED * t
    b = min(t, 4.0) - 2.0                                  # stopped at t = 4 s, 44 m from the ego's start
    return 20.0 + SPEED * 2.0 + SPEED * b - 2.0 * b * b


def ped(t):
    """A pedestrian on the kerb side 30 m ahead of the start, walking across from t = 1 s at 1.5 m/s."""
    return 30.0, LANE_Y - 3.0 + (1.5 * (t - 1.0) if t > 1.0 else 0.0)


@pytest.fixture
def recon(tmp_path):
    d = tmp_path / 'recon' / 'dashcam-synthetic'
    (d / 'posed').mkdir(parents=True)
    frames, det = [], []
    for k in range(40):
        t = k * 0.1
        e = SPEED * t
        V = view_of(e, LANE_Y)
        ego = np.eye(4)
        ego[:3, 3] = [e, LANE_Y, 0.0]
        frames.append({'index': 3 * k, 't': round(t, 3), 'video_t': t, 'K': K, 'size': [640, 360],
                       'viewmat': V.reshape(-1).tolist(), 'ego': ego.reshape(-1).tolist()})
        boxes = []
        near = [lead(t) - 4.6 / 2.0, LANE_Y, 0.0]                     # the car's near face on the road
        u, v = pixel(V, near)
        boxes.append({'box': [u - 30, v - 40, u + 30, v], 'kind': 'car', 'track': 'c1'})
        px, py = ped(t)
        u, v = pixel(V, [px, py, 0.0])
        boxes.append({'box': [u - 5, v - 30, u + 5, v], 'kind': 'pedestrian'})          # no id: joined
        det.append({'index': 3 * k, 'boxes': boxes})
    (d / 'posed' / 'posed.json').write_text(json.dumps({'geo': {'lat0': LAT0, 'lon0': LON0}, 'frames': frames}))
    (d / 'dashcam.json').write_text(json.dumps({'schema': 'falcon.dashcam_recon/1'}))
    dp = tmp_path / 'det.json'
    dp.write_text(json.dumps({'schema': INC.DETECTIONS_SCHEMA, 'frames': det}))
    return str(d), str(dp)


@pytest.fixture
def world_project(tmp_path):
    from falcon_autonomy.core.project import Project
    p = Project.new('inc')
    p.save(str(tmp_path / 'inc.falcon.json'))
    wdir = os.path.join(p.workspace, 'worlds', 'w-test')
    os.makedirs(wdir, exist_ok=True)
    with open(os.path.join(wdir, 'world.xodr'), 'w', encoding='utf-8') as fh:
        fh.write(XODR)
    p.doc['worlds'].append({'id': 'w-test', 'name': 'Synthetic road', 'kind': 'osm',
                            'xodr': 'worlds/w-test/world.xodr',
                            'osm': {'bbox': [-0.01, -0.01, 0.01, 0.01], 'centre': [LAT0, LON0], 'handedness': 'RHT'}})
    p.save(str(tmp_path / 'inc.falcon.json'))
    return p


def test_ground_point_recovers_a_road_point():
    V = view_of(0.0, 0.0)
    u, v = pixel(V, [12.0, -1.0, 0.0])
    p, C = INC.ground_point(K, V, u, v)
    assert p == pytest.approx([12.0, -1.0], abs=1e-6) and C == pytest.approx([0.0, 0.0])
    assert INC.ground_point(K, V, 320.0, 100.0) is None                    # above the horizon


def test_extract_places_the_event_on_the_matched_road(recon, world_project):
    rec, det = recon
    inc = INC.extract(rec, world_project, 'w-test', detections=det)
    assert inc['match']['matched_share'] == 1.0
    ego = inc['ego']['track']
    # CARLA's frame: y = −north; the right lane's centre is at +1.75
    assert ego['y'][0] == pytest.approx(1.75, abs=0.05) and ego['yaw'][0] == pytest.approx(0.0, abs=1.0)
    assert ego['x'][-1] - ego['x'][0] == pytest.approx(SPEED * 3.9, abs=0.3)
    kinds = sorted(a['kind'] for a in inc['actors'])
    assert kinds == ['car', 'pedestrian']
    car = next(a for a in inc['actors'] if a['kind'] == 'car')
    # the car's centre 20 m ahead of the ego at t = 0, on the ego's lane
    assert car['track']['x'][0] - ego['x'][0] == pytest.approx(20.0, abs=0.3)
    assert car['track']['y'][0] == pytest.approx(1.75, abs=0.2)
    f = {a['kind']: a for a in inc['fit']['actors']}
    assert f['car']['place']['lane'] == 'ego' and f['car']['place']['ahead_m'] == pytest.approx(20.0, abs=0.5)
    assert f['car']['speed_kmh'] == pytest.approx(SPEED * 3.6, abs=1.5)
    brake = next(x for x in f['car']['actions'] if x['do'] == 'speed')
    assert brake['t_s'] == pytest.approx(2.0, abs=0.4) and brake['accel'] > 2.0
    assert f['pedestrian']['heading'] == 'across' and f['pedestrian']['place']['lane'] in ('kerb', 'sidewalk')
    walk = next(x for x in f['pedestrian']['actions'] if x['do'] == 'walk')
    assert walk['t_s'] == pytest.approx(1.1, abs=0.3) and walk['speed_mps'] == pytest.approx(1.5, abs=0.3)


def test_window_and_refusals(recon, world_project):
    rec, det = recon
    inc = INC.extract(rec, world_project, 'w-test', start=1.0, end=2.5, detections=det)
    assert inc['duration_s'] == pytest.approx(1.5) and inc['ego']['track']['t'][0] == 0.0
    with pytest.raises(INC.IncidentError, match='fewer than two'):
        INC.extract(rec, world_project, 'w-test', start=10.0, end=11.0)
    with pytest.raises(INC.IncidentError, match='detections file'):
        INC.read_detections({'frames': []})


def test_scenarios_validate_and_centre_their_ranges(recon, world_project):
    rec, det = recon
    inc = INC.extract(rec, world_project, 'w-test', detections=det)
    rp, vr = INC.scenarios(inc, 'Test event')
    assert not SC.validate(rp) and not SC.validate(vr)
    assert rp['ego']['driver'] == 'replay' and rp['start']['x'] == inc['ego']['track']['x'][0]
    assert all(a.get('track') for a in rp['actors']) and rp['source']['format'] == 'dashcam'
    for name, p in vr['params'].items():
        assert p['dist'] == 'uniform' and p['lo'] <= p['value'] <= p['hi'], name
        if p['lo'] > 0:                                                     # not clipped at a bound: centred
            assert (p['lo'] + p['hi']) / 2 == pytest.approx(p['value'], abs=1e-6), name
    space = {d['name'] for d in SC.search_space(vr)}
    assert 'ego_speed' in space and any(n.endswith('_ahead') for n in space)
    conc = SC.sample(vr, 3)
    assert isinstance(conc['actors'][0]['place']['ahead_m'], float)


def test_openscenario_round_trip_keeps_start_route_and_tracks(recon, world_project):
    rec, det = recon
    inc = INC.extract(rec, world_project, 'w-test', detections=det)
    rp, _ = INC.scenarios(inc, 'Test event')
    text = OS.export_xosc(SC.concrete(rp, {}))
    assert '<FollowTrajectoryAction>' in text and '<WorldPosition' in text and '<AssignRouteAction>' in text
    back = OS.import_xosc(text)
    assert back['start']['x'] == pytest.approx(rp['start']['x'], abs=1e-3)
    assert back['start']['y'] == pytest.approx(rp['start']['y'], abs=1e-3)
    assert back['ego']['driver'] == 'replay'
    assert back['ego']['track']['x'] == pytest.approx(rp['ego']['track']['x'], abs=1e-3)
    car = next(a for a in back['actors'] if a['kind'] == 'car')
    orig = next(a for a in rp['actors'] if a['kind'] == 'car')
    assert car['track']['y'] == pytest.approx(orig['track']['y'], abs=1e-3)
    assert car['place']['lane'] == orig['place']['lane'] and back['timeline'] == []
    assert len(back['route_xy']) >= 2
    # a scenario without tracks exports as before
    assert '<FollowTrajectoryAction>' not in OS.export_xosc(SC.sample(SC.builtin('lead_brake'), 1))


def test_write_marks_the_folder_private_and_summary_has_no_positions(recon, world_project):
    from falcon_autonomy.assistant.privacy import private_reason
    rec, det = recon
    inc = INC.extract(rec, world_project, 'w-test', detections=det)
    name = INC.default_name(rec)
    assert 'synthetic' not in name and name.startswith('incident-')
    out = INC.incident_dir(world_project.workspace, name)
    summ = INC.write(inc, out, name)
    assert sorted(os.listdir(out)) == ['incident.json', 'incident.xosc', 'run.json', 'scenario.json',
                                       'variations.json']
    assert private_reason(out) and private_reason(os.path.join(out, 'scenario.json'))
    assert summ['actors'] == 2 and summ['kinds'] == {'car': 1, 'pedestrian': 1}
    flat = json.dumps(summ)
    assert 'x' not in summ and 'lat' not in flat and 'track' not in flat
    # a drive of the scenario, and a study of its variations, are private too
    run = os.path.join(world_project.workspace, 'runs', 'r1')
    os.makedirs(run)
    with open(os.path.join(run, 'drive.json'), 'w', encoding='utf-8') as fh:
        json.dump({'scenario': json.load(open(os.path.join(out, 'scenario.json')))}, fh)
    assert private_reason(run)


def _drive(project, tmp_path, scn, name, values=None):
    from falcon_autonomy.core.drive import scenario_config
    project.doc['rigs'][0]['sensors'][0].update(width=64, height=36, labels=[])
    cfg = scenario_config(project, scn, 'world-town10', 'rig-default', str(tmp_path / name), 2310, 8310, seed=1,
                          values=values)
    p = tmp_path / f'{name}.json'
    p.write_text(json.dumps(cfg))
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy.workers', 'drive', str(p)], capture_output=True,
                       text=True, env=worker_env(), timeout=180)
    return r, json.loads((tmp_path / name / 'run.json').read_text())


def _fake_road_event():
    """An event on the fake's straight road (Town10 on the fake: a road along +x through y = 0)."""
    t = [round(0.1 * k, 3) for k in range(31)]
    ego = {'t': t, 'x': [8.0 * s for s in t], 'y': [0.0] * len(t), 'yaw': [0.0] * len(t)}
    car = {'t': t[5:], 'x': [25.0 + 6.0 * s for s in t[5:]], 'y': [0.0] * len(t[5:]), 'yaw': [0.0] * len(t[5:])}
    return {'schema': SC.SCENARIO_SCHEMA, 'id': 'ev', 'name': 'Event', 'needs': [], 'duration_s': 3.0,
            'source': dict(INC.SOURCE), 'start': {'x': 0.0, 'y': 0.0, 'yaw': 0.0},
            'route_xy': [[0.0, 0.0], [30.0, 0.0]], 'params': {}, 'motion': 'kinematic',
            'ego': {'speed_kmh': 28.8, 'driver': 'replay', 'route': 'straight', 'track': ego},
            'actors': [{'id': 'a1', 'kind': 'car', 'place': {'lane': 'ego', 'ahead_m': 25.0, 'lateral_m': 0.0},
                        'heading': 'lane', 'speed_kmh': 21.6, 'start': 'moving', 'track': car}],
            'timeline': [], 'pass': {'no_collision': True}}


def test_director_replays_tracks_at_the_start_on_the_fake(project, tmp_path):
    scn = _fake_road_event()
    r, run = _drive(project, tmp_path, scn, 'replay')
    assert r.returncode == P.OK, r.stdout + r.stderr
    tj = json.loads((tmp_path / 'replay' / 'trajectories.json').read_text())
    ego, car = np.asarray(tj['actors']['ego']), np.asarray(tj['actors']['a1'])
    # the recorded tracks, to the step: the ego at 8 m/s from x = 0, the car extrapolated back before its first sample
    k = 20
    t = ego[k, 0]
    assert ego[k, 1] == pytest.approx(8.0 * t, abs=0.45) and abs(ego[k, 2]) < 0.05
    assert car[k, 1] == pytest.approx(25.0 + 6.0 * t, abs=0.35)
    assert car[0, 1] == pytest.approx(25.0 + 6.0 * car[0, 0], abs=0.35)
    # the follower drives the same start when the ego is not replayed
    scn['ego']['driver'] = 'follower'
    r, run = _drive(project, tmp_path, scn, 'follow')
    assert r.returncode == P.OK, r.stdout + r.stderr
    tj = json.loads((tmp_path / 'follow' / 'trajectories.json').read_text())
    assert tj['actors']['ego'][0][1] == pytest.approx(0.0, abs=0.6)


def test_a_start_off_the_map_is_refused(project, tmp_path):
    scn = _fake_road_event()
    scn['start'] = {'x': 0.0, 'y': 0.0, 'yaw': 90.0}                     # across the road
    r, run = _drive(project, tmp_path, scn, 'off')
    assert r.returncode == P.FAILED and 'start' in run['error']


def test_cli_extract_show(recon, world_project):
    rec, det = recon
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHONPATH')}
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy', 'dashcam', 'incident', 'extract', rec, '--project',
                        world_project.path, '--world', 'w-test', '--detections', det, '--name', 'ev1'],
                       capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads(r.stdout)['actors'] == 2
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy', 'dashcam', 'incident', 'show', world_project.path,
                        'ev1'], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0 and json.loads(r.stdout)['kinds'] == {'car': 1, 'pedestrian': 1}
    argv = INC.search_argv(world_project.path, world_project.workspace, 'ev1', budget=4)
    assert argv[:2] == ['search', world_project.path] and argv[3].endswith('variations.json')
    assert '--world' in argv and argv[argv.index('--world') + 1] == 'w-test'
