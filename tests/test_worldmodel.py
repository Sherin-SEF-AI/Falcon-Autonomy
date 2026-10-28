"""World-model data (core.worldmodel): occupancy with visibility, BEV maps, scene flow, future trajectories, ego actions
and clips — on hand-made scenes with known answers, and end to end on a drive the drive worker records against the
fake carla.

The run: the test project's rig with the pinhole camera at 160 × 90 with every label and a 16-channel lidar (40 000
points/s, 60 m, 10 Hz), 8 traffic vehicles and 2 walkers, 1.5 s at 20 fps, seed 3. Its actors.jsonl is the drive
worker's own (scene.ActorLog: every vehicle and walker per step from the world snapshot); nothing is synthesised.
"""
import json
import math
import os

import numpy as np
import pytest

from falcon_autonomy.core import groundtruth as GT
from falcon_autonomy.core import protocol as P
from falcon_autonomy.core import worldmodel as WM


# --------------------------------------------------------------------------------------------------------- fixtures
def _project(path):
    from falcon_autonomy.core.project import Project
    from falcon_autonomy.core.rig import new_sensor
    p = Project.new('Test')
    rig = p.doc['rigs'][0]
    rig['sensors'][0].update(width=160, height=90, labels=['semantic', 'instance', 'depth', 'boxes2d', 'boxes3d'])
    rig['sensors'].append(new_sensor('lidar', 'lid', 'top', {'x': 0.0, 'z': 1.9}))
    rig['sensors'][-1].update(channels=16, points_per_second=40000, range=60.0, rotation_frequency=10.0,
                              upper_fov=5.0, lower_fov=-20.0)
    p.doc['traffic_profiles'][0].update(vehicles=8, walkers=2)
    p.save(str(path))
    return p


@pytest.fixture(scope='module')
def run_dir(tmp_path_factory):
    from test_workers import _cfg, _run
    base = tmp_path_factory.mktemp('wm')
    project = _project(base / 'test.falcon.json')
    code, _, err = _run(_cfg(project, base / 'run', seconds=1.5, seed=3))
    assert code == P.OK, err
    assert (base / 'run' / 'actors.jsonl').is_file()
    return str(base / 'run')


@pytest.fixture(scope='module')
def derived(run_dir, tmp_path_factory):
    out = str(tmp_path_factory.mktemp('wm_out'))
    return out, WM.derive(run_dir, out, {'res': 0.4})


def _pose(x=0.0, y=0.0, z=0.0, yaw=0.0):
    return {'x': x, 'y': y, 'z': z, 'roll': 0.0, 'pitch': 0.0, 'yaw': yaw}


def _actor(aid, x, y, yaw=0.0, ext=(2.0, 1.0, 0.75), cat='vehicle.car', vx=0.0, vy=0.0, z=0.0, role=''):
    return dict(_pose(x, y, z, yaw), id=aid, vx=vx, vy=vy, vz=0.0, type_id='vehicle.x', category=cat, role=role,
                extent=list(ext), bbox_center=[0.0, 0.0, ext[2]])


# --------------------------------------------------------------------------------------------------------- occupancy
def test_traversal_is_exact_against_dense_sampling():
    g = WM.Grid(0.5, (-10, 10), (-10, 10), (-2, 4))
    rng = np.random.default_rng(0)
    o = np.array([0.13, -0.27, 1.21])
    ends = rng.uniform([-14, -14, -3], [14, 14, 5], (60, 3))
    free, hit = WM.traverse(g, o, ends)
    ref = np.zeros(g.shape, bool)
    for e in ends:
        end = g.index(e)
        s = np.linspace(0.0, 1.0, 40001)[:, None]
        ijk = g.index(o + (e - o) * s)
        before = ~np.all(ijk == end, axis=1)
        k = np.argmin(before) if not before.all() else len(before)     # up to the first sample in the end voxel
        ijk = ijk[:k]
        ijk = ijk[g.inside(ijk)]
        ref[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    # dense sampling can only miss a voxel clipped at a corner, never add one
    assert not (ref & ~free).any()
    assert (free & ~ref).sum() <= 0.005 * free.sum()
    assert hit.sum() >= 1


def test_occupancy_of_a_hand_made_box():
    g = WM.Grid(0.5, (-10, 20), (-5, 5), (-1, 3))
    origin = np.array([0.0, 0.0, 1.2])
    # a car's box 10–12 m ahead, ±1 m, z 0–2; the lidar sees its front face
    box = {'center': [11.0, 0.0, 1.0], 'R': np.eye(3), 'extent': [1.0, 1.0, 1.0], 'class': 14}
    yy, zz = np.meshgrid(np.linspace(-0.9, 0.9, 7), np.linspace(0.1, 1.9, 7))
    face = np.c_[np.full(yy.size, 10.001), yy.ravel(), zz.ravel()]
    # one voxel far to the side with 3 road points and 1 sign point: the majority is road
    vote = np.array([[2.1, 4.1, 0.1], [2.2, 4.2, 0.2], [2.3, 4.3, 0.3], [2.4, 4.4, 0.4]])
    pts = np.r_[face, vote]
    cls = np.r_[np.full(len(face), 14), [1, 1, 8, 1]]
    occ = WM.occupancy(pts, cls, origin, g, [box])
    sem, state, obs = occ['semantics'], occ['state'], occ['observed']

    def at(x, y, z):
        i, j, k = g.index([x, y, z])
        return int(sem[i, j, k]), int(state[i, j, k]), bool(obs[i, j, k])
    assert at(10.25, 0.25, 1.25) == (14, WM.OCCUPIED, True)            # the visible face: hit
    assert at(11.75, 0.25, 1.25) == (14, WM.OCCUPIED, False)           # inside the box, behind its face
    assert at(5.25, 0.25, 1.25) == (WM.EMPTY, WM.FREE, True)           # in front: rays passed through
    assert at(15.25, 0.25, 1.25) == (WM.EMPTY, WM.UNOBSERVED, False)   # behind the box: never reached
    assert at(-5.25, 0.25, 1.25) == (WM.EMPTY, WM.UNOBSERVED, False)   # no ray there
    assert at(2.25, 4.25, 0.25)[0] == 1                                # majority vote
    # the box is filled whole: 2 × 2 × 2 m at 0.5 m
    assert int((sem == 14).sum()) == 4 * 4 * 4
    # BEV, image convention: row = nx − 1 − ix, ego heading up
    i, j, _ = g.index([11.0, 0.25, 0.0])
    bs = occ['bev_state']
    assert bs[g.shape[0] - 1 - i, j] == WM.OCCUPIED and occ['bev_semantics'][g.shape[0] - 1 - i, j] == 14
    i5, _, _ = g.index([5.25, 0.25, 0.0])
    assert bs[g.shape[0] - 1 - i5, j] == WM.FREE
    i15, _, _ = g.index([15.25, 0.25, 0.0])
    assert bs[g.shape[0] - 1 - i15, j] == WM.UNOBSERVED


def test_classes_of_categories():
    assert WM.class_of_category('vehicle.car') == 14 and WM.class_of_category('vehicle.bus.rigid') == 16
    assert WM.class_of_category('human.pedestrian.child') == 12 and WM.class_of_category('vehicle.bicycle') == 19
    assert WM.bev_class_of('vehicle.motorcycle') == 'cyclist' and WM.bev_class_of('vehicle.truck') == 'vehicle'
    assert WM.bev_class_of('movable_object.trafficcone') == 'other'


# --------------------------------------------------------------------------------------------------------- BEV
def test_bev_raster_places_and_orients_a_known_box():
    """Ego at (100, 50) heading +y (yaw 90°): its right is world −x (CARLA, left-handed). A car 10 m ahead and 5 m to
    its right is drawn 10 m up and 5 m right of the centre; aligned with the ego it is tall, crosswise wide."""
    spec = WM.BevSpec(0.5, 20, 20, 20, 20)
    ego = _actor(1, 100.0, 50.0, yaw=90.0, ext=(2.4, 1.0, 0.75), role='hero')
    fwd, right = np.array([0.0, 1.0]), np.array([-1.0, 0.0])
    c = np.array([100.0, 50.0]) + 10 * fwd + 5 * right                # (95, 60)
    car = _actor(7, *c, yaw=90.0, ext=(2.0, 1.0, 0.75))
    ped_c = np.array([100.0, 50.0]) + 6 * fwd - 4 * right             # 6 m ahead, 4 m LEFT
    ped = _actor(8, *ped_c, ext=(0.2, 0.2, 0.9), cat='human.pedestrian.adult')
    b = WM.bev_raster(_pose(100.0, 50.0, yaw=90.0), [ego, car, ped], spec)
    veh = (b['bits'] & WM.BEV_BIT['vehicle']) > 0
    rr, cc = np.nonzero(veh)
    # x 8..12 m -> rows 16..23; y 4..6 m -> cols 48..51 (pixel centres inside)
    assert (rr.min(), rr.max(), cc.min(), cc.max()) == (16, 23, 48, 51) and veh.sum() == 32
    assert set(np.unique(b['instance'][veh])) == {7}
    pr, pc = np.nonzero(b['bits'] & WM.BEV_BIT['pedestrian'])
    assert len(pr) and pc.max() < spec.w // 2 and pr.max() < spec.h // 2      # ahead (up) and left
    er, ec = np.nonzero(b['bits'] & WM.BEV_BIT['ego'])
    assert er.min() < spec.h / 2 < er.max() and ec.min() < spec.w / 2 < ec.max()
    # the same car turned crosswise (world yaw 0 = 90° to the ego's left of heading): wide, not tall
    b2 = WM.bev_raster(_pose(100.0, 50.0, yaw=90.0), [dict(car, yaw=0.0)], spec)
    rr, cc = np.nonzero(b2['bits'] & WM.BEV_BIT['vehicle'])
    assert (rr.min(), rr.max(), cc.min(), cc.max()) == (18, 21, 46, 53)


def test_bev_lanes_from_lanes_json():
    """A lane along world +y through the ego (yaw 90): drivable 3.5 m wide around the centre column, boundaries at
    its edges, the centre line at the middle."""
    cols = ['x', 'y', 'z', 'yaw', 'width', 'road', 'section', 'lane', 's', 'left_type', 'left_color', 'right_type',
            'right_color']
    pts = [[100.0, float(y), 0.0, 90.0, 3.5, 1, 0, -1, float(y), 'Solid', 'White', 'Broken', 'White']
           for y in range(0, 101, 2)]
    lanes = {'columns': cols, 'points': pts, 'step_m': 2.0}
    spec = WM.BevSpec(0.5, 20, 20, 20, 20)
    lm = WM.lane_map_for(lanes, [_pose(100.0, 50.0, yaw=90.0)], spec)
    b = WM.bev_raster(_pose(100.0, 50.0, yaw=90.0), [], spec, lm)
    drv = (b['bits'] & WM.BEV_BIT['drivable']) > 0
    assert drv[:, 37:43].all() and not drv[:, :34].any() and not drv[:, 46:].any()
    lb = (b['bits'] & WM.BEV_BIT['lane_boundary']) > 0
    assert lb[:, 35:38].any(1).all() and lb[:, 42:45].any(1).all() and not lb[:, 39:41].any()
    cl = (b['bits'] & WM.BEV_BIT['centerline']) > 0
    assert cl[:, 39:41].any(1).all()


# --------------------------------------------------------------------------------------------------------- flow
def test_scene_flow_of_a_rigid_actor():
    from falcon_autonomy.core.rig import carla_rotation
    M_ws = WM.pose_matrix({'x': 1.0, 'y': -2.0, 'z': 1.9, 'roll': 0.0, 'pitch': 0.0, 'yaw': 30.0})
    A0 = WM.pose_matrix(_pose(10.0, 0.0, yaw=0.0))
    A1 = WM.pose_matrix(_pose(10.5, 0.2, yaw=0.0))                     # pure translation (0.5, 0.2, 0)
    A2 = WM.pose_matrix(_pose(10.5, 0.2, yaw=5.0))                     # and a turn
    rng = np.random.default_rng(1)
    pw = np.c_[rng.uniform(9, 11, 20), rng.uniform(-1, 1, 20), rng.uniform(0, 1.5, 20)]
    ps = (pw - M_ws[:3, 3]) @ M_ws[:3, :3]                              # into the sensor frame
    ids = np.r_[np.full(10, 7), np.zeros(5), np.full(5, 9)]            # 9: not in the second sweep's poses
    f = WM.scene_flow(ps, ids, M_ws, {7: A0, 9: A0}, {7: A1})
    disp_sensor = M_ws[:3, :3].T @ np.array([0.5, 0.2, 0.0])
    assert np.allclose(f[:10], disp_sensor, atol=1e-5) and np.allclose(f[10:], 0.0)
    assert np.allclose(np.linalg.norm(f[:10], axis=1), math.hypot(0.5, 0.2), atol=1e-5)
    f2 = WM.scene_flow(ps, ids, M_ws, {7: A0}, {7: A2})
    D = A2 @ np.linalg.inv(A0)
    want = (pw[:10] @ D[:3, :3].T + D[:3, 3] - pw[:10]) @ M_ws[:3, :3]
    assert np.allclose(f2[:10], want, atol=1e-5)
    assert np.allclose(carla_rotation(0, 0, 30.0), M_ws[:3, :3])


# --------------------------------------------------------------------------------------------------------- trajectories
def _timeline(n=40, fps=20.0, ego_v=(0.0, 0.0), ego_yaw=0.0):
    tl = []
    for i in range(n):
        t = i / fps
        acts = [_actor(5, 10 + 4 * t, 2 + 1 * t, vx=4.0, vy=1.0),
                _actor(1, ego_v[0] * t, ego_v[1] * t, yaw=ego_yaw, role='hero')]
        if i < 20:
            acts.append(_actor(6, -5.0, 0.0))
        tl.append({'step': i, 'sim_t': t, 'ego': _pose(ego_v[0] * t, ego_v[1] * t, yaw=ego_yaw), 'actors': acts})
    return tl


def test_future_trajectories_of_constant_velocity_agents():
    tr = WM.future_trajectories(_timeline(), horizon_s=1.0, rate_hz=10.0)
    assert np.allclose(tr['t_offsets'], np.arange(1, 11) / 10.0)
    assert 1 not in set(tr['agent_id'].tolist())                       # the ego is not an agent
    k = int(np.flatnonzero((tr['agent_step'] == 0) & (tr['agent_id'] == 5))[0])
    fut = tr['future'][k]
    assert tr['valid'][k].all()
    assert np.allclose(fut[:, 0], 10 + 0.4 * np.arange(1, 11), atol=1e-4)
    assert np.allclose(fut[:, 1], 2 + 0.1 * np.arange(1, 11), atol=1e-4)
    steps = np.diff(np.r_[tr['present'][k][None, :2], fut[:, :2]], axis=0)
    assert np.allclose(steps, [0.4, 0.1], atol=1e-4)                  # a straight line, evenly spaced
    # agent 6 is destroyed after step 19: from step 15 its future is valid up to 0.2 s only
    k6 = int(np.flatnonzero((tr['agent_step'] == 15) & (tr['agent_id'] == 6))[0])
    assert tr['valid'][k6].tolist() == [True, True] + [False] * 8
    # near the run's end nothing is valid beyond the last step
    k_end = int(np.flatnonzero((tr['agent_step'] == 35) & (tr['agent_id'] == 5))[0])
    assert tr['valid'][k_end].tolist() == [True, True] + [False] * 8
    # a rate that does not divide the fps: interpolated, still exact on a straight line
    tr3 = WM.future_trajectories(_timeline(), horizon_s=1.0, rate_hz=3.0)
    k = int(np.flatnonzero((tr3['agent_step'] == 0) & (tr3['agent_id'] == 5))[0])
    assert np.allclose(tr3['future'][k][:, 0], 10 + 4 * np.arange(1, 4) / 3.0, atol=1e-4)


def test_future_trajectories_in_the_ego_frame_left_handed():
    """The ego heads +y at 2 m/s (yaw 90°): its own future is straight ahead (x), an agent moving world +x goes to
    the ego's LEFT (−y: CARLA's right of +y is −x)."""
    tr = WM.future_trajectories(_timeline(ego_v=(0.0, 2.0), ego_yaw=90.0), horizon_s=1.0, rate_hz=10.0)
    ef = tr['ego_future'][0]
    assert tr['ego_valid'][0].all()
    assert np.allclose(ef[:, 0], 0.2 * np.arange(1, 11), atol=1e-4) and np.allclose(ef[:, 1:], 0.0, atol=1e-4)
    k = int(np.flatnonzero((tr['agent_step'] == 0) & (tr['agent_id'] == 5))[0])
    assert np.all(np.diff(tr['future'][k][:, 1]) < 0)
    assert np.allclose(tr['present'][k][:2], [2.0, -10.0], atol=1e-4)   # world (10, 2) from (0, 0) heading +y


# --------------------------------------------------------------------------------------------------------- actions
def test_ego_actions_derived_and_recorded():
    frames = [{'step': i, 'sim_t': i / 20.0, 'ego': _pose(yaw=10.0 * i / 20.0), 'speed_mps': 5.0 + 0.1 * i}
              for i in range(20)]
    acts = WM.ego_actions(frames)
    assert all(a['control'] is None and a['control_source'] == 'not recorded' for a in acts)
    a = acts[10]
    assert a['yaw_rate_dps'] == pytest.approx(10.0) and a['accel_mps2'] == pytest.approx(2.0)
    assert a['curvature_1pm'] == pytest.approx(math.radians(10.0) / 6.0, rel=1e-4)
    stopped = WM.ego_actions([dict(f, speed_mps=0.1) for f in frames])
    assert stopped[3]['curvature_1pm'] is None
    live = [dict(f, control={'throttle': 0.5, 'steer': 0.1, 'brake': 0.0, 'hand_brake': False, 'reverse': False})
            for f in frames]
    live[4]['control'] = None
    la = WM.ego_actions(live)
    assert la[0]['control']['throttle'] == 0.5 and la[0]['control_source'] == 'recorded'
    assert la[4]['control'] is None and la[4]['control_source'] == 'recorded: none applied'


# --------------------------------------------------------------------------------------------------------- clips
def _frames(n=40, cams=('a', 'b')):
    return [{'step': i, 'sim_t': i / 20.0, 'ego': _pose(i * 0.4), 'speed_mps': 8.0, 'recorded': True,
             'files': {f'{c}/rgb': f'rgb/{c}/{i:06d}.jpg' for c in cams}} for i in range(n)]


def test_clip_manifests_and_refusal():
    cm = WM.clip_manifest(_frames(), 20.0, 10.0, 1.0, 0.5)
    assert cm['step_stride'] == 2 and cm['frames_per_clip'] == 10 and cm['cameras'] == ['a', 'b']
    assert [c['start_step'] for c in cm['clips']] == [0, 10, 20]
    c0 = cm['clips'][0]
    assert [f['step'] for f in c0['frames']] == list(range(0, 20, 2))
    assert [f['t'] for f in c0['frames']] == pytest.approx([k / 10.0 for k in range(10)])
    assert c0['frames'][3]['images'] == {'a': 'rgb/a/000006.jpg', 'b': 'rgb/b/000006.jpg'}
    assert c0['frames'][3]['sim_t'] == pytest.approx(0.3)
    # 2 Hz: every 10th step
    cm2 = WM.clip_manifest(_frames(), 20.0, 2.0, 1.0, 1.0)
    assert [f['step'] for f in cm2['clips'][0]['frames']] == [0, 10]
    # a camera's frame missing: the clip is left out, counted
    fr = _frames()
    del fr[4]['files']['b/rgb']
    cm3 = WM.clip_manifest(fr, 20.0, 10.0, 1.0, 0.5)
    assert [c['start_step'] for c in cm3['clips']] == [10, 20] and cm3['dropped_incomplete'] == 1
    for bad in (3.0, 8.0, 15.0):                                        # 20 / rate not an integer
        with pytest.raises(ValueError, match='does not divide'):
            WM.clip_manifest(_frames(), 20.0, bad, 1.0, 0.5)
    with pytest.raises(ValueError, match='stride'):
        WM.clip_manifest(_frames(), 20.0, 10.0, 1.0, 0.025)


# --------------------------------------------------------------------------------------------------------- end to end
def test_derive_refuses_a_non_divisor_rate_before_writing(run_dir, tmp_path):
    with pytest.raises(ValueError, match='does not divide'):
        WM.derive(run_dir, str(tmp_path / 'out'), {'clip_rate_hz': 3.0})
    assert not (tmp_path / 'out').exists()
    with pytest.raises(ValueError, match='unknown options'):
        WM.derive(run_dir, str(tmp_path / 'out'), {'resolution': 0.1})


def test_derive_end_to_end(run_dir, derived):
    out, m = derived
    data = WM.load_run(run_dir)
    assert m['schema'] == WM.SCHEMA and m['skipped'] == {} and set(m['products']) == set(WM.PRODUCTS)
    assert m['ego_id'] == data['ego_id'] and m['steps'] == 30 and m['lidars'] == ['lid']
    ins = m['provenance']['inputs']
    assert all(ins[k]['sha256'] for k in ('run', 'frames', 'lanes', 'actors'))
    with open(os.path.join(out, 'worldmodel.json')) as fh:
        assert json.load(fh)['products'].keys() == m['products'].keys()
    tl = [json.loads(x) for x in open(os.path.join(out, 'timeline.jsonl'))]
    assert [r['step'] for r in tl] == list(range(30)) and all('bev' in r and 'actions' in r for r in tl)
    assert sum('occupancy' in r for r in tl) == m['products']['occupancy']['sweeps'] == 15

    # occupancy: the voxels of a car's lidar points carry the car class; its box is filled
    grid = WM.Grid(0.4)
    sweeps = WM.lidar_sweeps(data, 'lid')
    seen_car = False
    for n, step, _t, rel, M_ws in sweeps:
        xyz, tag, ids = WM.read_sweep(run_dir, rel)
        occ = np.load(os.path.join(out, 'occupancy', 'lid', f'{n:06d}.npz'))
        assert occ['semantics'].shape == grid.shape == occ['state'].shape
        assert occ['bev_state'].shape == grid.shape[:2]
        frame = next(f for f in data['frames'] if f['step'] == step)
        T_es = np.linalg.inv(WM.pose_matrix(frame['ego'])) @ M_ws
        cars = [a for a in data['actors'][step] if a['id'] != data['ego_id'] and a['category'] == 'vehicle.car']
        for a in cars:
            sel = ids == a['id']
            if sel.sum() < 5:
                continue
            P = xyz[sel] @ T_es[:3, :3].T + T_es[:3, 3]
            ijk = grid.index(P)
            ijk = ijk[grid.inside(ijk)]
            if not len(ijk):
                continue
            assert np.mean(occ['semantics'][ijk[:, 0], ijk[:, 1], ijk[:, 2]] == 14) > 0.9
            assert (occ['state'][ijk[:, 0], ijk[:, 1], ijk[:, 2]] == WM.OCCUPIED).all()
            assert occ['observed'][ijk[:, 0], ijk[:, 1], ijk[:, 2]].all()
            seen_car = True
        # ground in front of the lidar is observed; the ego itself is not in the grid by default
        i, j, k = grid.index([0.0, 0.0, 0.6])
        assert occ['semantics'][i, j, k] == WM.EMPTY
        assert (occ['semantics'] == 1).sum() > 100 and (occ['state'] == WM.FREE).sum() > 1000
    assert seen_car

    # BEV: every car the camera saw (objects JSON box3d, vehicle frame — camgt's own geometry) is drawn as a vehicle
    # with its track id at its centre, and the ego at the raster's centre
    spec = WM.BevSpec(0.25, 40, 40, 40, 40)
    checked = 0
    for f in data['frames'][::5]:
        b = np.load(os.path.join(out, 'bev', f'{f["step"]:06d}.npz'))
        assert b['bits'][spec.h // 2, spec.w // 2] & WM.BEV_BIT['ego']
        assert b['bits'][spec.h // 2, spec.w // 2] & WM.BEV_BIT['drivable']
        objs = json.load(open(os.path.join(run_dir, f['files']['cam-front/objects'])))['objects']
        for o in objs:
            if o['class'] != 'car' or not isinstance(o['track_id'], int):
                continue
            r, c = np.round(spec.pixel(o['box3d']['center'][:2])).astype(int)
            if 0 <= r < spec.h and 0 <= c < spec.w:
                assert b['bits'][r, c] & WM.BEV_BIT['vehicle'] and b['instance'][r, c] == o['track_id']
                checked += 1
    assert checked >= 3

    # scene flow: a moving car's points move by its displacement between the sweeps
    flows = 0
    for (n, s0, t0, rel, M_ws), (_n1, s1, _t1, _r1, _M1) in zip(sweeps, sweeps[1:]):
        xyz, tag, ids = WM.read_sweep(run_dir, rel)
        fl = np.load(os.path.join(out, 'flow', 'lid', f'{n:06d}.npy'))
        assert fl.shape == xyz.shape and fl.dtype == np.float32
        assert np.allclose(fl[ids == 0], 0.0)
        a0 = {a['id']: a for a in data['actors'][s0]}
        a1 = {a['id']: a for a in data['actors'][s1]}
        for aid in set(np.unique(ids).tolist()) & set(a0) & set(a1):
            if aid == data['ego_id']:
                continue
            d = np.array([a1[aid]['x'] - a0[aid]['x'], a1[aid]['y'] - a0[aid]['y'], 0.0])
            if a0[aid]['yaw'] == a1[aid]['yaw'] and np.linalg.norm(d) > 0.1:
                assert np.allclose(fl[ids == aid], M_ws[:3, :3].T @ d, atol=2e-3)
                flows += 1
    assert flows >= 3

    # trajectories: an agent's future at +0.5 s is its actors.jsonl pose 10 steps later, in the ego frame now
    tr = np.load(os.path.join(out, 'trajectories.npz'))
    assert tr['future'].shape[1:] == (30, 4) and len(tr['ego_future']) == 30
    k = int(np.flatnonzero(tr['agent_step'] == 0)[0])
    aid = int(tr['agent_id'][k])
    later = next(a for a in data['actors'][10] if a['id'] == aid)
    Ti = np.linalg.inv(WM.pose_matrix(data['frames'][0]['ego']))
    want = Ti[:3, :3] @ np.array([later['x'], later['y'], later['z']]) + Ti[:3, 3]
    assert tr['valid'][k, 4] and np.allclose(tr['future'][k, 4, :3], want, atol=1e-3)
    assert not tr['valid'][tr['agent_step'] == 29].any()                # the run ends

    # actions: an autopilot drive records the traffic manager's control; the fake's hero turns at 3°/s at 8 m/s
    acts = [json.loads(x) for x in open(os.path.join(out, 'ego_actions.jsonl'))]
    assert len(acts) == 30 and all(a['control_source'] == 'recorded' and a['control']['steer'] > 0 for a in acts)
    assert acts[15]['yaw_rate_dps'] == pytest.approx(3.0, abs=0.05)
    assert acts[15]['curvature_1pm'] == pytest.approx(math.radians(3.0) / 8.0, rel=0.02)

    # clips: 10 Hz from 20 fps, 1 s every 0.5 s
    clips = json.load(open(os.path.join(out, 'clips.json')))
    assert clips['step_stride'] == 2 and len(clips['clips']) == 2
    for c in clips['clips']:
        for fr in c['frames']:
            assert os.path.isfile(os.path.join(run_dir, fr['images']['cam-front']))
            assert fr['action']['step'] == fr['step']
    assert GT.CLASS_NAMES[14] == 'car'
