"""Level-k road users in the neural closed loop (core.recon.levelk_actors): tiny synthetic dynamic reconstructions —
a street of Gaussians, the ego's logged drive along it at 9 m/s, a car ahead and a two-wheeler that waited on a side
road crossing the street 30 m ahead (CARLA's y = +5.5, facing the street) — written directly as reconstruction folders
(no training), driven in NeuralSim by ego policies on the observation's lead features.

* The ego 22 m from the crossing (2.4 s): an ego that gives way (slows to a crawl for the side road) gets the level-1
  rider creeping across in front of it; an ego that keeps its speed gets it waiting and crossing behind it — for every
  seed tried.
* The ego 40 m from the crossing: the rider inches into a gap shorter than its own gap acceptance, predicting (level-k)
  that the ego, a level-0 driver to it, yields; an ego that brakes for what enters its path does, one that ignores it
  passes behind it — the rider's choice cannot depend on what the ego will do, only on its state.

The same seed gives the same motion; each episode is a run folder the traffic-rule checker accepts; parameters from a
dashcam calibration make the episode private. Everything on the CPU, about 2 s an episode."""
import json
import math
import os

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from falcon_autonomy.core.recon import actors as A  # noqa: E402

FPS = 10.0
SECONDS = 7.0
EGO_V = 9.0
CROSS_X = 30.0
RIDER_Y_RH = -5.5                 # the side road: the rider waits right of the street (CARLA's y = +5.5)
NEAR = {'ego_x0': 8.0, 'lead_x0': 45.0}       # the ego 22 m (2.4 s) from the crossing, a car well past it
FAR = {'ego_x0': -10.0, 'lead_x0': None}      # the ego 40 m (4.4 s) from it, nobody else on the street
PARAMS = {'car': {'k': 0}, 'two_wheeler': {'k': 1, 'alpha': 0.6}}
LANE_EDGE = -1.75                 # CARLA's y of the ego lane's far edge: the rider is across it


@pytest.fixture(autouse=True)
def _threads():
    n = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(n)


# ================================================================================================ the scene
def _quat_yaw(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def write_crossing_recon(path: str, ego_x0: float, lead_x0: float | None, seconds: float = SECONDS) -> str:
    """A finished dynamic reconstruction (point_cloud.ply, views.json, actors.pt, run.json) of the crossing: the
    street a checkerboard of Gaussians, one forward camera, the ego along +x at EGO_V from ego_x0, track 1 a car from
    lead_x0 at the same speed (None: no car), track 2 a two-wheeler standing on the side road facing the street (it
    waited in the log)."""
    from falcon_autonomy.core.provenance import write_json
    from falcon_autonomy.core.recon.gaussians import Gaussians, inverse_sigmoid, rgb_to_sh0
    from falcon_autonomy.core.recon.raster import fov_K
    from falcon_autonomy.core.recon.scene import VIEWS_SCHEMA, mount_from_carla
    os.makedirs(path, exist_ok=True)
    xs, ys = np.meshgrid(np.arange(-20.0, 90.0, 1.0), np.arange(-12.0, 12.0, 1.0))
    P = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], 1)
    chk = ((np.floor(P[:, 0] / 2) + np.floor(P[:, 1] / 2)) % 2)[:, None]
    col = np.clip(np.array([0.35, 0.36, 0.38]) * (0.7 + 0.5 * chk), 0, 1)
    n = len(P)
    t32 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731
    gs = Gaussians(t32(P), torch.log(t32(np.tile([0.6, 0.6, 0.02], (n, 1)))), t32(np.tile([1.0, 0, 0, 0], (n, 1))),
                   inverse_sigmoid(t32(np.full(n, 0.95))), rgb_to_sh0(t32(col))[:, None, :], torch.zeros(n, 0, 3))
    gs.write_ply(os.path.join(path, 'point_cloud.ply'))
    times = np.arange(int(round(seconds * FPS)) + 1) / FPS
    tracks = []
    if lead_x0 is not None:
        tracks.append(A.Track(1, 'car', 'vehicle.car', True, (4.5, 1.9, 1.4), times,
                              np.stack([lead_x0 + EGO_V * times, np.zeros(len(times)), np.full(len(times), 0.7)], 1),
                              np.tile(_quat_yaw(0.0), (len(times), 1))))
    tracks.append(A.Track(2, 'motorcycle', 'vehicle.motorcycle', True, (2.0, 0.8, 1.5), times,
                          np.tile([CROSS_X, RIDER_Y_RH, 0.75], (len(times), 1)),
                          np.tile(_quat_yaw(math.pi / 2), (len(times), 1))))
    actors = [A.init_actor(tr, None, None, n_surface=150, seed=k, device='cpu') for k, tr in enumerate(tracks)]
    torch.save(A.DynamicModel(gs, actors).state(), os.path.join(path, 'actors.pt'))
    w, h = 64, 36
    K = fov_K(w, h, 90.0)
    mount = mount_from_carla({'x': 1.5, 'y': 0.0, 'z': 1.6, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0})
    frames = []
    for k, t in enumerate(times):
        E = np.eye(4)
        E[0, 3] = ego_x0 + EGO_V * t
        frames.append({'index': k, 't': round(float(t), 6), 'name': f'{k:06d}', 'split': 'train', 'image': None,
                       'K': list(K), 'size': [w, h], 'viewmat': np.linalg.inv(E @ mount).reshape(-1).tolist(),
                       'ego': E.reshape(-1).tolist(), 'mount': mount.reshape(-1).tolist(), 'semantic': None,
                       'range': None, 'camera': 'cam-front'})
    write_json(os.path.join(path, 'views.json'), {
        'schema': VIEWS_SCHEMA, 'world': None, 'dynamic': True, 'carla_exact': False, 'town': 'Synthetic',
        'source': {'kind': 'synthetic', 'path': '', 'camera': 'cam-front', 'options': {}}, 'work_size': [w, h],
        'frames': frames})
    write_json(os.path.join(path, 'run.json'), {'schema': 'falcon.run/1', 'status': 'ok', 'name': 'crossing'})
    return path


class IdmPolicy:
    """An ego policy on the observation's lead features: IDM (Treiber et al. 2000) with a time headway T (negative:
    blind to what is ahead), a desired speed (approached from above at no more than b_free m/s²), steering back to
    the route as core.recon.closedloop.ConstantSpeedPolicy."""

    def __init__(self, headway_s: float, v_des: float, a_max: float = 2.0, b_free: float = 1.5):
        self.T, self.v_des, self.a_max, self.b_free = headway_s, v_des, a_max, b_free

    def act(self, obs, names):
        f = np.asarray(obs['state'], float)
        ix = {nm: i for i, nm in enumerate(names)}
        v = f[ix['speed_mps']]
        a = max(self.a_max * (1.0 - (v / self.v_des) ** 4), -self.b_free)
        if f[ix['lead_present']] > 0.5 and self.T >= 0:
            gap, dv = max(f[ix['lead_gap_m']], 0.1), f[ix['lead_closing_mps']]
            s_star = 2.0 + max(v * self.T + v * dv / (2.0 * math.sqrt(self.a_max * 2.0)), 0.0)
            a -= self.a_max * (s_star / gap) ** 2
        accel = a / 3.0 if a > 0 else a / 8.0          # the kinematic ego: 3 m/s² full throttle, 8 m/s² full brake
        steer = -0.3 * f[ix['lateral_offset_m']] - 1.2 * f[ix['heading_error_rad']]
        return np.clip([steer, accel], -1.0, 1.0).astype(np.float32)


KEEPS = (1.2, 9.0)                # keeps 9 m/s and brakes for what enters its path
GIVES_WAY = (1.2, 1.0, 2.0, 3.0)  # slows to a crawl (3 m/s²) for the side road: it gives way
IGNORES = (-1.0, 9.0)             # keeps 9 m/s and brakes for nothing


@pytest.fixture(scope='module')
def recons(tmp_path_factory):
    root = tmp_path_factory.mktemp('lk')
    return {'near': write_crossing_recon(str(root / 'near'), **NEAR),
            'far': write_crossing_recon(str(root / 'far'), **FAR)}


def drive(recon, out, policy, seed=0, params=PARAMS, mode='sample'):
    from falcon_autonomy.core.policy import rollout
    from falcon_autonomy.core.recon.closedloop import NeuralSim
    from falcon_autonomy.core.recon.levelk_actors import LevelKActors
    sim = NeuralSim(recon, out, obs_size=(48, 27), device='cpu', controller=LevelKActors(params, mode=mode))
    try:
        doc = rollout(IdmPolicy(*policy), sim, 1, seed=seed, out_dir=out)
    finally:
        sim.close()
    ep = doc['episodes'][0]
    with open(os.path.join(ep['run_dir'], 'levelk.json')) as fh:
        return ep, json.load(fh)


def ego_x_when_across(lk) -> float | None:
    """The ego's x (CARLA) when the rider is first across the ego lane's far edge."""
    tr = lk['trajectories']
    k = next((k for k, y in enumerate(tr['2']['y']) if y < LANE_EDGE), None)
    return None if k is None else tr['ego']['x'][k]


def check_episode(ep, lk):
    from falcon_autonomy.core import rules
    assert not lk['realism']['collisions'], lk['realism']['collisions']
    assert lk['realism']['levelk_realistic'], lk['realism']['agents']
    assert not ep['infraction_counts'].get('collision_vehicle'), ep['infraction_counts']
    res = rules.check_run(ep['run_dir'], xodr='')
    assert res['rules'] and res['run'] == os.path.abspath(ep['run_dir'])
    with open(os.path.join(ep['run_dir'], 'run.json')) as fh:
        run = json.load(fh)
    assert run['actors']['model'] == 'levelk' and 'source' not in run
    acts = [json.loads(x) for x in open(os.path.join(ep['run_dir'], 'actors.jsonl'))]
    last = [a for a in acts[-1]['actors'] if a['id'] == 2][0]        # drawn where level-k put it
    assert last['x'] == pytest.approx(lk['trajectories']['2']['x'][-1], abs=1e-3)
    assert last['y'] == pytest.approx(lk['trajectories']['2']['y'][-1], abs=1e-3)


# ================================================================================================ frames
def test_poses_round_trip_the_logged_frame(recons, tmp_path):
    """At the start every actor is drawn exactly where its logged box is (the frames convert both ways), and the ego
    and the tracks are where they should be in CARLA's convention."""
    from falcon_autonomy.core.recon.closedloop import NeuralSim
    from falcon_autonomy.core.recon.levelk_actors import LevelKActors
    ctl = LevelKActors(PARAMS)
    sim = NeuralSim(recons['near'], str(tmp_path / 'cl'), obs_size=(48, 27), device='cpu', controller=ctl)
    sim.reset(seed=0)
    t0 = sim.ep['t0']
    po = ctl.poses(t0)
    for tid in (1, 2):
        R, c, v = po[tid]
        Rl, cl = sim.tracks[tid].pose(t0)
        assert np.allclose(c, cl, atol=1e-6) and np.allclose(R, Rl, atol=1e-6)
    assert np.allclose(po[1][2], [EGO_V, 0.0, 0.0], atol=1e-6) and np.allclose(po[2][2], 0.0)
    a1, a2 = ctl.agents[1], ctl.agents[2]
    assert a1['type'] == 'car' and a1['priority'] and a1['params']['level'] == 0
    assert a1['params']['v_des'] == pytest.approx(EGO_V)          # the logged median speed
    assert a2['type'] == 'two_wheeler' and not a2['priority'] and a2['params']['level'] == 1
    assert a2['params']['alpha'] == 0.6 and a2['params']['v_des'] == 11.0     # it never moved: the type's
    x, y, yaw = a2['path'].at(a2['s'], a2['d'])
    assert (float(x), float(y), float(yaw)) == pytest.approx((CROSS_X, -RIDER_Y_RH, -90.0))   # CARLA: y right
    assert sim.ep['state'][:2] == pytest.approx([NEAR['ego_x0'], 0.0])
    # a step moves the rider along its path and the pose follows (mirrored back: y left)
    sim.step([0.0, 0.0])
    R, c, _ = ctl.poses(sim.ep['t0'] + sim.dt)[2]
    x, y, yaw = (float(q) for q in a2['path'].at(a2['s'], a2['d']))
    assert c[:2] == pytest.approx([x, -y]) and math.degrees(math.atan2(R[1, 0], R[0, 0])) == pytest.approx(-yaw)
    sim.close()


# ================================================================================================ inching
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_two_wheeler_crosses_in_front_of_an_ego_that_gives_way_and_waits_for_one_that_does_not(recons, tmp_path,
                                                                                                seed):
    ep_y, lk_y = drive(recons['near'], str(tmp_path / 'gives'), GIVES_WAY, seed)
    ep_n, lk_n = drive(recons['near'], str(tmp_path / 'keeps'), KEEPS, seed)
    iy, i_n = lk_y['inching']['2'], lk_n['inching']['2']
    # the ego gives way: the rider creeps out (under 2.5 m/s on its way) and is across while the ego is short of it
    assert iy['entered'] and iy['yielded_by'] == 'ego' and iy['mean_approach_speed_mps'] < 2.5, iy
    assert ego_x_when_across(lk_y) < CROSS_X - 4.0
    # the ego keeps its speed: the rider waits and crosses behind it
    assert not i_n['entered'], i_n
    assert ego_x_when_across(lk_n) > CROSS_X + 4.0
    for ep, lk in ((ep_y, lk_y), (ep_n, lk_n)):
        check_episode(ep, lk)


def test_two_wheeler_inches_into_a_short_gap_predicting_that_the_ego_yields(recons, tmp_path):
    ep_y, lk_y = drive(recons['far'], str(tmp_path / 'keeps'), KEEPS)
    ep_n, lk_n = drive(recons['far'], str(tmp_path / 'ignores'), IGNORES)
    iy, i_n = lk_y['inching']['2'], lk_n['inching']['2']
    # a gap under its own gap acceptance (3 s), creeping, and the ego brakes for it
    assert iy['inched'] and iy['vehicle_time_to_reach_s'] < 3.0 and iy['vehicle_max_decel_after_mps2'] > 0.5, iy
    # the same choice against an ego that will not brake (it cannot know): the ego passes behind it, at speed
    assert i_n['entered'] and i_n['entry_t'] == iy['entry_t'] and i_n['vehicle_min_speed_mps'] == pytest.approx(EGO_V)
    assert not i_n['inched']
    for ep, lk in ((ep_y, lk_y), (ep_n, lk_n)):
        check_episode(ep, lk)


def test_same_seed_same_motion(recons, tmp_path):
    _, a = drive(recons['near'], str(tmp_path / 'a'), GIVES_WAY, seed=5)
    _, b = drive(recons['near'], str(tmp_path / 'b'), GIVES_WAY, seed=5)
    assert a['trajectories'] == b['trajectories'] and a['seed'] == b['seed'] == 5


# ================================================================================================ the command
def test_closedloop_cli_with_levelk_actors(recons, tmp_path):
    from falcon_autonomy.core.recon.commands import main_recon
    from falcon_autonomy.core.recon.levelk_actors import load_params
    recon = recons['near']
    d = str(tmp_path / 'cli')
    assert main_recon(['closedloop', recon, '--policy', 'constant', '--out', d, '--episodes', '1', '--obs-size',
                       '32x18', '--device', 'cpu', '--actors', 'levelk', '--levelk-params',
                       json.dumps(PARAMS)]) == 0
    with open(os.path.join(d, 'closedloop.json')) as fh:
        assert json.load(fh)['actors'] == 'levelk'
    with open(os.path.join(d, 'ep-0000', 'levelk.json')) as fh:
        lk = json.load(fh)
    assert lk['parameters'] == PARAMS and lk['decisions'] > 10 and lk['mode'] == 'sample'
    # a parameters file: a calibration's behaviour.levelk block
    pf = str(tmp_path / 'params.json')
    with open(pf, 'w') as fh:
        json.dump({'behaviour': {'model': 'levelk', 'levelk': PARAMS}}, fh)
    assert load_params(pf)[0] == PARAMS and load_params(pf)[1]['private'] is None
    assert main_recon(['closedloop', recon, '--out', str(tmp_path / 'x'), '--actors', 'levelk', '--levelk-params',
                       '{"car": {"k": 9}}', '--device', 'cpu']) == 2          # outside the parameter bounds
    assert main_recon(['closedloop', recon, '--out', str(tmp_path / 'y'), '--levelk-params', '{}',
                       '--device', 'cpu']) == 2                                # needs --actors levelk
    assert main_recon(['closedloop', '--help']) == 0


def test_dashcam_calibrated_parameters_make_the_episode_private(recons, tmp_path):
    """Parameters from a calibration on dashcam footage mark every episode's run.json: the assistant refuses it."""
    from falcon_autonomy.assistant.privacy import private_reason
    from falcon_autonomy.core.recon.commands import main_recon
    pf = str(tmp_path / 'cal' / 'levelk_params.json')
    os.makedirs(os.path.dirname(pf))
    with open(pf, 'w') as fh:
        json.dump({'behaviour': {'model': 'levelk', 'levelk': PARAMS, 'levelk_source': {'format': 'dashcam'}}}, fh)
    d = str(tmp_path / 'priv')
    assert main_recon(['closedloop', recons['near'], '--out', d, '--obs-size', '32x18', '--device', 'cpu',
                       '--actors', 'levelk', '--levelk-params', pf]) == 0
    ep = os.path.join(d, 'ep-0000')
    with open(os.path.join(ep, 'run.json')) as fh:
        assert json.load(fh)['source']['format'] == 'dashcam'
    assert private_reason(ep) and private_reason(os.path.join(ep, 'frames.jsonl'))
