"""4D reconstruction (core.recon.actors, lens_raster, lidar, closedloop): scenes rendered by Falcon's own rasteriser —
a moving textured car with wheels whose pose and wheel angles are recovered, a pedestrian's deformation, fisheye
cube faces against the lens fast path, a lidar on a plane with exact ranges, ray drop learned on a planted mask —,
the dynamic sources (a synthetic Falcon run with actors.jsonl and lidar sweeps, a synthetic nuScenes folder with six
cameras, LIDAR_TOP and instance tracks — no nuScenes data is on this machine), a dynamic training with the neural
lidar end to end, its re-rendered run, the CLI, and the closed-loop simulator driving a policy through it into a run
folder the traffic-rule checker accepts. Everything on the CPU, tiny. The real-data acceptance checks (nuScenes mini,
CARLA at a 3 m offset) are marked nuscenes / real_server and skipped here."""
import json
import math
import os

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from falcon_autonomy.core.recon import actors as A  # noqa: E402
from falcon_autonomy.core.recon import lidar as LD  # noqa: E402
from falcon_autonomy.core.recon import lens_raster as LR  # noqa: E402
from falcon_autonomy.core.recon.gaussians import Gaussians, inverse_sigmoid, rgb_to_sh0  # noqa: E402

S3 = np.diag([1.0, -1.0, 1.0])


@pytest.fixture(autouse=True)
def _threads():
    n = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(n)


# ================================================================================================ scenes
def static_gaussians(means, scales, cols, ops=None, quats=None) -> Gaussians:
    """Degree-0 Gaussians (no view dependence) from plain arrays."""
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731
    n = len(means)
    q = np.tile([1.0, 0, 0, 0], (n, 1)) if quats is None else quats
    o = np.full(n, 0.95) if ops is None else ops
    return Gaussians(t(means), torch.log(t(scales)), t(q), inverse_sigmoid(t(o)), rgb_to_sh0(t(cols))[:, None, :],
                     torch.zeros(n, 0, 3))


def ground(x0=-12, x1=12, y0=-10, y1=10, step=0.5, seed=0):
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step))
    P = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], 1)
    chk = ((np.floor(P[:, 0]) + np.floor(P[:, 1])) % 2)[:, None]
    col = np.clip(np.array([0.35, 0.36, 0.38]) * (0.7 + 0.5 * chk) + rng.normal(0, 0.03, (len(P), 3)), 0, 1)
    sc = np.tile([0.6 * step, 0.6 * step, 0.02], (len(P), 1))
    return P, sc, col


def car_track(t=np.arange(0, 1.61, 0.2), x0=-4.0, speed=5.0, y=0.0, yaw_rate=0.0, size=(4.0, 1.8, 1.4), tid=7,
              cls='car'):
    yaw = yaw_rate * t
    c = np.stack([x0 + speed * t, np.full(len(t), y), np.full(len(t), size[2] / 2)], 1)
    q = np.stack([np.cos(yaw / 2), np.zeros(len(t)), np.zeros(len(t)), np.sin(yaw / 2)], 1)
    return A.Track(tid, cls, f'vehicle.{cls}', True, size, np.asarray(t, float), c, q)


def textured_car(track, seed=0, n=500):
    """A car's actor: a box of Gaussians with a coarse colour pattern (so its pose is observable), wheels with a bright
    marker on their tread (so their angle is)."""
    P = A.box_surface(track.size, n, seed)
    pat = ((np.floor(P[:, 0] / 0.5) + np.floor(P[:, 2] / 0.35)) % 3) / 2.0
    col = np.stack([0.2 + 0.7 * pat, 0.3 + 0.2 * (1 - pat), 0.8 - 0.6 * pat], 1)
    a = A.ActorModel(track, P, np.full((n, 3), 0.11), np.tile([1.0, 0, 0, 0], (n, 1)), np.full(n, 0.95), col)
    with torch.no_grad():
        wheel = a.part >= 0
        idx = torch.nonzero(wheel).squeeze(1)
        around = (torch.arange(len(idx)) // A.WHEEL_ACROSS) % A.WHEEL_AROUND
        marker = idx[(around <= 2)]
        a.p['sh_dc'][marker] = rgb_to_sh0(torch.tensor([0.95, 0.95, 0.2]))
        a.p['opacity_logits'][wheel] = float(inverse_sigmoid(torch.tensor(0.97)))
    return a


def side_cameras(w=96, h=64, dist=8.0, height=1.0, fov=70.0):
    """World -> optical of two cameras looking at the origin from each side (world: x forward, y left, z up)."""
    from falcon_autonomy.core.recon.raster import fov_K
    out = []
    for side in (-1.0, 1.0):
        c = np.array([0.0, side * dist, height])
        z = -c / np.linalg.norm(c)
        x = np.cross(z, [0, 0, 1.0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z])
        vm = np.eye(4)
        vm[:3, :3], vm[:3, 3] = R, -R @ c
        out.append((torch.tensor(vm, dtype=torch.float32), fov_K(w, h, fov), w, h))
    return out


# ================================================================================================ tracks and wheels
def test_tracks_interpolate_and_the_wheels_follow_the_motion():
    tr = car_track(yaw_rate=0.3)
    R, c = tr.pose(0.5)
    assert np.allclose(c[0], -4 + 2.5, atol=1e-9) and abs(math.atan2(R[1, 0], R[0, 0]) - 0.15) < 1e-3
    assert tr.pose(5.0) is None and tr.pose(1.65) is not None            # 0.11 s of padding at the end
    a = textured_car(tr)
    st = a.wheel_state(1.0)
    assert len(st) == 4
    # spin = distance / radius (the box went ~5 m along its heading), front wheels steer by the bicycle model
    assert st[2][0] == pytest.approx(5.0 / a.wheels[2]['radius'], rel=0.02)
    steer = float(a._interp(a.steer_keys, 1.0))         # atan(L ψ̇ / v), v along the heading
    assert steer == pytest.approx(math.atan(a.wheelbase * 0.3 / 5.0), rel=0.1)
    left, right = st[0][1], st[1][1]
    assert left > steer > right > 0 and st[2][1] == 0.0                   # Ackermann: the inner (left) wheel more
    assert A.kind_of('autorickshaw', 'vehicle.autorickshaw') == 'autorickshaw'
    auto = textured_car(car_track(cls='autorickshaw', size=(2.6, 1.4, 1.7)))
    assert len(auto.wheels) == 3 and sum(w['steers'] for w in auto.wheels) == 1
    assert A.kind_of('pedestrian', 'human.pedestrian.adult') == 'pedestrian'


def test_moving_car_pose_and_wheel_angles_are_recovered():
    """A textured car with wheels driving past two side cameras, rendered by Falcon's rasteriser at its true pose and
    wheel phases; then the same car with its logged box 0.25 m and 4° off and its wheels' phases unknown: the learned
    correction and phases, fitted to the images alone, recover the true pose and wheel angles."""
    torch.manual_seed(0)
    Pg, sg, cg = ground(-8, 8, -7, 7, 0.6)
    gs = static_gaussians(Pg, sg, cg)
    truth = textured_car(car_track())
    phases = torch.tensor([0.35, -0.3, 0.3, -0.35])
    with torch.no_grad():
        truth.phase.copy_(phases)
    cams = side_cameras(80, 56)
    times = truth.track.t[1:-1:3]
    gt_model = A.DynamicModel(gs, [truth])
    with torch.no_grad():
        imgs = [[gt_model.render(vm, K, w, h, float(t))['image'] for vm, K, w, h in cams] for t in times]
    # the logged box is off: translated and turned
    off = car_track()
    off.centre = off.centre + np.array([0.25, -0.2, 0.0])
    dq = np.array([math.cos(math.radians(2)), 0, 0, math.sin(math.radians(2))])
    off.quat = np.array([A.rot_to_quat_np(A.quat_to_rot_np(dq)[0] @ A.quat_to_rot_np(q)[0])[0] for q in off.quat])
    model_a = textured_car(off)
    model = A.DynamicModel(gs, [model_a])
    opt = torch.optim.Adam([model_a.xi, model_a.phase], lr=0.03)
    for it in range(110):
        loss = 0.0
        for k, t in enumerate(times):
            for (vm, K, w, h), gt in zip(cams, imgs[k]):
                loss = loss + (model.render(vm, K, w, h, float(t))['image'] - gt).abs().mean()
        loss = loss + 1e-4 * model_a.regulariser()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        for t in times:
            wt, wm = truth.world(float(t)), model_a.world(float(t))
            body = truth.part < 0
            err = (wt['means'][body] - wm['means'][body]).norm(dim=1)
            assert float(err.mean()) < 0.03, float(err.mean())
            # wheel angles: the learned phase + the shared spin against the truth's (wrapped)
        d = torch.remainder(model_a.phase - phases + math.pi, 2 * math.pi) - math.pi
        assert float(d.abs().max()) < math.radians(6), np.degrees(d.numpy())


def test_pedestrian_deformation_replays_a_limb_and_keeps_the_body_rigid():
    """A pedestrian whose 'arm' (a cluster of Gaussians) swings forward and back: canonical Gaussians plus the
    deformation network, fitted to two side views at four instants, reproduce the arm's motion; the rigidity and
    size regularisers keep the torso where it is."""
    torch.manual_seed(0)
    tr = car_track(t=np.arange(0, 1.21, 0.4), x0=0.0, speed=0.0, size=(0.5, 0.6, 1.8), tid=3, cls='pedestrian')
    tr.dynamic = True
    tr.kind = 'pedestrian'
    rng = np.random.default_rng(1)
    torso = np.c_[rng.uniform(-0.1, 0.1, 120), rng.uniform(-0.2, 0.2, 120), rng.uniform(-0.9, 0.6, 120)]
    arm = np.c_[rng.uniform(-0.05, 0.05, 40), np.full(40, 0.28), rng.uniform(-0.2, 0.4, 40)]
    P = np.r_[torso, arm]
    col = np.r_[np.tile([0.2, 0.3, 0.8], (120, 1)), np.tile([0.9, 0.5, 0.2], (40, 1))]
    n = len(P)

    def swing(t):
        return 0.25 * math.sin(2 * math.pi * t / 1.2)
    Pg, sg, cg = ground(-4, 4, -4, 4)
    gs = static_gaussians(Pg, sg, cg)
    cams = side_cameras(64, 64, 4.0, 0.9, 50.0)
    imgs = []
    for t in tr.t:
        Pt = P.copy()
        Pt[120:, 0] += swing(t)
        g = A.ActorModel(tr, Pt, np.full((n, 3), 0.05), np.tile([1.0, 0, 0, 0], (n, 1)), np.full(n, 0.95), col,
                         wheels=False, deform=False)
        with torch.no_grad():
            imgs.append([A.DynamicModel(gs, [g]).render(vm, K, w, h, float(t))['image'] for vm, K, w, h in cams])
    ped = A.ActorModel(tr, P, np.full((n, 3), 0.05), np.tile([1.0, 0, 0, 0], (n, 1)), np.full(n, 0.95), col,
                       wheels=False)
    assert ped.deform is not None and len(ped.wheels) == 0
    model = A.DynamicModel(gs, [ped])
    opt = torch.optim.Adam(ped.deform.parameters(), lr=3e-3)
    first = None
    for it in range(150):
        loss = 0.0
        for k, t in enumerate(tr.t):
            for (vm, K, w, h), gt in zip(cams, imgs[k]):
                loss = loss + (model.render(vm, K, w, h, float(t))['image'] - gt).abs().mean()
            loss = loss + 0.05 * ped.regulariser(w_def=0.1, w_temporal=0.1, w_rigid=1.0, t=float(t))
        first = float(loss.detach()) if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < 0.5 * first
    with torch.no_grad():
        mu = ped.p['means']
        for t in tr.t:
            d = ped.deform(mu, ped._tnorm(float(t)))
            arm_dx = float(d[120:, 0].mean())
            assert arm_dx == pytest.approx(swing(t), abs=0.08), (t, arm_dx, swing(t))
            assert float(d[:120].norm(dim=1).mean()) < 0.05


# ================================================================================================ fisheye
def test_fisheye_cube_faces_and_the_lens_fast_path_agree_in_the_centre():
    from test_recon import street_gaussians
    from falcon_autonomy.core.lens import output_rays
    means, scales, quats, ops, cols, _ = street_gaussians(spacing=0.5, length=30.0)
    R = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)
    vm = np.eye(4)
    vm[:3, :3], vm[:3, 3] = R, -R @ np.array([0, 0, 1.6])
    vm = torch.tensor(vm, dtype=torch.float32)
    intr = LR.fisheye_intrinsics(160, 120, 190.0)
    a = LR.render_cube(means, scales, quats, ops, cols, vm, intr)
    b = LR.render_lens(means, scales, quats, ops, cols, vm, intr)
    assert a['views'] == 5
    d, ok = output_rays(intr)
    th = np.degrees(np.arccos(np.clip(d[..., 2], -1, 1)))

    def psnr(lo, hi):
        m = torch.as_tensor((th >= lo) & (th < hi) & ok)
        return -10 * math.log10(float(((a['image'] - b['image']) ** 2).mean(-1)[m].mean()))
    centre, rim = psnr(0, 60), psnr(90, 96)
    assert centre > 28.0 and rim < centre - 3.0, (centre, rim)
    assert torch.equal(a['field'], b['field'])
    # within the field both see the street; outside it nothing
    assert float(a['alpha'][~a['field']].abs().max()) == 0.0 and float(a['alpha'][a['field']].mean()) > 0.5
    # the pinhole case of the reference is the rasteriser itself
    from falcon_autonomy.core.lens import intrinsics
    from falcon_autonomy.core.recon.raster import fov_K, render
    pin = intrinsics({'type': 'camera.pinhole', 'width': 96, 'height': 64, 'fov': 80.0})
    c = LR.render_cube(means, scales, quats, ops, cols, vm, pin)
    r = render(means, scales, quats, ops, cols, vm, fov_K(96, 64, 80.0), 96, 64)
    assert c['views'] == 1
    assert float((c['image'] - r['image'])[2:-2, 2:-2].abs().max()) < 0.05


# ================================================================================================ lidar
def wall(x=10.0, half_y=8.0, half_z=3.0, step=0.1):
    ys, zs = np.meshgrid(np.arange(-half_y, half_y + 1e-9, step), np.arange(-half_z, half_z + 1e-9, step))
    P = np.stack([np.full(ys.size, x), ys.ravel(), zs.ravel()], 1)
    n = len(P)
    return {'means': torch.tensor(P, dtype=torch.float32),
            'scales': torch.tensor(np.tile([0.005, 0.7 * step, 0.7 * step], (n, 1)), dtype=torch.float32),
            'quats': torch.tensor(np.tile([1.0, 0, 0, 0], (n, 1)), dtype=torch.float32),
            'opacity': torch.full((n,), 0.95), 'lidar_int': torch.zeros(n), 'lidar_drop': torch.full((n,), -6.0),
            'velocity': torch.zeros(n, 3)}


def test_lidar_on_a_plane_has_exact_ranges():
    G = wall()
    spec = LD._spec(np.linspace(-15, 15, 16), 512)
    out = LD.render_sweep(lambda t: G, spec, np.eye(4), 0.0)
    D = spec.directions()
    truth = np.where(D[..., 0] > 0.3, 10.0 / np.maximum(D[..., 0], 1e-9), np.nan)
    on = np.isfinite(truth) & (np.abs(D[..., 1] * truth) < 7.5) & (np.abs(D[..., 2] * truth) < 2.5)
    pts = LD.to_points(spec, out)
    err = np.abs(pts['range'] - truth)[on]
    assert np.isfinite(pts['range'][on]).all() and err.max() < 0.01 and np.median(err) < 0.002, err.max()
    off = ~np.isfinite(truth)
    assert float(out['p_return'].numpy()[off].max()) < 1e-6
    # the points lie on the wall, in the sensor frame
    assert np.allclose(pts['points'][:, 0], 10.0, atol=0.01)
    # the beam targets of those points give back the same ranges, and a moved sensor sees the wall nearer
    tgt = LD.beam_targets(spec, pts['points'][:, :3], pts['points'][:, 3])
    assert np.nanmax(np.abs(tgt['range'] - pts['range'])) < 1e-4
    T = np.eye(4)
    T[0, 3] = 2.0
    out2 = LD.to_points(spec, LD.render_sweep(lambda t: G, spec, T, 0.0))
    assert np.nanmedian(out2['range'][on]) < np.nanmedian(pts['range'][on]) - 1.9


def test_ray_drop_is_learned_on_a_planted_mask():
    """A wall the recording sees except a window of beams (as glass would drop them): the Gaussians' drop logits,
    fitted by the cross-entropy of p_return, learn it — the window drops, the rest returns, the ranges stay."""
    G = wall()
    G['lidar_drop'] = torch.full_like(G['lidar_drop'], -2.0).requires_grad_()
    spec = LD._spec(np.linspace(-15, 15, 16), 512)
    D = spec.directions()
    truth = np.where(D[..., 0] > 0.3, 10.0 / np.maximum(D[..., 0], 1e-9), np.nan)
    el = np.degrees(np.arcsin(D[..., 2]))
    az = np.degrees(np.arctan2(D[..., 1], D[..., 0]))
    window = (np.abs(az) < 12) & (el > -2) & (el < 8)
    on = np.isfinite(truth) & (np.abs(D[..., 1] * truth) < 7.5) & (np.abs(D[..., 2] * truth) < 2.5)
    rec = np.where(on & ~window, truth, np.nan)
    tgt = {'range': rec, 'intensity': np.zeros_like(truth), 'returned': np.isfinite(rec),
           'observed': np.isfinite(rec).any(0)}
    opt = torch.optim.Adam([G['lidar_drop']], lr=0.2)
    for _ in range(40):
        out = LD.render_sweep(lambda t: G, spec, np.eye(4), 0.0, grad=True)
        loss, parts = LD.lidar_loss(out, tgt, w_range=0.0, w_drop=1.0, w_int=0.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        out = LD.render_sweep(lambda t: G, spec, np.eye(4), 0.0)
    p = out['p_return'].numpy()
    inner = window & on & (np.abs(az) < 9) & (el > 0) & (el < 6)
    assert p[inner].mean() < 0.15 and p[on & ~window & (np.abs(az) > 16)].mean() > 0.85, \
        (p[inner].mean(), p[on & ~window].mean())
    cmp = LD.compare_sweep(spec, out, tgt)
    assert cmp['drop_accuracy'] > 0.9 and cmp['range_median_abs_m'] < 0.01


def test_radar_is_geometric_with_track_velocities():
    G = wall(x=20.0, half_y=3.0, half_z=1.5, step=0.2)
    G['velocity'] = torch.tensor([[-5.0, 0.0, 0.0]]).repeat(len(G['means']), 1)       # coming towards the sensor
    det = LD.render_radar(lambda t: G, {'horizontal_fov': 30.0, 'vertical_fov': 10.0, 'range': 100.0}, np.eye(4),
                          0.0, ego_velocity=(2.0, 0.0, 0.0))
    d = det['detections']
    assert det['learned'] is False and len(d) > 20
    centre = np.abs(d[:, 1]) < 0.05
    assert np.allclose(d[centre, 0], -7.0, atol=0.2) and np.allclose(d[centre, 3], 20.0, atol=0.1)


# ================================================================================================ sources
def write_dynamic_run(path, n=20, w=96, h=54, fps=10.0):
    """A finished falcon-raw run rendered by Falcon's rasteriser: a street, the ego at 2 m/s along +x, a car (with
    wheels) overtaking in the next lane at 5 m/s; rgb, a lidar (16 rings × 256 columns: its sweeps rendered from the
    true scene), actors.jsonl, frames.jsonl, drive.json, dataset.json, run.json."""
    import cv2
    from test_recon import carla_cam_viewmat, street_gaussians
    from falcon_autonomy.core.recon.raster import fov_K
    os.makedirs(path, exist_ok=True)
    means, scales, quats, ops, cols, _ = street_gaussians(length=40.0)
    gs = static_gaussians(means.numpy(), scales.numpy(), cols.numpy(), ops.numpy(), quats.numpy())
    times = np.arange(n) / fps
    car = textured_car(car_track(t=times, x0=4.0, speed=5.0, y=2.6, size=(3.8, 1.8, 1.4), tid=42))
    model = A.DynamicModel(gs, [car])
    K = fov_K(w, h, 90.0)
    cam = {'id': 'cam-front', 'name': 'front', 'type': 'camera.pinhole', 'width': w, 'height': h, 'fov': 90.0,
           'pose': {'x': 1.5, 'y': 0.0, 'z': 1.6, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}}
    lid = {'id': 'lid', 'name': 'top', 'type': 'lidar', 'channels': 16, 'upper_fov': 5.0, 'lower_fov': -15.0,
           'points_per_second': 16 * 256 * 10, 'rotation_frequency': 10.0, 'range': 60.0,
           'pose': {'x': 0.0, 'y': 0.0, 'z': 1.9, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}}
    spec = LD.spec_of_sensor(lid)
    os.makedirs(os.path.join(path, 'rgb', 'cam-front'), exist_ok=True)
    os.makedirs(os.path.join(path, 'lidar', 'lid'), exist_ok=True)
    frames, actor_rows = [], []
    for k, t in enumerate(times):
        x = 2.0 * t
        ego = {'x': x, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        vm = torch.tensor(carla_cam_viewmat(ego), dtype=torch.float32)
        with torch.no_grad():
            out = model.render(vm, K, w, h, float(t))
        rgb = out['image'] + (1 - out['alpha'])[..., None] * torch.tensor([0.55, 0.7, 0.9])
        cv2.imwrite(os.path.join(path, f'rgb/cam-front/{k:06d}.png'),
                    cv2.cvtColor((rgb.clamp(0, 1).numpy() * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
        rec = {'step': k, 'frame': k, 'sim_t': round(float(t), 6), 'ego': ego, 'speed_mps': 2.0, 'recorded': True,
               'files': {'cam-front/rgb': f'rgb/cam-front/{k:06d}.png'}}
        T = np.eye(4)
        T[:3, 3] = [x, 0.0, 1.9]
        with torch.no_grad():
            sw = LD.to_points(spec, LD.render_sweep(model.gaussians_at, spec, T, float(t)))
        pts = sw['points'].copy()
        pts[:, 1] *= -1.0
        pts[:, 3] = 0.5
        pts.astype(np.float32).tofile(os.path.join(path, f'lidar/lid/{k:06d}.bin'))
        rec['files']['lid/lidar'] = f'lidar/lid/{k:06d}.bin'
        rec['lidar_pose'] = {'lid': [1, 0, 0, x, 0, 1, 0, 0.0, 0, 0, 1, 1.9]}
        frames.append(rec)
        R, c = car.track.pose(float(t))
        yaw = -math.degrees(math.atan2(R[1, 0], R[0, 0]))
        actor_rows.append({'step': k, 'sim_t': round(float(t), 6), 'actors': [
            {'id': 1, 'type_id': 'vehicle.lincoln.mkz_2020', 'role': 'hero', 'category': 'vehicle.car', **ego,
             'vx': 2.0, 'vy': 0.0, 'vz': 0.0, 'extent': [2.45, 1.05, 0.75], 'bbox_center': [0.0, 0.0, 0.75]},
            {'id': 42, 'type_id': 'vehicle.test.car', 'role': '', 'category': 'vehicle.car', 'x': float(c[0]),
             'y': float(-c[1]), 'z': float(c[2]), 'roll': 0.0, 'pitch': 0.0, 'yaw': yaw, 'vx': 5.0, 'vy': 0.0,
             'vz': 0.0, 'extent': [1.9, 0.9, 0.7], 'bbox_center': [0.0, 0.0, 0.0]}]})
    for name, rows in (('frames.jsonl', frames), ('actors.jsonl', actor_rows)):
        with open(os.path.join(path, name), 'w') as fh:
            for r in rows:
                fh.write(json.dumps(r) + '\n')
    rig = {'id': 'rig', 'name': 'synthetic', 'vehicle': 'vehicle.lincoln.mkz_2020', 'sensors': [cam, lid]}
    json.dump({'town': 'Synthetic', 'rig': rig, 'fps': fps}, open(os.path.join(path, 'drive.json'), 'w'))
    Kc = [[K[0], 0, K[2]], [0, K[1], K[3]], [0, 0, 1]]
    json.dump({'schema': 'falcon.dataset/1', 'layout': 'falcon-raw', 'frames': n, 'fps': fps,
               'sensors': [dict(cam, K=Kc, streams=['rgb'])]}, open(os.path.join(path, 'dataset.json'), 'w'))
    json.dump({'schema': 'falcon.run/1', 'status': 'ok', 'started': '2026-11-12T00:00:00Z', 'name': 'dyn'},
              open(os.path.join(path, 'run.json'), 'w'))
    return path, car


def test_falcon_source_in_dynamic_mode(tmp_path):
    from falcon_autonomy.core.recon import data as D
    run, car = write_dynamic_run(str(tmp_path / 'run'), n=8)
    cap = D.open_capture(f'falcon:{run}?dynamic=1')
    assert cap.meta['dynamic'] and set(cap.tracks) == {42} and 'learned' in cap.meta['masks']
    tr = cap.tracks[42]
    assert tr.kind == 'car' and tr.dynamic and tr.size == pytest.approx((3.8, 1.8, 1.4))
    R, c = tr.pose(0.35)
    Rt, ct = car.track.pose(0.35)
    assert np.allclose(c, ct, atol=1e-6) and np.allclose(R, Rt, atol=1e-6)
    assert len(cap.sweeps) == 8 and cap.lidar_spec.rings == 16 and cap.lidar_spec.columns == 256
    P, inten, ids = cap.sweeps[3].load()
    tgt = LD.beam_targets(cap.lidar_spec, P, inten)
    assert tgt['returned'].sum() > 0.5 * len(P) and ids is None
    pts = D.actor_points(cap)
    assert 42 in pts and len(pts[42][0]) > 30
    L, W, H = tr.size
    assert np.all(np.abs(pts[42][0]) <= np.array([L, W, H]) / 2 + 0.11)
    # the static source is unchanged: no tracks, no sweeps
    st = D.open_capture(f'falcon:{run}')
    assert st.tracks is None and not st.sweeps


def _write_nuscenes_dynamic(root, n=4):
    """nuScenes-shaped tables: one scene, n key frames of the six cameras and LIDAR_TOP (x y z intensity ring), one
    car driving past (an instance with n annotations) and one parked."""
    import cv2
    v = os.path.join(root, 'v1.0-mini')
    os.makedirs(v, exist_ok=True)
    chans = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
    yaws = [0, -55, -110, 180, 110, 55]
    tabs = {'scene': [{'token': 'sc', 'name': 'scene-0001', 'log_token': 'lg', 'nbr_samples': n}],
            'log': [{'token': 'lg', 'location': 'boston-seaport'}],
            'sensor': [{'token': f's{c}', 'channel': c} for c in chans + ['LIDAR_TOP']],
            'calibrated_sensor': [], 'category': [{'token': 'k1', 'name': 'vehicle.car'}],
            'instance': [{'token': 'i1', 'category_token': 'k1'}, {'token': 'i2', 'category_token': 'k1'}],
            'sample': [], 'sample_data': [], 'ego_pose': [], 'sample_annotation': []}
    for c, yaw in zip(chans, yaws):
        a = math.radians(yaw)
        Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        Rc = Rz @ np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
        q = A.rot_to_quat_np(Rc)[0].tolist()
        tabs['calibrated_sensor'].append({'token': f'c{c}', 'sensor_token': f's{c}', 'translation': [1.0, 0.0, 1.5],
                                          'rotation': q, 'camera_intrinsic': [[400, 0, 399.5], [0, 400, 224.5],
                                                                              [0, 0, 1]]})
        os.makedirs(os.path.join(root, 'samples', c), exist_ok=True)
    tabs['calibrated_sensor'].append({'token': 'cL', 'sensor_token': 'sLIDAR_TOP', 'translation': [0.9, 0.0, 1.8],
                                      'rotation': [1, 0, 0, 0], 'camera_intrinsic': []})
    os.makedirs(os.path.join(root, 'samples', 'LIDAR_TOP'), exist_ok=True)
    for k in range(n):
        ts = 1_500_000_000_000_000 + k * 500_000
        tabs['sample'].append({'token': f'sm{k}', 'scene_token': 'sc', 'timestamp': ts})
        tabs['ego_pose'].append({'token': f'e{k}', 'translation': [100.0 + 2 * k, 50.0, 0.0], 'rotation': [1, 0, 0, 0]})
        for c in chans:
            img = f'samples/{c}/{k}.jpg'
            cv2.imwrite(os.path.join(root, img), np.full((450, 800, 3), 100 + 10 * k, np.uint8))
            tabs['sample_data'].append({'token': f'd{c}{k}', 'sample_token': f'sm{k}', 'calibrated_sensor_token':
                                        f'c{c}', 'ego_pose_token': f'e{k}', 'timestamp': ts, 'filename': img,
                                        'is_key_frame': True, 'width': 800, 'height': 450})
        rings = np.repeat(np.arange(32), 90)
        el = np.radians(np.linspace(-30.67, 10.67, 32))[rings]
        az = np.radians(np.tile(np.arange(90) * 4.0, 32))
        r = np.where(el < 0, 1.8 / np.maximum(np.sin(-el), 0.05), 30.0)
        pts = np.stack([r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el),
                        np.full(len(r), 40.0), rings.astype(float)], 1).astype(np.float32)
        lid = f'samples/LIDAR_TOP/{k}.pcd.bin'
        pts.tofile(os.path.join(root, lid))
        tabs['sample_data'].append({'token': f'l{k}', 'sample_token': f'sm{k}', 'calibrated_sensor_token': 'cL',
                                    'ego_pose_token': f'e{k}', 'timestamp': ts, 'filename': lid, 'is_key_frame': True})
        tabs['sample_annotation'] += [
            {'token': f'a{k}', 'sample_token': f'sm{k}', 'instance_token': 'i1',
             'translation': [110.0 + 4 * k, 53.0, 0.8], 'size': [1.9, 4.5, 1.6], 'rotation': [1, 0, 0, 0]},
            {'token': f'b{k}', 'sample_token': f'sm{k}', 'instance_token': 'i2',
             'translation': [95.0, 45.0, 0.8], 'size': [1.9, 4.5, 1.6], 'rotation': [1, 0, 0, 0]}]
    for name, rows in tabs.items():
        with open(os.path.join(v, name + '.json'), 'w') as fh:
            json.dump(rows, fh)
    return root


def test_nuscenes_six_cameras_lidar_top_and_instance_tracks(tmp_path):
    from falcon_autonomy.core.recon import data as D
    root = _write_nuscenes_dynamic(str(tmp_path / 'nusc'))
    cap = D.open_capture(f'nuscenes:{root}?scene=scene-0001&dynamic=1')
    assert cap.meta['cameras'] == ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT',
                                   'CAM_FRONT_LEFT']
    assert len(cap.frames) == 24 and {f.camera for f in cap.frames} == set(cap.meta['cameras'])
    assert all(f.mask is None for f in cap.frames)                              # learned, not masked
    assert len(cap.tracks) == 2
    moving = [t for t in cap.tracks.values() if t.dynamic]
    parked = [t for t in cap.tracks.values() if not t.dynamic]
    assert len(moving) == 1 and len(parked) == 1 and moving[0].size == pytest.approx((4.5, 1.9, 1.6))
    assert moving[0].velocity(0.7)[0] == pytest.approx(8.0)                     # 4 m per 0.5 s
    # LIDAR_TOP: the beam grid measured from the ring indices; sweeps in the world through ego ∘ calibration
    assert cap.lidar_spec.rings == 32 and cap.lidar_spec.columns == 90
    assert cap.lidar_spec.elevations[0] == pytest.approx(-30.67, abs=0.05)
    assert len(cap.sweeps) == 4 and np.allclose(cap.sweeps[1].T[:3, 3], [102.9, 50.0, 1.8])
    P, inten, ids = cap.sweeps[0].load()
    assert P.shape[1] == 3 and inten.max() <= 1.0 and ids is None
    tgt = LD.beam_targets(cap.lidar_spec, P, inten)
    assert tgt['returned'].sum() == len(P)
    # the front camera sees straight ahead of the ego
    f = next(fr for fr in cap.frames if fr.camera == 'CAM_FRONT')
    pc = f.viewmat @ np.array([100.0 + 1.0 + 20.0, 50.0, 1.5, 1.0])
    assert np.allclose(pc[:3], [0, 0, 20.0], atol=1e-6)
    # the single-camera static source is unchanged
    st = D.open_capture(f'nuscenes:{root}?scene=scene-0001')
    assert st.camera == 'CAM_FRONT' and len(st.frames) == 4 and st.tracks is None


# ======================================================================================== training, CLI, closed loop
def _settings(**kw):
    s = {'iterations': 80, 'width': 96, 'device': 'cpu', 'eval_every': 40, 'log_every': 20, 'sky_points': 300,
         'densify_from': 30, 'densify_every': 20, 'max_gaussians': 20000, 'renders': 1, 'voxel_m': 0.3,
         'holdout': 5, 'actor_points': 300}
    s.update(kw)
    return s


@pytest.fixture(scope='module')
def dynamic_recon(tmp_path_factory):
    from falcon_autonomy.core.recon.train import train
    root = tmp_path_factory.mktemp('dyn')
    run, car = write_dynamic_run(str(root / 'run'))
    out = str(root / 'recon')
    torch.set_num_threads(4)
    summary = train(f'falcon:{run}', out, _settings(dynamic=True, lidar=True))
    return run, out, summary


def test_dynamic_training_writes_actors_and_a_lidar(dynamic_recon):
    from falcon_autonomy.core.provenance import read_json
    from falcon_autonomy.core.recon.scene import Recon
    run, out, summary = dynamic_recon
    for f in ('point_cloud.ply', 'extras.npz', 'actors.pt', 'actors.json', 'lidar.json', 'views.json',
              'evaluation.json'):
        assert os.path.isfile(os.path.join(out, f)), f
    rj = read_json(os.path.join(out, 'run.json'))
    assert rj['dynamic']['actors'] == 1 and rj['dynamic']['kinds'] == {'car': 1}
    assert rj['lidar']['heldout_sweeps'] == 4 and summary['lidar']['sweeps'] == 4
    ev = read_json(os.path.join(out, 'evaluation.json'))
    assert ev['heldout']['mean']['psnr'] > 18.0, ev['heldout']['mean']
    lm = ev['lidar']['mean']
    assert lm['range_median_abs_m'] is not None and lm['range_median_abs_m'] < 0.5, lm
    curves = [json.loads(x) for x in open(os.path.join(out, 'curves.jsonl'))]
    assert any('lidar' in c for c in curves)
    rec = Recon.load(out, 'cpu')
    assert rec.dynamic and len(rec.actors) == 1 and rec.lidar_spec.rings == 16
    # the car is drawn where its track is at the frame's time: two instants differ where it drove
    f = rec.frame(4)
    cam = rec.camera(f)
    a, b = rec.render(cam, t=0.2), rec.render(cam, t=1.2)
    assert np.abs(a['image'].astype(int) - b['image'].astype(int)).mean() > 0.5
    assert (a['actor'] == 0).any()
    sw = rec.lidar_sweep(np.asarray(f['ego']).reshape(4, 4), f['t'])
    assert sw['returns'] > 100


def test_dynamic_cli_render_fisheye_lidar_eval_and_rerender(dynamic_recon, tmp_path):
    from falcon_autonomy.core.provenance import read_json
    from falcon_autonomy.core.recon.commands import main_recon
    run, out, _ = dynamic_recon
    img = str(tmp_path / 'fish.png')
    assert main_recon(['render', out, '--frame', '3', '--camera', 'fisheye', '--fisheye-size', '128x128',
                       '--lidar', '--radar', '--out', img, '--device', 'cpu']) == 0
    import cv2
    assert cv2.imread(img).shape == (128, 128, 3)
    pts = np.fromfile(str(tmp_path / 'fish-lidar.bin'), np.float32).reshape(-1, 4)
    assert len(pts) > 100
    assert os.path.isfile(str(tmp_path / 'fish-radar.npy'))
    assert main_recon(['render', out, '--frame', '3', '--camera', 'fisheye', '--fisheye-method', 'lens',
                       '--fisheye-size', '96x96', '--out', str(tmp_path / 'f2.png'), '--device', 'cpu']) == 0
    assert main_recon(['eval', out, '--lidar', '--device', 'cpu']) == 0
    assert 'lidar' in read_json(os.path.join(out, 'evaluation.json'))
    rr = str(tmp_path / 'rerender')
    assert main_recon(['rerender', out, '--out', rr, '--lateral', '0.5', '--every', '4', '--device', 'cpu']) == 0
    ds = read_json(os.path.join(rr, 'dataset.json'))
    na = ds['reconstruction']['not_available']
    assert 'actors.jsonl' not in na and 'lidar' not in na and 'radar' in na
    rows = [json.loads(x) for x in open(os.path.join(rr, 'actors.jsonl'))]
    assert rows and rows[0]['actors'][0]['role'] == 'hero' and any(a['id'] == 42 for r in rows for a in r['actors'])
    fr = [json.loads(x) for x in open(os.path.join(rr, 'frames.jsonl'))]
    assert 'lidar_pose' in fr[0] and os.path.isfile(os.path.join(rr, fr[0]['files']['recon-lidar/lidar']))
    # train refuses the lidar without the dynamic mode
    assert main_recon(['train', '--source', f'falcon:{run}', '--out', str(tmp_path / 'x'), '--lidar', '--device',
                       'cpu', '--iterations', '1']) == 2


def test_trust_envelope_from_the_lateral_shift_measure():
    from falcon_autonomy.core.recon.closedloop import expected_psnr, trust_envelope
    ev = {'novel_view': {'lateral_warped': {'shifts': [
        {'lateral_m': 0.0, 'psnr': 30.0}, {'lateral_m': 0.5, 'psnr': 29.0}, {'lateral_m': -0.5, 'psnr': 28.5},
        {'lateral_m': 1.0, 'psnr': 27.5}, {'lateral_m': -1.0, 'psnr': 26.0}, {'lateral_m': 2.0, 'psnr': 24.0},
        {'lateral_m': -2.0, 'psnr': 25.0}]}}}
    env = trust_envelope(ev)
    assert env['trusted_m'] == 0.5 and env['control_psnr'] == 30.0
    assert env['curve'] == [[0.0, 30.0], [0.5, 28.5], [1.0, 26.0], [2.0, 24.0]]
    assert expected_psnr(env, 0.75) == pytest.approx(27.25) and expected_psnr(env, 3.0) is None
    assert trust_envelope(None)['trusted_m'] is None


def test_closed_loop_drives_a_policy_into_a_run_the_rules_accept(dynamic_recon, tmp_path):
    from falcon_autonomy.core import rules
    from falcon_autonomy.core.policy import rollout
    from falcon_autonomy.core.provenance import write_json
    from falcon_autonomy.core.recon.closedloop import ConstantSpeedPolicy, LoggedActors, NeuralSim, kin_step
    run, out, _ = dynamic_recon
    ev_p = os.path.join(out, 'evaluation.json')
    with open(ev_p) as fh:
        ev = json.load(fh)
    ev['novel_view'] = {'lateral_warped': {'shifts': [{'lateral_m': s, 'psnr': 25.0 - abs(s) * 3}
                                                       for s in (0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0)]}}
    write_json(ev_p, ev)
    # the kinematic model is the director's: straight at 2 m/s, then a right turn turns right (CARLA: yaw grows)
    st = kin_step([0.0, 0.0, 0.0, 2.0], {'throttle': 0.0, 'brake': 0.0, 'steer': 0.0}, 0.1, 2.9)
    assert st == pytest.approx([0.2, 0.0, 0.0, 2.0])
    assert kin_step([0, 0, 0, 2.0], {'steer': 0.5}, 0.1, 2.9)[2] > 0
    sim = NeuralSim(out, str(tmp_path / 'cl'), obs_size=(64, 36), device='cpu')
    assert sim.envelope['trusted_m'] == 1.0 and isinstance(sim.controller, LoggedActors)
    doc = rollout(ConstantSpeedPolicy(2.0), sim, 1, seed=3, out_dir=str(tmp_path / 'cl'))
    ep = doc['episodes'][0]
    assert ep['termination'] in ('route_complete', 'time_limit') and ep['steps'] > 5, ep['termination']
    assert ep['deviation']['max_m'] < 0.5 and ep['deviation']['steps_beyond_limit'] == 0
    ep_dir = ep['run_dir']
    for f in ('frames.jsonl', 'actors.jsonl', 'drive.json', 'dataset.json', 'run.json', 'closedloop.json'):
        assert os.path.isfile(os.path.join(ep_dir, f)), f
    frames = [json.loads(x) for x in open(os.path.join(ep_dir, 'frames.jsonl'))]
    assert os.path.isfile(os.path.join(ep_dir, frames[3]['files']['cam-front/rgb']))
    res = rules.check_run(ep_dir, xodr='')
    assert res['rules'] and res['run'] == os.path.abspath(ep_dir)
    from falcon_autonomy.core.exporters import Run
    assert len(Run(ep_dir).frames) == len(frames)
    # off the path beyond the limit: flagged, and the episode ends outside the reconstruction
    obs, info = sim.reset(seed=1, options={'lateral_m': 3.5})
    assert obs['state'].shape == (len(sim.feature_names),) and obs['camera'].shape == (36, 64, 3)
    obs, r, term, trunc, info = sim.step([0.0, 0.0])
    assert trunc and not term and info['termination'] == 'outside_reconstruction'
    assert info['deviation']['beyond_limit'] and info['episode_summary']['deviation']['steps_beyond_limit'] >= 1
    # a controller moves the actors: drawn where it says
    class Parked(LoggedActors):
        def poses(self, t):
            return {k: (np.eye(3), np.array([30.0, -2.6, 0.7]), np.zeros(3)) for k in self.tracks}
    sim2 = NeuralSim(out, str(tmp_path / 'cl2'), obs_size=(64, 36), device='cpu', controller=Parked(sim.tracks))
    obs, _ = sim2.reset(seed=0)
    rows = [json.loads(x) for x in open(os.path.join(tmp_path, 'cl2', 'ep-0000', 'frames.jsonl'))] \
        if os.path.isfile(os.path.join(tmp_path, 'cl2', 'ep-0000', 'frames.jsonl')) else None
    sim2.close()
    act = [json.loads(x) for x in open(os.path.join(tmp_path, 'cl2', 'ep-0000', 'actors.jsonl'))]
    car = [a for a in act[0]['actors'] if a['id'] == 42][0]
    assert car['x'] == pytest.approx(30.0) and car['y'] == pytest.approx(2.6)
    sim.close()
    del rows


def test_closedloop_cli(dynamic_recon, tmp_path):
    from falcon_autonomy.core.recon.commands import main_recon
    _, out, _ = dynamic_recon
    d = str(tmp_path / 'cli')
    assert main_recon(['closedloop', out, '--policy', 'constant:7.2', '--out', d, '--episodes', '1',
                       '--obs-size', '48x27', '--deviate', '3.0', '--device', 'cpu']) == 0
    doc = json.load(open(os.path.join(d, 'rollout.json')))
    assert doc['aggregate']['episodes'] == 1 and os.path.isfile(os.path.join(d, 'closedloop.json'))
    assert main_recon(['closedloop', '--help']) == 0


from test_gui import pump, qapp, slot_exceptions, window  # noqa: E402,F401  (fixtures)


@pytest.mark.gui
def test_replay_page_dynamic_training_and_closed_loop_jobs(window, qapp, dynamic_recon, tmp_path):
    """The reconstruction section queues a dynamic training (with the lidar: the run has one) and a closed-loop drive
    through an opened dynamic reconstruction."""
    from falcon_autonomy.core.logreplay import import_log
    from falcon_autonomy.ui.pages.replay import ReplayPage
    run, out, _ = dynamic_recon
    page = ReplayPage(window)
    window.stack.addWidget(page)
    os.makedirs(page.logs_dir(), exist_ok=True)
    import_log('falcon', run).save(os.path.join(page.logs_dir(), 'dyn'))
    page.refresh()
    page.logs.setCurrentIndex(page.logs.findText('dyn'))
    panel = page.recon
    panel.device.setCurrentIndex(panel.device.findData('cpu'))
    panel.dynamic.setChecked(True)
    panel.iters.setValue(10)
    panel.train()
    argv = panel.jobs['train'].spec.argv
    assert '--dynamic' in argv and '--lidar' in argv and argv[argv.index('--camera') + 1] == 'all'
    window.queue.cancel(panel.jobs['train'])
    panel.rec_dir = out
    panel.open_recon()
    assert pump(qapp, lambda: panel.rec is not None, 60)
    assert panel.rec.actors
    d = panel.closedloop(str(tmp_path / 'cl'))
    cl = panel.jobs['closedloop'].spec.argv
    assert cl[cl.index('closedloop') + 1] == out and cl[cl.index('--out') + 1] == d and '--deviate' in cl
    window.queue.cancel(panel.jobs['closedloop'])


# ================================================================================================ real data (skipped)
@pytest.mark.nuscenes
def test_nuscenes_mini_dynamic_reconstruction_acceptance(tmp_path):
    """nuScenes mini (CC BY-NC-SA 4.0; FALCON_NUSCENES=1 and FALCON_NUSCENES_ROOT): a dynamic training with the
    neural lidar on scene-0061's six cameras; acceptance: held-out PSNR ≥ 22 dB and the lidar's median range error on
    held-out sweeps ≤ 0.3 m."""
    from falcon_autonomy.core.recon.train import train
    root = os.environ.get('FALCON_NUSCENES_ROOT', '')
    if not os.path.isdir(root):
        pytest.skip('FALCON_NUSCENES_ROOT is not a nuScenes folder')
    s = train(f'nuscenes:{root}?scene=scene-0061', str(tmp_path / 'r'),
              {'dynamic': True, 'lidar': True, 'camera': 'all', 'iterations': 15000})
    assert s['heldout']['psnr'] >= 22.0 and s['lidar']['range_median_abs_m'] <= 0.3


@pytest.mark.real_server
def test_carla_three_metre_offset_acceptance(tmp_path):
    """A dynamic reconstruction of a recorded CARLA drive measured 3 m off its path against CARLA's own frames there
    (FALCON_REAL_CARLA=1 and FALCON_RECON_ACCEPT=<recon folder>,<reference run 3 m to the side>)."""
    from falcon_autonomy.core.recon.commands import evaluate_command
    spec = os.environ.get('FALCON_RECON_ACCEPT', '')
    if ',' not in spec:
        pytest.skip('FALCON_RECON_ACCEPT=<recon>,<run 3 m to the side> is not set')
    recon, ref = spec.split(',', 1)
    res = evaluate_command(recon, [f'falcon:{ref}'], allow_other_world=False)
    assert res
