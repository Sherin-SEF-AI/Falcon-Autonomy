"""Neural reconstruction (core.recon): the pure-PyTorch rasteriser against finite differences and the pinhole model,
SSIM against scikit-image, the 3DGS .ply layout, density control, an optimisation that recovers a known scene, the
sources (a synthetic Falcon run, KITTI raw and nuScenes folders written here — never real dataset files), training,
evaluation, the depth warp, the re-rendered run, the CLI and the Replay page's section. Everything runs on the CPU
on scenes rendered here; GPU tests run only with FALCON_GPU_TESTS=1."""
import json
import math
import os

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from falcon_autonomy.core.recon import raster as R  # noqa: E402

S3 = np.diag([1.0, -1.0, 1.0])


# ================================================================================================ scenes
def street_gaussians(seed=0, spacing=0.5, length=40.0):
    """A synthetic street in a right-handed world (x forward, y left, z up): a textured ground, two textured walls
    and a back wall, as flat Gaussians. Returns (means, scales, quats, opacities, colours, classes)."""
    rng = np.random.default_rng(seed)
    pts, nrm, cls = [], [], []
    xs = np.arange(-4.0, length, spacing)
    for x in xs:
        for y in np.arange(-6.0, 6.01, spacing):
            pts.append((x, y, 0.0)); nrm.append(2); cls.append(1)
    for side in (-6.0, 6.0):
        for x in xs:
            for z in np.arange(0.25, 5.0, spacing):
                pts.append((x, side, z)); nrm.append(1); cls.append(3)
    for y in np.arange(-6.0, 6.01, spacing):
        for z in np.arange(0.25, 5.0, spacing):
            pts.append((length, y, z)); nrm.append(0); cls.append(3)
    pts = np.array(pts)
    n = len(pts)
    sc = np.full((n, 3), spacing * 0.6)
    sc[np.arange(n), nrm] = 0.02
    checker = ((np.floor(pts[:, 0]) + np.floor(pts[:, 1]) + np.floor(pts[:, 2])) % 2)[:, None]
    base = np.where(np.array(cls)[:, None] == 1, [0.35, 0.35, 0.38], [0.7, 0.45, 0.3])
    col = np.clip(base * (0.6 + 0.5 * checker) + rng.normal(0, 0.05, (n, 3)), 0, 1)
    t = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    return (t(pts), t(sc), t(np.tile([1.0, 0, 0, 0], (n, 1))), t(np.full(n, 0.95)), t(col), np.array(cls))


SKY = np.array([0.55, 0.7, 0.9])


def carla_cam_viewmat(ego_carla: dict, mount=(1.5, 0.0, 1.6)):
    """world (right-handed, the CARLA world mirrored) -> optical, for a CARLA ego pose and a forward camera mount."""
    from falcon_autonomy.core.exporters import pose_matrix, rh
    from falcon_autonomy.core.recon.scene import CAM_OPTICAL
    ego = rh(pose_matrix(ego_carla))
    m = np.eye(4)
    m[:3, 3] = [mount[0], -mount[1], mount[2]]
    m[:3, :3] = CAM_OPTICAL
    T = ego @ m
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def make_falcon_run(path, n=24, w=96, h=54, speed=1.0, fps=10.0, actor=True, lidar=True, lateral=0.0):
    """A finished falcon-raw run of the synthetic street (rendered here): rgb, semantic, instance, range, objects,
    lidar sweeps with labels, frames.jsonl, drive.json, dataset.json, run.json. An actor (a red block, id 42) drives
    ahead of the ego in the first half; the ego moves along +x (CARLA), `lateral` m to the right."""
    import cv2
    os.makedirs(path, exist_ok=True)
    means, scales, quats, ops, cols, cls = street_gaussians()
    fov = 90.0
    K = R.fov_K(w, h, fov)
    cam = {'id': 'cam-front', 'name': 'front', 'type': 'camera.pinhole', 'width': w, 'height': h, 'fov': fov,
           'pose': {'x': 1.5, 'y': 0.0, 'z': 1.6, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}}
    lid = {'id': 'lid', 'name': 'top', 'type': 'lidar', 'pose': {'x': 0.0, 'y': 0.0, 'z': 1.9, 'roll': 0.0,
                                                                 'pitch': 0.0, 'yaw': 0.0}, 'channels': 32}
    for d in ('rgb', 'semantic', 'instance', 'range', 'objects'):
        os.makedirs(os.path.join(path, d, 'cam-front'), exist_ok=True)
    os.makedirs(os.path.join(path, 'lidar', 'lid'), exist_ok=True)
    onehot = torch.nn.functional.one_hot(torch.as_tensor(cls), 30).float()
    frames = []
    for k in range(n):
        x = k * speed / fps
        ego = {'x': x, 'y': lateral, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        vm = torch.tensor(carla_cam_viewmat(ego), dtype=torch.float32)
        mm, ss, qq, oo, cc, hh, aa = means, scales, quats, ops, cols, onehot, torch.zeros(len(means))
        if actor and k < n // 2:
            g = np.mgrid[0:4, 0:3, 0:3].reshape(3, -1).T * 0.4
            am = torch.tensor(g + [x + 8.0, -0.6, 0.2], dtype=torch.float32)
            na = len(am)
            mm = torch.cat([means, am]); ss = torch.cat([scales, torch.full((na, 3), 0.25)])
            qq = torch.cat([quats, torch.tensor([[1.0, 0, 0, 0]]).repeat(na, 1)])
            oo = torch.cat([ops, torch.full((na,), 0.99)])
            cc = torch.cat([cols, torch.tensor([[0.9, 0.1, 0.1]]).repeat(na, 1)])
            hh = torch.cat([onehot, torch.nn.functional.one_hot(torch.full((na,), 14), 30).float()])
            aa = torch.cat([aa, torch.ones(na)])
        feats = torch.cat([cc, hh, aa[:, None]], 1)
        with torch.no_grad():
            out = R.render(mm, ss, qq, oo, feats, vm, K, w, h)
        alpha = out['alpha']
        rgb = out['image'][..., :3] + (1 - alpha)[..., None] * torch.tensor(SKY, dtype=torch.float32)
        sem = out['image'][..., 3:33].argmax(-1)
        sem = torch.where(alpha > 0.5, sem, torch.full_like(sem, 11)).numpy().astype(np.uint8)
        inst = np.where((out['image'][..., 33] > 0.3).numpy(), 5, 0).astype(np.uint16)
        depth = (out['depth'] / alpha.clamp_min(1e-6)) * R.ray_norms(K, w, h)
        rng = np.where(alpha.numpy() > 0.5, np.round(depth.numpy() * 256), 0).astype(np.uint16)
        cv2.imwrite(os.path.join(path, f'rgb/cam-front/{k:06d}.png'),
                    cv2.cvtColor((rgb.clamp(0, 1).numpy() * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(path, f'semantic/cam-front/{k:06d}.png'), sem)
        cv2.imwrite(os.path.join(path, f'instance/cam-front/{k:06d}.png'), inst)
        cv2.imwrite(os.path.join(path, f'range/cam-front/{k:06d}.png'), rng)
        objs = [{'id': 5, 'track_id': 42, 'actor_id': 42, 'class': 'car', 'bbox': [0, 0, 1, 1],
                 'pixels': int((inst == 5).sum())}] if (inst == 5).any() else []
        with open(os.path.join(path, f'objects/cam-front/{k:06d}.json'), 'w') as fh:
            json.dump({'objects': objs, 'width': w, 'height': h}, fh)
        files = {f'cam-front/{s}': f'{s}/cam-front/{k:06d}.{"json" if s == "objects" else "png"}'
                 for s in ('rgb', 'semantic', 'instance', 'range', 'objects')}
        rec = {'step': k, 'frame': 100 + k, 'sim_t': round(k / fps, 6), 'files': files, 'ego': ego,
               'speed_mps': speed, 'recorded': True}
        if lidar and k % 4 == 1:
            # the street's centres within 25 m, in the lidar's frame (CARLA axes), with tags and actor ids
            wc = (means.numpy() @ S3.T)
            lp = np.array([x, lateral, 1.9])
            near = np.linalg.norm(wc - lp, axis=1) < 25
            pts = wc[near] - lp
            tags = cls[near].astype(np.uint8)
            ids = np.zeros(len(pts), np.uint32)
            pts = np.concatenate([pts, [[8.0, 0.6, -1.0]]])           # one return on the actor: dropped
            tags = np.concatenate([tags, [14]]).astype(np.uint8)
            ids = np.concatenate([ids, [42]]).astype(np.uint32)
            sw = len([f for f in frames if 'lidar_pose' in f])
            np.c_[pts, np.ones(len(pts))].astype(np.float32).tofile(os.path.join(path, f'lidar/lid/{sw:06d}.bin'))
            with open(os.path.join(path, f'lidar/lid/{sw:06d}.labels.bin'), 'wb') as fh:
                fh.write(tags.tobytes() + ids.tobytes())
            rec['files']['lid/lidar'] = f'lidar/lid/{sw:06d}.bin'
            rec['lidar_pose'] = {'lid': [1, 0, 0, x, 0, 1, 0, lateral, 0, 0, 1, 1.9]}
        frames.append(rec)
    with open(os.path.join(path, 'frames.jsonl'), 'w') as fh:
        for r in frames:
            fh.write(json.dumps(r) + '\n')
    rig = {'id': 'rig', 'name': 'synthetic', 'vehicle': 'vehicle.test', 'sensors': [cam, lid]}
    weather = {'preset': 'ClearNoon'}
    with open(os.path.join(path, 'drive.json'), 'w') as fh:
        json.dump({'town': 'Synthetic', 'weather': weather, 'rig': rig, 'fps': fps}, fh)
    Kc = [[K[0], 0, K[2]], [0, K[1], K[3]], [0, 0, 1]]
    with open(os.path.join(path, 'dataset.json'), 'w') as fh:
        json.dump({'schema': 'falcon.dataset/1', 'layout': 'falcon-raw', 'frames': n, 'fps': fps,
                   'sensors': [dict(cam, K=Kc, streams=['instance', 'objects', 'range', 'rgb', 'semantic'])],
                   'labels': {'semantic': {'encoding': 'uint8'}}}, fh)
    with open(os.path.join(path, 'run.json'), 'w') as fh:
        json.dump({'schema': 'falcon.run/1', 'status': 'ok', 'started': '2026-10-29T00:00:00Z', 'name': 'synthetic'},
                  fh)
    return path


# ================================================================================================ the rasteriser
def _small_scene(dtype=torch.float64):
    torch.manual_seed(0)
    n = 6
    means = torch.tensor([[0.1, 0.0, 3.0], [-0.4, 0.2, 4.0], [0.3, -0.3, 3.5], [0.0, 0.1, 5.0], [-0.2, -0.2, 2.5],
                          [0.5, 0.4, 4.5]], dtype=dtype)
    ls = torch.log(torch.full((n, 3), 0.15, dtype=dtype) + torch.rand(n, 3, dtype=dtype) * 0.1)
    q = torch.randn(n, 4, dtype=dtype)
    ol = torch.tensor([0.5, 1.0, -0.3, 0.2, 0.8, 0.1], dtype=dtype)
    col = torch.rand(n, 3, dtype=dtype)
    return means, ls, q, ol, col


def test_rasteriser_gradients_match_finite_differences():
    means, ls, q, ol, col = _small_scene()
    W, H = 40, 28
    vm = torch.eye(4, dtype=torch.float64)
    vm[:3, 3] = torch.tensor([0.05, -0.02, 0.1], dtype=torch.float64)
    K = (30.0, 31.0, 20.3, 13.8)
    wt = torch.linspace(0, 1, W * H * 3, dtype=torch.float64).reshape(H, W, 3)

    def f(means, ls, q, ol, col):
        out = R.render(means, torch.exp(ls), q, torch.sigmoid(ol), col, vm, K, W, H, tile=8, chunk_elems=3000)
        return (out['image'] * wt).sum() + 0.1 * out['depth'].sum() + 0.3 * out['alpha'].sum()
    ps = [p.requires_grad_(True) for p in (means, ls, q, ol, col)]
    f(*ps).backward()
    for p in ps:
        num = torch.zeros_like(p)
        for i in range(p.numel()):
            with torch.no_grad():
                p.view(-1)[i] += 1e-6
                a = f(*ps)
                p.view(-1)[i] -= 2e-6
                b = f(*ps)
                p.view(-1)[i] += 1e-6
            num.view(-1)[i] = (a - b) / 2e-6
        assert torch.allclose(p.grad, num, atol=1e-6, rtol=1e-5), (p.grad - num).abs().max()


def test_spherical_harmonics_gradient_and_view_dependence():
    means, ls, q, ol, _ = _small_scene()
    sh = (torch.randn(len(means), 16, 3, dtype=torch.float64) * 0.2).requires_grad_(True)
    vm = torch.eye(4, dtype=torch.float64)
    K = (30.0, 30.0, 20.0, 14.0)

    def f(sh):
        return R.render(means, torch.exp(ls), q, torch.sigmoid(ol), sh, vm, K, 40, 28, sh_degree=3)['image'].sum()
    f(sh).backward()
    num = torch.zeros_like(sh)
    for i in range(0, sh.numel(), 7):
        with torch.no_grad():
            sh.view(-1)[i] += 1e-6
            a = f(sh)
            sh.view(-1)[i] -= 2e-6
            b = f(sh)
            sh.view(-1)[i] += 1e-6
        num.view(-1)[i] = (a - b) / 2e-6
    sel = torch.zeros(sh.numel(), dtype=torch.bool)
    sel[::7] = True
    assert torch.allclose(sh.grad.view(-1)[sel], num.view(-1)[sel], atol=1e-6)
    # degree 0 is view-independent: SH_C0·dc + ½
    d = torch.nn.functional.normalize(torch.randn(len(means), 3, dtype=torch.float64), dim=-1)
    c0 = R.sh_to_color(sh.detach(), d, 0)
    assert torch.allclose(c0, (R.SH_C0 * sh.detach()[:, 0] + 0.5).clamp_min(0))


def test_projection_and_compositing_follow_the_pinhole_model():
    W, H = 64, 48
    K = (50.0, 52.0, 30.25, 25.5)                 # pixel-edge coordinates
    vm = torch.eye(4, dtype=torch.float64)
    m = torch.tensor([[0.4, -0.3, 5.0]], dtype=torch.float64)
    s = torch.full((1, 3), 0.01, dtype=torch.float64)
    q = torch.tensor([[1.0, 0, 0, 0]], dtype=torch.float64)
    out = R.render(m, s, q, torch.tensor([0.9], dtype=torch.float64), torch.ones(1, 1, dtype=torch.float64), vm, K,
                   W, H)
    img = out['image'][..., 0]
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float64) + 0.5, torch.arange(W, dtype=torch.float64) + 0.5,
                            indexing='ij')
    u, v = (img * xs).sum() / img.sum(), (img * ys).sum() / img.sum()
    assert abs(float(u) - (50.0 * 0.4 / 5.0 + 30.25)) < 0.02 and abs(float(v) - (52.0 * -0.3 / 5.0 + 25.5)) < 0.02
    # two Gaussians on one ray: front to back, C = α₁c₁ + (1 − α₁)α₂c₂; the near one listed second still wins
    m2 = torch.tensor([[0.0, 0.0, 8.0], [0.0, 0.0, 4.0]], dtype=torch.float64)
    s2 = torch.full((2, 3), 0.5, dtype=torch.float64)
    c2 = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    o2 = torch.tensor([0.6, 0.5], dtype=torch.float64)
    q2 = torch.tensor([[1.0, 0, 0, 0]] * 2, dtype=torch.float64)
    out = R.render(m2, s2, q2, o2, c2, vm, (40.0, 40.0, 20.0, 20.0), 40, 40)
    centre = out['image'][19:21, 19:21].mean((0, 1))
    a_near = 0.5 * math.exp(-0.5 * 2 * 0.5 ** 2 / (0.5 ** 2 * 100 + 0.3))   # the pixel centres are ½ px off in x and y
    assert float(centre[0]) > float(centre[1]) > 0
    assert abs(float(centre[0]) - a_near) < 0.01
    assert float(out['alpha'][20, 20]) <= 1.0


def test_ssim_matches_scikit_image():
    from skimage.metrics import structural_similarity
    from falcon_autonomy.core.recon.metrics import psnr, ssim
    rng = np.random.default_rng(0)
    a = rng.random((40, 50, 3))
    b = np.clip(a + rng.normal(0, 0.1, a.shape), 0, 1)
    ref = structural_similarity(a, b, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0,
                                channel_axis=2)
    assert abs(ssim(torch.tensor(a), torch.tensor(b)) - ref) < 1e-6
    assert abs(psnr(torch.tensor(a), torch.tensor(b)) - 10 * math.log10(1 / np.mean((a - b) ** 2))) < 1e-9


# ================================================================================================ Gaussians
def test_ply_layout_round_trip(tmp_path):
    from falcon_autonomy.core.recon.gaussians import Gaussians, ply_properties
    rng = np.random.default_rng(1)
    g = Gaussians.from_points(rng.random((50, 3)), rng.random((50, 3)), sh_degree=3, sky=rng.random(50) > 0.8)
    with torch.no_grad():
        g.p['sh_rest'].copy_(torch.tensor(rng.normal(size=(50, 15, 3)), dtype=torch.float32))
        g.p['quats'].copy_(torch.tensor(rng.normal(size=(50, 4)), dtype=torch.float32))
    p = str(tmp_path / 'point_cloud.ply')
    g.write_ply(p)
    head = open(p, 'rb').read().split(b'end_header\n')[0].decode()
    props = [ln.split()[2] for ln in head.splitlines() if ln.startswith('property float')]
    assert props == ply_properties(3) and len(props) == 62 and 'binary_little_endian' in head
    assert props[:9] == ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2']
    assert props[-8:] == ['opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
    raw = np.frombuffer(open(p, 'rb').read().split(b'end_header\n', 1)[1], '<f4').reshape(50, 62)
    # f_rest is channel-major: red's 15 coefficients, then green's, then blue's
    assert np.allclose(raw[:, 9:24], g.p['sh_rest'].detach().numpy()[:, :, 0])
    assert np.allclose(raw[:, 24:39], g.p['sh_rest'].detach().numpy()[:, :, 1])
    assert np.allclose(raw[:, 54], g.p['opacity_logits'].detach().numpy())
    back = Gaussians.read_ply(p)
    for k, v in g.p.items():
        assert torch.allclose(back.p[k], v.detach()), k
    assert torch.equal(back.sky, g.sky)


def test_density_control_clones_splits_and_prunes():
    from falcon_autonomy.core.recon.gaussians import Adam, Densifier, Gaussians
    xyz = np.array([[0, 0, 5.0], [1, 0, 5.0], [2, 0, 5.0], [3, 0, 5.0]])
    g = Gaussians.from_points(xyz, np.full((4, 3), 0.5), sh_degree=1)
    with torch.no_grad():
        g.p['log_scales'].fill_(math.log(0.05))             # small: cloned
        g.p['log_scales'][1] = math.log(2.0)              # large: split
        g.p['opacity_logits'][3] = -10.0                    # transparent: pruned
    opt = Adam(g, {k: 1e-3 for k in g.p})
    d = Densifier(g, extent=10.0, grad_threshold=0.1, generator=torch.Generator().manual_seed(0))
    grad = torch.tensor([[1.0, 0], [1.0, 0], [0.0, 0], [0.0, 0]])
    d.add_stats(grad, torch.ones(4, dtype=torch.bool), torch.ones(4), 20, 20)
    res = d.step(opt)
    assert res['cloned'] == 1 and res['split'] == 1 and res['pruned'] == 1
    assert len(g) == 4 + 1 + 2 - 1 - 1 and opt.m['means'].shape[0] == len(g)
    big = g.scales().max(-1).values
    assert torch.isclose(big.max(), torch.tensor(2.0 / 1.6), atol=1e-5)


def test_optimisation_recovers_a_known_scene():
    """Gaussians rendered from known cameras are recovered from a perturbed start: positions, colours, opacities."""
    from falcon_autonomy.core.recon.gaussians import Adam, Gaussians
    from falcon_autonomy.core.recon.metrics import psnr
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    gx, gy = np.meshgrid(np.linspace(-1.5, 1.5, 7), np.linspace(-1.0, 1.0, 5))
    xyz = np.c_[gx.ravel(), gy.ravel(), 6.0 + 0.3 * np.sin(gx.ravel())]
    col = rng.random((len(xyz), 3))
    truth = Gaussians.from_points(xyz, col, sh_degree=0, opacity=0.9)
    with torch.no_grad():
        truth.p['log_scales'].fill_(math.log(0.18))
    W, H = 48, 36
    K = (40.0, 40.0, 24.0, 18.0)
    views = []
    for a in np.linspace(-0.25, 0.25, 7):
        vm = np.eye(4)
        c, s = math.cos(a), math.sin(a)
        vm[:3, :3] = [[c, 0, -s], [0, 1, 0], [s, 0, c]]
        vm[:3, 3] = -vm[:3, :3] @ np.array([6.0 * math.sin(a), 0.0, 6.0 - 6.0 * math.cos(a)])
        vt = torch.tensor(vm, dtype=torch.float32)
        with torch.no_grad():
            views.append((vt, truth.render(vt, K, W, H)['image']))
    test = views.pop(3)
    start = Gaussians.from_points(xyz + rng.normal(0, 0.08, xyz.shape), np.full_like(col, 0.5), sh_degree=0,
                                  opacity=0.5)
    with torch.no_grad():
        start.p['log_scales'].fill_(math.log(0.15))
    opt = Adam(start, {'means': 2e-3, 'sh_dc': 2e-2, 'sh_rest': 0.0, 'opacity_logits': 5e-2, 'log_scales': 5e-3,
                       'quats': 1e-3})
    err0 = float(np.linalg.norm(start.p['means'].detach().numpy() - xyz, axis=1).mean())
    with torch.no_grad():
        p0 = psnr(start.render(test[0], K, W, H)['image'].clamp(0, 1), test[1])
    for it in range(400):
        vt, img = views[it % len(views)]
        loss = (start.render(vt, K, W, H)['image'] - img).abs().mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
    with torch.no_grad():
        p1 = psnr(start.render(test[0], K, W, H)['image'].clamp(0, 1), test[1])
    err1 = float(np.linalg.norm(start.p['means'].detach().numpy() - xyz, axis=1).mean())
    assert p1 > 30.0 and p1 > p0 + 12, (p0, p1)
    assert err1 < 0.5 * err0, (err0, err1)
    assert np.abs(torch.sigmoid(start.p['opacity_logits']).detach().numpy() - 0.9).mean() < 0.1


# ================================================================================================ sources
def test_falcon_source_poses_masks_and_lidar(tmp_path):
    from falcon_autonomy.core.recon import data as D
    run = make_falcon_run(str(tmp_path / 'run'), n=12)
    cap = D.open_capture(f'falcon:{run}')
    assert cap.kind == 'falcon' and len(cap.frames) == 12 and cap.meta['carla_exact']
    f = cap.frames[2]
    # a world point on the ground 10 m ahead of the camera lands where the pinhole says
    ego_x = 2 * 0.1
    wpt = np.array([ego_x + 1.5 + 10.0, 0.0, 0.0])                 # right-handed world: (x, −y, z) of CARLA's
    pc = f.viewmat[:3, :3] @ wpt + f.viewmat[:3, 3]
    assert np.allclose(pc, [0.0, 1.6, 10.0], atol=1e-9)
    m = D.load_mask(f)
    inst = __import__('cv2').imread(str(f.image).replace('rgb', 'instance'), -1)
    assert inst.any() and not m[inst == 5].any() and m[inst == 0].all()
    xyz, rgb = cap.lidar()
    assert len(xyz) > 1000 and not np.any((np.abs(xyz[:, 1] + 0.6) < 1e-3) & (np.abs(xyz[:, 2] - 0.9) < 1e-3))
    assert abs(np.median(xyz[:, 2][xyz[:, 2] < 0.1])) < 1e-6            # the ground at z = 0
    assert rgb.min() >= 0 and rgb.max() <= 1


def _write_kitti(root, n=6):
    """A KITTI-raw-shaped drive (calibration, OXTS, images, velodyne) with known numbers — not KITTI data."""
    import cv2
    date = os.path.join(root, '2011_09_26')
    drive = os.path.join(date, '2011_09_26_drive_0001_sync')
    for d in ('image_02/data', 'oxts/data', 'velodyne_points/data'):
        os.makedirs(os.path.join(drive, d), exist_ok=True)
    P2 = np.array([[700.0, 0, 600.0, 45.0], [0, 700.0, 170.0, 0.2], [0, 0, 1, 0.003]])
    Rr = np.eye(3)
    with open(os.path.join(date, 'calib_cam_to_cam.txt'), 'w') as fh:
        fh.write('calib_time: x\nS_rect_02: 1242 375\nR_rect_00: ' + ' '.join(map(str, Rr.ravel())) + '\n'
                 'P_rect_02: ' + ' '.join(map(str, P2.ravel())) + '\n')
    Rv = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0.0]])
    with open(os.path.join(date, 'calib_velo_to_cam.txt'), 'w') as fh:
        fh.write('R: ' + ' '.join(map(str, Rv.ravel())) + '\nT: 0.0 -0.08 -0.27\n')
    with open(os.path.join(date, 'calib_imu_to_velo.txt'), 'w') as fh:
        fh.write('R: 1 0 0 0 1 0 0 0 1\nT: -0.81 0.32 -0.80\n')
    stamps = []
    for k in range(n):
        lat = 49.0 + k * 1e-5                                    # driving north
        vals = [lat, 8.4, 110.0, 0.0, 0.0, math.pi / 2] + [0.0] * 24
        with open(os.path.join(drive, f'oxts/data/{k:010d}.txt'), 'w') as fh:
            fh.write(' '.join(f'{v:.10f}' for v in vals))
        cv2.imwrite(os.path.join(drive, f'image_02/data/{k:010d}.png'), np.full((375, 1242, 3), 90, np.uint8))
        np.array([[10.0, 0, 0, 1], [20, 2, -1, 1]], np.float32).tofile(
            os.path.join(drive, f'velodyne_points/data/{k:010d}.bin'))
        stamps.append(f'2011-09-26 13:02:{25 + k * 0.1:09.6f}000')
    for sub in ('oxts', 'image_02'):
        with open(os.path.join(drive, sub, 'timestamps.txt'), 'w') as fh:
            fh.write('\n'.join(stamps) + '\n')
    return drive, P2, Rv


def test_kitti_raw_source_follows_the_devkit_chain(tmp_path):
    from falcon_autonomy.core.recon import data as D
    drive, P2, Rv = _write_kitti(str(tmp_path))
    cap = D.open_capture(f'kitti_raw:{drive}')
    assert cap.kind == 'kitti_raw' and len(cap.frames) == 6 and cap.lidar is not None
    f = cap.frames[3]
    # a velodyne point of frame 3 projects as P_rect_02 · R_rect_00 · T_velo→cam says (OpenCV pixel centres)
    X = np.array([20.0, 2.0, -1.0, 1.0])
    Tvc = np.eye(4)
    Tvc[:3, :3] = Rv
    Tvc[:3, 3] = [0.0, -0.08, -0.27]
    x = P2 @ (Tvc @ X)
    uv_devkit = x[:2] / x[2]
    Tiv = np.eye(4)
    Tiv[:3, 3] = [-0.81, 0.32, -0.80]
    world = f.ego @ np.linalg.inv(Tiv) @ X
    pc = f.viewmat @ world
    uv = np.array([f.K[0] * pc[0] / pc[2] + f.K[2], f.K[1] * pc[1] / pc[2] + f.K[3]]) - 0.5
    assert np.allclose(uv, uv_devkit, atol=1e-6)
    # driving north: yaw 90° — the camera looks along +y (north) in the east-north-up world
    assert abs(f.viewmat[2, 1] - 1.0) < 1e-6
    assert np.allclose(cap.frames[1].centre - cap.frames[0].centre, [0, 1.112, 0], atol=0.01)


def _write_nuscenes(root):
    """nuScenes-shaped tables (one scene, three key frames of CAM_FRONT and LIDAR_TOP, one annotated car)."""
    import cv2
    v = os.path.join(root, 'v1.0-mini')
    os.makedirs(v, exist_ok=True)
    os.makedirs(os.path.join(root, 'samples/CAM_FRONT'), exist_ok=True)
    os.makedirs(os.path.join(root, 'samples/LIDAR_TOP'), exist_ok=True)
    q_cam = [0.5, -0.5, 0.5, -0.5]                  # nuScenes' front camera: optical axes in the ego frame
    tabs = {'scene': [{'token': 'sc', 'name': 'scene-0001', 'log_token': 'lg', 'nbr_samples': 3}],
            'log': [{'token': 'lg', 'location': 'boston-seaport'}],
            'sensor': [{'token': 's1', 'channel': 'CAM_FRONT'}, {'token': 's2', 'channel': 'LIDAR_TOP'}],
            'calibrated_sensor': [{'token': 'c1', 'sensor_token': 's1', 'translation': [1.7, 0.0, 1.5],
                                   'rotation': q_cam, 'camera_intrinsic': [[800, 0, 799.5], [0, 800, 449.5],
                                                                           [0, 0, 1]]},
                                  {'token': 'c2', 'sensor_token': 's2', 'translation': [0.9, 0.0, 1.8],
                                   'rotation': [1, 0, 0, 0], 'camera_intrinsic': []}],
            'category': [{'token': 'k1', 'name': 'vehicle.car'}], 'instance': [{'token': 'i1', 'category_token': 'k1'}],
            'sample': [], 'sample_data': [], 'ego_pose': [], 'sample_annotation': []}
    for k in range(3):
        ts = 1_500_000_000_000_000 + k * 500_000
        tabs['sample'].append({'token': f's{k}', 'scene_token': 'sc', 'timestamp': ts})
        tabs['ego_pose'].append({'token': f'e{k}', 'translation': [100.0 + 2 * k, 50.0, 0.0],
                                 'rotation': [1, 0, 0, 0]})
        img = f'samples/CAM_FRONT/{k}.jpg'
        cv2.imwrite(os.path.join(root, img), np.full((900, 1600, 3), 120, np.uint8))
        lid = f'samples/LIDAR_TOP/{k}.pcd.bin'
        np.array([[6, 0, -1.8, 1, 0], [15, 3, -1.8, 1, 0], [12, 0, -1.0, 1, 0]], np.float32).tofile(
            os.path.join(root, lid))
        tabs['sample_data'] += [{'token': f'd{k}', 'sample_token': f's{k}', 'calibrated_sensor_token': 'c1',
                                 'ego_pose_token': f'e{k}', 'timestamp': ts, 'filename': img, 'is_key_frame': True,
                                 'width': 1600, 'height': 900},
                                {'token': f'l{k}', 'sample_token': f's{k}', 'calibrated_sensor_token': 'c2',
                                 'ego_pose_token': f'e{k}', 'timestamp': ts, 'filename': lid, 'is_key_frame': True}]
        tabs['sample_annotation'].append({'token': f'a{k}', 'sample_token': f's{k}', 'instance_token': 'i1',
                                          'translation': [100.0 + 2 * k + 12.9, 50.0, 0.8], 'size': [1.9, 4.5, 1.6],
                                          'rotation': [1, 0, 0, 0]})
    for name, rows in tabs.items():
        with open(os.path.join(v, name + '.json'), 'w') as fh:
            json.dump(rows, fh)
    return root


def test_nuscenes_source_poses_masks_and_lidar(tmp_path):
    from falcon_autonomy.core.recon import data as D
    root = _write_nuscenes(str(tmp_path / 'nusc'))
    cap = D.open_capture(f'nuscenes:{root}?scene=scene-0001')
    assert cap.kind == 'nuscenes' and len(cap.frames) == 3 and cap.lidar is not None
    f = cap.frames[1]
    # a point 20 m ahead of the ego (x forward), at the camera's height, lands at the principal point
    world = np.array([102.0 + 1.7 + 20.0, 50.0, 1.5, 1.0])
    pc = f.viewmat @ world
    assert np.allclose(pc[:3], [0, 0, 20.0], atol=1e-9)
    m = D.load_mask(f)
    assert not m[450, 800] and m[50, 50]          # the car ahead is masked; the corner is not
    xyz, _ = cap.lidar()
    # the lidar return inside the car's box (12 m ahead, 1 m below the sensor) is left out
    assert len(xyz) == 6 and np.all(np.abs(xyz[:, 2]) < 1e-6)


# ================================================================================================ training
def _settings(**kw):
    s = {'iterations': 120, 'width': 96, 'device': 'cpu', 'eval_every': 60, 'log_every': 20, 'sky_points': 300,
         'densify_from': 40, 'densify_every': 30, 'max_gaussians': 30000, 'renders': 1, 'voxel_m': 0.3}
    s.update(kw)
    return s


@pytest.fixture(scope='module')
def trained(tmp_path_factory):
    from falcon_autonomy.core.recon.train import train
    root = tmp_path_factory.mktemp('recon')
    run = make_falcon_run(str(root / 'run'), n=24)
    out = str(root / 'recon')
    torch.set_num_threads(2)
    summary = train(f'falcon:{run}', out, _settings())
    return run, out, summary


def test_training_writes_a_reconstruction(trained):
    from falcon_autonomy.core.provenance import read_json
    run, out, summary = trained
    for f in ('point_cloud.ply', 'extras.npz', 'views.json', 'curves.jsonl', 'evaluation.json', 'run.json'):
        assert os.path.isfile(os.path.join(out, f)), f
    rj = read_json(os.path.join(out, 'run.json'))
    assert rj['status'] == 'ok' and rj['summary']['init'] == 'lidar' and rj['summary']['heldout_frames'] == 3
    assert rj['timings']['steps_per_s'] > 0 and rj['device']['type'] == 'cpu'
    assert any(k.endswith('/drive.json') for k in rj['provenance']['inputs']) and rj['summary']['semantic_votes']
    ev = read_json(os.path.join(out, 'evaluation.json'))
    h = ev['heldout']['mean']
    assert h['frames'] == 3 and h['psnr'] > 20.0 and h['ssim'] > 0.6, h
    assert h['depth_median_abs_rel'] < 0.1 and 'LPIPS' not in json.dumps(h) and 'not computed' in ev['lpips']
    curves = [json.loads(x) for x in open(os.path.join(out, 'curves.jsonl'))]
    assert curves[-1]['step'] == 120 and curves[-1]['loss'] < curves[0]['loss']


def test_render_api_and_shifted_cameras(trained):
    from falcon_autonomy.core.recon.scene import Recon, carla_pose_of
    run, out, _ = trained
    rec = Recon.load(out, 'cpu')
    f = rec.frame(10)
    cam = rec.camera(f, lateral_m=1.0)
    base = rec.camera(f)
    # the ego moved 1 m to the right: CARLA's y + 1, the same heading
    pe, pb = carla_pose_of(cam['ego']), carla_pose_of(base['ego'])
    assert abs(pe['y'] - pb['y'] - 1.0) < 1e-9 and abs(pe['x'] - pb['x']) < 1e-9 and abs(pe['yaw']) < 1e-9
    r = rec.render(cam)
    assert r['image'].shape == (54, 96, 3) and r['range'].shape == (54, 96) and 'semantic' in r
    # the ground ahead: range close to the truth for a camera 1.6 m up (the bottom row's ray)
    bottom = r['range'][-3, 40:56]
    ray_z = (54 - 2.5 - 27.0) / 48.0
    truth = 1.6 / ray_z * np.sqrt(1 + ray_z ** 2 + ((np.arange(40, 56) + 0.5 - 48) / 48.0) ** 2)
    assert np.median(np.abs(bottom - truth) / truth) < 0.1
    sem = r['semantic']
    assert (sem[-5:] == 1).mean() > 0.8              # road at the bottom
    # a new mount and field of view
    c2 = rec.camera(f, mount={'x': 1.5, 'y': 0.0, 'z': 2.5, 'roll': 0, 'pitch': -10.0, 'yaw': 0}, fov=70, size=(80, 60))
    assert c2['size'] == (80, 60) and abs(c2['K'][0] - 40 / math.tan(math.radians(35))) < 1e-9
    assert abs(carla_pose_of(c2['mount'], optical=True)['pitch'] + 10.0) < 1e-6
    # a camera at a world pose (CARLA's convention) is the frame's own camera when the pose is the frame's camera's
    world_cam = carla_pose_of(base['ego'] @ base['mount'], optical=True)
    c3 = rec.camera_at(world_cam, fov=90.0)
    assert np.allclose(c3['viewmat'], base['viewmat'], atol=1e-6) and np.allclose(c3['K'], base['K'])


def test_warp_reference_reproduces_the_frame_and_its_shifts(trained):
    from falcon_autonomy.core.recon import evaluate as EV
    from falcon_autonomy.core.recon.scene import Recon
    run, out, _ = trained
    rec = Recon.load(out, 'cpu')
    src = rec.capture().frames[8]
    # to its own camera: the image itself wherever the warp defines it
    cam = {'viewmat': src.viewmat, 'K': src.K, 'size': src.size}
    ref, valid, rref = EV.warp(src, cam)
    img = __import__('falcon_autonomy.core.recon.data', fromlist=['x']).load_rgb(src.image) / 255.0
    assert valid.mean() > 0.9 and np.abs(ref - img)[valid].mean() < 0.01
    # the warp's own error against a real frame further along the path
    chk = EV.warp_check(rec.capture(), rec.work_size, (0.5,), limit=4)
    assert chk[0]['frames'] and chk[0]['psnr'] > 20 and chk[0]['coverage'] > 0.6
    lat = EV.lateral_shift(rec, (0.5,), limit=2)
    shifts = {s['lateral_m']: s for s in lat['shifts']}
    assert set(shifts) == {0.0, 0.5, -0.5} and shifts[0.0]['psnr'] >= shifts[0.5]['psnr'] - 3


def test_against_another_run_beside_the_path(trained, tmp_path):
    from falcon_autonomy.core.recon import evaluate as EV
    from falcon_autonomy.core.recon.scene import Recon
    run, out, _ = trained
    other = make_falcon_run(str(tmp_path / 'beside'), n=24, actor=False, lidar=False, lateral=0.7)
    rec = Recon.load(out, 'cpu')
    res = EV.against_run(rec, f'falcon:{other}', limit=6)
    assert res['frames'] and all(abs(r['lateral_m'] - 0.7) < 1e-6 for r in res['frames'])
    assert res['by_lateral'][0]['lateral_m'] == [0.25, 0.75] and res['mean']['psnr'] > 15
    with pytest.raises(ValueError):
        bad = make_falcon_run(str(tmp_path / 'elsewhere'), n=4, actor=False, lidar=False)
        dj = os.path.join(bad, 'drive.json')
        d = json.load(open(dj))
        d['town'] = 'Town01'
        json.dump(d, open(dj, 'w'))
        EV.against_run(rec, f'falcon:{bad}')


def test_rerendered_run_is_a_falcon_raw_run(trained, tmp_path):
    from falcon_autonomy.core.exporters import Run
    from falcon_autonomy.core.recon.rerender import rerender
    run, out, _ = trained
    res = rerender(out, str(tmp_path / 'shifted'), lateral_m=1.0, every=4, device='cpu')
    r = Run(res['run'])
    assert len(r.frames) == 6 and r.cams['cam-front']['width'] == 96
    assert set(res['labels']) == {'rgb', 'range', 'semantic'} and 'objects' in res['not_available']
    steps = r.camera_steps('cam-front')
    assert len(steps) == 6 and r.file(steps[0], 'cam-front', 'semantic')
    src0 = json.loads(open(os.path.join(run, 'frames.jsonl')).readline())
    assert abs(r.frames[0]['ego']['y'] - (src0['ego']['y'] + 1.0)) < 1e-6
    assert r.run['status'] == 'ok' and 'not ground truth' in json.dumps(r.ds['labels'])
    # a segmenter can be scored on it by the harness: the semantic labels are the reconstruction's
    from falcon_autonomy.core import harness as HZ
    import cv2

    class Seg:
        providers = ['test']

        def predict(self, rgb, meta):
            return np.full(rgb.shape[:2], 1, np.uint8)
    col = HZ.collect([res['run']], {'kind': 'segmenter', 'runtime': 'python', 'falcon_classes': ['road']},
                     runner=Seg())
    assert len(col['frames']) == 6
    sc = HZ.score(col['frames'], 'segmenter', n_boot=20)
    assert 0.0 < sc['overall']['mIoU'] <= 1.0 if 'overall' in sc else True
    assert cv2.imread(os.path.join(res['run'], 'range', 'cam-front', '000000.png'), -1).dtype == np.uint16


def test_cli_train_info_render_and_eval(tmp_path, capsys):
    from falcon_autonomy.cli import main
    run = make_falcon_run(str(tmp_path / 'run'), n=12, actor=False)
    out = str(tmp_path / 'r')
    assert main(['recon', 'train', '--source', f'falcon:{run}', '--out', out, '--iterations', '30', '--width', '64',
                 '--device', 'cpu', '--sky-points', '100', '--no-semantic']) == 0
    assert main(['recon', 'info', out]) == 0
    assert '"status": "ok"' in capsys.readouterr().out
    png = str(tmp_path / 'view.png')
    assert main(['recon', 'render', out, '--frame', '3', '--lateral', '0.5', '--out', png, '--depth-out',
                 str(tmp_path / 'd.png'), '--device', 'cpu']) == 0
    assert __import__('cv2').imread(png).shape == (36, 64, 3)
    assert main(['recon', 'eval', out, '--warp-shift', '0.5', '--warp-check', '0.3', '--limit', '2',
                 '--device', 'cpu']) == 0
    ev = json.load(open(os.path.join(out, 'evaluation.json')))
    assert ev['novel_view']['lateral_warped']['shifts'] and ev['novel_view']['warp_checks']
    assert main(['recon', 'train', '--source', str(tmp_path / 'nothing'), '--out', out]) == 2


def test_training_is_repeatable_on_the_cpu(tmp_path):
    from falcon_autonomy.core.recon.gaussians import Gaussians
    from falcon_autonomy.core.recon.train import train
    run = make_falcon_run(str(tmp_path / 'run'), n=8, actor=False)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)                     # several threads accumulate in another order: not bit for bit
    try:
        s = _settings(iterations=40, width=64, densify_from=10, densify_every=10, sky_points=100, semantic=False)
        train(f'falcon:{run}', str(tmp_path / 'a'), s)
        train(f'falcon:{run}', str(tmp_path / 'b'), s)
    finally:
        torch.set_num_threads(threads)
    a = Gaussians.read_ply(str(tmp_path / 'a' / 'point_cloud.ply'))
    b = Gaussians.read_ply(str(tmp_path / 'b' / 'point_cloud.ply'))
    assert len(a) == len(b) and all(torch.equal(a.p[k], b.p[k]) for k in a.p)


@pytest.mark.skipif(os.environ.get('FALCON_GPU_TESTS') != '1', reason='GPU tests need FALCON_GPU_TESTS=1')
def test_gpu_rasteriser_matches_the_cpu():
    means, ls, q, ol, col = _small_scene(torch.float32)
    vm = torch.eye(4)
    a = R.render(means, torch.exp(ls), q, torch.sigmoid(ol), col, vm, (30.0, 30.0, 20.0, 14.0), 40, 28)
    b = R.render(*(t.cuda() for t in (means, torch.exp(ls), q, torch.sigmoid(ol), col, vm)), (30.0, 30.0, 20.0, 14.0),
                 40, 28)
    assert torch.allclose(a['image'], b['image'].cpu(), atol=1e-5)


# ================================================================================================ the Replay page
from test_gui import pump, qapp, slot_exceptions, window  # noqa: E402,F401


@pytest.mark.gui
def test_replay_page_reconstruction_section(window, qapp, tmp_path):
    from falcon_autonomy.core.jobs import DONE, FINAL
    from falcon_autonomy.core.logreplay import import_log
    from falcon_autonomy.ui.pages.replay import ReplayPage
    page = ReplayPage(window)
    window.stack.addWidget(page)
    window.stack.setCurrentWidget(page)
    run = make_falcon_run(str(tmp_path / 'src'), n=12, actor=False)
    os.makedirs(page.logs_dir(), exist_ok=True)
    import_log('falcon', run).save(os.path.join(page.logs_dir(), 'synthetic'))
    page.refresh()
    page.logs.setCurrentIndex(page.logs.findText('synthetic'))
    panel = page.recon
    assert panel.source == f'falcon:{run}' and panel.train_btn.isEnabled()
    assert panel.info.vals['Reconstruction'].text() == 'none yet'
    panel.iters.setValue(30)
    panel.width_px.setValue(64)
    panel.device.setCurrentIndex(panel.device.findData('cpu'))
    panel.train()
    job = panel.jobs['train']
    assert job.spec.needs_gpu and job.spec.argv[job.spec.argv.index('--device') + 1] == 'cpu'
    assert pump(qapp, lambda: job.state in FINAL, 180), job.summary()
    assert job.state == DONE, job.log[-5:]
    assert pump(qapp, lambda: panel.last is not None, 60)
    assert panel.last['image'].shape == (36, 64, 3)
    assert 'Gaussians' in panel.info.vals['Reconstruction'].text() and 'PSNR' in panel.info.vals['Held out'].text()
    panel.frame.setValue(5)
    panel.lateral.setValue(1.0)
    panel.lateral.valueChanged.emit(1.0)
    assert pump(qapp, lambda: panel.last['lateral_m'] == 1.0 and panel.last['frame'] == 5, 60)
    out = panel.rerender(str(tmp_path / 'shifted'))
    rj = panel.jobs['rerender']
    assert pump(qapp, lambda: rj.state in FINAL, 120) and rj.state == DONE, rj.log[-5:]
    assert os.path.isfile(os.path.join(out, 'frames.jsonl')) and os.path.isfile(os.path.join(out, 'run.json'))
    # a log whose images Falcon cannot read (a ROS 2 bag's) says so
    from falcon_autonomy.core.logreplay import Log, Track
    t = np.arange(10) * 0.1
    Log(Track('ego', 'car', t, t, 0 * t, 0 * t, 0 * t), [], {'name': 'bag', 'source': {'format': 'ros2',
                                                                                      'path': run}}).save(
        os.path.join(page.logs_dir(), 'bag'))
    page.refresh()
    page.logs.setCurrentIndex(page.logs.findText('bag'))
    assert not panel.train_btn.isEnabled() and 'trajectories only' in panel.info.vals['Source'].text()
