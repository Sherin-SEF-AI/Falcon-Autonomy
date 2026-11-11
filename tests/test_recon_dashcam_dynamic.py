"""Dynamic reconstruction from a dashcam's near-miss tracks (core.recon.dashcam_tracks; `falcon dashcam nearmiss
reconstruct`, `falcon recon from-dashcam --dynamic`): the ingest's tracked road users carried into a reconstruction's
world and learned as moving actors, a policy driven through the incident in closed loop, and everything made from the
footage private (assistant.privacy, cluster bundles).

The coordinate chain is checked exactly on a synthetic perception record whose reconstruction world is a rigid
transform of the ingest's (rotated, shifted, a pitched camera, posed frames out of phase with the perception's).
End to end: a synthetic dashcam clip (the textured street of test_dashcam_recon, ray-cast, with a lead car closing in
the ego's lane and a truck in the next lane drawn as textured boxes; a GPS index of 1 Hz fixes from the true track),
ingested with a ground-truth detector sidecar and flat-ground ranges, its top event reconstructed on the CPU (a few
training steps) and driven through. No real footage, no network, no model download."""
import json
import math
import os
import shutil

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')
torch = pytest.importorskip('torch')
pytest.importorskip('scipy')

from falcon_autonomy.core.recon import dashcam_tracks as DTK  # noqa: E402

from test_dashcam_recon import LAT0, LON0, Street, trajectory  # noqa: E402

CAM_OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])   # vehicle ← optical


def _rz(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _T(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


# ================================================================================================ the chain, exactly
def _ego_nm(t):
    """The ingest's ego (its world): straight at 8 m/s, heading 20°."""
    yaw = math.radians(20.0)
    return 8.0 * t * math.cos(yaw), 8.0 * t * math.sin(yaw), math.degrees(yaw)


def _car_nm(t):
    """A car's centre in the ingest's world: ahead in the ego's lane, slower (5 m/s), 1 m to its left."""
    yaw = math.radians(20.0)
    along, left = 20.0 + 5.0 * t, 1.0
    return np.array([along * math.cos(yaw) - left * math.sin(yaw), along * math.sin(yaw) + left * math.cos(yaw)])


def test_the_chain_places_the_track_where_the_reconstruction_saw_it():
    """C_recon · E⁻¹ · A: with the reconstruction's world a rigid transform G of the ingest's (and its camera pitched
    3°, posed frames at other instants), each box lands at G · its ingest pose; sizes are the kind's, times on the
    posed frames' clock."""
    from falcon_autonomy.core import nearmiss as NM
    ts = 10.0 + np.arange(0, 3.0, 0.1)                      # video seconds of the perception's frames
    frames, ego = [], {k: [] for k in ('t', 'x', 'y', 'yaw', 'vx', 'vy', 'speed', 'accel')}
    for t in ts:
        x, y, yaw = _ego_nm(t)
        E = NM.ego_pose({'x': [x], 'y': [y], 'yaw': [yaw]}, 0)
        c = np.linalg.inv(E) @ np.r_[_car_nm(t), 0.0, 1.0]   # the centre in the camera's levelled frame
        u = c[:2] / np.linalg.norm(c[:2])
        near = c[:2] - u * 4.6 / 2.0                          # what the ingest measured: the near face
        frames.append({'n': len(frames), 't': float(t), 'boxes': [
            {'track': 5, 'forward': float(near[0]), 'left': float(near[1]), 'sigma': 0.05, 'kind': 'car',
             'class_source': 'model', 'box': [0, 0, 1, 1]}]})
        for k, v in (('t', t), ('x', x), ('y', y), ('yaw', yaw), ('vx', 8 * math.cos(math.radians(yaw))),
                     ('vy', 8 * math.sin(math.radians(yaw))), ('speed', 8.0), ('accel', 0.0)):
            ego[k].append(v)
    perc = {'created': 'x', 'frames': frames}
    mined = {'perception_created': 'x', 'ego_track': dict(ego, ok=[True] * len(ts))}
    G = _T(_rz(math.radians(-35.0)), [120.0, -40.0, 0.0])     # the reconstruction's world ← the ingest's
    pitch = math.radians(3.0)
    Rp = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]])
    posed = []
    for j, vt in enumerate(10.05 + np.arange(0, 2.9, 1 / 3)):   # 3 a second, out of phase with the perception
        x, y, yaw = _ego_nm(vt)
        Tw = G @ NM.ego_pose({'x': [x], 'y': [y], 'yaw': [yaw]}, 0) @ _T(CAM_OPTICAL @ Rp, [0.0, 0.0, 1.4])
        posed.append({'video_t': float(vt), 't': float(vt - 10.05), 'viewmat': np.linalg.inv(Tw).reshape(-1).tolist()})
    tracks, rep = DTK.recon_tracks(posed, perc, mined)
    assert set(tracks) == {5} and rep['tracks'] == 1 and rep['kinds'] == {'car': 1} and 'chain' in rep
    tr = tracks[5]
    assert tr.size == pytest.approx((4.6, 2.0, 1.5)) and tr.kind == 'car' and tr.dynamic
    assert tr.category == 'vehicle.car' and tr.meta['observations'] == len(ts)
    # times on the posed clock, every box where G puts the ingest's pose, heading 20° − 35°
    inside = [t for t in ts if 10.05 - DTK.PAD_S <= t <= posed[-1]['video_t'] + DTK.PAD_S]
    assert len(tr.t) == len(inside) and tr.t[0] == pytest.approx(inside[0] - 10.05, abs=1e-9)
    errs, yerr = [], []
    for t in inside[2:-2]:                                    # the smoother's ends aside
        R, c = tr.pose(t - 10.05)
        want = G @ np.r_[_car_nm(t), 0.75, 1.0]
        errs.append(np.linalg.norm(c - want[:3]))
        yerr.append(abs(math.degrees(math.atan2(R[1, 0], R[0, 0])) - (20.0 - 35.0)))
    assert max(errs) < 0.05 and max(yerr) < 0.5, (max(errs), max(yerr))
    # a track outside the segment, or seen too rarely, is left out and counted
    perc['frames'][0]['boxes'].append({'track': 9, 'forward': 9.0, 'left': 0.0, 'sigma': 0.1, 'kind': 'bus',
                                       'class_source': 'model', 'box': [0, 0, 1, 1]})
    _, rep2 = DTK.recon_tracks(posed, perc, mined)
    assert rep2['left_out']['too_few_observations'] == 1
    assert DTK.size_of('bus') == (10.3, 3.9, 3.2) and DTK.size_of('pedestrian')[2] == 1.7


def _posed_folder(root, with_tracks=True, masks=True):
    """A posed folder of 4 frames looking along +x from the origin (64 × 48, f 40), a red box 8 m ahead (tracked),
    a grey background, all-usable masks."""
    from falcon_autonomy.core.recon import data as D
    from falcon_autonomy.core.recon.actors import Track, rot_to_quat_np
    os.makedirs(os.path.join(root, 'images'), exist_ok=True)
    os.makedirs(os.path.join(root, 'masks'), exist_ok=True)
    K = (40.0, 40.0, 32.0, 24.0)
    frames = []
    for k in range(4):
        Tw = _T(CAM_OPTICAL, [0.2 * k, 0.0, 1.4])
        vm = np.linalg.inv(Tw)
        img = np.full((48, 64, 3), 128, np.uint8)
        corners = D.box_corners(_T(np.eye(3), [8.0, 0.0, 0.75]), (2.0, 4.6, 1.5))
        uv = D.project_box_hull(corners, vm, K)
        x0, y0 = np.floor(uv.min(0)).astype(int)
        x1, y1 = np.ceil(uv.max(0)).astype(int)
        img[max(y0, 0):y1, max(x0, 0):x1] = (220, 30, 30)
        cv2.imwrite(os.path.join(root, f'images/{k}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if masks:
            cv2.imwrite(os.path.join(root, f'masks/{k}.png'), np.full((48, 64), 255, np.uint8))
        frames.append({'index': k, 't': 0.1 * k, 'image': f'images/{k}.png', 'mask': f'masks/{k}.png' if masks
                       else None, 'K': list(K), 'size': [64, 48], 'viewmat': vm.reshape(-1).tolist(),
                       'ego': _T(np.eye(3), [0.2 * k, 0, 0]).reshape(-1).tolist(),
                       'mount': _T(CAM_OPTICAL, [0, 0, 1.4]).reshape(-1).tolist()})
    doc = {'schema': D.POSED_SCHEMA, 'name': 'synthetic', 'camera': 'cam', 'frames': frames, 'masks': 'test masks'}
    if with_tracks:
        q = rot_to_quat_np(np.eye(3))[0]
        tr = Track(3, 'car', 'vehicle.car', True, (4.6, 2.0, 1.5), np.array([0.0, 0.1, 0.2, 0.3]),
                   np.array([[8.0, 0.0, 0.75]] * 4), np.tile(q, (4, 1)))
        doc['tracks'] = DTK.write(root, {3: tr}, {'tracks': 1})
    with open(os.path.join(root, 'posed.json'), 'w') as fh:
        json.dump(doc, fh)
    return root


def test_a_posed_folder_with_tracks_learns_or_masks_them(tmp_path):
    from falcon_autonomy.core.recon import data as D
    p = _posed_folder(str(tmp_path / 'posed'))
    dyn = D.open_capture(f'posed:{p}?dynamic=1')
    assert set(dyn.tracks) == {3} and dyn.meta['dynamic'] and 'learned' in dyn.meta['masks']
    st = D.open_capture(f'posed:{p}')
    assert st.tracks is None and 'tracked road users' in st.meta['masks']
    m_st, m_dyn = D.load_mask(st.frames[0]), D.load_mask(dyn.frames[0])
    img = D.load_rgb(st.frames[0].image)
    red = (img[..., 0] > 200) & (img[..., 1] < 60)
    assert red.sum() > 50
    assert not m_st[red].any() and m_st[~red].mean() > 0.6     # static mode: the box (dilated) masked
    assert m_dyn.all()                                         # dynamic mode: its pixels learned
    # no lidar: the actor starts from its box's surface coloured by the frames that see it (red where seen)
    P, rgb = D.actor_points(dyn, n_surface=300)[3]
    assert len(P) == 300 and np.all(np.abs(P) <= np.array([4.6, 2.0, 1.5]) / 2 + 1e-9)
    seen = np.any(np.abs(rgb - 0.5) > 1e-6, axis=1)
    assert seen.mean() > 0.3 and np.median(rgb[seen][:, 0]) > 0.75
    # a folder without tracks has no dynamic mode
    q = _posed_folder(str(tmp_path / 'plain'), with_tracks=False)
    with pytest.raises(D.DataError, match='brings no tracks'):
        D.open_capture(f'posed:{q}?dynamic=1')
    with pytest.raises(D.DataError, match='KITTI'):
        D.open_capture(('kitti_raw', str(tmp_path)), dynamic='1')


# ================================================================================================ a synthetic clip
W, H, FPS, N = 480, 270, 10, 80
NORTH = -1.75
EGO_V = 6.0
ACTORS = [  # id, COCO label, (L, W, H), lane (north, m), x at t = 0 (east, m), speed (m/s)
    ('lead', 'car', (4.6, 2.0, 1.5), NORTH, -2.0, 4.5),
    ('truck', 'truck', (6.0, 2.6, 3.0), NORTH + 3.5, -8.0, 7.0),
]


def actor_centre(a, t):
    _n, _l, (L, Wd, Hh), north, x0, v = a
    return np.array([x0 + v * t, north, Hh / 2.0])


def _box_hits(a, t, c, d):
    """Ray parameters (N,) where rays c + s·d hit the actor's (axis-aligned) box, inf elsewhere; and the hit's
    local coordinates for a texture."""
    _n, _l, (L, Wd, Hh), *_ = a
    ctr = actor_centre(a, t)
    lo, hi = ctr - [L / 2, Wd / 2, Hh / 2], ctr + [L / 2, Wd / 2, Hh / 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        t1, t2 = (lo - c) / d, (hi - c) / d
    tn = np.nanmax(np.minimum(t1, t2), axis=1)
    tf = np.nanmin(np.maximum(t1, t2), axis=1)
    hit = (tn <= tf) & (tf > 0)
    s = np.where(hit, np.maximum(tn, 0.0), np.inf)
    p = c + d * np.where(np.isfinite(s), s, 0.0)[:, None]
    return s, p - ctr


def render_frame(st, intr, R, c, t, ss=2):
    """The street (test_dashcam_recon.Street) with the actors' boxes over it: a box is always nearer than the street
    behind it (it stands on the road between the walls)."""
    from falcon_autonomy.core.recon import vo as V
    img = st.render(intr, R, c, supersample=ss).astype(np.float32)
    big = V.Intrinsics(intr.width * ss, intr.height * ss, intr.f * ss, 0.0, 0.0)
    u, v = np.meshgrid(np.arange(big.width, dtype=np.float64), np.arange(big.height, dtype=np.float64))
    d = np.c_[big.normalise(np.c_[u.ravel(), v.ravel()]), np.ones(u.size)] @ np.asarray(R).T
    best = np.full(len(d), np.inf)
    val = np.zeros(len(d), np.float32)
    for k, a in enumerate(ACTORS):
        s, loc = _box_hits(a, t, c, d)
        near = s < best
        stripes = ((np.floor(loc[:, 0] * 2.5) + np.floor(loc[:, 2] * 2.5) + np.floor(loc[:, 1] * 2.5)) % 2)
        val[near] = (40 + 150 * stripes[near] + 40 * k)
        best[near] = s[near]
    cover = cv2.resize(np.isfinite(best).reshape(big.height, big.width).astype(np.float32), (intr.width,
                       intr.height), interpolation=cv2.INTER_AREA)
    boxv = cv2.resize(np.where(np.isfinite(best), val, 0).reshape(big.height, big.width), (intr.width, intr.height),
                      interpolation=cv2.INTER_AREA)
    out = img * (1 - cover) + boxv
    return np.clip(out, 0, 255).astype(np.uint8)


def gt_box(a, t, intr, R, c):
    """The actor's bounding box in the image (pixel-edge, core.dashcam_depth's convention), or None."""
    _n, _l, (L, Wd, Hh), *_ = a
    ctr = actor_centre(a, t)
    P = np.array([ctr + [sx * L / 2, sy * Wd / 2, sz * Hh / 2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    pc = (P - c) @ np.asarray(R)
    if (pc[:, 2] <= 0.5).any():
        return None
    u = intr.f * pc[:, 0] / pc[:, 2] + intr.cx + 0.5
    v = intr.f * pc[:, 1] / pc[:, 2] + intr.cy + 0.5
    b = [max(0.0, u.min()), max(0.0, v.min()), min(W - 1.0, u.max()), min(H - 1.0, v.max())]
    return b if b[2] - b[0] > 2 and b[3] - b[1] > 2 else None


def make_clip(folder):
    """The clip (as test_dashcam_recon.make_dashcam_clip, with the actors), its GPS index and the detector's
    ground-truth sidecar. Returns (clip, sidecar, true camera centres, intrinsics)."""
    import datetime as dt
    import subprocess
    from falcon_autonomy.core import dashcam as DC
    from falcon_autonomy.core.osm import local_projection
    from falcon_autonomy.core.recon import vo as V
    intr = V.Intrinsics.from_hfov(W, H, 100.0)
    st = Street(seed=11, half_width=9.0)
    ts, pos, rots = trajectory(n=N, fps=FPS, speed=EGO_V, weave=4.0)
    pos[:, 0] -= 20.0
    pos[:, 1] += NORTH
    os.makedirs(folder, exist_ok=True)
    utc0 = dt.datetime(2026, 3, 21, 6, 30, 0, tzinfo=dt.timezone.utc)
    path = os.path.join(folder, (utc0 + dt.timedelta(hours=5, minutes=30)).strftime('%Y%m%d%H%M%S') + '_000001F.MP4')
    p = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray',
                          '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-g', '5', '-bf', '0',
                          '-crf', '12', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', path], stdin=subprocess.PIPE)
    side = {}
    for n in range(N):
        t = ts[n]
        p.stdin.write(render_frame(st, intr, rots[n], pos[n], t).tobytes())
        side[str(n)] = [{'box': b, 'score': 0.9, 'label': a[1], 'truth': a[0]}
                        for a in ACTORS for b in [gt_box(a, t, intr, rots[n], pos[n])] if b is not None]
    p.stdin.close()
    assert p.wait() == 0
    proj = local_projection(LAT0, LON0)
    blocks = []
    for k in range(int(N / FPS)):
        j = min(int(round(k * FPS)), N - 1)
        dd = pos[min(j + 1, N - 1)] - pos[max(j - 1, 0)]
        course = (90.0 - math.degrees(math.atan2(dd[1], dd[0]))) % 360
        lat, lon = proj.to_geo(pos[j, 0], pos[j, 1])
        blocks.append(DC.encode_block(utc0 + dt.timedelta(seconds=k), lat, lon, EGO_V / 0.514444, course))
    DC.add_gps(path, blocks)
    sp = os.path.join(os.path.dirname(os.path.dirname(folder)), 'sidecar.json')
    with open(sp, 'w') as fh:
        json.dump({'frames': side}, fh)
    return path, sp, pos, intr


def _cfg(tmp, intr):
    from falcon_autonomy.core.recon import fromdashcam as F
    nat = intr.scaled(*F.NATIVE)
    p = os.path.join(tmp, 'recon-cfg.json')
    with open(p, 'w') as fh:
        json.dump({'image_size': [W, H], 'vo_width': W, 'out_width': 320, 'image_every': 2,
                   'intrinsics': f'f={nat.f},k1=0', 'novel_frames': 1, 'novel_lateral_m': [1.0], 'novel_up_m': [0.5]},
                  fh)
    return p


@pytest.fixture(scope='module')
def incident(tmp_path_factory):
    """The clip ingested (a ground-truth detector, flat-ground ranges with the mount's true horizon), its top event
    reconstructed with `falcon dashcam nearmiss reconstruct … --device cpu` (30 steps at 96 px)."""
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('needs ffmpeg')
    from falcon_autonomy.core import dashcam_ingest as I
    from falcon_autonomy.core.dashcam_detect import FileDetector
    torch.set_num_threads(4)
    root = str(tmp_path_factory.mktemp('nmrecon'))
    folder = os.path.join(root, 'footage')
    clip, side, pos, intr = make_clip(os.path.join(folder, 'Movie_F'))
    s = {'detector': f'file:{side}', 'depth': 'none', 'rate': FPS, 'size': [W, H], 'focal_px': intr.f,
         'camera_height_m': 1.4, 'horizon_v': H / 2.0 - intr.f * math.tan(math.radians(3.0)), 'batch': 4,
         'vo': False}
    summ = I.ingest(folder, s, detector=FileDetector(side))
    out = os.path.join(root, 'recon')
    rc = I.main_ingest(['nearmiss', 'reconstruct', folder, '1', '--out', out, '--config', _cfg(root, intr),
                        '--device', 'cpu', '--iterations', '30', '--width', '96', '--seed', '0'])
    return {'root': root, 'folder': folder, 'clip': clip, 'pos': pos, 'intr': intr, 'summary': summ, 'out': out,
            'rc': rc}


def test_nearmiss_reconstruct_learns_the_tracked_road_users(incident):
    from falcon_autonomy.core.osm import local_projection
    from falcon_autonomy.core.provenance import read_json
    from falcon_autonomy.core.recon import data as D
    from falcon_autonomy.core.recon.scene import Recon
    assert incident['summary']['events'] >= 1 and incident['rc'] == 0
    out = incident['out']
    rep = read_json(os.path.join(out, 'dashcam.json'))
    ru = rep['road_users']
    assert ru['tracks'] == 2 and ru['kinds'] == {'car': 1, 'truck': 1} and ru['moving'] == 2, ru
    assert ru['sizes'] == {'car': [4.6, 2.0, 1.5], 'truck': [6.0, 2.6, 3.0]}
    assert rep['summary']['road_users']['tracks'] == 2 and rep['summary']['recon']['actors'] == 2
    cap = D.open_capture(f'posed:{out}/posed?dynamic=1')
    assert len(cap.tracks) == 2
    # every box where the clip drew it: the reconstruction's world is east-north-up about its first fix
    e0, n0 = local_projection(LAT0, LON0).to_local(rep['geo']['lat0'], rep['geo']['lon0'])
    posed = read_json(os.path.join(out, 'posed', 'posed.json'))['frames']
    clock0 = posed[0]['video_t'] - posed[0]['t']               # the clip's time of the posed clock's zero
    by_kind = {}
    for tr in cap.tracks.values():
        truth = next(a for a in ACTORS if a[1] == tr.cls)
        assert tr.size == pytest.approx(truth[2])
        errs, yerr = [], []
        for t in tr.t[2:-2]:
            R, c = tr.pose(float(t))
            want = actor_centre(truth, t + clock0) - [e0, n0, 0.0]
            errs.append(float(np.linalg.norm(c[:2] - want[:2])))
            yerr.append(abs((math.degrees(math.atan2(R[1, 0], R[0, 0])) + 180) % 360 - 180))
            assert c[2] == pytest.approx(truth[2][2] / 2)
        by_kind[tr.cls] = (float(np.median(errs)), float(np.median(yerr)), len(errs))
    for k, (e, y, n) in by_kind.items():          # measured: car 0.73 m, 1.5°; truck 0.45 m, 1.9° (63 boxes each)
        assert n >= 10 and e < 1.2 and y < 6.0, by_kind
    rec = Recon.load(out, 'cpu')
    assert len(rec.actors) == 2 and rec.views.get('dynamic')
    aj = read_json(os.path.join(out, 'actors.json'))
    assert sorted(a['kind'] for a in aj['actors']) == ['car', 'car']        # a truck is a four-wheeled actor
    assert all(a['wheels'] == 4 for a in aj['actors'])
    img = rec.render(rec.camera(rec.frame(len(rec.frames) // 2)))
    assert img['image'].shape[2] == 3 and (img['actor'] >= 0).any()        # the actors are drawn in a view
    assert rep['heldout_by_horizon'].get('below') and len(rep['novel_views']['files']) == 3


def test_a_policy_drives_through_the_incident_in_closed_loop(incident, tmp_path):
    from falcon_autonomy.core.recon.closedloop import run_closedloop
    cl = str(tmp_path / 'cl')
    res = run_closedloop(incident['out'], cl, 'constant', episodes=1, seed=1, obs_size=(48, 27), device='cpu')
    assert len(res['episodes']) == 1
    ep = res['episodes'][0]
    for f in ('frames.jsonl', 'actors.jsonl', 'drive.json', 'run.json', 'closedloop.json'):
        assert os.path.isfile(os.path.join(ep, f)), f
    rows = [json.loads(x) for x in open(os.path.join(ep, 'actors.jsonl'))]
    seen = {a['id'] for r in rows for a in r['actors'] if a['role'] != 'hero'}
    assert len(seen) == 2 and len(rows) > 5                                 # the incident's road users replayed
    lead = [a for r in rows for a in r['actors'] if a['role'] != 'hero' and a['extent'][0] == pytest.approx(2.3)]
    assert lead
    from falcon_autonomy.core import rules
    assert rules.check_run(ep, xodr='')['rules']


def test_the_reconstruction_and_every_drive_through_it_stay_private(incident, tmp_path):
    from falcon_autonomy.assistant import privacy
    from falcon_autonomy.assistant.tool_api import ToolError
    from falcon_autonomy.cluster import bundle as B
    from falcon_autonomy.core.jobs import JobSpec
    from falcon_autonomy.core.recon.closedloop import run_closedloop
    from falcon_autonomy.core.recon.rerender import rerender
    out = incident['out']
    for p in (out, os.path.join(out, 'posed', 'tracks.json'), os.path.join(out, 'actors.pt'),
              os.path.join(out, 'renders')):
        assert privacy.private_reason(p), p
    cl = str(tmp_path / 'runs' / 'neural')
    res = run_closedloop(out, cl, 'constant', episodes=1, seed=2, obs_size=(48, 27), device='cpu')
    ep = res['episodes'][0]
    img = next(os.path.join(dp, f) for dp, _, fs in os.walk(os.path.join(ep, 'rgb')) for f in fs)
    for p in (cl, ep, img, os.path.join(ep, 'frames.jsonl')):
        assert privacy.private_reason(p), p
        with pytest.raises(ToolError, match='private'):
            privacy.guard(p)
    for f in ('drive.json', 'run.json', 'dataset.json', 'closedloop.json'):
        assert json.load(open(os.path.join(ep, f)))['source']['format'] == 'dashcam', f   # survives the recon
    assert json.load(open(os.path.join(cl, 'closedloop.json')))['source']['format'] == 'dashcam'
    rr = str(tmp_path / 'runs' / 'shifted')
    rerender(out, rr, lateral_m=0.5, every=4, device='cpu')
    assert json.load(open(os.path.join(rr, 'drive.json')))['source']['format'] == 'dashcam'
    assert privacy.private_reason(rr)
    # an older run without the marker is private while its reconstruction is (the record names it) …
    old = tmp_path / 'old-run'
    old.mkdir()
    (old / 'drive.json').write_text(json.dumps({'reconstruction': {'reconstruction': out}}))
    assert 'dashcam' in privacy.private_reason(str(old))
    plain = tmp_path / 'plain-recon'
    plain.mkdir()
    (plain / 'run.json').write_text('{}')
    (old / 'drive.json').write_text(json.dumps({'reconstruction': {'reconstruction': str(plain)}}))
    assert privacy.private_reason(str(old)) is None
    # … and nothing of it goes into a cluster bundle
    for k, v in (('tracks', os.path.join(out, 'posed', 'tracks.json')), ('route', os.path.join(ep, 'frames.jsonl')),
                 ('hybrid', {'reconstruction': out})):
        d = tmp_path / f'job-{k}'
        d.mkdir()
        cfg = {'out': str(d), 'seed': 1, k: v}
        (d / 'drive.json').write_text(json.dumps(cfg))
        spec = JobSpec('drive', 'd', ['python', '-m', 'falcon_autonomy.workers', 'drive', 'x'], run_dir=str(d),
                       meta={'config': str(d / 'drive.json'), 'worker': 'drive'})
        with pytest.raises(B.BundleError, match='private'):
            B.make(spec)
    spec = JobSpec('drive', 'd', ['python', '-m', 'falcon_autonomy.workers', 'drive', 'x'], run_dir=ep,
                   meta={'config': os.path.join(ep, 'drive.json'), 'worker': 'drive'})
    with pytest.raises(B.BundleError, match='private'):
        B.make(spec)


def test_the_commands_explain_what_they_need(incident, tmp_path, capsys):
    from falcon_autonomy.core import dashcam_ingest as I
    from falcon_autonomy.core.recon.commands import main_recon
    clip = incident['clip']
    assert main_recon(['from-dashcam', '--clip', clip, '--tracks', incident['folder'], '--out',
                       str(tmp_path / 'x'), '--no-train']) == 2
    assert 'give --dynamic too' in capsys.readouterr().out
    lone = tmp_path / 'lone' / 'Movie_F'
    lone.mkdir(parents=True)
    shutil.copy(clip, lone / os.path.basename(clip))
    with pytest.raises(DTK.TracksError, match='no ingest of this clip'):
        DTK.find_ingest(str(lone / os.path.basename(clip)))
    src = DTK.find_ingest(clip)                                   # found beside the clip, and from an event folder
    assert src['perception'].endswith('perception.json') and src['nearmiss']
    assert DTK.find_ingest(clip, I.event_dir(incident['folder'], '1')) == src
    assert I.main_ingest(['nearmiss', 'reconstruct', incident['folder'], '99', '--out', str(tmp_path / 'y')]) == 2
    assert 'no event ranked 99' in capsys.readouterr().out
    seg = I.event_segment(incident['folder'], '1')
    ev = seg['event']
    assert seg['clip'] == clip and seg['start'] == pytest.approx(max(0.0, ev['window_s'][0] - I.RECON_PAD_S))
    assert seg['start'] + seg['seconds'] <= N / FPS + 1e-6
    spec = I.reconstruct_spec(incident['folder'], ev['id'], str(tmp_path / 'z'), 'cpu', 30, 96)
    assert spec.kind == 'recon' and not spec.needs_gpu and spec.meta['source']['format'] == 'dashcam'
    assert spec.argv[spec.argv.index('reconstruct') + 1:spec.argv.index('reconstruct') + 3] == [incident['folder'],
                                                                                                  ev['id']]
    assert spec.placement == 'local'
    assert I.main_ingest(['nearmiss', 'reconstruct', '--help']) == 0
    assert main_recon(['from-dashcam', '--help']) == 0
