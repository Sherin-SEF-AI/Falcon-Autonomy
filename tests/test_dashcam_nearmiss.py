"""Near-miss mining on a synthetic dashcam clip (core.dashcam_ingest, core.nearmiss, core.dashcam_depth): a forward
camera driving east at 10 m/s over a textured road (OpenCV's perspective warp of a noise texture), vehicles drawn as
boxes with known kinematics — a lead car that brakes and closes to 7 m, a truck and an autorickshaw-sized car in the
next lane — a GPS track written into the MP4 as the verified camera writes it, a deterministic fake detector reading
the ground-truth sidecar and a fake depth network (the scene's true inverse depth, affine-distorted). Checked: stable
track ids, the horizon from the focus of expansion, the ranges, the autorickshaw heuristic, TTC against the ground
truth's, the ranking, the exported event (stream-copied clip, posed frames, detections) becoming a scenario with an
OpenSCENARIO file on an imported road, the variation search's job, resuming, privacy, and the command."""
import datetime as dt
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from falcon_autonomy.core import criticality as CR
from falcon_autonomy.core import dashcam as DC
from falcon_autonomy.core import dashcam_depth as DD
from falcon_autonomy.core import dashcam_ingest as I
from falcon_autonomy.core import nearmiss as NM
from falcon_autonomy.core import protocol as P

pytestmark = pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='needs ffmpeg')
cv2 = pytest.importorskip('cv2')
pytest.importorskip('scipy')

W, H, FPS, SECONDS = 320, 180, 10, 7
F = W / 2 / math.tan(math.radians(90) / 2)          # 90° horizontal field of view: f = 160 px
CAM_H = 1.3
HORIZON = 80.0                                       # the camera is pitched down: the horizon above the centre (90)
EGO_V = 10.0
LANE = -1.75                                         # east-north metres: the right lane of a road along +x
UTC0 = dt.datetime(2026, 3, 21, 6, 30, 0, tzinfo=dt.timezone.utc)
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
ALPHA, BETA = 3.0, 0.2                               # the fake network's affine distortion of 1/Z


def cam():
    return DD.Camera(W, H, f=F, height_m=CAM_H, horizon_v=HORIZON)


def lead_x(t):
    """The lead car's near face (east, m): 25 m ahead at 10 m/s, braking 3 m/s² for 2 s from t = 1, then 4 m/s
    for 1 s, then accelerating 3 m/s² for 2 s — the gap closes from 25 m to 7 m."""
    s = EGO_V * t
    if t > 1:
        b = min(t, 3) - 1
        s -= 1.5 * b * b
    if t > 3:
        s -= 6.0 * (min(t, 4) - 3)
    if t > 4:
        b = min(t, 6) - 4
        s -= 6.0 * b - 1.5 * b * b
    return 25.0 + s


ACTORS = [  # id, label, width, height, length, lateral (north), near face (east) at t
    ('lead', 'car', 1.8, 1.5, 4.6, LANE, lead_x),
    ('truck', 'truck', 2.5, 3.0, 6.0, LANE + 3.5, lambda t: 18.0 + EGO_V * t),
    ('auto', 'car', 1.3, 1.7, 2.6, LANE - 3.0, lambda t: 12.0 + EGO_V * t),     # an autorickshaw, called a car
]


def ego_x(t):
    return EGO_V * t


def project(c, P_v):
    """Vehicle-frame points (N, 3) → pixels (N, 2) and camera depth."""
    pc = (c.mount() @ (np.asarray(P_v, float) - [0, 0, c.height_m]).T).T
    return np.c_[c.f * pc[:, 0] / pc[:, 2] + c.cx, c.f * pc[:, 1] / pc[:, 2] + c.cy], pc[:, 2]


def actor_box(c, t, a):
    _id, _lab, w, h, _l, lat, xf = a
    X = xf(t) - ego_x(t)
    Y = lat - LANE
    pts, z = project(c, [[X, Y - w / 2, 0], [X, Y + w / 2, 0], [X, Y - w / 2, h], [X, Y + w / 2, h]])
    if (z <= 0).any():
        return None
    x1, y1 = pts.min(0)
    x2, y2 = pts.max(0)
    return [float(x1), float(y1), float(x2), float(y2)], X


def frame_scene(c, t, texture, tex_res=0.1, tex_x0=-20.0, tex_y0=-30.0, ss=3):
    """The rendered frame (RGB) and the true camera depth per pixel. The road is the texture warped by the camera's
    ground homography at 3 × the size and area-averaged down (no aliasing)."""
    ex = ego_x(t)
    M = c.mount()
    A = np.array([[tex_res, 0, tex_x0 - ex], [0, tex_res, tex_y0 - LANE], [0, 0, 1.0]])     # texture → vehicle
    G = np.c_[M[:, 0], M[:, 1], -M[:, 2] * c.height_m]                                    # (X, Y, 1) → camera
    K = np.array([[c.f * ss, 0, c.cx * ss + (ss - 1) / 2], [0, c.f * ss, c.cy * ss + (ss - 1) / 2], [0, 0, 1]])
    big = cv2.warpPerspective(texture, K @ G @ A, (W * ss, H * ss), flags=cv2.INTER_LINEAR,
                              borderValue=(90, 90, 90))
    img = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)
    uu, vv = np.meshgrid(np.arange(W) + 0.0, np.arange(H) + 0.0)
    rays = np.stack([(uu - c.cx) / c.f, (vv - c.cy) / c.f, np.ones_like(uu)], -1) @ M      # vehicle frame, cam-z 1
    Z = np.where(rays[..., 2] < -1e-6, -c.height_m / np.minimum(rays[..., 2], -1e-9), 1e4)
    img[vv < c.horizon_v + 1] = (170, 200, 235)
    for a in sorted(ACTORS, key=lambda a: -(a[6](t) - ex)):
        got = actor_box(c, t, a)
        if got is None:
            continue
        (x1, y1, x2, y2), X = got
        colour = {'lead': (200, 40, 40), 'truck': (40, 40, 160), 'auto': (230, 200, 30)}[a[0]]
        cv2.rectangle(img, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), colour, -1)
        cv2.rectangle(img, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), (20, 20, 20), 1)
        inside = (uu >= x1) & (uu <= x2) & (vv >= y1) & (vv <= y2)
        Z = np.where(inside, X / np.maximum(rays[..., 0], 1e-6), Z)
    return img, Z


def make_clip(folder, name_time):
    c = cam()
    rng = np.random.default_rng(3)
    tex = cv2.GaussianBlur(rng.integers(0, 255, (600, 1500, 3)).astype(np.uint8), (0, 0), 3.0)     # 0.1 m a pixel
    tex = cv2.normalize(tex, None, 0, 255, cv2.NORM_MINMAX)
    path = os.path.join(folder, name_time.strftime('%Y%m%d%H%M%S') + '_000001F.MP4')
    os.makedirs(folder, exist_ok=True)
    p = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                          '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-g', '5', '-bf', '0',
                          '-crf', '12', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', path], stdin=subprocess.PIPE)
    sidecar, depths = {}, {}
    for n in range(FPS * SECONDS):
        t = n / FPS
        img, Z = frame_scene(c, t, tex)
        p.stdin.write(img.tobytes())
        boxes = []
        for a in ACTORS:
            got = actor_box(c, t, a)
            if got is None:
                continue
            b = [max(0.0, got[0][0]), max(0.0, got[0][1]), min(W - 1.0, got[0][2]), min(H - 1.0, got[0][3])]
            boxes.append({'box': b, 'score': 0.9, 'label': a[1], 'truth': a[0]})
        sidecar[str(n)] = boxes
        depths[n] = (ALPHA / Z + BETA).astype(np.float32)
    p.stdin.close()
    assert p.wait() == 0
    t0 = name_time.astimezone(dt.timezone.utc)
    blocks = []
    for k in range(SECONDS + 1):                  # one fix a second along the lane, course east
        e = ego_x(k)
        lat = math.degrees(LANE / 6371008.8)
        lon = math.degrees(e / 6371008.8)
        blocks.append(DC.encode_block(t0 + dt.timedelta(seconds=k), lat, lon, EGO_V * 3.6 / DC.KNOT_KMH, 90.0))
    DC.add_gps(path, blocks)
    side = os.path.join(os.path.dirname(folder), 'sidecar.json')
    with open(side, 'w') as fh:
        json.dump({'frames': sidecar}, fh)
    return path, side, depths


class FakeDepth:
    """The scene's true inverse depth through an affine map — what a relative depth network is assumed to give."""
    kind = 'relative'
    info = {'model': 'fake (the synthetic scene\'s depth)', 'kind': 'relative'}

    def __init__(self, depths):
        self.depths = depths

    def __call__(self, frames):
        return [self.depths[int(n)] for n, _ in frames]


SETTINGS = {'detector': 'file', 'depth': 'fake', 'rate': FPS, 'size': [W, H], 'focal_px': F, 'camera_height_m': CAM_H,
            'batch': 4, 'top': 20, 'horizon_v': HORIZON}             # a calibrated mount (docs/nearmiss.md)


@pytest.fixture(scope='module')
def mined(tmp_path_factory):
    root = tmp_path_factory.mktemp('nearmiss')
    folder = str(root / 'footage')
    clip, side, depths = make_clip(os.path.join(folder, 'Movie_F'), UTC0.astimezone(IST))
    from falcon_autonomy.core.dashcam_detect import FileDetector
    det = FileDetector(side)
    s = dict(SETTINGS, detector=f'file:{side}')
    res = I.ingest(folder, s, detector=det, depth=FakeDepth(depths))
    return {'folder': folder, 'clip': clip, 'side': side, 'depths': depths, 'summary': res, 'settings': s,
            'root': str(root)}


def _perc(m):
    od = I.out_dir(m['folder'])
    ck = I.clip_key(m['clip'])
    return json.load(open(os.path.join(od, 'clips', ck, 'perception.json')))


def test_tracks_keep_their_ids_and_the_horizon_is_found(mined):
    perc = _perc(mined)
    side = json.load(open(mined['side']))['frames']
    truth_of = {}
    for fr in perc['frames']:
        gt = side[str(fr['n'])]
        for b in fr['boxes']:
            hit = min(gt, key=lambda g: max(abs(p - q) for p, q in zip(g['box'], b['box'])))
            truth_of.setdefault(hit['truth'], set()).add(b['track'])
    assert {k: len(v) for k, v in truth_of.items()} == {'lead': 1, 'truck': 1, 'auto': 1}    # one id each, all along
    assert len(set().union(*truth_of.values())) == 3
    assert perc['horizon']['given'] and perc['camera']['horizon_v'] == HORIZON
    assert perc['counts']['frames'] == FPS * SECONDS and perc['counts']['plane_fits'] == FPS * SECONDS
    # without a calibration: the horizon from the focus of expansion while the car drives straight
    from falcon_autonomy.core.dashcam_detect import FileDetector
    s = dict(mined['settings'], horizon_v=None)
    est = I.perceive(mined['clip'], s, FileDetector(mined['side']), None)
    assert est['horizon']['pairs'] >= 30 and 'focus of expansion' in est['camera']['horizon_source']
    assert abs(est['horizon']['horizon_v'] - HORIZON) < 6.0          # a few pixels: see docs/nearmiss.md, Limits


def test_ranges_and_the_autorickshaw_heuristic(mined):
    perc = _perc(mined)
    errs, kinds = [], {}
    for fr in perc['frames']:
        t = fr['t']
        for b in fr['boxes']:
            kinds.setdefault(b['track'], set()).add((b['kind'], b['class_source']))
    lead_track = next(b['track'] for fr in perc['frames'] for b in fr['boxes'] if b['kind'] == 'car')
    for fr in perc['frames']:
        for b in fr['boxes']:
            if b['track'] == lead_track:
                true = lead_x(fr['t']) - ego_x(fr['t'])
                errs.append(abs(b['forward'] - true) / true)
                assert b['source'] == 'fused' and not b['disagree']
    assert np.median(errs) < 0.03 and max(errs) < 0.08
    by_kind = {next(iter(v)) for v in kinds.values()}
    assert ('autorickshaw', 'heuristic') in by_kind and ('car', 'model') in by_kind and ('truck', 'model') in by_kind
    auto = next(b for fr in perc['frames'] for b in fr['boxes'] if b['kind'] == 'autorickshaw' and 'heuristic' in b)
    assert 'wide' in auto['heuristic'] and 1.1 <= auto['size_m'][0] <= 1.6


def ground_truth_ttc():
    """The lead's minimum TTC by core.criticality on the true trajectories (ego box centred on the camera)."""
    t = np.arange(0, SECONDS, 1 / FPS)
    e = np.c_[t, ego_x(t), np.full_like(t, LANE), np.zeros_like(t), np.full_like(t, EGO_V), np.zeros_like(t),
              np.full_like(t, 4.6), np.full_like(t, 2.0)]
    lx = np.array([lead_x(x) for x in t]) + 2.3
    a = np.c_[t, lx, np.full_like(t, LANE), np.zeros_like(t), np.gradient(lx, t), np.zeros_like(t),
              np.full_like(t, 4.6), np.full_like(t, 2.0)]
    tt = CR.ttc_series(e, a)[1]
    return float(tt.min()), float(t[np.argmin(tt)])


def test_ttc_ranking_and_the_ranked_list(mined):
    rk = I.ranking(mined['folder'])
    assert rk and rk[0]['rank'] == 1 and rk[0]['kind'] == 'car'
    top = rk[0]
    gt, gt_t = ground_truth_ttc()
    assert 1.5 < gt < 3.0
    assert abs(top['min_ttc_s'] - gt) < 0.5, (top['min_ttc_s'], gt)
    assert abs(top['min_ttc_t'] - gt_t) < 1.0
    assert top['score_ci'][0] <= top['score_median'] <= top['score_ci'][1]
    assert top['score_ci'][1] - top['score_ci'][0] > 0                          # the ±10 % depth scale matters
    assert 'closing in the corridor' in top['why'] and top['vo']['used']
    assert top['vo']['departure_from_gps_deg'] < 5.0 and top['vo']['position_departure_m'] < 3.0   # 320 px: noisy
    others = [e for e in rk if e['kind'] != 'car']
    assert others and all(e['score_median'] < top['score_median'] for e in others)
    assert [e['rank'] for e in rk] == list(range(1, len(rk) + 1))


def test_the_event_folder_scenario_and_openscenario(mined, tmp_path):
    from falcon_autonomy.core import openscenario as OS
    from test_incident import XODR
    top = I.ranking(mined['folder'])[0]
    d = I.event_dir(mined['folder'], '1')
    assert d == I.event_dir(mined['folder'], top['id'])
    ev = json.load(open(os.path.join(d, 'event.json')))
    assert ev['rank'] == 1 and ev['clip_cut']['how'].startswith('ffmpeg -c copy')
    pr = DC.probe(os.path.join(d, 'clip.mp4'))
    assert pr['duration_s'] and pr['duration_s'] >= (ev['window_s'][1] - ev['window_s'][0])
    det = json.load(open(os.path.join(d, 'detections.json')))
    assert det['schema'] == 'falcon.dashcam_detections/1' and all('range_m' in b for f in det['frames']
                                                                  for b in f['boxes'])
    posed = json.load(open(os.path.join(d, 'posed', 'posed.json')))
    assert len(posed['frames']) == len(det['frames']) and posed['geo']['lat0'] == pytest.approx(
        math.degrees(LANE / 6371008.8), abs=2e-5)
    # a project with an imported road where the footage drove (null island: no real place)
    from falcon_autonomy.core.project import Project
    p = Project.new('nm')
    p.save(str(tmp_path / 'nm.falcon.json'))
    wdir = os.path.join(p.workspace, 'worlds', 'w-test')
    os.makedirs(wdir, exist_ok=True)
    with open(os.path.join(wdir, 'world.xodr'), 'w') as fh:
        fh.write(XODR)
    p.doc['worlds'].append({'id': 'w-test', 'name': 'Synthetic road', 'kind': 'osm', 'xodr': 'worlds/w-test/world.xodr',
                            'osm': {'bbox': [-0.01, -0.01, 0.01, 0.01], 'centre': [0.0, 0.0], 'handedness': 'RHT'}})
    p.save(str(tmp_path / 'nm.falcon.json'))
    res = NM.make_scenario(d, p, 'w-test')
    inc_dir = res['folder']
    assert res['actors'] >= 1 and os.path.isfile(os.path.join(inc_dir, 'incident.xosc'))
    inc = json.load(open(os.path.join(inc_dir, 'incident.json')))
    lead = min(inc['actors'], key=lambda a: abs(a['track']['y'][0] - inc['ego']['track']['y'][0]))
    gap0 = lead['track']['x'][0] - inc['ego']['track']['x'][0]
    t_ev = posed['frames'][0]['video_t']
    assert abs(gap0 - (lead_x(t_ev) - ego_x(t_ev) + 2.3)) < 1.5          # the fused range carried into the scenario
    with open(os.path.join(inc_dir, 'incident.xosc'), encoding='utf-8') as fh:
        text = fh.read()
    assert '<FollowTrajectoryAction>' in text and '<WorldPosition' in text
    back = OS.import_xosc(text)                                   # Falcon's importer reads it back
    assert back['ego']['driver'] == 'replay' and any(a['kind'] == 'car' for a in back['actors'])
    from falcon_autonomy.assistant.privacy import private_reason
    assert private_reason(inc_dir) and private_reason(d)
    spec = NM.variations_spec(p.path, p.workspace, res['name'], budget=4)
    assert spec.kind == 'study' and 'search' in spec.argv and spec.meta['source'] == {'format': 'dashcam'}
    assert getattr(spec, 'placement', 'local') == 'local'


def test_resume_privacy_and_summary(mined, monkeypatch):
    from falcon_autonomy.assistant.privacy import private_reason
    from falcon_autonomy.core.dashcam_detect import FileDetector
    res = I.ingest(mined['folder'], mined['settings'], detector=FileDetector(mined['side']),
                   depth=FakeDepth(mined['depths']))
    st = json.load(open(os.path.join(I.out_dir(mined['folder']), 'ingest.json')))
    assert st['counts']['reused'] == 1 and st['counts']['perceived'] == 0 and st['counts']['mined'] == 0
    assert st['worker'] == 'dashcam-ingest' and st['status'] == 'ok'
    text = json.dumps(res)
    assert os.path.basename(mined['clip']) not in text and 'lat' not in text and 'Movie_F' not in text
    assert res['events'] >= 3 and res['top'][0]['rank'] == 1 and set(res['top'][0]) == {'rank', 'id', 'kind',
                                                                                        'score_median'}
    od = I.out_dir(mined['folder'])
    for p in (mined['folder'], mined['clip'], od, os.path.join(od, 'ranking.json'),
              os.path.join(od, 'clips', I.clip_key(mined['clip']), 'perception.json')):
        assert private_reason(p), p
    from falcon_autonomy.assistant.privacy import guard
    from falcon_autonomy.assistant.tool_api import ToolError
    with pytest.raises(ToolError, match='private'):
        guard(os.path.join(od, 'ranking.json'))


def test_command_lists_and_refuses(mined, capsys):
    assert I.main_ingest(['nearmiss', 'list', mined['folder'], '--top', '3']) == P.OK
    out = capsys.readouterr().out
    assert out.splitlines()[0].lstrip().startswith('1 ') and 'TTC' in out
    assert I.main_ingest(['nearmiss', 'list', mined['root']]) == P.OK
    assert 'no near-misses' in capsys.readouterr().out
    assert I.main_ingest(['ingest', mined['folder'], '--detector', 'yolo11n']) == P.FAILED
    assert 'AGPL' in capsys.readouterr().out
    assert I.main_ingest(['ingest', mined['folder'], '--detector', 'rtdetr-v2-r18', '--models',
                          os.path.join(mined['root'], 'no-models'), '--redo']) == P.FAILED
    assert 'not downloaded' in capsys.readouterr().out
    assert I.main_ingest(['nearmiss', 'scenario', mined['folder'], '99', '--project', 'x', '--world', 'w']) == P.FAILED
    assert I.main_ingest(['nearmiss', 'models']) == P.OK
    assert 'REFUSED' in capsys.readouterr().out
    assert I.main_ingest(['ingest', '--help']) == P.OK and I.main_ingest(['nearmiss', '--help']) == P.OK


def test_cli_ingest_with_a_file_detector_and_no_depth(tmp_path):
    """The command end to end in a subprocess: the protocol lines, flat-ground ranges only."""
    folder = str(tmp_path / 'f')
    clip, side, _ = make_clip(os.path.join(folder, 'Movie_F'), (UTC0 + dt.timedelta(hours=1)).astimezone(IST))
    from conftest import worker_env
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy', 'dashcam', 'ingest', folder, '--detector',
                        f'file:{side}', '--depth', 'none', '--size', f'{W}x{H}', '--rate', str(FPS), '--focal-px',
                        str(F), '--camera-height', str(CAM_H), '--device', 'cpu', '--no-clips'],
                       capture_output=True, text=True, env=worker_env(), timeout=600)
    assert r.returncode == P.OK, r.stdout[-2000:] + r.stderr[-2000:]
    msgs = [P.parse_line(x) for x in r.stdout.splitlines() if x.startswith('{"t"')] if hasattr(P, 'parse_line') \
        else [json.loads(x) for x in r.stdout.splitlines() if x.startswith('{"t"')]
    assert any(m.get('t') == 'result' and m.get('status') == 'ok' for m in msgs)
    rk = I.ranking(folder)
    assert rk and rk[0]['kind'] == 'car'
    perc = json.load(open(os.path.join(I.out_dir(folder), 'clips', I.clip_key(clip), 'perception.json')))
    assert {b['source'] for fr in perc['frames'] for b in fr['boxes']} == {'geometric'}
    assert not os.path.isfile(os.path.join(I.event_dir(folder, '1'), 'clip.mp4'))
