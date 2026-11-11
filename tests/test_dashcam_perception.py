"""The parts of dashcam perception on their own (core.perception_models, core.dashcam_detect, core.dashcam_depth,
core.nearmiss): licence refusals (YOLO's AGPL, Depth Anything V2 Base/Large's CC-BY-NC, SMPL), the opt-in for weights
whose terms are unstated, a pinned download from a local server checked against its checksums (nothing reaches the
network), the ground-plane scale fit, the fusion of the two ranges, the autorickshaw heuristic and rider folding, the
smoother and the odometry's heading sign, and the ingest folder's privacy. The real models run only with
FALCON_MODELS=1 and their weights downloaded (marker `models`)."""
import hashlib
import http.server
import json
import math
import os
import threading

import numpy as np
import pytest

from falcon_autonomy.core import dashcam_depth as DD
from falcon_autonomy.core import dashcam_detect as DT
from falcon_autonomy.core import nearmiss as NM
from falcon_autonomy.core import perception_models as PM


# ------------------------------------------------------------------------------------------------ licences
@pytest.mark.parametrize('name', ['yolo11n', 'yolov8s', 'ultralytics/yolo11', 'YOLO11x-seg'])
def test_yolo_is_refused_everywhere(name):
    assert 'AGPL' in PM.refused_reason(name)
    with pytest.raises(DT.DetectError, match='AGPL'):
        DT.make_detector(name)
    with pytest.raises(PM.PerceptionModelError, match='AGPL'):
        PM.fetch(name, accept_licence=True)


@pytest.mark.parametrize('name', ['da2-base', 'da2-large', 'depth-anything/Depth-Anything-V2-Large-hf',
                                  'Depth-Anything-V2-Base'])
def test_non_commercial_depth_is_refused(name):
    assert 'non-commercial' in PM.refused_reason(name)
    with pytest.raises(DD.DepthError, match='non-commercial'):
        DD.make_depth(name, 500.0)


def test_smpl_refused_and_the_table():
    assert 'non-commercial' in PM.refused_reason('smplx-neutral')
    rows = {r['name']: r for r in PM.list_models('/nonexistent')}
    assert rows['rtdetr-v2-r18']['licence'] == 'Apache-2.0' and rows['da2-small']['licence'] == 'Apache-2.0'
    assert rows['metric3d-small']['opt_in'] and rows['yolo']['state'] == 'refused'
    for n, c in PM.CANDIDATES.items():                       # every candidate says where its licence was read, when
        assert c['licence_source'].count('https://') >= 1 and c['licence_checked'] == PM.LICENCE_CHECKED
        assert len(c['revision']) == 40 and all(f['git_oid'] and len(f['git_oid']) == 40 for f in c['files'])
        assert any(f['lfs_sha256'] and len(f['lfs_sha256']) == 64 for f in c['files'])
    assert PM.check_licence(PM.CANDIDATES['metric3d-small'])
    assert PM.check_licence(PM.CANDIDATES['metric3d-small'], accept_unstated=True) == []
    assert PM.check_licence({'licence': 'AGPL-3.0'}) and PM.check_licence({'licence': 'CC-BY-NC-4.0'})
    text = PM.describe('rtdetr-v2-r18')
    assert 'Apache-2.0' in text and PM.CANDIDATES['rtdetr-v2-r18']['revision'] in text


def test_nothing_is_fetched_without_accepting(tmp_path):
    with pytest.raises(PM.PerceptionModelError, match='accept'):
        PM.fetch('da2-small', str(tmp_path), accept_licence=False)
    with pytest.raises(PM.PerceptionModelError, match='not stated'):
        PM.fetch('metric3d-small', str(tmp_path), accept_licence=True)
    assert os.listdir(tmp_path) == []
    with pytest.raises(PM.PerceptionModelError, match='not downloaded'):
        PM.load_folder('rtdetr-v2-r18', str(tmp_path))
    with pytest.raises(DT.DetectError, match='not downloaded'):
        DT.make_detector('rtdetr-v2-r50', str(tmp_path))


def _git_oid(data):
    return hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()


def test_a_pinned_download_from_a_local_server(tmp_path, monkeypatch):
    """fetch() against a loopback server holding a stand-in candidate's files: sizes and the publisher's checksums
    checked, pins.json written, load_folder verifies; a tampered file is refused and not installed."""
    files = {'config.json': b'{"x": 1}', 'model.safetensors': os.urandom(4096)}
    served = dict(files)

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            name = self.path.rsplit('/', 1)[-1]
            data = served.get(name)
            self.send_response(200 if data is not None else 404)
            self.end_headers()
            if data is not None:
                self.wfile.write(data)

        def log_message(self, *a):
            pass
    srv = http.server.HTTPServer(('127.0.0.1', 0), H)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        fake = dict(PM.CANDIDATES['rtdetr-v2-r18'], repo='test/stand-in', revision='0' * 40,
                    files=[{'name': 'config.json', 'size': len(files['config.json']), 'lfs_sha256': None,
                            'git_oid': _git_oid(files['config.json'])},
                           {'name': 'model.safetensors', 'size': len(files['model.safetensors']),
                            'lfs_sha256': hashlib.sha256(files['model.safetensors']).hexdigest(),
                            'git_oid': 'unused'}],
                    bytes=sum(len(v) for v in files.values()))
        monkeypatch.setitem(PM.CANDIDATES, 'rtdetr-v2-r18', fake)
        url = f'http://127.0.0.1:{srv.server_port}'
        folder = PM.fetch('rtdetr-v2-r18', str(tmp_path), accept_licence=True, base_url=url)
        pins = json.load(open(os.path.join(folder, 'pins.json')))
        assert pins['files']['model.safetensors']['sha256'] == fake['files'][1]['lfs_sha256']
        assert pins['accepted_licence'] and pins['accepted_unstated_terms'] is None
        assert PM.load_folder('rtdetr-v2-r18', str(tmp_path)) == folder
        with open(os.path.join(folder, 'model.safetensors'), 'ab') as fh:
            fh.write(b'x')
        with pytest.raises(PM.PerceptionModelError, match='bytes'):
            PM.load_folder('rtdetr-v2-r18', str(tmp_path))
        served['model.safetensors'] = files['model.safetensors'][:-1] + b'\0'       # the server sends other bytes
        with pytest.raises(PM.PerceptionModelError, match='not installed'):
            PM.fetch('rtdetr-v2-r18', str(tmp_path / 'again'), accept_licence=True, base_url=url)
        assert not os.path.exists(os.path.join(str(tmp_path / 'again'), 'rtdetr-v2-r18', 'model.safetensors'))
    finally:
        srv.shutdown()


# ------------------------------------------------------------------------------------------------ geometry
def scene(cam, alpha=2.5, beta=0.4, boxes=()):
    """The exact inverse depth of a flat road (affine-distorted like a relative network), boxes as fronto-parallel
    faces at their forward distance."""
    uu, vv = np.meshgrid(np.arange(cam.width) + 0.0, np.arange(cam.height) + 0.0)
    rays = np.stack([(uu - cam.cx) / cam.f, (vv - cam.cy) / cam.f, np.ones_like(uu)], -1) @ cam.mount()
    Z = np.where(rays[..., 2] < -1e-6, -cam.height_m / np.minimum(rays[..., 2], -1e-9), 1e4)
    for (x1, y1, x2, y2), X in boxes:
        inside = (uu >= x1) & (uu <= x2) & (vv >= y1) & (vv <= y2)
        Z = np.where(inside, X / np.maximum(rays[..., 0], 1e-6), Z)
    return alpha / Z + beta, Z


def face_box(cam, X, Y, w=1.8, h=1.5):
    pts = np.array([[X, Y - w / 2, 0], [X, Y + w / 2, 0], [X, Y - w / 2, h], [X, Y + w / 2, h]])
    pc = (cam.mount() @ (pts - [0, 0, cam.height_m]).T).T
    uv = np.c_[cam.f * pc[:, 0] / pc[:, 2] + cam.cx, cam.f * pc[:, 1] / pc[:, 2] + cam.cy]
    return [*uv.min(0), *uv.max(0)]


def test_ground_plane_fit_recovers_the_scale_and_shift():
    cam = DD.Camera(640, 360, hfov_deg=100, height_m=1.4, horizon_v=160)
    b = face_box(cam, 18.0, 1.0)
    disp, Z = scene(cam, boxes=[(b, 18.0)])
    fit = DD.fit_ground(disp.astype(np.float32), cam, boxes=[b])
    assert fit['alpha'] == pytest.approx(2.5, rel=1e-3) and fit['beta'] == pytest.approx(0.4, abs=1e-3)
    assert fit['alpha_sigma'] < 0.01 and abs(fit['roll_slope']) < 1e-6
    net = DD.net_point(DD.box_value(disp, b), 'relative', fit, cam, b)
    geo = DD.geo_point(cam, b)
    assert net['forward'] == pytest.approx(18.0, rel=0.02) and geo['forward'] == pytest.approx(18.0, rel=0.01)
    assert geo['left'] == pytest.approx(1.0, abs=0.05) and net['left'] == pytest.approx(1.0, abs=0.1)
    fz = DD.fuse(geo, net)
    assert fz['source'] == 'fused' and fz['sigma'] < min(geo['sigma'], net['sigma'])
    assert fz['forward'] == pytest.approx(18.0, rel=0.02) and not fz['disagree']
    # metric depth: the camera height the road implies, and the scale that corrects a wrong focal guess
    m = DD.fit_ground((Z * 1.25).astype(np.float32), cam, boxes=[b], metric=True)
    assert m['height_implied_m'] == pytest.approx(1.4 * 1.25, rel=1e-3) and m['scale'] == pytest.approx(0.8, rel=1e-3)
    assert DD.net_point(DD.box_value(Z * 1.25, b), 'metric', m, cam, b)['forward'] == pytest.approx(18.0, rel=0.02)
    # no road: no fit; a horizon-only shift breaks no fit (the plane is independent of it)
    assert DD.fit_ground(np.zeros((360, 640), np.float32), cam) is None
    pl = DD.fit_plane(disp.astype(np.float32), cam, [b], v_min=200)
    assert DD.scale_plane(pl, cam)['alpha'] == pytest.approx(2.5, rel=1e-3)


def test_fusion_disagreement_keeps_the_trustworthy_one():
    geo = {'forward': 20.0, 'left': 1.0, 'sigma': 1.0, 'cut': False}
    net = {'forward': 35.0, 'left': 1.8, 'sigma': 2.0}
    f = DD.fuse(geo, net)
    assert f['disagree'] and f['source'] == 'geometric' and f['sigma'] >= 7.5
    f = DD.fuse(dict(geo, cut=True), net)
    assert f['source'] == 'network'
    assert DD.fuse(None, None) is None and DD.fuse(geo, None)['source'] == 'geometric'
    cam = DD.Camera(640, 360, hfov_deg=100, horizon_v=160)
    assert DD.geo_point(cam, [100, 50, 140, 150]) is None           # its bottom above the horizon
    assert cam.doc()['pitch_deg'] > 0 and DD.Camera(640, 360).horizon_source.startswith('image centre')


def test_horizon_estimate_needs_enough_pairs():
    cam = DD.Camera(320, 180)
    assert DD.estimate_horizon([{'u': 160, 'v': 70}] * 3, cam)['pairs'] == 3 and cam.horizon_v == 90
    r = DD.estimate_horizon([{'u': 160, 'v': 70 + k % 3} for k in range(9)], cam)
    assert cam.horizon_v == pytest.approx(71) and r['pairs'] == 9 and 'focus of expansion' in cam.horizon_source


def test_autorickshaw_heuristic_and_riders():
    cam = DD.Camera(960, 540, hfov_deg=100, height_m=1.3, horizon_v=270)
    auto = face_box(cam, 12.0, 0.0, w=1.3, h=1.75)
    car = face_box(cam, 12.0, 3.5, w=1.8, h=1.45)
    raw = [{'box': auto, 'score': 0.8, 'label': 'car'}, {'box': car, 'score': 0.9, 'label': 'car'},
           {'box': [400, 300, 420, 360], 'score': 0.9, 'label': 'traffic light'}]
    dets, folded = DT.postprocess(raw, cam)
    assert [d['kind'] for d in dets] == ['autorickshaw', 'car'] and folded == 0
    assert dets[0]['class_source'] == 'heuristic' and dets[0]['coco'] == 'car' and 'wide' in dets[0]['heuristic']
    assert dets[1]['class_source'] == 'model' and 1.6 < dets[1]['size_m'][0] < 2.0
    bike = face_box(cam, 10.0, -2.0, w=0.8, h=1.2)
    rider = [bike[0] + 2, bike[1] - 60, bike[2] - 2, bike[3] - 15]
    walker = [700, 250, 730, 330]
    dets, folded = DT.postprocess([{'box': bike, 'score': 0.9, 'label': 'motorcycle'},
                                   {'box': rider, 'score': 0.9, 'label': 'person'},
                                   {'box': walker, 'score': 0.9, 'label': 'person'}], cam)
    assert folded == 1 and sorted(d['kind'] for d in dets) == ['motorcycle', 'pedestrian']
    assert DT.map_detections([{'box': [0, 0, 1, 1], 'label': 'cow'}])[0]['kind'] == 'cattle'


# ------------------------------------------------------------------------------------------------ smoothing, odometry
def test_rts_smooths_a_constant_velocity_track():
    rng = np.random.default_rng(0)
    t = np.arange(0, 10, 0.1)
    truth = np.c_[3.0 * t, -1.0 * t]
    meas = [[(np.eye(2, 4), p + rng.normal(0, 1.0, 2), np.eye(2))] for p in truth]
    xs, P = NM.rts(t, meas, 0.5)
    assert np.sqrt(np.mean(np.sum((xs[:, :2] - truth) ** 2, 1))) < 0.5
    assert np.abs(xs[10:-10, 2] - 3.0).max() < 0.5 and np.abs(xs[10:-10, 3] + 1.0).max() < 0.5
    d1, d2 = NM.savgol_derivs(10.0 - 2.0 * t ** 2 / 2, 0.1)              # braking 2 m/s²: accel −2, jerk 0
    assert np.allclose(d1[5:-5], -2.0 * t[5:-5], atol=1e-6) and np.allclose(d2[5:-5], -2.0, atol=1e-6)


def test_odometry_heading_turns_the_right_way():
    """A camera over a textured road yawing left (counter-clockwise from above) by 1° a frame while moving 0.5 m:
    +1° increments, from the road's homography."""
    import cv2
    cam = DD.Camera(320, 180, hfov_deg=90, height_m=1.3)
    rng = np.random.default_rng(1)
    tex = cv2.normalize(cv2.GaussianBlur(rng.integers(0, 255, (800, 800)).astype(np.uint8), (0, 0), 3.0), None, 0,
                        255, cv2.NORM_MINMAX)                              # 0.1 m a pixel
    grays = []
    ss = 3                                                                 # rendered 3 × and area-averaged: no aliasing
    for k in range(6):
        a = math.radians(1.0 * k)
        Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        M = cam.mount()
        A = np.array([[0.1, 0, -20 - 0.5 * k], [0, 0.1, -40.0], [0, 0, 1.0]])
        Rv = Rz.T                                                         # world → vehicle
        G = np.c_[(M @ Rv)[:, 0], (M @ Rv)[:, 1], -M[:, 2] * cam.height_m]
        K = np.array([[cam.f * ss, 0, cam.cx * ss + 1], [0, cam.f * ss, cam.cy * ss + 1], [0, 0, 1]])
        big = cv2.warpPerspective(tex, K @ G @ A, (320 * ss, 180 * ss), flags=cv2.INTER_LINEAR)
        img = cv2.resize(big, (320, 180), interpolation=cv2.INTER_AREA)
        img[:95] = 128
        grays.append(img)
    vo = NM.vo_heading(grays, cam)
    d = np.array(vo['dyaw_deg'])
    assert vo['failed'] == 0 and np.all(np.abs(d - 1.0) < 0.3) and abs(d.mean() - 1.0) < 0.15, d
    assert vo['methods'] == {'road homography': 5}


# ------------------------------------------------------------------------------------------------ privacy
def test_an_ingest_folder_is_private_at_any_depth(tmp_path):
    from falcon_autonomy.assistant.privacy import private_reason
    root = tmp_path / 'clips'
    deep = root / 'falcon-ingest' / 'events' / 'abc' / 'posed'
    deep.mkdir(parents=True)
    (deep / 'posed.json').write_text('{}')
    (root / 'Movie_F').mkdir()
    (root / 'Movie_F' / 'x.MP4').write_bytes(b'')
    assert private_reason(str(root)) is None                         # no ingest record yet
    (root / 'falcon-ingest' / 'ingest.json').write_text(json.dumps({'worker': 'dashcam-ingest'}))
    for p in (root, root / 'Movie_F' / 'x.MP4', deep / 'posed.json', root / 'falcon-ingest'):
        assert private_reason(str(p)), p
    assert private_reason(str(tmp_path)) is None                     # the folder above is not


# ------------------------------------------------------------------------------------------------ real weights
@pytest.mark.models
@pytest.mark.skipif(os.environ.get('FALCON_MODELS') != '1', reason='real model weights: FALCON_MODELS=1')
def test_real_models_run_offline():
    """RT-DETRv2-S and Depth Anything V2 Small from their pinned, verified folders (downloaded by the lead at
    acceptance: falcon dashcam nearmiss fetch NAME --accept-licence) on a synthetic frame, on the CPU."""
    import cv2
    for n in ('rtdetr-v2-r18', 'da2-small'):
        try:
            PM.load_folder(n)
        except PM.PerceptionModelError as e:
            pytest.skip(str(e))
    img = np.full((360, 640, 3), 120, np.uint8)
    cv2.rectangle(img, (280, 180), (360, 240), (200, 30, 30), -1)
    det = DT.make_detector('rtdetr-v2-r18', device='cpu')
    out = det([(0, img)])
    assert isinstance(out[0], list) and det.info['licence'] == 'Apache-2.0'
    dep = DD.make_depth('da2-small', 500.0, device='cpu')
    m = dep([(0, img)])[0]
    assert m.shape == (360, 640) and np.isfinite(m).all()
