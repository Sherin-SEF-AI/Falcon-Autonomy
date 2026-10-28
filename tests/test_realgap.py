"""The sim-to-real gap (core.realgap): the readers on hand-made sets in each format's layout, the statistics against
signals with known answers, the distances against closed forms, appearance matching, and the command end to end on
a drive the worker records against the fake carla. The feature extractor is never downloaded here: its use is
tested only when its weights are already in FALCON_FEATURES_WORKSPACE."""
import json
import math
import os

import numpy as np
import pytest

from falcon_autonomy.core import protocol as P
from falcon_autonomy.core import realgap as G

cv2 = pytest.importorskip('cv2')


def _img(path, rgb):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _scene(rng, h=96, w=160, light=120, noise=0.0, blur=0.0):
    """A textured test image: smooth gradients, a few boxes, optional blur and noise."""
    y, x = np.mgrid[0:h, 0:w]
    im = np.stack([light + 40 * np.sin(x / 9.0), light + 30 * np.cos(y / 7.0), light + 20 * np.sin((x + y) / 11.0)],
                  -1)
    for _ in range(4):
        x0, y0 = rng.integers(0, w - 30), rng.integers(0, h - 20)
        im[y0:y0 + 20, x0:x0 + 30] = rng.integers(0, 255, 3)
    if blur:
        im = cv2.GaussianBlur(im, (0, 0), blur)
    im = im + rng.normal(0, noise, im.shape[:2])[..., None] if noise else im
    return np.clip(np.round(im), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------------------------------------------- readers
def test_kitti_object_reader(tmp_path):
    _img(tmp_path / 'image_2' / '000000.png', np.zeros((40, 60, 3), np.uint8))
    (tmp_path / 'label_2').mkdir()
    (tmp_path / 'label_2' / '000000.txt').write_text(
        'Car 0.00 0 -1.58 587.01 173.33 614.12 200.12 1.65 1.67 3.64 -0.65 1.71 46.70 -1.59\n'
        'Cyclist 0.00 0 1.0 10 20 30 60 1.7 0.6 1.8 3.0 1.6 12.0 0.0\n'
        'DontCare -1 -1 -10 503.89 169.71 590.61 190.13 -1 -1 -1 -1000 -1000 -1000 -10\n')
    s = G.read_set('kitti_object', str(tmp_path))
    assert s['labelled'] and s['n_total'] == 1
    objs = s['items'][0]['objects']
    assert [o['class'] for o in objs] == ['car', 'bicycle']
    assert objs[0]['bbox'] == [587.01, 173.33, 614.12, 200.12]
    assert objs[0]['range_m'] == pytest.approx(math.sqrt(0.65 ** 2 + 1.71 ** 2 + 46.7 ** 2))


def test_nuscenes_reader_projects_boxes(tmp_path):
    """A car 20 m ahead and 2 m left of a front camera 1.5 m up (nuScenes' CAM_FRONT rotation): its projected box
    is the near face's corners through K, exactly."""
    rel = tmp_path / 'v1.0-mini'
    rel.mkdir()
    K = [[1000.0, 0, 800.0], [0, 1000.0, 450.0], [0, 0, 1]]
    tables = {
        'sensor': [{'token': 's1', 'channel': 'CAM_FRONT', 'modality': 'camera'},
                   {'token': 's2', 'channel': 'CAM_BACK', 'modality': 'camera'}],
        'calibrated_sensor': [{'token': 'c1', 'sensor_token': 's1', 'translation': [0, 0, 1.5],
                               'rotation': [0.5, -0.5, 0.5, -0.5], 'camera_intrinsic': K},
                              {'token': 'c2', 'sensor_token': 's2', 'translation': [0, 0, 1.5],
                               'rotation': [0.5, -0.5, -0.5, 0.5], 'camera_intrinsic': K}],
        'ego_pose': [{'token': 'e1', 'translation': [0, 0, 0], 'rotation': [1, 0, 0, 0]}],
        'category': [{'token': 'k1', 'name': 'vehicle.car'}, {'token': 'k2', 'name': 'movable_object.barrier'}],
        'instance': [{'token': 'i1', 'category_token': 'k1'}, {'token': 'i2', 'category_token': 'k1'},
                     {'token': 'i3', 'category_token': 'k2'}],
        'scene': [{'token': 'sc', 'description': 'Night, rain, parked cars'}],
        'sample': [{'token': 'sa', 'scene_token': 'sc', 'timestamp': 1}],
        'sample_annotation': [
            {'sample_token': 'sa', 'instance_token': 'i1', 'translation': [20, 2, 1.5], 'size': [2, 4, 1.5],
             'rotation': [1, 0, 0, 0]},
            {'sample_token': 'sa', 'instance_token': 'i2', 'translation': [-10, 0, 1.5], 'size': [2, 4, 1.5],
             'rotation': [1, 0, 0, 0]},                                   # behind the camera
            {'sample_token': 'sa', 'instance_token': 'i3', 'translation': [10, 0, 0.5], 'size': [1, 1, 1],
             'rotation': [1, 0, 0, 0]}],                                  # no Falcon class
        'sample_data': [{'token': 'd1', 'sample_token': 'sa', 'ego_pose_token': 'e1', 'calibrated_sensor_token': 'c1',
                         'filename': 'samples/CAM_FRONT/a.jpg', 'width': 1600, 'height': 900, 'is_key_frame': True,
                         'timestamp': 1},
                        {'token': 'd2', 'sample_token': 'sa', 'ego_pose_token': 'e1', 'calibrated_sensor_token': 'c2',
                         'filename': 'samples/CAM_BACK/a.jpg', 'width': 1600, 'height': 900, 'is_key_frame': True,
                         'timestamp': 1}],
    }
    for k, v in tables.items():
        (rel / f'{k}.json').write_text(json.dumps(v))
    s = G.read_set('nuscenes', str(tmp_path))
    assert s['n_total'] == 1 and s['items'][0]['image'].endswith('samples/CAM_FRONT/a.jpg')
    it = s['items'][0]
    assert it['conditions'] == {'night': True, 'rain': True}
    (o,) = it['objects']
    assert o['class'] == 'car'
    assert o['bbox'] == pytest.approx([800 - 1000 * 3 / 18, 450 - 1000 * 0.75 / 18, 800 - 1000 * 1 / 22,
                                       450 + 1000 * 0.75 / 18])
    assert o['range_m'] == pytest.approx(math.hypot(20, 2))
    back = G.read_set('nuscenes', str(tmp_path), camera='CAM_BACK')['items'][0]['objects']
    assert [o['class'] for o in back] == ['car']                      # the car behind is ahead of the back camera


def test_bdd_coco_folder_and_kitti_raw_readers(tmp_path):
    _img(tmp_path / 'img' / 'a.jpg', np.zeros((20, 30, 3), np.uint8))
    _img(tmp_path / 'img' / 'sub' / 'b.png', np.zeros((20, 30, 3), np.uint8))
    (tmp_path / 'bdd.json').write_text(json.dumps([
        {'name': 'a.jpg', 'attributes': {'weather': 'rainy', 'timeofday': 'night', 'scene': 'city street'},
         'labels': [{'category': 'car', 'box2d': {'x1': 1, 'y1': 2, 'x2': 10, 'y2': 15}},
                    {'category': 'motor', 'box2d': {'x1': 1, 'y1': 2, 'x2': 5, 'y2': 9}},
                    {'category': 'drivable area', 'poly2d': []}]},
        {'name': 'missing.jpg', 'labels': []}]))
    s = G.read_set('bdd100k', str(tmp_path / 'img'), labels=str(tmp_path / 'bdd.json'))
    assert s['n_total'] == 1
    assert [o['class'] for o in s['items'][0]['objects']] == ['car', 'motorcycle']
    assert s['items'][0]['conditions']['timeofday'] == 'night'
    (tmp_path / 'coco.json').write_text(json.dumps({
        'images': [{'id': 2, 'file_name': 'sub/b.png'}, {'id': 1, 'file_name': 'a.jpg'}],
        'categories': [{'id': 1, 'name': 'person'}, {'id': 2, 'name': 'umbrella'}],
        'annotations': [{'image_id': 1, 'category_id': 1, 'bbox': [1, 2, 3, 4], 'iscrowd': 0},
                        {'image_id': 1, 'category_id': 2, 'bbox': [1, 2, 3, 4], 'iscrowd': 0},
                        {'image_id': 2, 'category_id': 1, 'bbox': [0, 0, 5, 5], 'iscrowd': 1}]}))
    c = G.read_set('coco', str(tmp_path / 'img'), labels=str(tmp_path / 'coco.json'))
    assert [it['image'].endswith('a.jpg') for it in c['items']] == [True, False]
    assert c['items'][0]['objects'] == [{'class': 'pedestrian', 'bbox': [1, 2, 4, 6]}]
    assert c['items'][1]['objects'] == []
    assert 'umbrella' in c['notes'][0]
    f = G.read_set('folder', str(tmp_path / 'img'))
    assert f['n_total'] == 2 and not f['labelled']
    for n in range(3):
        _img(tmp_path / 'raw' / 'drive_a' / 'image_02' / 'data' / f'{n:010d}.png', np.zeros((8, 8, 3), np.uint8))
    assert G.read_set('kitti_raw', str(tmp_path / 'raw'))['n_total'] == 3
    assert G.read_set('kitti_raw', str(tmp_path / 'raw' / 'drive_a'))['n_total'] == 3
    sub = G.read_set('kitti_raw', str(tmp_path / 'raw'), limit=2, seed=5)
    assert len(sub['items']) == 2 and sub['n_total'] == 3
    assert sub['items'] == G.read_set('kitti_raw', str(tmp_path / 'raw'), limit=2, seed=5)['items']
    with pytest.raises(G.GapError):
        G.read_set('bdd100k', str(tmp_path / 'img'))
    with pytest.raises(G.GapError):
        G.read_set('nope', str(tmp_path))
    with pytest.raises(G.GapError):
        G.read_set('folder', str(tmp_path / 'absent'))


# ---------------------------------------------------------------------------------------------------- statistics
def test_noise_estimate_spectrum_slope_and_sharpness():
    rng = np.random.default_rng(0)
    flat = np.full((480, 640), 128.0) + rng.normal(0, 5.0, (480, 640))
    assert G.noise_sigma(flat) == pytest.approx(5.0, rel=0.05)
    # a field whose power falls as f^-2
    h = w = 256
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    f = np.sqrt(fx * fx + fy * fy)
    f[0, 0] = 1.0
    field = np.real(np.fft.ifft2(np.exp(2j * np.pi * rng.random((h, w))) / f))
    assert G.spectrum_slope(field) == pytest.approx(-2.0, abs=0.15)
    im = _scene(rng, 240, 640)
    assert G.image_stats(cv2.GaussianBlur(im, (0, 0), 2.0))['sharpness'] < G.image_stats(im)['sharpness'] - 0.3
    st = G.image_stats(np.full((60, 80, 3), 255, np.uint8))
    assert st['bright_share'] == 1.0 and st['L_mean'] == pytest.approx(100.0, abs=0.01)


def test_frechet_and_kernel_distance_against_closed_forms():
    rng = np.random.default_rng(1)
    n = 200_000
    f1 = rng.normal(0, 1, (n, 2)) * [1.0, 2.0]
    f2 = rng.normal(0, 1, (n, 2)) * [2.0, 1.0] + [1.0, 0.0]
    # |Δμ|² + tr(S1 + S2 − 2 (S1 S2)^½) = 1 + 5 + 5 − 2 (2 + 2) = 3
    assert G.frechet(f1, f2) == pytest.approx(3.0, abs=0.05)
    a = rng.normal(0, 1, (400, 16))
    b = rng.normal(0, 1, (400, 16))
    kd, sd = G.kernel_distance(a, b, subsets=50)
    assert abs(kd) < 3 * sd + 1e-3 and sd > 0
    kd2, sd2 = G.kernel_distance(a, b + 0.5, subsets=50)
    assert kd2 > 10 * sd2
    gap = G.feature_gap(b + 0.5, np.concatenate([a, b]))
    assert gap['n_sim'] == gap['n_real'] == 400
    assert gap['fd'] > 10 * gap['fd_floor'] and gap['kd'] > 10 * abs(gap['kd_floor'])
    with pytest.raises(G.GapError):
        G.feature_gap(a, a[:15])


def test_features_need_a_fetched_extractor(tmp_path):
    with pytest.raises(G.GapError, match='fetch'):
        G.Features(str(tmp_path))
    p = G.features_path(str(tmp_path))
    os.makedirs(os.path.dirname(p))
    open(p, 'wb').write(b'not the weights')
    with pytest.raises(G.GapError, match='SHA-256'):
        G.Features(str(tmp_path))


@pytest.mark.skipif(not os.environ.get('FALCON_FEATURES_WORKSPACE'), reason='no fetched extractor '
                    '(FALCON_FEATURES_WORKSPACE); never downloaded by the tests')
def test_features_extract():
    fx = G.Features(os.environ['FALCON_FEATURES_WORKSPACE'])
    rng = np.random.default_rng(0)
    f = fx([_scene(rng) for _ in range(3)])
    assert f.shape == (3, 2048) and np.isfinite(f).all()


# ------------------------------------------------------------------------------------------- matching and study
def _sets(tmp_path, n=12):
    rng = np.random.default_rng(2)
    for i in range(n):
        _img(tmp_path / 'sim' / f'{i:03d}.png', _scene(rng, 240, 640, light=170))
        _img(tmp_path / 'real' / f'{i:03d}.png', _scene(rng, 240, 640, light=90, noise=2.0, blur=1.5))
    return G.read_set('folder', str(tmp_path / 'sim')), G.read_set('folder', str(tmp_path / 'real'))


def test_appearance_matching_moves_the_statistics(tmp_path):
    sim, real = _sets(tmp_path)
    ss = G.set_stats(sim, keep_images=True)
    rs = G.set_stats(real)
    p = G.fit_appearance(ss, rs)
    assert p['blur_sigma'] > 0 and p['noise_sigma'] > 0
    per = {k: [] for k in G.IMAGE_STATS}
    for i, im in enumerate(ss['images']):
        for k, v in G.image_stats(G.apply_appearance(im, p, i)).items():
            per[k].append(v)
    before = {r['stat']: r for r in G.compare(ss, rs)['appearance']}
    after = {r['stat']: r for r in G.compare({'per_image': per}, rs)['appearance']}
    for k in ('L_mean', 'sharpness', 'noise_sigma'):
        assert after[k]['w1'] < 0.5 * before[k]['w1'], k
    assert np.array_equal(G.apply_appearance(ss['images'][0], p, 7), G.apply_appearance(ss['images'][0], p, 7))


def test_study_document(tmp_path):
    sim, real = _sets(tmp_path, n=6)
    seen = []
    doc = G.study([sim], real, match=True, progress=lambda f, t: seen.append(f))
    assert doc['schema'] == G.SCHEMA and doc['features'] is None
    assert {r['stat'] for r in doc['gap']['appearance']} == set(G.IMAGE_STATS)
    shifts = [r['shift'] for r in doc['gap']['appearance'] if r['shift'] is not None]
    assert shifts == sorted(shifts, reverse=True)
    assert doc['gap']['appearance'][0]['stat'] in ('L_mean', 'noise_sigma', 'sharpness', 'spectrum_slope', 'L_std',
                                                   'a_mean', 'b_mean', 'saturation')
    assert doc['matched']['params']['noise_sigma'] > 0
    assert 'images' not in doc['sim'] and 'per_image' not in doc['real']
    assert seen[-1] == 1.0
    json.dumps(doc)


def test_gap_command_on_a_recorded_drive(project, tmp_path):
    """falcon gap on a drive recorded against the fake carla, against a labelled KITTI-layout set; then the same
    drive as both sides: its gaps are zero."""
    from falcon_autonomy.cli import main
    from test_workers import _cfg, _run
    cam = project.doc['rigs'][0]['sensors'][0]
    cam.update(labels=['semantic', 'instance', 'depth', 'boxes2d'], width=160, height=90)
    project.doc['traffic_profiles'][0]['vehicles'] = 6
    code, _, err = _run(_cfg(project, tmp_path / 'run', seconds=0.5, seed=2))
    assert code == P.OK, err
    s = G.read_set('falcon', str(tmp_path / 'run'))
    assert s['labelled'] and s['n_total'] == 10 and s['camera'] is None
    assert all(o['class'] in G.THINGS for it in s['items'] for o in it['objects'])
    rng = np.random.default_rng(3)
    (tmp_path / 'kitti' / 'label_2').mkdir(parents=True)
    for i in range(8):
        _img(tmp_path / 'kitti' / 'image_2' / f'{i:06d}.png', _scene(rng, 90, 160, light=80, noise=4))
        (tmp_path / 'kitti' / 'label_2' / f'{i:06d}.txt').write_text(
            'Car 0 0 0 10 20 60 60 1.5 1.6 4 1 1.6 15 0\nPedestrian 0 0 0 80 30 90 70 1.7 .6 .8 -2 1.6 9 0\n')
    out = tmp_path / 'gap'
    assert main(['gap', '--sim', str(tmp_path / 'run'), '--real', f'kitti_object:{tmp_path / "kitti"}', '--out',
                 str(out), '--match', '--seed', '4']) == 0
    doc = json.loads((out / 'gap.json').read_text())
    run = json.loads((out / 'run.json').read_text())
    assert run['status'] == 'ok' and 'gap.json' in run['outputs'] and run['provenance']['seed'] == 4
    assert 'sim0/run.json' in run['provenance']['inputs']
    obj = doc['gap']['objects']
    assert obj['class_share']['pedestrian']['real'] == pytest.approx(0.5)
    assert {r['stat'] for r in obj['rows']} == {'box_height_share', 'per_image', 'range_m'}
    assert doc['matched'] and doc['real']['kind'] == 'kitti_object'
    same = tmp_path / 'same'
    assert main(['gap', '--sim', str(tmp_path / 'run'), '--real', f'falcon:{tmp_path / "run"}', '--out',
                 str(same)]) == 0
    d2 = json.loads((same / 'gap.json').read_text())
    assert all(r['w1'] == pytest.approx(0, abs=1e-9) for r in d2['gap']['appearance'])
    assert d2['gap']['objects']['class_share_tvd'] == pytest.approx(0)
    assert main(['gap', '--sim', str(tmp_path / 'run'), '--real', 'folder', '--out', str(tmp_path / 'x')]) == 2
    assert main(['gap', '--sim', str(tmp_path / 'run'), '--real', f'folder:{tmp_path / "nothing"}', '--out',
                 str(tmp_path / 'y')]) == 2
    assert json.loads((tmp_path / 'y' / 'run.json').read_text())['status'] == 'failed'
    assert main(['gap', '--sim', str(tmp_path / 'run'), '--real', f'folder:{tmp_path}', '--out', str(tmp_path / 'z'),
                 '--features']) == 2
