"""Dataset exporters (core.exporters, core.mcap_bag) on drives the drive worker records against the fake carla.

Validated against the official devkits where they can be installed, by running them on the exported data: pycocotools
(COCO), nuscenes-devkit (nuScenes, -lidarseg, -panoptic), mcap + mcap-ros2-support (the bags). They are looked for in
this Python, then in the repository's .venv-devkits (or FALCON_DEVKITS_PYTHON); a check whose devkit is missing is
skipped with the reason, never faked. KITTI object and YOLO have no devkit to install: they are checked against their
published specs (KITTI: 15 label fields, calib keys P0–P3 R0_rect Tr_velo_to_cam Tr_imu_to_velo; YOLO: class cx cy w
h normalised to [0, 1]) and against the geometry (3D boxes through P2 cover the 2D boxes; lidar points of an actor
fall inside its box).

The runs: the test project's rig with the pinhole camera at 192 × 108 with every label, a 16-channel lidar, a radar,
GNSS and IMU, 8 traffic vehicles, 1 s at 20 fps; two seeds, one per split (train, val). Recorded once per module.
"""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import ROOT
from falcon_autonomy.core import exporters as E
from falcon_autonomy.core import groundtruth as GT
from falcon_autonomy.core import mcap_bag as B
from falcon_autonomy.core import protocol as P
from falcon_autonomy.core.provenance import sha256_file

cv2 = pytest.importorskip('cv2')

VENV_PY = os.environ.get('FALCON_DEVKITS_PYTHON') or os.path.join(ROOT, '.venv-devkits', 'bin', 'python')


# --------------------------------------------------------------------------------------------------------- fixtures
def _project(path):
    """The conftest `project` (module scope cannot use the function-scoped fixture), with the exporters' rig."""
    from falcon_autonomy.core.project import Project
    from falcon_autonomy.core.rig import new_sensor
    p = Project.new('Test')
    rig = p.doc['rigs'][0]
    rig['sensors'][0].update(width=192, height=108, labels=['semantic', 'instance', 'depth', 'boxes2d', 'boxes3d'])
    rig['sensors'].append(new_sensor('lidar', 'lid', 'top', {'x': 0.0, 'z': 1.9}))
    rig['sensors'][-1].update(channels=16, points_per_second=40000, range=60.0, upper_fov=5.0, lower_fov=-20.0)
    rig['sensors'].append(new_sensor('radar', 'rad', 'front_radar', {'x': 2.3, 'z': 0.6}))
    rig['sensors'].append(new_sensor('gnss', 'gnss', 'gnss'))
    rig['sensors'].append(new_sensor('imu', 'imu', 'imu'))
    p.doc['traffic_profiles'][0]['vehicles'] = 8
    p.save(str(path))
    return p


@pytest.fixture(scope='module')
def runs(tmp_path_factory):
    from test_workers import _cfg, _run
    base = tmp_path_factory.mktemp('drives')
    project = _project(base / 'test.falcon.json')
    out = {}
    for name, seed in (('run_a', 3), ('run_b', 4)):
        code, _, err = _run(_cfg(project, base / name, seconds=1.0, seed=seed))
        assert code == P.OK, err
        out[name] = str(base / name)
    return out


@pytest.fixture(scope='module')
def exported(runs, tmp_path_factory):
    """fmt -> (out dir, report): each format exported once, train = run_a, val = run_b."""
    base = tmp_path_factory.mktemp('exports')
    cache = {}

    def get(fmt):
        if fmt not in cache:
            out = str(base / fmt)
            rep = E.export([], fmt, out, {'splits': {'train': [runs['run_a']], 'val': [runs['run_b']]}})
            cache[fmt] = (out, rep)
        return cache[fmt]
    return get


_DEVKITS = {}


def devkit_python(*modules) -> str | None:
    """A Python that imports every module: this one, else the devkit venv's; None when neither does."""
    key = tuple(modules)
    if key not in _DEVKITS:
        code = 'import ' + ', '.join(modules)
        found = None
        for py in (sys.executable, VENV_PY):
            if py and os.path.isfile(py) and subprocess.run([py, '-P', '-c', code], capture_output=True,
                                                            timeout=60).returncode == 0:
                found = py
                break
        _DEVKITS[key] = found
    return _DEVKITS[key]


def run_devkit(modules, script: str, *args) -> dict:
    """Run a check script in a Python that has the devkit (skip when none); it prints one JSON line last."""
    py = devkit_python(*modules)
    if py is None:
        pytest.skip(f'devkit not installed ({", ".join(modules)}): neither this Python nor {VENV_PY} imports it')
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHON')}
    r = subprocess.run([py, '-P', '-c', script, *[str(a) for a in args]], capture_output=True, text=True, timeout=120,
                       env=env)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def _objects(run_dir, cam, step):
    with open(os.path.join(run_dir, 'objects', cam, f'{step:06d}.json')) as fh:
        return json.load(fh)['objects']


# ------------------------------------------------------------------------------------------------ manifest and API
@pytest.mark.parametrize('fmt', E.FORMATS)
def test_manifest_provenance_splits_and_left_out(exported, runs, fmt):
    out, rep = exported(fmt)
    with open(os.path.join(out, E.MANIFEST)) as fh:
        doc = json.load(fh)
    assert doc['schema'] == E.EXPORT_SCHEMA and doc['format'] == fmt and doc['files'] > 0
    assert doc['splits'] == {'train': ['run_a'], 'val': ['run_b']}
    ins = doc['provenance']['inputs']
    for name, d in runs.items():
        for f in ('run.json', 'dataset.json', 'frames.jsonl', 'lanes.json'):
            assert ins[f'{name}/{f}']['sha256'] == sha256_file(os.path.join(d, f))
    assert {r['key']: r['run_json_sha256'] for r in doc['runs']} == {
        n: sha256_file(os.path.join(d, 'run.json')) for n, d in runs.items()}
    assert doc['left_out'] and all(x['what'] and x['why'] for x in doc['left_out'])
    assert doc['coordinates'] and doc['load']
    assert rep['format'] == fmt and rep['counts'] == doc['counts']


def test_splits_out_dir_and_errors(runs, tmp_path):
    a, b = runs['run_a'], runs['run_b']
    with pytest.raises(E.ExportError, match='one split only'):
        E.export([], 'coco', str(tmp_path / 'x'), {'splits': {'train': [a], 'val': [a]}})
    with pytest.raises(E.ExportError, match='unknown split'):
        E.export([], 'coco', str(tmp_path / 'x'), {'splits': {'holdout': [a]}})
    with pytest.raises(E.ExportError, match='in no split'):
        E.export([b], 'coco', str(tmp_path / 'x'), {'splits': {'train': [a]}})
    with pytest.raises(E.ExportError, match='unknown format'):
        E.export([a], 'pascal', str(tmp_path / 'x'))
    with pytest.raises(E.ExportError, match='not a finished drive'):
        E.export([str(tmp_path)], 'coco', str(tmp_path / 'x'))
    (tmp_path / 'busy').mkdir()
    (tmp_path / 'busy' / 'mine.txt').write_text('x')
    with pytest.raises(E.ExportError, match='did not export'):
        E.export([a], 'coco', str(tmp_path / 'busy'))
    out = str(tmp_path / 'once')
    seen = []
    rep = E.export([a], 'yolo', out, {}, progress=lambda f, t: seen.append(f))
    assert rep['splits'] == {'all': ['run_a']} and seen[-1] == 1.0
    with pytest.raises(E.ExportError, match='earlier export'):
        E.export([a], 'yolo', out)
    assert E.export([a], 'yolo', out, {'overwrite': True})['frames_written'] == {'all': 20}


def test_main_export_cli(runs, tmp_path):
    sj = tmp_path / 'splits.json'
    sj.write_text(json.dumps({'train': [runs['run_a']], 'test': [runs['run_b']]}))
    assert E.main_export(['--format', 'kitti', '--out', str(tmp_path / 'k'), '--split-json', str(sj),
                          '--option', 'every=2']) == 0
    doc = json.loads((tmp_path / 'k' / E.MANIFEST).read_text())
    assert doc['provenance']['settings'] == {'every': 2} and set(doc['frames_written']) == {'training', 'testing'}
    assert (tmp_path / 'k' / 'testing' / 'ImageSets' / 'test.txt').exists()
    assert E.main_export([runs['run_a'], '--format', 'coco', '--out', str(tmp_path / 'k')]) == 2   # not empty
    assert E.main_export([str(tmp_path / 'nope'), '--format', 'coco', '--out', str(tmp_path / 'c')]) == 2
    assert E.main_export(['--format', 'gif', '--out', str(tmp_path / 'c')]) == 2


# ------------------------------------------------------------------------------------------------------------ COCO
def test_coco_structure_and_masks_round_trip(exported, runs):
    out, rep = exported('coco')
    coco = json.loads(open(os.path.join(out, 'annotations', 'instances_train.json')).read())
    assert {c['id'] for c in coco['categories']} == {c[0] for c in E.THING_CLASSES}
    assert len(coco['images']) == 20 and coco['annotations']
    imgs = {i['id']: i for i in coco['images']}
    for a in coco['annotations']:
        im = imgs[a['image_id']]
        assert os.path.isfile(os.path.join(out, 'images', 'train', im['file_name']))
        inst = cv2.imread(os.path.join(runs['run_a'], im['falcon']['instance']), cv2.IMREAD_UNCHANGED)
        m = GT.rle_decode(a['segmentation'])
        assert np.array_equal(m, inst == a['falcon']['object_id'])
        ys, xs = np.nonzero(m)
        assert a['bbox'] == [xs.min(), ys.min(), xs.max() + 1 - xs.min(), ys.max() + 1 - ys.min()]
        assert a['area'] == m.sum()


COCO_CHECK = r'''
import json, sys, numpy as np, cv2, pycocotools
from pycocotools.coco import COCO
ann_file, run_dir = sys.argv[1], sys.argv[2]
c = COCO(ann_file)
n = bad = 0
for img in c.loadImgs(c.getImgIds()):
    inst = cv2.imread(run_dir + '/' + img['falcon']['instance'], cv2.IMREAD_UNCHANGED)
    for a in c.loadAnns(c.getAnnIds(imgIds=[img['id']])):
        n += 1
        m = c.annToMask(a).astype(bool)
        bad += int(not np.array_equal(m, inst == a['falcon']['object_id']) or c.annToRLE(a)['size'] != list(inst.shape))
print(json.dumps({'images': len(c.getImgIds()), 'anns': n, 'bad': bad, 'cats': len(c.getCatIds()),
                  'version': getattr(pycocotools, '__version__', '?')}))
'''


def test_coco_with_pycocotools(exported, runs):
    out, rep = exported('coco')
    for split, run in (('train', 'run_a'), ('val', 'run_b')):
        r = run_devkit(('pycocotools', 'cv2'), COCO_CHECK, os.path.join(out, 'annotations', f'instances_{split}.json'),
                       runs[run])
        assert r['images'] == 20 and r['anns'] > 0 and r['bad'] == 0 and r['cats'] == len(E.THING_CLASSES)


# ------------------------------------------------------------------------------------------------------------ YOLO
def test_yolo_against_its_spec(exported):
    out, rep = exported('yolo')
    yaml = open(os.path.join(out, 'data.yaml')).read()
    assert 'train: images/train' in yaml and 'val: images/val' in yaml and f'nc: {len(E.THING_CLASSES)}' in yaml
    coco_out, _ = exported('coco')
    coco = json.loads(open(os.path.join(coco_out, 'annotations', 'instances_train.json')).read())
    per_img = {}
    for a in coco['annotations']:
        per_img.setdefault(a['image_id'], []).append(a)
    by_name = {i['file_name'][:-4]: (i, per_img.get(i['id'], [])) for i in coco['images']}
    labels = sorted(os.listdir(os.path.join(out, 'labels', 'train')))
    assert len(labels) == 20 and len(os.listdir(os.path.join(out, 'images', 'train'))) == 20
    n = 0
    for f in labels:
        img, anns = by_name[f[:-4]]
        rows = [x.split() for x in open(os.path.join(out, 'labels', 'train', f)).read().splitlines()]
        assert len(rows) == len(anns)
        for row, a in zip(rows, anns):
            assert len(row) == 5
            k, cx, cy, w, h = int(row[0]), *map(float, row[1:])
            assert 0 <= k < len(E.THING_CLASSES) and E.THING_CLASSES[k][0] == a['category_id']
            assert all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)) and w > 0 and h > 0
            x, y, bw, bh = a['bbox']
            assert cx == pytest.approx((x + bw / 2) / img['width'], abs=1e-6)
            assert h == pytest.approx(bh / img['height'], abs=1e-6)
            n += 1
    assert n > 0


# ----------------------------------------------------------------------------------------------------------- KITTI
CALIB_KEYS = {'P0': 12, 'P1': 12, 'P2': 12, 'P3': 12, 'R0_rect': 9, 'Tr_velo_to_cam': 12, 'Tr_imu_to_velo': 12}
KITTI_TYPES = {'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare'}


def _kitti_calib(path):
    out = {}
    for line in open(path).read().splitlines():
        k, _, v = line.partition(':')
        out[k] = np.array([float(x) for x in v.split()])
    return out


def _kitti_corners(h, w, l, x, y, z, ry):
    """The KITTI devkit's box corners (camera frame) of a label."""
    xs = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    ys = [0, 0, 0, 0, -h, -h, -h, -h]
    zs = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    R = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    return R @ np.array([xs, ys, zs]) + np.array([[x], [y], [z]])


def test_kitti_against_its_spec_and_geometry(exported):
    out, rep = exported('kitti')
    tr = os.path.join(out, 'training')
    names = sorted(f[:-4] for f in os.listdir(os.path.join(tr, 'label_2')))
    assert len(names) == 20                                 # 10 lidar sweeps per run, both runs (train + val)
    sets = {s: open(os.path.join(tr, 'ImageSets', f'{s}.txt')).read().split() for s in ('train', 'val')}
    assert sorted(sets['train'] + sets['val']) == names and len(sets['train']) == len(sets['val']) == 10
    covered = labels = inside = 0
    for n in names:
        cal = _kitti_calib(os.path.join(tr, 'calib', f'{n}.txt'))
        assert {k: len(v) for k, v in cal.items()} == CALIB_KEYS
        P2 = cal['P2'].reshape(3, 4)
        assert np.allclose(cal['R0_rect'].reshape(3, 3), np.eye(3))
        Tv = np.vstack([cal['Tr_velo_to_cam'].reshape(3, 4), [0, 0, 0, 1]])
        assert np.allclose(Tv[:3, :3] @ Tv[:3, :3].T, np.eye(3), atol=1e-9)
        img = cv2.imread(os.path.join(tr, 'image_2', f'{n}.png'))
        H, W = img.shape[:2]
        assert (W, H) == (192, 108)
        raw = open(os.path.join(tr, 'velodyne', f'{n}.bin'), 'rb').read()
        assert len(raw) % 16 == 0 and len(raw) > 0
        velo = np.frombuffer(raw, np.float32).reshape(-1, 4)
        pc = (Tv[:3, :3] @ velo[:, :3].T.astype(np.float64)).T + Tv[:3, 3]          # lidar points, camera frame
        for line in open(os.path.join(tr, 'label_2', f'{n}.txt')).read().splitlines():
            f = line.split()
            assert len(f) == 15 and f[0] in KITTI_TYPES
            trunc, occ, alpha = float(f[1]), int(f[2]), float(f[3])
            l_, t_, r_, b_, h, w, ln, x, y, z, ry = map(float, f[4:])
            assert 0 <= trunc <= 1 and occ in (0, 1, 2, 3)
            assert -math.pi <= alpha <= math.pi and -math.pi <= ry <= math.pi
            assert 0 <= l_ <= r_ <= W - 1 and 0 <= t_ <= b_ <= H - 1 and min(h, w, ln) > 0 and z > 0
            C = _kitti_corners(h, w, ln, x, y, z, ry)
            uv = P2 @ np.vstack([C, np.ones(8)])
            u, v = uv[0] / uv[2], uv[1] / uv[2]
            labels += 1
            # the 3D box through P2 covers the tight 2D box (both pixel-centre coordinates)
            if (np.clip(u.min(), 0, W - 1) <= l_ + 1.0 and np.clip(u.max(), 0, W - 1) >= r_ - 1.0 and
                    np.clip(v.min(), 0, H - 1) <= t_ + 1.0 and np.clip(v.max(), 0, H - 1) >= b_ - 1.0):
                covered += 1
            # lidar points in the box (camera frame, the box's own axes)
            R = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
            q = (pc - [x, y - h / 2, z]) @ R
            m = (np.abs(q[:, 0]) <= ln / 2 + 0.1) & (np.abs(q[:, 1]) <= h / 2 + 0.1) & (np.abs(q[:, 2]) <= w / 2 + 0.1)
            inside += int(m.any())
    assert labels > 10 and covered == labels and inside >= 0.5 * labels
    doc = rep
    assert any(x['what'] == 'P0, P1, P3' for x in doc['left_out'])


def test_kitti_rotation_convention(runs):
    """A car 10 m ahead heading 30° to the right (CARLA yaw +30) is KITTI rotation_y −60° in a forward camera:
    its heading (cos ry, 0, −sin ry) = (sin 30°, 0, cos 30°)."""
    run = E.Run(runs['run_a'])
    o = {'class': 'car', 'bbox': [80, 40, 120, 70], 'occluded': 0.6, 'truncated': 0.0,
         'box3d': {'center': [11.5, 0.0, 0.75], 'size': [4.0, 2.0, 1.5], 'yaw_deg': 30.0}}
    f = E._kitti_label(o, run, 'cam-front', 192, 108).split()
    assert float(f[14]) == pytest.approx(-math.radians(60), abs=0.01) and f[2] == '2'
    x, y, z = map(float, f[11:14])
    assert (x, y, z) == pytest.approx((0.0, 1.6, 10.0), abs=0.01)         # camera at x 1.5, z 1.6; bottom centre
    assert float(f[3]) == pytest.approx(float(f[14]) - math.atan2(x, z), abs=0.01)


# -------------------------------------------------------------------------------------------------------- nuScenes
def test_nuscenes_tables_reference_each_other(exported):
    out, rep = exported('nuscenes')
    assert rep['versions'] == ['v1.0-falcon-train', 'v1.0-falcon-val']
    for ver in rep['versions']:
        T = {f[:-5]: json.load(open(os.path.join(out, ver, f))) for f in os.listdir(os.path.join(out, ver))}
        tok = {k: {r['token'] for r in v} for k, v in T.items()}
        assert len(T['category']) == 32 and [c['index'] for c in T['category']] == list(range(32))
        assert len(T['sample']) == 10 and len(T['scene']) == 1 and len(T['lidarseg']) == 10
        assert all(r['sample_token'] in tok['sample'] and r['ego_pose_token'] in tok['ego_pose'] and
                   r['calibrated_sensor_token'] in tok['calibrated_sensor'] for r in T['sample_data'])
        assert all(os.path.isfile(os.path.join(out, r['filename'])) for r in T['sample_data'])
        assert all(a['instance_token'] in tok['instance'] and a['sample_token'] in tok['sample'] for a in
                   T['sample_annotation'])
        assert sum(i['nbr_annotations'] for i in T['instance']) == len(T['sample_annotation']) > 0
        sd = {r['token']: r for r in T['sample_data']}
        assert all(sd[r['sample_data_token']]['is_key_frame'] for r in T['lidarseg'])
        for r in T['lidarseg']:
            n = os.path.getsize(os.path.join(out, sd[r['sample_data_token']]['filename'])) // 20
            assert os.path.getsize(os.path.join(out, r['filename'])) == n


def test_nuscenes_rotation_convention():
    """CARLA yaw +30° (to the right) is −30° about z in nuScenes' right-handed frames."""
    q = E.quat_of(E.yaw_matrix(-math.radians(30.0)))
    assert 2 * math.degrees(math.atan2(q[3], q[0])) == pytest.approx(-30.0)
    T = E.rh(E.pose_matrix({'x': 1.0, 'y': 2.0, 'z': 3.0, 'roll': 0, 'pitch': 0, 'yaw': 30.0}))
    assert T[:3, 3].tolist() == [1.0, -2.0, 3.0]
    assert math.degrees(math.atan2(T[1, 0], T[0, 0])) == pytest.approx(-30.0)
    assert E.K_opencv([[96, 0, 96], [0, 96, 54], [0, 0, 1]])[:2, 2].tolist() == [95.5, 53.5]


NUSC_CHECK = r'''
import json, sys, numpy as np, nuscenes, importlib.metadata as md
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from nuscenes.utils.data_io import load_bin_file, panoptic_to_lidarseg
from nuscenes.utils.geometry_utils import view_points, points_in_box, BoxVisibility
root, version, expected = sys.argv[1], sys.argv[2], json.load(open(sys.argv[3]))
nusc = NuScenes(version=version, dataroot=root, verbose=False)
res = {'samples': 0, 'boxes': 0, 'covered': 0, 'lidar_sweeps': 0, 'seg_ok': 0, 'pan_ok': 0, 'box_pts': 0,
       'box_pts_ok': 0, 'radar_pts': 0, 'on_map': 0, 'egos': 0, 'devkit': md.version('nuscenes-devkit')}
for sample in nusc.sample:
    res['samples'] += 1
    for chan, sd in sample['data'].items():
        path, boxes, K = nusc.get_sample_data(sd, box_vis_level=BoxVisibility.ANY)
        rec = nusc.get('sample_data', sd)
        if rec['sensor_modality'] == 'camera':
            exp = expected.get(rec['filename'], {})
            for b in boxes:
                C = b.corners()
                inst = nusc.get('sample_annotation', b.token)['instance_token']
                if inst not in exp or C[2].min() < 0.1:
                    continue
                uv = view_points(C, np.array(K), normalize=True)[:2] + 0.5          # pixel-edge coordinates
                x0, y0, x1, y1 = exp[inst]
                res['boxes'] += 1
                W, H = rec['width'], rec['height']
                lo, hi = np.clip(uv.min(1), 0, [W, H]), np.clip(uv.max(1), 0, [W, H])
                res['covered'] += int(lo[0] <= x0 + 1 and lo[1] <= y0 + 1 and hi[0] >= x1 - 1 and hi[1] >= y1 - 1)
        elif rec['sensor_modality'] == 'lidar':
            pc = LidarPointCloud.from_file(path)
            seg = load_bin_file(nusc.dataroot + '/' + nusc.get('lidarseg', sd)['filename'], type='lidarseg')
            pan = load_bin_file(nusc.dataroot + '/' + nusc.get('panoptic', sd)['filename'], type='panoptic')
            res['lidar_sweeps'] += 1
            res['seg_ok'] += int(len(seg) == pc.points.shape[1])
            res['pan_ok'] += int(len(pan) == len(seg) and (panoptic_to_lidarseg(pan) == seg).all())
            for b in boxes:
                m = points_in_box(b, pc.points[:3])
                if m.sum() < 5:
                    continue
                idx = nusc.lidarseg_name2idx_mapping[b.name]
                res['box_pts'] += int(m.sum())
                res['box_pts_ok'] += int((seg[m] == idx).sum())
        elif rec['sensor_modality'] == 'radar':
            res['radar_pts'] += RadarPointCloud.from_file(path).points.shape[1]
for log in nusc.log:
    mask = nusc.get('map', log['map_token'])['mask']
    scene = [s for s in nusc.scene if s['log_token'] == log['token']][0]
    poses = [nusc.get('ego_pose', nusc.get('sample_data', s['data']['LIDAR_TOP'])['ego_pose_token'])
             for s in nusc.sample if s['scene_token'] == scene['token']]
    xy = np.array([p['translation'][:2] for p in poses])
    res['on_map'] += int(mask.is_on_mask(xy[:, 0], xy[:, 1]).sum())
    res['egos'] += len(xy)
nusc.list_lidarseg_categories(sort_by='index')
print(json.dumps(res))
'''


def test_nuscenes_with_the_devkit(exported, runs, tmp_path):
    """The official devkit loads each split's version, every sample's data through get_sample_data, the lidarseg and
    panoptic files through its own loaders; its boxes projected into the camera cover Falcon's 2D boxes, the lidar
    points in its boxes carry the boxes' categories, and the ego drives on the map mask."""
    out, rep = exported('nuscenes')
    for ver, run_key in (('v1.0-falcon-train', 'run_a'), ('v1.0-falcon-val', 'run_b')):
        run = E.Run(runs[run_key])
        sd = json.load(open(os.path.join(out, ver, 'sample_data.json')))
        step_of = {run.us(r, 'cam-front'): r['step'] for r in run.frames}
        expected = {}
        for r in sd:
            if '/CAM_FRONT/' in r['filename'] and r['is_key_frame']:
                step = step_of[r['timestamp']]
                expected[r['filename']] = {E.token('falcon', 'instance', run_key, str(o['track_id'])): o['bbox']
                                           for o in _objects(runs[run_key], 'cam-front', step)}
        ej = tmp_path / f'{ver}.json'
        ej.write_text(json.dumps(expected))
        r = run_devkit(('nuscenes', 'pyquaternion'), NUSC_CHECK, out, ver, ej)
        assert r['samples'] == 10 and r['lidar_sweeps'] == 10
        assert r['seg_ok'] == r['pan_ok'] == 10
        assert r['boxes'] > 5 and r['covered'] == r['boxes']
        assert r['box_pts'] > 20 and r['box_pts_ok'] >= 0.95 * r['box_pts']
        assert r['radar_pts'] > 0 and r['on_map'] == r['egos'] == 10


def test_lidarseg_and_panoptic_labels(runs):
    """Every lidar point's category: its actor's when a camera labelled the actor, else its tag's; things carry one
    instance number per actor."""
    run = E.Run(runs['run_a'])
    rec = next(r for r in run.frames if run.file(r, 'lid', 'lidar'))
    pts, tags, ids = run.lidar(rec, 'lid')
    M = run.mount('lid')
    Pv = pts[:, :3] @ M[:3, :3].T + M[:3, 3]
    seg, pan = E.lidar_labels(run, tags, ids, Pv, {})
    c, size = E._ego_box(run)
    inside = GT.in_box(Pv, c, size, np.eye(3)) & (ids > 0)   # the fake lidar hits its own car, as CARLA's can
    ego = ids == np.bincount(ids[inside]).argmax()
    assert (~inside & ego).any()                              # motion compensation moved some out of its box
    assert (seg[ego] == GT.NUSC_INDEX['vehicle.ego']).all() and (pan[ego] == 31000).all()
    cars = (tags == 14) & (ids > 0) & ~ego
    assert cars.any() and (seg[cars] == GT.NUSC_INDEX['vehicle.car']).all()
    assert (seg[tags == 1] == GT.NUSC_INDEX['flat.driveable_surface']).all()
    for aid in np.unique(ids[cars]):
        assert len(np.unique(pan[ids == aid])) == 1 and pan[ids == aid][0] % 1000 > 0
    assert (pan[tags == 1] == GT.NUSC_INDEX['flat.driveable_surface'] * 1000).all()


# ----------------------------------------------------------------------------------------------------------- Waymo
def test_waymo_style_layout(exported, runs):
    out, rep = exported('waymo')
    assert any('TFRecord' in x['what'] for x in rep['left_out'])
    seg = os.path.join(out, 'training', 'segment-run_a')
    ctx = json.load(open(os.path.join(seg, 'context.json')))
    cam = ctx['camera_calibrations'][0]
    assert cam['name'] == 1 and cam['intrinsic'][:4] == pytest.approx([96.0, 96.0, 95.5, 53.5])
    assert np.reshape(cam['extrinsic']['transform'], (4, 4))[:3, 3].tolist() == [1.5, 0.0, 1.6]
    lc = ctx['laser_calibrations'][0]
    assert len(lc['beam_inclinations']) == 16 and lc['beam_inclination_max'] == pytest.approx(math.radians(5.0))
    frames = [json.loads(x) for x in open(os.path.join(seg, 'frames.jsonl'))]
    assert len(frames) == 20 and sum(1 for f in frames if f['lasers']) == 10
    f = next(f for f in frames if f['lasers'] and f['laser_labels'])
    objs = {str(o['track_id']): o for o in _objects(runs['run_a'], 'cam-front', f['falcon_step'])}
    for lab in f['laser_labels']:
        b = objs[lab['id']]['box3d']
        assert [lab['box']['center_x'], lab['box']['center_y']] == pytest.approx([b['center'][0], -b['center'][1]])
        assert lab['box']['heading'] == pytest.approx(E._wrap(-math.radians(b['yaw_deg'])))
        assert lab['type'] in (1, 2, 3, 4)
    pts = np.load(os.path.join(seg, f['lasers'][0]['points']))
    assert pts.shape[1] == 4 and pts.dtype == np.float32
    assert any(lab['num_lidar_points_in_box'] > 0 for lab in f['laser_labels'])
    assert all(os.path.isfile(os.path.join(seg, im['image'])) for im in f['images'])
    assert os.path.isdir(os.path.join(out, 'validation', 'segment-run_b'))


# ------------------------------------------------------------------------------------------------------- BDD lanes
BDD_CATEGORIES = {'crosswalk', 'double other', 'double white', 'double yellow', 'road curb', 'single other',
                  'single white', 'single yellow'}


def test_bdd_lanes(exported, runs):
    """BDD100K lane labels: the map's lane boundaries in each frame; their vertices lie on the ground (road, road line,
    sidewalk pixels of Falcon's semantic image)."""
    out, rep = exported('bdd_lanes')
    frames = json.load(open(os.path.join(out, 'labels', 'lane_train.json')))
    assert len(frames) == 20
    on = total = 0
    cats = set()
    for fr in frames:
        assert os.path.isfile(os.path.join(out, 'images', 'train', fr['name']))
        sem = cv2.imread(os.path.join(runs['run_a'], 'semantic', 'cam-front', f'{fr["frameIndex"]:06d}.png'),
                         cv2.IMREAD_UNCHANGED)
        for lab in fr['labels']:
            assert lab['category'] in BDD_CATEGORIES and lab['attributes']['laneStyle'] in ('solid', 'dashed')
            cats.add(lab['category'])
            for p in lab['poly2d']:
                assert p['closed'] is False and len(p['types']) == len(p['vertices']) >= 2
                for x, y in p['vertices']:
                    assert 0 <= x <= 192 and 0 <= y <= 108
                    total += 1
                    on += int(sem[min(int(y), 107), min(int(x), 191)] in (1, 2, 24))
    assert total > 50 and on >= 0.9 * total
    assert {'double yellow', 'single white'} <= cats                  # the fake town's centre line and lane lines


def test_lane_categories():
    assert E.lane_category('SolidSolid', 'Yellow') == ('double yellow', 'solid')
    assert E.lane_category('Broken', 'Standard') == ('single white', 'dashed')
    assert E.lane_category('Curb', 'White') == ('road curb', 'solid')
    assert E.lane_category('NONE', 'White') == (None, None)


# ------------------------------------------------------------------------------------------------------------ MCAP
def _expected_topics(run_dir):
    frames = [json.loads(x) for x in open(os.path.join(run_dir, 'frames.jsonl'))]
    n = len(frames)
    count = lambda key: sum(1 for f in frames if key in f['files'])                           # noqa: E731
    lines = lambda kind, sid: len(open(os.path.join(run_dir, kind, f'{sid}.jsonl')).read().splitlines())  # noqa
    return {'/tf_static': ('tf2_msgs/msg/TFMessage', 1), '/tf': ('tf2_msgs/msg/TFMessage', n),
            '/front/image/compressed': ('sensor_msgs/msg/CompressedImage', count('cam-front/rgb')),
            '/front/camera_info': ('sensor_msgs/msg/CameraInfo', count('cam-front/rgb')),
            '/top/points': ('sensor_msgs/msg/PointCloud2', count('lid/lidar')),
            '/front_radar/detections': ('sensor_msgs/msg/PointCloud2', count('rad/radar')),
            '/imu/imu': ('sensor_msgs/msg/Imu', lines('imu', 'imu')),
            '/gnss/fix': ('sensor_msgs/msg/NavSatFix', lines('gnss', 'gnss'))}


def test_mcap_with_the_mcap_reader(exported, runs):
    mcap_reader = pytest.importorskip('mcap.reader', reason='the mcap package is not installed')
    out, rep = exported('mcap')
    for split, name in (('train', 'run_a'), ('val', 'run_b')):
        bag = os.path.join(out, split, name)
        meta = open(os.path.join(bag, 'metadata.yaml')).read()
        assert 'storage_identifier: mcap' in meta and f'relative_file_paths:\n    - {name}_0.mcap' in meta
        with open(os.path.join(bag, f'{name}_0.mcap'), 'rb') as fh:
            r = mcap_reader.make_reader(fh)
            assert r.get_header().profile == 'ros2'
            s = r.get_summary()
            got = {c.topic: (s.schemas[c.schema_id].name, s.statistics.channel_message_counts[c.id])
                   for c in s.channels.values()}
            assert all(c.message_encoding == 'cdr' for c in s.channels.values())
            assert all(sc.encoding == 'ros2msg' for sc in s.schemas.values())
            n = sum(1 for _ in r.iter_messages())
        exp = _expected_topics(runs[name])
        assert got == exp and n == sum(v[1] for v in exp.values()) == rep['bags'][f'{split}/{name}']['messages']
        assert exp['/top/points'][1] == 10 and exp['/imu/imu'][1] >= 19 and exp['/gnss/fix'][1] >= 9


MCAP_CHECK = r'''
import json, sys, numpy as np, importlib.metadata as md
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
path = sys.argv[1]
out = {'types': {}, 'tf_static': [], 'clouds': [], 'imu': [], 'fix': [], 'images': [], 'info': [],
       'versions': {p: md.version(p) for p in ('mcap', 'mcap-ros2-support')}}
with open(path, 'rb') as fh:
    for schema, ch, msg, d in make_reader(fh, decoder_factories=[DecoderFactory()]).iter_decoded_messages():
        out['types'][ch.topic] = schema.name
        if ch.topic == '/tf_static':
            out['tf_static'] = [[t.header.frame_id, t.child_frame_id, t.transform.translation.x,
                                 t.transform.translation.y, t.transform.translation.z] for t in d.transforms]
        elif ch.topic == '/top/points':
            out['clouds'].append([d.header.stamp.sec * 10**9 + d.header.stamp.nanosec, d.header.frame_id, d.width,
                                  d.point_step, [f.name for f in d.fields]])
        elif ch.topic == '/imu/imu' and not out['imu']:
            out['imu'] = [d.linear_acceleration.x, d.linear_acceleration.y, d.linear_acceleration.z,
                          d.angular_velocity.x, d.angular_velocity.y, d.angular_velocity.z,
                          d.orientation_covariance[0]]
        elif ch.topic == '/gnss/fix' and not out['fix']:
            out['fix'] = [d.latitude, d.longitude, d.altitude, d.status.status]
        elif ch.topic == '/front/image/compressed':
            out['images'].append([d.format, len(d.data), d.header.frame_id])
        elif ch.topic == '/front/camera_info' and not out['info']:
            out['info'] = [d.width, d.height, list(d.k), d.distortion_model]
print(json.dumps(out))
'''


def test_mcap_with_the_ros2_decoder(exported, runs):
    out, rep = exported('mcap')
    d = runs['run_a']
    r = run_devkit(('mcap', 'mcap_ros2.decoder'), MCAP_CHECK, os.path.join(out, 'train', 'run_a', 'run_a_0.mcap'))
    assert r['types'] == {k: v[0] for k, v in _expected_topics(d).items()}
    tf = {(p, c): xyz for p, c, *xyz in r['tf_static']}
    assert tf[('base_link', 'top_link')] == pytest.approx([0.0, 0.0, 1.9])
    assert tf[('base_link', 'front_radar_link')] == pytest.approx([2.3, 0.0, 0.6])
    assert ('front_link', 'front_optical') in tf
    frames = [json.loads(x) for x in open(os.path.join(d, 'frames.jsonl'))]
    sweeps = [f for f in frames if 'lid/lidar' in f['files']]
    assert [c[2] for c in r['clouds']] == [os.path.getsize(os.path.join(d, f['files']['lid/lidar'])) // 16
                                           for f in sweeps]
    assert r['clouds'][0][1] == 'top_link' and r['clouds'][0][4] == ['x', 'y', 'z', 'intensity', 'tag', 'actor_id']
    imu = json.loads(open(os.path.join(d, 'imu', 'imu.jsonl')).readline())
    a, g = imu['accel'], imu['gyro']
    assert r['imu'] == pytest.approx([a[0], -a[1], a[2], -g[0], g[1], -g[2], -1.0])
    fix = json.loads(open(os.path.join(d, 'gnss', 'gnss.jsonl')).readline())
    assert r['fix'] == pytest.approx([fix['lat'], fix['lon'], fix['alt'], 0])
    jpgs = [os.path.getsize(os.path.join(d, f['files']['cam-front/rgb'])) for f in frames]
    assert [x[1] for x in r['images']] == jpgs and r['images'][0][0] == 'jpeg'
    assert r['info'][:2] == [192, 108] and r['info'][2][2] == pytest.approx(95.5) and r['info'][3] == 'plumb_bob'


def test_cdr_encoding_is_plain_cdr():
    """XCDR1 little-endian: header 00 01 00 00; Time (int32, uint32); a string's length counts its NUL; float64
    aligned to 8 from the payload's start."""
    b = B.encode('std_msgs/msg/Header', {'stamp': {'sec': 1, 'nanosec': 2}, 'frame_id': 'ab'})
    assert b == bytes([0, 1, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]) + b'ab\x00'
    v = B.encode('geometry_msgs/msg/TransformStamped', {'header': {'frame_id': 'a'}, 'child_frame_id': 'b'})
    # header: 8 (time) + 4 + 2 ('a\0'); child: pad to 16, 4 + 2 -> 22; doubles from 24
    assert len(v) == 4 + 24 + 7 * 8 and v[4 + 24 + 48:4 + 24 + 56] == np.float64(1.0).tobytes()    # rotation w = 1
    with pytest.raises(ValueError, match='no field'):
        B.encode('std_msgs/msg/Header', {'stamp': {}, 'frameid': 'x'})
    assert 'MSG: geometry_msgs/Quaternion' in B.schema_text('sensor_msgs/msg/Imu')
