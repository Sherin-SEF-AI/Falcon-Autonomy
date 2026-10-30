"""Semantic-occupancy exports (core.occexport): the camera ray cast on hand-made grids, the class mapping against the
nuScenes devkit's own, and Occ3D-nuScenes, OpenOcc v2 and SurroundOcc exports of a drive against the fake carla,
read back as their loaders read them (np.load, the keys, shapes, dtypes and value ranges they assume)."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from falcon_autonomy.core import exporters as E
from falcon_autonomy.core import occexport as O
from falcon_autonomy.core import protocol as P
from falcon_autonomy.core import worldmodel as WM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PY = os.environ.get('FALCON_DEVKITS_PYTHON') or os.path.join(ROOT, '.venv-devkits', 'bin', 'python')


# ------------------------------------------------------------------------------------------------- ray casting
def test_a_wall_hides_what_is_behind_it():
    g = WM.Grid(1.0, (0.0, 20.0), (-5.0, 5.0), (-2.0, 3.0))
    occ = np.zeros(g.shape, bool)
    occ[10, :, :] = True                                       # a wall across the grid at x 10–11 m
    dirs = O.camera_rays([[50.0, 0, 50.0], [0, 50.0, 25.0], [0, 0, 1]], 100, 50, np.eye(3), 0.01)
    seen = O.cast_to_first_occupied(g, occ, (0.5, 0.0, 0.5), dirs)
    assert seen[:10].any() and seen[10].any() and not seen[11:].any()
    # without the wall the rays run on to the edge of the grid
    assert O.cast_to_first_occupied(g, np.zeros(g.shape, bool), (0.5, 0.0, 0.5), dirs)[19].any()
    # a single voxel straight ahead: seen, the voxels right behind it on the axis not
    occ = np.zeros(g.shape, bool)
    occ[5, 5, 2] = True
    seen = O.cast_to_first_occupied(g, occ, (0.5, 0.5, 0.5), [(1.0, 0.0, 0.0)])     # voxel (0, 5, 2)
    assert seen[:6, 5, 2].all() and not seen[6:, 5, 2].any()


def test_rays_from_outside_enter_the_grid_and_camera_rays_point_forward():
    g = WM.Grid(1.0, (0.0, 10.0), (0.0, 10.0), (0.0, 10.0))
    seen = O.cast_to_first_occupied(g, np.zeros(g.shape, bool), (-5.0, 5.5, 5.5), [(1.0, 0.0, 0.0)])
    assert seen[:, 5, 5].all() and seen.sum() == 10
    assert not O.cast_to_first_occupied(g, np.zeros(g.shape, bool), (-5.0, 5.5, 5.5), [(-1.0, 0.0, 0.0)]).any()
    K = [[100.0, 0, 50.0], [0, 100.0, 25.0], [0, 0, 1]]
    d = O.camera_rays(K, 100, 50, np.eye(3), 0.5)              # f = 100 px: every 50th pixel
    assert len(d) == 2 and np.allclose(d[:, 1], [-0.495, 0.005]) and np.allclose(d[:, 0], 1.0)
    centre = O.camera_rays(K, 1, 1, np.eye(3), 0.0)            # the pixel at the principal point: straight ahead
    assert np.allclose(centre, [[1.0, (0.5 - 50.0) / 100.0, -(0.5 - 25.0) / 100.0]])


# ------------------------------------------------------------------------------------------------- classes
def test_class_mapping_follows_the_devkit():
    assert O.coarse_of_category('vehicle.car') == 4 and O.coarse_of_category('human.pedestrian.child') == 7
    assert O.coarse_of_category('animal') == 0 and O.coarse_of_category('vehicle.emergency.ambulance') == 0
    assert O.TAG_LUT[1] == O.COARSE.index('driveable_surface') and O.TAG_LUT[24] == O.TAG_LUT[1]
    assert O.TAG_LUT[9] == O.COARSE.index('vegetation') and O.TAG_LUT[3] == O.COARSE.index('manmade')
    assert O.TAG_LUT[0] == O.TAG_LUT[11] == -1                 # unlabelled and sky vote for nothing
    assert O.target_lut('openocc')[O.COARSE.index('car')] == 0 and O.target_lut('openocc')[0] == 16
    if not (os.path.isfile(VENV_PY) and subprocess.run([VENV_PY, '-c', 'import nuscenes'], capture_output=True,
                                                       timeout=60).returncode == 0):
        pytest.skip('the nuScenes devkit is not installed (.venv-devkits or FALCON_DEVKITS_PYTHON)')
    code = ('import json; from nuscenes.eval.lidarseg.utils import LidarsegClassMapper as M; m = M.__new__(M); '
            'm.ignore_class = M.get_ignore_class(); print(json.dumps([m.get_fine2coarse(), m.get_coarse2idx()]))')
    r = subprocess.run([VENV_PY, '-c', code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    f2c, c2i = json.loads(r.stdout.strip().splitlines()[-1])
    assert {k: ('others' if v == 'ignore' else v) for k, v in f2c.items()} == O.FINE2COARSE
    assert {('others' if k == 'ignore' else k): v for k, v in c2i.items()} == {c: i for i, c in enumerate(O.COARSE)}


# ------------------------------------------------------------------------------------------------- a drive
@pytest.fixture(scope='module')
def run_dir(tmp_path_factory):
    from test_worldmodel import _project
    from test_workers import _cfg, _run
    base = tmp_path_factory.mktemp('occ')
    project = _project(base / 'test.falcon.json')
    code, _, err = _run(_cfg(project, base / 'run', seconds=1.0, seed=3))
    assert code == P.OK, err
    return str(base / 'run')


def test_occ3d_export_reads_as_the_occ3d_loaders_read_it(run_dir, tmp_path):
    out = str(tmp_path / 'occ3d')
    rep = E.export([run_dir], 'occ3d', out, {'every': 2})
    ann = json.load(open(os.path.join(out, 'annotations.json')))
    scene = os.path.basename(run_dir)
    assert set(ann) >= {'train_split', 'val_split', 'scene_infos'} and ann['train_split'] == [scene]
    infos = ann['scene_infos'][scene]
    assert len(infos) == rep['frames_written'][scene] >= 4
    first = next(t for t, i in infos.items() if i['prev'] == '')
    one = infos[first]
    assert set(one) >= {'timestamp', 'camera_sensor', 'ego_pose', 'gt_path', 'next', 'prev'}
    cam = one['camera_sensor']['CAM_FRONT']
    assert np.asarray(cam['intrinsic']).shape == (3, 3) and len(cam['extrinsic']['rotation']) == 4
    assert os.path.isfile(os.path.join(out, cam['img_path']))
    for tok, info in infos.items():
        # what LoadOccGTFromFile and evaluate_miou do
        lab = np.load(os.path.join(out, info['gt_path']))
        sem, ml, mc = lab['semantics'], lab['mask_lidar'], lab['mask_camera']
        assert set(lab.files) == {'semantics', 'mask_lidar', 'mask_camera'}
        assert sem.shape == ml.shape == mc.shape == (200, 200, 16)
        assert sem.dtype == ml.dtype == mc.dtype == np.uint8
        assert sem.max() <= 17 and set(np.unique(ml)) <= {0, 1} and set(np.unique(mc)) <= {0, 1}
        assert (sem != 17).any() and ml.astype(bool).any() and mc.astype(bool).any()
        assert info['gt_path'] == f'gts/{scene}/{tok}/labels.npz'
    assert rep['coordinates']['grid']['shape'] == [200, 200, 16] and rep['coordinates']['free'] == 17
    assert any(x['what'] == 'class traffic_cone' for x in rep['left_out'])


def test_y_is_mirrored_and_boxes_carry_their_class(run_dir, tmp_path):
    """An actor's box to the ego's right (CARLA +y) lands at iy < 100 (the formats' y is left); its voxels hold its
    class; the camera sees the near side of it and not the voxels hidden behind."""
    data = WM.load_run(run_dir)
    sweeps = WM.lidar_sweeps(data, 'lid')
    k = len(sweeps) // 2
    g = O.sample_grids(data, 'lid', sweeps, k, 'occ3d', [])
    f = O.to_format(g, 'occ3d')
    step = sweeps[k][1]
    frame = next(fr for fr in data['frames'] if fr['step'] == step)
    T_ew = np.linalg.inv(WM.pose_matrix(frame['ego']))
    grid = g['grid']
    checked = 0
    for a in data['actors'][step]:
        if a['id'] == data['ego_id']:
            continue
        c, _, _ = WM.actor_box(a)
        e = T_ew[:3, :3] @ c + T_ew[:3, 3]
        ijk = grid.index(e)
        if not grid.inside(ijk) or abs(e[1]) < 2.0:
            continue
        ix, iy, iz = ijk
        assert f['semantics'][ix, 199 - iy, iz] == O.coarse_of_category(a['category'])
        assert (199 - iy < 100) == (e[1] > 0)
        checked += 1
    assert checked >= 1


def test_openocc_flow_and_surroundocc_sparse(run_dir, tmp_path):
    out = str(tmp_path / 'openocc')
    E.export([run_dir], 'occ3d', out, {'variant': 'openocc', 'every': 3})
    ann = json.load(open(os.path.join(out, 'annotations.json')))
    infos = next(iter(ann['scene_infos'].values()))
    moving = 0
    for info in infos.values():
        lab = np.load(os.path.join(out, info['gt_path']))          # what OccNet's loader reads
        sem, flow = lab['semantics'], lab['flow']
        assert info['gt_path'].startswith('openocc_v2/') and sem.shape == (200, 200, 16) and sem.dtype == np.uint8
        assert flow.shape == (200, 200, 16, 2) and flow.dtype == np.float32 and sem.max() <= 16
        assert np.all(flow[sem == 16] == 0)                         # free space does not move
        moving += int((np.linalg.norm(flow, axis=-1) > 0.5).sum())
    assert moving > 0                                               # the fake's traffic drives
    out = str(tmp_path / 'surroundocc')
    rep = E.export([run_dir], 'occ3d', out, {'variant': 'surroundocc', 'every': 3})
    infos = next(iter(json.load(open(os.path.join(out, 'annotations.json')))['scene_infos'].values()))
    for info in infos.values():
        occ = np.load(os.path.join(out, info['gt_path'])).astype(np.float32)   # SurroundOcc's LoadOccupancy
        assert occ.ndim == 2 and occ.shape[1] == 4 and len(occ)
        assert occ[:, :3].min() >= 0 and occ[:, 0].max() < 200 and occ[:, 1].max() < 200 and occ[:, 2].max() < 16
        assert set(np.unique(occ[:, 3]).astype(int)) <= set(range(17))
    assert rep['coordinates']['grid']['voxel_m'] == 0.5


def test_an_unknown_variant_and_a_run_without_lidar_are_refused(run_dir, tmp_path, project):
    with pytest.raises(E.ExportError, match='variant'):
        E.export([run_dir], 'occ3d', str(tmp_path / 'x'), {'variant': 'openoccupancy'})
    from test_workers import _cfg, _run
    project.doc['rigs'][0]['sensors'][0].update(width=64, height=36, labels=['semantic'])
    code, _, err = _run(_cfg(project, tmp_path / 'nolidar', seconds=0.3, seed=1))
    assert code == P.OK, err
    with pytest.raises(E.ExportError, match='nothing to export'):
        E.export([str(tmp_path / 'nolidar')], 'occ3d', str(tmp_path / 'y'))


def test_the_cli_exports_occupancy(run_dir, tmp_path):
    out = str(tmp_path / 'cli')
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy', 'export', run_dir, '--format', 'occ3d', '--out', out,
                        '--option', 'every=5', '--option', 'aggregate=1'], capture_output=True, text=True,
                       timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.load(open(os.path.join(out, 'falcon_export.json')))
    assert rep['format'] == 'occ3d' and rep['counts']['samples'] >= 2
    assert any(h['what'] == 'aggregation' for h in rep['heuristic'])
