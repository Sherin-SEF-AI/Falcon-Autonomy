"""Fault tolerance campaigns (core.faulttol): the search (bisection, the monotonicity check), the graded fault models
(determinism, common random numbers, the rotation homography against the mount convention), an offline campaign on a
recorded fake-carla drive — the model-free recall criterion, a detector's AP drop with a blob detector that really
reads the image, the lidar retention — a closed-loop campaign through study_service on the fake carla, the result
document, its GSN evidence and the command line."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import FAKE, worker_env
from falcon_autonomy.core import faults as F
from falcon_autonomy.core import faulttol as FT
from falcon_autonomy.core import protocol as P

BLOB = {'id': 'blob', 'name': 'Blob', 'kind': 'detector', 'runtime': 'python', 'entry': 'test_faulttol:BlobDetector',
        'classes': ['car'], 'licence': 'GPL-3.0-or-later'}


class BlobDetector:
    """Finds bright rectangles on a dark image (the fixture paints every labelled box white): it reads pixels only."""

    def predict(self, rgb, meta):
        import cv2
        m = (rgb.max(axis=2) > 128).astype(np.uint8)
        n, _lab, st, _c = cv2.connectedComponentsWithStats(m, 8)
        return [{'bbox': [float(x), float(y), float(x + w), float(y + h)], 'class': 'car', 'score': 0.9}
                for x, y, w, h, a in st[1:] if a >= 12]


# ------------------------------------------------------------------------------------------------------ the search
def test_bisection_brackets_a_planted_threshold():
    calls = []

    def ev(x):
        calls.append(x)
        return 10.0 * x                                   # fails above 0.37 (threshold 3.7)
    r = FT.tolerance(ev, 3.7, 0.0, 1.0, grid=5, steps=8)
    assert r['status'] == 'bracketed' and r['monotone']
    lo, hi = r['bracket']
    assert lo <= 0.37 < hi and hi - lo < 0.25 / 2 ** 7 + 1e-9 and r['tolerated'] == lo
    assert len(calls) == len(set(calls)) == r['evaluations']            # every severity evaluated once
    assert [c[0] for c in r['curve']] == sorted(calls)


def test_no_failure_a_nominal_failure_and_a_response_that_is_not_monotone():
    r = FT.tolerance(lambda x: 0.0, 1.0, 0.0, 2.0)
    assert r['status'] == 'tolerates_max' and r['tolerated'] == 2.0 and r['bracket'] == [2.0, None]
    r = FT.tolerance(lambda x: 5.0, 1.0, 0.0, 2.0)
    assert r['status'] == 'fails_nominal' and r['tolerated'] is None and r['evaluations'] == 1
    # fails at 0.25 and 0.5, passes again at 0.75: the lowest failing grid point is reported, nothing bisected
    r = FT.tolerance(lambda x: 2.0 if 0.2 < x < 0.6 or x > 0.9 else 0.0, 1.0, 0.0, 1.0, grid=5)
    assert r['status'] == 'not_monotone' and not r['monotone'] and r['bracket'] == [0.0, 0.25]
    assert r['evaluations'] == 5 and 'not monotone' in r['note']
    # resolution stops the bisection
    r = FT.tolerance(lambda x: x, 0.4, 0.0, 1.0, grid=3, steps=20, resolution=0.1)
    assert r['bracket'][1] - r['bracket'][0] <= 0.1 + 1e-9 and r['evaluations'] <= 7


# ------------------------------------------------------------------------------------------------ the fault models
def test_the_mount_rotation_is_carlas_and_the_homography_moves_a_box_the_right_way():
    from falcon_autonomy.core.exporters import carla_rotation
    for deg in ([0, 0, 7], [0, 4, 0], [3, 0, 0], [2, -5, 11]):
        assert np.allclose(F.mount_rotation_matrix(deg), carla_rotation(*deg), atol=1e-12)
    w, h, fov = 640, 360, 90.0
    f = w / (2 * np.tan(np.radians(fov / 2)))
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    H = FT.misalign_homography(K, [0, 0, 2.0])                 # turned 2° to the right: the scene moves left
    u = H @ np.array([w / 2, h / 2, 1.0])
    assert u[0] / u[2] == pytest.approx(w / 2 - f * np.tan(np.radians(2.0)), abs=1e-6)
    assert u[1] / u[2] == pytest.approx(h / 2, abs=1e-6)
    box = [300.0, 160.0, 340.0, 200.0]
    moved = FT.misalign_box(box, H, w, h)
    assert moved[0] < box[0] and FT.iou(box, moved) < FT.iou(box, FT.misalign_box(box, FT.misalign_homography(
        K, [0, 0, 0.5]), w, h)) < 1.0
    assert FT.misalign_box(box, FT.misalign_homography(K, [0, 0, 60.0]), w, h) is None       # left the image
    img = np.zeros((h, w, 3), np.uint8)
    img[160:200, 300:340] = 255
    out = FT.misalign_image(img, K, [0, 0, 2.0])
    xs = np.nonzero(out[180].max(axis=1) > 128)[0]
    assert xs.min() == pytest.approx(moved[0], abs=1.5) and xs.max() == pytest.approx(moved[2], abs=1.5)
    # a lidar turned to the right sees a point straight ahead on its left
    q = F.rotate_points(np.array([[10.0, 0.0, 0.0, 1.0]], np.float32), [0, 0, 5.0])
    assert q[0, 1] < 0 and np.linalg.norm(q[0, :3]) == pytest.approx(10.0, abs=1e-4) and q[0, 3] == 1.0


def test_timing_faults_are_seeded_and_nested_in_the_severity():
    t = [0.05 * k for k in range(40)]

    def idx(fault, sev):
        return FT.delivered_index(fault, sev, t, FT.cell_rng(3, 'cam', fault, 0))
    assert idx('dropped', 0.3) == idx('dropped', 0.3)
    low, high = idx('dropped', 0.2), idx('dropped', 0.5)
    lost_low = {k for k, j in enumerate(low) if j != k}
    lost_high = {k for k, j in enumerate(high) if j != k}
    assert lost_low and lost_low < lost_high                    # a frame lost at 0.2 is lost at 0.5 too
    assert idx('clock_offset', 100.0) == [max(k - 2, 0) for k in range(40)]
    assert idx('clock_offset', 0.0) == list(range(40))
    a, b = idx('frozen', 0.2), idx('frozen', 0.5)
    fa, fb = [k for k, j in enumerate(a) if j != k], [k for k, j in enumerate(b) if j != k]
    assert fa and set(fa) < set(fb) and fa[0] == fb[0] and len({a[k] for k in fa}) == 1
    img = np.full((40, 60, 3), 100, np.uint8)
    assert int(FT.exposure(img, 1.0)[0, 0, 0]) > 100 and int(FT.exposure(img, 5.0)[0, 0, 0]) == 255


def test_stream_faults_drift_and_bias_validate_and_apply():
    assert F.validate([{'kind': 'extrinsic_drift', 'deg': [0, 0, 2.0], 'onset_s': 1.0},
                       {'kind': 'bias', 'north_m': 3.0, 'onset_s': 0.5}]) == []
    assert F.validate([{'kind': 'extrinsic_drift'}]) and F.validate([{'kind': 'bias', 'north_m': 500}])
    assert F.applies_to('extrinsic_drift', 'lidar') and not F.applies_to('extrinsic_drift', 'gnss')
    assert F.applies_to('bias', 'gnss') and not F.applies_to('bias', 'lidar')
    pts = np.array([[10.0, 0.0, 0.0, 1.0], [0.0, 5.0, 1.0, 1.0]], np.float32)
    fl = [{'kind': 'extrinsic_drift', 'deg': [0, 0, 90.0], 'onset_s': 1.0}]
    assert F.apply_points(pts, fl, 0.5, None) is pts or np.array_equal(F.apply_points(pts, fl, 0.5, None), pts)
    out = F.apply_points(pts, fl, 1.5, None)
    assert np.allclose(out[0, :3], [0.0, -10.0, 0.0], atol=1e-4) and np.allclose(out[1, :3], [5.0, 0.0, 1.0], atol=1e-4)
    assert F.gnss_offset([{'kind': 'bias', 'north_m': 3.0, 'east_m': -1.0, 'onset_s': 0.5}], 0.2, 0) == (0.0, 0.0)
    assert F.gnss_offset([{'kind': 'bias', 'north_m': 3.0, 'east_m': -1.0, 'onset_s': 0.5}], 0.7, 0) == (3.0, -1.0)


# ------------------------------------------------------------------------------------------------ offline campaign
@pytest.fixture(scope='module')
def recorded(tmp_path_factory):
    """A fake-carla drive with a labelled camera and a lidar; then every image repainted: black, with each labelled
    box white — so the blob detector is exact on the clean frames and every image fault really costs it."""
    import cv2
    from falcon_autonomy.core.exporters import Run
    from falcon_autonomy.core.project import Project
    from test_workers import _cfg, _run
    base = tmp_path_factory.mktemp('ft')
    p = Project.new('FT')
    rig = p.doc['rigs'][0]
    rig['sensors'][0].update(width=320, height=180, labels=['boxes2d'])
    rig['sensors'].append({'id': 'lid', 'name': 'top', 'type': 'lidar', 'pose': {'x': 0.0, 'y': 0.0, 'z': 2.0,
                           'roll': 0, 'pitch': 0, 'yaw': 0}, 'channels': 16, 'range': 60.0,
                           'points_per_second': 60000, 'rotation_frequency': 20.0, 'upper_fov': 5.0,
                           'lower_fov': -20.0, 'horizontal_fov': 360.0, 'noise_stddev': 0.0})
    p.doc['traffic_profiles'][0]['vehicles'] = 10
    p.doc['models'] = [BLOB]
    p.save(str(base / 'ft.falcon.json'))
    code, _, err = _run(_cfg(p, base / 'a', seconds=1.5, seed=3))
    assert code == P.OK, err
    run = Run(str(base / 'a'))
    cid = next(iter(run.cams))
    for rec in run.camera_steps(cid):
        img = np.zeros((180, 320, 3), np.uint8)
        objs = run.objects(rec, cid) or []
        for o in objs:
            o['class'] = 'car'
            x0, y0, x1, y1 = (int(round(v)) for v in o['bbox'])
            img[y0:y1, x0:x1] = 255
        cv2.imwrite(run.p(run.file(rec, cid, 'rgb')), img)
    return {'project': p, 'path': str(base / 'ft.falcon.json'), 'run': str(base / 'a'), 'cam': cid, 'base': base}


def _cells(cells):
    return {(c['sensor'], c['fault']): c for c in cells}


def test_offline_without_a_model_recall_and_lidar_retention(recorded):
    cells = FT.offline_campaign([recorded['run']], seed=1)
    by = _cells(cells)
    cam = recorded['cam']
    for f in ('cam_yaw', 'cam_pitch', 'cam_roll'):
        c = by[(cam, f)]
        assert c['criterion'] == 'recall_drop' and c['status'] in ('bracketed', 'tolerates_max'), c
        assert c['curve'][0][:2] == [0.0, 0.0] and c['monotone']
    yaw = by[(cam, 'cam_yaw')]
    assert yaw['status'] == 'bracketed' and 0 < yaw['bracket'][0] < yaw['bracket'][1] <= 5.0
    assert yaw['bracket'][1] - yaw['bracket'][0] <= 1.25 / 2 ** 6 + 1e-9
    for f in ('dirty_lens', 'noise', 'exposure'):                  # refused by name: they need a detector
        assert by[(cam, f)]['status'] == 'refused' and '--model' in by[(cam, f)]['note']
    assert by[(cam, 'clock_offset')]['resolution'] == pytest.approx(50.0)
    ly = by[('lid', 'lidar_yaw')]
    assert ly['criterion'] == 'retention_drop' and ly['status'] in ('bracketed', 'not_monotone')
    sec = by[('lid', 'lidar_sector')]
    assert sec['monotone'] and sec['curve'][-1][1] == pytest.approx(50.0, abs=25.0)     # half the circle gone
    # deterministic: the same campaign gives the same cells
    again = _cells(FT.offline_campaign([recorded['run']], sensors=[cam], faults=['cam_yaw', 'dropped'], seed=1))
    assert again[(cam, 'cam_yaw')]['curve'] == yaw['curve']
    assert again[(cam, 'dropped')]['curve'] == by[(cam, 'dropped')]['curve']
    with pytest.raises(FT.FaultTolError):
        FT.offline_campaign([recorded['run']], sensors=['nope'])
    with pytest.raises(FT.FaultTolError):
        FT.offline_campaign([recorded['run']], sensors=['lid'], faults=['cam_yaw'])


def test_offline_with_a_detector_the_ap_drop_and_the_result_document(recorded, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(os.path.dirname(__file__))
    cam = recorded['cam']
    ws = recorded['project'].workspace
    cells = FT.offline_campaign([recorded['run']], sensors=[cam], faults=['cam_yaw', 'dirty_lens', 'exposure'],
                                model=BLOB, workspace=ws, seed=1, options={'max_frames': 12, 'steps': 4})
    by = _cells(cells)
    yaw, dirt, exp = by[(cam, 'cam_yaw')], by[(cam, 'dirty_lens')], by[(cam, 'exposure')]
    assert yaw['criterion'] == 'ap_drop' and yaw['baseline'] > 70.0 and not yaw['synthetic_model']
    assert yaw['status'] == 'bracketed' and yaw['curve'][-1][1] > 5.0          # 5° off: the boxes are elsewhere
    assert dirt['status'] in ('bracketed', 'not_monotone') and dirt['curve'][-1][2]
    assert exp['status'] == 'tolerates_max'                         # white on black: over-exposure changes nothing
    doc = FT.result(cells, 'offline', 'ap', 1, {'cam_yaw': 0.1, 'dirty_lens': 0.45})
    assert doc['tolerance'][cam]['cam_yaw']['tolerated'] == yaw['tolerated']
    assert FT.requirement_status(yaw, doc['requirements']) == ('met', 'good')
    assert FT.requirement_status(dirt, doc['requirements']) == ('not met', 'critical')
    assert FT.requirement_status(exp, doc['requirements'])[0] == 'no requirement'
    mf = doc['most_fragile']
    assert mf['sensor'] == cam and mf['fault'] == 'dirty_lens' and mf['basis'] == 'requirement' and mf['value'] < 1
    assert FT.most_fragile(cells)['basis'] == 'tested range'
    st = doc['calibration'][0]
    assert st['derived'] and 'extrinsic yaw must hold within' in st['statement'] and 'not a general' in st['note']
    out = str(tmp_path / 'c')
    p = FT.write(doc, out)
    assert FT.load(out)['cells'] == json.loads(json.dumps(cells)) and os.path.isfile(os.path.join(out, 'tolerance.csv'))
    text = FT.table_text(FT.load(p))
    assert 'most fragile' in text and 'derived —' in text and 'not met' in text


def test_gsn_binds_a_tolerance_bound(recorded, tmp_path):
    from falcon_autonomy.core import gsn as G
    ws = str(tmp_path / 'ws')
    cells = FT.offline_campaign([recorded['run']], sensors=[recorded['cam']], faults=['cam_yaw'], seed=1)
    FT.write(FT.result(cells, 'offline', 'c1'), FT.campaign_dir(ws, 'c1'))
    assert FT.campaigns(ws) == ['c1'] and G.evidence_candidates(ws)['faulttol'] == ['faulttol/c1']
    doc = G.template('blank', ws, 'T')
    doc['nodes'] += [{'id': 'S1', 'type': 'Solution', 'text': 'yaw tolerance'}]
    doc['links'] += [{'from': doc['nodes'][0]['id'], 'to': 'S1', 'kind': 'SupportedBy'}]
    path = f'tolerance.{recorded["cam"]}.cam_yaw.tolerated'
    G.bind(ws, doc, 'S1', 'faulttol', 'faulttol/c1', {'measure': 'field', 'path': path, 'op': '>=', 'limit': 0.05})
    assert G.verify(ws, doc)['nodes']['S1']['status'] == 'green'
    G.bind(ws, doc, 'S1', 'faulttol', 'faulttol/c1', {'measure': 'field', 'path': path, 'op': '>=', 'limit': 4.9})
    assert G.verify(ws, doc)['nodes']['S1']['status'] == 'red'


# -------------------------------------------------------------------------------------------- closed loop and CLI
def test_closed_loop_on_the_fake_carla_the_monitor_trips_past_its_sector(tmp_path):
    """The built-in follower with its degraded mode: a blocked lidar sector is tolerated up to the health monitor's
    45° (ros2/degraded.py), a GNSS bias up to about its 4 m jump (beyond what the odometry moved); a fault the worker cannot inject is refused by name."""
    from falcon_autonomy.core import scenarios as SC
    from falcon_autonomy.core.project import Project
    from falcon_autonomy.sim.service import study_service
    from test_stack import stack_rig
    p = Project.new('CL')
    stack_rig(p)
    p.save(str(tmp_path / 'cl.falcon.json'))
    svc = study_service(workspace=p.workspace, fake_carla=FAKE, lock_dir=str(tmp_path / 'locks'))
    out = FT.campaign_dir(p.workspace, 'cl')
    try:
        kw = dict(criterion='degraded', seeds=(3,), values={'speed': 30.0, 'gap': 30.0, 'decel': 3.0,
                                                           'brake_at': 100.0}, fps=10, options={'steps': 3})
        cells = FT.closed_campaign(svc, p, SC.builtin('lead_brake'), out, 'world-town10', 'rig-default',
                                   sensors=['lid', 'gnss'], faults=['lidar_sector', 'gnss_bias', 'clock_offset'],
                                   **kw)
        by = _cells(cells)
        sec = by[('lid', 'lidar_sector')]
        assert sec['status'] == 'bracketed' and sec['bracket'][0] <= 45.0 <= sec['bracket'][1] + 1.0, sec
        gn = by[('gnss', 'gnss_bias')]
        assert gn['status'] == 'bracketed' and 3.5 <= gn['bracket'][0] < gn['bracket'][1] <= 5.5, gn
        assert by[('lid', 'clock_offset')]['status'] == 'refused' and 'closed-loop' in by[('lid', 'clock_offset')]['note']
        n_runs = len(os.listdir(os.path.join(out, 'runs')))
        again = FT.closed_campaign(svc, p, SC.builtin('lead_brake'), out, 'world-town10', 'rig-default',
                                   sensors=['lid'], faults=['lidar_sector'], **kw)
        assert again[0]['curve'] == sec['curve'] and len(os.listdir(os.path.join(out, 'runs'))) == n_runs  # cached
        doc = FT.result(cells, 'closed_loop', 'cl')
        assert '[closed loop, degraded]' in doc['calibration'][0]['statement']
    finally:
        svc.shutdown()


def test_the_command_line(recorded, tmp_path):
    def cli(*a):
        return subprocess.run([sys.executable, '-m', 'falcon_autonomy', 'faulttol', *a], capture_output=True,
                              text=True, env=worker_env({'XDG_CONFIG_HOME': str(tmp_path / 'cfg')}), timeout=120)
    r = cli('models')
    assert r.returncode == 0 and 'cam_yaw' in r.stdout and '[offline only]' in r.stdout
    r = cli('run', recorded['path'], '--name', 'cli', '--runs', recorded['run'], '--sensor', recorded['cam'],
            '--fault', 'cam_yaw', '--fault', 'frozen', '--require', 'cam_yaw=0.1', '--steps', '3')
    assert r.returncode == 0, r.stdout + r.stderr
    d = FT.campaign_dir(recorded['project'].workspace, 'cli')
    doc = FT.load(d)
    assert doc['mode'] == 'offline' and len(doc['cells']) == 2 and doc['requirements'] == {'cam_yaw': 0.1}
    r = cli('show', d)
    assert r.returncode == 0 and 'cam_yaw' in r.stdout and 'derived —' in r.stdout
    assert json.loads(cli('show', d, '--json').stdout)['schema'] == FT.SCHEMA
    assert cli('run', recorded['path'], '--name', 'x').returncode == P.REFUSED            # neither runs nor a loop
    assert cli('run', recorded['path'], '--name', 'x', '--runs', recorded['run'], '--fault', 'nope').returncode == \
        P.REFUSED
    assert cli('show', str(tmp_path / 'none')).returncode == P.FAILED
