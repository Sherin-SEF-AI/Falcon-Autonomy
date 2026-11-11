"""Rig variants compared by perception: reading a Vantage project (its own GPL sample, tests/data/tigor-sample.vantage,
made by vantage.core.samples.tigor_sample_project without its photo set), its cameras as Falcon rigs in CARLA's frame,
the seams against Vantage's own analysis, the file never written; the comparison's statistics on synthetic detections
whose true difference is known (and on null data, how often a difference is claimed); a whole study against the fake
carla: `falcon rigs compare`, `report`, `note`; and the research tool."""
import copy
import hashlib
import io
import json
import math
import os
import shutil
import zipfile

import numpy as np
import pytest

from conftest import FAKE
from falcon_autonomy.core import metrics as M
from falcon_autonomy.core import protocol as P
from falcon_autonomy.core import rigcompare as RC
from falcon_autonomy.core import vantagerig as VR
from falcon_autonomy.sim.frames import carla_matrix

SAMPLE = os.path.join(os.path.dirname(__file__), 'data', 'tigor-sample.vantage')
# Vantage's own seams for its sample (vantage.core.analysis.Engine.analyse, run on 2026-11-02): azimuth about the
# body centre, degrees, counter-clockwise from ahead
VANTAGE_SEAMS = {
    '6-cam · v3': {'QR|MR': -97.95, 'MR|F': -23.27, 'F|ML': 23.41, 'ML|QL': 92.51, 'QL|R': 156.16, 'R|QR': -156.23},
    '4-cam': {'MR|F': -24.19, 'F|ML': 24.34, 'ML|R': 152.05, 'R|MR': -152.08},
    '4-cam, mirror cameras 5° off': {'MR|F': -23.16, 'F|ML': 25.11, 'ML|R': 156.74, 'R|MR': -156.8},
    '6-cam front-heavy': {'MR|FR': -33.43, 'FR|F': -23.31, 'F|FL': 22.19, 'FL|ML': 37.42, 'ML|R': 134.72,
                          'R|MR': -134.55},
    'Data rig · 6-cam + lidar + radars': {'QR|MR': -97.95, 'MR|F': -23.28, 'F|ML': 23.41, 'ML|QL': 92.51,
                                          'QL|R': 156.16, 'R|QR': -156.23},
}


def _sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def _variant_of(path, edit):
    """A copy of the sample with project.json changed by edit(data) (the fixture itself is never touched)."""
    src = zipfile.ZipFile(SAMPLE)
    d = json.loads(src.read('project.json'))
    edit(d)
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('project.json', json.dumps(d))
        z.writestr('mesh.npz', src.read('mesh.npz'))
    return str(path)


# ================================================================================================ reading Vantage
def test_reads_the_sample_variants_lenses_body_and_sensors():
    before = _sha(SAMPLE)
    doc = VR.read(SAMPLE)
    assert [len(v['cameras']) for v in doc['variants']] == [6, 4, 4, 6, 6]
    assert [len(v['sensors']) for v in doc['variants']] == [0, 0, 0, 0, 6]          # an OS1-128 and five radars
    b = doc['body']
    assert b['source'] == 'mesh' and b['length'] == pytest.approx(3.993, abs=1e-3)
    assert b['width'] == pytest.approx(1.676, abs=1e-3) and b['wheelbase'] == pytest.approx(2.45)
    L = VR.camera_lens(doc, doc['variants'][0]['cameras'][0])
    # STURDeCAM31 (3.0 µm, 1920 × 1536) with the H05R2 lens: f 2.62 mm, Kannala-Brandt, the image circle half the
    # datasheet diagonal (170.1°) + 2°
    assert L['model'] == 'kannala_brandt' and L['f_px'] == pytest.approx(2.62 / 0.003)
    assert L['max_half_deg'] == pytest.approx(170.1 / 2 + 2) and (L['cx'], L['cy']) == (960.0, 768.0)
    veh = VR.match_vehicle(b)
    assert veh['blueprint'] == 'vehicle.audi.a2' and veh['size_error'] < 0.05
    rigs = VR.rigs_for(doc)
    from falcon_autonomy.core import rig as R
    assert all(R.validate_rig(r) == [] for r in rigs)
    data = rigs[4]
    kinds = sorted(s['type'] for s in data['sensors'])
    assert kinds.count('camera.fisheye') == 6 and kinds.count('radar') == 5 and kinds.count('lidar') == 1
    lid = next(s for s in data['sensors'] if s['type'] == 'lidar')
    assert lid['channels'] == 128 and lid['upper_fov'] == 22.5 and lid['lower_fov'] == -22.5
    corner = next(s for s in data['sensors'] if s['type'] == 'radar' and 'corner' in s['vantage']['model'])
    assert corner['horizontal_fov'] == 150.0                                       # ±75° zone
    cam = rigs[0]['sensors'][0]
    assert (cam['width'], cam['height']) == (960, 768)                           # image scale 0.5
    assert cam['lens']['f_px'] == pytest.approx(2.62 / 0.003 / 2) and cam['lens']['fov'] == pytest.approx(174.1)
    assert {'instance', 'depth', 'boxes2d'} <= set(cam['labels'])
    assert _sha(SAMPLE) == before                                                 # read only


def test_poses_in_carlas_frame_from_the_rear_axle():
    doc = VR.read(SAMPLE)
    veh = VR.match_vehicle(doc['body'])
    rig = VR.rigs_for(doc, ['6-cam · v3'])[0]
    ml_v = next(c for c in doc['variants'][0]['cameras'] if c['tag'] == 'ML')
    ml = next(s for s in rig['sensors'] if s['name'] == 'ML')
    # Vantage: y left, yaw left, pitch down; CARLA: y right, yaw right, pitch up; x from the CARLA car's rear axle
    assert ml['pose']['x'] == pytest.approx(ml_v['x'] + veh['rear_axle_x'])
    assert ml['pose']['y'] == pytest.approx(-ml_v['y']) and ml['pose']['z'] == pytest.approx(ml_v['z'])
    assert ml['pose']['yaw'] == pytest.approx(-ml_v['yaw']) and ml['pose']['pitch'] == pytest.approx(-ml_v['pitch'])
    # the optical axis (body x) agrees as a vector: Vantage's, mirrored, is CARLA's actor x
    fwd_v = VR.rot_zyx(ml_v['yaw'], ml_v['pitch'], ml_v['roll'])[:, 0]
    p = ml['pose']
    fwd_c = carla_matrix(p['pitch'], p['yaw'], p['roll'])[:, 0]
    assert np.allclose(fwd_c, fwd_v * np.array([1, -1, 1]), atol=1e-9)
    assert fwd_c[1] < 0 and fwd_c[2] < 0                                # the left mirror camera looks left and down
    # a roll goes through the conjugation as roll
    c = dict(ml_v, yaw=30.0, pitch=20.0, roll=10.0)
    q = VR.falcon_pose(c, 0.0)
    Rv = VR.rot_zyx(30.0, 20.0, 10.0)
    S = np.diag([1.0, -1.0, 1.0])
    assert np.allclose(carla_matrix(q['pitch'], q['yaw'], q['roll']), S @ Rv @ S, atol=1e-9)


def test_seams_estimated_by_vantages_rule_agree_with_vantage():
    doc = VR.read(SAMPLE)
    veh = VR.match_vehicle(doc['body'])
    diffs = []
    for v in doc['variants']:
        sm = VR.estimate_seams(doc, v, veh)
        assert sm['centre_x'] == pytest.approx(1.222, abs=0.01)                 # Vantage's body centre
        got = {f'{s["a"]}|{s["b"]}': s['angle_deg'] for s in sm['seams']}
        assert set(got) == set(VANTAGE_SEAMS[v['name']]), v['name']               # the same neighbours
        assert all(s['source'] == 'estimated' for s in sm['seams'])
        diffs += [abs(got[k] - a) for k, a in VANTAGE_SEAMS[v['name']].items()]
    # measured 2026-11-02: 26 seams, median 1.1°, mean 1.8°, 23 within 5°, largest 6.5° (the front-heavy variant's
    # FL|ML)
    assert len(diffs) == 26 and np.median(diffs) < 1.5 and max(diffs) < 7.0 and sum(x <= 5 for x in diffs) >= 23


def test_seams_set_in_vantage_are_used_and_odd_ones_reported(tmp_path):
    doc0 = VR.read(SAMPLE)
    uid = {c['tag']: c['uid'] for c in doc0['variants'][0]['cameras']}

    def edit(d):
        d['variants'][0]['seams'] = {f'{uid["QR"]}|{uid["MR"]}': -90.0, f'{uid["F"]}|{uid["R"]}': 0.0}
    doc = VR.read(_variant_of(tmp_path / 'seams.vantage', edit))
    sm = VR.estimate_seams(doc, doc['variants'][0], VR.match_vehicle(doc['body']))
    qr = next(s for s in sm['seams'] if (s['a'], s['b']) == ('QR', 'MR'))
    assert qr['angle_deg'] == -90.0 and qr['source'] == 'vantage'
    assert [(u['a'], u['b']) for u in sm['unused']] == [('F', 'R')]               # F and R are not neighbours
    rig = VR.rigs_for(doc, ['1'])[0]
    assert any(s['source'] == 'vantage' for s in rig['vantage']['seams'])


def test_a_unit_calibration_replaces_the_library_lens(tmp_path):
    def edit(d):
        d['variants'][0]['cameras'][0]['calib'] = {'model': 'kannala_brandt', 'width': 960, 'height': 768,
                                                   'fx': 450.0, 'fy': 452.0, 'cx': 479.5, 'cy': 383.5,
                                                   'k': [0.01, 0.02, 0.0, 0.0], 'serial': 'SN7'}
    doc = VR.read(_variant_of(tmp_path / 'calib.vantage', edit))
    L = VR.camera_lens(doc, doc['variants'][0]['cameras'][0])
    # scaled to the module's 1920 × 1536, pixel centres to pixel edges (+0.5), as vantage.core.calib.unit_lens
    assert L['f_px'] == pytest.approx(900.0) and L['cx'] == pytest.approx(960.0) and L['cy'] == pytest.approx(768.0)
    assert L['aspect'] == pytest.approx(904.0 / 900.0) and L['k'] == [0.01, 0.02, 0.0, 0.0]
    assert L['source'] == 'unit calibration SN7'


def test_what_is_not_a_vantage_project_is_refused(tmp_path):
    (tmp_path / 'junk.vantage').write_bytes(b'not a zip')
    with pytest.raises(VR.VantageError, match='cannot read'):
        VR.read(str(tmp_path / 'junk.vantage'))
    with zipfile.ZipFile(tmp_path / 'other.vantage', 'w') as z:
        z.writestr('project.json', json.dumps({'format': 'falcon.project'}))
    with pytest.raises(VR.VantageError, match='not a Vantage project'):
        VR.read(str(tmp_path / 'other.vantage'))
    with pytest.raises(VR.VantageError, match='no variants'):
        VR.read(_variant_of(tmp_path / 'empty.vantage', lambda d: d.update(variants=[])))
    with pytest.raises(VR.VantageError, match='no variant'):
        VR.find_variant(VR.read(SAMPLE), 'nope')
    # no mesh: a typical small car and the centre from the wheels; still rigs
    with zipfile.ZipFile(tmp_path / 'nomesh.vantage', 'w') as z:
        z.writestr('project.json', zipfile.ZipFile(SAMPLE).read('project.json'))
    doc = VR.read(str(tmp_path / 'nomesh.vantage'))
    assert doc['body']['source'].startswith('spec') and VR.rigs_for(doc, ['1'])


def test_import_command_adds_rigs_replaces_them_on_reimport_and_never_writes_the_file(tmp_path):
    from falcon_autonomy.cli import main
    from falcon_autonomy.core.project import Project
    p = Project.new('T')
    p.save(str(tmp_path / 't.falcon.json'))
    src = str(tmp_path / 'copy.vantage')
    shutil.copy(SAMPLE, src)
    before = _sha(src)
    assert main(['rigs', 'import', p.path, src, '--variant', '6-cam · v3', '--variant', '3']) == P.OK
    q = Project.load(p.path)                                                       # validates against the schema
    assert len(q.doc['rigs']) == 3 and all(r.get('vantage') for r in q.doc['rigs'][1:])
    assert main(['rigs', 'import', p.path, src, '--variant', '3', '--scale', '0.25']) == P.OK
    q = Project.load(p.path)
    assert len(q.doc['rigs']) == 3 and q.doc['rigs'][2]['sensors'][0]['width'] == 480
    assert _sha(src) == before
    assert main(['rigs', 'vantage', src]) == P.OK
    assert main(['rigs', 'import', p.path, src, '--variant', 'nope']) == P.FAILED


# ================================================================================================ the statistics
def test_match_frame_classes_thresholds_and_ignored_objects():
    f = {'gt': [{'bbox': [0, 0, 10, 10], 'class': 'car', 'ignore': 0},
                {'bbox': [50, 50, 10, 10], 'class': 'car', 'ignore': 1},
                {'bbox': [100, 0, 10, 10], 'class': 'pedestrian', 'ignore': 0}],
         'det': [{'bbox': [0, 0, 10, 10], 'class': 'car', 'score': 0.9},
                 {'bbox': [50, 50, 10, 10], 'class': 'car', 'score': 0.8},          # on an ignored GT: neither
                 {'bbox': [100, 0, 10, 10], 'class': 'car', 'score': 0.9},          # wrong class: a false positive
                 {'bbox': [100, 0, 10, 10], 'class': 'pedestrian', 'score': 0.3}]}  # below the threshold
    hit, tp, fp = RC.match_frame(f)
    assert hit == [True, False, False] and (tp, fp) == (1, 1)
    hit, tp, fp = RC.match_frame(f, thr=0.2)
    assert hit == [True, False, True] and (tp, fp) == (2, 1)


def test_owner_and_sectors_follow_vantages_seams():
    seams = sorted([{'a': 'R', 'b': 'MR', 'angle_deg': -152.0}, {'a': 'MR', 'b': 'F', 'angle_deg': -22.0},
                    {'a': 'F', 'b': 'ML', 'angle_deg': 22.0}, {'a': 'ML', 'b': 'R', 'angle_deg': 152.0}],
                   key=lambda s: s['angle_deg'])
    assert RC.owner(seams, 0.0) == 'F' and RC.owner(seams, 90.0) == 'ML' and RC.owner(seams, -90.0) == 'MR'
    assert RC.owner(seams, 179.0) == 'R' and RC.owner(seams, -179.0) == 'R'
    assert VR.sector_of(0) == 'front' and VR.sector_of(90) == 'left' and VR.sector_of(-90) == 'right'
    assert VR.sector_of(180) == 'rear' and VR.sector_of(-135) == 'rear-right'


def _null_or_drop(rng, n_units, n, pa, pb):
    items = []
    for u in range(n_units):
        ra = rng.random(n) < pa
        rb = rng.random(n) < pb
        items.append({'u': u, 'a': {'recall': [int(ra.sum()), n]}, 'b': {'recall': [int(rb.sum()), n]}})
    return items


def test_the_paired_test_keeps_its_level_with_few_runs_and_finds_a_real_drop():
    """Null data (both rigs detect with 0.7, independently) on 6 runs of 60 objects: how often is a difference
    claimed? The percentile bootstrap over 6 units claims one more often than its 5 %; the cluster t-test keeps
    about its level. A 0.15 drop on the same design is found nearly always."""
    rng = np.random.default_rng(1)
    reps, claims, boot_claims, drops = 200, 0, 0, 0
    for k in range(reps):
        items = _null_or_drop(rng, 6, 60, 0.7, 0.7)
        claims += RC.cluster_test(items, 'recall')['p'] < 0.05
        if k < 60:
            r = M.paired_bootstrap(lambda its, w: RC.ratios(its, w), items, 'a', 'b', n=300, seed=k, unit='image')
            boot_claims += not r['metrics']['recall']['ci_low'] <= 0 <= r['metrics']['recall']['ci_high']
        drops += RC.cluster_test(_null_or_drop(rng, 6, 60, 0.7, 0.55), 'recall')['p'] < 0.05
    assert claims / reps <= 0.09 and drops / reps >= 0.85, (claims, drops)
    assert boot_claims / 60 > claims / reps, (boot_claims, claims)                # why the verdict is the t-test's
    # Holm over a family: one real effect among nine null metrics stays found; adjusted p never below the raw one
    items = []
    for u in range(6):
        a = {f'm{j}': [int((rng.random(60) < 0.7).sum()), 60] for j in range(10)}
        b = {f'm{j}': [int((rng.random(60) < (0.3 if j == 0 else 0.7)).sum()), 60] for j in range(10)}
        items.append({'u': u, 'a': a, 'b': b})
    rows = {r['metric']: r for r in RC.paired_rows(items, n_boot=200)}
    assert rows['m0']['verdict'] == 'worse' and rows['m0']['p_holm'] < 0.01
    assert all(rows[f'm{j}']['p_holm'] >= rows[f'm{j}']['p'] for j in range(10))


def _synthetic_study(tmp_path, n_runs=6, steps=10, p_a=0.9, p_b=0.9, weak=None, fp_b=0):
    """Runs whose road users stand on a 4 m ring (every 22.5°) round a car at the origin, and two rigs' frame
    records: each object seen by the camera owning its azimuth, detected with probability p (rig B: p_weak within
    ±10° of weak[0] when weak = (azimuth, p_weak)), fp_b extra false positives per B frame."""
    rng = np.random.default_rng(7)
    seams = [{'a': 'R', 'b': 'Rt', 'angle_deg': -135.0, 'source': 'vantage'},
             {'a': 'Rt', 'b': 'F', 'angle_deg': -45.0, 'source': 'vantage'},
             {'a': 'F', 'b': 'L', 'angle_deg': 45.0, 'source': 'vantage'},
             {'a': 'L', 'b': 'R', 'angle_deg': 135.0, 'source': 'vantage'}]

    def rig(rid):
        return {'id': rid, 'name': f'rig {rid}', 'vehicle': 'vehicle.audi.a2',
                'sensors': [{'id': f'cam-{t.lower()}', 'name': t, 'type': 'camera.fisheye', 'pose': {}}
                            for t in ('F', 'L', 'R', 'Rt')],
                'vantage': {'centre': {'falcon_x': 0.0}, 'seams': seams}}
    rigs = [rig('A'), rig('B')]
    azs = np.arange(0, 360, 22.5)
    runs = {'A': {}, 'B': {}}
    frames = {'A': [], 'B': []}
    for k in range(n_runs):
        lab = f's-{k}'
        for rid in ('A', 'B'):
            d = tmp_path / rid / lab
            d.mkdir(parents=True)
            with open(d / 'actors.jsonl', 'w') as fh:
                for st in range(steps):
                    actors = [{'id': 1, 'type_id': 'vehicle.audi.a2', 'role': 'hero', 'category': 'vehicle.car',
                               'x': 0.0, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                               'extent': [1.85, 0.9, 0.77], 'bbox_center': [0.0, 0.0, 0.77]}]
                    for i, a in enumerate(azs):
                        r = 4.0 + 0.02 * k
                        actors.append({'id': 100 + i + (7 if rid == 'B' else 0),   # ids differ between runs
                                       'type_id': 'walker.pedestrian.0001', 'role': 'falcon_scenario',
                                       'category': 'human.pedestrian.adult',
                                       'x': r * math.cos(math.radians(a)), 'y': -r * math.sin(math.radians(a)),
                                       'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                                       'extent': [0.2, 0.2, 0.9], 'bbox_center': [0.0, 0.0, 0.9]})
                    fh.write(json.dumps({'step': st, 'actors': actors}) + '\n')
            runs[rid][lab] = str(d)
            for st in range(steps):
                per_cam = {c: {'gt': [], 'det': []} for c in ('cam-f', 'cam-l', 'cam-r', 'cam-rt')}
                for i, a in enumerate(azs):
                    a = (a + 180) % 360 - 180
                    cam = 'cam-' + RC.owner(seams, a).lower()
                    box = [10.0 * i, 10.0, 8.0, 20.0]
                    per_cam[cam]['gt'].append({'bbox': box, 'class': 'pedestrian', 'ignore': 0,
                                               'track_id': 100 + i + (7 if rid == 'B' else 0)})
                    p = p_a if rid == 'A' else (weak[1] if weak and abs(a - weak[0]) <= 10 else p_b)
                    if rng.random() < p:
                        per_cam[cam]['det'].append({'bbox': box, 'class': 'pedestrian', 'score': 0.9})
                if rid == 'B':
                    for _ in range(fp_b):
                        per_cam['cam-f']['det'].append({'bbox': [500, 500, 5, 5], 'class': 'pedestrian',
                                                        'score': 0.9})
                for c, fr in per_cam.items():
                    frames[rid].append({'run': lab, 'drive': lab, 'step': st, 'camera': c,
                                        'image_id': f'{lab}/{c}/{st}', 'conditions': {}, **fr})
    return rigs, frames, runs


def test_a_known_drop_near_one_seam_is_found_there_and_nowhere_else(tmp_path):
    rigs, frames, runs = _synthetic_study(tmp_path, p_a=0.9, p_b=0.9, weak=(45.0, 0.2), fp_b=1)
    rep = RC.analyse(rigs, frames, runs, fps=10, n_boot=400, ap=False)
    assert rep['unit'] == 'run' and rep['n_units'] == 6
    va = {r['metric']: r for r in rep['variants']['A']['table']}
    assert va['seen']['value'] == 1.0 and va['recall']['n'] == 6 * 10 * 16
    assert va['recall']['ci_low'] <= va['recall']['value'] <= va['recall']['ci_high']
    p = rep['pairs'][0]
    assert (p['a'], p['b']) == ('A', 'B') and p['n_objects'] == 960 and p['only_a'] == p['only_b'] == 0
    rows = {r['metric']: r for r in p['rows']}
    band = 'recall.seam[F|L +45°]'
    assert rows[band]['verdict'] == 'worse' and rows[band]['diff'] == pytest.approx(-0.7, abs=0.15)
    assert rows['stitched_recall.seam[F|L +45°]']['verdict'] == 'worse'
    assert rows['precision']['verdict'] == 'worse'                                    # B's false positives
    for k in ('recall.rear', 'recall.right', 'recall.seam[R|Rt -135°]', 'recall.seam[Rt|F -45°]'):
        assert rows[k]['verdict'] == 'no detectable change', k
    assert rows['seen']['diff'] == 0.0
    md = RC.markdown(dict(rep, study='syn', model={'id': 'm'}))
    assert '| recall.seam[F|L +45°] |' in md and 'worse' in md
    note = RC.design_log(dict(rep, model={'id': 'm'}, rigs=[]), date='2026-11-02')
    e = note['entries'][0]
    assert note['schema'] == 'vantage.design_log/1' and e['kind'] == 'note' and len(e['text']) <= 400
    assert 'F|L' in e['text'] and set(e['about']) == {'F', 'L'}


def test_the_same_detections_on_both_rigs_show_no_change_and_few_runs_are_flagged(tmp_path):
    rigs, frames, runs = _synthetic_study(tmp_path, n_runs=2, p_a=0.8, p_b=0.8)
    frames['B'] = [dict(f, gt=[dict(g, track_id=g['track_id'] + 7) for g in fa['gt']], det=fa['det'])
                   for f, fa in zip(frames['B'], frames['A'])]
    rep = RC.analyse(rigs, frames, runs, fps=10, n_boot=200, block_s=0.5, ap=False)
    assert rep['unit'] == 'block' and rep['n_units'] == 4 and 'too narrow' in rep['notes'][0]
    assert all(r['diff'] in (0.0, None) and r['verdict'] == 'no detectable change' for r in rep['pairs'][0]['rows'])


def test_present_objects_in_the_cars_frame(tmp_path):
    d = tmp_path / 'run'
    d.mkdir()
    ego = {'id': 1, 'type_id': 'vehicle.audi.a2', 'role': 'hero', 'x': 10.0, 'y': 5.0, 'z': 0.0, 'roll': 0.0,
           'pitch': 0.0, 'yaw': 90.0, 'extent': [1.85, 0.9, 0.77], 'bbox_center': [0.0, 0.0, 0.77]}
    left = {'id': 2, 'type_id': 'walker.pedestrian.0002', 'role': '', 'category': 'human.pedestrian.child',
            'x': 13.0, 'y': 5.0, 'z': 0.0, 'roll': 0, 'pitch': 0, 'yaw': 0, 'extent': [0.2, 0.2, 0.6],
            'bbox_center': [0, 0, 0.6]}
    far = dict(left, id=3, x=10.0, y=30.0)
    (d / 'actors.jsonl').write_text(json.dumps({'step': 0, 'actors': [ego, left, far]}) + '\n')
    rows = RC.present_objects(str(d))[0]
    # the ego faces +y (CARLA yaw 90): a walker 3 m along +x stands on its LEFT (CARLA's y right is world −x)
    assert len(rows) == 1 and rows[0]['class'] == 'pedestrian' and rows[0]['sector'] == 'left'
    assert rows[0]['az'] == pytest.approx(90.0) and rows[0]['clearance'] == pytest.approx(2.1)


# ================================================================================================ a whole study
@pytest.fixture(scope='module')
def study(tmp_path_factory):
    from falcon_autonomy.cli import main
    from falcon_autonomy.core.project import Project
    base = tmp_path_factory.mktemp('rigstudy')
    p = Project.new('T')
    p.doc['models'] = [{'id': 'syn', 'name': 'Synthetic', 'kind': 'detector', 'runtime': 'python',
                        'entry': 'falcon_autonomy.plugins.synthetic_detector:SyntheticDetector',
                        'licence': 'GPL-3.0-or-later'}]
    p.save(str(base / 't.falcon.json'))
    assert main(['rigs', 'import', p.path, SAMPLE, '--variant', '2', '--variant', '3', '--scale', '0.05']) == P.OK
    rigs = [r['id'] for r in Project.load(p.path).doc['rigs'][1:]]
    code = main(['rigs', 'compare', p.path, '--rig', rigs[0], '--rig', rigs[1], '--model', 'syn', '--scenario',
                 'svm_children_round', '--seeds', '2', '--seconds', '1.5', '--boot', '200', '--name', 'mirror',
                 '--fake-carla', FAKE])
    assert code == P.OK
    return {'project': p.path, 'dir': str(base / 't.falcon' / 'rigstudies' / 'mirror'), 'rigs': rigs}


def test_a_study_against_the_fake_carla(study, tmp_path):
    from falcon_autonomy.cli import main
    rep = json.load(open(os.path.join(study['dir'], 'report.json')))
    assert rep['labels'] == ['svm_children_round-s0', 'svm_children_round-s1'] and rep['synthetic_model']
    assert set(rep['variants']) == set(study['rigs'])
    for v in rep['variants'].values():
        t = {r['metric']: r for r in v['table']}
        assert v['n_objects'] > 0 and len(v['cameras']) == 4 and len(v['seams']) == 4
        assert t['recall']['n'] == v['n_objects'] and 0.0 <= t['recall']['value'] <= 1.0
    p = rep['pairs'][0]
    assert p['only_a'] == p['only_b'] == 0 and p['n_objects'] > 0              # kinematic: the same road users
    runs = os.listdir(os.path.join(study['dir'], 'runs', study['rigs'][0]))
    assert sorted(runs) == rep['labels']
    drv = json.load(open(os.path.join(study['dir'], 'runs', study['rigs'][0], runs[0], 'drive.json')))
    assert drv['scenario']['motion'] == 'kinematic' and drv['traffic']['vehicles'] == 0
    assert drv['rig']['vehicle'] == 'vehicle.audi.a2'
    assert os.path.isfile(os.path.join(study['dir'], 'report.md'))
    # re-scored without the model, with wider seam bands
    assert main(['rigs', 'report', study['dir'], '--band', '20', '--boot', '100']) == P.OK
    assert json.load(open(os.path.join(study['dir'], 'report.json')))['settings']['band_deg'] == 20.0
    # the note: to its own file, never into a .vantage project
    assert main(['rigs', 'note', study['dir'], '--out', str(tmp_path / 'x.vantage')]) == P.FAILED
    assert main(['rigs', 'note', study['dir'], '--out', str(tmp_path / 'note.json')]) == P.OK
    note = json.load(open(tmp_path / 'note.json'))
    assert note['schema'] == 'vantage.design_log/1' and note['entries'][0]['text'].startswith('Falcon perception')
    # resumed: nothing driven again
    import time
    t0 = time.monotonic()
    code = main(['rigs', 'compare', study['project'], '--rig', study['rigs'][0], '--rig', study['rigs'][1],
                 '--model', 'syn', '--scenario', 'svm_children_round', '--seeds', '2', '--seconds', '1.5', '--boot',
                 '100', '--name', 'mirror', '--fake-carla', FAKE])
    assert code == P.OK and time.monotonic() - t0 < 30


def test_the_research_tool_queues_a_study_and_reads_it(study):
    from falcon_autonomy.assistant import tools as TOOLS
    from falcon_autonomy.assistant.tool_api import REGISTRY, call
    ctx = TOOLS.context(study['project'], FAKE)
    try:
        t = REGISTRY['research_rig_compare']
        assert t.group == 'research' and t.tier == 'act' and t.preview is not None
        pv = t.preview(ctx, rigs=study['rigs'], model='syn')
        assert pv.startswith('falcon rigs compare') and '--rig' in pv
        assert 'two different rigs' in call(ctx, 'research_rig_compare', {'rigs': [study['rigs'][0]] * 2,
                                                                          'model': 'syn'})['error']
        assert 'instance and depth' in call(ctx, 'research_rig_compare', {'rigs': ['rig-default', study['rigs'][0]],
                                                                          'model': 'syn'})['error']
        r = call(ctx, 'research_rig_compare', {'rigs': study['rigs'], 'model': 'syn', 'seeds': 1})
        assert r['job'] and r['rigstudy'].startswith('rigs-')
        ctx.queue.cancel(next(j for j in ctx.queue.jobs if j.id == r['job']))
        lst = call(ctx, 'analysis_outputs', {'kind': 'rigstudy'})
        assert 'mirror' in [x['id'] for x in lst['items']]
        got = call(ctx, 'analysis_outputs', {'kind': 'rigstudy', 'id': 'mirror'})
        assert set(got['rigs']) == set(study['rigs']) and got['pairs'][0]['objects'] > 0
    finally:
        ctx.service.shutdown(grace_s=10.0)
