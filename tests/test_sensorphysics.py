"""Weather and sensor physics (core.sensorphysics): each model against its own documented formula — the lidar's
points kept per range bin against the model's expectation at several fog densities, clutter labelled weather_noise
and real returns keeping their labels, the published constants (Kim's q, Carbonneau, ITU-R P.838-3), the radar's
rain, the camera's droplets, contrast loss and glare, headlight projection — and a drive on the fake carla with the
physics on (recorded per frame and in run.json, never applied when off, the label consistent through the nuScenes
exporter)."""
import json
import math

import numpy as np
import pytest

from falcon_autonomy.core import sensorphysics as SP
from falcon_autonomy.core import weather as W
from test_gui import pump, qapp, slot_exceptions, window  # noqa: F401  (the GUI fixtures)

cv2 = pytest.importorskip('cv2')


def _cond(**w):
    return SP.conditions(W.resolve(dict({'preset': 'ClearNoon', 'fog_density': 0.0}, **w)))


def _rays(n, d_lo=2.0, d_hi=99.0, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.uniform(d_lo, d_hi, n)
    az = rng.uniform(-math.pi, math.pi, n)
    el = rng.uniform(-0.3, 0.1, n)
    xyz = np.c_[np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)] * d[:, None]
    tags = rng.choice([1, 9, 14, 12], n).astype(np.uint8)
    ids = np.where(np.isin(tags, [14, 12]), rng.integers(100, 200, n), 0).astype(np.uint32)
    return xyz, np.exp(-0.004 * d).astype(np.float32), tags, ids, d


# ====================================================================================== constants and conditions
def test_conditions_and_published_constants():
    c = _cond(fog_density=20.0)
    assert c['visibility_m'] == pytest.approx(350.0)
    assert c['alpha_fog_per_m'] == pytest.approx(-math.log(0.05) / 350.0)        # q = 0 below 500 m
    assert SP.kim_q(60) == 1.6 and SP.kim_q(10) == 1.3 and SP.kim_q(3) == pytest.approx(0.82)
    assert SP.kim_q(0.8) == pytest.approx(0.3) and SP.kim_q(0.2) == 0.0
    v = 3000.0                                                                   # q = 0.16 · 3 + 0.34
    assert SP.fog_extinction(v, 905) == pytest.approx(-math.log(0.05) / v * (905 / 550) ** -0.82, rel=1e-9)
    assert SP.rain_extinction_optical(25.0) == pytest.approx(1.076 * 25 ** 0.67 * math.log(10) / 10 / 1000)
    r = _cond(precipitation=60.0)
    assert r['rain_mm_h'] == pytest.approx(30.0) and 'heuristic' in r['rain_source']
    assert SP.conditions(W.resolve({'preset': 'ClearNoon'}), {'rain_mm_h': 12})['rain_mm_h'] == 12.0
    k, a = SP.p838_coefficients(77.0)                    # P.838-3 eq. (4), (5) at 45°, horizontal path
    assert k == pytest.approx((1.1320 + 1.1276) / 2) and a == pytest.approx(
        (1.1320 * 0.7177 + 1.1276 * 0.7073) / (1.1320 + 1.1276))
    assert SP.radar_rain_db_per_km(25.0) == pytest.approx(k * 25 ** a)
    assert SP.swerling1_pd(SP.radar_snr_ref(SP.RADAR_DEFAULTS), 1e-6) == pytest.approx(0.5)
    assert _cond(sun_altitude_angle=-20.0)['night'] and not _cond()['night']


def test_settings_validation_and_opt_out():
    assert SP.validate({'enabled': True, 'nonsense': 1}) == ["sensor_physics: unknown key 'nonsense'"]
    assert SP.validate({'lidar_params': {'pulse_ns': 2, 'bogus': 1}})[0].endswith("'bogus'")
    cam = {'id': 'c', 'type': 'camera.pinhole'}
    assert not SP.enabled_for(None, cam) and not SP.enabled_for({'enabled': False}, cam)
    assert SP.enabled_for({'enabled': True}, cam) and not SP.enabled_for({'enabled': True, 'camera': False}, cam)
    assert not SP.enabled_for({'enabled': True}, dict(cam, weather_physics=False))
    assert not SP.enabled_for({'enabled': True}, {'id': 'g', 'type': 'gnss'})
    assert SP.Session.from_config({'weather': {}, 'rig': {'sensors': []}}) is None


# ====================================================================================== lidar
@pytest.mark.parametrize('fog', [10.0, 20.0, 50.0, 100.0])
def test_lidar_points_kept_per_range_match_the_model(fog):
    """The implementation draws exactly the documented model: per range bin, the real returns kept against
    expected_real_kept (binomial tolerance), and with backscatter off against the survival formula itself."""
    xyz, inten, tags, ids, d = _rays(60000, seed=int(fog))
    cond = _cond(fog_density=fog)
    edges = np.array([0, 10, 20, 30, 45, 60, 80, 100])
    for p in (dict(SP.LIDAR_DEFAULTS), dict(SP.LIDAR_DEFAULTS, backscatter=False)):
        tab = SP.clutter_table(cond, 100.0, p)
        out = SP.apply_lidar(xyz, inten, tags, ids, cond, 100.0, np.random.default_rng(7), p, tab)
        x2, i2, t2, id2, st = out
        real = t2 != SP.WEATHER_NOISE_TAG
        assert st['points_in'] == len(xyz) and len(x2) == len(xyz) - st['dropped']
        got = np.histogram(np.linalg.norm(x2[real], axis=1), edges)[0]
        rho = SP.reflectivity(tags)
        exp = np.histogram(d, edges, weights=SP.expected_real_kept(d, rho, cond, 100.0, p, tab))[0]
        if not p['backscatter']:
            exp2 = np.histogram(d, edges, weights=SP.lidar_survival(d, rho, cond['alpha_lidar_per_m'], 100.0, p))[0]
            assert np.allclose(exp, exp2) and st['clutter'] == 0
        n = np.histogram(d, edges)[0]
        sd = np.sqrt(np.maximum(exp * (1 - exp / np.maximum(n, 1)), 1.0))
        assert np.all(np.abs(got - exp) <= 4.5 * sd + 2), (fog, got, exp)
    kept = SP.expected_real_kept(np.array([10.0, 40.0, 80.0]), 0.12, cond, 100.0, SP.LIDAR_DEFAULTS)
    assert kept[0] > kept[1] > kept[2]                                       # farther: more lost


def test_lidar_labels_stay_correct_and_clutter_is_labelled():
    xyz, inten, tags, ids, d = _rays(20000, seed=3)
    cond = _cond(fog_density=60.0)
    x2, i2, t2, id2, st = SP.apply_lidar(xyz, inten, tags, ids, cond, 100.0, np.random.default_rng(1))
    assert st['clutter'] > st['replaced'] > 0          # dense fog: a lost return is usually replaced by fog's own
    noise = t2 == SP.WEATHER_NOISE_TAG
    assert noise.sum() == st['clutter'] and np.all(id2[noise] == 0)
    # a real return keeps its own point, tag and id: find each by its exact position in the input
    key = {tuple(np.round(p, 6)): (t, i) for p, t, i in zip(xyz, tags, ids)}
    for p, t, i in zip(x2[~noise][:3000], t2[~noise][:3000], id2[~noise][:3000]):
        assert key[tuple(np.round(p, 6))] == (t, i)
    # clutter lies along a real ray, nearer than the surface it replaced or hid, and its intensity is low
    r = np.linalg.norm(x2[noise], axis=1)
    assert r.max() < 100.0 and np.median(r) < 15.0 and np.all(i2[noise] <= 1.0)
    # intensity: CARLA's times the two-way transmission
    keep = ~noise
    dk = np.linalg.norm(x2[keep], axis=1)
    i_in = np.exp(-0.004 * dk)
    assert np.allclose(i2[keep], i_in * np.exp(-2 * cond['alpha_lidar_per_m'] * dk), rtol=1e-4)


def test_lidar_clear_air_is_untouched_and_seeded():
    xyz, inten, tags, ids, _ = _rays(3000)
    out = SP.apply_lidar(xyz, inten, tags, ids, _cond(), 100.0, np.random.default_rng(0))
    assert np.array_equal(out[0], xyz) and out[4]['dropped'] == out[4]['clutter'] == 0
    c = _cond(fog_density=40.0, precipitation=80.0)
    a = SP.apply_lidar(xyz, inten, tags, ids, c, 100.0, SP.sensor_rng(5, 'lid', 'lidar'))
    b = SP.apply_lidar(xyz, inten, tags, ids, c, 100.0, SP.sensor_rng(5, 'lid', 'lidar'))
    assert all(np.array_equal(x, y) for x, y in zip(a[:4], b[:4]))
    tab = SP.clutter_table(_cond(precipitation=100.0), 100.0, SP.LIDAR_DEFAULTS)
    assert tab['p_rain'].max() > 0 and tab['p_fog'].max() == 0               # rain drops, no fog
    assert np.all(np.diff(tab['F']) >= -1e-15) and tab['F'][-1] < 0.2


# ====================================================================================== radar
def test_radar_rain_attenuation_and_clutter():
    sensor = {'id': 'rad', 'type': 'radar', 'range': 100.0, 'horizontal_fov': 30.0, 'vertical_fov': 10.0}
    rng = np.random.default_rng(0)
    n = 20000
    dets = np.c_[rng.normal(0, 3, n), rng.uniform(-0.25, 0.25, n), rng.uniform(-0.08, 0.08, n),
                 rng.uniform(1, 100, n)].astype(np.float32)
    dry = SP.apply_radar(dets, _cond(), sensor, 10.0, np.random.default_rng(1))
    assert np.array_equal(dry[0], dets) and dry[2]['clutter'] == 0
    c = _cond(precipitation=100.0)
    out, flags, st = SP.apply_radar(dets, c, sensor, 10.0, np.random.default_rng(1))
    assert len(out) == n - st['dropped'] + st['clutter'] and flags.sum() == st['clutter'] > 0
    surv = SP.radar_survival(dets[:, 3], c['rain_mm_h'], 100.0, SP.RADAR_DEFAULTS)
    assert abs((n - st['dropped']) - surv.sum()) < 4 * math.sqrt(n * 0.25)
    near, far = SP.radar_survival([10.0, 90.0], 50.0, 100.0, SP.RADAR_DEFAULTS)
    assert 1.0 >= near > far > 0.0
    cl = out[flags == 1]
    assert np.all(cl[:, 3] < 100.0) and np.all(np.abs(cl[:, 1]) <= math.radians(15) + 1e-6)
    assert np.median(cl[:, 0]) < 0                                          # the radar drives into the rain
    assert SP.fall_speed(2.0) == pytest.approx(9.65 - 10.3 * math.exp(-1.2))


# ====================================================================================== camera
def _scene(h=180, w=320, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 90, np.uint8)
    for _ in range(40):
        x, y = int(rng.uniform(0, w)), int(rng.uniform(0, h))
        cv2.rectangle(img, (x, y), (x + 20, y + 12), [int(v) for v in rng.integers(0, 200, 3)], -1)
    return img


def test_droplet_layer_reaches_its_steady_cover_and_refracts():
    p = SP.CAMERA_DEFAULTS
    lay = SP.DropletLayer(np.random.default_rng(0), 640, 360)
    lay.warm(25.0, seconds=40.0)
    covers = []
    for _ in range(200):
        drops = lay.advance(0.05, 25.0)
        m = np.zeros((360, 640), np.uint8)
        for x, y, r in drops:
            cv2.circle(m, (int(x), int(y)), int(round(r)), 1, -1)
        covers.append(len(drops) * 0 + (np.pi * drops[:, 2] ** 2).sum() / (640 * 360))
    want = p['cover_max'] * 25 / (25 + p['rain_half_mm_h'])
    assert np.mean(covers) == pytest.approx(want, rel=0.25)                 # the drops' summed area (overlaps counted)
    img = _scene()
    drop = np.array([[160.0, 90.0, 20.0]])
    out = SP.render_droplets(img, drop)
    yy, xx = np.mgrid[0:180, 0:320]
    outside = np.hypot(xx - 160, yy - 90) > 21
    assert np.array_equal(out[outside], img[outside])                         # only inside the drop
    assert np.abs(out[~outside].astype(int) - img[~outside]).mean() > 3
    assert SP.DropletLayer(np.random.default_rng(0), 64, 64).advance(0.05, 0.0).shape == (0, 3)


def test_wet_contrast_and_glare():
    from falcon_autonomy.core.realgap import image_stats
    img = _scene()
    wet = SP.wet_contrast(img, 0.25)
    assert image_stats(wet)['L_std'] < image_stats(img)['L_std']
    dark = np.full((180, 320, 3), 20, np.uint8)
    dark[88:93, 158:163] = 255                                                 # a lamp
    g, veil = SP.glare(dark, 90.0, 0.0)
    gw, veil_w = SP.glare(dark, 90.0, 1.0)
    assert veil > 0 and veil_w > veil * 3
    prof = g[90, 165:310, 0].astype(int)
    assert prof[0] > prof[40] >= prof[140] >= 20                               # falls off with angle
    assert SP.glare(np.full((50, 50, 3), 30, np.uint8), 90.0, 1.0)[1] == 0.0    # no source, no veil


def test_headlights_project_and_hide():
    cam = {'id': 'c', 'type': 'camera.pinhole', 'width': 320, 'height': 180, 'fov': 90.0,
           'pose': {'x': 1.5, 'y': 0.0, 'z': 1.5, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}}
    facing = {'center': [21.5, 0.0, 0.75], 'R': [[-1, 0, 0], [0, -1, 0], [0, 0, 1]], 'size': [4.5, 2.0, 1.5],
              'category': 'vehicle.car'}
    away = dict(facing, R=np.eye(3).tolist())
    lamps = SP.headlight_points({1: facing, 2: away}, cam)
    assert len(lamps) == 2                                                     # only the oncoming car's two
    f = 160.0
    for u, v, x, dist in lamps:
        assert x == pytest.approx(21.5 - 2.25 - 1.5) and abs(u - 160) == pytest.approx(f * 0.7 / x, rel=1e-6)
        assert v == pytest.approx(90 + f * (1.5 - 0.65) / x, rel=1e-6)
    cond = _cond(sun_altitude_angle=-30.0)
    cp = SP.CameraPhysics(cam, cond, 0, warm=False)
    plan = cp.plan(0.05, {1: facing})
    assert 'headlight_glare' in plan['effects'] and 'glare' in plan['effects'] and 'droplets' not in plan['effects']
    img = np.full((180, 320, 3), 10, np.uint8)
    lit, eff = cp.apply(img, plan)
    hidden, eff2 = cp.apply(img, plan, depth=np.full((180, 320), 5.0))         # a wall 5 m ahead hides them
    assert 'headlight_glare' in eff and 'headlight_glare' not in eff2
    assert lit.mean() > hidden.mean() + 1


# ====================================================================================== a drive
def _rig(project):
    from falcon_autonomy.core.rig import new_sensor
    rig = project.doc['rigs'][0]
    rig['sensors'][0].update(labels=['semantic', 'depth'], width=160, height=90)
    rig['sensors'].append(new_sensor('lidar', 'lid', 'top', {'x': 0.0, 'z': 1.9}))
    rig['sensors'][-1].update(channels=16, points_per_second=40000, range=60.0, rotation_frequency=10.0,
                              upper_fov=5.0, lower_fov=-20.0)
    rig['sensors'].append(new_sensor('radar', 'rad', 'front_radar', {'x': 2.3, 'z': 0.6}))
    project.doc['traffic_profiles'][0]['vehicles'] = 6


def test_drive_with_physics_records_what_touched_each_frame(project, tmp_path):
    from falcon_autonomy.core import protocol as P
    from test_workers import _cfg, _run
    _rig(project)
    wf = W.resolve({'preset': 'HardRainNight', 'fog_density': 60.0})
    code, _, err = _run(_cfg(project, tmp_path / 'wet', seconds=1.0, seed=4, weather=wf,
                             sensor_physics={'enabled': True}))
    assert code == P.OK, err
    out = tmp_path / 'wet'
    run = json.loads((out / 'run.json').read_text())
    sp = run['sensor_physics']
    assert sp['enabled'] and sp['sensors'] == ['cam-front', 'lid', 'rad'] and sp['conditions']['night']
    assert sp['conditions']['visibility_m'] == pytest.approx(7000 / 60) and sp['conditions']['rain_mm_h'] == 50.0
    assert sp['frames']['cam-front']['droplets'] == 20 and sp['frames']['lid']['clutter'] > 0
    assert 'ITU-R P.838-3' in sp['sources']['radar_rain'] and sp['order'].startswith('weather physics first')
    frames = [json.loads(x) for x in (out / 'frames.jsonl').read_text().splitlines()]
    assert all('droplets' in f['physics']['cam-front'] for f in frames)
    assert any('lidar_clutter' in (f.get('physics') or {}).get('lid', []) for f in frames)
    lab_all = []
    for f in frames:
        rel = f['files'].get('lid/lidar')
        if rel and rel not in lab_all:
            lab_all.append(rel)
    tags = []
    for rel in lab_all:
        n = len(np.fromfile(out / rel, np.float32)) // 4
        b = (out / rel.replace('.bin', '.labels.bin')).read_bytes()
        t = np.frombuffer(b[:n], np.uint8)
        i = np.frombuffer(b[n:], np.uint32)
        assert np.all(i[t == SP.WEATHER_NOISE_TAG] == 0)
        tags.append(t)
    tags = np.concatenate(tags)
    assert (tags == SP.WEATHER_NOISE_TAG).sum() > 0
    ds = json.loads((out / 'dataset.json').read_text())
    assert 'weather_noise' in ds['labels']['lidar']['encoding']
    # the same drive without physics: nothing applied, said so
    code, _, err = _run(_cfg(project, tmp_path / 'dry', seconds=1.0, seed=4, weather=wf))
    assert code == P.OK, err
    run2 = json.loads((tmp_path / 'dry' / 'run.json').read_text())
    assert run2['sensor_physics'] == {'enabled': False}
    assert not any('physics' in json.loads(x) for x in (tmp_path / 'dry' / 'frames.jsonl').read_text().splitlines())
    # the label through the nuScenes-lidarseg exporter: weather_noise is 'noise' (index 0)
    from falcon_autonomy.core import groundtruth as GT
    from falcon_autonomy.core.exporters import Run, lidar_labels
    r = Run(str(out))
    rec = next(f for f in r.frames if r.file(f, 'lid', 'lidar'))
    pts, t, i = r.lidar(rec, 'lid')
    seg, pan = lidar_labels(r, t, i, pts[:, :3].astype(np.float64), {})
    assert np.all(seg[t == SP.WEATHER_NOISE_TAG] == GT.NUSC_INDEX['noise'])
    assert np.all(pan[t == SP.WEATHER_NOISE_TAG] == 0)
    assert GT.CLASS_NAMES[SP.WEATHER_NOISE_TAG] == 'weather_noise'


def test_physics_then_faults_in_a_drive(project, tmp_path):
    """Both on one camera: the frame says which of each touched it, apart; a bad physics block refuses the run."""
    from falcon_autonomy.core import protocol as P
    from test_workers import _cfg, _run
    project.doc['rigs'][0]['sensors'][0]['faults'] = [{'kind': 'noise', 'sigma': 3.0}, {'kind': 'drop', 'rate': 0.3}]
    wf = W.resolve({'preset': 'HardRainNoon'})
    code, _, err = _run(_cfg(project, tmp_path / 'r', seconds=0.5, seed=2, weather=wf,
                             sensor_physics={'enabled': True, 'lidar': False}))
    assert code == P.OK, err
    run = json.loads((tmp_path / 'r' / 'run.json').read_text())
    assert run['faults_applied']['cam-front']['frames']['noise'] >= 1
    assert run['sensor_physics']['frames']['cam-front']['droplets'] >= 1
    frames = [json.loads(x) for x in (tmp_path / 'r' / 'frames.jsonl').read_text().splitlines()]
    dropped = [f for f in frames if (f.get('faults') or {}).get('cam-front') == 'drop']
    assert dropped and all('physics' not in f for f in dropped)                 # a dropped frame: nothing applied
    code, _, err = _run(_cfg(project, tmp_path / 'bad', seconds=0.5, seed=2, weather=wf,
                             sensor_physics={'enabled': True, 'nonsense': 1}))
    assert code == P.FAILED and 'nonsense' in (tmp_path / 'bad' / 'run.json').read_text()


def test_rig_page_weather_physics_switch(window, qapp):
    from PyQt6.QtWidgets import QCheckBox
    page = window.go('rigs')
    pump(qapp, timeout=0.2)
    boxes = [b for b in page.findChildren(QCheckBox) if b.text().startswith('when the world enables it')]
    assert boxes and boxes[0].isChecked()
    boxes[0].setChecked(False)
    pump(qapp, timeout=0.2)
    s = window.project.doc['rigs'][0]['sensors'][0]
    assert s['weather_physics'] is False and not SP.enabled_for({'enabled': True}, s)
