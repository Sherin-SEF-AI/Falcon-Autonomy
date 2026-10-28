"""Runtime ODD monitoring (core.oddmonitor, workers.oddwatch): the condition monitors against the ODD's ranges and
values, the timeline, the Loop's ODD mapped and its cells flagged, the data monitor's calibration (a measured
false-alarm rate on held-out drives, detection of shifted frames), fog against clear drives of the fake carla through
`falcon odd calibrate` and `falcon odd monitor`, the workers' runtime records (drive, live), the safety-case section
and the Live and Stack testing pages' indicators."""
import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest

from conftest import worker_env
from test_gui import pump, qapp, slot_exceptions, window  # noqa: F401  (the GUI fixtures)
from falcon_autonomy.core import oddmonitor as OM
from falcon_autonomy.core import protocol as P
from falcon_autonomy.core import weather as W
from falcon_autonomy.core.safetycase import example_odd

cv2 = pytest.importorskip('cv2')


def _odd(**dims):
    o = example_odd()
    o['dimensions'].update(dims)
    return o


# ====================================================================================== conditions
def test_condition_monitor_ranges_values_and_unmonitored():
    mon = OM.ConditionMonitor(OM.monitor_odd(example_odd()))
    st = OM.state_from(W.resolve({'preset': 'ClearNoon'}), 10.0, ['Town10HD_Opt'], ['straight', 'multi_lane'], 'RHT',
                       5)
    inside, reasons, unmon = mon.check(st)
    assert inside and not reasons and set(unmon) == {'scenario', 'actors'}
    st2 = dict(st, fog_density=40.0, ego_speed_kmh=72.0, place=['Town01'], road_type=['curve'])
    inside, reasons, _ = mon.check(st2)
    dims = {r['dim'] for r in reasons}
    assert not inside and dims == {'fog_density', 'ego_speed_kmh', 'place', 'road_type'}
    fog = next(r['text'] for r in reasons if r['dim'] == 'fog_density')
    assert '40' in fog and 'above' in fog and '0–20' in fog
    night = OM.state_from(W.resolve({'preset': 'ClearNight'}))
    assert night['sun_altitude_deg'] == -90.0
    assert any(r['dim'] == 'sun_altitude_deg' and 'below' in r['text'] for r in mon.check(night)[1])
    with pytest.raises(OM.OddError):
        OM.monitor_odd({'schema': 'falcon.odd/1', 'dimensions': {'x': {'kind': 'range', 'lo': 2, 'hi': 1}}})


def test_timeline_segments_and_time_outside():
    tl = OM.Timeline(0.1)
    fog = [{'dim': 'fog_density', 'text': 'fog'}]
    for i in range(10):
        tl.add(i * 0.1, i < 4 or i >= 8, [] if i < 4 or i >= 8 else fog, data_flag=(i == 9) if i % 3 == 0 else None)
    s = tl.summary()
    assert s['steps'] == 10 and s['time_outside_s'] == pytest.approx(0.5)           # 4 by fog, 1 by data
    assert s['outside_by_dimension_s'] == {'data (out of distribution)': pytest.approx(0.1),
                                           'fog_density': pytest.approx(0.4)}
    assert [g['inside'] for g in s['segments']] == [True, False, True, False]
    assert s['segments'][1]['t0'] == pytest.approx(0.4) and s['segments'][1]['t1'] == pytest.approx(0.8)
    assert s['data'] == {'frames_scored': 4, 'frames_flagged': 1}


def test_loop_odd_mapped_and_cells_flagged(tmp_path):
    from falcon_autonomy.core.project import Project
    proj = Project.new('T')
    loop_odd = {'schema': 'falcon.odd/1', 'id': 'dusk', 'name': 'Dusk', 'worlds': ['world-town10'],
                'weathers': [{'name': 'monsoon', 'preset': 'MidRainSunset', 'ranges': {'precipitation': [50, 90]}},
                             {'name': 'clear', 'preset': 'ClearNoon'}]}
    m = OM.from_loop_odd(loop_odd, proj)
    d = m['dimensions']
    assert d['precipitation']['lo'] == 0 and d['precipitation']['hi'] == 90
    assert d['sun_altitude_deg']['lo'] == 15 and d['sun_altitude_deg']['hi'] == 45
    assert set(d['place']['values']) == {'world-town10', 'Town10HD_Opt'} and 'hull' in m['about']
    assert OM.monitor_odd(loop_odd, proj)['dimensions'] == d
    rows = {r['weather']: r for r in OM.loop_cells(loop_odd, example_odd(), proj)}
    assert rows['clear']['inside'] and not rows['clear']['reasons']
    assert rows['monsoon']['partly'] and not rows['monsoon']['inside']             # 50–90 against the ODD's 0–60
    assert any('precipitation 90' in r for r in rows['monsoon']['reasons'])


# ====================================================================================== the data monitor
def _drives(n, rng, offset=0.0, frames=40, spread=0.3):
    out = {}
    for k in range(n):
        mu = rng.normal(0, spread, 11) + offset
        out[f'd{k}'] = mu + rng.normal(0, 1.0, (frames, 11))
    return out


def test_calibration_measures_false_alarms_and_detection():
    rng = np.random.default_rng(0)
    ins = _drives(8, rng)
    shifted = rng.normal(0, 1.0, (200, 11)) + 3.0
    near = rng.normal(0, 1.0, (200, 11)) + 0.2
    ref = OM.calibrate(ins, {'shifted': shifted, 'near': near}, far=0.05, seed=1)
    c = ref['calibration']
    assert c['unit'] == 'drive' and c['units'] == 8 and c['test']['frames'] == 320
    assert c['test']['false_alarm_rate'] <= 0.15 and c['test']['ci95'][0] <= c['test']['false_alarm_rate']
    assert c['detection']['shifted']['rate'] > 0.95 and c['detection']['near']['rate'] < 0.3
    assert sum(u['false_alarms'] for u in c['test']['per_unit'].values()) == c['test']['false_alarms']
    # the threshold is the conformal quantile of the out-of-fold scores
    s = np.arange(1, 101, dtype=float)
    assert OM.conformal_threshold(s, 0.05) == 96.0 and OM.conformal_threshold(s[:10], 0.01) > 10.0
    # the monitor applied to one frame (through the image statistics) agrees with the score function
    dm = OM.DataMonitor(ref)
    img = np.random.default_rng(2).integers(0, 255, (90, 160, 3), dtype=np.uint8)
    a = OM.appearance_score(ref['monitors']['appearance']['model'], OM.stats_vector(img)[None])[0]
    assert dm.score(img)['appearance'] == pytest.approx(a)
    two = OM.calibrate(_drives(2, rng, frames=30), far=0.1)
    assert two['calibration']['unit'] == 'block' and 'optimistic' in two['calibration']['note']
    with pytest.raises(OM.OddError):
        OM.calibrate({'a': np.zeros((2, 11))})
    with pytest.raises(OM.OddError):
        OM.DataMonitor({'schema': 'nope'})


def test_knn_feature_monitor():
    rng = np.random.default_rng(3)
    ins = _drives(5, rng)
    feats = {'in': {k: np.abs(rng.normal(1, 0.1, (40, 32))) for k in ins},
             'out': {'other': np.abs(rng.normal(0, 1.0, (50, 32)))}}
    ref = OM.calibrate(ins, {'other': rng.normal(0, 1, (50, 11)) + 5}, far=0.05, features=feats)
    assert ref['monitors']['features']['kind'] == 'knn_cosine' and ref['monitors']['features']['k'] == OM.KNN_K
    det = ref['calibration']['detection']['other']
    assert det['per_monitor']['features'] > 40 and det['rate'] > 0.9
    assert 'features' in next(iter(ref['calibration']['test']['per_unit'].values()))['per_monitor']
    dm = OM.DataMonitor(ref, '')                      # the weights are not here: appearance only, and it says so
    assert dm.feat is None and dm.unavailable


# ====================================================================================== fake drives, the CLI
@pytest.fixture(scope='module')
def fog_drives(tmp_path_factory):
    """Three clear drives and a fog drive (fog_density 50: the fake renders it by CARLA's visibility relation)."""
    from falcon_autonomy.core.project import Project
    from test_workers import _cfg, _run
    base = tmp_path_factory.mktemp('oddruns')
    p = Project.new('Test')
    p.doc['rigs'][0]['sensors'][0].update(width=160, height=90, labels=['semantic'])
    p.doc['traffic_profiles'][0]['vehicles'] = 4
    out = {}
    for name, seed, fog in (('c1', 1, 0.0), ('c2', 2, 0.0), ('c3', 3, 0.0), ('c4', 4, 0.0), ('fog', 5, 50.0)):
        code, _, err = _run(_cfg(p, base / name, seconds=1.0, seed=seed,
                                 weather=W.resolve({'preset': 'ClearNoon', 'fog_density': fog})))
        assert code == P.OK, err
        out[name] = str(base / name)
    odd = example_odd()
    odd['dimensions']['place']['values'].append('Town10HD_Opt')
    (base / 'odd.json').write_text(json.dumps(odd))
    return {'runs': out, 'base': base, 'odd': str(base / 'odd.json'), 'project': p}


def _cli(*args):
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy', *args], capture_output=True, text=True,
                       env=worker_env(), timeout=300)
    return r.returncode, r.stdout, r.stderr


def test_fog_against_clear_drives_through_the_cli(fog_drives):
    R, base = fog_drives['runs'], fog_drives['base']
    ref_path = base / 'reference.json'
    code, out, err = _cli('odd', 'calibrate', '--in', R['c1'], R['c2'], R['c3'], R['c4'], '--out-of',
                          f'fog={R["fog"]}', '--far', '0.05', '--every', '1', '--out', str(ref_path))
    assert code == 0, err
    ref = json.loads(ref_path.read_text())
    c = ref['calibration']
    assert ref['schema'] == OM.REFERENCE_SCHEMA and c['unit'] == 'drive' and c['test']['frames'] == 80
    assert c['detection']['fog']['rate'] >= 0.9, c['detection']
    assert 'false alarms (measured' in out and 'detected fog' in out
    code, out, err = _cli('odd', 'monitor', R['fog'], R['c1'], '--odd', fog_drives['odd'], '--reference',
                          str(ref_path), '--every', '2')
    assert code == 0, err
    doc = json.loads(open(os.path.join(R['fog'], OM.MONITOR_FILE)).read())
    assert doc['schema'] == OM.TIMELINE_SCHEMA and doc['share_outside'] == pytest.approx(1.0)
    assert doc['outside_by_dimension_s']['fog_density'] == pytest.approx(doc['time_s'])
    assert doc['data']['frames_scored'] == 10 and doc['data']['frames_flagged'] >= 9
    assert any('road type from the scenario' in n for n in doc['notes'])
    clear = json.loads(open(os.path.join(R['c1'], OM.MONITOR_FILE)).read())
    assert 'fog_density' not in clear['outside_by_dimension_s']
    assert _cli('odd', 'monitor', R['c1'], '--odd', str(base / 'nope.json'))[0] == 2


def test_drive_worker_monitors_at_runtime(fog_drives, tmp_path):
    from test_workers import _cfg, _run
    p = fog_drives['project']
    odd = json.loads(open(fog_drives['odd']).read())
    R = fog_drives['runs']
    X = {k: OM.frame_stats([q for _, _, q in OM.run_frames(R[k])]) for k in ('c1', 'c2', 'c3', 'c4')}
    ref = tmp_path / 'ref.json'
    ref.write_text(json.dumps(OM.calibrate(X, far=0.05)))
    code, msgs, err = _run(_cfg(p, tmp_path / 'r', seconds=1.0, seed=7, odd=odd, odd_reference=str(ref), odd_every=4,
                                weather=W.resolve({'preset': 'ClearNoon', 'fog_density': 30.0})))
    assert code == P.OK, err
    run = json.loads((tmp_path / 'r' / 'run.json').read_text())
    om = run['odd_monitor']
    assert om['steps'] == 20 and om['data_monitor'] and om['data']['frames_scored'] == 5
    assert om['outside_by_dimension_s']['fog_density'] == pytest.approx(1.0)
    frames = [json.loads(x) for x in (tmp_path / 'r' / 'frames.jsonl').read_text().splitlines()]
    assert all(f['odd']['in'] is False and any('fog_density' in w for w in f['odd']['why']) for f in frames)
    assert set(frames[0]['odd']['road']) >= {'multi_lane', 'two_way', 'straight'}    # the fake's road
    assert sum('score' in f['odd'] for f in frames) == 5
    sent = [m for m in msgs if m and m.get('t') == 'odd']
    assert sent and sent[0]['inside'] is False
    doc = json.loads((tmp_path / 'r' / OM.MONITOR_FILE).read_text())
    assert doc['source'] == 'runtime' and doc['time_outside_s'] == pytest.approx(1.0)
    # offline over the same run, from its frames' records (the road observed, not the heuristic)
    off = OM.monitor_run(str(tmp_path / 'r'), OM.monitor_odd(odd), write=False, data=False)
    assert off['time_outside_s'] == pytest.approx(1.0) and not off['notes']


def test_safety_case_time_outside(fog_drives):
    from falcon_autonomy.core import safetycase as SC
    R = fog_drives['runs']
    odd = OM.load_odd(fog_drives['odd'])
    rows = [{'status': 'ok', 'run': R['fog'], 'scenario': 'free', 'scenario_name': 'Free'},
            {'status': 'ok', 'run': R['c1'], 'scenario': 'free', 'scenario_name': 'Free'},
            {'status': 'failed', 'run': R['c2'], 'scenario': 'free'}]
    res = OM.suite_outside(rows, odd)
    g = res['scenarios']['free']
    assert res['runs'] == 2 and g['runs_outside'] >= 1 and g['by_dimension_s']['fog_density'] == pytest.approx(1.0)
    html = SC._odd_monitor_html(res)
    assert 'Time outside the ODD' in html and 'fog_density' in html and SC._odd_monitor_html(None) == ''


# ====================================================================================== live, pages
def test_live_session_odd_and_physics(project, tmp_path):
    from test_live import _frames, _wait
    import socket
    import threading
    from falcon_autonomy.core.live import live_config
    from falcon_autonomy.core.livelink import Reader
    path = str(tmp_path / 'l.sock')
    srv = socket.socket(socket.AF_UNIX)
    srv.bind(path)
    srv.listen(1)
    got = {'msgs': []}

    def serve():
        conn, _ = srv.accept()
        r = Reader()
        while True:
            try:
                b = conn.recv(1 << 20)
            except OSError:
                break
            if not b:
                break
            got['msgs'] += r.feed(b)
    threading.Thread(target=serve, daemon=True).start()
    cfg = live_config(project, 'world-town10', 'rig-default', str(tmp_path / 'live'), 2310, 8310, path,
                      str(tmp_path / 'recs'), fps=20.0, mode='autopilot',
                      view={'kind': 'sensor', 'sensor': 'cam-front', 'width': 96, 'height': 64, 'fov': 90.0})
    cfg['odd'] = example_odd()
    cfg['weather'] = W.resolve({'preset': 'HardRainNoon'})
    cfg['sensor_physics'] = {'enabled': True}
    cp = tmp_path / 'live.json'
    cp.write_text(json.dumps(cfg))
    p = subprocess.Popen([sys.executable, '-m', 'falcon_autonomy.workers', 'live', str(cp)], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=worker_env())
    try:
        assert _wait(lambda: len(_frames(got)) > 5), p.stderr.read() if p.poll() is not None else 'no frames'
        f = _frames(got)[-1]
        assert f['odd']['in'] is False and any('precipitation' in w for w in f['odd']['why'])
        assert 'droplets' in f['physics']
        p.stdin.write(json.dumps({'cmd': 'event', 'kind': 'weather', 'fields': {'preset': 'ClearNoon'}}) + '\n')
        p.stdin.write(json.dumps({'cmd': 'stop'}) + '\n')
        p.stdin.flush()
        p.wait(60)
    finally:
        if p.poll() is None:
            p.kill()
        srv.close()
    assert p.returncode == P.OK, p.stderr.read()
    run = json.load(open(tmp_path / 'live' / 'run.json'))
    assert run['sensor_physics']['enabled'] and run['odd_monitor']['time_outside_s'] > 0


def test_pages_show_the_odd(window, qapp, tmp_path, fog_drives):
    page = window.go('live')
    page.view.set_frame(_qimage(), {'odd': {'in': False, 'why': ['fog_density 50 above the ODD\'s 0–20 %']},
                                    'speed_kmh': 10.0})
    page.view.repaint()
    pump(qapp, timeout=0.1)
    st = window.go('stack')
    st.odd.setText(fog_drives['odd'])
    R = fog_drives['runs']
    st.odd_summary([{'status': 'ok', 'run': R['fog'], 'scenario': 'free', 'scenario_name': 'Free'}])
    assert '1.0 s of 1.0 s (100%)' in st.summary.vals['Outside the ODD'].text()
    st.odd.setText('')
    st.odd_summary([])
    assert 'choose an ODD' in st.summary.vals['Outside the ODD'].text()


def test_loop_page_flags_cells_outside(window, qapp):
    page = window.go('loop')
    loop_odd = {'schema': 'falcon.odd/1', 'id': 'dusk', 'name': 'Dusk', 'worlds': ['world-town10'],
                'weathers': [{'name': 'monsoon', 'preset': 'MidRainSunset', 'ranges': {'precipitation': [50, 90]}},
                             {'name': 'clear', 'preset': 'ClearNoon'}]}
    assert 'not checked' in page._odd_flags(loop_odd, window.project)
    os.makedirs(page.ws(), exist_ok=True)
    with open(os.path.join(page.ws(), 'odd.json'), 'w') as fh:
        json.dump(example_odd(), fh)
    text = page._odd_flags(loop_odd, window.project)
    assert 'world-town10 · monsoon — partly outside' in text and 'clear' not in text
    assert list(page.outside_cells) == ['world-town10 · monsoon']


def _qimage():
    from PyQt6.QtGui import QImage
    img = QImage(96, 64, QImage.Format.Format_RGB888)
    img.fill(0)
    return img
