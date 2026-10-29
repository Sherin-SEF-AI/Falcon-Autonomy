"""Night: the dashcam's night and twilight sets, exposure per frame, vehicle lights in a night drive, the gap's night
statistics and the translator's drift at night — all on synthetic data and the fake carla."""
import datetime as dt
import json
import os

import numpy as np
import pytest

from test_workers import _cfg, _run
from falcon_autonomy.core import protocol as P


# ------------------------------------------------------------------------------------------------- vehicle lights
def _lights(tmp_path, project, name, **kw):
    code, _, err = _run(_cfg(project, tmp_path / name, **kw))
    assert code == P.OK, err
    return json.loads((tmp_path / name / 'run.json').read_text())['lights']


def test_night_drive_switches_vehicle_lights_on(project, tmp_path):
    from falcon_autonomy.core import weather as W
    night = W.resolve({'preset': 'ClearNight'})
    L = _lights(tmp_path, project, 'night', weather=night)
    assert L['mode'] == 'auto' and L['resolved'] == 'on'
    assert L['vehicles'] >= 1 and L['vehicles_lit'] == L['vehicles']          # read back from every vehicle
    assert L['map_lights'] is None                                              # the fake has no LightManager
    day = _lights(tmp_path, project, 'day')                                     # the world's own (daytime) weather
    assert day['resolved'] == 'off' and day['vehicles'] == 0 and day['vehicles_lit'] == 0
    on = _lights(tmp_path, project, 'forced', vehicle_lights='on')
    assert on['resolved'] == 'on' and on['vehicles_lit'] == on['vehicles'] >= 1
    tm = _lights(tmp_path, project, 'tm', weather=night, vehicle_lights='tm')
    assert tm['resolved'] == 'tm' and tm['vehicles'] >= 1


def test_an_unknown_lights_mode_fails_cleanly(project, tmp_path):
    code, msgs, _ = _run(_cfg(project, tmp_path / 'bad', vehicle_lights='disco'))
    assert code == P.FAILED
    assert 'vehicle_lights' in json.loads((tmp_path / 'bad' / 'run.json').read_text())['error']


# ------------------------------------------------------------------------------------------------- night image sets
from test_dashcam import clips  # noqa: E402,F401  (synthetic clips: one at noon, one parked, one at 22:00 IST)
from test_gui import pump, qapp, slot_exceptions, window  # noqa: E402,F401


def test_night_and_twilight_bounds_and_exposure(clips, tmp_path):
    from falcon_autonomy.core import dashcam as D
    cs = D.list_clips(clips['root'])
    pl = D.plan(cs, rate=2, min_speed_kmh=None, max_sun_deg=D.NIGHT_SUN_DEG)
    assert {os.path.basename(i['path']) for i in pl['items']} == {os.path.basename(clips['night'])}
    assert pl['counts']['bright'] == pl['counts']['candidates'] - pl['counts']['kept']
    assert all(i['sun_deg'] < D.NIGHT_SUN_DEG for i in pl['items'])
    tw = D.plan(cs, rate=2, min_speed_kmh=None, min_sun_deg=D.NIGHT_SUN_DEG, max_sun_deg=0.0)
    assert tw['items'] == [] and tw['counts']['dark'] + tw['counts']['bright'] == tw['counts']['candidates']
    out = str(tmp_path / 'night')
    run = D.extract(cs, out, rate=2, size=(32, 18), min_speed_kmh=None, max_sun_deg=D.NIGHT_SUN_DEG, hwaccel='none')
    assert run['counts']['written'] == 8 and run['provenance']['settings']['max_sun_deg'] == D.NIGHT_SUN_DEG
    rows = D.read_set(out)['items']
    e = rows[0]['exposure']
    assert set(e) == {'luma_mean', 'luma_p01', 'luma_p50', 'luma_p99', 'dark_share', 'clipped_share'}
    assert e['luma_mean'] == pytest.approx(16.0, abs=2.0) and e['dark_share'] > 0.5     # frame 0: luma 16
    assert run['exposure']['median']['luma_mean'] > 16 and 'Rec. 601' in run['exposure']['method']
    # the CLI's --night and --twilight are exclusive with --day
    from falcon_autonomy.cli import main
    assert main(['dashcam', 'extract', '--root', clips['root'], '--out', str(tmp_path / 'cli'), '--size', '32x18',
                 '--night', '--min-speed', '-1', '--hwaccel', 'none']) == P.OK
    assert len(D.read_set(str(tmp_path / 'cli'))['items']) == 8
    assert main(['dashcam', 'extract', '--root', clips['root'], '--out', str(tmp_path / 'x'), '--day',
                 '--night']) == P.FAILED


def test_training_page_dashcam_when(window, qapp, clips, monkeypatch):
    tab = window.go('training').translation
    dc = tab.dashcam
    seen = []
    monkeypatch.setattr(tab, 'submit', lambda spec: seen.append(spec) or type('J', (), {'id': 1})())
    dc.root.setText(clips['root'])
    assert dc.when.currentData() == 'day'
    dc.when.setCurrentIndex(dc.when.findData('night'))
    assert dc.name.text() == 'dashcam-night'                       # the default name follows the time of day
    dc.extract()
    assert '--night' in seen[-1].argv and '--day' not in seen[-1].argv and seen[-1].run_dir.endswith('dashcam-night')
    dc.name.setText('mine')
    dc.when.setCurrentIndex(dc.when.findData('twilight'))
    assert dc.name.text() == 'mine'                                # a name the user chose is kept
    dc.moving.setChecked(False)
    dc.extract()
    argv = seen[-1].argv
    assert '--twilight' in argv and argv[argv.index('--min-speed') + 1] == '-1'


# ------------------------------------------------------------------------------------------------- night statistics
def _night_scene(seed=0, lights=6, glow=0.0, noise=0.0, size=(360, 640)):
    """A dark street: black-blue background, a dim road, `lights` small bright discs with an optional glow, noise."""
    import cv2
    rng = np.random.default_rng(seed)
    h, w = size
    im = np.zeros((h, w, 3), np.float32)
    im[:] = (6, 6, 10)
    im[int(h * 0.6):, :] = (22, 22, 24)
    for _ in range(lights):
        cx, cy = int(rng.integers(20, w - 20)), int(rng.integers(20, int(h * 0.6)))
        if glow:
            g = np.zeros((h, w), np.float32)
            cv2.circle(g, (cx, cy), 4, 1.0, -1)
            g = cv2.GaussianBlur(g, (0, 0), glow)
            im += (g / g.max())[..., None] * 160
        cv2.circle(im, (cx, cy), 4, (255, 250, 230), -1)
    if noise:
        im += rng.normal(0, noise, im.shape[:2])[..., None]
    return np.clip(im, 0, 255).astype(np.uint8)


def test_night_statistics_see_lights_glow_and_shadow_noise():
    from falcon_autonomy.core import realgap as G
    plain = G.night_stats(_night_scene())
    assert plain['light_blobs'] == 6 and plain['light_share'] > 0
    assert plain['dark_share_25'] > 0.9 and plain['dark_share_10'] > 0.5
    glow = G.night_stats(_night_scene(glow=6.0))
    assert glow['halo_L'] > plain['halo_L'] + 5                    # the glow lifts the ring round the lights
    noisy = G.night_stats(_night_scene(noise=4.0))
    assert noisy['noise_dark'] > plain['noise_dark'] + 1.5
    none = G.night_stats(np.full((90, 160, 3), 128, np.uint8))     # no light, no shadow
    assert none['light_blobs'] == 0 and none['halo_L'] is None and none['noise_dark'] is None


def test_the_gap_adds_night_rows_for_a_night_real_set(tmp_path):
    import cv2
    from falcon_autonomy.core import realgap as G
    for name, kw in (('sim', {}), ('real', {'glow': 6.0, 'noise': 4.0})):
        os.makedirs(tmp_path / name)
        for i in range(12):
            cv2.imwrite(str(tmp_path / name / f'{i:02d}.png'), _night_scene(seed=i, **kw)[..., ::-1])
    sim, real = G.read_set('folder', str(tmp_path / 'sim')), G.read_set('folder', str(tmp_path / 'real'))
    assert not G.is_night_set(real)                                 # a plain folder says nothing about the time
    day = G.study([dict(sim, path=str(tmp_path / 'sim'))], real)
    assert 'night' not in day['gap'] and not day['method']['night_stats']
    doc = G.study([dict(sim, path=str(tmp_path / 'sim'))], real, night=True)
    rows = {r['stat']: r for r in doc['gap']['night']}
    assert set(rows) == set(G.NIGHT_STATS) and doc['method']['night_stats']
    assert rows['noise_dark']['real_median'] > rows['noise_dark']['sim_median']
    assert {r['stat'] for r in doc['gap']['appearance']} == set(G.IMAGE_STATS)       # the daytime rows unchanged
    for it in real['items']:
        it['conditions'] = {'night': True}
    assert G.is_night_set(real)


# ------------------------------------------------------------------------------------------------- the drift at night
def test_drift_sees_lights_removed_or_invented_at_night_only():
    import cv2
    from falcon_autonomy.core import translate as TR
    im = _night_scene(seed=3, lights=8)
    same = TR.frame_drift(im, im.copy())
    assert same['drift'] == pytest.approx(0, abs=1e-5) and same['light_recall'] == 1.0 and same['light_precision'] == 1.0
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    gone = im.copy()
    gone[cv2.dilate((g > 200).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0] = (6, 6, 10)
    d = TR.frame_drift(im, gone)
    assert d['light_recall'] == 0.0 and d['drift'] > TR.DRIFT_THRESHOLD           # refused: every light removed
    more = im.copy()
    for x in range(40, 600, 70):                                                    # eight lamps where it was dark
        cv2.circle(more, (x, 30), 4, (255, 255, 255), -1)
    m = TR.frame_drift(im, more)
    assert m['light_recall'] == 1.0 and m['light_precision'] <= 0.6 and m['drift'] >= 0.4
    day = np.clip(_night_scene(seed=3).astype(int) + 120, 0, 255).astype(np.uint8)  # a bright frame: no light terms
    dd = TR.frame_drift(day, day.copy())
    assert dd['light_recall'] is None and dd['light_precision'] is None
    rows = [d, m, same]
    s = TR.drift_summary(rows)
    assert s['light_recall']['n'] == 3 and s['flagged'] == 2
