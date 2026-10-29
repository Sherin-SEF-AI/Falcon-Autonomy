"""Dashcam routes (core.dashcam_route) on synthetic footage: two consecutive clips made with ffmpeg whose GPS drives
east along a straight road, a tiny OpenStreetMap file of that road, the import through the fake converter, the route
snapped to its lanes, replays against the fake CARLA (whose own road lies where the file's does) and the frame pairs.
No real footage, place or track is used."""
import datetime as dt
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from conftest import FAKE
from falcon_autonomy.core import dashcam as D
from falcon_autonomy.core import dashcam_route as R
from falcon_autonomy.core import osm as O

pytestmark = pytest.mark.skipif(not shutil.which('ffmpeg'), reason='needs ffmpeg')

LAT0, LON0 = 12.9716, 77.5946          # a synthetic point (the test world's centre)
UTC0 = dt.datetime(2026, 3, 21, 6, 30, 0, tzinfo=dt.timezone.utc)
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
V = 10.0                                # m/s, east
CLIP_S = 12


def _ll(e, n):
    return O.local_projection(LAT0, LON0).to_geo(e, n)


def make_clip(folder, k, seconds=CLIP_S, fps=30, seed=0):
    """Clip k of a drive east at V m/s along north = 0 (± 0.8 m of seeded GPS noise), starting at x = −120 m; its
    file named as the camera names clips."""
    start = UTC0 + dt.timedelta(seconds=k * seconds)
    path = os.path.join(folder, start.astimezone(IST).strftime('%Y%m%d%H%M%S') + f'_{k + 1:06d}F.MP4')
    os.makedirs(folder, exist_ok=True)
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i',
                    f'color=c=gray:s=64x36:r={fps}:d={seconds}', '-vf',
                    "format=gray,geq=lum='16+16*mod(N\\,14)',format=yuv420p", '-c:v', 'libx264', '-g', '15',
                    '-bf', '0', '-crf', '0', '-preset', 'ultrafast', path], check=True)
    rng = np.random.default_rng(seed + k)
    blocks = []
    for j in range(seconds):
        t = k * seconds + j
        lat, lon = _ll(-120.0 + V * t, float(rng.normal(0, 0.8)))
        blocks.append(D.encode_block(start + dt.timedelta(seconds=j), float(lat), float(lon), V * 3.6 / D.KNOT_KMH,
                                     90.0))
    D.add_gps(path, blocks)
    return path


def osm_file(path):
    """The drive's road (two lanes each way, its centre line 1.675 m north of the track, so that the eastbound inner
    lane's centre is on it — where the fake server's lane is), running 400 m past both ends; a road 200 m north (in
    the box, not near the track); a north-south road whose only two nodes are 120 m either side of the track (in the
    box, beyond the corridor: only its segment passes the track)."""
    nodes, ways = {}, []

    def node(i, e, n):
        nodes[i] = _ll(e, n)
    for k in range(17):
        node(1 + k, -400.0 + 50.0 * k, 1.675)
    for k in range(5):
        node(100 + k, -100.0 + 50.0 * k, 200.0)
    node(200, 30.0, -120.0)
    node(201, 30.0, 120.0)
    ways.append((7, list(range(1, 18)), {'highway': 'primary', 'lanes': '4', 'name': 'Test Road'}))
    ways.append((8, list(range(100, 105)), {'highway': 'residential', 'name': 'Far Road'}))
    ways.append((9, [200, 201], {'highway': 'residential', 'name': 'Cross Road'}))
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<osm version="0.6">']
    for i, (la, lo) in nodes.items():
        lines.append(f'<node id="{i}" lat="{la:.9f}" lon="{lo:.9f}"/>')
    for wid, refs, tags in ways:
        lines.append(f'<way id="{wid}">' + ''.join(f'<nd ref="{r}"/>' for r in refs) +
                     ''.join(f'<tag k="{k}" v="{v}"/>' for k, v in tags.items()) + '</way>')
    lines.append('</osm>')
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines))
    return path


@pytest.fixture(scope='module')
def footage(tmp_path_factory):
    base = tmp_path_factory.mktemp('route')
    clips = [make_clip(str(base / 'Movie_F'), k) for k in range(2)]
    return {'base': base, 'clips': clips, 'osm': osm_file(str(base / 'area.osm'))}


# ------------------------------------------------------------------------------------------------------- window
def test_window_joins_clips_on_gps_time(footage):
    win = R.window(footage['clips'], start=2.0, seconds=18.0)
    assert [c['offset_s'] for c in win['clips']] == pytest.approx([0.0, 12.0], abs=1e-3)
    assert win['start_s'] == 2.0 and win['end_s'] == 20.0
    assert all(c['fingerprint']['sha256_head'] for c in win['clips'])
    assert win['holes'] == [] and win['uncovered'] == []
    T = [f['T'] for f in win['fixes']]
    assert T == sorted(T) and T[0] <= 2.0 and T[-1] >= 20.0
    s, w, n, e = win['bbox']
    assert all(s <= f['lat'] <= n and w <= f['lon'] <= e for f in win['fixes'])
    # the box: the track's extent plus the margin on each side
    assert win['size_m'][0] == pytest.approx((win['fixes'][-1]['T'] - win['fixes'][0]['T']) * V + 2 * R.MARGIN_M,
                                             abs=3.0)
    with pytest.raises(D.DashcamError, match='shorter window'):
        R.window(footage['clips'], max_side_m=300.0)
    with pytest.raises(D.DashcamError, match='shorter than a second'):
        R.window(footage['clips'], start=30.0)
    tr = R.dense_track(win)
    assert tr['clip'][0] == 0 and tr['clip'][-1] == 1
    k = int(np.argmax(tr['clip'] == 1))
    assert tr['video_t'][k] == pytest.approx(tr['t'][k] - 12.0, abs=1e-6)


def test_corridor_keeps_the_roads_near_the_track_cut_to_the_box(footage, tmp_path):
    win = R.window(footage['clips'])
    info = R.corridor_osm(footage['osm'], win, str(tmp_path / 'corridor.osm'))
    assert info['counts']['ways_in_corridor'] == 2              # the drive's road and the crossing; not Far Road
    d = O.read_osm(str(tmp_path / 'corridor.osm'))
    names = sorted(w['tags'].get('name') for w in d.highways)
    assert names == ['Cross Road', 'Test Road']
    road = next(w for w in d.highways if w['tags']['name'] == 'Test Road')
    assert road['id'] == 700 and road['tags']['falcon:source_way'] == '7'
    s, w, n, e = win['bbox']
    lons = [d.nodes[r][1] for r in road['nodes']]
    inside = [w <= x <= e for x in lons]
    assert not inside[0] and not inside[-1] and all(inside[1:-1])     # one node beyond the box at each end
    assert info['counts']['cut'] == 1


# ------------------------------------------------------------------------------------------------------- snapping
def _lanes(step=1.0):
    xs = np.arange(-200.0, 200.0, step)
    rows = []
    for y, yaw in ((0.0, 0.0), (3.5, 0.0), (-3.5, 180.0), (-7.0, 180.0)):
        rows += [(x, y, yaw, 3.5) for x in xs]
    return np.array(rows)


def test_snapping_keeps_one_lane_through_gps_noise_and_follows_a_real_lane_change():
    rng = np.random.default_rng(3)
    x = np.linspace(-150, 150, 301)
    y = 0.8 + rng.normal(0, 1.2, len(x))                  # biased and noisy, on the inner eastbound lane
    yaw = np.zeros(len(x))
    sn = R.snap_to_lanes(x, y, yaw, _lanes())
    assert sn['matched'].all() and sn['lane_changes'] == 0
    assert np.abs(sn['y']).max() < 1e-9 and sn['yaw'] == pytest.approx(0.0)
    # a sustained move to the outer lane: one change, late
    y2 = np.where(x < 0, 0.3, 3.3) + rng.normal(0, 0.5, len(x))
    sn = R.snap_to_lanes(x, y2, yaw, _lanes())
    assert sn['lane_changes'] == 1
    k = int(np.argmax(sn['y'] > 1.75))
    assert x[k] >= 0.0 and sn['y'][-1] == pytest.approx(3.5)
    # driving west on the same positions: only the westbound lanes are candidates
    sn = R.snap_to_lanes(x, -3.5 + rng.normal(0, 0.5, len(x)), np.full(len(x), 180.0), _lanes())
    assert set(np.round(sn['y'], 6)) == {-3.5}
    # off the map: nothing to snap to, the GPS position is kept
    sn = R.snap_to_lanes(x, y + 40.0, yaw, _lanes())
    assert not sn['matched'].any() and sn['y'] == pytest.approx(y + 40.0)


# ------------------------------------------------------------------------------------------------------- download
def test_download_needs_permission_and_records_the_licence(tmp_path, monkeypatch):
    with pytest.raises(PermissionError, match='overpass-api.de'):
        R.fetch_overpass((12.9, 77.5, 12.91, 77.51), str(tmp_path / 'a.osm'), allowed=False)
    seen = {}

    class Resp:
        def __init__(self, data):
            self.data = data

        def read(self):
            return self.data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(req, timeout=0):
        seen['url'], seen['body'] = req.full_url, req.data.decode()
        return Resp(b'<?xml version="1.0"?><osm version="0.6"></osm>')
    import urllib.request
    monkeypatch.setattr(urllib.request, 'urlopen', urlopen)
    got = R.fetch_overpass((12.9, 77.5, 12.91, 77.51), str(tmp_path / 'a.osm'), allowed=True)
    assert seen['url'] == R.OVERPASS_URL and 'highway' in seen['body'] and '12.9000000' in seen['body']
    assert got['licence'].startswith('ODbL') and (tmp_path / 'a.osm').read_text().startswith('<?xml')


# ------------------------------------------------------------------------------------------------------- end to end
def _cli(*a):
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHONPATH')}
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy', 'dashcam', 'route', *a], capture_output=True,
                       text=True, env=env, timeout=600)
    last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else '{}'
    try:
        res = json.loads(last)
    except ValueError:
        res = {}
    return r, res


@pytest.fixture(scope='module')
def imported(footage):
    from falcon_autonomy.core.project import Project
    base = footage['base']
    p = Project.new('Route')
    p.doc['rigs'][0]['sensors'][0].update(width=96, height=54)
    pp = str(base / 'route.falcon.json')
    p.save(pp)
    route = str(base / 'r')
    r, res = _cli('plan', *footage['clips'], '--out', route)
    assert r.returncode == 0, r.stdout + r.stderr
    assert res['summary']['fixes'] == 2 * CLIP_S
    r, res = _cli('import', pp, route, '--osm', footage['osm'], '--rht', '--world-id', 'w-route', '--fake-carla',
                  FAKE)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return {'project': pp, 'route': route, 'import': res}


def test_import_adds_the_world_and_matches_the_route(imported):
    from falcon_autonomy.core.project import Project
    p = Project.load(imported['project'])
    w = p.item('worlds', 'w-route')
    assert w['kind'] == 'osm' and w['osm']['source'].endswith('corridor.osm') and w['osm']['bbox']
    assert w['traffic_profile'] == 'tp-default'                      # right-hand: --rht
    route = json.loads(open(os.path.join(imported['route'], 'route.json')).read())
    f = route['fit']
    assert f['gps_against_lanes']['verdict'] == 'match'
    assert f['snapped_share'] == 1.0 and f['lane_changes'] == 0
    assert f['gps_to_lane_m']['median'] < 1.5
    assert f['route_against_lanes']['verdict'] == 'match' and f['route_against_lanes']['median_lateral_m'] < 0.2
    assert f['path_m']['route'] == pytest.approx(f['path_m']['speed_integral'], rel=0.05)
    s = route['samples']
    assert len(s['t']) == (2 * CLIP_S - 1) * 10 + 1 and set(s['clip']) == {0, 1}
    assert imported['import']['summary']['gps_verdict'] == 'match'


@pytest.mark.parametrize('driver', ['log', 'follower'])
def test_drive_records_a_replay_with_provenance_and_pairs(imported, driver, footage):
    run = os.path.join(os.path.dirname(imported['route']), f'run-{driver}')
    r, res = _cli('drive', imported['project'], imported['route'], '--driver', driver, '--out', run,
                  '--fake-carla', FAKE, *(['--frames', '--world', 'w-route'] if driver == 'log' else []))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    rj = json.loads(open(os.path.join(run, 'run.json')).read())
    assert rj['worker'] == 'replay' and rj['status'] == 'ok'
    prov = rj['provenance']['settings']['dashcam_route']
    assert prov['driver'] == driver and [c['fingerprint']['sha256_head'] for c in prov['clips']] == \
        [D.fingerprint(c)['sha256_head'] for c in footage['clips']]
    rp = json.loads(open(os.path.join(run, 'replay.json')).read())
    assert rp['placement']['mode'] == 'geo' and rp['map_match']['verdict'] == 'match'
    pairs_dir = os.path.join(imported['route'], 'pairs', os.path.basename(run))
    doc = json.loads(open(os.path.join(pairs_dir, 'pairs.json')).read())
    rows = [json.loads(x) for x in open(os.path.join(pairs_dir, 'pairs.jsonl'))]
    assert doc['pairs'] == len(rows) and len(rows) >= 2 * (2 * CLIP_S - 2)
    for row in rows:
        assert row['step'] == round(row['T'] * 20)                  # T0 = 0, 20 steps a second
        assert row['sim_image'] and os.path.isfile(os.path.join(run, row['sim_image']))
        assert row['video_t'] == pytest.approx(row['T'] - 12.0 * row['clip'], abs=1e-6)
    if driver == 'log':
        assert doc['sim_minus_route_m']['max'] < 0.05             # placed on the route every step
        assert all(row['real_image'] and os.path.isfile(os.path.join(pairs_dir, row['real_image'])) for row in rows)
        import cv2
        img = cv2.imread(os.path.join(pairs_dir, rows[3]['real_image']), cv2.IMREAD_GRAYSCALE)
        assert img.shape == (360, 640)
        n = rows[3]['video_frame']                                  # frame n's luma is 16 + 16 (n mod 14)
        assert abs(float(np.median(img)) - (16 + 16 * (n % 14))) < 6
    else:
        assert doc['sim_across_m']['max'] < 0.2                   # on the route's lane …
        assert doc['sim_along_m']['median'] < 2.5                 # … a little behind it after starting
        assert all(r_['real_image'] is None for r_ in rows)


def test_pairs_without_a_match_and_a_wrong_world_are_refused(imported, tmp_path):
    r, _ = _cli('drive', imported['project'], imported['route'], '--world', 'world-town10', '--out',
                str(tmp_path / 'x'), '--fake-carla', FAKE)
    assert r.returncode != 0 and 'no geographic reference' in r.stdout
    r, _ = _cli('pairs', imported['route'], str(tmp_path / 'nothing'))
    assert r.returncode != 0


def test_download_is_not_done_without_confirmation(imported):
    r, res = _cli('import', imported['project'], imported['route'], '--download-osm', '--fake-carla', FAKE)
    assert r.returncode != 0 and 'confirmation' in r.stderr
    assert res.get('error') == 'download not confirmed'


# ------------------------------------------------------------------------------------------------------- the dialog
from test_gui import pump, qapp, slot_exceptions, window  # noqa: E402,F401


@pytest.mark.gui
def test_route_dialog_plans_imports_drives_and_pairs(window, qapp, footage):
    from falcon_autonomy.core.jobs import FINAL
    from falcon_autonomy.ui.dashcam_route import open_route_dialog
    dlg = open_route_dialog(window)
    for c in footage['clips']:
        dlg.clips.addItem(c)
    dlg.start.setValue(0.0)
    dlg.seconds.setValue(24.0)
    assert not dlg.drive_btn.isEnabled() and dlg.drive_btn.toolTip()          # says why
    dlg.plan()
    assert pump(qapp, lambda: dlg.win_doc is not None, 30)
    assert '2 clip(s)' in dlg.plan_kv.vals['Route'].text()
    assert os.path.isfile(os.path.join(dlg.route_dir, 'window.json'))
    dlg.source.setText(footage['osm'])
    dlg.lht.setChecked(False)
    dlg.import_roads()
    assert pump(qapp, lambda: dlg.job is not None and dlg.job.state in FINAL, 120), dlg.import_kv.vals['Import'].text()
    assert dlg.job.state == 'done'
    assert pump(qapp, lambda: dlg.drive_btn.isEnabled(), 60), dlg.import_kv.vals['Fit'].text()
    w = window.project.item('worlds', dlg.world_id)
    assert w and w['kind'] == 'osm' and window.undo.canUndo()
    assert 'match' in dlg.import_kv.vals['Fit'].text()
    dlg.driver.setCurrentIndex(dlg.driver.findData('log'))
    dlg.drive()
    assert pump(qapp, lambda: dlg.drive_job is not None and dlg.drive_job.state in FINAL, 300), \
        dlg.drive_kv.vals['Run'].text()
    assert dlg.drive_job.state == 'done', dlg.drive_job.log[-10:]
    assert pump(qapp, lambda: 'pairs →' in dlg.drive_kv.vals['Pairs'].text(), 60), dlg.drive_kv.vals['Pairs'].text()
    run = json.loads(open(os.path.join(dlg.drive_out, 'run.json')).read())
    assert run['provenance']['settings']['dashcam_route']['driver'] == 'log'
    dlg.close()
    dlg.deleteLater()
    pump(qapp, timeout=0.2)


@pytest.mark.gui
def test_worlds_page_opens_the_route_dialog(window, qapp):
    from falcon_autonomy.ui.dashcam_route import RouteDialog
    dlg = window.go('worlds').dashcam_route()
    assert isinstance(dlg, RouteDialog) and dlg.isVisible()
    dlg.close()
    dlg.deleteLater()
    pump(qapp, timeout=0.1)
