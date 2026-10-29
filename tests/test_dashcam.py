"""Dashcam ingest (core.dashcam) on synthetic clips: small MP4s made with ffmpeg, whose frame n carries its index in
its brightness, with synthetic freeGPS blocks written in the verified layout (never the user's footage). GPS
decoding, the time base, extraction (frame indices, GPS stamps, static/dark/limit filters), the image set, the
realgap kind, the map matching against an imported world, the command and the Training page's dashcam source."""
import datetime as dt
import json
import math
import os
import shutil
import struct
import subprocess

import numpy as np
import pytest

from falcon_autonomy.core import dashcam as D
from test_gui import pump, qapp, slot_exceptions, window  # noqa: F401  (the GUI fixtures)

pytestmark = pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='needs ffmpeg')
cv2 = pytest.importorskip('cv2')

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
LAT0, LON0 = 12.9716, 77.5946          # the imported test world's centre
UTC0 = dt.datetime(2026, 3, 21, 6, 30, 0, tzinfo=dt.timezone.utc)      # noon IST: the sun is high


def make_clip(folder, start_local, seconds=4, fps=30, gop=15, size=(64, 36), fixes=None, gps=True):
    """A synthetic clip named as the camera names them; frame n's luma is 16 + 16 (n mod 14). fixes: per second, a
    (lat, lon, knots, course) or None (an empty block); the UTC of block k is the name's time + k s."""
    name = start_local.strftime('%Y%m%d%H%M%S') + '_000001F.MP4'
    path = os.path.join(folder, name)
    os.makedirs(folder, exist_ok=True)
    w, h = size
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i',
                    f'color=c=black:s={w}x{h}:r={fps}:d={seconds}', '-vf',
                    "format=gray,geq=lum='16+16*mod(N\\,14)',format=yuv420p", '-c:v', 'libx264', '-g', str(gop),
                    '-keyint_min', str(gop), '-sc_threshold', '0', '-bf', '0', '-crf', '0', '-preset', 'ultrafast',
                    path], check=True)
    if gps:
        t0 = start_local.astimezone(dt.timezone.utc)
        blocks = []
        for k, f in enumerate(fixes or []):
            blocks.append(D.empty_block() if f is None else
                          D.encode_block(t0 + dt.timedelta(seconds=k), f[0], f[1], f[2], f[3]))
        D.add_gps(path, blocks)
    return path


def straight(n, lat=LAT0, lon=LON0, kmh=36.0, course=90.0):
    """n one-second fixes along a straight line from (lat, lon) at kmh on course (degrees from north)."""
    out = []
    v = kmh / 3.6
    for k in range(n):
        d = v * k
        dn, de = d * math.cos(math.radians(course)), d * math.sin(math.radians(course))
        out.append((lat + math.degrees(dn / 6371008.8),
                    lon + math.degrees(de / (6371008.8 * math.cos(math.radians(lat)))), kmh / D.KNOT_KMH, course))
    return out


@pytest.fixture(scope='module')
def clips(tmp_path_factory):
    root = str(tmp_path_factory.mktemp('dashcam'))
    a = make_clip(os.path.join(root, 'Movie_F'), UTC0.astimezone(IST), fixes=straight(4))
    stop = [(LAT0 + 0.01, LON0, 0.0, 0.0)] * 4                                       # parked
    b = make_clip(os.path.join(root, 'Movie_F'), (UTC0 + dt.timedelta(minutes=1)).astimezone(IST), fixes=stop)
    night = (UTC0 + dt.timedelta(hours=10)).astimezone(IST)                          # 22:00 IST
    c = make_clip(os.path.join(root, 'Park_F'), night, fixes=straight(4, lat=LAT0 + 0.02))
    e = make_clip(os.path.join(root, 'Movie_F'), (UTC0 + dt.timedelta(minutes=3)).astimezone(IST), gps=False)
    return {'root': root, 'moving': a, 'parked': b, 'night': c, 'nogps': e}


# ------------------------------------------------------------------------------------------------------- GPS
def test_block_layout_round_trips_and_garbage_is_refused():
    t = dt.datetime(2026, 1, 2, 3, 4, 5, tzinfo=dt.timezone.utc)
    b = D.encode_block(t, 12.9716, 77.5946, 52.40, 90.37)
    assert len(b) == D.BLOCK_SIZE and b[4:12] == b'freeGPS ' and b[16:26] == b'YOUQINGGPS'
    assert struct.unpack('<ff', b[16 + 24:16 + 32]) == pytest.approx((1258.296, 7735.676), abs=1e-3)
    f = D.decode_block(b)
    assert f['utc'] == t and f['valid'] and f['lat'] == pytest.approx(12.9716, abs=2e-6)
    assert f['lon'] == pytest.approx(77.5946, abs=1e-5)
    assert f['speed_knots'] == pytest.approx(52.40, abs=1e-4) and f['speed_kmh'] == pytest.approx(52.40 * 1.852, 1e-4)
    assert f['course_deg'] == pytest.approx(90.37, abs=1e-4)
    assert f['camera_local'] == '2026-01-02T08:34:05'                               # IST = UTC + 5:30
    south = D.decode_block(D.encode_block(t, -33.9, -70.6))
    assert south['lat'] == pytest.approx(-33.9, abs=1e-4) and south['lon'] == pytest.approx(-70.6, abs=1e-4)
    assert D.decode_block(D.empty_block()) is None
    bad = bytearray(b)
    struct.pack_into('<f', bad, 16 + 24, 1275.0)                                   # 75 minutes: not ddmm
    assert D.decode_block(bytes(bad)) is None
    bad = bytearray(b)
    bad[16 + 57:16 + 58] = b'X'                                                     # neither N nor S
    assert D.decode_block(bytes(bad)) is None


def test_clips_listed_by_name_time_with_headers(clips):
    cs = D.list_clips(clips['root'], headers=True)
    assert [os.path.basename(c['path']) for c in cs] == [os.path.basename(clips[k]) for k in
                                                         ('moving', 'parked', 'nogps', 'night')]
    c = cs[0]
    assert c['start_utc'] == UTC0 and c['camera'] == 'F' and c['folder'] == 'Movie_F'
    assert c['fps'] == 30 and c['n_frames'] == 120 and (c['width'], c['height']) == (64, 36)
    assert c['duration_s'] == pytest.approx(4.0, abs=0.1) and c['gps_blocks'] == 4
    assert D.parse_name('x.mp4')['start_utc'] is None
    fp = D.fingerprint(clips['moving'])
    assert fp['size'] == os.path.getsize(clips['moving']) and len(fp['sha256_head']) == 64


def test_track_time_base_and_plausibility(clips):
    trk = D.track(clips['moving'])
    assert trk['n_fixes'] == 4 and trk['t0_utc'].startswith('2026-03-21T06:30:00')
    assert [f['t'] for f in trk['fixes']] == [0.0, 1.0, 2.0, 3.0]
    chk = D.check_track(trk)
    assert chk['jumps'] == 0 and chk['pairs'] == 3
    assert chk['dist_over_speed_p10_50_90'][1] == pytest.approx(1.0, abs=0.1)      # knots; float32 ≈ 0.9 m steps
    assert abs(chk['heading_minus_course_p10_50_90'][1]) < 1.0
    p = D.position_at(trk, 1.5)
    assert p['gps'] == 'interpolated' and p['speed_kmh'] == pytest.approx(36.0, abs=1e-3)
    assert p['lon'] == pytest.approx((trk['fixes'][1]['lon'] + trk['fixes'][2]['lon']) / 2, abs=1e-9)
    assert p['utc'].startswith('2026-03-21T06:30:01.500')
    assert D.position_at(trk, 3.6)['gps'] == 'nearest' and D.position_at(trk, 9.0) is None
    assert D.track(clips['nogps'])['n_fixes'] == 0


def test_repeated_and_missing_seconds(tmp_path):
    """A repeated UTC second keeps its first fix; an empty block is a hole; the time base is the GPS time."""
    t0 = UTC0
    fx = straight(5)
    blocks = [D.encode_block(t0, *fx[0]), D.encode_block(t0 + dt.timedelta(seconds=1), *fx[1]),
              D.encode_block(t0 + dt.timedelta(seconds=1), *fx[2]), D.empty_block(),
              D.encode_block(t0 + dt.timedelta(seconds=4), *fx[4])]
    p = make_clip(str(tmp_path), t0.astimezone(IST), seconds=5, gps=False)
    D.add_gps(p, blocks)
    g = D.read_gps(p)
    assert g['blocks'] == 5 and g['empty'] == 1 and len(g['fixes']) == 4
    trk = D.track(p)
    assert trk['repeated'] == 1 and [f['t'] for f in trk['fixes']] == [0.0, 1.0, 4.0]
    assert trk['fixes'][1]['lon'] == pytest.approx(fx[1][1])


def test_sun_altitude_at_known_times():
    noon = dt.datetime(2026, 6, 21, 6, 48, tzinfo=dt.timezone.utc)                 # ≈ solar noon at 77.59 E
    assert D.sun_altitude(LAT0, LON0, noon) == pytest.approx(90 - abs(LAT0 - 23.44), abs=1.0)
    assert D.sun_altitude(LAT0, LON0, noon + dt.timedelta(hours=12)) == pytest.approx(-(90 - LAT0 - 23.44), abs=1.0)


# ------------------------------------------------------------------------------------------------------- frames
def _index(rgb):
    """n mod 14 from the frame's brightness (the gray source passes through as RGB ≈ Y)."""
    return int(round((float(rgb.mean()) - 16) / 16)) % 14


def test_frames_carry_their_own_index(clips):
    got = [(n, t, _index(rgb)) for n, t, rgb in D.iter_frames(clips['moving'], rate=2, size=(32, 18),
                                                               hwaccel='none')]
    assert [g[0] for g in got] == list(range(0, 120, 15))
    assert all(g[1] == pytest.approx(g[0] / 30) for g in got)
    assert all(g[2] == g[0] % 14 for g in got)
    info = {}
    seg = list(D.iter_frames(clips['moving'], rate=None, start=2.0, duration=0.1, size=(32, 18), hwaccel='none',
                             info=info))
    assert [n for n, _t, _r in seg] == [60, 61, 62] and [_index(r) for _n, _t, r in seg] == [60 % 14, 61 % 14, 62 % 14]
    assert info['hwaccel'] == 'none' and not info['keyframes']
    k = list(D.iter_frames(clips['moving'], rate=2, size=(32, 18), hwaccel='none', keyframes=False))
    assert [n for n, _t, _r in k] == [g[0] for g in got]


def test_extract_writes_a_stamped_image_set_and_the_gap_reads_it(clips, tmp_path):
    from falcon_autonomy.core import realgap as G
    cs = D.list_clips(clips['root'])
    out = str(tmp_path / 'set')
    run = D.extract(cs, out, rate=2, size=(32, 18), min_speed_kmh=2.0, min_sun_deg=0.0, hwaccel='none', seed=3)
    c = run['counts']
    assert c['clips'] == 4 and c['candidates'] == 32
    assert c['static'] == 8 and c['dark'] == 8                                      # parked and night clips
    assert c['no_gps'] == 8 and c['written'] == 16                                  # no fix: kept by default
    st = D.read_set(out)
    rows = st['items']
    assert len(rows) == 16 and all(os.path.isfile(r['image']) for r in rows)
    mv = [r for r in rows if r['clip'] == os.path.basename(clips['moving'])]
    assert [r['frame'] for r in mv] == list(range(0, 120, 15))
    r = mv[3]                                                                       # t = 1.5 s
    assert r['t'] == 1.5 and r['gps'] == 'interpolated' and r['speed_kmh'] == pytest.approx(36.0, abs=1e-3)
    assert r['utc'].startswith('2026-03-21T06:30:01.500') and r['local'].startswith('2026-03-21T12:00:01.500')
    assert r['sun_deg'] > 60
    assert _index(cv2.imread(r['image'])) == 45 % 14
    assert run['provenance']['seed'] == 3 and len(run['clips']) == 4
    assert all(len(cl['fingerprint']['sha256_head']) == 64 for cl in run['clips'])
    assert run['outputs']['frames'] == 16
    with pytest.raises(D.DashcamError):
        D.extract(cs, out)                                                          # never over a set
    s = G.read_set('dashcam', out)
    assert s['n_total'] == 16 and not s['labelled']
    cond = [i['conditions'] for i in s['items'] if i['conditions']['lat'] is not None][0]
    assert cond['night'] is False and cond['speed_kmh'] == pytest.approx(36.0, abs=1e-3)
    from falcon_autonomy.core import translate as TR
    ims, src, _notes = TR.real_frames(f'dashcam:{out}?limit=5')
    assert len(ims) == 5 and src['kind'] == 'dashcam' and 'dashcam' in TR.terms_of([f'dashcam:{out}'])[0]
    lim = D.extract(cs, str(tmp_path / 'lim'), rate=2, size=(32, 18), min_speed_kmh=None, max_frames=5,
                    require_gps=True, hwaccel='none', crop_bottom=0.25)
    assert lim['counts']['written'] == 5 and lim['counts']['no_gps'] == 8
    assert lim['provenance']['settings']['image_size'] == [32, 14]                 # 18 · 0.75, even
    assert cv2.imread(D.read_set(str(tmp_path / 'lim'))['items'][0]['image']).shape[:2] == (14, 32)


# ------------------------------------------------------------------------------------------------------- the world
@pytest.fixture(scope='module')
def blr_project(tmp_path_factory):
    """A project with an imported world: the small Bengaluru fixture converted offline (core.xodr) when it can be;
    else a hand-written straight road along +x through the centre."""
    from falcon_autonomy.core.project import Project
    d = tmp_path_factory.mktemp('blrproj')
    p = Project.new('blr')
    p.save(str(d / 'blr.falcon.json'))
    wdir = os.path.join(p.workspace, 'worlds', 'w-test')
    os.makedirs(wdir, exist_ok=True)
    proj = f'+proj=tmerc +lat_0={LAT0} +lon_0={LON0} +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs'
    xodr = f"""<?xml version="1.0"?>
<OpenDRIVE><header revMajor="1" revMinor="4"><geoReference><![CDATA[{proj}]]></geoReference></header>
<road id="1" length="400" junction="-1"><planView><geometry s="0" x="-200" y="0" hdg="0" length="400"><line/></geometry>
</planView><lanes><laneSection s="0"><left><lane id="1" type="driving"><width sOffset="0" a="3.5" b="0" c="0" d="0"/>
</lane></left><center><lane id="0" type="none"/></center><right><lane id="-1" type="driving">
<width sOffset="0" a="3.5" b="0" c="0" d="0"/></lane></right></laneSection></lanes></road></OpenDRIVE>"""
    with open(os.path.join(wdir, 'world.xodr'), 'w', encoding='utf-8') as fh:
        fh.write(xodr)
    from falcon_autonomy.core.osm import bbox_around
    bb = list(bbox_around(LAT0, LON0, 1000.0))
    p.doc['worlds'].append({'id': 'w-test', 'name': 'Test centre', 'kind': 'osm', 'xodr': 'worlds/w-test/world.xodr',
                            'osm': {'bbox': bb, 'centre': [LAT0, LON0], 'handedness': 'RHT'}})
    p.save(str(d / 'blr.falcon.json'))
    return str(d / 'blr.falcon.json')


def test_match_finds_the_clip_on_the_imported_road(clips, blr_project):
    """The moving clip drives east along the hand-written road through the centre, 1.75 m right of the reference
    line (on its right lane); the night clip runs 2.2 km north: outside, and no lane near."""
    cs = D.list_clips(clips['root'])
    w = D.world_geo(blr_project, 'w-test')
    assert len(w['lanes']) and w['lat0'] == LAT0
    x, y, yaw = D.to_world(w, [LAT0], [LON0 + 0.001], [90.0])
    assert x[0] == pytest.approx(108.5, abs=0.5) and abs(y[0]) < 0.1 and yaw[0] == 0.0
    x, y, yaw = D.to_world(w, [LAT0 + 0.001], [LON0], [180.0])
    assert y[0] == pytest.approx(-110.6, abs=0.5) and yaw[0] == 90.0                # north is −y; south is yaw 90
    off = math.degrees(1.75 / 6371008.8)                                            # 1.75 m south: the right lane
    p = make_clip(os.path.join(clips['root'], 'Lane'), (UTC0 + dt.timedelta(minutes=5)).astimezone(IST),
                  fixes=straight(4, lat=LAT0 - off, lon=LON0 - 0.0005))
    doc = D.match_world(blr_project, 'w-test', cs + D.list_clips(p))
    by = {r['clip']: r for r in doc['clips']}
    lane = by[os.path.basename(p)]
    assert lane['inside_box'] == 4 and lane['box_distance_m'] == 0.0 and lane['near_lanes'] == 4
    seg = lane['segments'][0]
    assert seg['lane_distance_m']['median'] < 0.5 and seg['match']['verdict'] == 'match'
    night = by[os.path.basename(clips['night'])]
    assert night['inside_box'] == 0 and night['near_lanes'] == 0 and night['box_distance_m'] > 1500
    assert os.path.basename(p) in doc['inside'] and os.path.basename(clips['night']) not in doc['inside']
    assert doc['suggest']['fixes'] >= 4
    assert 'clips drive inside the imported box' in D.match_text(doc)
    os.remove(p)


def test_suggest_area_takes_the_densest_square():
    trk = lambda name, lat, n: {'clip': name, 'fixes': [{'lat': lat, 'lon': LON0 + i * 1e-4, 'speed_kmh': 20.0,
                                                          'utc': '2026-03-21T06:30:00Z', 't': i, 'course_deg': 90.0}
                                                         for i in range(n)]}
    s = D.suggest_area([trk('a', LAT0, 30), trk('b', LAT0 + 0.2, 10)], size_m=1000)
    assert s['fixes'] == 30 and s['clips'] == ['a'] and abs(s['centre'][0] - LAT0) < 0.01


# ------------------------------------------------------------------------------------------------------- command
def test_command(clips, blr_project, tmp_path, capsys):
    from falcon_autonomy.cli import main
    from falcon_autonomy.core import protocol as P
    assert main(['dashcam', 'list', '--root', clips['root']]) == P.OK
    assert '4 clips' in capsys.readouterr().out
    assert main(['dashcam', 'gps', clips['moving']]) == P.OK
    assert 'check: distance / speed' in capsys.readouterr().out
    out = str(tmp_path / 'cli-set')
    assert main(['dashcam', 'extract', '--root', clips['root'], '--out', out, '--size', '32x18', '--day',
                 '--hwaccel', 'none', '--include', '*F.MP4', '--holdout', '2', '--split', 'train']) == P.OK
    assert os.path.isfile(os.path.join(out, 'frames.jsonl'))
    assert main(['dashcam', 'match', '--root', clips['root'], '--project', blr_project, '--world', 'w-test', '--out',
                 str(tmp_path / 'm')]) == P.OK
    doc = json.load(open(tmp_path / 'm' / 'match.json'))
    assert doc['world']['id'] == 'w-test' and doc['provenance']['settings']['world'] == 'w-test'
    capsys.readouterr()
    assert main(['dashcam', 'gps', str(tmp_path / 'missing.mp4')]) == P.FAILED


# ------------------------------------------------------------------------------------------------------- GUI
def test_training_page_dashcam_source(window, qapp, clips, monkeypatch):
    page = window.go('training')
    tab = page.translation
    dc = tab.dashcam
    dc.root.setText(clips['root'])
    dc.read()
    assert pump(qapp, lambda: dc.tracks is not None, 20)
    assert '4 clips' in dc.summary.text() and len(dc.map.tracks) == 3
    seen = []
    monkeypatch.setattr(tab, 'submit', lambda spec: seen.append(spec) or type('J', (), {'id': 1})())
    dc.max_frames.setValue(10)
    dc.extract()
    assert seen and seen[0].argv[3:5] == ['dashcam', 'extract'] and '--day' in seen[0].argv
    out = seen[0].run_dir
    dc.use(out)
    assert tab.real.text() == f'dashcam:{out}'
