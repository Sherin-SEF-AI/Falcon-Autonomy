"""The city around an imported world's roads (core.cityprops): features from OSM, the plan (buildings fitted to
footprints and kept off the lanes, trees, lamps, props, the ground), determinism, the drive config and spawning on
the fake carla. Synthetic data only."""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import worker_env
from falcon_autonomy.core import cityprops as C
from falcon_autonomy.core import osm as O
from falcon_autonomy.core import protocol as P
from falcon_autonomy.core import xodr as X

LAT0, LON0 = 12.9716, 77.5946          # a synthetic place (the test world's centre)
KX, KY = 111319.49 * math.cos(math.radians(LAT0)), 110574.0

# one straight road along x (east), -150 … 150 m: a sidewalk (2 m) and a driving lane (3.5 m) each side
XODR = """<?xml version="1.0"?>
<OpenDRIVE><header revMajor="1" revMinor="4"/>
<road name="Main" length="300" id="1" junction="-1" rule="LHT">
 <planView><geometry s="0" x="-150" y="0" hdg="0" length="300"><line/></geometry></planView>
 <lanes><laneSection s="0">
  <left><lane id="2" type="sidewalk"><width sOffset="0" a="2" b="0" c="0" d="0"/></lane>
        <lane id="1" type="driving"><width sOffset="0" a="3.5" b="0" c="0" d="0"/></lane></left>
  <center><lane id="0" type="none"/></center>
  <right><lane id="-1" type="driving"><width sOffset="0" a="3.5" b="0" c="0" d="0"/></lane>
         <lane id="-2" type="sidewalk"><width sOffset="0" a="2" b="0" c="0" d="0"/></lane></right>
 </laneSection></lanes>
</road>
</OpenDRIVE>"""
EDGE = 5.5                             # the paved edge each side of the reference line

CATALOGUE = {
    'schema': C.CATALOGUE_SCHEMA, 'carla': '0.9.16', 'measured': 'synthetic',
    'buildings': [{'path': '/Game/Test/SM_House.SM_House', 'size': [10.0, 8.0, 7.0], 'center': [1.0, -0.5, 3.5],
                   'bottom': 0.0},
                  {'path': '/Game/Test/SM_Block.SM_Block', 'size': [30.0, 15.0, 20.0], 'center': [0.0, 0.0, 10.0],
                   'bottom': 0.0}],
    'trees': [{'path': '/Game/Test/SM_Palm.SM_Palm', 'size': [4.6, 4.8, 11.0], 'center': [0.5, 0.2, 5.5],
               'bottom': 0.0}],
    'bushes': [],
    'lamps': [{'path': '/Game/Test/SM_Lamp.SM_Lamp', 'size': [7.9, 0.7, 11.0], 'center': [3.6, 0.0, 5.5],
               'bottom': 0.0, 'arm': 'x'}],
    'ground': {'path': '/Game/Test/SM_Plane.SM_Plane', 'size': [100.0, 100.0, 0.0], 'center': [0.0, 0.0, 0.0],
               'bottom': 0.0},
    'props': {'static.prop.busstop': 'bus_stop', 'static.prop.bench01': 'bench'}, 'tags': {}}


def _geo(x, y):
    return LAT0 + y / KY, LON0 + x / KX


def _osm(path):
    """Buildings (one clear of the road, one across it, one tiny), a park, an open way, and point features."""
    nodes, ways = {}, []
    nid = [100]

    def node(x, y, tags=None):
        nid[0] += 1
        nodes[nid[0]] = (x, y, tags or {})
        return nid[0]

    def ring(pts, tags, wid):
        ids = [node(x, y) for x, y in pts]
        ways.append((wid, ids + [ids[0]], tags))

    ring([(-40, 12), (-30, 12), (-30, 20), (-40, 20)], {'building': 'house', 'building:levels': '2'}, 1)
    ring([(10, 9), (40, 9), (40, 24), (10, 24)], {'building': 'apartments', 'height': '19'}, 2)
    ring([(60, -3), (70, -3), (70, 3), (60, 3)], {'building': 'yes'}, 3)          # across the road
    ring([(90, 30), (92, 30), (92, 32), (90, 32)], {'building': 'shed'}, 4)       # 4 m²
    ring([(-120, -60), (-60, -60), (-60, -20), (-120, -20)], {'leisure': 'park'}, 5)
    ids = [node(0, 50), node(5, 50)]
    ways.append((6, ids, {'building': 'yes'}))                                    # not closed
    node(-100, 30, {'natural': 'tree'})
    node(0, 0.5, {'natural': 'tree'})                                             # on the carriageway
    node(20, -1.0, {'highway': 'bus_stop'})                                       # mapped on the road
    node(-10, -8.0, {'amenity': 'bench'})
    node(500, 500, {'natural': 'tree'})                                           # outside the box
    lines = ['<?xml version="1.0"?>', '<osm version="0.6">']
    for i, (x, y, t) in nodes.items():
        la, lo = _geo(x, y)
        tg = ''.join(f'<tag k="{k}" v="{v}"/>' for k, v in t.items())
        lines.append(f'<node id="{i}" lat="{la:.9f}" lon="{lo:.9f}">{tg}</node>')
    for wid, refs, t in ways:
        lines.append(f'<way id="{wid}">' + ''.join(f'<nd ref="{r}"/>' for r in refs) +
                     ''.join(f'<tag k="{k}" v="{v}"/>' for k, v in t.items()) + '</way>')
    lines.append('</osm>')
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines))
    return path


def _bbox():
    s, w = _geo(-200, -200)
    n, e = _geo(200, 200)
    return (s, w, n, e)


@pytest.fixture
def features(tmp_path):
    return C.extract(_osm(str(tmp_path / 'city.osm')), _bbox(), (LAT0, LON0))


# ------------------------------------------------------------------------------------------------- geometry
def test_min_rect_overlap_distance_and_inside():
    a = math.radians(30)
    u, v = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    P = np.array([5 * u * sx + 2 * v * sy for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]) + (3, 4)
    cx, cy, L, W, ang = C.min_rect(P)
    assert (cx, cy, L, W) == pytest.approx((3, 4, 10, 4), abs=1e-6) and ang == pytest.approx(a, abs=1e-6)
    assert C.rects_overlap((0, 0, 4, 2, 0), (2.5, 0, 4, 2, math.pi / 2))
    assert not C.rects_overlap((0, 0, 4, 2, 0), (3, 0, 4, 2, math.pi / 2))          # touching
    assert not C.rects_overlap((0, 0, 4, 2, 0), (5, 0, 4, 2, 0))
    assert C.point_rect_distance([[0, 0], [5, 0], [5, 4]], (0, 0, 4, 2, 0)) == pytest.approx([0, 3, math.hypot(3, 3)])
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert C.in_poly([[5, 5], [15, 5], [-1, -1]], sq).tolist() == [True, False, False]
    assert C.poly_area(sq) == pytest.approx(100.0)


def test_building_height_from_tags():
    assert C.building_height({'height': '19 m'}) == (19.0, 'height')
    assert C.building_height({'building:levels': '4'}) == (pytest.approx(4 * C.LEVEL_M), 'levels')
    assert C.building_height({'building': 'apartments'}) == (pytest.approx(5 * C.LEVEL_M), 'typical')
    assert C.building_height({'height': 'tall', 'building': 'yes'})[1] == 'typical'


def test_place_mesh_puts_the_scaled_box_centre_on_the_footprint():
    m = {'center': [1.0, -0.5, 3.5], 'bottom': 0.2}
    for ang in (0.0, 0.7, 2.0):
        tf = C._place_mesh(m, 12.0, 30.0, ang, 1.5)
        r = math.radians(tf['yaw'])
        ox, oy = 1.5 * 1.0, 1.5 * -0.5
        wx = tf['x'] + ox * math.cos(r) - oy * math.sin(r)
        wy = tf['y'] + ox * math.sin(r) + oy * math.cos(r)
        assert (wx, wy) == pytest.approx((12.0, -30.0), abs=1e-3)          # CARLA's y is OpenDRIVE's −y
        assert tf['yaw'] == pytest.approx(((-math.degrees(ang) + 180) % 360) - 180, abs=1e-3)
        assert tf['z'] == pytest.approx(-1.5 * 0.2)


# ------------------------------------------------------------------------------------------------- features
def test_osm_reader_keeps_city_features_only_when_asked(tmp_path):
    src = _osm(str(tmp_path / 'c.osm'))
    plain = O.read_osm(src, bbox=_bbox())
    assert not plain.points and {w['kind'] for w in plain.ways} == {'building'}
    city = O.read_osm(src, bbox=_bbox(), city=True)
    assert {w['kind'] for w in city.ways} == {'building', 'area'}
    assert sorted(p['tags'].get('natural') or p['tags'].get('highway') or p['tags'].get('amenity')
                  for p in city.points) == ['bench', 'bus_stop', 'tree', 'tree']        # the far tree is outside
    assert city.counts['points'] == 4 and city.counts['areas'] == 1


def test_extract_gives_footprints_areas_and_points_in_the_local_frame(features):
    f = features
    assert f['schema'] == C.FEATURES_SCHEMA and f['counts']['buildings'] == 4
    b1 = next(b for b in f['buildings'] if b['id'] == 1)
    assert np.asarray(b1['poly']).mean(0) == pytest.approx((-35, 16), abs=0.05)
    assert b1['tags'] == {'building': 'house', 'building:levels': '2'}
    assert f['areas'][0]['kind'] == 'park' and f['counts']['by_point'] == {'bench': 1, 'bus_stop': 1, 'tree': 2}
    assert any('not closed' in n for n in f['notes'])


# ------------------------------------------------------------------------------------------------- the plan
def test_plan_fits_buildings_off_the_lanes_and_is_deterministic(features):
    doc = C.plan(features, XODR, CATALOGUE, seed=3)
    again = C.plan(features, XODR, CATALOGUE, seed=3)
    assert C.plan_sha(doc) == C.plan_sha(again) and doc['schema'] == C.PLAN_SCHEMA
    blds = [i for i in doc['items'] if i['kind'] == 'building']
    assert sorted(b['osm_id'] for b in blds) == [1, 2]
    sk = doc['summary']['skipped']['building']
    assert sk.get('would cover a lane or sidewalk') == 1 and sk.get('footprint under 12 m²') == 1
    b2 = next(b for b in blds if b['osm_id'] == 2)
    assert b2['mesh'].endswith('SM_Block') and b2['height'] == [19.0, 'height']        # 30 × 15 footprint, 19 m
    b1 = next(b for b in blds if b['osm_id'] == 1)
    assert b1['mesh'].endswith('SM_House') and b1['height'][1] == 'levels'
    lanes = C.Lanes(XODR)
    for b in blds:                       # every placed box keeps off the paved area (|y| ≤ 5.5 m) by the margin
        L, W = b['placed'][:2]
        yaw = -math.radians(b['yaw'])
        m = next(c for c in CATALOGUE['buildings'] if c['path'] == b['mesh'])
        ox, oy = b['scale'] * m['center'][0], b['scale'] * m['center'][1]
        r = math.radians(b['yaw'])
        cx = b['x'] + ox * math.cos(r) - oy * math.sin(r)
        cy = -(b['y'] + ox * math.sin(r) + oy * math.cos(r))
        assert lanes.clearance('paved', (cx, cy, L, W, yaw)) >= C.LANE_MARGIN_M - 1e-6
    assert doc['summary']['height_from'] == {'height': 1, 'levels': 1}


def test_plan_trees_lamps_props_and_ground(features):
    doc = C.plan(features, XODR, CATALOGUE, seed=0)
    by = doc['summary']['by_kind']
    assert by['tree']['osm'] == 1 and doc['summary']['skipped']['tree']['on a carriageway'] == 1
    assert by['tree']['osm-area'] >= 10                     # the park (60 × 40 m) at 9 m spacing
    trees = [i for i in doc['items'] if i['kind'] == 'tree' and i['source'] == 'osm-area']
    xy = np.array([[t['x'], t['y']] for t in trees])
    assert ((xy[:, 0] > -121) & (xy[:, 0] < -59) & (-xy[:, 1] > -61) & (-xy[:, 1] < -19)).all()
    lamps = [i for i in doc['items'] if i['kind'] == 'lamp']
    assert lamps and all(i['source'] == 'procedural' for i in lamps)
    for la in lamps:                     # on a sidewalk (OpenDRIVE |y| 3.5 … 5.5), the arm (+x) towards the road
        assert 3.4 <= abs(la['y']) <= 5.6
        arm_y = math.sin(math.radians(la['yaw']))
        assert arm_y * la['y'] < 0
    same_side = sorted(la['x'] for la in lamps if la['y'] > 0)
    assert all(b - a >= C.LAMP_SPACING_M * 0.6 for a, b in zip(same_side, same_side[1:]))
    stop = next(i for i in doc['items'] if i['kind'] == 'bus_stop')
    assert stop['blueprint'] == 'static.prop.busstop' and 3.4 <= abs(stop['y']) <= 5.6     # moved onto the sidewalk
    ground = [i for i in doc['items'] if i['kind'] == 'ground']
    assert len(ground) == 1 and ground[0]['z'] == pytest.approx(-C.GROUND_DROP_M) and ground[0]['scale'] > 3


def test_plan_options_and_seed(features):
    off = C.plan(features, XODR, CATALOGUE, options={'trees': False, 'lamps': 'off', 'ground': False})
    assert {i['kind'] for i in off['items']} == {'building', 'bus_stop', 'bench'}
    a = C.plan(features, XODR, CATALOGUE, seed=1)
    b = C.plan(features, XODR, CATALOGUE, seed=2)
    assert C.plan_sha(a) != C.plan_sha(b)                   # tree positions and turns are seeded


def test_catalogue_from_probe_keeps_whole_buildings():
    probe = {'boxes': {
        '/Game/Carla/Static/Building/00_Opt/SM_House01_Opt.SM_House01_Opt':
            {'extent': [4.8, 6.05, 6.3], 'center': [0, 0, 6.32], 'valid': True},
        '/Game/Carla/Static/Building/SM_House01.SM_House01': {'extent': [4.8, 6.05, 6.3], 'center': [0, 0, 6.32],
                                                              'valid': True},
        '/Game/Carla/Static/Building/Hotel/SM_Hotel_BodyWin01.SM_Hotel_BodyWin01':
            {'extent': [2.9, 1.3, 1.5], 'center': [0, 0, 1.5], 'valid': True},
        '/Game/Carla/Static/Building/SM_Skycraper01.SM_Skycraper01': {'extent': [5, 5, 15.5], 'center': [0, 0, -14.5],
                                                                      'valid': True},
        '/Game/Carla/Static/Vegetation/Trees/SM_CoconutPalm_01.SM_CoconutPalm_01':
            {'extent': [2.3, 2.4, 5.5], 'center': [0.58, 0.2, 5.52], 'valid': True},
        '/Game/Carla/Static/Pole/StreetLights/Group1/SM_Streetlight1_group1.SM_Streetlight1_group1':
            {'extent': [3.95, 0.35, 5.5], 'center': [3.59, 0, 5.48], 'valid': True},
        '/Game/Carla/Static/Other/SM_Plane.SM_Plane': {'extent': [50, 50, 0], 'center': [0, 0, 0], 'valid': True}},
        'labels': {'/Game/Carla/Static/Building/SM_House01.SM_House01': {'semantic': {'3': 900, '11': 100}}},
        'static_props': ['static.prop.busstop', 'static.prop.bench01']}
    cat = C.catalogue_from_probe(probe, measured='test')
    assert [b['path'].rsplit('.', 1)[1] for b in cat['buildings']] == ['SM_House01_Opt']   # one copy, no piece,
    assert cat['trees'][0]['size'] == [4.6, 4.8, 11.0] and cat['lamps'][0]['arm'] == 'x'  # no sunk skyscraper
    assert cat['tags'] == {'/Game/Carla/Static/Building/SM_House01.SM_House01': 3}
    assert cat['props'] == {'static.prop.busstop': 'bus_stop', 'static.prop.bench01': 'bench'}


def test_the_shipped_catalogue_is_measured():
    cat = C.load_catalogue()
    assert cat['carla'] == '0.9.16' and cat['measured'] and len(cat['buildings']) >= 30 and cat['trees']
    assert cat['lamps'] and cat['ground'] is None              # no plausible ground mesh (docs/worlds.md)
    for b in cat['buildings']:
        assert min(b['size'][:2]) >= 4.0 and abs(b['bottom']) <= 1.0


# ------------------------------------------------------------------------------------------------- runs
def _world(tmp_path, features, enabled=True):
    ws = tmp_path / 'proj.falcon'
    wd = ws / 'worlds' / 'w-city'
    wd.mkdir(parents=True)
    (wd / 'world.xodr').write_text(XODR)
    doc = C.plan(features, XODR, CATALOGUE, seed=0)
    (wd / 'city').mkdir()
    (wd / 'city' / 'plan.json').write_text(json.dumps(doc))
    return ws, {'id': 'w-city', 'name': 'City', 'kind': 'osm', 'xodr': 'worlds/w-city/world.xodr',
                'osm': {'handedness': 'LHT'}, 'traffic_profile': 'tp-default', 'weather': {'preset': 'ClearNoon'},
                'city': {'enabled': enabled, 'plan': 'city/plan.json', 'plan_sha256': C.plan_sha(doc),
                         'summary': doc['summary']}}, doc


def test_drive_config_carries_the_plan_only_when_on(project, tmp_path, features):
    from falcon_autonomy.core.drive import drive_config
    ws, w, doc = _world(tmp_path, features)
    project.doc['worlds'].append(w)
    cfg = drive_config(project, 'w-city', 'rig-default', str(tmp_path / 'r'), 2310, 8310, workspace=str(ws))
    assert cfg['city']['plan'] == str(ws / 'worlds' / 'w-city' / 'city' / 'plan.json')
    assert cfg['city']['plan_sha256'] == C.plan_sha(doc)
    w['city']['enabled'] = False
    assert 'city' not in drive_config(project, 'w-city', 'rig-default', str(tmp_path / 'r'), 2310, 8310,
                                      workspace=str(ws))
    w['city']['enabled'] = True
    os.remove(ws / 'worlds' / 'w-city' / 'city' / 'plan.json')
    cfg = drive_config(project, 'w-city', 'rig-default', str(tmp_path / 'r'), 2310, 8310, workspace=str(ws))
    assert 'city' not in cfg and any('city plan' in n for n in cfg['notes'])


def test_a_drive_spawns_the_city_and_tidies_it(project, tmp_path, features):
    from falcon_autonomy.core.drive import drive_config
    ws, w, doc = _world(tmp_path, features)
    project.doc['worlds'].append(w)
    cfg = drive_config(project, 'w-city', 'rig-default', str(tmp_path / 'run'), 2310, 8310, seconds=0.3, seed=2,
                       workspace=str(ws))
    p = tmp_path / 'run.cfg.json'
    p.write_text(json.dumps(cfg))
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy.workers', 'drive', str(p)], capture_output=True,
                       text=True, env=worker_env(), timeout=120)
    assert r.returncode == P.OK, r.stderr
    run = json.loads((tmp_path / 'run' / 'run.json').read_text())
    city = run['city']
    n = len(doc['items'])
    assert city['plan_sha256'] == C.plan_sha(doc)
    assert city['spawned'] + city['failed'] + city['missing_blueprints'] == n
    assert city['missing_blueprints'] == 1 and city['failed'] == 0      # the fake's library has no bus stop
    assert city['by_kind'] == doc['summary']['by_kind']


def test_city_only_build_worker(tmp_path):
    """The import worker's city-only mode: a world's folder dressed from a file, its summary written."""
    from falcon_autonomy.workers.osm_import import IMPORT_SCHEMA
    src = _osm(str(tmp_path / 'c.osm'))
    wd = tmp_path / 'w'
    wd.mkdir()
    (wd / 'world.xodr').write_text(XODR)
    (wd / 'import.json').write_text(json.dumps({'bbox': list(_bbox()), 'centre': [LAT0, LON0], 'source': src}))
    cfg = {'schema': IMPORT_SCHEMA, 'city_only': True, 'world_dir': str(wd), 'out': str(wd / 'city'), 'source': src,
           'seed': 4, 'options': {'lamps': 'off'}}
    p = tmp_path / 'c.json'
    p.write_text(json.dumps(cfg))
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy.workers', 'osm_import', str(p)], capture_output=True,
                       text=True, env=worker_env(), timeout=120)
    assert r.returncode == P.OK, r.stderr
    summ = json.loads((wd / 'city' / 'summary.json').read_text())
    assert summ['enabled'] and summ['plan'] == 'city/plan.json' and summ['seed'] == 4
    assert 'lamp' not in summ['summary']['by_kind'] and summ['summary']['by_kind']['building'] == {'osm': 2}
    plan = json.loads((wd / 'city' / 'plan.json').read_text())
    assert C.plan_sha(plan) == summ['plan_sha256']
    assert (wd / 'city' / 'run.json').is_file() and not (wd / 'run.json').exists()   # the import's record untouched


from test_gui import pump, qapp, slot_exceptions, window  # noqa: E402,F401


@pytest.mark.gui
def test_worlds_page_builds_and_switches_the_city(window, qapp, tmp_path):
    from falcon_autonomy.core.jobs import FINAL
    src = _osm(str(tmp_path / 'c.osm'))
    ws = window.workspace()
    wd = os.path.join(ws, 'worlds', 'w-city')
    os.makedirs(wd)
    with open(os.path.join(wd, 'world.xodr'), 'w') as fh:
        fh.write(XODR)
    with open(os.path.join(wd, 'import.json'), 'w') as fh:
        json.dump({'bbox': list(_bbox()), 'centre': [LAT0, LON0], 'source': src}, fh)
    window.project.doc['worlds'].append({'id': 'w-city', 'name': 'City', 'kind': 'osm',
                                         'xodr': 'worlds/w-city/world.xodr', 'osm': {'source': src},
                                         'weather': {'preset': 'ClearNoon'}, 'traffic_profile': 'tp-default'})
    page = window.go('worlds')
    page.refresh()
    page.list.setCurrentRow(len(window.project.doc['worlds']) - 1)
    pump(qapp, timeout=0.2)
    sec = page.city
    assert sec.isVisible() and not sec.on.isEnabled() and sec.build_btn.isEnabled()
    job = sec.build()
    assert pump(qapp, lambda: job.state in FINAL and 'city' in window.project.item('worlds', 'w-city'), 60), \
        job.log[-5:]
    c = window.project.item('worlds', 'w-city')['city']
    assert c['enabled'] and c['summary']['by_kind']['building'] == {'osm': 2}
    assert sec.on.isEnabled() and sec.on.isChecked() and 'building' in sec.kv.vals['Placed'].text()
    sec.on.setChecked(False)
    assert window.project.item('worlds', 'w-city')['city']['enabled'] is False
    window.undo.undo()
    assert window.project.item('worlds', 'w-city')['city']['enabled'] is True
