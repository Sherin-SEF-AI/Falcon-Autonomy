"""Indian road assets within the packaged CARLA (core.assets, core.roadfeatures, workers.assets): the measured
inventory and the fake's agreement with it, the stand-ins and proxies chosen by size, the class limits in traffic and
scenarios, the subclass in the labels, speed breakers in OpenDRIVE (and from OSM tags), the speed-breaker rule, the
Worlds page's road features, and the CLI."""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import worker_env
from falcon_autonomy.core import assets as A
from falcon_autonomy.core import protocol as P
from falcon_autonomy.core import roadfeatures as RF
from falcon_autonomy.core import scenarios as SC
from falcon_autonomy.core.drive import scenario_config
from test_gui import pump, qapp, slot_exceptions, window  # noqa: F401 (fixtures)


@pytest.fixture(scope='module')
def inv():
    return A.load_inventory()


# ------------------------------------------------------------------------------------------------ the inventory
def test_the_measured_inventory(inv):
    assert inv['schema'] == A.INVENTORY_SCHEMA and inv['server'].startswith('0.9.16')
    assert (len(inv['vehicles']), len(inv['walkers']), len(inv['props'])) == (41, 52, 98)
    assert all(r['spawned'] for r in inv['vehicles'] + inv['walkers'] + inv['props'])
    assert A.sizes(inv, 'vehicles')['vehicle.micro.microlino'] == (2.207, 1.481, 1.376)
    ms = inv['mesh_spawning']
    assert {'mesh_path', 'scale'} <= set(ms['blueprint'])
    scaled = next(t for t in ms['trials'] if t.get('scale') == 2.0)
    assert scaled['box']['size_m'] == pytest.approx([1.304, 1.304, 1.383], abs=2e-3)
    missing = ms['trials'][-1]
    assert 'does not exist' in missing['what'] and missing['box']['extent'][0] is None    # ±inf stored as null
    assert all(p.get('mesh_path', '').startswith('/Game/Carla/') for p in inv['props'])
    assert not any(b['id'].startswith(('animal', 'static.prop.cow')) for b in inv['blueprints'])


def test_the_fake_carla_agrees_with_the_inventory(inv):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'fake_carla'))
    import carla as fake
    meas = A.sizes(inv, 'vehicles')
    # the stand-ins (the fake's older entries keep the sizes earlier tests were written against: the ego
    # mkz_2020, the bus, the yzf, the cone and the barrier differ from the measured inventory; docs/indian-assets.md)
    for bid in ('vehicle.micro.microlino', 'vehicle.audi.a2', 'vehicle.citroen.c3', 'vehicle.tesla.model3'):
        assert tuple(fake.VEHICLES_ALL[bid][1:4]) == pytest.approx(meas[bid], abs=2e-3), bid
    for bid, row in fake.TWO_WHEELERS_NAMED.items():
        assert tuple(row[1:4]) == pytest.approx(meas[bid], abs=2e-3), bid
    props = {r['id']: r for r in inv['props']}
    for pid in ('static.prop.haybale', 'static.prop.foodcart', 'static.prop.maptable', 'static.prop.bench01'):
        _, ext, zc = fake.PROPS[pid]
        assert list(ext) == pytest.approx(props[pid]['box']['extent'], abs=1e-3)
        assert zc == pytest.approx(props[pid]['box']['center'][2], abs=1e-3)
    for path, (_, ext, zc) in fake.MESHES.items():
        p = next(r for r in inv['props'] if r.get('mesh_path') == path)
        assert list(ext) == pytest.approx(p['box']['extent'], abs=1e-3)


# ------------------------------------------------------------------------------------------------ choosing
def test_stand_ins_and_proxies_by_size(inv):
    auto = A.choose('autorickshaw', inv)
    assert auto['blueprint'] == 'vehicle.micro.microlino' and auto['label'] == 'autorickshaw (stand-in)'
    assert auto['score'] == pytest.approx(A.size_score((2.207, 1.481, 1.376), (2.635, 1.3, 1.7)), abs=1e-4)
    cyc = A.choose('cycle_rickshaw', inv)
    assert cyc['base_type'] == 'bicycle' and cyc['blueprint'] == 'vehicle.bh.crossbike'
    assert cyc['nearest_any'][0]['base_type'] == 'motorcycle'      # nearer, but a motorcyclist: refused by class
    cow = A.choose('cattle', inv)
    assert cow['blueprint'] == 'static.prop.mesh' and A.SCALE_RANGE[0] <= cow['scale'] <= A.SCALE_RANGE[1]
    cart = A.choose('handcart', inv)
    assert cart['blueprint'].startswith('static.prop.') and cart['scale'] == 1.0 and not cart['mesh_path']
    assert A.choose('street_vendor', inv)['blueprint'] == 'static.prop.foodcart'
    for k, spec in A.CLASSES.items():
        assert spec['label'].endswith(('(stand-in)', '(proxy)', '(proxy obstacle)')) and spec['size_source']


def test_best_scale_is_the_minimum():
    size, want = (1.573, 0.617, 1.181), (1.61, 0.44, 0.85)
    s = A.best_scale(size, want, 0.01, 100.0)
    grid = np.linspace(0.3, 1.5, 12001)
    best = grid[np.argmin([A.size_score([g * x for x in size], want) for g in grid])]
    assert s == pytest.approx(best, abs=2e-4)
    assert A.best_scale((10.0, 10.0, 10.0), (1.0, 1.0, 1.0)) == A.SCALE_RANGE[0]


def test_desired_speed_is_the_limit_capped_at_the_class():
    assert A.desired_speed_kmh(30.0, 10.0, 63.0) == pytest.approx(27.0)
    assert A.desired_speed_kmh(80.0, 0.0, 63.0) == 63.0
    assert A.desired_speed_kmh(30.0, -50.0, 15.0) == 15.0 and A.desired_speed_kmh(30.0, 0.0, None) == 30.0


def test_placement_rests_the_box_on_the_ground(inv):
    assert A.placement_z(inv, 'static.prop.haybale') == pytest.approx(0.5936, abs=1e-3)   # its origin at its centre
    assert A.placement_z(inv, 'static.prop.maptable', 0.5) == pytest.approx(0.0, abs=1e-3)


# ------------------------------------------------------------------------------------------------ road features
def test_profiles():
    e = RF.profile_entries([{'kind': 'irc_hump', 's': 100.0}, {'kind': 'table', 's': 200.0},
                            {'kind': 'dip', 's': 300.0}])
    h = lambda s: float(RF.height_at(e, s)[0])          # noqa: E731
    assert h(100.0) == pytest.approx(0.10) and h(98.15) == pytest.approx(0.0, abs=1e-9)
    assert h(101.85) == pytest.approx(0.0, abs=1e-9) and h(50.0) == 0.0
    # IRC:99's 17 m radius: the parabola's curvature at the top, 8h/L²
    assert 1.0 / (8 * 0.10 / 3.7 ** 2) == pytest.approx(17.1, abs=0.05)
    assert h(197.0) == pytest.approx(0.05) and h(200.0) == pytest.approx(0.10) and h(204.5) == pytest.approx(0.0)
    assert h(300.0) == pytest.approx(-0.08) and h(301.0) == 0.0
    s = np.linspace(0, 360, 36001)
    z = RF.height_at(e, s)
    assert np.max(np.abs(np.diff(z))) < 0.003     # continuous: no step (the steepest slope, the dip's 0.27, × 1 cm)


def test_apply_read_strip_and_skips():
    base = RF.test_road(360.0)
    text, placed, skipped = RF.apply(base, [{'road': '1', 's': 100.0, 'kind': 'irc_hump', 'source': 'x'},
                                            {'road': '1', 's': 101.0, 'kind': 'osm_bump'},
                                            {'road': '1', 's': 1.0, 'kind': 'dip'},
                                            {'road': '9', 's': 50.0, 'kind': 'dip'}])
    assert [p['kind'] for p in placed] == ['irc_hump'] and placed[0]['x'] == pytest.approx(100.0)
    assert {s['why'].split()[0] for s in skipped} == {'overlaps', 'the', 'no'}
    feats = RF.read(text)
    assert len(feats) == 1 and feats[0]['height_m'] == 0.1 and feats[0]['half_width_right_m'] == 3.5
    from falcon_autonomy.core import xodr as X
    assert X.parse(text).roads[0].length == 360.0                  # still a valid OpenDRIVE file for the parser
    back = RF.strip(text)
    assert RF.read(back) == [] and 'userData' not in back
    with pytest.raises(ValueError, match='unknown road feature'):
        RF.apply(base, [{'road': '1', 's': 100.0, 'kind': 'pothole'}])


def test_lht_conversion_keeps_the_elevation():
    from falcon_autonomy.core import xodr as X
    text, _, _ = RF.apply(RF.test_road(200.0), [{'road': '1', 's': 80.0, 'kind': 'irc_hump'}])
    lht, _ = X.to_lht(text)
    assert RF.read(lht)[0]['s'] == 80.0 and lht.count('<elevation ') == text.count('<elevation ')


def _osm_world(tmp_path, tags):
    """An imported world's folder: the straight test road and an OSM area whose way runs along it with tagged nodes
    [(x, value)]."""
    from falcon_autonomy.core import osm as O
    d = tmp_path / 'worlds' / 'w-test'
    d.mkdir(parents=True)
    (d / 'world.xodr').write_text(RF.test_road(360.0))
    centre = (12.9716, 77.5946)
    pr = O.local_projection(*centre)
    nodes, nd_tags = [], {}
    for k, x in enumerate(np.arange(0.0, 361.0, 20.0)):
        nodes.append((k + 1, *pr.to_geo(float(x), 0.0)))
    for j, (x, val) in enumerate(tags):
        nid = 1000 + j
        nodes.append((nid, *pr.to_geo(float(x), 0.0)))
        nd_tags[nid] = val
    order = sorted(nodes, key=lambda n: pr.to_local(n[1], n[2])[0])
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<osm version="0.6">']
    for nid, la, lo in nodes:
        t = f'<tag k="traffic_calming" v="{nd_tags[nid]}"/>' if nid in nd_tags else ''
        lines.append(f'<node id="{nid}" lat="{la:.9f}" lon="{lo:.9f}">{t}</node>')
    lines.append('<way id="7">' + ''.join(f'<nd ref="{n[0]}"/>' for n in order) +
                 '<tag k="highway" v="residential"/><tag k="name" v="Test Road"/></way>')
    lines.append('</osm>')
    (d / 'area.osm').write_text('\n'.join(lines))
    (d / 'import.json').write_text(json.dumps({'centre': list(centre)}))
    return d


def test_osm_traffic_calming_becomes_elevation(tmp_path):
    d = _osm_world(tmp_path, [(100.0, 'hump'), (180.0, 'bump'), (250.0, 'rumble_strip'), (300.0, 'table')])
    doc = RF.apply_to_world(str(d), {'osm': True, 'osm_bump': 'osm_bump'})
    kinds = sorted((round(p['s']), p['kind']) for p in doc['placed'])
    assert [k for _, k in kinds] == ['irc_hump', 'osm_bump', 'table']
    assert all(abs(s - x) <= 0.5 for (s, _), x in zip(kinds, (100, 180, 300)))
    assert doc['not_represented'] == {'rumble_strip': 1}
    assert (d / 'world.base.xodr').exists() and RF.read((d / 'world.xodr').read_text())
    # applied again with other choices: from the base, not on top of the first
    doc2 = RF.apply_to_world(str(d), {'osm': True, 'osm_bump': 'irc_hump', 'spacing_m': 110.0})
    k2 = [p['kind'] for p in doc2['placed']]
    assert k2.count('osm_bump') == 0 and len(RF.read((d / 'world.xodr').read_text())) == len(doc2['placed'])
    assert RF.remove_from_world(str(d)) and RF.read((d / 'world.xodr').read_text()) == []


def test_spaced_breakers_follow_irc_spacing():
    from falcon_autonomy.core import xodr as X
    od = X.parse(RF.test_road(360.0))
    pl = RF.spaced(od, 'irc_hump', 110.0)
    assert [round(p['s']) for p in pl] == [55, 165, 275]
    assert RF.mesh_for({'vertex_distance': 2.0}, True)['vertex_distance'] == RF.FEATURE_VERTEX_DISTANCE_M
    assert RF.mesh_for({'vertex_distance': 2.0}, False)['vertex_distance'] == 2.0


def test_world_entry_takes_the_feature_mesh():
    from falcon_autonomy.core.worlds import import_config, world_entry
    cfg = import_config('/x/area.osm', '/tmp/w', None, True)
    assert cfg['settings']['speed_breakers'] == 'osm'
    e = world_entry('w', 'W', cfg, {'road_features': {'placed': 3}})
    assert e['osm']['mesh']['vertex_distance'] == RF.FEATURE_VERTEX_DISTANCE_M
    assert world_entry('w', 'W', cfg, {})['osm']['mesh']['vertex_distance'] == 2.0


# ------------------------------------------------------------------------------------------------ the rule
def test_speed_breaker_rule():
    from falcon_autonomy.core import rules as R
    text, _, _ = RF.apply(RF.test_road(360.0), [{'road': '1', 's': 100.0, 'kind': 'irc_hump'},
                                                {'road': '1', 's': 200.0, 'kind': 'dip'}])
    t = np.arange(0.0, 35.0, 0.1)
    x = 20.0 + 8.0 * t                              # 28.8 km/h along +x in the right lane (CARLA y = +1.75)
    rs = R.load_ruleset('india')

    def run(speed):
        tr = R.Trace(t, x, np.full_like(t, 1.75), np.zeros_like(t), speed)
        tr.map.add_xodr(text, 'test', make_lanes=True)
        return next(r for r in R.check(tr, rs)['rules'] if r['id'] == 'speed_breaker')
    fast = run(np.full_like(t, 8.0))
    assert fast['state'] == 'violated' and fast['count'] == 1           # the dip is no breaker
    v = fast['violations'][0]
    assert 9.4 <= v['start'] <= 9.8 and 10.2 <= v['end'] <= 10.8         # body over 98.15–101.85 m (± half the ego)
    slow = run(np.where((x > 90) & (x < 110), 6.0, 8.0))
    assert slow['state'] == 'satisfied'
    tr = R.Trace(t, x, np.full_like(t, 1.75), np.zeros_like(t), np.full_like(t, 8.0))
    none = next(r for r in R.check(tr, rs)['rules'] if r['id'] == 'speed_breaker')
    assert none['state'] == 'not evaluable' and 'on_breaker' in none['missing_signals']


# ------------------------------------------------------------------------------------------------ runs on the fake
def _run(project, tmp_path, key, seed=1, values=None, name='run', labels=()):
    project.doc['rigs'][0]['sensors'][0].update(width=64, height=36, labels=list(labels))
    cfg = scenario_config(project, SC.builtin(key), 'world-town10', 'rig-default', str(tmp_path / name), 2310, 8310,
                          seed=seed, values=values)
    p = tmp_path / f'{name}.json'
    p.write_text(json.dumps(cfg))
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy.workers', 'drive', str(p)], capture_output=True,
                       text=True, env=worker_env(), timeout=240)
    return r, json.loads((tmp_path / name / 'run.json').read_text()), cfg


def test_stray_cattle_proxy_is_labelled(project, tmp_path):
    r, run, _ = _run(project, tmp_path, 'stray_cattle', values={'speed': 30.0, 'distance': 25.0, 'lateral': 0.0},
                     labels=('semantic', 'instance', 'depth', 'boxes2d', 'boxes3d'))
    assert r.returncode == P.OK, r.stderr
    sc = run['scenario']
    assert sc['blueprints']['cow'] == 'static.prop.mesh' and sc['subclasses']['cow'] == 'cattle (proxy obstacle)'
    aid = sc['actor_ids']['cow']
    rows = [json.loads(x) for x in (tmp_path / 'run' / 'actors.jsonl').read_text().splitlines()]
    cow = next(a for a in rows[0]['actors'] if a['id'] == aid)
    want = A.choose('cattle')
    assert cow['subclass'] == 'cattle (proxy obstacle)'
    # the true box: the scaled mesh's centre (CARLA reports it unscaled)
    assert 2 * cow['extent'][2] == pytest.approx(want['size_m'][2], abs=1e-3)
    assert cow['bbox_center'][2] == pytest.approx(cow['extent'][2], abs=1e-3)
    seen = []
    for f in sorted((tmp_path / 'run' / 'objects' / 'cam-front').iterdir()):
        seen += [o for o in json.loads(f.read_text())['objects'] if o.get('actor_id') == aid]
    assert seen and all(o['subclass'] == 'cattle (proxy obstacle)' for o in seen)
    assert {o['class'] for o in seen} == {'static'}                     # CARLA's class kept (tag 20)
    traj = json.loads((tmp_path / 'run' / 'trajectories.json').read_text())['actors']['cow']
    assert traj[0][6] == pytest.approx(want['size_m'][0], abs=1e-3) and traj[0][1] == traj[-1][1]   # stands still


def test_handcart_moves_at_walking_pace_with_its_pusher(project, tmp_path):
    r, run, _ = _run(project, tmp_path, 'handcart_kerb_lane', values={'speed': 25.0, 'distance': 40.0,
                                                                      'cart_speed': 5.0, 'lateral': 0.6})
    assert r.returncode == P.OK, r.stderr
    sc = run['scenario']
    assert sc['subclasses'] == {'cart': 'handcart (proxy)'} and sc['blueprints']['cart'].startswith('static.prop.')
    traj = json.loads((tmp_path / 'run' / 'trajectories.json').read_text())['actors']
    cart, pusher = np.array(traj['cart']), np.array(traj['pusher'])
    v = np.hypot(np.diff(cart[:, 1]), np.diff(cart[:, 2])) / np.diff(cart[:, 0])
    assert np.median(v) * 3.6 == pytest.approx(5.0, abs=0.05)
    assert np.hypot(*(pusher[-1, 1:3] - pusher[0, 1:3])) == pytest.approx(np.hypot(*(cart[-1, 1:3] - cart[0, 1:3])),
                                                                           abs=0.5)
    assert sc['checks'] and not any(c['name'] == 'no collision' and not c['ok'] for c in sc['checks'])


def test_autorickshaw_stand_in_keeps_its_class_limits(project, tmp_path):
    r, run, _ = _run(project, tmp_path, 'autorickshaw_cut_in', seed=4,
                     values={'speed': 40.0, 'faster': 30.0, 'start_offset': 0.0, 'cut_gap': 3.0, 'cut_time': 1.5})
    assert r.returncode == P.OK, r.stderr
    sc = run['scenario']
    assert sc['blueprints']['auto'] == 'vehicle.micro.microlino'
    assert sc['subclasses']['auto'] == 'autorickshaw (stand-in)'
    a = np.array(json.loads((tmp_path / 'run' / 'trajectories.json').read_text())['actors']['auto'])
    sp = np.hypot(a[:, 4], a[:, 5])
    assert sp.max() * 3.6 <= A.CLASSES['autorickshaw']['top_kmh'] + 0.5


def test_monsoon_variant_sets_its_weather_and_says_what_is_not_drawn(project, tmp_path):
    r, run, cfg = _run(project, tmp_path, 'monsoon_waterlogged', seed=2)
    assert r.returncode == P.OK, r.stderr
    assert cfg['weather']['precipitation_deposits'] == 100.0 and cfg['weather_source']['scenario'] == \
        'monsoon_waterlogged'
    assert not cfg.get('notes')                       # a town: puddles are drawn
    from falcon_autonomy.core.project import Project
    from falcon_autonomy.core.worlds import import_config, world_entry
    d = tmp_path / 'ws' / 'worlds' / 'w1'
    d.mkdir(parents=True)
    (d / 'world.xodr').write_text(RF.test_road(300.0))
    project.doc['worlds'].append(world_entry('w1', 'Imported', import_config('/x.osm', str(d), None, False), {}))
    c2 = scenario_config(project, SC.builtin('monsoon_waterlogged'), 'w1', 'rig-default', str(tmp_path / 'r2'), 2310,
                         8310, seed=2, workspace=str(tmp_path / 'ws'))
    assert any('not drawn on an imported' in n for n in c2['notes'])


def test_indian_traffic_gets_class_limits_and_subclasses(project, tmp_path):
    from falcon_autonomy.core.drive import drive_config
    from falcon_autonomy.core.traffic import preset
    prof = preset('tp-india-mixed')
    prof.update(vehicles=12, walkers=0, handedness='map')
    prof['fleet'] = {f'blueprint:{prof["fleet_classes"] and next(iter(prof["fleet_classes"])).split(":", 1)[1]}': 1.0}
    project.doc['traffic_profiles'].append(prof)
    project.doc['rigs'][0]['sensors'][0].update(width=64, height=36, labels=[])
    cfg = drive_config(project, 'world-town10', 'rig-default', str(tmp_path / 'd'), 2310, 8310, seconds=2.0,
                       traffic_id=prof['id'])
    p = tmp_path / 'd.json'
    p.write_text(json.dumps(cfg))
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy.workers', 'drive', str(p)], capture_output=True,
                       text=True, env=worker_env(), timeout=240)
    assert r.returncode == P.OK, r.stderr
    run = json.loads((tmp_path / 'd' / 'run.json').read_text())
    n = run['traffic']['stand_ins']['autorickshaw']
    assert n >= 1 and run['traffic']['limits']['autorickshaw']['top_kmh'] == A.CLASSES['autorickshaw']['top_kmh']
    assert sum(1 for v in run['subclasses'].values() if v == 'autorickshaw (stand-in)') == n
    rows = [json.loads(x) for x in (tmp_path / 'd' / 'actors.jsonl').read_text().splitlines()]
    tagged = [a for a in rows[0]['actors'] if a.get('subclass') == 'autorickshaw (stand-in)']
    assert len(tagged) == n and all(a['type_id'] == 'vehicle.micro.microlino' for a in tagged)


def test_the_assets_worker_inventory_on_the_fake(tmp_path):
    cfg = {'schema': 'falcon.assets/1', 'mode': 'inventory', 'port': 2310, 'tm_port': 8310, 'out': str(tmp_path)}
    (tmp_path / 'a.json').write_text(json.dumps(cfg))
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy.workers', 'assets', str(tmp_path / 'a.json')],
                       capture_output=True, text=True, env=worker_env(), timeout=120)
    assert r.returncode == P.OK, r.stderr
    doc = json.loads((tmp_path / 'inventory.json').read_text())
    micro = next(v for v in doc['vehicles'] if v['id'] == 'vehicle.micro.microlino')
    assert micro['box']['size_m'] == [2.207, 1.481, 1.376]
    trials = doc['mesh_spawning']['trials']
    box2 = next(t for t in trials if t.get('scale') == 2.0)
    assert box2['box']['size_m'] == pytest.approx([1.304, 1.304, 1.383], abs=2e-3)
    assert box2['box']['center'][2] == pytest.approx(0.3402)              # reported unscaled, as CARLA does
    assert math.isinf(trials[-1]['box']['extent'][0]) or trials[-1]['box']['extent'][0] is None


def test_cli_list_and_show(capsys):
    from falcon_autonomy.cli import main
    assert main(['assets', 'list']) == 0
    out = capsys.readouterr().out
    assert 'autorickshaw (stand-in)' in out and 'vehicle.micro.microlino' in out and 'cattle (proxy obstacle)' in out
    assert main(['assets', 'show', 'autorickshaw']) == 0
    assert 'bajajauto.com' in capsys.readouterr().out


# ------------------------------------------------------------------------------------------------ the Worlds page
def test_worlds_page_road_features(window, qapp, tmp_path, slot_exceptions):  # noqa: F811
    from falcon_autonomy.core.project import Edit
    from falcon_autonomy.core.worlds import import_config, world_entry
    window.project.path = str(tmp_path / 'proj.falcon.json')
    window.save()
    ws = window.workspace()
    d = _osm_world(tmp_path, [(100.0, 'hump')])
    target = os.path.join(ws, 'worlds', 'w-test')
    os.makedirs(os.path.dirname(target), exist_ok=True)
    os.rename(d, target)
    w = world_entry('w-test', 'Test place', import_config(os.path.join(target, 'area.osm'), target, None, False), {})
    window.edit(Edit('insert', ['worlds'], w, text='add'))
    page = window.go('worlds')
    page.list.setCurrentRow(page.list.count() - 1)
    assert page.features.isVisibleTo(page) and 'none' in page.features.kv.vals['Placed'].text()
    page.features.spacing.setValue(110.0)
    page.features.apply()
    doc = json.load(open(os.path.join(target, 'features.json')))
    assert doc['counts'].get('irc_hump', 0) >= 3
    cur = window.project.doc['worlds'][-1]['osm']
    assert cur['mesh']['vertex_distance'] == RF.FEATURE_VERTEX_DISTANCE_M and cur['road_features']['spacing_m'] == 110
    assert 'irc hump' in page.features.kv.vals['Placed'].text()
    page.features.remove()
    assert window.project.doc['worlds'][-1]['osm']['mesh']['vertex_distance'] == 2.0
    assert not os.path.exists(os.path.join(target, 'features.json'))
