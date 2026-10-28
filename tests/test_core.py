"""The process-neutral core: project format, edits, provenance, budget, weather, labels, settings, catalogue."""
import json
import os
import subprocess

import pytest

from falcon_autonomy.core import budget, labels, protocol as P, weather
from falcon_autonomy.core.catalog import Catalog
from falcon_autonomy.core.paths import tilde, workspace_for
from falcon_autonomy.core.project import Edit, Project, ProjectError, migrate, new_project, validate
from falcon_autonomy.core.provenance import block, hash_tree, sha256_json
from falcon_autonomy.core.settings import RESERVED_PORTS, Secrets, Settings, port_allowed


# ------------------------------------------------------------------------------------------------ project
def test_new_project_is_valid_and_round_trips(tmp_path):
    p = Project.new('Bengaluru study')
    assert validate(p.doc) == []
    path = str(tmp_path / 'b')
    p.save(path)
    assert p.path.endswith('.falcon.json') and os.path.isfile(p.path)
    q = Project.load(p.path)
    assert q.doc['name'] == 'Bengaluru study' and not q.dirty
    assert q.workspace == workspace_for(p.path) and q.workspace.endswith('b.falcon')


def test_schema_errors_are_readable():
    d = new_project('x')
    d['rigs'][0]['sensors'][0]['type'] = 'camera.magic'
    d['worlds'][0].pop('town')
    errs = validate(d)
    assert any(e.startswith('rigs/0/sensors/0/type') for e in errs)
    assert any('town' in e for e in errs)


def test_save_refuses_an_invalid_project(tmp_path):
    p = Project.new('x')
    p.doc['traffic_profiles'][0]['vehicles'] = -1
    with pytest.raises(ProjectError):
        p.save(str(tmp_path / 'x.falcon.json'))


def test_migration_from_version_0(tmp_path):
    v0 = {'format': 'falcon.project', 'version': 0, 'name': 'old', 'worlds': [],
          'rigs': [{'id': 'r1', 'name': 'r', 'vehicle': 'vehicle.audi.a2',
                    'cameras': [{'id': 'c1', 'name': 'c', 'pose': {'x': 0, 'y': 0, 'z': 1, 'roll': 0, 'pitch': 0,
                                                                     'yaw': 0}, 'width': 64, 'height': 48,
                                 'fov': 90}]}]}
    path = tmp_path / 'old.falcon.json'
    path.write_text(json.dumps(v0))
    p = Project.load(str(path))
    assert p.doc['version'] == 1 and p.dirty and p.notes == ['migrated from format version 0 to 1']
    assert p.doc['rigs'][0]['sensors'][0]['type'] == 'camera.pinhole'
    assert validate(p.doc) == []


def test_a_newer_format_is_refused():
    with pytest.raises(ProjectError, match='Update Falcon'):
        migrate({'format': 'falcon.project', 'version': 99})
    with pytest.raises(ProjectError):
        migrate({'format': 'something.else', 'version': 1})


def test_edits_undo_and_redo():
    p = Project.new('x')
    fov = ['rigs', 0, 'sensors', 0, 'fov']
    p.apply(Edit('set', fov, 120.0))
    p.apply(Edit('insert', ['scenarios'], {'id': 's1', 'name': 'cut-in'}))
    p.apply(Edit('set', ['loop', 'note'], 'new key'))
    assert p.doc['loop']['note'] == 'new key'
    assert p.undo() and 'note' not in p.doc['loop']
    assert p.undo() and p.doc['scenarios'] == []
    assert p.undo() and p.doc['rigs'][0]['sensors'][0]['fov'] == 90.0
    assert not p.undo()
    assert p.redo() and p.doc['rigs'][0]['sensors'][0]['fov'] == 120.0
    assert p.redo() and p.doc['scenarios'][0]['id'] == 's1'
    p.apply(Edit('remove', ['scenarios'], index=0))
    assert not p.redo()                       # a new edit clears the redo history
    assert p.undo() and p.doc['scenarios'][0]['name'] == 'cut-in'


def test_edits_copy_their_values():
    p = Project.new('x')
    v = {'id': 'm1', 'name': 'det'}
    p.apply(Edit('insert', ['models'], v))
    v['name'] = 'changed outside'
    assert p.doc['models'][0]['name'] == 'det'


# ------------------------------------------------------------------------------------------------ provenance, paths
def test_provenance_block_hashes_inputs(tmp_path):
    f = tmp_path / 'in.json'
    f.write_text('{"a": 1}')
    b = block({'config': str(f), 'gone': str(tmp_path / 'missing')}, seed=7, settings={'k': 1})
    import hashlib
    assert b['inputs']['config']['sha256'] == hashlib.sha256(b'{"a": 1}').hexdigest()
    assert b['inputs']['gone']['missing'] is True and b['seed'] == 7 and b['app'] == 'Falcon Autonomy'
    assert sha256_json({'b': 1, 'a': 2}) == sha256_json({'a': 2, 'b': 1})
    (tmp_path / 'd').mkdir()
    (tmp_path / 'd' / 'x.txt').write_text('x')
    (tmp_path / 'd' / 'run.json').write_text('{}')
    assert list(hash_tree(str(tmp_path / 'd'))) == ['x.txt']


def test_tilde():
    home = os.path.expanduser('~')
    assert tilde(os.path.join(home, 'a', 'b')) == os.path.join('~', 'a', 'b')
    assert tilde('/opt/x') == '/opt/x'


# ------------------------------------------------------------------------------------------------ budget
def test_render_budget():
    rig = {'sensors': [{'type': 'camera.pinhole', 'width': 1280, 'height': 720, 'labels': ['semantic']},
                       {'type': 'lidar'}]}
    # any label layer: the instance and the depth camera beside the image (each rendered once)
    assert budget.rig_mpx(rig) == pytest.approx(1280 * 720 * 3 / 1e6)
    s = {'type': 'camera.pinhole', 'width': 1000, 'height': 1000, 'labels': ['instance', 'boxes2d', 'depth']}
    assert budget.sensor_mpx(s) == pytest.approx(3.0)
    assert budget.sensor_mpx(dict(s, labels=[])) == pytest.approx(1.0)
    # six full-size fisheyes with instance + depth, written as rigs write them (the lens block): refused
    fe = {'type': 'camera.fisheye', 'width': 1920, 'height': 1536, 'labels': ['instance', 'depth'],
          'lens': {'model': 'equidistant', 'fov': 190.0, 'f_px': 900.0}}
    big = budget.rig_mpx({'sensors': [fe] * 6})
    assert big > 300
    ok, why = budget.check_budget({'sensors': [fe] * 6})
    assert not ok and 'over the budget' in why
    half = dict(fe, width=960, height=768, lens=dict(fe['lens'], f_px=450.0))
    assert 70 < budget.rig_mpx({'sensors': [half] * 6}) < 90 and budget.check_budget({'sensors': [half] * 6})[0]


def test_throughput_labels_typical_until_measured():
    t = budget.Throughput.from_settings({})
    assert not t.measured and t.estimate(100, 2.0)['label'] == 'typical' and 'typical' in t.source
    # one rendering load: no overhead can be told apart, all of the time is rendering
    m = t.add(1.84, 0.02, 'run A')
    assert m.measured and m.overhead_s == 0 and m.mpx_per_s == pytest.approx(92.0)
    assert m.estimate(100, 1.84)['label'] == 'measured'
    assert m.estimate(100, 77.0)['label'] == 'extrapolated from measured'
    # two loads: the line through them
    m2 = m.add(20.0, 0.2016, 'run B')
    assert m2.overhead_s == pytest.approx(0.0016, abs=1e-9) and m2.mpx_per_s == pytest.approx(100.0, rel=1e-6)
    back = budget.Throughput.from_settings(m2.to_dict())
    assert back.measured and back.mpx_per_s == pytest.approx(m2.mpx_per_s) and len(back.points) == 2
    assert 'measured on this machine (2 drives' in back.source


# ------------------------------------------------------------------------------------------------ weather, labels
def test_weather_resolution_and_fog_visibility():
    d = weather.resolve({'preset': 'HardRainNoon', 'fog_density': 35})
    assert d['precipitation'] == 100 and d['fog_density'] == 35
    assert weather.fog_visibility_m(35) == pytest.approx(200.0)
    assert weather.fog_visibility_m(0) is None
    assert 'fog' in weather.describe({'preset': 'ClearNoon', 'fog_density': 50})
    with pytest.raises(KeyError):
        weather.resolve({'preset': 'Nope'})


@pytest.mark.carla_python
def test_weather_and_tags_match_the_real_wheel(carla_python):
    code = ('import carla, json\n'
            f'F = {weather.FIELDS!r}\n'
            'out = {"presets": {n: [getattr(getattr(carla.WeatherParameters, n), f) for f in F] for n in '
            f'{list(weather.PRESETS)!r}' + '}, "tags": {int(v): k for k, v in carla.CityObjectLabel.names.items()}}\n'
            'print(json.dumps(out))')
    r = subprocess.run([carla_python, '-I', '-c', code], capture_output=True, text=True, timeout=60)
    real = json.loads(r.stdout)
    for name, vals in real['presets'].items():
        assert vals == pytest.approx(list(weather.preset_values(name).values()), abs=1e-4), name
    for k, v in real['tags'].items():
        if int(k) != 255:
            assert labels.CARLA_NAMES[int(k)] == v


def test_labels_palette():
    import numpy as np
    t = np.array([[1, 14], [11, 200]], np.uint8)
    rgb = labels.colorize(t)
    assert tuple(rgb[0, 0]) == (128, 64, 128) and tuple(rgb[0, 1]) == (0, 0, 142)
    assert tuple(rgb[1, 1]) == (255, 0, 255) and labels.name(24) == 'road line'


# ------------------------------------------------------------------------------------------------ settings
def test_reserved_ports_need_the_users_choice(tmp_path):
    s = Settings(str(tmp_path / 's.json'))
    assert 2000 in RESERVED_PORTS and 8100 in RESERVED_PORTS
    assert port_allowed(2310, s) and not port_allowed(2000, s) and not port_allowed(8100, s)
    s.set('allowed_ports', [2000])
    assert port_allowed(2000, Settings(s.path))
    assert not port_allowed(80, s)


def test_secrets_are_private_and_apart_from_settings(tmp_path):
    sec = Secrets(str(tmp_path / 'secrets.json'))
    sec.set('jetson_password', 'hunter2')
    assert sec.get('jetson_password') == 'hunter2'
    assert os.stat(sec.path).st_mode & 0o777 == 0o600
    s = Settings(str(tmp_path / 'settings.json'))
    s.save()
    assert 'hunter2' not in (tmp_path / 'settings.json').read_text()
    sec.set('jetson_password', None)
    assert not sec.has('jetson_password')


def test_settings_live_in_the_throwaway_config():
    assert Settings().path.startswith(os.environ['XDG_CONFIG_HOME'])


# ------------------------------------------------------------------------------------------------ protocol
def test_protocol_lines_and_codes():
    assert P.parse_line('{"t": "progress", "f": 0.5}\n')['f'] == 0.5
    assert P.parse_line('plain text') == {'t': 'log', 'level': 'info', 'text': 'plain text'}
    assert P.parse_line('{"no": "type"}')['t'] == 'log'
    assert P.status_of(P.SERVER_LOST) == 'server_lost'
    for code in P.EXIT_CODES:
        assert f'    {code}  ' in P.__doc__
    assert P.describe_exit(-9) == 'killed by signal 9'


def test_emitter_progress_has_an_eta():
    import io
    import time
    buf = io.StringIO()
    e = P.Emitter(buf, min_interval=0.0)
    e.progress(0.1, 'a')
    time.sleep(0.05)
    e.progress(0.2, 'b')
    lines = [json.loads(x) for x in buf.getvalue().splitlines()]
    assert lines[0]['eta_s'] is None and lines[1]['eta_s'] > 0


# ------------------------------------------------------------------------------------------------ catalogue
def test_catalog_indexes_runs_and_rebuilds(tmp_path):
    ws = tmp_path / 'ws'
    run = ws / 'runs' / 'r1'
    run.mkdir(parents=True)
    (run / 'run.json').write_text(json.dumps({'status': 'ok', 'worker': 'drive', 'started': '2026-01-01T00:00:00Z',
                                              'name': 'drive A', 'summary': {'steps': 10}}))
    (ws / 'runs' / 'junk').mkdir()
    c = Catalog(str(ws / 'catalog.sqlite'))
    assert c.rebuild(str(ws)) == 1
    r = c.list('run')[0]
    assert r['name'] == 'drive A' and r['summary']['steps'] == 10 and r['path'].endswith('r1')
    c.link('run:r1', 'dataset:d1', 'produced')
    assert c.links('run:r1')[0]['role'] == 'produced'
    c.close()
