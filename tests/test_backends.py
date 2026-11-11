"""CARLA backends (sim.backends, sim.server, sim.service, workers.compat): the capability table, folder and version
detection, the server command per backend, refusals before and during a run, semantic tags read at run time, the
cache key, `falcon backend list|check|probe`, the Settings section — and a smoke drive, a scenario sweep and a
benchmark route run on Town10HD_Opt through the unchanged CarlaService / studio / planbench paths on both fakes
(tests/fake_carla: 0.9.16; tests/fake_carla_ue5: the UE5 line). Real UE5 acceptance tests are marked ue5."""
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import FAKE, ROOT
from falcon_autonomy.core import protocol as P
from falcon_autonomy.core.settings import Settings
from falcon_autonomy.sim import backends as B
from falcon_autonomy.sim import server as S

FAKE_UE5 = os.path.join(ROOT, 'tests', 'fake_carla_ue5')
UE5_ROOT = os.path.join(FAKE_UE5, 'root')
FAKES = {'carla-0.9.16': FAKE, 'carla-ue5': FAKE_UE5}


def _env(fake, extra=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHON')}
    env['PYTHONPATH'] = os.pathsep.join([fake, ROOT])
    env['PYTHONUNBUFFERED'] = '1'
    env.update(extra or {})
    return env


def _worker(cfg: dict, path, fake, extra=None, worker='drive'):
    with open(path, 'w') as fh:
        json.dump(cfg, fh)
    r = subprocess.run([sys.executable, '-m', 'falcon_autonomy.workers', worker, str(path)], capture_output=True,
                       text=True, env=_env(fake, extra), timeout=300)
    run = json.loads(open(os.path.join(cfg['out'], 'run.json')).read())
    return r, run


def _cli(*a, config=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHONPATH')}
    if config:
        env['XDG_CONFIG_HOME'] = str(config)
    return subprocess.run([sys.executable, '-m', 'falcon_autonomy', *a], capture_output=True, text=True, env=env,
                          timeout=600)


def _venv_python(tmp_path, fake) -> str:
    """A stand-in for a CARLA venv's python: this interpreter with the fake carla importable (check_client runs it
    with -I, which the wrapper drops so PYTHONPATH reaches it)."""
    py = tmp_path / 'venv' / 'bin' / 'python'
    py.parent.mkdir(parents=True)
    py.write_text(f'#!/bin/sh\n[ "$1" = "-I" ] && shift\nPYTHONPATH="{fake}" exec "{sys.executable}" "$@"\n')
    py.chmod(0o755)
    return str(py)


def _root_0916(tmp_path) -> str:
    root = tmp_path / 'CARLA_0.9.16'
    (root / os.path.dirname(S.BINARY)).mkdir(parents=True)
    (root / S.BINARY).write_text('')
    return str(root)


# ------------------------------------------------------------------------------------------------- the table
def test_capability_table_and_detection(tmp_path):
    assert set(B.BACKENDS) == {'carla-0.9.16', 'carla-ue5'} and B.DEFAULT == 'carla-0.9.16'
    old, ue5 = B.get('carla-0.9.16'), B.get('carla-ue5')
    assert old.binary == S.BINARY and old.measured and all(old.caps[c] for c in B.CAPS)
    assert ue5.binary.endswith('CarlaUnreal-Linux-Shipping') and ue5.project == 'CarlaUnreal' and not ue5.measured
    assert ue5.maps == ('Town10HD_Opt', 'Mine_01') and not ue5.cap('ros2') and not ue5.cap('osm2odr')
    with pytest.raises(B.BackendError):
        B.get('carla-0.8')
    root = _root_0916(tmp_path)
    assert B.detect_root(root) == S.is_carla_root(root) == 'carla-0.9.16'
    assert B.detect_root(UE5_ROOT) == S.is_carla_root(UE5_ROOT) == 'carla-ue5'
    assert S.is_carla_root(UE5_ROOT, 'carla-0.9.16') == '' and not S.is_carla_root(str(tmp_path))
    assert B.detect_version('0.9.16') == 'carla-0.9.16' and B.detect_version('0.10.0-12-g1') == 'carla-ue5'
    assert B.detect_version('0.9.15') == '' and B.detect(version='0.10.1') == 'carla-ue5'
    assert B.fake_backend(FAKE) == 'carla-0.9.16' and B.fake_backend(FAKE_UE5) == 'carla-ue5'
    # candidate folders per backend
    os.symlink(UE5_ROOT, tmp_path / 'Carla-0.10.0-Linux-Shipping')
    assert S.candidate_roots(str(tmp_path)) == [root]
    assert S.candidate_roots(str(tmp_path), backend='carla-ue5') == [str(tmp_path / 'Carla-0.10.0-Linux-Shipping')]


def test_server_command_per_backend(tmp_path):
    old = S.Server(_root_0916(tmp_path), 2310, 12.0, 'Low')
    cmd = old.command()
    # exactly what Falcon always started for 0.9.16
    i = cmd.index(os.path.join(old.root, S.BINARY))
    assert cmd[i:] == [os.path.join(old.root, S.BINARY), 'CarlaUE4', '-carla-rpc-port=2310', '-nosound',
                       '-RenderOffScreen', '-quality-level=Low']
    new = S.Server(UE5_ROOT, 2320, 10.0, 'Epic', backend='carla-ue5', texture_pool_mb=512)
    cmd = new.command()
    i = cmd.index(os.path.join(UE5_ROOT, B.get('carla-ue5').binary))
    assert cmd[i + 1:i + 5] == ['CarlaUnreal', '-carla-rpc-port=2320', '-nosound', '-RenderOffScreen']
    assert '-ini:Engine:[/Script/Engine.RendererSettings]:r.Streaming.PoolSize=512' in cmd
    assert '-ExecCmds=r.Streaming.PoolSize 512' in cmd
    assert '-ExecCmds=r.Streaming.PoolSize 512' not in S.Server(UE5_ROOT, backend='carla-ue5', no_render=True,
                                                               texture_pool_mb=512).command()
    with pytest.raises(RuntimeError, match='no native ROS 2'):
        S.Server(UE5_ROOT, backend='carla-ue5', ros2=True).command()
    assert '--ros2' in S.Server(old.root, ros2=True).command()
    # a folder of the other backend is refused before anything starts
    with pytest.raises(RuntimeError, match='not CARLA 0.10'):
        S.Server(old.root, 2330, backend='carla-ue5').start()
    argv, env = S.worker_command(sys.executable, 'probe', ['--port', 2310], scope=False, backend='carla-ue5')
    assert env['FALCON_BACKEND'] == 'carla-ue5' and 'FALCON_BACKEND' not in S.worker_command(
        sys.executable, 'probe', scope=False)[1]


def test_check_client_by_backend(tmp_path):
    ue5 = S.check_client(_venv_python(tmp_path / 'a', FAKE_UE5), backend='carla-ue5')
    assert ue5['ok'] and ue5['carla_version'] == '0.10.0', ue5
    wrong = S.check_client(_venv_python(tmp_path / 'b', FAKE_UE5))
    assert not wrong['ok'] and 'not CARLA 0.9.16' in wrong['text']
    assert S.check_client(_venv_python(tmp_path / 'c', FAKE))['ok']


def test_settings_choose_the_root_and_python_per_backend(tmp_path, settings):
    from falcon_autonomy.core.jobs import JobQueue, SubprocessLauncher
    from falcon_autonomy.sim.service import CarlaService
    old = _root_0916(tmp_path)
    settings.set('carla_root', old, save=False)
    settings.set('client_python', '/old/python', save=False)
    settings.set('carla_roots', {'carla-ue5': UE5_ROOT}, save=False)
    settings.set('client_pythons', {'carla-ue5': '/ue5/python'}, save=False)
    svc = CarlaService(settings, JobQueue(SubprocessLauncher(), str(tmp_path / 'locks')))
    try:
        assert (svc.backend, svc.carla_root, svc.python) == ('carla-0.9.16', old, '/old/python')
        settings.set('default_backend', 'carla-ue5', save=False)
        assert (svc.backend, svc.carla_root, svc.python) == ('carla-ue5', UE5_ROOT, '/ue5/python')
        srv = svc.ensure_server_object()
        assert srv.backend == 'carla-ue5' and srv.root == UE5_ROOT
        svc.use_backend('carla-0.9.16')
        assert svc.backend == 'carla-0.9.16' and svc.ensure_server_object().backend == 'carla-0.9.16'
        spec = svc.worker_spec('probe', [], 'p', 'probe')
        assert spec.env['FALCON_BACKEND'] == 'carla-0.9.16'
        settings.set('texture_pool_mb', 512, save=False)
        assert svc.ensure_server_object().texture_pool_mb == 512
    finally:
        svc.shutdown()
    fake = CarlaService(settings, JobQueue(SubprocessLauncher(), str(tmp_path / 'locks2')), fake_carla=FAKE_UE5)
    assert fake.backend == 'carla-ue5' and fake.python == sys.executable
    fake.shutdown()


def test_check_needs_refusals():
    assert B.check_needs('carla-0.9.16', {'town': 'Town05', 'opendrive_world': True, 'ros2': True}, probe={}) == []
    r = B.check_needs('carla-ue5', {'town': 'Town05'}, probe={})
    assert len(r) == 1 and 'no map Town05' in r[0] and 'Town10HD_Opt, Mine_01' in r[0]
    assert B.check_needs('carla-ue5', {'town': 'Town10HD_Opt'}, probe={}) == []
    r = B.check_needs('carla-ue5', {'opendrive_world': True, 'ros2': True, 'osm2odr': True}, probe={})
    assert len(r) == 3 and all('falcon backend probe' in x for x in r)
    # a probe's maps and API capabilities replace the declared ones (never opendrive_world)
    p = {'maps': ['Town10HD_Opt', 'Town03'], 'caps': {'osm2odr': True, 'lane_offset': False}}
    assert B.check_needs('carla-ue5', {'town': 'Town03', 'osm2odr': True}, probe=p) == []
    assert B.check_needs('carla-ue5', {'lane_offset': True, 'opendrive_world': True}, probe=p) != []
    cfg = {'town': 'Town01', 'traffic': {'lateral_filtering': {'enabled': True}}}
    assert B.needs_of(cfg) == {'town': 'Town01', 'lane_offset': True}
    assert B.needs_of({'xodr': '/w.xodr', 'town': ''}) == {'opendrive_world': True}
    assert B.needs_of({'schema': 'falcon.osm_import/1'}) == {'osm2odr': True}


# ------------------------------------------------------------------------------------------------- the adapter
@pytest.fixture
def ue5_module(monkeypatch):
    monkeypatch.syspath_prepend(FAKE_UE5)
    sys.modules.pop('carla', None)
    import carla
    carla.reset()
    yield carla
    sys.modules.pop('carla', None)


class _W:
    def __init__(self):
        self.notes = []

    def note(self, t):
        self.notes.append(t)


def test_compat_aliases_tags_weather_and_attributes(ue5_module, monkeypatch):
    from falcon_autonomy.core import groundtruth as GT
    from falcon_autonomy.workers import compat as C
    carla = ue5_module
    monkeypatch.delenv('FALCON_BACKEND', raising=False)
    w = _W()
    c = C.attach(w, carla)
    assert c.backend_id == 'carla-ue5' and not c.default and C.active() is c and w.compat is c
    lib = carla.Client().get_world().get_blueprint_library()
    assert lib.find('vehicle.lincoln.mkz_2020').id == 'vehicle.lincoln.mkz'             # an alias
    assert lib.find('vehicle.micro.microlino').id == 'vehicle.dodge.charger'            # same base type, first by id
    assert len(lib.filter('vehicle.mini.cooper_s')) == 1                                # filter by an alias
    with pytest.raises(IndexError):
        lib.find('static.prop.nothing')
    assert c.substitutions == {'vehicle.lincoln.mkz_2020': 'vehicle.lincoln.mkz',
                               'vehicle.micro.microlino': 'vehicle.dodge.charger'}
    # tags: read at run time, mapped to Falcon's classes by name
    raw = np.array([carla.CityObjectLabel.Car, carla.CityObjectLabel.Roads, carla.CityObjectLabel.Pedestrians])
    assert raw.tolist() != [14, 1, 12] and c.canon_tags(raw).tolist() == [14, 1, 12]
    assert GT.tags_for('carla-0.9.16')['Car'] == 14 and GT.tags_for('carla-ue5') == {}
    assert GT.tags_for('carla-ue5', carla.CityObjectLabel)['Car'] == carla.CityObjectLabel.Car
    lut, unknown = GT.tag_lut({'Car': 3, 'TrafficSign': 9, 'Debris': 40, 'Unlabeled': 0})
    assert lut[3] == 14 and lut[9] == 8 and lut[40] == 0 and unknown == ['Debris']
    assert lut[GT.WEATHER_NOISE_CLASS] == GT.WEATHER_NOISE_CLASS
    # capabilities, weather, camera attributes, tone mapping
    assert not c.has('osm2odr') and not c.has('tm_osm') and c.has('lane_offset') and not c.has('ros2')
    assert not c.set_osm_mode(carla.Client().get_trafficmanager(8310), False)
    wp = carla.WeatherParameters()
    assert c.weather_missing(wp, {'dust_storm': 100.0, 'fog_density': 5.0}) == ['dust_storm']
    assert c.weather_missing(wp, {'dust_storm': 0.0}) == []
    c.missing_attribute('sensor.camera.rgb', 'lens_circle_falloff')
    c.missing_attribute('sensor.camera.rgb', 'image_size_x')                             # not a lens attribute
    assert not c.tonemap_measured()
    rep = c.report()
    assert rep['labels_read'] and len([n for n in rep['notes'] if 'lens_circle_falloff' in n]) == 1
    assert any('dust_storm' in n for n in rep['notes']) and any('vignette' in n for n in rep['notes'])
    assert w.notes == rep['notes']


def test_compat_is_the_identity_on_0916(fake_module):
    from falcon_autonomy.workers import compat as C
    w = _W()
    c = C.attach(w, fake_module)
    assert c.default and c.lut is None and c.report() is None and c.doc() == {'id': 'carla-0.9.16',
                                                                              'carla_client': '0.9.16'}
    tags = np.arange(5, dtype=np.uint8)
    assert c.canon_tags(tags) is tags and c.tonemap_measured() and c.has('osm2odr') and c.has('tm_osm')
    assert c.set_osm_mode(fake_module.Client().get_trafficmanager(8310), True)
    c.missing_attribute('sensor.camera.rgb', 'lens_k')
    c.weather_missing(fake_module.WeatherParameters(), {'dust_storm': 50.0, 'nope': 1.0})
    assert w.notes == []
    with pytest.raises(IndexError):                         # no aliases on 0.9.16
        fake_module.Client().get_world().get_blueprint_library().find('vehicle.lincoln.mkz')


# ------------------------------------------------------------------------------------------- workers on both fakes
def _drive_cfg(project, out, **kw):
    from falcon_autonomy.core.drive import drive_config
    cfg = drive_config(project, 'world-town10', 'rig-default', str(out), 2310, 8310, seconds=kw.pop('seconds', 0.5),
                       seed=kw.pop('seed', 3))
    cfg.update(kw)
    return cfg


def test_labels_are_falcons_classes_on_both_backends(project, tmp_path):
    rig = project.doc['rigs'][0]
    rig['sensors'][0].update(width=96, height=54, labels=['semantic', 'instance', 'depth', 'boxes2d'])
    rig['sensors'].append({'id': 'lid', 'name': 'top', 'type': 'lidar', 'channels': 16, 'points_per_second': 30000,
                           'pose': {'x': 0.0, 'y': 0.0, 'z': 1.9, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}})
    project.doc['traffic_profiles'][0].update(vehicles=0, walkers=0)
    sem, seg = {}, {}
    for bid, fake in FAKES.items():
        out = tmp_path / bid
        r, run = _worker(_drive_cfg(project, out), tmp_path / f'{bid}.json', fake)
        assert r.returncode == P.OK, r.stderr
        assert run['provenance']['backend']['id'] == bid
        sem[bid] = [open(f, 'rb').read() for f in sorted(glob.glob(str(out / 'semantic' / '**' / '*.png'),
                                                                     recursive=True))]
        seg[bid] = []
        for f in sorted(glob.glob(str(out / 'lidar' / 'lid' / '*.labels.bin'))):   # uint8 tags, then uint32 ids
            n = os.path.getsize(f) // 5
            seg[bid].append(np.fromfile(f, np.uint8)[:n])
    assert sem['carla-0.9.16'] and sem['carla-0.9.16'] == sem['carla-ue5']        # pixel for pixel
    assert seg['carla-0.9.16'] and all(np.array_equal(a, b) for a, b in zip(seg['carla-0.9.16'], seg['carla-ue5']))
    assert any((a == 1).any() for a in seg['carla-ue5'])


def test_fisheye_on_ue5_leaves_out_what_it_lacks(project, tmp_path):
    rig = project.doc['rigs'][0]
    rig['sensors'] = [{'id': 'cam-f', 'name': 'front_fisheye', 'type': 'camera.fisheye', 'width': 96, 'height': 80,
                       'labels': ['semantic', 'instance', 'depth'],
                       'pose': {'x': 2.0, 'y': 0.0, 'z': 1.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
                       'lens': {'model': 'equidistant', 'fov': 190.0, 'f_px': 30.0, 'k': [0.0, 0.0, 0.0, 0.0]}}]
    project.doc['traffic_profiles'][0].update(vehicles=0, walkers=0)
    r, run = _worker(_drive_cfg(project, tmp_path / 'f', seconds=0.2), tmp_path / 'f.json', FAKE_UE5)
    assert r.returncode == P.OK, r.stderr
    notes = run['compat']['notes']
    assert any('vignette correction is off' in n for n in notes)
    assert any('lens_k' in n and 'sensor.camera.rgb' in n for n in notes)
    r, run = _worker(_drive_cfg(project, tmp_path / 'g', seconds=0.2), tmp_path / 'g.json', FAKE)
    assert r.returncode == P.OK and 'compat' not in run and not any('lens' in n for n in run['notes'])


def test_refusals_at_run_time(project, tmp_path):
    project.doc['rigs'][0]['sensors'][0].update(width=32, height=18, labels=[])
    cfg = _drive_cfg(project, tmp_path / 'a', town='Town05')
    r, run = _worker(cfg, tmp_path / 'a.json', FAKE_UE5)
    assert r.returncode == P.REFUSED and 'has no map Town05' in run['error'] and 'Mine_01' in run['error']
    r, run = _worker(_drive_cfg(project, tmp_path / 'b', town='Town05'), tmp_path / 'b.json', FAKE)
    assert r.returncode == P.OK, r.stderr                       # 0.9.16: as before
    r, run = _worker(_drive_cfg(project, tmp_path / 'c', backend='carla-0.9.16'), tmp_path / 'c.json', FAKE_UE5)
    assert r.returncode == P.REFUSED and 'is for CARLA 0.9.16' in run['error']
    xodr = tmp_path / 'w.xodr'
    xodr.write_text('<OpenDRIVE/>')
    r, run = _worker(_drive_cfg(project, tmp_path / 'd', town='', xodr=str(xodr)), tmp_path / 'd.json', FAKE_UE5)
    assert r.returncode == P.REFUSED and 'imported (OpenDRIVE) world' in run['error']
    r, run = _worker(_drive_cfg(project, tmp_path / 'e'), tmp_path / 'e.json', FAKE,
                     extra={'FALCON_BACKEND': 'carla-ue5'})
    assert r.returncode == P.NO_CARLA and 'started for CARLA 0.10' in run['error']
    assert run['provenance']['backend']['id'] == 'carla-0.9.16'
    osm = {'schema': 'falcon.osm_import/1', 'source': str(tmp_path / 'x.osm'), 'out': str(tmp_path / 'osm')}
    r, run = _worker(osm, tmp_path / 'osm.json', FAKE_UE5, worker='osm_import')
    assert r.returncode == P.REFUSED and 'importing a place' in run['error']


def test_two_wheeler_filtering_runs_on_ue5(project, tmp_path):
    from falcon_autonomy.core import scenarios as SC
    from falcon_autonomy.core.drive import scenario_config
    project.doc['rigs'][0]['sensors'][0].update(width=48, height=32, labels=[])
    cfg = scenario_config(project, SC.builtin('two_wheeler_filtering'), 'world-town10', 'rig-default',
                          str(tmp_path / 'run'), 2310, 8310, seed=2)
    r, run = _worker(cfg, tmp_path / 'run.json', FAKE_UE5)
    assert r.returncode == P.OK, r.stderr
    assert run['provenance']['backend'] == {'id': 'carla-ue5', 'carla_client': '0.10.0'}


def test_cache_key_separates_backends(project, tmp_path):
    from falcon_autonomy.core import datasets as D
    cfg = _drive_cfg(project, tmp_path / 'x')
    assert 'backend' not in cfg
    k = D.drive_key(cfg)
    assert D.drive_key(dict(cfg, backend='carla-0.9.16')) == k                # 0.9.16 keys as before
    assert D.drive_key(dict(cfg, backend='carla-ue5')) != k
    project.doc['simulator'] = {'backend': 'carla-ue5'}                       # the project names its simulator
    ue = _drive_cfg(project, tmp_path / 'y')
    assert ue['backend'] == 'carla-ue5' and D.drive_key(ue) != k
    assert D.drive_key(ue) == D.drive_key(dict(cfg, backend='carla-ue5'))
    del project.doc['simulator']
    st = Settings(str(tmp_path / 'settings.json'))
    st.set('default_backend', 'carla-ue5')                                    # or the setting, when it names none
    assert B.run_backend(project, st) == 'carla-ue5' and B.run_backend(project, Settings(str(tmp_path / 'n.json'))) \
        == 'carla-0.9.16'


# ------------------------------------------------------------------------------------------------- the unchanged paths
@pytest.mark.parametrize('bid', list(FAKES))
def test_smoke_scenario_and_benchmark_on_town10(project, tmp_path, bid):
    """The smoke drive (CarlaService), a scenario sweep (studio) and a benchmark route (planbench) on Town10HD_Opt,
    through the CLI, on each backend's fake."""
    fake = FAKES[bid]
    project.doc['rigs'][0]['sensors'][0].update(width=48, height=32, labels=[])
    path = str(tmp_path / 'p.falcon.json')
    project.save(path)
    ws = project.workspace
    r = _cli('--fake-carla', fake, 'smoke', path, '--seed', '1', '--max-wall', '4', config=tmp_path / 'cfg')
    assert r.returncode == 0, r.stdout + r.stderr
    run = json.load(open(os.path.join(r.stdout.strip().splitlines()[-1], 'run.json')))
    assert run['provenance']['backend']['id'] == bid and run['status'] == 'ok'
    assert ('compat' in run) == (bid != 'carla-0.9.16')
    r = _cli('sweep', path, '--scenario', 'lead_brake', '--name', 'lb', '--samples', '1', '--fake-carla', fake,
             config=tmp_path / 'cfg')
    assert r.returncode == 0, r.stdout + r.stderr
    res = [json.loads(x) for x in open(os.path.join(ws, 'studies', 'lb', 'results.jsonl'))]
    assert len(res) == 1 and res[0]['status'] == 'ok'
    runs = glob.glob(os.path.join(ws, 'studies', 'lb', 'runs', '*', 'run.json'))
    assert runs and json.load(open(runs[0]))['provenance']['backend']['id'] == bid
    r = _cli('bench', 'run', path, '--name', 'mini', '--set', 'falcon-pdm-mini', '--planner', 'none', '--route',
             't10-lead_brake-s1', '--boot', '20', '--fake-carla', fake, config=tmp_path / 'cfg')
    assert r.returncode == 0, r.stdout + r.stderr
    bdir = os.path.join(ws, 'benchmarks', 'mini')
    res = json.load(open(os.path.join(bdir, 'results.json')))
    assert res['planners']['none']['n_routes'] == 1 and res['planners']['none']['partial_routes'] == 0
    brun = json.load(open(os.path.join(bdir, 'runs', 'none', 't10-lead_brake-s1', 'run.json')))
    assert brun['provenance']['backend']['id'] == bid and brun['status'] == 'ok'


# ------------------------------------------------------------------------------------------------- the command
def test_backend_list_check_and_probe(tmp_path):
    cfg = tmp_path / 'cfg'
    r = _cli('backend', 'list', '--json', config=cfg)
    assert r.returncode == 0, r.stderr
    rows = {x['id']: x for x in json.loads(r.stdout)}
    assert rows['carla-0.9.16']['default'] and rows['carla-ue5']['maps'] == ['Town10HD_Opt', 'Mine_01']
    assert rows['carla-ue5']['caps']['osm2odr'] is False and rows['carla-ue5']['probed'] == ''
    r = _cli('backend', 'list', config=cfg)
    assert r.returncode == 0 and 'carla-ue5: CARLA 0.10 (Unreal Engine 5)' in r.stdout
    py = _venv_python(tmp_path, FAKE_UE5)
    r = _cli('backend', 'check', 'carla-ue5', '--root', UE5_ROOT, '--python', py, config=cfg)
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'holds CarlaUnreal' in r.stdout and 'carla 0.10.0' in r.stdout
    r = _cli('backend', 'check', 'carla-0.9.16', '--root', UE5_ROOT, '--python', py, '--json', config=cfg)
    out = json.loads(r.stdout)
    assert r.returncode == P.FAILED and not out['ok'] and 'is a CARLA 0.10' in out['rows'][0][2]
    r = _cli('backend', 'probe', 'carla-ue5', '--fake-carla', FAKE_UE5, '--json', config=cfg)
    assert r.returncode == 0, r.stdout + r.stderr
    info = json.loads(r.stdout)
    assert info['maps'] == ['Mine_01', 'Town10HD_Opt'] and 'vehicle.lincoln.mkz' in info['blueprints']
    assert info['caps'] == {'ros2': False, 'osm2odr': False, 'tm_osm': False, 'lane_offset': True}
    assert 'dust_storm' not in info['weather_fields'] and len(info['labels']) == 29
    assert 'lens_k' not in info['sensor_attributes']['sensor.camera.rgb']
    saved = json.load(open(cfg / 'FalconAutonomy' / 'backends' / 'carla-ue5.json'))
    assert saved['probed'] and saved['backend'] == 'carla-ue5'
    r = _cli('backend', 'list', '--json', config=cfg)
    assert {x['id']: x for x in json.loads(r.stdout)}['carla-ue5']['probed'] == saved['probed']
    r = _cli('backend', 'probe', 'carla-ue5', '--fake-carla', FAKE, config=cfg)
    assert r.returncode == P.FAILED and 'is carla-0.9.16, not carla-ue5' in r.stderr
    r = _cli('backend', 'check', 'nope', config=cfg)
    assert r.returncode == P.FAILED and 'unknown backend' in r.stderr
    assert _cli('backend', '--help', config=cfg).returncode == 0


# ------------------------------------------------------------------------------------------------- the GUI
@pytest.fixture(scope='module')
def qapp():
    from falcon_autonomy.ui.app import make_app
    from PyQt6.QtCore import QThreadPool
    app = make_app()
    yield app
    QThreadPool.globalInstance().waitForDone(10000)
    app.processEvents()


@pytest.mark.gui
def test_settings_simulator_section(qapp, settings, tmp_path):
    from types import SimpleNamespace
    from falcon_autonomy.ui import settings_backend as SB
    page = SimpleNamespace(win=SimpleNamespace(settings=settings))
    sec = SB.section(page)
    assert sec.title == 'Simulator' and set(sec.roots) == {'carla-ue5'}
    assert 'CARLA 0.10 (Unreal Engine 5): maps Town10HD_Opt, Mine_01' in sec.table.text()
    sec.roots['carla-ue5'].setText(_root_0916(tmp_path))
    assert 'is a CARLA 0.9.16 folder' in sec.apply(settings)
    sec.roots['carla-ue5'].setText(str(tmp_path))
    assert 'not a CARLA 0.10' in sec.apply(settings)
    sec.roots['carla-ue5'].setText(UE5_ROOT)
    sec.pys['carla-ue5'].setText('/ue5/venv/bin/python')
    sec.default.setCurrentIndex(sec.default.findData('carla-ue5'))
    assert sec.apply(settings) == ''
    assert settings.get('default_backend') == 'carla-ue5'
    assert settings.get('carla_roots') == {'carla-ue5': UE5_ROOT}
    assert settings.get('client_pythons') == {'carla-ue5': '/ue5/venv/bin/python'}
    sec.deleteLater()


@pytest.mark.ue5
def test_real_ue5_backend_checks_and_probes(tmp_path):
    """Acceptance on a real CARLA 0.10 installation (FALCON_REAL_UE5=1; found in the usual places, its venv beside
    it): the folder and Python check, then a probe of Falcon's own server of that line."""
    r = _cli('backend', 'check', 'carla-ue5')
    assert r.returncode == 0, r.stdout + r.stderr
    r = _cli('backend', 'probe', 'carla-ue5', '--json', '--wait', '180', config=tmp_path / 'cfg')
    assert r.returncode == 0, r.stdout + r.stderr
    info = json.loads(r.stdout)
    assert info['backend'] == 'carla-ue5' and 'Town10HD_Opt' in info['maps'] and info['labels']
