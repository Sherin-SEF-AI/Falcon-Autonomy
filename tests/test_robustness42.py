"""4.2 robustness and compliance, the core: the director degrades when a lane ends before an actor's distance (no
IndexError; the regulation pack's cut-out runs on the fake carla), the sensitivity checks of an unverified value
(core.regpacks) and of a fault tolerance verdict (core.faulttol) as pure functions, a campaign's flags, the safety
case's fidelity rule (core.gsn: evidence from a fake, kinematic or mixed result verifies amber at best) and attach /
detach, and the monitor's dry run and rollback on a faked SSH layer (core.monitorgen)."""
import copy
import json
import os
import types

import pytest

from conftest import FAKE
from falcon_autonomy.core import faulttol as FT
from falcon_autonomy.core import gsn as G
from falcon_autonomy.core import monitorgen as MG
from falcon_autonomy.core import regpacks as RP


# ------------------------------------------------------------------------------------------ the director's lane end
def _director(carla, notes):
    from falcon_autonomy.workers import director as D
    d = D.Director.__new__(D.Director)
    d.carla, d.lht, d.placed_short = carla, False, []
    d.w = types.SimpleNamespace(note=notes.append)
    return d


def test_an_actor_beyond_the_lane_end_is_placed_at_its_end_and_said(fake_module):
    from falcon_autonomy.workers import director as D
    carla, notes = fake_module, []
    d = _director(carla, notes)
    wp = carla.Waypoint(-1, 100.0)                       # the fake's road ends at x = 150
    path, _off = d.actor_path({'id': 'target', 'kind': 'car', 'place': {'lane': 'ego', 'ahead_m': 120.0}}, wp)
    assert path.pts[0][0] == pytest.approx(150.0)        # at the lane's end, not an IndexError
    assert d.placed_short == [{'actor': 'target', 'lane': 'ego', 'ahead_m': 120.0, 'placed_m': 50.0}]
    assert 'lane ends 50 m' in notes[0] and 'not the one asked' in notes[0]
    # behind the start and in the oncoming lane: the same walk, the same note
    d.actor_path({'id': 'b', 'kind': 'car', 'place': {'lane': 'ego', 'ahead_m': -300.0}}, wp)
    assert d.placed_short[-1]['placed_m'] == -250.0
    d.actor_path({'id': 'o', 'kind': 'car', 'place': {'lane': 'opposite', 'ahead_m': 80.0}}, wp)
    assert d.placed_short[-1]['actor'] == 'o' and d.placed_short[-1]['placed_m'] == 50.0
    # inside the road: placed where asked, nothing noted
    n = len(notes)
    path, _ = d.actor_path({'id': 'near', 'kind': 'car', 'place': {'lane': 'ego', 'ahead_m': 30.0}}, wp)
    assert path.pts[0][0] == pytest.approx(130.0) and len(notes) == n
    at, got = D.walk(wp, 500.0)
    assert got == 50.0 and at.transform.location.x == pytest.approx(150.0)


def test_the_start_is_chosen_so_the_farthest_actor_fits(fake_module):
    carla, notes = fake_module, []
    d = _director(carla, notes)
    d.map, d.breakers = carla.Map('Town10HD_Opt'), []
    d.scn = {'name': 'far', 'needs': [], 'ego': {}, 'actors': [{'id': 't', 'kind': 'car',
                                                                 'place': {'lane': 'ego', 'ahead_m': 190.0}}]}
    spawns = [carla.Transform(carla.Location(x, 0.0, 0.3), carla.Rotation(0, 0, 0)) for x in (60.0, 0.0, -60.0)]
    order = [0, 1, 2]
    tf, wp = d.place_ego(spawns, order)
    assert tf.location.x == -60.0 and order == [0, 1]   # the only start 190 m from the road's end
    d.scn['actors'][0]['place']['ahead_m'] = 260.0      # none fits: the first suitable start, the actor degrades
    order = [0, 1, 2]
    tf, wp = d.place_ego(spawns, order)
    assert tf.location.x == 0.0 and order == [0, 2]      # (x = 60 has only 90 m of lane: not a start at all)


def test_a_moved_actor_makes_the_pack_incomplete():
    p = RP.pack('r157')
    row = RP._row(next(c for c in p['cases'] if c['id'] == 'stationary-20'), result='pass', word='passed',
                  placed_short=[{'actor': 'target', 'ahead_m': 40.0, 'placed_m': 30.0}])
    rows = [dict(RP._row(c), result='pass', word='passed') for c in p['cases'] if c['id'] != 'stationary-20']
    v = RP.verdict_of(p, {'cases': rows + [row]}, True)
    assert v['verdict'] == 'incomplete' and 'placed nearer than asked' in v['verdict_note']
    assert RP.verdict_of(p, {'cases': rows + [dict(row, placed_short=None)]}, True)['verdict'] == 'passed'


@pytest.fixture(scope='module')
def svc_project(tmp_path_factory):
    from falcon_autonomy.core.project import Project
    from falcon_autonomy.sim.service import study_service
    base = tmp_path_factory.mktemp('rb42')
    p = Project.new('RB42')
    p.doc['rigs'][0]['sensors'][0].update(width=48, height=32, labels=[])
    p.save(str(base / 'rb.falcon.json'))
    svc = study_service(workspace=p.workspace, fake_carla=FAKE, lock_dir=str(base / 'locks'))
    yield p, svc, base
    svc.shutdown()


def test_the_cut_out_case_runs_on_the_fake_and_a_far_actor_degrades(svc_project):
    """The report's crash (`falcon regpack run … --case cut-out-60-60-1`: IndexError in actor_path) on the fake
    carla: the case runs; a scenario whose actor is beyond every lane's end runs too, with the actor moved and said."""
    from falcon_autonomy.core import stack as ST
    from falcon_autonomy.sim.stacksuite import run_one
    p, svc, base = svc_project
    out = RP.result_dir(p.workspace, 'co')
    doc = RP.run_pack(svc, p, 'r157', out, p.doc['worlds'][0]['id'], p.doc['rigs'][0]['id'],
                      cases=['cut-out-60-60-1'], fps=10)
    r = doc['cases'][0]
    assert r['result'] == 'pass' and 'IndexError' not in json.dumps(r), r
    assert r['fidelity'] == 'fake' and doc['fidelity'] == 'fake'
    scn = RP.scn_stationary('far', 30.0, 400.0)          # 400 m ahead: no lane of the fake's 300 m road has it
    cfg = ST.stack_config(p, p.doc['worlds'][0]['id'], p.doc['rigs'][0]['id'], '/pending', svc.port, svc.tm_port,
                          'follower', scn, seed=0, fps=10, workspace=p.workspace)
    cfg['scenario']['duration_s'] = 3.0
    cfg['out'], cfg['name'] = str(base / 'far'), 'far'
    run = run_one(svc, cfg, title='far')
    assert run['status'] == 'ok', run.get('error')
    moved = run['scenario']['placed_short']
    assert moved[0]['actor'] == 'target' and moved[0]['placed_m'] < 400.0
    assert any('lane ends' in n for n in run['notes'])


# --------------------------------------------------------------------------------- sensitivity of unverified values
def _ncap_doc(impact_kmh):
    """A one-case Euro NCAP result: CCRs 30 km/h, 100 % overlap, hit at impact_kmh (bands 5 / 15 / 25 there)."""
    p = RP.pack('ncap_aeb_c2c')
    c = next(c for c in p['cases'] if c['id'] == 'ccrs-30-100')
    word = RP.band(impact_kmh, c['criteria'][0]['bands'])
    row = RP._row(c, result=word, word=word, factor=RP.BAND_SCORE['value'][word], speed=30, overlap=100,
                  measured={'impact_speed_kmh': impact_kmh, 'relative_impact_kmh': impact_kmh, 'collision': True})
    return {'pack': 'ncap_aeb_c2c', 'cases': [row], 'inputs': {'cases_selected': 1}}


def test_a_chart_read_band_flips_a_case_near_its_bound():
    s = RP.sensitivity('ncap_aeb_c2c', 'bands_ccrs', _ncap_doc(14.0))
    c = s['cases'][0]
    assert c['id'] == 'ccrs-30-100' and c['nominal'] == 'yellow' and c['low'] == 'orange' and c['high'] == 'yellow'
    assert c['flips'] and s['flips'] and s['band'].startswith('±10 %') and s['basis'] == 'figure'
    assert s['document'].startswith('Euro NCAP') and s['clause'] == '3.3.2'
    s = RP.sensitivity('ncap_aeb_c2c', 'bands_ccrs', _ncap_doc(8.0))       # far from a bound: holds
    assert not s['flips'] and s['cases'][0]['nominal'] == s['cases'][0]['low'] == 'yellow'
    assert 'nothing flips' in RP.sensitivity_text(s)
    s = RP.sensitivity('ncap_aeb_c2c', 'bands_ccrs', _ncap_doc(14.0), rel=0.03)    # a stated narrower band
    assert not s['flips'] and s['band'].startswith('±3 %')
    # the pack itself (not run): the classification only, nothing measured to re-judge
    s = RP.sensitivity('ncap_aeb_c2c', 'bands_ccrs')
    assert s['cases'] == [] and s['verdict'] is None and not s['run']


def test_performance_model_dy0_reclassifies_and_rejudges_a_run():
    p = RP.pack('r157')
    s = RP.sensitivity('r157', 'pm_dy0')
    assert {c['id'] for c in s['cases']} == {c['id'] for c in p['cases'] if c['group'] == 'cut-in'}
    assert s['low'] == pytest.approx(1.44) and s['high'] == pytest.approx(1.76) and not s['flips']
    # a run: the informative cut-in collided; were it mandatory at an end, the verdict would flip to failed
    rows = []
    for c in p['cases']:
        rows.append(dict(RP._row(c), result='informative' if not c['mandatory'] else 'pass',
                         measured={'collision': not c['mandatory']}))
    doc = {'cases': rows, 'inputs': {'cases_selected': len(p['cases'])}}
    s = RP.sensitivity('r157', 'pm_dy0', doc)
    assert s['verdict'] == {'nominal': 'passed', 'low': 'passed', 'high': 'passed'}
    s = RP.sensitivity('r157', 'pm_dy0', doc, rel=0.0)
    assert not s['flips']
    # a value no check reads, a direction, a statement
    s = RP.sensitivity('r157', 'pm_vy_max')
    assert s['applicable'] and not s['flips'] and 'no check' in s['checked']
    s = RP.sensitivity('ncap_aeb_c2c', 'overlap_sign', _ncap_doc(14.0))
    assert 'both signs' in s['band'] and not s['flips']
    s = RP.sensitivity('iso34502', 'same_in_iso')
    assert not s['applicable'] and s['low'] is None and 'not a number' in s['checked']
    with pytest.raises(RP.RegPackError):
        RP.sensitivity('r157', 'nope')
    assert RP.value_band({'value': 2.0, 'range': [1.5, 2.5]}) == (1.5, 2.5, 'the range the document gives')


def test_a_fault_tolerance_verdict_inside_its_band_flips():
    cell = {'sensor': 'lid', 'fault': 'lidar_yaw', 'title': 'Lidar extrinsic drift (yaw)', 'unit': 'deg',
            'status': 'bracketed', 'tolerated': 1.05, 'bracket': [1.05, 1.2], 'max': 5.0}
    s = FT.sensitivity(cell, {'lidar_yaw': 1.0})
    assert s['nominal'] == 'met' and s['flips']                  # +10 % asks 1.1 > 1.05
    assert [e['verdict'] for e in s['ends']] == ['met', 'not met', 'met', 'met']
    assert 'FLIPS' in FT.sensitivity_text(s)
    s = FT.sensitivity(cell, {'lidar_yaw': 0.5})
    assert s['nominal'] == 'met' and not s['flips'] and 'holds' in s['why']
    s = FT.sensitivity(cell, {'lidar_yaw': 1.1})                  # not met, but the bracket reaches above it
    assert s['nominal'] == 'not met' and s['flips']
    for c, why in ((dict(cell, status='refused', note='no lidar'), 'not measured'),
                   (dict(cell, status='fails_nominal'), 'without the fault'), (cell, 'no requirement')):
        s = FT.sensitivity(c, {} if why == 'no requirement' else {'lidar_yaw': 1.0})
        assert not s['applicable'] and why in s['why']
    s = FT.sensitivity(dict(cell, status='tolerates_max', tolerated=5.0, bracket=[5.0, None]), {'lidar_yaw': 1.0})
    assert len(s['ends']) == 3 and not s['flips']


def test_a_campaign_s_flags_lead_with_the_worst():
    base = {'title': 'X', 'unit': 'deg', 'max': 5.0, 'bracket': [None, None]}
    doc = {'cells': [dict(base, sensor='cam', fault='cam_yaw', status='bracketed', tolerated=0.4),
                     dict(base, sensor='cam', fault='cam_roll', status='bracketed', tolerated=1.05),
                     dict(base, sensor='lid', fault='lidar_yaw', status='refused', tolerated=None, note='no lidar'),
                     dict(base, sensor='gnss', fault='gnss_bias', status='fails_nominal', tolerated=0.0),
                     dict(base, sensor='cam', fault='cam_pitch', status='bracketed', tolerated=2.0)],
           'requirements': {'cam_yaw': 0.5, 'cam_roll': 1.0}}
    fl = FT.flags(doc)
    kinds = [f['kind'] for f in fl]
    assert kinds[:2] == ['not_met', 'fails_nominal'] and set(kinds) == {'not_met', 'fails_nominal', 'thin_margin',
                                                                         'missing', 'no_requirement'}
    nm = fl[0]
    assert nm['tone'] == 'critical' and nm['field'] == 'requirement' and nm['sensitivity'] and 'not met' in nm['text']
    thin = next(f for f in fl if f['kind'] == 'thin_margin')
    assert 'no safety factor' in thin['text'] and thin['sensitivity']
    assert 'cam_pitch' in next(f for f in fl if f['kind'] == 'no_requirement')['text']
    assert FT.flags({'cells': [], 'requirements': {}}) == []


def test_campaign_fidelity_is_recorded_or_its_runs(tmp_path):
    run = tmp_path / 'r1'
    run.mkdir()
    (run / 'run.json').write_text(json.dumps({'provenance': {'fidelity': 'fake'}}))
    assert FT.campaign_fidelity([], runs=[str(run)]) == ('recorded', 'fake')
    assert FT.campaign_fidelity([{'runs': ['r1']}], str(tmp_path)) == ('fake', '')
    doc = FT.result([], 'offline', fidelity='recorded', source_fidelity='fake')
    assert doc['fidelity'] == 'recorded' and doc['source_fidelity'] == 'fake'


# ------------------------------------------------------------------------------------- the safety case's fidelity
def _ws_with(tmp_path, level):
    ws = tmp_path / f'ws-{level}'
    d = ws / 'regpacks' / 'a'
    d.mkdir(parents=True)
    doc = {'schema': RP.SCHEMA, 'verdict': 'passed', 'score': {'share': 1.0}}
    if level:
        doc['fidelity'] = level
    (d / 'regpack.json').write_text(json.dumps(doc))
    return str(ws)


@pytest.mark.parametrize('level,status', [('fake', 'amber'), ('kinematic', 'amber'), ('mixed', 'amber'),
                                          ('carla', 'green'), (None, 'green')])
def test_evidence_from_a_test_double_verifies_amber(tmp_path, level, status):
    ws = _ws_with(tmp_path, level)
    doc = G.template('blank', ws, 'T')
    nid = G.attach(ws, doc, 'regpack', 'regpacks/a', {'measure': 'verdict', 'equals': 'passed'}, 'R157 pre-check')
    r = G.verify(ws, doc)['nodes'][nid]
    assert r['status'] == status, r
    assert r['fidelity'] == (level or 'unknown')
    if status == 'amber':
        from falcon_autonomy.core import fidelity as F
        assert f'evidence from {F.label(level)}, not a real simulator' in r['why']
        assert 'verdict passed' in r['why']                    # what held is still said
    # a violated claim stays red whatever the fidelity
    G.bind(ws, doc, nid, 'regpack', 'regpacks/a', {'measure': 'verdict', 'equals': 'failed'})
    assert G.verify(ws, doc)['nodes'][nid]['status'] == 'red'


def test_an_offline_analysis_of_fake_runs_is_amber_too(tmp_path):
    ws = tmp_path / 'ws'
    d = ws / 'faulttol' / 'f'
    d.mkdir(parents=True)
    (d / 'faulttol.json').write_text(json.dumps({'schema': FT.SCHEMA, 'fidelity': 'recorded',
                                                 'source_fidelity': 'fake', 'tolerance': {'cam': {'cam_yaw': {
                                                     'tolerated': 1.0}}}}))
    doc = G.template('blank', str(ws), 'T')
    nid = G.attach(str(ws), doc, 'faulttol', 'faulttol/f', {'measure': 'field', 'path': 'tolerance.cam.cam_yaw.'
                                                            'tolerated', 'op': '>=', 'limit': 0.5})
    r = G.verify(str(ws), doc)['nodes'][nid]
    assert r['status'] == 'amber' and 'offline analysis of Fake simulator runs' in r['why']


def test_attach_and_detach_a_run_as_evidence(tmp_path):
    ws = tmp_path / 'ws'
    run = ws / 'runs' / 'r1'
    run.mkdir(parents=True)
    (run / 'run.json').write_text(json.dumps({'status': 'ok', 'scenario': {'passed': True},
                                              'provenance': {'fidelity': 'carla'}}))
    doc = G.template('blank', str(ws), 'T')
    before = copy.deepcopy(doc)
    nid = G.attach(str(ws), doc, 'run', 'runs/r1', {'measure': 'field', 'path': 'scenario.passed', 'equals': True},
                   'the drive')
    assert nid == 'Sn1' and G.validate(doc) == [] and {'from': 'G1', 'to': 'Sn1', 'kind': 'SupportedBy'} in doc['links']
    assert G.node(doc, 'G1').get('undeveloped') is None
    v = G.verify(str(ws), doc)
    assert v['nodes'][nid]['status'] == 'green' and v['top']['G1'] == 'green'
    G.detach(doc, nid)
    assert [n['id'] for n in doc['nodes']] == [n['id'] for n in before['nodes']] and doc['links'] == before['links']
    with pytest.raises(G.GSNError):
        G.attach(str(ws), doc, 'run', 'runs/nope')
    assert [n['id'] for n in doc['nodes']] == [n['id'] for n in before['nodes']]       # nothing left half-made
    with pytest.raises(G.GSNError, match='Goal or Strategy'):
        G.attach(str(ws), doc, 'run', 'runs/r1', parent='J1')


# --------------------------------------------------------------------------------- the monitor's dry run, rollback
class _Device:
    """A fake SSH layer standing for the device's folder tree: ssh() understands the commands deploy and rollback
    send (and records them); put() copies into a dict. No network."""

    def __init__(self):
        self.commands, self.dirs, self.loaded = [], {}, 'falcon-monitor ok 12\n'

    def ssh(self, remote):
        self.commands.append(remote)
        cur, prev = '~/falcon-monitor/india', '~/falcon-monitor/.previous/india'
        if remote.startswith('if [ -d'):
            if cur in self.dirs:
                self.dirs[prev] = dict(self.dirs[cur])
                return 'falcon-kept\n'
            return 'falcon-none\n'
        if remote.startswith('mkdir -p'):
            self.dirs.setdefault(cur, {})
            return ''
        if remote.startswith('[ -d'):
            if prev not in self.dirs:
                return 'falcon-no-previous\n'
            self.dirs[cur], self.dirs[prev] = self.dirs[prev], self.dirs.get(cur, {})
            return 'falcon-swapped\n'
        if remote.startswith('cat '):
            return self.dirs[cur].get('manifest.json', '{}')
        if 'falcon_monitor' in remote:
            return self.loaded
        if remote.startswith('systemctl --user try-restart'):
            return ''
        raise AssertionError(remote)

    def put(self, local, remote_dir, name):
        with open(local, encoding='utf-8') as fh:
            self.dirs.setdefault(remote_dir, {})[name] = fh.read()


@pytest.fixture
def package(tmp_path):
    rules, src = MG.compile_rules('india')
    out = str(tmp_path / 'india')
    MG.generate(out, rules, src, 'india')
    return out


def test_a_dry_run_validates_here_and_says_what_would_be_copied(package, capsys):
    v = MG.validate_package(package)
    assert v['ok'] and {c['name'] for c in v['checks']} == {'manifest', 'pinned files', 'mapping.json', 'rules.json',
                                                            'systemd unit'}
    names = [f['name'] for f in v['files']]
    assert 'falcon-monitor-india.service' in names and 'mapping.json' in names and 'manifest.json' in names
    assert all(len(f['sha256']) == 64 and f['bytes'] > 0 for f in v['files'])
    assert MG.main_monitor(['deploy', package, '--dry-run']) == 0
    text = capsys.readouterr().out
    assert 'ok   systemd unit' in text and 'would keep the version on the device' in text
    assert 'falcon_monitor.py' in text and 'sha256' in text and 'no SSH' in text and 'systemctl --user enable' in text
    # a changed unit, a broken mapping: refused here, before any device
    unit = os.path.join(package, 'falcon-monitor-india.service')
    with open(unit, 'a', encoding='utf-8') as fh:
        fh.write('User=root\n')
    with open(os.path.join(package, 'mapping.json'), 'w', encoding='utf-8') as fh:
        fh.write('{"signals": {"x": {"from": "y"}}}')
    v = MG.validate_package(package)
    bad = {c['name'] for c in v['checks'] if not c['ok']}
    assert not v['ok'] and {'pinned files', 'systemd unit', 'mapping.json'} <= bad
    assert MG.main_monitor(['deploy', package, '--dry-run']) == 2
    assert 'not deployable' in capsys.readouterr().out
    assert not MG.validate_package(str(package) + '-nowhere')['ok']


def test_deploy_keeps_the_previous_version_and_rollback_restores_it(package, capsys):
    dev = _Device()
    out = MG.deploy(package, transport=dev)
    assert out['previous'] is None and out['selftest'] == 'falcon-monitor ok 12'
    assert not any('sudo' in c or 'systemctl' in c for c in dev.commands)
    first = json.loads(dev.dirs['~/falcon-monitor/india']['manifest.json'])['hash']
    rules, src = MG.compile_rules('india', ['speed_limit'])             # a second, different generation
    MG.generate(package, rules, src, 'india')
    out = MG.deploy(package, transport=dev)
    assert out['previous'] == '~/falcon-monitor/.previous/india'
    second = json.loads(dev.dirs['~/falcon-monitor/india']['manifest.json'])['hash']
    assert second != first and json.loads(dev.dirs[out['previous']]['manifest.json'])['hash'] == first
    dev.commands.clear()
    rb = MG.rollback(package, transport=dev)
    assert rb['restored'] == first and rb['selftest'].startswith('falcon-monitor ok')
    assert dev.commands[-1] == 'systemctl --user try-restart falcon-monitor-india.service'
    assert not any('sudo' in c for c in dev.commands) and sum('systemctl' in c for c in dev.commands) == 1
    assert MG.rollback(package, transport=dev)['restored'] == second      # rolling back again returns to it
    with pytest.raises(MG.MonitorGenError, match='nothing to roll back'):
        MG.rollback(package, transport=_Device())
    dev.loaded = 'Traceback: no module'
    with pytest.raises(MG.MonitorGenError, match='does not load'):
        MG.rollback(package, transport=dev)
    # the dry run of a rollback sends nothing; without a device in the settings the real one refuses
    assert MG.main_monitor(['rollback', package, '--dry-run']) == 0
    text = capsys.readouterr().out
    assert 'would swap' in text and 'try-restart' in text and 'no SSH' in text
    assert MG.main_monitor(['rollback', package]) == 2
    assert 'address is not set' in capsys.readouterr().err
