"""Regulation packs (core.regpacks): the data (every document value carries its source, clause and whether it was
verified; the unverified ones are listed), the regulation's formulas and driver performance model, the Euro NCAP
scoring, the scenarios (valid Falcon scenarios, exportable as OpenSCENARIO), packs run on the fake carla through
study_service with their verdicts, the report, the GSN evidence, the assistant's tools and the command line.

The numbers asserted here are the ones read in the source documents (docs/regulation-packs.md lists them)."""
import json
import os
import subprocess
import sys

import pytest

from conftest import FAKE, worker_env
from falcon_autonomy.core import protocol as P
from falcon_autonomy.core import regpacks as RP


# ------------------------------------------------------------------------------------------------------- the data
def test_every_value_names_its_document_and_the_unverified_are_listed():
    for pid in RP.PACKS:
        p = RP.pack(pid)
        assert set(p['sources']) <= set(RP.SOURCES) and p['limits'] and p['about']
        vals = RP.values_of(p)
        assert vals, pid
        for name, v in vals:
            assert v['source'] in RP.SOURCES, name
            if v['source'] == 'falcon':
                assert v['verified'] is None and v['note'], name          # a choice says why, and is not a document's
            else:
                assert v['clause'], name
                assert v['verified'] is (v['basis'] == 'text'), name
        for u in RP.unverified(p):
            assert u['basis'] in RP.BASIS_TEXT and u['why']
        for c in p['cases']:
            assert c['clause'] and c['source'] in RP.SOURCES and (c.get('scenario') or c.get('refuse')), c['id']
    names = {u['name'] for u in RP.unverified(RP.pack('ncap_aeb_c2c'))}
    assert names == {'bands_ccrs', 'bands_ccrm', 'overlap_sign'}       # the chart-read bands and the recalled sign
    assert {u['name'] for u in RP.unverified(RP.pack('r157'))} == {'pm_dy0', 'pm_vy_max'}
    assert [u['basis'] for u in RP.unverified(RP.pack('iso34502'))] == ['recalled']
    assert 'not a type-approval' in RP.DISCLAIMER and 'not a qualified tool' in RP.DISCLAIMER


def test_r157_following_distance_and_cut_in_formula():
    # 5.2.3.3: the table's rows (d = v · t_front) and the 2 m floor
    for kmh, d in ((7.2, 2.0), (10, 3.1), (20, 6.7), (30, 10.8), (40, 15.6), (50, 20.8), (60, 26.7)):
        assert RP.r157_following_m(kmh / 3.6) == pytest.approx(d, abs=0.06), kmh
    assert RP.r157_following_m(1.0) == 2.0
    assert RP.r157_following_m(35 / 3.6) == pytest.approx(35 / 3.6 * 1.35, abs=1e-6)        # interpolated
    assert RP.R157['mrm_decel']['value'] == 4.0 and RP.R157['em_decel']['value'] == 5.0
    assert RP.R157['lane_keep_s']['value'] == 300.0 and RP.R157['lead_decel']['value'] == 6.0
    # 5.2.5.2(c): v_rel 20 km/h needs TTC > 5.556 / 12 + 0.35 = 0.813 s at the intrusion (0.85 s after the start)
    assert RP.r157_cut_in_required(40, 20, 30.0, 1.0)
    assert not RP.r157_cut_in_required(40, 20, 9.0, 1.0)             # TTC there (9 − 4.72) / 5.56 = 0.77 s
    assert not RP.r157_cut_in_required(40, 20, 30.0, 2.0)            # visible for 0.425 s < 0.72 s


def test_performance_model_1_separates_avoidable_from_unavoidable():
    assert RP.pm1_cut_in_avoidable(60, 40, 30.0, 1.0) and not RP.pm1_cut_in_avoidable(60, 20, 10.0, 2.0)
    # monotone in the gap: once avoidable, a larger gap stays avoidable
    flags = [RP.pm1_cut_in_avoidable(60, 20, d, 1.5) for d in range(2, 60, 2)]
    assert not flags[0] and flags[-1] and flags == sorted(flags)
    # the regulation's own statements about its data sheets, at the 2.0 s headway
    for v in (10, 20, 40, 60):
        assert RP.pm1_decel_avoidable(v, 1.0 * RP.G), v              # "possible to avoid … -1.0G … at THW 2.0sec"
    assert RP.pm1_cut_out_avoidable(60, 60.0, 1.0) and not RP.pm1_cut_out_avoidable(60, 2.0, 3.0)
    p = RP.pack('r157')
    by = {c['id']: c for c in p['cases']}
    assert by['cut-in-60-20-10-2']['mandatory'] is False and 'not a failure' in by['cut-in-60-20-10-2']['note']
    assert by['cut-in-60-40-30-1']['mandatory'] and by['cut-in-60-40-30-1']['pm1_avoidable']
    assert by['mrm']['criteria'][0]['limit'] is not None and by['mrm']['fault']['kind'] == 'dropout'


def test_euro_ncap_grids_bands_and_scoring():
    p = RP.pack('ncap_aeb_c2c')
    ids = [c['id'] for c in p['cases']]
    ccrs = [i for i in ids if i.startswith('ccrs-') and 'fcw' not in i]
    ccrm = [i for i in ids if i.startswith('ccrm-')]
    assert len(ccrs) == 9 * 5 and len(ccrm) == 11 * 5 and len([i for i in ids if i.startswith('ccrb-')]) == 4
    assert {c['id'] for c in p['cases'] if c['group'] == 'CCRb'} == {'ccrb-12-2', 'ccrb-12-6', 'ccrb-40-2',
                                                                    'ccrb-40-6'}
    assert sum(RP.CCRS_POINTS['value'].values()) == 14 and sum(RP.CCRM_POINTS['value'].values()) == 15
    fcw = [c for c in p['cases'] if c['group'] == 'CCRs FCW']
    assert len(fcw) == 6 and all('braking-robot' in c['refuse'] for c in fcw)
    # the 50 km/h bands of the assessment protocol's text table
    b50 = RP.CCRS_BANDS[50]
    assert [RP.band(v, b50) for v in (None, 0.0, 4.9, 5.0, 14.9, 15.0, 29.9, 30.0, 39.9, 40.0)] == \
        ['green', 'green', 'green', 'yellow', 'yellow', 'orange', 'orange', 'brown', 'brown', 'red']
    assert RP.band(3.0, RP.CCRS_BANDS[20]) == 'red' and RP.band(None, RP.CCRS_BANDS[20]) == 'green'   # avoid only
    assert RP.band(7.0, RP.CCRM_BANDS[40]) == 'orange' and RP.band(20.0, RP.CCRM_BANDS[40]) == 'red'
    assert RP.overlap_offset(100) == 0.0 and RP.overlap_offset(50) == pytest.approx(0.9) == -RP.overlap_offset(-50)
    # all green: the three scenario points; yellow at every 100 % overlap of CCRs: 1 − (2 × 0.25) / 6
    rows = [{'id': c['id'], 'factor': 1.0} for c in p['cases'] if c.get('scenario')]
    s = RP.aeb_score(rows)
    assert s['complete'] and s['points'] == 3.0 and s['share'] == 1.0
    for r in rows:
        if r['id'].startswith('ccrs-') and r['id'].endswith('-100'):
            r['factor'] = 0.75
    s = RP.aeb_score(rows)
    assert s['scenarios']['CCRs']['points'] == pytest.approx(1 - 0.5 / 6, abs=1e-4) and s['points'] == \
        pytest.approx(3 - 0.5 / 6, abs=1e-4)
    part = RP.aeb_score(rows[:7])
    assert not part['complete'] and part['points'] is None and 'partial' in part['note']
    lss = RP.pack('ncap_lss')
    assert len(lss['cases']) == 30 and not any(RP.runnable(c) for c in lss['cases'])
    assert RP.LSS['lka_dtle']['value'] == -0.3 and RP.LSS['elk_edge_dtle']['value'] == -0.1
    assert RP.LSS['speed']['value'] == 72.0 and RP.LSS['vlat']['value'] == [0.2, 0.3, 0.4, 0.5, 0.6]


def test_scenarios_are_valid_and_export_as_openscenario(tmp_path):
    from falcon_autonomy.core import openscenario as OX
    from falcon_autonomy.core import scenarios as SC
    n = 0
    for pid in ('r157', 'ncap_aeb_c2c'):
        for c in RP.pack(pid)['cases']:
            if c.get('scenario'):
                assert SC.validate(c['scenario']) == [], c['id']
                n += 1
    assert n == 22 + 104
    out = RP.export('r157', str(tmp_path / 'x'))
    assert out['scenarios'] == 22 and out['refused'] == []
    suite = json.load(open(out['suite']))
    assert suite['schema'] == 'falcon.suite/1' and len(suite['entries']) == 22 and 'not a type-approval' in \
        suite['about']
    x = open(tmp_path / 'x' / 'xosc' / 'regpack-r157-cut-in-60-40-30-1.xosc').read()
    back = OX.import_xosc(x)
    assert [a['id'] for a in back['actors']] == ['cutter'] and 'LaneChangeAction' in x
    assert RP.export('ncap_lss', str(tmp_path / 'l'))['scenarios'] == 0


def test_the_iso_checklist_is_a_checklist():
    doc = RP.checklist_result(RP.pack('iso34502'), '', 'c')
    assert doc['verdict'] == 'checklist' and 'nothing was run' in doc['verdict_note']
    traffic = [i for i in doc['items'] if i['category'] == 'traffic']
    assert len(traffic) == 24 and sum(i['status'] == 'covered' for i in traffic) == 3
    assert {i['category'] for i in doc['items']} == {'traffic', 'perception', 'vehicle control'}
    assert all(i['note'] for i in doc['items'] if i['status'] != 'covered')
    assert doc['score']['items'] == 33 and doc['unverified'][0]['basis'] == 'recalled'
    assert 'JAMA' in doc['limits'][0]


# ---------------------------------------------------------------------------------------------- run on the fake
@pytest.fixture(scope='module')
def ran(tmp_path_factory):
    from falcon_autonomy.core.project import Project
    from falcon_autonomy.sim.service import study_service
    from test_stack import stack_rig
    base = tmp_path_factory.mktemp('rp')
    p = Project.new('RP')
    stack_rig(p)
    p.save(str(base / 'rp.falcon.json'))
    svc = study_service(workspace=p.workspace, fake_carla=FAKE, lock_dir=str(base / 'locks'))
    out = RP.result_dir(p.workspace, 'r157-a')
    cases = ['lane-keeping', 'stationary-40', 'following-20', 'lead-brake-40', 'cut-in-60-40-30-1',
             'cut-in-60-20-10-2', 'mrm']
    doc = RP.run_pack(svc, p, 'r157', out, 'world-town10', 'rig-default', cases=cases, fps=10, duration_scale=0.05,
                      name='r157-a')
    RP.write(doc, out)
    yield {'project': p, 'svc': svc, 'doc': doc, 'out': out, 'base': base, 'cases': cases}
    svc.shutdown()


def test_r157_cases_on_the_fake_carla(ran):
    doc = ran['doc']
    by = {r['id']: r for r in doc['cases']}
    assert list(by) == ran['cases'] and doc['disclaimer'] == RP.DISCLAIMER
    lk = by['lane-keeping']
    assert lk['result'] == 'pass' and lk['shortened'] and 'NOT the document\'s duration' in lk['note']
    assert lk['measured']['lane_excursion_m'] < 0
    assert by['stationary-40']['result'] == 'pass' and by['stationary-40']['measured']['collision'] is False
    assert by['following-20']['result'] == 'pass' and by['following-20']['measured']['following_shortfall_m'] <= 0
    assert by['lead-brake-40']['result'] == 'pass' and by['cut-in-60-40-30-1']['result'] == 'pass'
    inf = by['cut-in-60-20-10-2']                                   # the model cannot avoid it: informative
    assert inf['result'] == 'informative' and not inf['mandatory'] and inf['measured']['collision']
    assert 0 < inf['measured']['relative_impact_kmh'] <= 60.0 and inf['unverified'] == ['dy0']
    mrm = by['mrm']
    assert mrm['result'] == 'pass' and (mrm['measured']['mrm_max_decel_mps2'] or 0) <= 4.0
    # nothing failed, but a selection and a shortened case: never 'passed'
    assert doc['verdict'] == 'incomplete' and 'selection' in doc['verdict_note'] and 'shortened' in doc['verdict_note']
    assert doc['score'] == {'passed': 6, 'mandatory': 6, 'share': 1.0}
    assert os.path.isfile(os.path.join(ran['out'], 'report.html'))
    # the full duration does not fit the runner's route: refused by name, not silently cut
    p, svc = ran['project'], ran['svc']
    d2 = RP.run_pack(svc, p, 'r157', ran['out'], 'world-town10', 'rig-default', cases=['lane-keeping'], fps=10)
    assert d2['cases'][0]['result'] == 'refused' and '400 m' in d2['cases'][0]['why']
    assert d2['verdict'] == 'incomplete'
    with pytest.raises(RP.RegPackError):
        RP.run_pack(svc, p, 'r157', ran['out'], 'world-town10', 'rig-default', cases=['nope'])


def test_a_failed_mandatory_case_fails_the_pack_and_ncap_is_scored(ran):
    p, svc = ran['project'], ran['svc']
    out = RP.result_dir(p.workspace, 'r157-b')
    doc = RP.run_pack(svc, p, 'r157', out, 'world-town10', 'rig-default', cases=['cut-in-40-20-12-1'], fps=10)
    r = doc['cases'][0]
    assert r['result'] == 'fail' and r['word'] == 'failed' and r['tone'] == 'critical'       # the follower hits it
    assert doc['verdict'] == 'failed' and 'cut-in-40-20-12-1' in doc['verdict_note']
    out = RP.result_dir(p.workspace, 'ncap-a')
    doc = RP.run_pack(svc, p, 'ncap_aeb_c2c', out, 'world-town10', 'rig-default', fps=10,
                      cases=['ccrb-12-2', 'ccrb-12-6', 'ccrb-40-2', 'ccrb-40-6', 'ccrs-30-100', 'ccrs-fcw-60'])
    RP.write(doc, out)
    by = {r['id']: r for r in doc['cases']}
    assert by['ccrs-fcw-60']['result'] == 'refused' and by['ccrs-30-100']['word'] in RP.BAND_NAMES
    sc = doc['score']
    assert doc['verdict'] == 'partial' and sc['points'] is None
    assert sc['scenarios']['CCRb']['complete'] and 0.0 <= sc['scenarios']['CCRb']['points'] <= 1.0
    assert sc['scenarios']['CCRs']['points'] is None and sc['scenarios']['CCRs']['grid_points_run'] == 1
    text = RP.text_report(doc)
    assert RP.DISCLAIMER in text and 'NOT verified' in text and 'recalled, not checked against the document' in text


def test_the_report_states_the_disclaimer_and_the_unverified_values(ran, tmp_path):
    from falcon_autonomy.core import reports as R
    meta = R.write_report('regpack', ran['out'], str(tmp_path / 'r.html'))
    html = open(tmp_path / 'r.html').read()
    assert meta['kind'] == 'regpack' and R.read_meta(str(tmp_path / 'r.html'))['inputs']
    for want in ('not a type-approval', 'not a qualified tool', 'Values not verified', 'Annex 5, 4.1',
                 'read from a chart or figure', 'Falcon&#x27;s own choices', 'incomplete'):
        assert want in html, want
    assert ('r157-a', ran['out']) in R.sources(ran['project'].workspace)['regpack']


def test_gsn_binds_a_pack_result(ran):
    from falcon_autonomy.core import gsn as G
    ws = ran['project'].workspace
    assert 'regpacks/r157-a' in G.evidence_candidates(ws)['regpack']
    doc = G.template('blank', ws, 'T')
    doc['nodes'] += [{'id': 'S1', 'type': 'Solution', 'text': 'R157 pre-check'}]
    doc['links'] += [{'from': doc['nodes'][0]['id'], 'to': 'S1', 'kind': 'SupportedBy'}]
    G.bind(ws, doc, 'S1', 'regpack', 'regpacks/r157-a', {'measure': 'verdict', 'equals': 'passed'})
    assert G.verify(ws, doc)['nodes']['S1']['status'] == 'red'                 # incomplete is not passed
    G.bind(ws, doc, 'S1', 'regpack', 'regpacks/r157-a', {'measure': 'field', 'path': 'score.share', 'op': '>=',
                                                         'limit': 1.0})
    assert G.verify(ws, doc)['nodes']['S1']['status'] == 'green'


# ------------------------------------------------------------------------------------------ the tools and the CLI
def test_the_assistant_tools(ran):
    from falcon_autonomy.assistant import tools as TOOLS
    from falcon_autonomy.assistant.tool_api import REGISTRY, call
    for name, tier in (('formal_regpack_run', 'act'), ('formal_regpack_show', 'read'),
                       ('research_faulttol_run', 'act'), ('research_faulttol_show', 'read')):
        assert REGISTRY[name].tier == tier
    c = TOOLS.context(ran['project'].path, FAKE)
    try:
        r = call(c, 'formal_regpack_show', {})
        assert [p['id'] for p in r['packs']] == list(RP.PACKS) and r['disclaimer'] == RP.DISCLAIMER
        assert 'r157-a' in r['results']
        r = call(c, 'formal_regpack_show', {'pack': 'r157', 'group': 'cut-in'})
        assert r['total'] == 6 and {x['kind'] for x in r['items']} == {'mandatory', 'informative'}
        unv = [v for v in r['values'] if v['verified'] is False]
        assert {v['name'] for v in unv} == {'pm_dy0', 'pm_vy_max'} and all(v['why'] for v in unv)
        r = call(c, 'formal_regpack_show', {'result': 'r157-a'})
        assert r['verdict'] == 'incomplete' and r['disclaimer'] == RP.DISCLAIMER and r['total'] == 7
        assert any(x['uses_unverified_value'] for x in r['items']) and r['unverified_values']
        assert 'no regulation pack result' in call(c, 'formal_regpack_show', {'result': 'nope'})['error']
        assert 'no pack' in call(c, 'formal_regpack_show', {'pack': 'nope'})['error']
        t = REGISTRY['formal_regpack_run']
        pre = t.preview(c, pack='r157', name='x', groups=['cut-in'], duration_scale=0.1)
        assert pre.startswith('falcon regpack run') and '--group cut-in' in pre and 'not a type-approval' in pre
        assert t.estimate(c, pack='iso34502', name='x')['seconds'] == 0.0
        assert 'exists already' in call(c, 'formal_regpack_run', {'pack': 'r157', 'name': 'r157-a'})['error']
        j = call(c, 'formal_regpack_run', {'pack': 'iso34502', 'name': 'iso-a'})
        assert j['disclaimer'] == RP.DISCLAIMER and j['result'] == 'iso-a'
        w = call(c, 'jobs_wait', {'jobs': [j['job']], 'timeout_s': 120})
        assert w['finished'] == [j['job']], w
        r = call(c, 'formal_regpack_show', {'result': 'iso-a', 'group': 'traffic'})
        assert r['verdict'] == 'checklist' and len(r['items']) == 24
        # fault tolerance
        r = call(c, 'research_faulttol_show', {})
        assert r['campaigns'] == [] and len(r['fault_models']) == 12
        f = REGISTRY['research_faulttol_run']
        pre = f.preview(c, name='ft', scenario='lead_brake', faults=['lidar_sector'], require={'lidar_sector': 30})
        assert '--closed-loop --scenario lead_brake' in pre and '--require lidar_sector=30' in pre
        assert 'not both or neither' in call(c, 'research_faulttol_run', {'name': 'ft'})['error']
        assert 'unknown fault' in call(c, 'research_faulttol_run', {'name': 'ft', 'scenario': 'lead_brake',
                                                                    'faults': ['nope']})['error']
        j = call(c, 'research_faulttol_run', {'name': 'ft', 'scenario': 'lead_brake', 'sensors': ['gnss'],
                                              'faults': ['gnss_bias'], 'require': {'gnss_bias': 1.0}})
        w = call(c, 'jobs_wait', {'jobs': [j['job']], 'timeout_s': 240})
        assert w['finished'] == [j['job']], w
        r = call(c, 'research_faulttol_show', {'name': 'ft', 'cell': 'gnss:gnss_bias'})
        row = r['items'][0]
        assert r['mode'] == 'closed_loop' and row['status'] == 'bracketed' and row['requirement'] == 'met'
        assert len(row['curve']) >= 5 and r['most_fragile']['sensor'] == 'gnss' and r['derived_calibration']
        assert 'no cell' in call(c, 'research_faulttol_show', {'name': 'ft', 'cell': 'a:b'})['error']
    finally:
        c.service.shutdown(grace_s=10.0)


def test_the_command_line(ran, tmp_path):
    def cli(*a):
        return subprocess.run([sys.executable, '-m', 'falcon_autonomy', 'regpack', *a], capture_output=True,
                              text=True, env=worker_env({'XDG_CONFIG_HOME': str(tmp_path / 'cfg')}), timeout=120)
    r = cli('list')
    assert r.returncode == 0 and all(p in r.stdout for p in RP.PACKS) and 'not a type-approval' in r.stdout
    r = cli('show', 'ncap_aeb_c2c')
    assert r.returncode == 0 and 'NOT VERIFIED: recalled, not checked against the document' in r.stdout
    assert 'NOT VERIFIED: read from a chart' in r.stdout and 'ccrb-12-2' in r.stdout
    assert json.loads(cli('show', 'r157', '--json').stdout)['unverified']
    assert cli('show', 'nope').returncode == P.REFUSED
    r = cli('run', ran['project'].path, 'r157', '--name', 'cli', '--case', 'stationary-20', '--fps', '10',
            '--fake-carla', FAKE)
    assert r.returncode == 0, r.stdout + r.stderr
    d = RP.result_dir(ran['project'].workspace, 'cli')
    assert RP.load(d)['cases'][0]['result'] == 'pass'
    r = cli('report', d, '--out', str(tmp_path / 'c.html'))
    assert r.returncode == 0 and 'verdict: incomplete' in r.stdout and os.path.isfile(tmp_path / 'c.html')
    assert cli('run', ran['project'].path, 'r157', '--name', 'x', '--case', 'nope').returncode == P.REFUSED
    r = cli('export', 'ncap_aeb_c2c', '--out', str(tmp_path / 'e'))
    assert r.returncode == 0 and '104 scenarios' in r.stdout and 'refused: ccrs-fcw-55' in r.stdout
