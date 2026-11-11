"""The living GSN safety case (core.gsn): structure checks, binding and hash pinning (amber on a changed result), claims
on each kind of result — a Clopper–Pearson bound flipped red by a synthetic CI result — AND propagation, the ISO 26262
and SOTIF templates (with the not-a-qualified-tool statement and the triggering conditions), the append-only event log,
the command, the HTML and SVG exports, the CI gate end to end on the fake carla, and the Reports tab offscreen."""
from __future__ import annotations

import json
import os

import pytest

from falcon_autonomy.core import failrate as FR
from falcon_autonomy.core import gsn as G


_APP = []


@pytest.fixture(scope='module')
def qapp():
    """The application, kept alive for the session (another test file's may already exist)."""
    from PyQt6.QtWidgets import QApplication
    from falcon_autonomy.ui.app import make_app
    from falcon_autonomy.ui.theme import THEME
    app = QApplication.instance() or make_app()
    _APP.append(app)
    THEME.apply(app, 'dark')
    return app


def _ws(tmp_path):
    ws = tmp_path / 'ws'
    (ws / 'safety').mkdir(parents=True)
    return str(ws)


def _ci_result(ws, folder, name, failed, runs, verdict='passed', finished='2026-11-11T10:00:00Z'):
    d = os.path.join(ws, 'ci', folder)
    os.makedirs(d, exist_ok=True)
    doc = {'schema': 'falcon.ci_results/1', 'name': name, 'verdict': verdict, 'finished': finished,
           'summary': {'overall': {'runs': runs, 'failed': failed, 'passed': runs - failed, 'metrics': {
               'rule_violations_per_km': {'violations': 3, 'distance_km': 2.0, 'value': 1.5}}}}}
    with open(os.path.join(d, 'ci-results.json'), 'w', encoding='utf-8') as fh:
        json.dump(doc, fh)
    return d


def _suite(ws, name, passed: list):
    d = os.path.join(ws, 'suites', name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'results.jsonl'), 'w', encoding='utf-8') as fh:
        for i, p in enumerate(passed):
            fh.write(json.dumps({'key': f'k{i}', 'status': 'ok', 'passed': p, 'scenario': 'cut_in'}) + '\n')
    return d


def _doc(claim, ref='ci:aeb@latest', kind='ci'):
    return {'schema': G.SCHEMA, 'title': 't', 'template': 'blank', 'nodes': [
        {'id': 'G1', 'type': 'Goal', 'text': 'safe'},
        {'id': 'S1', 'type': 'Strategy', 'text': 'over evidence'},
        {'id': 'G2', 'type': 'Goal', 'text': 'CI failure rate bounded'},
        {'id': 'G3', 'type': 'Goal', 'text': 'suite failure rate bounded'},
        {'id': 'Sn1', 'type': 'Solution', 'text': 'the CI runs', 'evidence': {'kind': kind, 'ref': ref,
                                                                             'claim': claim}},
        {'id': 'Sn2', 'type': 'Solution', 'text': 'the suite'},
        {'id': 'C1', 'type': 'Context', 'text': 'the ODD'}],
        'links': [{'from': 'G1', 'to': 'S1', 'kind': 'SupportedBy'}, {'from': 'S1', 'to': 'G2', 'kind': 'SupportedBy'},
                  {'from': 'S1', 'to': 'G3', 'kind': 'SupportedBy'}, {'from': 'G2', 'to': 'Sn1', 'kind': 'SupportedBy'},
                  {'from': 'G3', 'to': 'Sn2', 'kind': 'SupportedBy'}, {'from': 'G1', 'to': 'C1', 'kind': 'InContextOf'}]}


CP = {'measure': 'failure_rate', 'bound': 'clopper_pearson_upper', 'conf': 0.95, 'limit': 0.1}


def test_structure_is_checked():
    assert G.validate(_doc(CP)) == []
    bad = _doc(CP)
    bad['links'].append({'from': 'Sn1', 'to': 'G1', 'kind': 'SupportedBy'})
    assert any('supports nothing' in p for p in G.validate(bad))
    bad = _doc(CP)
    bad['links'].append({'from': 'G2', 'to': 'G1', 'kind': 'SupportedBy'})
    assert any('cycle' in p for p in G.validate(bad))
    bad = _doc(CP)
    bad['links'].append({'from': 'G1', 'to': 'C1', 'kind': 'SupportedBy'})
    assert any('not a support' in p for p in G.validate(bad))
    bad = _doc(dict(CP, measure='nope'))
    assert any('claim measure' in p for p in G.validate(bad))
    bad = _doc(CP)
    bad['nodes'][0]['evidence'] = {'kind': 'ci', 'ref': 'x'}
    assert any('only a Solution' in p for p in G.validate(bad))
    assert G.top_goals(_doc(CP)) == ['G1']


def test_clopper_pearson_claim_flips_red_on_a_new_ci_result_and_propagates(tmp_path):
    """60 runs without a failure bound the failure rate at 1 − 0.05^(1/60) ≈ 0.049 ≤ 0.1: green, up to G1 (G3's
    undeveloped Solution keeps G1 grey until it is bound). A newer CI result with 4 failures in 60 lifts the upper
    bound to 0.146: the alias follows it and the bound is broken — red, propagated to G1."""
    ws = _ws(tmp_path)
    _ci_result(ws, 'a', 'aeb', 0, 60)
    _suite(ws, 'smoke', [True] * 40)
    doc = _doc(CP)
    G.bind(ws, doc, 'Sn1', 'ci', 'ci:aeb@latest', CP)
    v = G.verify(ws, doc)
    assert v['nodes']['Sn1']['status'] == 'green'
    assert v['nodes']['Sn1']['claim']['measured'] == pytest.approx(FR.binom_upper(0, 60, 0.95))
    assert v['nodes']['G2']['status'] == 'green' and v['nodes']['Sn2']['status'] == 'grey'
    assert v['nodes']['G1']['status'] == 'grey' and v['nodes']['C1']['status'] is None
    G.bind(ws, doc, 'Sn2', 'suite', 'suites/smoke', CP)
    v = G.verify(ws, doc)
    assert v['top'] == {'G1': 'green'} and v['counts']['green'] == 6
    _ci_result(ws, 'b', 'aeb', 4, 60, finished='2026-11-12T10:00:00Z')
    up = FR.binom_upper(4, 60, 0.95)
    assert up > 0.1
    v = G.verify(ws, doc)
    assert v['nodes']['Sn1']['status'] == 'red' and v['nodes']['Sn1']['claim']['measured'] == pytest.approx(up)
    assert v['nodes']['Sn1']['path'] == os.path.join('ci', 'b', 'ci-results.json')
    assert v['nodes']['G2']['status'] == 'red' and v['top'] == {'G1': 'red'} and 'S1' in v['nodes']['G1']['why']
    assert G.gate(v) == (False, {'G1': 'red'})
    # an in-memory override (a CI run at its end) wins over the files
    v = G.verify(ws, doc, {'ci:aeb@latest': ('x/ci-results.json', {'summary': {'overall': {'runs': 60,
                                                                                            'failed': 0}}})})
    assert v['top'] == {'G1': 'green'}


def test_a_changed_result_turns_amber_until_rebound_and_missing_is_amber(tmp_path):
    ws = _ws(tmp_path)
    _suite(ws, 'smoke', [True] * 40)
    doc = _doc(CP, 'suites/smoke', 'suite')
    ev = G.bind(ws, doc, 'Sn1', 'suite', 'suites/smoke', CP)
    assert ev['path'] == os.path.join('suites', 'smoke', 'results.jsonl') and len(ev['sha256']) == 64
    assert G.verify(ws, doc)['nodes']['Sn1']['status'] == 'green'
    _suite(ws, 'smoke', [True] * 39 + [False])                # the suite re-run: another file
    r = G.verify(ws, doc)['nodes']['Sn1']
    assert r['status'] == 'amber' and 're-bind' in r['why']
    assert G.verify(ws, doc)['nodes']['G2']['status'] == 'amber'
    G.bind(ws, doc, 'Sn1', 'suite', 'suites/smoke', keep_claim=True)
    r = G.verify(ws, doc)['nodes']['Sn1']
    assert r['status'] == 'red' and r['claim']['k'] == 1 and r['claim']['n'] == 40    # 1 of 40: upper 0.11 > 0.1
    os.remove(os.path.join(ws, 'suites', 'smoke', 'results.jsonl'))
    r = G.verify(ws, doc)['nodes']['Sn1']
    assert r['status'] == 'amber' and 'missing' in r['why']
    with pytest.raises(G.GSNError, match='missing'):
        G.bind(ws, doc, 'Sn1', 'suite', 'suites/smoke', CP)
    with pytest.raises(G.GSNError, match='Solution'):
        G.bind(ws, doc, 'G1', 'suite', 'suites/smoke', CP)


def test_claims_on_the_other_kinds(tmp_path):
    ws = _ws(tmp_path)
    st = os.path.join(ws, 'studies', 'fz')
    os.makedirs(st)
    with open(os.path.join(st, 'falsify.json'), 'w') as fh:
        json.dump({'schema': 'falcon.falsify/1', 'summary': {'best_rho': -0.4, 'known': 9, 'evaluated': 10,
                                                             'falsified': [{}, {}], 'near_misses': []}}, fh)
    fr = os.path.join(ws, 'studies', 'rate')
    os.makedirs(fr)
    with open(os.path.join(fr, 'failure-rate.json'), 'w') as fh:
        json.dump({'estimate': {'status': 'ok', 'p': 2e-4, 'ci95': [1e-4, 4e-4], 'failures': 12, 'n': 200}}, fh)
    ev = G.evaluate_claim
    with open(os.path.join(st, 'falsify.json')) as fh:
        fz = json.load(fh)
    assert ev('falsify', {'measure': 'min_robustness', 'limit': 0.0}, fz)['ok'] is False
    assert ev('falsify', {'measure': 'falsified_runs', 'limit': 2}, fz)['ok'] is True
    with open(os.path.join(fr, 'failure-rate.json')) as fh:
        rate = json.load(fh)
    assert ev('failrate', {'measure': 'failure_rate', 'limit': 5e-4}, rate)['measured'] == 4e-4
    assert ev('failrate', {'measure': 'failure_rate', 'limit': 3e-4}, rate)['ok'] is False
    assert ev('failrate', {'measure': 'failure_rate', 'bound': 'estimate', 'limit': 3e-4}, rate)['ok'] is True
    refused = {'estimate': {'status': 'refused', 'upper95': 0.01, 'reasons': ['too few']}}
    assert ev('failrate', {'measure': 'failure_rate', 'limit': 0.02}, refused)['ok'] is True
    assert ev('failrate', {'measure': 'failure_rate', 'bound': 'estimate', 'limit': 0.02}, refused)['ok'] is None
    ci = {'verdict': 'failed', 'summary': {'overall': {'runs': 10, 'failed': 1, 'metrics': {
        'rule_violations_per_km': {'violations': 3, 'distance_km': 2.0}}}}}
    assert ev('ci', {'measure': 'verdict'}, ci)['ok'] is False
    assert ev('ci', {'measure': 'violations_per_km', 'limit': 2.0}, ci)['measured'] == 1.5
    assert ev('ci', {'measure': 'pass_rate', 'limit': 0.5}, ci)['measured'] == pytest.approx(
        FR.binom_lower(9, 10, 0.95))
    assert ev('ci', {'measure': 'field', 'path': 'summary.overall.runs', 'op': '>=', 'limit': 10}, ci)['ok'] is True
    assert ev('ci', {'measure': 'field', 'path': 'summary.nope', 'limit': 1}, ci)['ok'] is None
    assert ev('suite', {'measure': 'min_robustness', 'limit': 0}, [])['ok'] is None
    # lineage: the hash alone
    with open(os.path.join(ws, 'data.json'), 'w') as fh:
        json.dump({'merkle_root': 'ab' * 32}, fh)
    doc = _doc({'measure': 'field', 'path': 'merkle_root', 'equals': 'ab' * 32}, 'data.json', 'lineage')
    G.bind(ws, doc, 'Sn1', 'lineage', 'data.json', {'measure': 'field', 'path': 'merkle_root', 'equals': 'ab' * 32})
    assert G.verify(ws, doc)['nodes']['Sn1']['status'] == 'green'


def test_templates_events_command_and_exports(tmp_path, capsys):
    ws = _ws(tmp_path)
    _ci_result(ws, 'a', 'aeb', 0, 80)
    _suite(ws, 'smoke', [True] * 80)
    sd = os.path.join(ws, 'studies', 'search-cut_in')
    os.makedirs(sd)
    with open(os.path.join(sd, 'study.json'), 'w') as fh:
        json.dump({'scenario': {'id': 'cut_in'}}, fh)
    with open(os.path.join(sd, 'results.jsonl'), 'w') as fh:
        for g, v in ((3.0, 0.9), (12.0, 0.1), (4.0, 0.7)):
            fh.write(json.dumps({'status': 'ok', 'values': {'cut_gap': g}, 'objective': v, 'run': 'r'}) + '\n')
    for kind in G.TEMPLATES:
        doc = G.template(kind, ws)
        assert G.validate(doc) == [], kind
        assert any(n['type'] == 'Justification' and 'ISO 26262-8 §11' in n['text'] for n in doc['nodes'])
    sotif = G.template('sotif', ws)
    texts = ' '.join(n['text'] for n in sotif['nodes'])
    assert 'Triggering condition' in texts and 'cut_gap 3–4' in texts             # from core.safetycase
    refs = {n['evidence']['ref'] for n in sotif['nodes'] if n.get('evidence')}
    assert {'suites/smoke', 'ci:aeb@latest'} <= refs
    iso = G.template('iso26262', ws)
    assert any(n['type'] == 'AwayGoal' for n in iso['nodes'])
    # the command: new (bound), show, verify --gate, bind, export
    assert G.main_gsn(['new', '--workspace', ws, '--template', 'sotif', '--bind']) == 0
    assert G.main_gsn(['new', '--workspace', ws]) == 2                          # exists
    capsys.readouterr()
    assert G.main_gsn(['verify', '--workspace', ws, '--json']) == 0
    v = json.loads(capsys.readouterr().out)
    assert v['schema'] == G.VERIFY_SCHEMA and set(v['top']) == {'G1'}
    doc = G.load(ws)
    sn = next(n['id'] for n in doc['nodes'] if (n.get('evidence') or {}).get('ref') == 'suites/smoke')
    assert G.main_gsn(['bind', '--workspace', ws, sn, '--measure', 'failure_rate', '--limit', '0.001']) == 0
    assert G.main_gsn(['verify', '--workspace', ws, '--gate']) == 1             # 0 of 80: 0.037 > 0.001: red
    assert G.main_gsn(['bind', '--workspace', ws, sn, '--unbind']) == 0
    assert G.main_gsn(['show', '--workspace', ws]) == 0
    html, svg = str(tmp_path / 'sc.html'), str(tmp_path / 'sc.svg')
    assert G.main_gsn(['export', '--workspace', ws, '--out', html]) == 0
    assert G.main_gsn(['export', '--workspace', ws, '--out', svg]) == 0
    from falcon_autonomy.core.reports import read_meta
    meta = read_meta(html)
    assert meta['kind'] == 'gsn' and any(p.endswith('gsn.json') for p in meta['inputs'])
    text = open(html, encoding='utf-8').read()
    assert '<svg' in text and 'ISO 26262-8 §11' in text
    assert open(svg).read().startswith('<svg')
    ev = G.events(ws)
    kinds = [e['event'] for e in ev]
    assert kinds[0] == 'new' and kinds.count('verify') == 2 and 'bind' in kinds and 'unbind' in kinds
    assert kinds[-2:] == ['export', 'export'] and all(e['schema'] == G.EVENTS_SCHEMA for e in ev)
    # append-only: the earlier lines are still the same bytes
    with open(G.events_path(ws), encoding='utf-8') as fh:
        before = fh.read()
    G.verify_and_log(ws)
    with open(G.events_path(ws), encoding='utf-8') as fh:
        after = fh.read()
    assert after.startswith(before) and len(after.splitlines()) == len(before.splitlines()) + 1
    from falcon_autonomy.core import reports as R
    assert ('safety case', G.gsn_path(ws)) in R.sources(ws)['gsn']


def test_layout_places_every_node_once():
    doc = G.template('iso26262')
    lay = G.layout(doc)
    assert set(lay) == {n['id'] for n in doc['nodes']}
    boxes = sorted((x, y) for x, y, _, _ in lay.values())
    assert len(set(boxes)) == len(boxes)
    assert lay['G1'][1] == 0 and lay['S1'][1] > lay['G1'][1]


def test_the_ci_gate_fails_when_a_top_goal_turns_red(tmp_path, fake_carla):
    """A follower suite through `falcon ci run` on the fake carla with the project's safety case bound to the CI's
    @latest alias: with a limit the two runs' Clopper–Pearson bound meets, the gate passes and the event is logged by
    the CI; with a stricter limit the same runs break the bound — the top goal is red and the gate fails (exit 1)."""
    from falcon_autonomy.core import ci as CI
    from falcon_autonomy.core.project import Project
    p = Project.new('CI GSN')
    p.doc['rigs'][0]['sensors'][0].update(width=48, height=32, labels=[])
    p.save(str(tmp_path / 'p.falcon.json'))
    ws = p.workspace
    suite = {'schema': 'falcon.suite/1', 'name': 'g', 'stack': {'driver': 'follower', 'bridge': {}, 'launch': [],
                                                               'domain_id': 0},
             'world': 'world-town10', 'rig': 'rig-default', 'fps': 20.0, 'motion': 'kinematic', 'traffic_vehicles': 0,
             'criteria': {'no_collision': True}, 'entries': [{'scenario': 'lead_brake', 'seeds': [0, 1]}]}
    (tmp_path / 'suite.json').write_text(json.dumps(suite))
    (tmp_path / 'ci.json').write_text(json.dumps({
        'schema': 'falcon.ci/1', 'name': 'gsn-smoke', 'project': 'p.falcon.json',
        'subject': {'kind': 'stack', 'suite': 'suite.json'}, 'carla': {'mode': 'fake'},
        'gates': {'gsn': {'fail_on': 'red'}}}))
    doc = _doc(dict(CP, limit=0.9), 'ci:gsn-smoke@latest')
    doc['nodes'] = [n for n in doc['nodes'] if n['id'] not in ('G3', 'Sn2')]
    doc['links'] = [ln for ln in doc['links'] if 'G3' not in (ln['from'], ln['to'])]
    os.makedirs(os.path.join(ws, 'safety'), exist_ok=True)
    G.save(ws, doc)
    res, code = CI.run_ci(str(tmp_path / 'ci.json'), {'out': str(tmp_path / 'o1'), 'fake_carla': fake_carla},
                          log=lambda t: None, env={})
    assert res['error'] is None, res.get('traceback') or res['error']
    n, k = res['summary']['overall']['runs'], res['summary']['overall']['failed']
    assert n == 2
    gate = next(g for g in res['gates'] if g['id'] == 'gsn')
    want = FR.binom_upper(k, n, 0.95) <= 0.9
    assert gate['status'] == ('passed' if want else 'failed')
    assert res['gsn']['bound'] == ['Sn1'] and res['gsn']['top'] == {'G1': 'green' if want else 'red'}
    assert G.events(ws)[-1]['by'] == 'ci' and G.events(ws)[-1]['aliases'] == ['ci:gsn-smoke@latest']
    G.node(doc, 'Sn1')['evidence']['claim'] = dict(CP, limit=0.5)     # 2 runs can never bound it at 0.5
    G.save(ws, doc)
    res, code = CI.run_ci(str(tmp_path / 'ci.json'), {'out': str(tmp_path / 'o2'), 'fake_carla': fake_carla},
                          log=lambda t: None, env={})
    gate = next(g for g in res['gates'] if g['id'] == 'gsn')
    assert gate['status'] == 'failed' and 'G1 is red' in gate['detail'] and code == 1
    assert res['gsn']['top'] == {'G1': 'red'}
    junit = (tmp_path / 'o2' / 'junit.xml').read_text()
    assert 'safety case: no top goal red' in junit
    # no safety case: the gate is an error (the run cannot be judged), never a pass
    os.remove(G.gsn_path(ws))
    res, code = CI.run_ci(str(tmp_path / 'ci.json'), {'out': str(tmp_path / 'o3'), 'fake_carla': fake_carla},
                          log=lambda t: None, env={})
    gate = next(g for g in res['gates'] if g['id'] == 'gsn')
    assert gate['status'] == 'error' and code == 2
    assert CI.validate_config({'schema': 'falcon.ci/1', 'subject': {'kind': 'stack', 'suite': 'x'},
                               'gates': {'gsn': {'fail_on': 'purple'}}})


@pytest.mark.gui
def test_the_reports_tab_offscreen(qapp, tmp_path):
    from types import SimpleNamespace
    from falcon_autonomy.ui.gsn_tab import GsnTab
    ws = _ws(tmp_path)
    _ci_result(ws, 'a', 'aeb', 0, 60)
    _suite(ws, 'smoke', [True] * 40)
    win = SimpleNamespace(project=SimpleNamespace(path=str(tmp_path / 'p.falcon.json')), workspace=lambda: ws)
    tab = GsnTab(SimpleNamespace(win=win))
    assert tab.new_btn.isEnabled() and 'No safety case' in tab.state.text()
    tab.template.setCurrentIndex(tab.template.findData('blank'))
    doc = _doc(CP)
    G.save(ws, doc)
    tab.reload()
    assert 'top goals G1 grey' in tab.state.text() and set(tab.items) == {n['id'] for n in doc['nodes']}
    d = G.load(ws)
    G.bind(ws, d, 'Sn1', 'ci', 'ci:aeb@latest', CP)
    G.bind(ws, d, 'Sn2', 'suite', 'suites/smoke', CP)
    G.save(ws, d)
    tab.verify_now()
    assert 'top goals G1 green' in tab.state.text()
    _suite(ws, 'smoke', [True] * 40 + [False])                 # the suite changed: Sn2 amber
    tab.reload()
    assert 'G1 amber' in tab.state.text()
    tab.select('Sn2')
    assert tab.rebind_btn.isEnabled() and tab.info.vals['Status'].text() == 'amber'
    old = G.node(G.load(ws), 'Sn2')['evidence']['sha256']
    tab.rebind()                                                 # pins the new result: 1 of 41, upper > 0.1: red
    assert G.node(G.load(ws), 'Sn2')['evidence']['sha256'] != old and 'G1 red' in tab.state.text()
    tab.undo.undo()                                              # the previous evidence is back
    assert G.node(G.load(ws), 'Sn2')['evidence']['sha256'] == old and 'G1 amber' in tab.state.text()
    out = tab.export_to(str(tmp_path / 'x.svg'))
    assert open(out).read().startswith('<svg')
    tab.grab()
