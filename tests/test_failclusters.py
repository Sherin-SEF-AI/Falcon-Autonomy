"""Failure clustering (core.failclusters): Gower's distance by hand, average linkage by the nearest-neighbour chain
against a brute-force agglomeration, a workspace with two planted failure modes across a rare-case study, a suite and
a CI run (and a root-cause answer) clustered and named deterministically, a new CI run's failures sorted into a known
mode and a new one (also at the end of `falcon ci run`), the commands; and the assistant's side: the failure tools
(private runs left out, the triage queue, comments, a bisect's preview and answer), the new-failure-mode suggestion,
and cluster names reworded by a fake fast model kept only when grounded."""
import json
import os
import subprocess

import numpy as np
import pytest

from conftest import FAKE
from falcon_autonomy.core import failclusters as FC


def _w(path, doc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        if isinstance(doc, list):
            fh.write(''.join(json.dumps(r) + '\n' for r in doc))
        else:
            json.dump(doc, fh)


def _run(ws, rel, sun=45.0, fog=0.0, rain=0.0, world='world-town10', private=False):
    d = os.path.join(ws, rel)
    st = {'weather': {'sun_altitude_angle': sun, 'fog_density': fog, 'precipitation': rain},
          'project': {'world': world}}
    if private:
        st['dashcam_route'] = {'route_sha256': 'x'}
    _w(os.path.join(d, 'run.json'), {'status': 'ok', 'name': os.path.basename(d), 'started': '2026-10-30T10:00:00Z',
                                     'provenance': {'settings': st}})
    return d


def lead_brake_row(ws, i, speed, crit=0.9, private=False):
    run = _run(ws, f'studies/search-lead_brake/runs/r{i}', sun=-20.0, fog=60.0 + i, private=private)
    return {'key': f'k{i}', 'values': {'speed': speed, 'distance': 30.0 + i}, 'run': run, 'status': 'ok',
            'objective': crit, 'criticality': crit, 'rss': {'label': 'avoidable'},
            'scenario': {'id': 'lead_brake', 'checks': [{'name': 'no collision', 'ok': False}],
                         'criticality': {'overall': {'collision': True, 'min_ttc_s': 0.0}}}}


def make_workspace(ws, private_one=False):
    """Mode A: lead_brake at night in fog, fast, colliding (a search's 10 failures, a CI run's 3); mode B: jaywalker by
    day, missing its TTC check without a collision (a suite's 6). Non-failures beside them."""
    rows = [lead_brake_row(ws, i, 25.0 + 0.5 * i, private=private_one and i == 0) for i in range(10)]
    rows += [dict(lead_brake_row(ws, 10 + i, 10.0), objective=0.2, criticality=0.2) for i in range(4)]
    _w(os.path.join(ws, 'studies', 'search-lead_brake', 'study.json'), {'scenario': 'lead_brake'})
    _w(os.path.join(ws, 'studies', 'search-lead_brake', 'search-bayes.json'), {'threshold': 0.5})
    _w(os.path.join(ws, 'studies', 'search-lead_brake', 'results.jsonl'), rows)
    suite = []
    for i in range(8):
        run = _run(ws, f'suites/nightly/runs/j{i}', sun=50.0)
        suite.append({'key': f'j{i}', 'entry': 0, 'scenario': 'jaywalker', 'seed': i, 'run': run, 'status': 'ok',
                      'passed': i >= 6, 'values': {'walker_speed': 1.2 + 0.05 * i},
                      'checks': [{'name': 'min TTC ≥ 1 s', 'ok': i >= 6}], 'criticality': 0.6, 'min_ttc_s': 0.6,
                      'collision': False})
    _w(os.path.join(ws, 'suites', 'nightly', 'results.jsonl'), suite)
    ci = []
    for i in range(4):
        run = _run(ws, f'ci/run-1/suite/runs/c{i}', sun=-20.0, fog=70.0)
        ci.append({'id': f'c{i}', 'pair': f'p{i}', 'group': 'lead_brake', 'seed': i, 'status': 'ok', 'passed': i == 3,
                   'values': {'speed': 27.0, 'distance': 35.0}, 'dir': run, 'failed_checks': ['no collision: hit'],
                   'metrics': {'collision': True, 'min_ttc_s': 0.0, 'criticality': 1.0}})
    _w(os.path.join(ws, 'ci', 'run-1', 'ci-runs.jsonl'), ci)
    _w(os.path.join(ws, 'rootcause', '20261030-why', 'rootcause.json'), {
        'kind': 'counterfactual', 'status': 'attributed', 'case': {'run': rows[1]['run']},
        'factors': [{'factor': 'fog', 'present': True, 'removing_fixes': True},
                    {'factor': 'lighting', 'present': True, 'removing_fixes': False}]})
    return rows


def new_ci(ws, name='run-2'):
    """Two failures like mode A (known) and two of a cut-in by day that misses its TTC (new)."""
    recs = []
    for i in range(2):
        run = _run(ws, f'ci/{name}/suite/runs/a{i}', sun=-20.0, fog=65.0)
        recs.append({'id': f'a{i}', 'pair': f'a{i}', 'group': 'lead_brake', 'seed': i, 'status': 'ok', 'passed': False,
                     'values': {'speed': 27.5, 'distance': 34.0}, 'dir': run, 'failed_checks': ['no collision: hit'],
                     'metrics': {'collision': True, 'min_ttc_s': 0.0, 'criticality': 1.0}})
    for i in range(2):
        run = _run(ws, f'ci/{name}/suite/runs/b{i}', sun=60.0, rain=80.0, world='world-blr')
        recs.append({'id': f'b{i}', 'pair': f'b{i}', 'group': 'cut_in', 'seed': i, 'status': 'ok', 'passed': False,
                     'values': {'gap': 4.0 + i}, 'dir': run, 'failed_checks': ['min TTC ≥ 1 s: 0.3'],
                     'metrics': {'collision': False, 'min_ttc_s': 0.3, 'criticality': 0.7}})
    _w(os.path.join(ws, 'ci', name, 'ci-runs.jsonl'), recs)
    return os.path.join(ws, 'ci', name), recs


# --------------------------------------------------------------------------------------------- the method
def test_gower_distance_by_hand():
    a = {'key': 'a', 'scenario': 's', 'values': {'speed': 10.0, 'kind': 'car'}, 'conditions': {'tod': 'night'},
         'outcome': {'collision': True, 'checks': ['x', 'y']}}
    b = {'key': 'b', 'scenario': 's', 'values': {'speed': 20.0, 'kind': 'bus'}, 'conditions': {'tod': 'night'},
         'outcome': {'collision': True, 'checks': ['y']}}
    c = {'key': 'c', 'scenario': 't', 'values': {'speed': 30.0}, 'outcome': {'checks': []}, 'causes': ['fog']}
    D = FC.distance_matrix([a, b, c])
    # a–b: scenario 0; parameters (|10−20|/20 + 1)/2 = 0.75; conditions 0; outcome (0 + Jaccard 1/2)/2 = 0.25
    assert D[0, 1] == pytest.approx((0 + 0.75 + 0 + 0.25) / 4)
    # a–c: scenario 1; parameters: speed only, 1.0; outcome: checks only, Jaccard 1; no conditions or cause in common
    assert D[0, 2] == pytest.approx((1 + 1 + 1) / 3)
    assert np.allclose(D, D.T) and np.all(np.diag(D) == 0) and D.min() >= 0 and D.max() <= 1


def test_average_linkage_matches_brute_force():
    def brute(D, cut):
        groups = [[i] for i in range(len(D))]
        while True:
            best = None
            for x in range(len(groups)):
                for y in range(x + 1, len(groups)):
                    d = D[np.ix_(groups[x], groups[y])].mean()
                    if best is None or d < best[0]:
                        best = (d, x, y)
            if best is None or best[0] > cut:
                return sorted(sorted(g) for g in groups)
            groups[best[1]] += groups.pop(best[2])
    for seed in range(12):
        X = np.random.RandomState(seed).rand(22, 3)
        D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1)) / np.sqrt(3)
        merges = FC.linkage(D)
        assert len(merges) == 21
        for cut in (0.1, 0.25, 0.4):
            lab = FC.labels_at(22, merges, cut)
            got: dict = {}
            for i, g in enumerate(lab):
                got.setdefault(g, []).append(i)
            assert sorted(sorted(v) for v in got.values()) == brute(D, cut), (seed, cut)
    assert FC.cluster([]) == [] and FC.cluster([{'key': 'x', 'source': 's', 'scenario': 'a'}])[0]['size'] == 1


# --------------------------------------------------------------------------------------------- a workspace
def test_two_planted_modes_found_and_named(tmp_path):
    ws = str(tmp_path / 'p.falcon')
    make_workspace(ws)
    fails = FC.collect(ws)
    by = {s: sum(1 for f in fails if f['source'] == s) for s in FC.SOURCES}
    assert by == {'search': 10, 'suite': 6, 'ci': 3}
    f1 = next(f for f in fails if f['key'] == 'search:search-lead_brake#1')
    assert f1['causes'] == ['fog'] and f1['conditions']['tod'] == 'night' and f1['outcome']['rss'] == 'avoidable'
    doc = FC.build(ws)
    cl = doc['clusters']
    assert [c['size'] for c in cl] == [13, 6], [(c['name'], c['size']) for c in cl]
    a, b = cl[:2]                                      # (the CI run's cut-in is a third)
    assert a['scenarios'] == {'lead_brake': 13} and a['sources'] == {'search': 10, 'ci': 3}
    assert a['name'].startswith('lead_brake, ') and 'collision' in a['name'], a['name']
    assert any(f['text'] == 'night' for f in a['features']) and any(f['text'].startswith('fog ≥') for f in
                                                                       a['features'])
    assert b['name'].startswith('jaywalker, ') and 'fails "min TTC ≥ 1 s"' in [f['text'] for f in b['features']]
    assert a['id'].startswith('C-') and a['cohesion'] < 0.2
    again = FC.build(ws, save=False)
    assert [(c['id'], c['name'], c['members']) for c in again['clusters']] == \
        [(c['id'], c['name'], c['members']) for c in cl]                                      # deterministic
    assert FC.load(ws)['clusters'][0]['id'] == a['id'] and not os.path.exists(FC.clusters_path(ws) + '.lock')
    assert FC.build(ws, sources=('suite',), save=False)['failures'] == 6


def test_a_ci_runs_new_failure_mode(tmp_path):
    from falcon_autonomy.core import ci as CI
    ws = str(tmp_path / 'p.falcon')
    make_workspace(ws)
    known = {c['name']: c['id'] for c in FC.build(ws)['clusters']}
    d, recs = new_ci(ws)
    doc = FC.check_ci(ws, d)
    assert doc['failures'] == 4 and doc['reference'] == {'failures': 19, 'clusters': 2}
    assert sorted(k['key'] for k in doc['known']) == ['ci:run-2:a0', 'ci:run-2:a1']
    assert {k['cluster'] for k in doc['known']} == {known[next(n for n in known if n.startswith('lead_brake'))]}
    assert sorted(k['key'] for k in doc['new']) == ['ci:run-2:b0', 'ci:run-2:b1']
    (m,) = doc['new_modes']
    assert m['size'] == 2 and m['name'].startswith('cut_in, '), m['name']
    assert json.load(open(os.path.join(d, 'failure-modes.json')))['new_modes'][0]['name'] == m['name']
    # what `falcon ci run` does at its end: OUT inside WORKSPACE/ci names the workspace
    res = {'notes': [], 'runs': recs}
    CI.failure_modes({'project': ''}, d, res, recs)
    assert res['failure_modes']['new'] == 2 and res['failure_modes']['new_modes'][0]['name'] == m['name']
    md = CI.markdown(dict(res, runs=[], verdict='failed', exit_code=1, name='x', gates=[], summary=None, commit={},
                          subject={'kind': 'stack', 'label': 'x'}, carla={'mode': 'fake'}))
    assert '**New failure modes**: 2 of 4 failing runs' in md
    res2 = {'notes': [], 'runs': recs}
    CI.failure_modes({'project': ''}, str(tmp_path / 'elsewhere'), res2, recs)        # no workspace: nothing
    assert 'failure_modes' not in res2


def test_commands(tmp_path, capsys):
    from falcon_autonomy.cli import main
    ws = str(tmp_path / 'p.falcon')
    make_workspace(ws)
    assert main(['clusters', 'build', ws]) == 0
    out = capsys.readouterr().out
    assert '19 failures in 2 clusters' in out
    cid = FC.load(ws)['clusters'][1]['id']
    assert main(['clusters', 'show', ws, cid]) == 0 and 'jaywalker' in capsys.readouterr().out
    d, _ = new_ci(ws)
    assert main(['clusters', 'check', ws, d]) == 0
    assert '2 new (1 new modes)' in capsys.readouterr().out
    assert main(['clusters', 'build', ws, '--sources', 'nope']) == 2
    assert main(['clusters', 'show', str(tmp_path / 'empty.falcon')]) == 2


# --------------------------------------------------------------------------------------------- the assistant
@pytest.fixture
def ctx(project):
    from falcon_autonomy.assistant import tools as TOOLS
    c = TOOLS.context(project.path, FAKE)
    yield c
    c.service.shutdown(grace_s=10.0)


def test_failure_tools(ctx, tmp_path, monkeypatch):
    from falcon_autonomy.assistant.tool_api import REGISTRY, call
    from falcon_autonomy.core import team as TM
    monkeypatch.setenv('FALCON_USER', 'asha')
    ws = ctx.workspace
    make_workspace(ws, private_one=True)
    r = call(ctx, 'research_clusters', {})
    assert r['failures'] == 18 and r['private_left_out'] == 1 and r['total_clusters'] == 2
    assert r['clusters'][0]['name'].startswith('lead_brake') and 'search:search-lead_brake#0' not in \
        json.dumps(r)                                                                       # the private one
    one = call(ctx, 'research_clusters', {'cluster': r['clusters'][1]['id']})
    assert one['members']['total'] == 6 and one['members']['items'][0]['run'].startswith('suites/nightly/runs/')
    assert 'no cluster' in call(ctx, 'research_clusters', {'cluster': 'C-0'})['error']
    new_ci(ws)
    r = call(ctx, 'research_clusters', {'ci': 'run-2'})
    assert len(r['new']) == 2 and r['new_modes'][0]['name'].startswith('cut_in')
    assert not os.path.exists(os.path.join(ws, 'ci', 'run-2', 'failure-modes.json'))      # a read tool writes nothing
    # comments and the triage queue
    edit = 'research_triage_edit'
    pv = REGISTRY[edit].preview(ctx, action='comment', ref='run:suites/nightly/runs/j0', text='late brake')
    assert 'via the assistant' in pv and 'late brake' in pv
    c = call(ctx, edit, {'action': 'comment', 'ref': 'run:suites/nightly/runs/j0', 'text': 'late brake'})
    assert c['by'] == 'asha' and TM.comments(ws, 'run:suites/nightly/runs/j0')[0]['via'] == 'assistant'
    priv = 'run:studies/search-lead_brake/runs/r0'
    assert 'private' in call(ctx, edit, {'action': 'comment', 'ref': priv, 'text': 'x'})['error']
    assert 'private' in call(ctx, 'research_triage', {'ref': priv})['error']
    it = call(ctx, edit, {'action': 'add', 'title': 'Lead brake at night', 'links': ['scenario:lead_brake'],
                          'to': 'ravi'})
    assert it['item'] == 'T-1' and it['state'] == 'assigned'
    assert 'not a step' in call(ctx, edit, {'action': 'state', 'item': 'T-1', 'to': 'verified'})['error']
    assert call(ctx, edit, {'action': 'state', 'item': 'T-1', 'to': 'investigating'})['state'] == \
        'investigating'
    TM.create_item(ws, 'secret drive', {'user': 'asha', 'host': 'h'}, [priv])
    q = call(ctx, 'research_triage', {})
    assert q['total'] == 2 and q['items'][0] == {'id': 'T-2', 'state': 'open', 'private':
                                                 'links a run from the dashcam footage: withheld'}
    full = call(ctx, 'research_triage', {'item': 'T-1'})
    assert full['history'][-1]['via'] == 'assistant' and full['assignee'] == 'ravi'
    assert 'secret' not in json.dumps(call(ctx, 'research_triage', {'item': 'T-2'}))
    assert call(ctx, 'research_triage', {'ref': 'run:suites/nightly/runs/j0'})['comments']['items'][0]['text'] == \
        'late brake'


def test_bisect_tool_preview_refusals_and_answer(ctx, tmp_path):
    from falcon_autonomy.assistant.tool_api import REGISTRY, call
    repo = tmp_path / 'stack'
    repo.mkdir()
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    (repo / 'falcon-ci.yml').write_text('schema: falcon.ci/1\n')
    pv = REGISTRY['research_bisect'].preview(ctx, repo=str(repo), config='falcon-ci.yml', good='v1', bad='HEAD',
                                         repeats=3, scenarios=['right_turn'], name='rt')
    assert pv == (f'falcon bisect run {repo} --config falcon-ci.yml --good v1 --bad HEAD --out '
                  f'{ctx.workspace}/bisect/rt --repeats 3 --scenario right_turn --fake-carla {FAKE}')
    assert 'git repository' in call(ctx, 'research_bisect', {'repo': str(tmp_path), 'config': 'x', 'good': 'a',
                                                        'bad': 'b'})['error']
    assert 'git repository' in call(ctx, 'research_bisect', {'repo': 'stack', 'config': 'x', 'good': 'a',
                                                        'bad': 'b'})['error']
    assert 'no nope.yml' in call(ctx, 'research_bisect', {'repo': str(repo), 'config': 'nope.yml', 'good': 'a',
                                                     'bad': 'b'})['error']
    assert 'config is' in call(ctx, 'research_bisect', {'repo': str(repo), 'config': '../x', 'good': 'a',
                                                   'bad': 'b'})['error']
    _w(os.path.join(ctx.workspace, 'bisect', 'b1', 'bisect.json'), {
        'good': {'short': 'aaa', 'subject': 'ok'}, 'bad': {'short': 'bbb'}, 'commits': [{}, {}], 'repeats': 1,
        'rule': 'deterministic', 'endpoints': [{'scenario': 'rt', 'regressed': True}],
        'bisect': {'rt': {'status': 'found', 'first_bad': {'short': 'ccc', 'subject': 'speed', 'author': 'A'},
                          'candidates': [{'short': 'ccc'}], 'tested': 2}}})
    r = call(ctx, 'analysis_outputs', {'kind': 'bisect', 'id': 'b1'})
    assert r['bisect']['rt']['first_bad']['short'] == 'ccc' and r['commits'] == 2
    assert call(ctx, 'analysis_outputs', {'kind': 'bisect'})['items'][0]['id'] == 'b1'


def test_suggestion_and_grounded_cluster_names(tmp_path):
    from falcon_autonomy.assistant import suggest as S
    ws = str(tmp_path / 'p.falcon')
    make_workspace(ws)
    d, _ = new_ci(ws)
    FC.check_ci(ws, d)
    (s,) = S.t_new_failure_mode(S.State(workspace=ws))
    assert s.tool == 'research_clusters' and s.args == {'ci': 'run-2'} and 'triage' in s.pages
    assert '2 of 4 failing runs fall in no known failure cluster' in s.why and S.grounded(s.why, s)
    assert s.key in {x.key for x in S.gather(S.State(workspace=ws))}
    cl = FC.build(ws, save=False)['clusters']
    a, b = cl[:2]                                      # (the CI run's cut-in is a third)
    num = next(f['text'] for f in a['features'] if f['text'].startswith('fog ≥'))

    class Svc:
        def quick(self, prompt, context=None, system='', max_tokens=0, cancel=None):
            self.ctx = context
            return json.dumps({'names': [{'id': a['id'], 'name': f'lead car braking at night, {num}'},
                                         {'id': b['id'], 'name': 'jaywalker at 99 km/h'}]})
    svc = Svc()
    res = S.phrase_clusters(svc, cl)
    assert res['names'] == {a['id']: f'lead car braking at night, {num}'} and b['id'] in res['rejected']
    assert [c['id'] for c in svc.ctx['clusters']][:2] == [a['id'], b['id']]
