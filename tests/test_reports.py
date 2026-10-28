"""Reports (core.reports): an evaluation, a dataset's coverage and a version, a rare-case study and a Leaderboard-style
result, each rendered from what Falcon wrote against the fake carla (or, for the Leaderboard, from scored hand-made
episodes): self-contained HTML (no network asset, light and dark), the numbers of the source in it, the embedded meta
read back with the hashes of every file read, the list of reports and the export, the command. The loop iteration's
report is tested with the loop (tests/test_loop.py), on its workspace."""
import json
import os
import re

import pytest

from falcon_autonomy.core import protocol as P
from falcon_autonomy.core import provenance as PV
from falcon_autonomy.core import reports as R

from conftest import FAKE

SYN = {'id': 'syn', 'name': 'Synthetic', 'kind': 'detector', 'runtime': 'python',
       'entry': 'falcon_autonomy.plugins.synthetic_detector:SyntheticDetector', 'licence': 'GPL-3.0-or-later'}


def self_contained(doc: str):
    assert doc.startswith('<!doctype html>') and 'prefers-color-scheme:dark' in doc
    assert not re.search(r'(src|href)\s*=\s*["\']?(https?:)?//', doc), 'a network asset'
    assert '<link' not in doc and '@import' not in doc and '<img' not in doc
    assert '<script' not in doc.replace('<script type="application/json" id="falcon-report">', '')


@pytest.fixture(scope='module')
def ws(tmp_path_factory):
    """A workspace with two drives of a dataset spec (recorded by the worker against the fake carla), a version of
    them, an evaluation of the synthetic detector on them, and a two-run rare-case search."""
    from falcon_autonomy.core.project import Project
    from falcon_autonomy.core import harness as H
    from falcon_autonomy.sim import engine as E
    from test_workers import _cfg, _run
    base = tmp_path_factory.mktemp('reports')
    p = Project.new('Reports')
    p.doc['rigs'][0]['sensors'][0].update(width=160, height=96, labels=['semantic', 'instance', 'depth', 'boxes2d'])
    p.doc['traffic_profiles'][0].update(vehicles=10, walkers=10)
    p.doc['models'] = [SYN]
    p.save(str(base / 'r.falcon.json'))
    w = p.workspace
    spec = {'id': 'spec1', 'name': 'Two drives', 'rig': 'rig-default', 'fps': 20.0, 'record_every': 1,
            'drive_s': 1.0, 'worlds': ['world-town10'], 'weathers': ['ClearNoon'], 'scenarios': [],
            'targets': {'car': 100, 'pedestrian': 40}, 'bins': {'distance_m': [0, 20, 40, 1000]}, 'bin_targets': {},
            'traffic_profile': None, 'seed': 0, 'max_drives': 2}
    runs = []
    os.makedirs(os.path.join(w, 'datasets', 'spec1', 'drives'))
    for i in range(2):
        out = os.path.join(w, 'datasets', 'spec1', 'drives', f'd{i}')
        code, _, err = _run(_cfg(p, out, seconds=1.0, seed=5 + i))
        assert code == P.OK, err
        runs.append(out)
    PV.write_json(os.path.join(w, 'datasets', 'spec1', 'plan.json'), {'spec': spec, 'plan': {}, 'drives': runs})
    version = E.new_version(w, spec, 'drive', {'train': 0.5, 'val': 0.5})
    ev = H.evaluate(runs, SYN, os.path.join(w, 'evaluations', 'e1'), w, n_boot=100)
    # a rare-case search: two runs of a scenario
    from falcon_autonomy.core import scenarios as SC
    from falcon_autonomy.core.jobs import JobQueue, SubprocessLauncher
    from falcon_autonomy.core.settings import Settings
    from falcon_autonomy.sim.service import CarlaService
    from falcon_autonomy.sim.studio import Study
    p.doc['rigs'][0]['sensors'][0].update(width=48, height=32, labels=[])
    svc = CarlaService(Settings(str(base / 'settings.json')), JobQueue(SubprocessLauncher()), w, fake_carla=FAKE)
    try:
        st = Study(svc, p, SC.builtin('lead_brake'), 'study1', 'world-town10', 'rig-default', seed=1)
        search = st.search('random', 2, batch=2, threshold=0.5)
    finally:
        svc.shutdown()
    return {'ws': w, 'runs': runs, 'eval': ev, 'version': version, 'search': search, 'base': base}


def test_evaluation_report(ws, tmp_path):
    out = str(tmp_path / 'e.html')
    meta = R.write_report('evaluation', os.path.join(ws['ws'], 'evaluations', 'e1'), out)
    doc = open(out, encoding='utf-8').read()
    self_contained(doc)
    tab = {r['metric']: r for r in ws['eval']['scores']['table']}
    for k in ('AP', 'AP50', 'AP.car'):
        assert f'{tab[k]["value"]:.3f}' in doc and f'[{tab[k]["ci_low"]:.3f}, {tab[k]["ci_high"]:.3f}]' in doc
    assert 'SYNTHETIC' in doc and 'not reliable' in doc and 'bootstrap over drives (2)' in doc
    assert '<svg' in doc and '<title>AP.car — estimate' in doc                  # hover values on the marks
    ev_json = os.path.join(ws['ws'], 'evaluations', 'e1', 'evaluation.json')
    assert meta['inputs'][ev_json] == PV.sha256_file(ev_json) and PV.sha256_file(ev_json) in doc
    back = R.read_meta(out)
    assert back['kind'] == 'evaluation' and back['title'] == 'Evaluation of Synthetic' and back['inputs'] == \
        meta['inputs']
    assert 'run0/run.json' in doc                                                 # the evaluation's own provenance


def test_dataset_reports(ws, tmp_path):
    from falcon_autonomy.core import datasets as D
    out = str(tmp_path / 'd.html')
    R.write_report('dataset', os.path.join(ws['ws'], 'datasets', 'spec1'), out)
    doc = open(out, encoding='utf-8').read()
    self_contained(doc)
    rows = [D.coverage_of_run(r, {'distance_m': [0, 20, 40, 1000]}) for r in ws['runs']]
    have = sum(r['objects'].get('car', 0) for r in rows)
    assert have > 0 and f'<td class="n">{have}</td>' in doc and '<td class="n">100</td>' in doc
    assert f'{have / 100:.0%}' in doc and 'class="tgt"' in doc
    near = sum((r['bins']['distance_m'].get('car') or [0])[0] for r in rows)
    assert f'<td class="n">{near}</td>' in doc
    vpath = os.path.join(ws['ws'], 'datasets', 'spec1', 'versions', 'v001.json')
    R.write_report('dataset', vpath, str(tmp_path / 'v.html'))
    vdoc = open(tmp_path / 'v.html', encoding='utf-8').read()
    assert ws['version']['sha256'] in vdoc and 'version 1' in vdoc and f'<td class="n">{have}</td>' in vdoc


def test_study_report(ws, tmp_path):
    out = str(tmp_path / 's.html')
    R.write_report('study', os.path.join(ws['ws'], 'studies', 'study1'), out)
    doc = open(out, encoding='utf-8').read()
    self_contained(doc)
    assert 'Lead vehicle brakes' in doc and 'Search: random' in doc and 'not where a perception model fails' in doc
    for h in ws['search']['history']:
        assert f'{h["value"]:.3f}' in doc
    assert 'polyline' in doc


def test_leaderboard_report(tmp_path):
    from falcon_autonomy.core import leaderboard as LB
    eps = [{'episode': i, 'seed': i, 'scenario': {'id': 'lead_brake'}, 'route_length_m': 100.0,
            'route_completion': rc, 'distance_m': 100.0 * rc, 'termination': 'route_complete',
            'infraction_counts': cnt, 'infractions': []}
           for i, (rc, cnt) in enumerate([(1.0, {}), (0.5, {'collision_vehicle': 1}), (1.0, {'red_light': 1})])]
    doc = LB.score(eps, n_boot=200, seed=0)
    doc['provenance'] = PV.block({}, 0, {'routes': 3})
    doc['failed'] = [{'scenario': 'cut_in', 'error': 'RuntimeError: no adjacent lane'}]
    PV.write_json(str(tmp_path / 'lb' / 'leaderboard.json'), doc)
    out = str(tmp_path / 'lb.html')
    R.write_report('leaderboard', str(tmp_path / 'lb'), out)
    html = open(out, encoding='utf-8').read()
    self_contained(html)
    ds = doc['metrics']['driving_score']
    assert ds['value'] == pytest.approx((1.0 + 0.5 * 0.6 + 0.7) / 3)
    assert f'{ds["value"]:.3f}' in html and f'[{ds["ci_low"]:.3f}, {ds["ci_high"]:.3f}]' in html
    assert 'no adjacent lane' in html and 'not comparable' in html and 'collision_vehicle' in html


def test_list_export_command_and_refusals(ws, tmp_path):
    from falcon_autonomy.cli import main
    folder = R.reports_dir(ws['ws'])
    a = os.path.join(folder, R.default_name('evaluation', os.path.join(ws['ws'], 'evaluations', 'e1')))
    assert main(['report', 'evaluation', os.path.join(ws['ws'], 'evaluations', 'e1'), '--out', a]) == P.OK
    assert main(['report', 'study', os.path.join(ws['ws'], 'studies', 'study1'), '--out',
                 os.path.join(folder, 'study.html')]) == P.OK
    with open(os.path.join(folder, 'notes.html'), 'w') as fh:
        fh.write('<html>not a report</html>')
    listed = R.list_reports(folder)
    assert {m['kind'] for m in listed} == {'evaluation', 'study'} and len(listed) == 2
    copies = R.export([m['path'] for m in listed], str(tmp_path / 'out'))
    assert sorted(os.listdir(tmp_path / 'out')) == sorted(os.path.basename(m['path']) for m in listed)
    assert all(R.read_meta(c) is not None for c in copies)
    src = R.sources(ws['ws'])
    assert [x[0] for x in src['evaluation']] == ['e1'] and len(src['dataset']) == 2 and src['study']
    assert main(['report', 'evaluation', str(tmp_path / 'nope'), '--out', str(tmp_path / 'x.html')]) == P.FAILED
    assert main(['report', 'leaderboard', os.path.join(ws['ws'], 'studies', 'study1'), '--out',
                 str(tmp_path / 'y.html')]) == P.FAILED
    assert not os.path.exists(tmp_path / 'x.html') and not os.path.exists(tmp_path / 'y.html')
    with pytest.raises(R.ReportError):
        R.build('poster', ws['ws'])
    assert json.loads(json.dumps(R.read_meta(a)))['schema'] == R.REPORT_SCHEMA
