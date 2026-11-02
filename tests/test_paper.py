"""Outputs ready for papers (core.paper): LaTeX (booktabs) and CSV tables and PDF, PNG and SVG figures of an
ablation, a mixed experiment, an evaluation, a comparison and a run comparison — byte-identical for the same input —
and the data card every export writes, which never names a run that came from the user's dashcam footage."""
import csv
import io
import json
import os
import re
import xml.etree.ElementTree as ET

import pytest

from falcon_autonomy.core import ablation as A
from falcon_autonomy.core import paper as PP
from falcon_autonomy.core import protocol as P

cv2 = pytest.importorskip('cv2')


def _ablation(ws):
    s = {'id': 'abl', 'hypothesis': {'text': 'the fine-tuned detector raises night recall'}, 'metric': 'AR100',
         'factors': [{'name': 'model', 'kind': 'value', 'levels': ['base', 'ft']},
                     {'name': 'light', 'kind': 'value', 'levels': ['day', 'night']}],
         'contrasts': [{'factor': 'model', 'a': 'base', 'b': 'ft', 'by': ['light']}],
         'design': {'replicates': 6}, 'analysis': {'n_boot': 500},
         'run': {'executor': 'simulated', 'simulate': {'base': 0.6, 'noise_sd': 0.02, 'unit_sd': 0.05,
                                                        'effects': [{'when': {'model': 'ft', 'light': 'night'},
                                                                     'add': 0.06}]}}}
    A.create(ws, s)
    A.run(ws, 'abl')
    return A.path_of(ws, 'abl')


def _files(d):
    return {f: open(os.path.join(d, f), 'rb').read() for f in sorted(os.listdir(d))}


def test_ablation_outputs_are_deterministic_and_well_formed(tmp_path):
    exp = _ablation(str(tmp_path / 'ws'))
    a, b = str(tmp_path / 'a'), str(tmp_path / 'b')
    doc = PP.export('experiment', exp, a)
    PP.export('experiment', exp, b)
    fa, fb = _files(a), _files(b)
    assert fa == fb                                                     # byte-identical: tables, figures, manifest
    assert set(fa) == {f'{n}.{x}' for n in ('contrasts', 'cells') for x in ('tex', 'csv', 'pdf', 'png', 'svg')} | \
        {'paper.json'}
    assert doc['files']['contrasts.tex'] == __import__('hashlib').sha256(fa['contrasts.tex']).hexdigest()
    tex = fa['contrasts.tex'].decode()
    for k in ('\\toprule', '\\midrule', '\\bottomrule', '\\begin{tabular}{lrrrrrrrl}', '\\label{tab:abl-contrasts}',
              'needs \\usepackage{booktabs}'):
        assert k in tex, k
    assert 'model: ft $-$ base \\textbar{} light=night' in tex and '95 \\%' in tex
    row = [ln for ln in tex.splitlines() if 'light=night &' in ln][0]
    # six units: the resamples are raised until the simultaneous interval's tail and the p-value can show the effect
    assert re.search(r'& \+0\.\d{3} \[\+0\.\d{3}, \+0\.\d{3}\] &', row) and row.endswith('& supported \\\\')
    rows = list(csv.DictReader(io.StringIO(fa['contrasts.csv'].decode())))
    assert [r['label'] for r in rows] == ['model: ft − base | light=day', 'model: ft − base | light=night']
    night = rows[1]
    assert float(night['diff_ci_low']) <= 0.06 <= float(night['diff_ci_high']) and night['verdict'] == 'supported'
    assert float(night['sim_ci_low']) <= float(night['diff_ci_low'])  # the simultaneous interval is wider
    assert re.fullmatch(r'-?\d+\.\d{6}', night['diff'])
    cells = list(csv.DictReader(io.StringIO(fa['cells.csv'].decode())))
    assert len(cells) == 4 and set(cells[0]) >= {'model', 'light', 'n', 'mean', 'mean_ci_low', 'mean_ci_high'}
    # the figures: valid SVG with a mark per estimate and its value, a PDF whose cross-reference table is right, PNGs
    root = ET.fromstring(fa['contrasts.svg'])
    marks = [e for e in root.iter() if e.tag.endswith('circle')]
    assert len(marks) == 2 and 'light=night' in marks[1][0].text
    pdf = fa['contrasts.pdf']
    assert pdf.startswith(b'%PDF-1.4') and b'/CreationDate' not in pdf and pdf.rstrip().endswith(b'%%EOF')
    xref = int(pdf.rsplit(b'startxref\n', 1)[1].split(b'\n')[0])
    offs = [int(x.split()[0]) for x in pdf[xref:].split(b'\n')[3:8]]
    assert all(pdf[o:].startswith(f'{k} 0 obj'.encode()) for k, o in enumerate(offs, 1))
    img = cv2.imdecode(__import__('numpy').frombuffer(fa['cells.png'], 'uint8'), cv2.IMREAD_COLOR)
    assert img.shape[1] == 1440 and img.shape[0] > 100
    man = json.loads(fa['paper.json'])
    assert man['kind'] == 'experiment' and set(man['inputs']) == {'results.json'} and 'contrasts' in man['captions']


def _hand_made(tmp_path):
    ev = tmp_path / 'ev'
    ev.mkdir()
    table = [{'metric': 'AP', 'value': 0.41, 'ci_low': 0.36, 'ci_high': 0.47, 'n': 6},
             {'metric': 'AR100', 'value': 0.55, 'ci_low': 0.5, 'ci_high': 0.6, 'n': 6},
             {'metric': 'AP.car', 'value': 0.6, 'ci_low': 0.52, 'ci_high': 0.66, 'n': 6},
             {'metric': 'AP.range_m[0,20)', 'value': 0.7, 'ci_low': 0.6, 'ci_high': 0.8, 'n': 6}]
    (ev / 'evaluation.json').write_text(json.dumps({
        'schema': 'falcon.evaluation/1', 'model': {'id': 'm', 'name': 'Detector & co_1'}, 'n_frames': 120,
        'runs': ['r1', 'r2'], 'synthetic_model': False,
        'scores': {'unit': 'drive', 'n_units': 6, 'n_boot': 1000, 'note': None, 'table': table}}))
    cmp_ = tmp_path / 'cmp'
    cmp_.mkdir()
    (cmp_ / 'comparison.json').write_text(json.dumps({
        'schema': 'falcon.comparison/1', 'unit': 'drive', 'n_units': 6, 'n_frames': 120, 'n_boot': 1000,
        'a': {'model': {'name': 'base'}}, 'b': {'model': {'name': 'ft'}},
        'rows': [{'metric': 'AP', 'a': 0.4, 'b': 0.45, 'diff': 0.05, 'ci_low': 0.01, 'ci_high': 0.09,
                  'p_b_gt_a': 0.99},
                 {'metric': 'AR100', 'a': 0.5, 'b': 0.48, 'diff': -0.02, 'ci_low': -0.05, 'ci_high': 0.01,
                  'p_b_gt_a': 0.1}]}))
    return str(ev), str(cmp_)


def test_evaluation_comparison_and_mixed_outputs(tmp_path):
    ev, cmp_ = _hand_made(tmp_path)
    doc = PP.export('evaluation', ev, str(tmp_path / 'pe'), formats=('svg',))
    tex = open(tmp_path / 'pe' / 'scores.tex').read()
    assert 'Detector \\& co\\_1 on 120 frames of 2 drive(s)' in tex and '0.410 [0.360, 0.470]' in tex
    assert 'AP.range\\_m[0,20)' in tex and set(doc['files']) == {'scores.tex', 'scores.csv', 'scores.svg'}
    svg = open(tmp_path / 'pe' / 'scores.svg').read()
    assert svg.count('<circle') == 3                                   # headline and per-class, no range bins
    PP.export('comparison', cmp_, str(tmp_path / 'pc'), formats=('pdf', 'svg'))
    tex = open(tmp_path / 'pc' / 'comparison.tex').read()
    assert '+0.050 [+0.010, +0.090]' in tex and '$-$0.020 [$-$0.050, +0.010]' in tex
    assert 'ft (B) against base (A)' in tex and open(tmp_path / 'pc' / 'comparison.svg').read().count(
        'stroke-dasharray') == 1                                       # the line at 0
    mixed = tmp_path / 'mixed'
    mixed.mkdir()
    ap = lambda v: {'value': v, 'ci_low': v - 0.03, 'ci_high': v + 0.03}         # noqa: E731
    sp = {'AP': {'sd': 0.01}}
    (mixed / 'results.json').write_text(json.dumps({
        'schema': 'falcon.mixed_results/1', 'experiment': 'mx', 'budget_images': 400, 'seeds': [0, 1, 2],
        'test': {'images': 50, 'unit': 'sequence', 'n_units': 9}, 'warnings': [],
        'arms': [{'arm': '0%', 'n_real': 400, 'n_synthetic': 0, 'AP': ap(0.3), 'seed_spread': sp,
                  'vs_real_only': None},
                 {'arm': '50%', 'n_real': 200, 'n_synthetic': 200, 'AP': ap(0.34), 'seed_spread': sp,
                  'vs_real_only': {'AP': {'diff': 0.04, 'ci_low': 0.01, 'ci_high': 0.07,
                                          'verdict': 'better than real only'}}}]}))
    PP.export('experiment', str(mixed), str(tmp_path / 'pm'), formats=('png',))
    tex = open(tmp_path / 'pm' / 'arms.tex').read()
    assert '50\\% & 200 & 200 & 0.340 [0.310, 0.370] & 0.010 & +0.040 [+0.010, +0.070] & better than real only' \
        in tex
    with pytest.raises(PP.PaperError, match='no results.json'):
        PP.export('experiment', str(tmp_path), str(tmp_path / 'x'))


def _fake_run(ws, name, settings, steps=3):
    d = os.path.join(ws, 'runs', name)
    os.makedirs(d)
    with open(os.path.join(d, 'run.json'), 'w') as fh:
        json.dump({'schema': 'falcon.run/1', 'worker': 'drive', 'status': 'ok', 'name': name,
                   'started': '2026-10-01T00:00:00Z', 'performance': {'steps': steps, 'sim_s': steps * 0.05},
                   'provenance': {'settings': settings}}, fh)
    with open(os.path.join(d, 'frames.jsonl'), 'w') as fh:
        for k in range(steps):
            fh.write(json.dumps({'step': k, 'frame': k, 'sim_t': 0.05 * k, 'speed_mps': 3.0 + k,
                                 'ego': {'x': k, 'y': 0, 'z': 0, 'roll': 0, 'pitch': 0, 'yaw': 0}, 'files': {}})
                     + '\n')
    return d


def test_a_run_comparison_never_names_a_dashcam_run(tmp_path):
    ws = str(tmp_path / 'ws')
    a = _fake_run(ws, 'plain-drive', {'seed': 1, 'fps': 20.0})
    b = _fake_run(ws, 'secret-route-name', {'seed': 2, 'fps': 20.0, 'dashcam_route': {'route_sha256': 'x'}})
    doc = PP.export('runs', [a, b], str(tmp_path / 'pr'))
    assert doc['no_figure'] and set(doc['files']) == {'runs.tex', 'runs.csv'}
    out = ''.join(open(os.path.join(tmp_path, 'pr', f)).read() for f in os.listdir(tmp_path / 'pr'))
    assert 'plain-drive' in out and 'secret-route-name' not in out and 'run 2' in out
    tex = open(tmp_path / 'pr' / 'runs.tex').read()
    assert 'Frames & 3.000 & 3.000' in tex
    with pytest.raises(PP.PaperError, match='two to four'):
        PP.export('runs', [a], str(tmp_path / 'x'))


# ------------------------------------------------------------------------------------------------ data cards
@pytest.fixture(scope='module')
def exported(tmp_path_factory):
    """Two drives on the fake carla, the second marked as driven along a dashcam route, exported as COCO."""
    from falcon_autonomy.core import exporters as E
    from falcon_autonomy.core.project import Project
    from test_workers import _cfg, _run
    base = tmp_path_factory.mktemp('card')
    p = Project.new('Card')
    p.doc['rigs'][0]['sensors'][0].update(width=160, height=96, labels=['semantic', 'instance', 'boxes2d'])
    p.doc['traffic_profiles'][0]['vehicles'] = 6
    p.save(str(base / 'card.falcon.json'))
    runs = {}
    for name, seed in (('day_run', 3), ('private_run', 4)):
        code, _, err = _run(_cfg(p, base / name, seconds=1.0, seed=seed))
        assert code == P.OK, err
        runs[name] = str(base / name)
    rj = os.path.join(runs['private_run'], 'run.json')
    doc = json.load(open(rj))
    doc['provenance']['settings']['dashcam_route'] = {'route_sha256': 'x'}
    json.dump(doc, open(rj, 'w'))
    out = str(base / 'coco')
    rep = E.export([], 'coco', out, {'splits': {'train': [runs['day_run']], 'val': [runs['private_run']]}})
    return out, rep, runs


def test_every_export_writes_its_data_card(exported):
    out, rep, runs = exported
    assert rep['datacard'] == 'DATACARD.md'
    md = open(os.path.join(out, 'DATACARD.md')).read()
    card = json.load(open(os.path.join(out, 'datacard.json')))
    for h in ('## Composition', '## Collection process', '## Labels', '## Conditions covered', '## Known gaps',
              '## Licence', '## Provenance'):
        assert h in md, h
    assert card['composition']['runs'] == 2 and card['composition']['objects_per_class']
    assert card['collection']['simulator'] == ['0.9.16'] and card['labels']['source'].startswith('the simulator')
    assert 'no night frames' in card['known_gaps'] and any('CC-BY' in x for x in card['licence'])
    assert card['provenance']['export_manifest_sha256'] and len(card['provenance']['runs']) == 2
    # the dashcam-derived run: no name, no folder, no place, no date
    assert 'private_run' not in md and 'private_run' not in json.dumps(card)
    assert runs['private_run'] not in json.dumps(card) and card['export_created'] is None
    assert '1 run(s) come from the user\'s own dashcam footage' in md
    assert [r['run'] for r in card['provenance']['runs']] == ['day_run', 'run 2']
    # the same export gives the same card; the command writes it again
    from falcon_autonomy.cli import main
    before = open(os.path.join(out, 'DATACARD.md'), 'rb').read()
    assert main(['paper', 'datacard', out]) == 0
    assert open(os.path.join(out, 'DATACARD.md'), 'rb').read() == before
    assert main(['paper', 'datacard', os.path.dirname(out)]) == 2


def test_the_paper_command_and_sources(tmp_path):
    from falcon_autonomy.cli import main
    ws = str(tmp_path / 'ws')
    exp = _ablation(ws)
    out = str(tmp_path / 'out')
    assert main(['paper', 'experiment', exp, '--out', out, '--formats', 'svg']) == 0
    assert sorted(os.listdir(out)) == ['cells.csv', 'cells.svg', 'cells.tex', 'contrasts.csv', 'contrasts.svg',
                                       'contrasts.tex', 'paper.json']
    assert main(['paper', 'experiment', exp, '--out', out, '--formats', 'gif']) == 2
    assert main(['paper', 'experiment', exp]) == 2                    # --out is needed
    src = PP.sources(ws)
    assert src['experiment'] == [('abl', exp)] and src['comparison'] == [] and src['datacard'] == []
