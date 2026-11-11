"""The safety case's evidence from elsewhere and from models (core.gsn): a failure-rate estimate run on two local
daemons is bound by hash like a local one and records where its runs were made (each node, the jobs' global ids), and
turns amber when its runs were since made elsewhere; one pulled back from a cloud burst records the provider and the
burst; a trained model's data_root claim — its ONNX falcon.data_merkle_root against the data it is bound to, a dataset
version or its own data.json — is green when they agree, amber when the model or the data changes, red when the roots
(or the version it names, or its frames) disagree; and the SOTIF template uses both."""
import json
import os
import sys

import pytest

from cluster_helpers import Daemons
from conftest import FAKE
from falcon_autonomy.core import gsn as G


def _doc(*solutions):
    nodes = [{'id': 'G1', 'type': 'Goal', 'text': 'safe'}]
    links = []
    for sid, ev in solutions:
        nodes.append({'id': sid, 'type': 'Solution', 'text': sid, **({'evidence': ev} if ev else {})})
        links.append({'from': 'G1', 'to': sid, 'kind': 'SupportedBy'})
    return {'schema': G.SCHEMA, 'title': 't', 'template': 'blank', 'nodes': nodes, 'links': links}


# ============================================================================================ where it ran
@pytest.fixture
def daemons(tmp_path):
    from falcon_autonomy.cluster.launcher import close_all
    d = Daemons(tmp_path / 'nodes')
    yield d
    close_all()
    d.stop()


def test_a_failure_rate_estimated_on_the_cluster_records_where_it_ran(tmp_path, daemons, monkeypatch):
    from falcon_autonomy.core.project import Project
    from falcon_autonomy.core.settings import Settings
    from falcon_autonomy.sim.estimate import main_failrate
    a, b = daemons.start('node-a'), daemons.start('node-b')
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
    Settings().set('cluster', {'nodes': [a, b], 'pools': {}})
    p = Project.new('T')
    p.doc['rigs'][0]['sensors'][0].update(width=48, height=32, labels=[])
    p.save(str(tmp_path / 'p.falcon.json'))
    assert main_failrate(['run', p.path, '--scenario', 'lead_brake', '--name', 'rate-c', '--stage-size', '10',
                          '--final', '10', '--max-stages', '1', '--batch', '4', '--on', 'cluster',
                          '--fake-carla', FAKE]) == 0
    ws = p.workspace
    sd = os.path.join(ws, 'studies', 'rate-c')
    assert os.path.isfile(os.path.join(sd, 'failure-rate.json'))
    doc = _doc(('Sn1', None))
    ev = G.bind(ws, doc, 'Sn1', 'failrate', 'studies/rate-c')
    w = ev['where']
    assert w['on'] == 'cluster' and set(w['nodes']) == {'node-a', 'node-b'} and w['local_runs'] == 0
    assert w['runs'] == sum(w['nodes'].values()) >= 20 and len(w['gids']) == w['runs']
    assert all(len(g) == 20 for g in w['gids']) and w['pools'] == ['default'] and w['kinds'] == ['local']
    assert ev['sha256'] == G._sha(os.path.join(sd, 'failure-rate.json'))           # bound by hash like a local one
    v = G.verify(ws, doc)
    assert v['nodes']['Sn1']['status'] == 'green' and v['nodes']['Sn1']['where'] == w
    assert G.where_text(w).startswith('on the cluster (node-a ') and f'; {w["runs"]} jobs)' in G.where_text(w)
    # one of its runs made again elsewhere (its node block changed): amber until re-bound
    rows = [json.loads(x) for x in open(os.path.join(sd, 'results.jsonl'))]
    rj = os.path.join(rows[0]['run'], 'run.json')
    run = json.load(open(rj))
    run['node'].update(name='node-z', dispatch='f' * 20 + '.1')
    json.dump(run, open(rj, 'w'))
    r = G.verify(ws, doc)['nodes']['Sn1']
    assert r['status'] == 'amber' and 'node-z 1' in r['why'] and 're-bind' in r['why']
    G.bind(ws, doc, 'Sn1', 'failrate', 'studies/rate-c')
    assert G.verify(ws, doc)['nodes']['Sn1']['status'] == 'green'
    sotif = G.template('sotif', ws)
    texts = [n['text'] for n in sotif['nodes'] if (n.get('evidence') or {}).get('ref') == 'studies/rate-c']
    assert texts and 'on the cluster (node-a' in texts[0]


def test_a_result_from_a_cloud_burst_names_the_provider_and_the_burst(tmp_path):
    from falcon_autonomy.cluster import cloud as CL
    from falcon_autonomy.cluster import protocol as W
    src = tmp_path / 'machine' / 'summary'
    src.mkdir(parents=True)
    (src / 'failure-rate.json').write_text(json.dumps({'schema': 'falcon.failure_rate/1', 'estimate': {
        'status': 'ok', 'p': 1e-4, 'ci95': [5e-5, 2e-4], 'failures': 30, 'n': 200}, 'final_runs': []}))
    (src / 'workspace.json').write_text(json.dumps({'workspace': '/machine/ws', 'files': ['failure-rate.json']}))
    ws = str(tmp_path / 'ws')
    CL._import_summary(W.pack_dir(str(src), skip=()), ws, {'name': 'rate-cloud', 'no_render': True},
                       {'provider': 'runpod', 'burst': '20261112100000-abc123', 'gpu': 'NVIDIA RTX A5000',
                        'estimate_id': '0123456789abcdef', 'instance_id': 'pod1'})
    doc = _doc(('Sn1', None))
    ev = G.bind(ws, doc, 'Sn1', 'failrate', 'studies/rate-cloud', {'measure': 'failure_rate', 'bound': 'ci95_upper',
                                                                   'limit': 1e-3})
    assert ev['where'] == {'on': 'cloud', 'provider': 'runpod', 'burst': '20261112100000-abc123',
                           'gpu': 'NVIDIA RTX A5000', 'estimate_id': '0123456789abcdef', 'instance_id': 'pod1'}
    r = G.verify(ws, doc)['nodes']['Sn1']
    assert r['status'] == 'green' and 'ci95_upper 0.0002 ≤ 0.001' in r['why']
    assert G.where_text(ev['where']) == 'on a cloud burst (runpod NVIDIA RTX A5000, burst 20261112100000-abc123)'
    ev2 = G.bind(ws, doc, 'Sn1', 'failrate', 'studies/rate-cloud', {'measure': 'failure_rate',
                                                                    'bound': 'ci95_upper', 'limit': 1e-4})
    assert ev2['where'] == ev['where'] and G.verify(ws, doc)['nodes']['Sn1']['status'] == 'red'


# ============================================================================================ a model's data
def _onnx(path):
    onnx = pytest.importorskip('onnx')
    from onnx import TensorProto, helper
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1])
    g = helper.make_graph([helper.make_node('Identity', ['x'], ['y'])], 'g', [x], [y])
    onnx.save(helper.make_model(g), path)


def _training(ws, name, frames, version_label, alg='s256'):
    """A training folder as core.training leaves it: data.json with per-sample addresses and roots, model.onnx whose
    metadata names them (core.training.lineage_props)."""
    from falcon_autonomy.core import cas as C
    from falcon_autonomy.core import lineage as L
    from falcon_autonomy.core.training import lineage_props
    d = os.path.join(ws, 'training', name)
    os.makedirs(d)
    rows = [{'id': f's{i}', 'image': f'/nowhere/{i}.jpg', 'frame': f,
             'label': C.hash_bytes(json.dumps({'boxes': [[i, 0, 1, 1]]}).encode(), alg)} for i, f in enumerate(frames)]
    root = L.pairs_root(((r['frame'], r['label']) for r in rows), alg)
    doc = {'hash_alg': alg, 'manifest_sha': 'm' * 64, 'dataset_version': version_label,
           'train': {'images': rows, 'merkle_root': root, 'hash_alg': alg}}
    with open(os.path.join(d, 'data.json'), 'w') as fh:
        json.dump(doc, fh)
    _onnx(os.path.join(d, 'model.onnx'))
    L.set_onnx_metadata(os.path.join(d, 'model.onnx'), lineage_props(d))
    return d


@pytest.fixture
def lineage_ws(tmp_path, monkeypatch):
    pytest.importorskip('pyarrow')
    pytest.importorskip('onnx')
    monkeypatch.setitem(sys.modules, 'blake3', None)                      # SHA-256, as on the reference machine
    from test_lineage import make_run
    from falcon_autonomy.core import datasets as D
    from falcon_autonomy.core import lineage as L
    ws = str(tmp_path / 'ws')
    runs = [make_run(ws, n, seed=k) for k, n in enumerate(('a', 'b', 'c'))]
    v = D.make_version({'id': 'ds1', 'name': 'Set', 'seed': 0}, runs, {'train': ['runs/a', 'runs/b'],
                                                                       'test': ['runs/c']}, 'drive', 1, ws)
    vpath = os.path.join(ws, 'datasets', 'ds1', 'versions', 'v001.json')
    os.makedirs(os.path.dirname(vpath))
    with open(vpath, 'w') as fh:
        json.dump(v, fh)
    rows, _meta = L.read_table(os.path.join(ws, v['lineage']['splits']))
    train = sorted({r['frame'] for r in rows if r['split'] == 'train' and r['sensor'] == 'cam'})
    test = sorted({r['frame'] for r in rows if r['split'] == 'test' and r['sensor'] == 'cam'})
    label = G._version_label(v)
    return {'ws': ws, 'version': os.path.relpath(vpath, ws), 'label': label, 'train': train, 'test': test}


def test_a_model_s_data_root_against_its_dataset_version(lineage_ws):
    from falcon_autonomy.core import lineage as L
    ws, ver = lineage_ws['ws'], lineage_ws['version']
    good = _training(ws, 'good', lineage_ws['train'], lineage_ws['label'])
    _training(ws, 'leaky', lineage_ws['train'][:3] + lineage_ws['test'][:1], lineage_ws['label'])
    _training(ws, 'other', lineage_ws['train'], 'Another v4 (ds9 sha256 0123456789ab)')
    claim = {'measure': 'data_root', 'data': ver}
    doc = _doc(('Sn1', None), ('Sn2', None), ('Sn3', None), ('Sn4', None))
    ev = G.bind(ws, doc, 'Sn1', 'lineage', 'training/good', claim)
    assert ev['path'] == os.path.join('training', 'good', 'model.onnx')
    assert ev['data']['kind'] == 'version' and set(ev['data']['sha256']) == {
        ver, os.path.join('datasets', 'ds1', 'splits', 'v001.parquet')}
    G.bind(ws, doc, 'Sn2', 'lineage', 'training/leaky', claim)
    G.bind(ws, doc, 'Sn3', 'lineage', 'training/other', claim)
    G.bind(ws, doc, 'Sn4', 'lineage', 'training/good', {'measure': 'data_root'})     # its own data.json
    v = G.verify(ws, doc)['nodes']
    meta = L.onnx_metadata(os.path.join(good, 'model.onnx'))
    assert v['Sn1']['status'] == 'green' and meta['falcon.data_merkle_root'] in v['Sn1']['why']
    assert 'every one a frame of Set v1' in v['Sn1']['why'] and v['Sn1']['claim']['measured'] == meta[
        'falcon.data_merkle_root']
    assert v['Sn2']['status'] == 'red' and "1 of the model's 4 training frames are not in the version's train" in \
        v['Sn2']['why']
    assert v['Sn3']['status'] == 'red' and "names its data as 'Another v4" in v['Sn3']['why']
    assert v['Sn4']['status'] == 'green' and "training/good/data.json's training samples" in v['Sn4']['why']
    # the model changes (another root in its metadata): amber; re-bound, the roots disagree: red
    L.set_onnx_metadata(os.path.join(good, 'model.onnx'), {'falcon.data_merkle_root': 's256:' + '0' * 64})
    r = G.verify(ws, doc)['nodes']['Sn1']
    assert r['status'] == 'amber' and 'changed since it was bound' in r['why']
    G.bind(ws, doc, 'Sn1', 'lineage', 'training/good', keep_claim=True)
    G.bind(ws, doc, 'Sn4', 'lineage', 'training/good', keep_claim=True)
    v = G.verify(ws, doc)['nodes']
    assert v['Sn1']['status'] == 'red' and 'ONNX falcon.data_merkle_root s256:000' in v['Sn1']['why']
    assert v['Sn4']['status'] == 'red' and v['G1']['status'] == 'red'
    # the data changes: amber until re-bound
    G.bind(ws, doc, 'Sn2', 'lineage', 'training/good', claim)                         # (a model that holds)
    vp = os.path.join(ws, ver)
    vd = json.load(open(vp))
    vd['note'] = 'edited'
    json.dump(vd, open(vp, 'w'))
    r = G.verify(ws, doc)['nodes']['Sn2']
    assert r['status'] == 'amber' and f'the data changed since it was bound ({ver}' in r['why']
    # the command binds one, the template uses them
    assert G.main_gsn(['new', '--workspace', ws, '--template', 'sotif']) == 0
    sotif = G.load(ws)
    sols = {n['evidence']['ref']: n['evidence']['claim'] for n in sotif['nodes'] if (n.get('evidence') or {})
            .get('kind') == 'lineage'}
    assert sols['training/good'] == {'measure': 'data_root', 'data': ver}                 # the version it names
    assert sols['training/other'] == {'measure': 'data_root', 'data': 'training/other/data.json'}
    assert any(n['id'] == 'G7' for n in sotif['nodes']) and G.validate(sotif) == []
    sn = next(n['id'] for n in sotif['nodes'] if (n.get('evidence') or {}).get('ref') == 'training/leaky')
    assert G.main_gsn(['bind', '--workspace', ws, sn, '--measure', 'data_root', '--data', ver]) == 0
    assert G.load(ws)['nodes'][[n['id'] for n in G.load(ws)['nodes']].index(sn)]['evidence']['claim'] == claim
    assert G.main_gsn(['verify', '--workspace', ws, '--gate']) == 1                       # leaky: red
    with pytest.raises(G.GSNError, match='data_root: on lineage evidence'):
        G.bind(ws, doc, 'Sn1', 'failrate', 'training/good', claim)
