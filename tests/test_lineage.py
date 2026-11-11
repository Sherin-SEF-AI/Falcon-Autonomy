"""Content-addressed lineage of runs and dataset versions (core.lineage) on synthetic run folders: the index (one row
per recorded step and sensor; run.json never touched; unchanged files not read again), a dataset version's splits file
with every file put into the store, materialising it by hard links after its runs are gone, verifying it, gc keeping
what a version refers to, the catalogue's dataset row, the index job after a run, the command, and the path without
pyarrow. Models' lineage is tested with a real (tiny) training in test_training."""
import json
import os
import shutil
import sys
import time

import pytest

from falcon_autonomy.core import cas as C
from falcon_autonomy.core import datasets as D
from falcon_autonomy.core import lineage as L
from falcon_autonomy.core import protocol as P

pytest.importorskip('pyarrow')


@pytest.fixture(autouse=True)
def sha256_only(monkeypatch):
    """The reference machine has no blake3: the SHA-256 path."""
    monkeypatch.setitem(sys.modules, 'blake3', None)


def make_run(root, name, steps=4, seed=0):
    """A recorded run in falcon-raw's layout: a camera (rgb, objects, semantic), a lidar (sweep + labels) and a step
    that was not recorded."""
    d = os.path.join(root, 'runs', name)
    os.makedirs(d)
    recs = []
    for i in range(steps):
        files = {}
        for kind, ext, data in (('rgb', 'jpg', f'jpeg {name} {i} {seed}'), ('objects', 'json', f'[{{"n": {i}}}]'),
                                ('semantic', 'png', f'png {i}')):
            rel = os.path.join(kind, 'cam', f'{i:06d}.{ext}')
            os.makedirs(os.path.join(d, os.path.dirname(rel)), exist_ok=True)
            with open(os.path.join(d, rel), 'w') as fh:
                fh.write(data)
            files[f'cam/{kind}'] = rel
        rel = os.path.join('lidar', 'top', f'{i:06d}.bin')
        os.makedirs(os.path.join(d, 'lidar', 'top'), exist_ok=True)
        with open(os.path.join(d, rel), 'wb') as fh:
            fh.write(bytes([i, seed]) * 16)
        with open(os.path.join(d, rel[:-4] + '.labels.bin'), 'wb') as fh:
            fh.write(bytes([7]) * 8)
        files['top/lidar'] = rel
        recs.append({'step': i, 'sim_t': i * 0.05, 'files': files})
    recs.append({'step': steps, 'sim_t': steps * 0.05, 'recorded': False, 'files': {}})
    with open(os.path.join(d, 'frames.jsonl'), 'w') as fh:
        fh.write(''.join(json.dumps(r) + '\n' for r in recs))
    with open(os.path.join(d, 'run.json'), 'w') as fh:
        json.dump({'schema': 'falcon.run/1', 'status': 'ok', 'worker': 'drive'}, fh)
    return d


def test_index_a_run_without_touching_run_json(tmp_path):
    d = make_run(str(tmp_path), 'r1')
    before = open(os.path.join(d, 'run.json'), 'rb').read()
    s = L.index_run(d)
    assert open(os.path.join(d, 'run.json'), 'rb').read() == before
    assert s['rows'] == 8 and s['frames'] == 8 and s['hash_alg'] == 's256' and s['files_read'] == 20
    rows, meta = L.read_index(d)
    assert meta == s and meta['schema'] == L.INDEX_SCHEMA
    cam = next(r for r in rows if r['sensor'] == 'cam' and r['step'] == 2)
    assert cam['frame'] == C.hash_file(os.path.join(d, 'rgb', 'cam', '000002.jpg'))
    labels = json.loads(cam['labels'])
    assert sorted(labels) == ['objects', 'semantic']
    assert cam['label'] == C.hash_bytes(json.dumps({k: v['addr'] for k, v in sorted(labels.items())},
                                                   sort_keys=True, separators=(',', ':')).encode())
    top = next(r for r in rows if r['sensor'] == 'top' and r['step'] == 0)
    assert list(json.loads(top['labels'])) == ['lidar_labels']
    assert meta['merkle_root'] == L.pairs_root([(r['frame'], r['label']) for r in rows], 's256')
    again = L.index_run(d)                                     # nothing changed: nothing read
    assert again['files_read'] == 0 and again['merkle_root'] == s['merkle_root']
    time.sleep(0.01)
    with open(os.path.join(d, 'objects', 'cam', '000001.json'), 'w') as fh:
        fh.write('[{"n": 99}]')                                # a relabelled frame
    third = L.index_run(d)
    assert third['files_read'] == 1 and third['merkle_root'] != s['merkle_root']
    assert L.verify_run(d)['ok']
    with open(os.path.join(d, 'semantic', 'cam', '000003.png'), 'w') as fh:
        fh.write('changed after the index')
    assert not L.verify_run(d)['ok']
    with pytest.raises(L.LineageError, match='frames.jsonl'):
        L.index_run(str(tmp_path))


def test_a_version_pins_its_bytes_in_the_store(tmp_path):
    ws = str(tmp_path / 'ws')
    a, b, c = (make_run(ws, n, seed=k) for k, n in enumerate(('a', 'b', 'c')))
    spec = {'id': 'ds1', 'name': 'Set', 'seed': 0}
    splits = {'train': ['runs/a', 'runs/b'], 'test': ['runs/c']}
    v = D.make_version(spec, [a, b, c], splits, 'drive', 1, ws)
    lin = v['lineage']
    assert lin['splits'] == os.path.join('datasets', 'ds1', 'splits', 'v001.parquet')
    assert lin['rows'] == 24 and lin['hash_alg'] == 's256' and set(lin['roots']) == {'train', 'test'}
    # stored once per distinct bytes: 12 images and 12 sweeps (each run's own), the 4 steps' objects and semantic
    # files and their 4 label sets (alike in the three runs), one lidar labels file and its one label set
    assert lin['objects'] == 12 + 12 + 4 + 4 + 4 + 1 + 1
    assert set(v['provenance']['store']) <= {'hardlink', 'reflink', 'present'}
    assert v['provenance']['store'].get('hardlink', 0) + v['provenance']['store'].get('reflink', 0) == 33
    rows, meta = L.read_table(os.path.join(ws, lin['splits']))
    assert meta['merkle_root'] == lin['merkle_root'] == L.pairs_root([(r['frame'], r['label']) for r in rows], 's256')
    assert lin['roots']['test'] == L.read_index(c)[1]['merkle_root']
    path = os.path.join(ws, 'datasets', 'ds1', 'versions', 'v001.json')
    os.makedirs(os.path.dirname(path))
    with open(path, 'w') as fh:
        json.dump(v, fh)
    assert L.verify_version(path)['ok']
    # the runs go; the version's bytes do not
    keep = {os.path.relpath(os.path.join(dp, f), a): open(os.path.join(dp, f), 'rb').read()
            for dp, _, fs in os.walk(a) for f in fs if f.endswith(('.jpg', '.json', '.png', '.bin'))
            and f not in ('run.json',)}
    shutil.rmtree(os.path.join(ws, 'runs'))
    out = str(tmp_path / 'mat')
    m = L.materialise(os.path.join(ws, lin['splits']), out, C.Store(C.store_dir(ws)), 'hardlink')
    assert m['files'] == 3 * 4 * 5 and m['merkle_root'] == lin['merkle_root']
    for rel, data in keep.items():
        p = os.path.join(out, 'train', 'runs', 'a', rel)
        assert open(p, 'rb').read() == data, rel
    assert set(m['methods']) <= {'hardlink', 'present'}
    # gc keeps every object the version refers to, even with its runs gone
    store = C.Store(C.store_dir(ws))
    extra = store.put_bytes(b'nobody refers to me')
    res = store.gc(L.roots(ws), now=time.time() + 2 * C.DEFAULT_GRACE_S)
    assert res['removed'] == [C.parse(extra)[1]] and res['kept'] == lin['objects']
    assert L.verify_version(path)['ok']
    # a corrupted object is found
    p = store.object_path(rows[0]['frame'])
    os.chmod(p, 0o644)
    with open(p, 'wb') as fh:
        fh.write(b'bit rot')
    bad = L.verify_version(path)
    assert not bad['ok'] and 'no longer hash' in bad['problems'][0]
    # the catalogue's row
    from falcon_autonomy.core.catalog import Catalog
    cat = Catalog(str(tmp_path / 'cat.sqlite'))
    row = cat.add_dataset(path)
    assert row['id'] == 'dataset:ds1@v1' and row['summary']['merkle_root'] == lin['merkle_root']
    assert row['summary']['splits'] == {'train': 2, 'test': 1} and row['summary']['samples'] == 24
    cat.close()


def test_diff_between_versions_and_runs(tmp_path):
    ws = str(tmp_path / 'ws')
    a, b = make_run(ws, 'a'), make_run(ws, 'b', seed=1)
    L.index_run(a)
    L.index_run(b)
    d = L.diff(a, a)
    assert d['same'] and d['frames_only_in_a'] == d['frames_only_in_b'] == 0
    d = L.diff(a, b)
    assert not d['same'] and d['frames_only_in_a'] == d['frames_only_in_b'] == 8 and d['frames_in_both'] == 0
    c = os.path.join(ws, 'runs', 'c')
    shutil.copytree(a, c)
    with open(os.path.join(c, 'objects', 'cam', '000002.json'), 'w') as fh:
        fh.write('[]')
    L.index_run(c)
    d = L.diff(a, c)
    assert d['relabelled'] == 1 and d['frames_only_in_a'] == 0 and d['labels_compared']


def test_the_index_job_after_a_run(tmp_path):
    d = make_run(str(tmp_path), 'r')
    spec = L.index_job_spec(d)
    assert spec.kind == 'lineage-index' and spec.argv[-3:] == ['lineage', 'index', d]
    assert not spec.needs_carla and not spec.needs_gpu and spec.memory_gb < 1 and spec.run_dir == ''

    class Q:
        def __init__(self):
            self.jobs = []

        def submit(self, s):
            self.jobs.append(s)
            return s

    class Svc:
        queue = Q()

    class Job:
        def __init__(self, kind, code, run_dir):
            self.spec = type('S', (), {'kind': kind, 'run_dir': run_dir})()
            self.exit_code = code
    svc = Svc()
    assert L.after_job(svc, Job('drive', P.OK, d)) is not None
    assert L.after_job(svc, Job('drive', P.FAILED, d)) is None
    assert L.after_job(svc, Job('lineage-index', P.OK, d)) is None        # never indexes its own job
    assert L.after_job(svc, Job('train', P.OK, d)) is None
    assert L.after_job(svc, Job('drive', P.OK, str(tmp_path))) is None    # no frames recorded
    assert len(svc.queue.jobs) == 1


def test_command(tmp_path, capsys):
    ws = str(tmp_path / 'ws')
    a = make_run(ws, 'a')
    assert L.main_lineage(['index', a]) == P.OK
    out = json.loads(capsys.readouterr().out.strip())
    assert out['rows'] == 8 and out['merkle_root'].startswith('s256:')
    assert L.main_lineage(['verify', a]) == P.OK
    capsys.readouterr()
    spec = {'id': 'ds', 'name': 'S', 'seed': 0}
    v = D.make_version(spec, [a], {'train': ['runs/a']}, 'drive', 1, ws)
    vp = os.path.join(ws, 'datasets', 'ds', 'versions', 'v001.json')
    os.makedirs(os.path.dirname(vp))
    with open(vp, 'w') as fh:
        json.dump(v, fh)
    assert L.main_lineage(['verify', vp]) == P.OK
    assert L.main_lineage(['materialise', vp, str(tmp_path / 'm'), '--copy']) == P.OK
    assert os.access(os.path.join(str(tmp_path / 'm'), 'train', 'runs', 'a', 'rgb', 'cam', '000000.jpg'), os.W_OK)
    assert L.main_lineage(['diff', vp, a]) == P.OK
    capsys.readouterr()
    assert L.main_lineage(['gc', ws, '--dry-run']) == P.OK
    assert json.loads(capsys.readouterr().out)['removed'] == 0
    assert L.main_lineage(['verify', str(tmp_path / 'nothing')]) == P.FAILED
    assert L.main_lineage(['prove', str(tmp_path), 'x']) == P.FAILED
    assert L.main_lineage(['--help']) == P.OK


def test_without_pyarrow_a_version_says_so(tmp_path, monkeypatch):
    def missing():
        raise L.LineageError('lineage indexes need pyarrow')
    monkeypatch.setattr(L, '_pa', missing)
    ws = str(tmp_path / 'ws')
    a = make_run(ws, 'a')
    v = D.make_version({'id': 'ds', 'name': 'S', 'seed': 0}, [a], {'train': ['runs/a']}, 'drive', 1, ws)
    assert v['lineage'] is None and 'pyarrow' in v['provenance']['notes'][0] and v['sha256']
    assert L.after_job(None, type('J', (), {'exit_code': P.OK, 'spec': type('S', (), {
        'kind': 'drive', 'run_dir': a})()})()) is None


def test_catalogue_rebuild_has_dataset_and_model_rows_linked(tmp_path):
    from falcon_autonomy.core.catalog import Catalog
    ws = str(tmp_path / 'ws')
    a = make_run(ws, 'a')
    v = D.make_version({'id': 'ds', 'name': 'Set', 'seed': 0}, [a], {'train': ['runs/a']}, 'drive', 2, ws)
    vp = os.path.join(ws, 'datasets', 'ds', 'versions', 'v002.json')
    os.makedirs(os.path.dirname(vp))
    with open(vp, 'w') as fh:
        json.dump(v, fh)
    td = os.path.join(ws, 'training', 'm1')
    os.makedirs(td)
    with open(os.path.join(td, 'model.json'), 'w') as fh:
        json.dump({'id': 'm1', 'name': 'Model 1', 'source': {'sha256': 'f' * 64}}, fh)
    with open(os.path.join(td, 'data.json'), 'w') as fh:
        json.dump({'hash_alg': 's256', 'manifest_sha': 'e' * 64,
                   'train': {'merkle_root': 's256:' + 'a' * 64, 'n_images': 4},
                   'dataset_version': f'Set v2 (ds sha256 {v["sha256"][:12]})'}, fh)
    c = Catalog(os.path.join(ws, 'catalog.sqlite'))
    c.rebuild(ws)
    ds, m = c.get('dataset:ds@v2'), c.get('model:m1')
    assert ds['summary']['merkle_root'] == v['lineage']['merkle_root'] and ds['name'] == 'Set v2'
    assert m['summary']['merkle_root'] == 's256:' + 'a' * 64 and m['summary']['images'] == 4
    assert {(x['src'], x['dst'], x['role']) for x in c.links('model:m1')} == {('model:m1', 'dataset:ds@v2',
                                                                              'trained_on')}
    c.rebuild(ws)                                      # again: the same rows, nothing doubled
    assert len(c.list('dataset')) == 1 and len(c.list('model')) == 1
    c.close()
