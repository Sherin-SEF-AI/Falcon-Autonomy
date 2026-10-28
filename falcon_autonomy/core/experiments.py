"""Experiments: local tracking, an MLflow-compatible export, and the mixed sim/real training experiment (standard
library + numpy at import; torch only through core.training, inside the functions).

Tracking. An experiment is a folder experiments/<id>/:

    experiment.json          the definition (and its SHA-256: an experiment's definition never changes under it)
    runs/<run>/run.json      status (running | ok | failed | cancelled), start and end, provenance
    runs/<run>/params.json   {name: value}
    runs/<run>/metrics.jsonl one line per value: {"key", "value", "step", "timestamp" (ms since the epoch)}
    runs/<run>/tags.json     {name: text}
    runs/<run>/artifacts.json  [{"path" (relative to the run when inside it), "sha256", "bytes"}]
    results.json             a mixed experiment's analysis

Experiment(path).runs() lists the runs with their latest metrics; best(metric) picks the finished run with the best
latest value.

MLflow export (export_mlflow) writes MLflow's file-store layout WITHOUT importing mlflow: the layout of
mlflow.store.tracking.file_store.FileStore in MLflow 2.x (as its source reads and writes it, 2.0–2.17):

    mlruns/<experiment_id>/meta.yaml          artifact_location, creation_time, experiment_id, last_update_time,
                                              lifecycle_stage, name
    mlruns/<experiment_id>/<run_id>/meta.yaml artifact_uri, end_time, entry_point_name, experiment_id,
                                              lifecycle_stage, run_id, run_name, run_uuid, source_name, source_type
                                              (4: LOCAL), source_version, start_time, status (3 FINISHED, 4 FAILED,
                                              5 KILLED, 1 RUNNING), tags: [], user_id
    mlruns/<experiment_id>/<run_id>/params/<name>    the value
    mlruns/<experiment_id>/<run_id>/metrics/<name>   lines "timestamp value step"
    mlruns/<experiment_id>/<run_id>/tags/<name>      the value (mlflow.runName, mlflow.user, mlflow.source.*)
    mlruns/<experiment_id>/<run_id>/artifacts/       copies of the run's artifacts

so that `mlflow ui --backend-store-uri file:///…/mlruns` reads it. The ids are stable (from SHA-256 of the
experiment's and run's names), so an export can be repeated into the same folder. Names MLflow would refuse
(characters other than letters, digits, _ - . space /) have those characters replaced by _; parameter values are cut
at 6000 characters and tag values at 5000 (MLflow 2.x's limits). Checked structurally by the tests; mlflow itself is
not a dependency and was not run against it.

The mixed experiment (run_mixed): real only against real + X % synthetic, at a FIXED training budget. Arms are the
synthetic shares (default 0, 25, 50, 100 %) of the same number N of training images, trained with the same settings
for the same number of iterations (epochs × ⌈N / batch⌉; early stopping is off so no arm trains less), K seeds per
arm. Per seed the real and the synthetic images are drawn by seeded permutations shared by every arm (an arm with more
synthetic images swaps the tail of the real list for the head of the synthetic one), so arms differ only in the share.
The model kept is the one with the best AP on a REAL validation set (given, or a val_share of the real training pool
carved off by sequence/image and kept out of every arm) — or the last epoch with select "last". Every arm and seed is
then scored on the SAME held-out REAL test set, which must share no image and no sequence with the training or
validation data (refused otherwise).

Honest intervals. Per arm, AP (and AP50, AP75, AR100) is the mean over its seeds. Its 95 % interval is a bootstrap over
the test set's units, resampling them jointly for all seeds and arms. The unit is the image, unless the set says which
sequence an image comes from (KITTI's object images through its devkit mapping to raw drives, nuScenes' logs, a folder
per sequence, or a map given): then the whole sequence is resampled (a cluster bootstrap, as the harness does with
drives), because frames of one sequence are not independent. Each arm is compared with the real-only arm (share 0) by
a paired bootstrap of the difference on the same resamples. These intervals cover the test sample only, not training
noise: so the seed-to-seed spread (the per-seed numbers, their standard deviation and standard error) is reported
beside them, and the verdict uses a combined interval, the difference ± 1.96 √(sd_boot² + se_arm² + se_real²)
(a normal approximation that treats the test-sample and seed noise as independent). With one seed the training noise
is not measured, and the verdict says so. A verdict is "better" or "worse" only when the interval excludes zero;
otherwise "no detectable difference at 95 %" — not "no difference".

Resumable: an arm and seed whose run finished is skipped; a cancelled one resumes inside its training (core.training
checkpoints); the analysis is recomputed only when a run's predictions changed.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import time

import numpy as np

SCHEMA = 'falcon.experiment/1'
RUN_SCHEMA = 'falcon.experiment_run/1'
RESULTS_SCHEMA = 'falcon.mixed_results/1'
MLFLOW_LAYOUT = 'MLflow 2.x FileStore (mlflow/store/tracking/file_store.py)'
MLFLOW_STATUS = {'running': 1, 'ok': 3, 'failed': 4, 'cancelled': 5}       # RunStatus: RUNNING FINISHED FAILED KILLED
MLFLOW_SOURCE_LOCAL = 4
MLFLOW_PARAM_MAX, MLFLOW_TAG_MAX = 6000, 5000
HEADLINE = ('AP', 'AP50', 'AP75', 'AR100')
MIN_UNITS = 5
MIXED_DEFAULTS = {
    'kind': 'mixed', 'name': None, 'synthetic': [], 'real_train': [], 'real_val': [], 'real_test': [],
    'shares': [0.0, 0.25, 0.5, 1.0], 'budget_images': None, 'seeds': 3, 'seed': 0, 'val_share': 0.15,
    'n_boot': 1000, 'boot_seed': 0, 'train': {},
}


class ExperimentError(Exception):
    pass


def _ms() -> int:
    return int(time.time() * 1000)


def _safe(name: str) -> str:
    s = re.sub(r'[^A-Za-z0-9_.-]', '_', str(name)).strip('.')
    if not s:
        raise ExperimentError(f'{name!r} is not a usable name')
    return s


# ================================================================================================ tracking
class Experiment:
    """An experiment folder: experiment.json and runs/<run>/."""

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        p = os.path.join(self.path, 'experiment.json')
        if not os.path.isfile(p):
            raise ExperimentError(f'{self.path} is not an experiment (no experiment.json)')
        with open(p, encoding='utf-8') as fh:
            self.doc = json.load(fh)

    @classmethod
    def create(cls, root: str, exp_id: str, definition: dict, name: str | None = None) -> 'Experiment':
        """A new experiment, or the existing one with exactly this definition (else refused)."""
        from . import provenance as PV
        path = os.path.join(os.path.abspath(root), _safe(exp_id))
        sha = PV.sha256_json(definition)
        p = os.path.join(path, 'experiment.json')
        if os.path.isfile(p):
            old = PV.read_json(p)
            if old.get('definition_sha256') != sha:
                raise ExperimentError(f'{path} holds an experiment with another definition: choose another id')
            return cls(path)
        os.makedirs(os.path.join(path, 'runs'), exist_ok=True)
        PV.write_json(p, {'schema': SCHEMA, 'id': _safe(exp_id), 'name': name or exp_id,
                          'kind': definition.get('kind', 'generic'), 'created': PV.utc_now(),
                          'created_ms': _ms(), 'definition': definition, 'definition_sha256': sha,
                          'provenance': PV.block(seed=definition.get('seed'), settings={'kind': definition.get(
                              'kind', 'generic')})})
        return cls(path)

    @property
    def id(self) -> str:
        return self.doc['id']

    def run(self, name: str) -> 'TrackedRun':
        return TrackedRun(self, name)

    def run_names(self) -> list:
        d = os.path.join(self.path, 'runs')
        return sorted(n for n in os.listdir(d) if os.path.isdir(os.path.join(d, n))) if os.path.isdir(d) else []

    def runs(self, status: str | None = None) -> list:
        """[{'name', 'status', 'params', 'tags', 'metrics': {key: value at the latest step}, 'started',
        'finished'}], by name; status filters."""
        out = []
        for n in self.run_names():
            r = self.run(n)
            info = r.info()
            if status and info.get('status') != status:
                continue
            out.append({'name': n, 'status': info.get('status'), 'started': info.get('started'),
                        'finished': info.get('finished'), 'params': r.params(), 'tags': r.tags(),
                        'metrics': r.latest()})
        return out

    def best(self, metric: str, mode: str = 'max', tags: dict | None = None) -> dict | None:
        """The finished run with the best latest value of metric (mode max or min), optionally among runs whose tags
        match; None when no run has it."""
        if mode not in ('max', 'min'):
            raise ExperimentError('mode is max or min')
        rows = [r for r in self.runs('ok') if isinstance(r['metrics'].get(metric), (int, float))
                and math.isfinite(r['metrics'][metric])
                and all(r['tags'].get(k) == str(v) for k, v in (tags or {}).items())]
        if not rows:
            return None
        return (max if mode == 'max' else min)(rows, key=lambda r: r['metrics'][metric])


class TrackedRun:
    """runs/<name>/ of an experiment: params, metrics with steps, tags, artifacts and run.json."""

    def __init__(self, exp: Experiment, name: str):
        self.exp = exp
        self.name = _safe(name)
        self.dir = os.path.join(exp.path, 'runs', self.name)

    def _p(self, f):
        return os.path.join(self.dir, f)

    def _read(self, f, default):
        try:
            with open(self._p(f), encoding='utf-8') as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return default

    def info(self) -> dict:
        return self._read('run.json', {})

    @property
    def status(self):
        return self.info().get('status')

    def start(self, params: dict | None = None, tags: dict | None = None, seed=None, inputs: dict | None = None):
        """(Re)start the run: params and tags set, metrics and artifacts cleared, run.json status running."""
        from . import provenance as PV
        os.makedirs(self.dir, exist_ok=True)
        for f in ('metrics.jsonl', 'artifacts.json'):
            if os.path.isfile(self._p(f)):
                os.remove(self._p(f))
        PV.write_json(self._p('params.json'), dict(params or {}))
        PV.write_json(self._p('tags.json'), {str(k): str(v) for k, v in (tags or {}).items()})
        PV.write_json(self._p('run.json'), {'schema': RUN_SCHEMA, 'name': self.name, 'status': 'running',
                                            'started': PV.utc_now(), 'started_ms': _ms(), 'finished': None,
                                            'finished_ms': None, 'error': None,
                                            'provenance': PV.block(inputs, seed, {'params': params or {}})})
        return self

    def log_param(self, key: str, value):
        from . import provenance as PV
        p = self.params()
        p[str(key)] = value
        PV.write_json(self._p('params.json'), p)

    def set_tag(self, key: str, value):
        from . import provenance as PV
        t = self.tags()
        t[str(key)] = str(value)
        PV.write_json(self._p('tags.json'), t)

    def log_metric(self, key: str, value, step: int = 0, timestamp: int | None = None):
        if value is None:
            return
        v = float(value)
        with open(self._p('metrics.jsonl'), 'a', encoding='utf-8') as fh:
            fh.write(json.dumps({'key': str(key), 'value': v if math.isfinite(v) else None, 'step': int(step),
                                 'timestamp': int(timestamp if timestamp is not None else _ms())},
                                separators=(',', ':')) + '\n')

    def log_metrics(self, values: dict, step: int = 0, timestamp: int | None = None):
        ts = timestamp if timestamp is not None else _ms()
        for k, v in values.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                self.log_metric(k, v, step, ts)

    def log_artifact(self, path: str):
        from . import provenance as PV
        if not os.path.isfile(path):
            raise ExperimentError(f'no artifact {path}')
        a = os.path.abspath(path)
        rel = os.path.relpath(a, self.dir)
        arts = [x for x in self.artifacts() if x['path'] != (a if rel.startswith('..') else rel)]
        arts.append({'path': a if rel.startswith('..') else rel, 'sha256': PV.sha256_file(a),
                     'bytes': os.path.getsize(a)})
        PV.write_json(self._p('artifacts.json'), arts)

    def finish(self, status: str = 'ok', error: str | None = None, **extra):
        from . import provenance as PV
        if status not in MLFLOW_STATUS:
            raise ExperimentError(f'status is one of {", ".join(MLFLOW_STATUS)}')
        doc = self.info()
        doc.update(status=status, error=error, finished=PV.utc_now(), finished_ms=_ms(), **extra)
        PV.write_json(self._p('run.json'), doc)

    def params(self) -> dict:
        return self._read('params.json', {})

    def tags(self) -> dict:
        return self._read('tags.json', {})

    def artifacts(self) -> list:
        return self._read('artifacts.json', [])

    def artifact_path(self, a: dict) -> str:
        return a['path'] if os.path.isabs(a['path']) else os.path.join(self.dir, a['path'])

    def metrics(self) -> list:
        try:
            with open(self._p('metrics.jsonl'), encoding='utf-8') as fh:
                return [json.loads(x) for x in fh if x.strip()]
        except OSError:
            return []

    def latest(self) -> dict:
        """{key: value at its largest step (the last logged among equal steps)}."""
        out, at = {}, {}
        for m in self.metrics():
            if m['key'] not in at or m['step'] >= at[m['key']]:
                at[m['key']], out[m['key']] = m['step'], m['value']
        return out


# ================================================================================================ MLflow export
def _yaml(d: dict) -> str:
    """A flat dict as YAML (strings single-quoted, as MLflow's own meta files are)."""
    lines = []
    for k in sorted(d):
        v = d[k]
        if isinstance(v, bool):
            s = 'true' if v else 'false'
        elif isinstance(v, int):
            s = str(v)
        elif isinstance(v, list):
            s = '[]' if not v else '[' + ', '.join(_yaml_str(x) for x in v) + ']'
        elif v is None:
            s = 'null'
        else:
            s = _yaml_str(v)
        lines.append(f'{k}: {s}')
    return '\n'.join(lines) + '\n'


def _yaml_str(v) -> str:
    return "'" + str(v).replace("'", "''") + "'"


def mlflow_name(name: str) -> str:
    """A metric, parameter or tag name MLflow accepts (letters, digits, _ - . space /; no leading /, no '..')."""
    s = re.sub(r'[^A-Za-z0-9_\-. /]', '_', str(name)).lstrip('/')
    s = s.replace('..', '__')
    return s or '_'


def _mlflow_id(*parts, n=32) -> str:
    from .provenance import sha256_json
    return sha256_json(list(parts))[:n]


def _flat(d: dict, prefix='') -> dict:
    out = {}
    for k, v in (d or {}).items():
        key = f'{prefix}{k}'
        if isinstance(v, dict) and v:
            out.update(_flat(v, key + '.'))
        else:
            out[key] = v if isinstance(v, str) else json.dumps(v, default=str, separators=(',', ':'))
    return out


def export_mlflow(exp_path: str, mlruns: str, copy_artifacts: bool = True, user: str | None = None) -> dict:
    """Write an experiment in MLflow's file-store layout under mlruns/ (see the module docstring). Returns
    {'experiment_id', 'path', 'runs': {name: run_id}, 'layout'}."""
    exp = Experiment(exp_path)
    root = os.path.abspath(mlruns)
    eid = str(int(_mlflow_id(exp.id, n=15), 16))                    # numeric, as MLflow's own ids
    edir = os.path.join(root, eid)
    os.makedirs(edir, exist_ok=True)
    created = int(exp.doc.get('created_ms') or _ms())
    user = user or os.environ.get('USER') or 'falcon'
    runs = {}
    last = created
    for r in exp.runs():
        tr = exp.run(r['name'])
        info = tr.info()
        rid = _mlflow_id(exp.id, r['name'])
        rdir = os.path.join(edir, rid)
        if os.path.isdir(rdir):
            shutil.rmtree(rdir)                                      # a re-export replaces the run
        for sub in ('params', 'metrics', 'tags', 'artifacts'):
            os.makedirs(os.path.join(rdir, sub), exist_ok=True)
        start = int(info.get('started_ms') or created)
        end = info.get('finished_ms')
        last = max(last, int(end or start))
        meta = {'artifact_uri': 'file://' + os.path.join(rdir, 'artifacts'), 'end_time': int(end) if end else None,
                'entry_point_name': '', 'experiment_id': eid, 'lifecycle_stage': 'active', 'run_id': rid,
                'run_name': r['name'], 'run_uuid': rid, 'source_name': '', 'source_type': MLFLOW_SOURCE_LOCAL,
                'source_version': '', 'start_time': start, 'status': MLFLOW_STATUS.get(r['status'] or 'running', 1),
                'tags': [], 'user_id': user}
        with open(os.path.join(rdir, 'meta.yaml'), 'w', encoding='utf-8') as fh:
            fh.write(_yaml(meta))
        for k, v in _flat(tr.params()).items():
            _write_value(os.path.join(rdir, 'params'), mlflow_name(k), v[:MLFLOW_PARAM_MAX])
        tags = {**tr.tags(), 'mlflow.runName': r['name'], 'mlflow.user': user, 'mlflow.source.type': 'LOCAL',
                'mlflow.source.name': 'falcon_autonomy', 'falcon.experiment': exp.id,
                'falcon.status': r['status'] or 'running'}
        for k, v in tags.items():
            _write_value(os.path.join(rdir, 'tags'), mlflow_name(k), str(v)[:MLFLOW_TAG_MAX])
        series = {}
        for m in tr.metrics():
            series.setdefault(mlflow_name(m['key']), []).append(m)
        for k, ms in series.items():
            p = os.path.join(rdir, 'metrics', k)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, 'w', encoding='utf-8') as fh:
                for m in ms:
                    v = m['value']
                    fh.write(f'{int(m["timestamp"])} {repr(float(v)) if v is not None else "nan"} {int(m["step"])}'
                             '\n')
        if copy_artifacts:
            for a in tr.artifacts():
                src = tr.artifact_path(a)
                if not os.path.isfile(src):
                    continue
                rel = a['path'] if not os.path.isabs(a['path']) else os.path.basename(a['path'])
                dst = os.path.join(rdir, 'artifacts', rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
        runs[r['name']] = rid
    with open(os.path.join(edir, 'meta.yaml'), 'w', encoding='utf-8') as fh:
        fh.write(_yaml({'artifact_location': 'file://' + edir, 'creation_time': created, 'experiment_id': eid,
                        'last_update_time': last, 'lifecycle_stage': 'active', 'name': exp.doc.get('name') or exp.id}))
    return {'experiment_id': eid, 'path': edir, 'runs': runs, 'layout': MLFLOW_LAYOUT}


def _write_value(d: str, name: str, value: str):
    p = os.path.join(d, name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', encoding='utf-8') as fh:
        fh.write(value)


# ================================================================================================ the mixed experiment
def normalise_mixed(d: dict) -> dict:
    from . import training as T
    m = {**MIXED_DEFAULTS, **{k: v for k, v in (d or {}).items() if v is not None}}
    sh = sorted({round(float(x) / 100.0 if float(x) > 1 else float(x), 6) for x in m['shares']})
    if not sh or sh[0] != 0.0:
        raise ExperimentError('the shares must include 0 (the real-only arm every other arm is compared with)')
    if sh[-1] > 1.0:
        raise ExperimentError('a share is at most 100 %')
    m['shares'] = sh
    if int(m['seeds']) < 1:
        raise ExperimentError('seeds must be at least 1')
    for k in ('real_train', 'real_test'):
        if not m[k]:
            raise ExperimentError(f'{k} needs at least one source')
    if sh[-1] > 0 and not m['synthetic']:
        raise ExperimentError('synthetic needs at least one source (a share above 0 is asked for)')
    if not 0 <= float(m['val_share']) < 1:
        raise ExperimentError('val_share must lie in [0, 1)')
    srcs = lambda xs: [T.parse_source(x) if isinstance(x, str) else dict(x) for x in xs]     # noqa: E731
    for k in ('synthetic', 'real_train', 'real_val', 'real_test'):
        m[k] = srcs(m[k])
    for k in ('real_train', 'real_val', 'real_test'):
        bad = [x['kind'] for x in m[k] if x['kind'] not in T.REAL_KINDS]
        if bad:
            raise ExperimentError(f'{k} must be real sets ({", ".join(T.REAL_KINDS)}); got {", ".join(bad)}')
    tr = dict(m['train'])
    tr.update(train=[], val=[], test=[], patience=0, keep_last=False)
    m['train'] = T.normalise(tr)
    return m


def arm_name(share: float) -> str:
    return f'{round(share * 100):g}%'


def run_name(share: float, seed: int) -> str:
    return f'share{round(share * 100):03d}-seed{seed}'


def _pools(m: dict, em):
    """The synthetic pool, the real training pool, the real validation set and the real test set, checked for
    leakage (no image and no sequence of the test set in training or validation)."""
    from . import training as T
    from .datasets import split
    classes = m['train']['classes']
    syn, n1, i1 = T.load_sources(m['synthetic'], classes)
    real, n2, i2 = T.load_sources(m['real_train'], classes)
    test, n3, i3 = T.load_sources(m['real_test'], classes)
    val, n4, i4 = T.load_sources(m['real_val'], classes)
    notes, ins = n1 + n2 + n3 + n4, {**i1, **i2, **i3, **i4}
    if not val and float(m['val_share']) > 0:
        units = sorted({s['unit'] for s in real})
        w = {}
        for s in real:
            w[s['unit']] = w.get(s['unit'], 0) + 1
        sp = split(units, {'train': 1.0 - float(m['val_share']), 'val': float(m['val_share'])}, int(m['seed']), w)
        vu = set(sp['val'])
        val = [s for s in real if s['unit'] in vu]
        real = [s for s in real if s['unit'] not in vu]
        notes.append(f'validation: {len(val)} real images ({len(vu)} units) carved from the real training pool '
                     f'(val_share {m["val_share"]}), kept out of every arm')
    t_ids, t_units = {s['id'] for s in test}, {s['unit'] for s in test}
    for name, pool in (('synthetic', syn), ('real training', real), ('validation', val)):
        ids = t_ids & {s['id'] for s in pool}
        units = t_units & {s['unit'] for s in pool}
        if ids or units:
            raise ExperimentError(f'the test set shares {len(ids)} images and {len(units)} sequences/units with the '
                                  f'{name} data: a held-out test set may share neither')
    return syn, real, val, test, notes, ins


def _subsets(m: dict, n_syn_pool: int, n_real_pool: int):
    """Per seed the seeded orders of both pools; per arm the first n_real real and first n_syn synthetic of them."""
    shares = m['shares']
    N = m['budget_images']
    if N is None:
        N = n_real_pool if shares[-1] == 0 else min(n_real_pool, int(n_syn_pool / shares[-1]))
    N = int(N)
    need_syn = round(shares[-1] * N)
    if N < 1 or n_real_pool < N or n_syn_pool < need_syn:
        raise ExperimentError(f'a budget of {N} images needs {N} real images (the real-only arm) and {need_syn} '
                              f'synthetic ones; the pools hold {n_real_pool} real and {n_syn_pool} synthetic')
    return N


def _perm(seed: int, salt: int, n: int) -> np.ndarray:
    from .training import _mix
    return np.random.default_rng(_mix(seed, salt)).permutation(n)


class _SubEmitter:
    """Maps a run's progress into its slice of the experiment's progress."""

    def __init__(self, em, lo, hi, label):
        self.em, self.lo, self.hi, self.label = em, lo, hi, label

    def progress(self, f, text='', force=False):
        self.em.progress(self.lo + (self.hi - self.lo) * float(f), f'{self.label}: {text}', force=force)

    def log(self, text, level='info'):
        self.em.log(f'{self.label}: {text}', level)

    def metric(self, name, value, unit='', source='measured', **extra):
        self.em.metric(name, value, unit, source, run=self.label, **extra)


def _test_numbers(frames: list) -> dict:
    from . import metrics as M
    if not frames:
        return {}
    r = M.coco_evaluate(frames)
    return {k: r.get(k) for k in HEADLINE}


def run_mixed(root: str, exp_id: str, definition: dict, emitter=None, cancel=None, workspace: str = '',
              weights_dirs_extra=()) -> dict:
    """Run (or resume) the mixed sim/real experiment experiments/<exp_id>/ and analyse it: returns results.json's
    document. Raises ExperimentError, training.TrainingError or training.Cancelled."""
    from . import provenance as PV
    from . import training as T
    em = emitter or T._Quiet()
    cancel = cancel or (lambda: False)
    m = normalise_mixed(definition)
    fixed = dict(m, train={k: v for k, v in m['train'].items() if k not in T.VOLATILE})   # what the results depend on
    exp = Experiment.create(root, exp_id, fixed, m.get('name'))
    syn, real, val, test, notes, ins = _pools(m, em)
    N = _subsets(m, len(syn), len(real))
    seeds = [int(m['seed']) + j for j in range(int(m['seeds']))]
    plan = [(sh, sd) for sd in seeds for sh in m['shares']]
    ipe = -(-N // int(m['train']['batch']))
    em.log(f'{len(m["shares"])} arms × {len(seeds)} seeds; {N} training images and {ipe * m["train"]["epochs"]} '
           f'iterations per run; {len(test)} real test images')
    done = skipped = 0
    for k, (sh, sd) in enumerate(plan):
        name = run_name(sh, sd)
        tr = exp.run(name)
        if tr.status == 'ok' and os.path.isfile(os.path.join(tr.dir, 'training', 'test_predictions.jsonl')):
            skipped += 1
            continue
        if cancel():
            raise T.Cancelled()
        n_syn = round(sh * N)
        n_real = N - n_syn
        rp, sp = _perm(sd, 1, len(real)), _perm(sd, 2, len(syn))
        train_set = [real[i] for i in rp[:n_real]] + [syn[i] for i in sp[:n_syn]]
        s = dict(m['train'], seed=sd)
        params = {'share_synthetic': sh, 'seed': sd, 'n_real': n_real, 'n_synthetic': n_syn, 'budget_images': N,
                  'iterations': ipe * s['epochs'], **{f'train.{kk}': vv for kk, vv in s.items()
                                                      if kk not in ('train', 'val', 'test')}}
        tr.start(params, {'arm': arm_name(sh), 'seed': sd, 'kind': 'mixed', 'base': str(sh == 0.0).lower()}, sd, ins)
        sub = _SubEmitter(em, k / len(plan), (k + 1) / len(plan), name)
        try:
            summ = T.train(s, os.path.join(tr.dir, 'training'),
                           data={'train': train_set, 'val': val, 'test': test, 'inputs': ins,
                                 'notes': notes + [f'arm {arm_name(sh)}, seed {sd}: {n_real} real + {n_syn} '
                                                   f'synthetic images']},
                           workspace=workspace, emitter=sub, cancel=cancel,
                           argv=['experiment', 'mixed', exp.id, name], weights_dirs_extra=weights_dirs_extra)
        except T.Cancelled:
            tr.finish('cancelled')
            raise
        except T.TrainingError as e:
            tr.finish('failed', str(e))
            raise
        _record_run(tr, summ, m)
        tr.finish('ok')
        done += 1
    em.progress(1.0, 'analysing', force=True)
    res = analyse(exp, m, seeds, N, ipe * int(m['train']['epochs']), notes)
    res['runs_trained'], res['runs_skipped'] = done, skipped
    return res


def _record_run(tr: TrackedRun, summ: dict, m: dict):
    """Copy a finished training into the tracked run: per-epoch metrics, test numbers with a per-run bootstrap
    interval, artifacts."""
    from . import metrics as M
    from .training import _read_metrics
    d = os.path.join(tr.dir, 'training')
    for row in _read_metrics(os.path.join(d, 'metrics.jsonl')):
        tr.log_metrics({'train_loss': row.get('train_loss'), 'images_per_s': row.get('images_per_s'),
                        **{f'val.{k}': (row.get('val') or {}).get(k) for k in HEADLINE}}, step=row['epoch'])
    frames = _frames(os.path.join(d, 'test_predictions.jsonl'))
    vals = {f'test.{k}': v for k, v in _test_numbers(frames).items()}
    if frames and any(not g['ignore'] for f in frames for g in f['gt']):
        b = M.bootstrap(lambda its: _test_numbers(its).get('AP'), frames, n=min(int(m['n_boot']), 500),
                        seed=int(m['boot_seed']), unit='drive')
        vals.update({'test.AP.ci_low': b['ci_low'], 'test.AP.ci_high': b['ci_high']})
    tr.log_metrics(vals, step=summ.get('epochs_run') or 0)
    for f in ('run.json', 'metrics.jsonl', 'test_predictions.jsonl', 'model.onnx', 'model.json'):
        if os.path.isfile(os.path.join(d, f)):
            tr.log_artifact(os.path.join(d, f))


def _frames(path: str) -> list:
    if not os.path.isfile(path):
        return []
    with open(path, encoding='utf-8') as fh:
        return [json.loads(x) for x in fh if x.strip()]


def _unit_description(frames: list) -> str:
    seq = [not str(f['drive']).startswith('image:') for f in frames]
    if all(seq):
        return 'sequence'
    if not any(seq):
        return 'image'
    return 'sequence where the set gives one, else image'


def analyse(exp: Experiment, m: dict, seeds: list, N: int, iterations: int, notes=()) -> dict:
    """The arms' AP with bootstrap intervals over the test set's units, the paired comparison of each arm with the
    real-only arm, the seed-to-seed spread and the verdicts (see the module docstring); results.json."""
    from . import metrics as M
    from . import provenance as PV
    arms = list(m['shares'])
    names = {(sh, sd): run_name(sh, sd) for sh in arms for sd in seeds}
    paths = {k: os.path.join(exp.path, 'runs', n, 'training', 'test_predictions.jsonl') for k, n in names.items()}
    missing = [names[k] for k, p in paths.items() if not os.path.isfile(p)]
    if missing:
        raise ExperimentError(f'runs not finished: {", ".join(missing)}')
    key = PV.sha256_json({'preds': {names[k]: PV.sha256_file(p) for k, p in paths.items()},
                          'n_boot': m['n_boot'], 'boot_seed': m['boot_seed']})
    out_p = os.path.join(exp.path, 'results.json')
    if os.path.isfile(out_p):
        old = PV.read_json(out_p)
        if old.get('key') == key:
            return old
    frames = {k: {f['image_id']: f for f in _frames(p)} for k, p in paths.items()}
    ids = sorted(next(iter(frames.values())))
    for k, fr in frames.items():
        if sorted(fr) != ids:
            raise ExperimentError(f'{names[k]} was tested on other images than the others')
    recs = {k: {r['image_id']: r for r in M.coco_match([fr[i] for i in ids])} for k, fr in frames.items()}
    first = frames[next(iter(frames))]
    items = [{'image_id': i, 'drive': first[i]['drive'], 'r': {k: recs[k][i] for k in recs}} for i in ids]
    n_obj = sum(1 for i in ids for g in first[i]['gt'] if not g['ignore'])
    unit_desc = _unit_description(list(first.values()))

    def numbers(its, sh):
        per = []
        for sd in seeds:
            rs = [it['r'][(sh, sd)] for it in its]
            try:
                s = M.coco_summarise(rs)
            except ValueError:
                s = {}
            per.append([s.get(k) for k in HEADLINE])
        out = {}
        for j, k in enumerate(HEADLINE):
            v = [p[j] for p in per if p[j] is not None]
            out[k] = float(np.mean(v)) if len(v) == len(per) else None
        return out

    def all_arms(its):
        return {f'{arm_name(sh)}|{k}': v for sh in arms for k, v in numbers(its, sh).items()}

    n_boot, bseed = int(m['n_boot']), int(m['boot_seed'])
    boot = M.bootstrap(all_arms, items, n=n_boot, seed=bseed, unit='drive')
    base = arms[0]
    rows = []
    K = len(seeds)
    for sh in arms:
        per_seed = []
        for sd in seeds:
            rs = [recs[(sh, sd)][i] for i in ids]
            s = M.coco_summarise(rs)
            per_seed.append({'seed': sd, **{k: s.get(k) for k in HEADLINE}})
        spread = {}
        for k in HEADLINE:
            v = [p[k] for p in per_seed if p[k] is not None]
            sd_ = float(np.std(v, ddof=1)) if len(v) > 1 else None
            spread[k] = {'sd': sd_, 'se': sd_ / math.sqrt(len(v)) if sd_ is not None else None,
                         'min': min(v) if v else None, 'max': max(v) if v else None}
        row = {'arm': arm_name(sh), 'share_synthetic': sh, 'n_real': N - round(sh * N), 'n_synthetic': round(sh * N),
               'runs': [names[(sh, sd)] for sd in seeds],
               **{k: {kk: boot['metrics'][f'{arm_name(sh)}|{k}'][kk] for kk in ('value', 'ci_low', 'ci_high')}
                  for k in HEADLINE},
               'per_seed': per_seed, 'seed_spread': spread, 'vs_real_only': None}
        rows.append(row)
    by = {r['share_synthetic']: r for r in rows}
    for sh in arms[1:]:
        pb = M.paired_bootstrap(lambda its, a: numbers(its, a), items, base, sh, n=n_boot, seed=bseed, unit='drive')
        cmp_ = {}
        for k in HEADLINE:
            r = pb['metrics'][k]
            se_a, se_b = by[sh]['seed_spread'][k]['se'], by[base]['seed_spread'][k]['se']
            comb = None
            if r['diff'] is not None and r['sd'] is not None and se_a is not None and se_b is not None:
                h = 1.96 * math.sqrt(r['sd'] ** 2 + se_a ** 2 + se_b ** 2)
                comb = [r['diff'] - h, r['diff'] + h]
            lo, hi = comb if comb else (r['ci_low'], r['ci_high'])
            if lo is None or hi is None:
                verdict = 'not measurable (nothing to score)'
            elif lo > 0:
                verdict = 'better than real only'
            elif hi < 0:
                verdict = 'worse than real only'
            else:
                verdict = 'no detectable difference at 95 %'
            cmp_[k] = {'diff': r['diff'], 'ci_low': r['ci_low'], 'ci_high': r['ci_high'], 'sd_boot': r['sd'],
                       'p_arm_gt_real_only': r['p_b_gt_a'], 'combined_ci': comb,
                       'verdict': verdict + ('' if comb else ' (one seed: training noise not measured)'
                                             if K == 1 else '')}
        by[sh]['vs_real_only'] = cmp_
    n_units = boot['n_units']
    warn = []
    if n_units < MIN_UNITS:
        warn.append(f'only {n_units} test units: a bootstrap over so few has few distinct resamples, so its '
                    f'intervals are not reliable')
    if K == 1:
        warn.append('one seed per arm: the seed-to-seed spread is not measured, and the intervals leave out training '
                    'noise')
    doc = {'schema': RESULTS_SCHEMA, 'experiment': exp.id, 'key': key, 'base_arm': arm_name(base),
           'budget_images': N, 'iterations_per_run': iterations, 'seeds': seeds, 'classes': m['train']['classes'],
           'test': {'images': len(ids), 'objects': n_obj, 'unit': unit_desc, 'n_units': n_units},
           'n_boot': n_boot, 'boot_seed': bseed, 'arms': rows, 'warnings': warn, 'notes': list(notes),
           'method': {
               'arm_value': 'mean over seeds of COCO AP on the real test set (core.metrics, pycocotools\' numbers)',
               'interval': f'95 % percentile bootstrap over test {unit_desc}s ({n_boot} resamples), all arms and '
                           f'seeds resampled together; covers the test sample, not training noise',
               'paired': 'arm − real only on the same resamples (core.metrics.paired_bootstrap)',
               'combined': 'diff ± 1.96 √(sd_boot² + se_arm² + se_real_only²), se = seed sd / √K (normal '
                           'approximation, independence assumed)',
               'verdict': 'better/worse only when the combined interval (the bootstrap one with one seed) excludes 0',
               'ignore_rule': 'real boxes under core.harness.MIN_PIXELS in area are ignored (core.training)'},
           'provenance': PV.block({n: p for n, p in ((names[k], paths[k]) for k in paths)}, bseed,
                                  {'n_boot': n_boot})}
    PV.write_json(out_p, doc)
    tr = exp.run('analysis')
    tr.start({'n_boot': n_boot, 'boot_seed': bseed, 'budget_images': N, 'seeds': len(seeds)},
             {'kind': 'mixed-analysis'})
    for r in rows:
        a = 'arm' + f'{round(r["share_synthetic"] * 100):03d}'
        for k in HEADLINE:
            tr.log_metrics({f'{a}.{k}': r[k]['value'], f'{a}.{k}.ci_low': r[k]['ci_low'],
                            f'{a}.{k}.ci_high': r[k]['ci_high'], f'{a}.{k}.seed_sd': r['seed_spread'][k]['sd']})
            if r['vs_real_only']:
                c = r['vs_real_only'][k]
                tr.log_metrics({f'{a}.{k}.diff': c['diff'], f'{a}.{k}.diff_ci_low': c['ci_low'],
                                f'{a}.{k}.diff_ci_high': c['ci_high']})
    tr.log_artifact(out_p)
    tr.finish('ok')
    return doc


def table(res: dict) -> str:
    """results.json as a text table."""
    lines = [f'{"arm":>6s} {"real":>6s} {"synth":>6s}  {"AP [95 % CI]":26s} {"seed sd":>8s}  vs real only',
             '-' * 96]
    for r in res['arms']:
        ap = r['AP']
        ci = (f'{ap["value"]:.4f} [{ap["ci_low"]:.4f}, {ap["ci_high"]:.4f}]' if ap['value'] is not None
              and ap['ci_low'] is not None else '—')
        sdv = r['seed_spread']['AP']['sd']
        vs = ''
        if r['vs_real_only']:
            c = r['vs_real_only']['AP']
            if c['diff'] is not None:
                vs = f'{c["diff"]:+.4f} [{c["ci_low"]:+.4f}, {c["ci_high"]:+.4f}] → {c["verdict"]}'
        lines.append(f'{r["arm"]:>6s} {r["n_real"]:6d} {r["n_synthetic"]:6d}  {ci:26s} '
                     f'{sdv if sdv is None else f"{sdv:.4f}"!s:>8s}  {vs}')
    t = res['test']
    lines.append(f'test: {t["images"]} images, {t["objects"]} objects, unit {t["unit"]} ({t["n_units"]}); '
                 f'{len(res["seeds"])} seeds; {res["budget_images"]} images and {res["iterations_per_run"]} '
                 f'iterations per run')
    lines += [f'note: {w}' for w in res.get('warnings') or ()]
    return '\n'.join(lines)


# ================================================================================================ the command
def main_experiment(argv) -> int:
    """`falcon experiment mixed|list|best|results|export …`: protocol lines on stdout; 0 ok, 2 failed, 3 cancelled."""
    import argparse
    from . import protocol as P
    from . import provenance as PV
    from . import training as T
    ap = argparse.ArgumentParser(prog='falcon experiment', description='experiments: the mixed sim/real training '
                                                                       'experiment, run tracking, MLflow export')
    sub = ap.add_subparsers(dest='action', required=True)
    mx = sub.add_parser('mixed', help='real only against real + X %% synthetic, at a fixed budget (resumable)')
    mx.add_argument('--root', default='', help='the experiments folder (default WORKSPACE/experiments)')
    mx.add_argument('--workspace', default='', help='the workspace (experiments/, models/pretrained/)')
    mx.add_argument('--id', required=True, help='the experiment\'s id (its folder)')
    mx.add_argument('--config', help='a JSON file of the definition (the flags below override it)')
    mx.add_argument('--synthetic', action='append', default=None, metavar='KIND:PATH[?k=v&…]')
    mx.add_argument('--real-train', action='append', default=None, metavar='KIND:PATH[?k=v&…]')
    mx.add_argument('--real-val', action='append', default=None, metavar='KIND:PATH[?k=v&…]')
    mx.add_argument('--real-test', action='append', default=None, metavar='KIND:PATH[?k=v&…]')
    mx.add_argument('--shares', help='synthetic shares in %%, e.g. 0,25,50,100 (0 is required)')
    mx.add_argument('--budget', type=int, help='training images per run (default: the most the pools allow)')
    mx.add_argument('--seeds', type=int, help='seeds per arm (default 3)')
    mx.add_argument('--seed', type=int, help='the first seed (default 0)')
    mx.add_argument('--boot', type=int, help='bootstrap resamples (default 1000)')
    mx.add_argument('--val-share', type=float)
    for k, t in (('epochs', int), ('batch', int), ('lr', float), ('workers', int)):
        mx.add_argument(f'--{k}', type=t)
    mx.add_argument('--arch', choices=tuple(T.ARCHS))
    mx.add_argument('--init', choices=T.INITS)
    mx.add_argument('--size', help='WIDTHxHEIGHT')
    mx.add_argument('--classes')
    mx.add_argument('--device')
    mx.add_argument('--no-export', action='store_true', help='do not export each run\'s model to ONNX')
    mx.add_argument('--weights-dir', action='append', default=[])
    for name, hlp in (('list', 'the runs and their latest metrics'), ('results', 'a mixed experiment\'s table')):
        p = sub.add_parser(name, help=hlp)
        p.add_argument('experiment')
    b = sub.add_parser('best', help='the finished run with the best latest value of a metric')
    b.add_argument('experiment')
    b.add_argument('--metric', required=True)
    b.add_argument('--min', action='store_true', help='lower is better')
    ex = sub.add_parser('export', help='write the experiment in MLflow\'s file-store layout')
    ex.add_argument('experiment')
    ex.add_argument('--mlflow', required=True, metavar='MLRUNS_DIR')
    ex.add_argument('--no-artifacts', action='store_true')
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return P.OK if e.code == 0 else P.FAILED
    em = P.Emitter()
    em.hello('experiment')
    try:
        if a.action == 'list':
            for r in Experiment(a.experiment).runs():
                ms = ', '.join(f'{k} {v:.4g}' for k, v in sorted(r['metrics'].items())
                               if isinstance(v, (int, float)) and k.startswith(('test.AP', 'val.AP'))
                               and not k.endswith(('ci_low', 'ci_high')))
                print(f'{r["name"]:24s} {r["status"] or "?":10s} {ms}', flush=True)
            em.result('ok')
            return P.OK
        if a.action == 'best':
            r = Experiment(a.experiment).best(a.metric, 'min' if a.min else 'max')
            if r is None:
                raise ExperimentError(f'no finished run has the metric {a.metric!r}')
            print(f'{r["name"]}: {a.metric} = {r["metrics"][a.metric]}', flush=True)
            em.result('ok', run=r['name'], value=r['metrics'][a.metric])
            return P.OK
        if a.action == 'results':
            p = os.path.join(a.experiment, 'results.json')
            if not os.path.isfile(p):
                raise ExperimentError(f'{a.experiment} has no results.json (not a finished mixed experiment)')
            print(table(PV.read_json(p)), flush=True)
            em.result('ok')
            return P.OK
        if a.action == 'export':
            r = export_mlflow(a.experiment, a.mlflow, not a.no_artifacts)
            print(f'wrote {r["path"]} ({len(r["runs"])} runs; {r["layout"]}); view with: mlflow ui '
                  f'--backend-store-uri file://{os.path.abspath(a.mlflow)}', flush=True)
            em.result('ok', path=r['path'], experiment_id=r['experiment_id'])
            return P.OK
        # mixed
        d = PV.read_json(a.config) if a.config else {}
        over = {'synthetic': a.synthetic, 'real_train': a.real_train, 'real_val': a.real_val,
                'real_test': a.real_test, 'budget_images': a.budget, 'seeds': a.seeds, 'seed': a.seed,
                'n_boot': a.boot, 'val_share': a.val_share}
        d.update({k: v for k, v in over.items() if v is not None})
        if a.shares:
            d['shares'] = [float(x) for x in a.shares.split(',') if x.strip()]
        trn = dict(d.get('train') or {})
        trn.update({k: v for k, v in {'epochs': a.epochs, 'batch': a.batch, 'lr': a.lr, 'workers': a.workers,
                                      'arch': a.arch, 'init': a.init, 'device': a.device}.items() if v is not None})
        if a.size:
            trn['size'] = T._size(a.size)
        if a.classes:
            trn['classes'] = [c.strip() for c in a.classes.split(',') if c.strip()]
        if a.no_export:
            trn['export'] = False
        d['train'] = trn
        root = a.root or (os.path.join(a.workspace, 'experiments') if a.workspace else '')
        if not root:
            raise ExperimentError('--root DIR or --workspace W is needed')
        flag = T.CancelFlag().install()
        locks = []
        try:
            m = normalise_mixed(d)
            if T._device(m['train']).type == 'cuda':
                locks = T.gpu_locks(em, flag, held_by_queue=os.environ.get('FALCON_GPU_LOCK_HELD') == '1')
            res = run_mixed(root, a.id, d, em, flag, a.workspace, a.weights_dir)
        finally:
            for lk in locks:
                lk.release()
        print(table(res), flush=True)
        em.result('ok', results=os.path.join(root, _safe(a.id), 'results.json'),
                  summary={r['arm']: r['AP']['value'] for r in res['arms']})
        return P.OK
    except T.Cancelled:
        em.result('cancelled')
        return P.CANCELLED
    except (ExperimentError, T.TrainingError, ValueError, OSError, KeyError) as e:
        print(f'failed: {e}', flush=True)
        em.result('failed', error=str(e))
        return P.FAILED
