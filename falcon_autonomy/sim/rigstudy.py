"""Rig studies: record rig variants on the same scenarios and seeds through the job queue, score a detector on every
camera, compare them (core.rigcompare), and the `falcon rigs` command (shared by the GUI, which queues it as a job,
and the CLI; docs/rig-compare.md).

A study is a folder <workspace>/rigstudies/<name>/: study.json (the rigs as they were recorded, the scenarios, seeds,
world, model and settings), runs/<rig>/<scenario>-s<seed>/ (one scenario run per rig, scenario and seed: kinematic
motion, no background traffic, so every rig sees the road users at the same places), predictions/<rig>.jsonl (the
model's detections and the ground truth of every camera frame), report.json and report.md. Re-running a study resumes
it: a run whose config is unchanged and which finished is not driven again, and the model's predictions are kept by
the runs' config hashes. `falcon rigs report` re-scores a study's predictions with other settings without the model.
"""
from __future__ import annotations

import copy
import json
import os
import re
import time

from ..core import protocol as P
from ..core import rigcompare as RC
from ..core import vantagerig as VR
from ..core.provenance import read_json, write_json

STUDIES = 'rigstudies'
FPS = 10.0
SEEDS = 3


def study_dir(workspace: str, name: str) -> str:
    return os.path.join(workspace, STUDIES, name)


def study_name(rigs: list) -> str:
    return 'rigs-' + re.sub(r'[^A-Za-z0-9_-]+', '-', '-'.join(r['id'][-12:] for r in rigs[:3])).strip('-')[:60]


def compare_argv(project_path: str, rigs: list, model: str, scenarios=(), seeds: int = SEEDS, world: str = '',
                 name: str = '', fake_carla: str = '') -> list:
    """`falcon rigs compare …` as the Rigs page and the assistant queue it (the arguments after `python -m
    falcon_autonomy`)."""
    argv = ['rigs', 'compare', project_path, '--model', model, '--seeds', str(int(seeds))]
    for r in rigs:
        argv += ['--rig', r]
    for s in scenarios:
        argv += ['--scenario', s]
    if world:
        argv += ['--world', world]
    if name:
        argv += ['--name', name]
    return argv + (['--fake-carla', fake_carla] if fake_carla else [])


class RigStudy:
    emit = None
    cancel = None

    def __init__(self, service, project, rigs: list, model: dict, scenarios=RC.DEFAULT_SCENARIOS,
                 seeds=(0, 1, 2), world: str = '', name: str = '', fps: float = FPS, seconds: float | None = None,
                 every: int = 1):
        if len(rigs) < 2:
            raise RC.CompareError('a comparison needs two rigs or more')
        ids = [r['id'] for r in rigs]
        if len(set(ids)) != len(ids):
            raise RC.CompareError('the same rig is named twice')
        self.svc, self.project, self.model = service, project, model
        self.rigs = [copy.deepcopy(r) for r in rigs]
        for r in self.rigs:
            cams = [s for s in r['sensors'] if s.get('type', '').startswith('camera.')]
            if not cams:
                raise RC.CompareError(f'rig {r["name"]!r} has no camera')
            bad = [s['name'] for s in cams if not {'instance', 'depth'} <= set(s.get('labels') or [])]
            if bad:
                raise RC.CompareError(f'rig {r["name"]!r}: camera(s) {", ".join(bad)} record no instance and depth '
                                      'labels (the objects a detector is scored against need them)')
        self.scenarios = {k: RC.scenario(k, project) for k in scenarios}
        if seconds:
            for s in self.scenarios.values():
                s['duration_s'] = float(seconds)
        self.seeds = [int(x) for x in seeds]
        self.world = world or project.doc['worlds'][0]['id']
        if project.item('worlds', self.world) is None:
            raise RC.CompareError(f'no world {self.world!r} in the project')
        self.fps, self.every = float(fps), int(every)
        self.name = name or study_name(self.rigs)
        self.dir = study_dir(project.workspace, self.name)
        self.notes = []
        vehicles = {r['vehicle'] for r in self.rigs}
        if len(vehicles) > 1:
            self.notes.append(f'the rigs are on different CARLA vehicles ({", ".join(sorted(vehicles))}): the car\'s '
                              'own body hides different ground, which is part of what is compared')

    # ------------------------------------------------------------------ plan
    def label(self, scn: str, seed: int) -> str:
        return f'{scn}-s{seed}'

    def configs(self) -> list:
        """[(rig id, label, config)] in recording order (rig by rig)."""
        from ..core.datasets import drive_key
        from ..core.drive import scenario_config
        proj = copy.deepcopy(self.project)            # the rigs as recorded (a Vantage file's never saved into it)
        for r in self.rigs:
            i, _ = proj.find('rigs', r['id'])
            if i is None:
                proj.doc['rigs'].append(copy.deepcopy(r))
            else:
                proj.doc['rigs'][i] = copy.deepcopy(r)
        out = []
        for r in self.rigs:
            for key, scn in self.scenarios.items():
                for seed in self.seeds:
                    lab = self.label(key, seed)
                    cfg = scenario_config(proj, scn, self.world, r['id'], '/pending', self.svc.port, self.svc.tm_port,
                                          seed=seed, fps=self.fps,
                                          render_budget_mpx=float(self.svc.settings.get('render_budget_mpx')),
                                          workspace=self.project.workspace)
                    cfg['scenario']['motion'] = 'kinematic'          # the same road users at every step, any rig
                    cfg['traffic'] = dict(cfg['traffic'], vehicles=0, walkers=0)
                    cfg['name'] = f'{self.name}: {r["name"]} · {scn["name"]} (seed {seed})'
                    cfg['cache_key'] = drive_key(cfg)
                    cfg['out'] = os.path.join(self.dir, 'runs', r['id'], lab)
                    out.append((r['id'], lab, cfg))
        return out

    def doc(self) -> dict:
        m = self.model
        return {'schema': RC.STUDY_SCHEMA, 'name': self.name, 'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'rigs': self.rigs, 'scenarios': self.scenarios, 'seeds': self.seeds, 'world': self.world,
                'fps': self.fps, 'every': self.every,
                'model': {k: m.get(k) for k in ('id', 'name', 'kind', 'runtime', 'entry', 'path', 'params',
                                                'class_map', 'falcon_classes')},
                'runs': {r['id']: {} for r in self.rigs}, 'notes': self.notes}

    # ------------------------------------------------------------------ recording
    def _done(self, cfg) -> bool:
        try:
            run = read_json(os.path.join(cfg['out'], 'run.json'))
            drv = read_json(os.path.join(cfg['out'], 'drive.json'))
        except (OSError, ValueError):
            return False
        return run.get('status') == 'ok' and drv.get('cache_key') == cfg['cache_key']

    def record(self, progress=None) -> dict:
        """Drive every run not already done; {rig: {label: run folder}}. A run that fails stops the study."""
        from ..core.jobs import FINAL
        os.makedirs(self.dir, exist_ok=True)
        plan = self.configs()
        doc = self.doc()
        for rid, lab, cfg in plan:
            doc['runs'][rid][lab] = os.path.relpath(cfg['out'], self.dir)
        write_json(os.path.join(self.dir, 'study.json'), doc)
        todo = [(rid, lab, cfg) for rid, lab, cfg in plan if not self._done(cfg)]
        done = len(plan) - len(todo)
        jobs = []
        for rid, lab, cfg in todo:
            jobs.append((rid, lab, cfg, self.svc.submit_drive(cfg, title=cfg['name'], kind='drive')))
        while any(j.state not in FINAL for *_, j in jobs):
            if self.cancel and self.cancel():
                for *_, j in jobs:
                    self.svc.queue.cancel(j)
            self.svc.queue.poll()
            if progress:
                fin = done + sum(j.state in FINAL for *_, j in jobs)
                progress(fin / max(len(plan), 1), f'{fin} of {len(plan)} runs recorded')
            time.sleep(0.2)
        failed = [(rid, lab, cfg, j) for rid, lab, cfg, j in jobs if not self._done(cfg)]
        if failed:
            rid, lab, cfg, j = failed[0]
            try:
                err = read_json(os.path.join(cfg['out'], 'run.json')).get('error')
            except (OSError, ValueError):
                err = None
            raise RuntimeError(f'{len(failed)} of {len(plan)} runs failed; first: {rid} / {lab}: '
                               f'{err or P.describe_exit(j.exit_code)}')
        return {rid: {lab: cfg['out'] for r2, lab, cfg in plan if r2 == rid} for rid in [r['id'] for r in self.rigs]}

    # ------------------------------------------------------------------ the model
    def predict(self, runs: dict, progress=None) -> dict:
        """{rig: frame records}: the model over every camera of every run (kept by the runs' config hashes)."""
        from ..core import milsearch as MS
        from ..core.models import Runner
        why = MS.evaluable(self.model, self.project.workspace)
        if why:
            raise RC.CompareError(f'the model cannot be evaluated: {why}')
        os.makedirs(os.path.join(self.dir, 'predictions'), exist_ok=True)
        out, locks, runner = {}, [], None
        try:
            for k, r in enumerate(self.rigs):
                path = os.path.join(self.dir, 'predictions', f'{r["id"]}.jsonl')
                key = _predictions_key(self.model, runs[r['id']])
                cached = _read_predictions(path, key)
                if cached is not None:
                    out[r['id']] = cached
                    continue
                if runner is None:
                    if self.model.get('runtime') != 'python':
                        from ..core.training import gpu_locks
                        locks = gpu_locks(self.emit, self.cancel)
                    runner = Runner(self.model, self.project.workspace)

                def sub(f, t, k=k):
                    if progress:
                        progress((k + f) / len(self.rigs), f'{r["name"]}: {t}')
                frames = RC.collect_rig(r, list(runs[r['id']].values()), self.model, self.project.workspace,
                                        self.every, runner, sub, self.cancel)
                _write_predictions(path, key, frames)
                out[r['id']] = frames
        finally:
            for lk in locks:
                lk.release()
        return out

    def run(self, progress=None, n_boot: int = 1000, seed: int = 0, **settings) -> dict:
        def sub(lo, hi):
            return (lambda f, t: progress(lo + (hi - lo) * f, t)) if progress else None
        runs = self.record(sub(0.0, 0.7))
        frames = self.predict(runs, sub(0.7, 0.95))
        if progress:
            progress(0.95, 'comparing')
        return write_report(self.dir, frames=frames, n_boot=n_boot, seed=seed, **settings)


def _predictions_key(model: dict, runs: dict) -> str:
    from ..core.provenance import sha256_json
    keys = {}
    for lab, d in sorted(runs.items()):
        try:
            keys[lab] = read_json(os.path.join(d, 'drive.json')).get('cache_key')
        except (OSError, ValueError):
            keys[lab] = None
    return sha256_json({'model': {k: model.get(k) for k in ('id', 'runtime', 'entry', 'path', 'params', 'class_map')},
                        'runs': keys})


def _read_predictions(path: str, key: str):
    try:
        with open(path, encoding='utf-8') as fh:
            head = json.loads(fh.readline() or '{}')
            if head.get('key') != key:
                return None
            return [json.loads(x) for x in fh if x.strip()]
    except (OSError, ValueError):
        return None


def _write_predictions(path: str, key: str, frames: list):
    from ..core.paths import atomic_write
    atomic_write(path, json.dumps({'key': key, 'schema': 'falcon.rig_predictions/1'}) + '\n'
                 + ''.join(json.dumps(f, separators=(',', ':')) + '\n' for f in frames))


def load(study: str) -> dict:
    p = os.path.join(study, 'study.json')
    if not os.path.isfile(p):
        raise RC.CompareError(f'{study}: no study.json (not a rig study)')
    return read_json(p)


LIMITS = (
    'CARLA renders its own car, not the project\'s body: cameras are mounted on the nearest CARLA car by size, from '
    'its rear axle; the body that hides ground near the car is CARLA\'s.',
    'Falcon renders no stitched bird\'s-eye image: "stitched recall" is detection in the camera the stitched view '
    'shows at the object\'s azimuth (Vantage\'s seams), not detection on a stitched image.',
    'Seams Vantage computed itself are not in its file: they are estimated by its rule on a box-shaped body (within '
    'a few degrees of Vantage\'s own on its sample project, docs/rig-compare.md); a seam set in Vantage is exact.',
    'Fisheye images are CARLA pinhole views remapped through the lens model, one instant per frame: no rolling '
    'shutter, motion blur, flare or the lens\'s real resolution fall-off beyond the model.',
)


def write_report(study: str, frames: dict | None = None, n_boot: int = 1000, seed: int = 0, pairs=None,
                 radius: float = RC.RADIUS_M, band_deg: float = RC.BAND_DEG, thr: float = RC.SCORE_THRESHOLD,
                 block_s: float = RC.BLOCK_S) -> dict:
    """Score a recorded study into report.json and report.md (frames: the predictions, else read from the study)."""
    from ..core import provenance as PV
    doc = load(study)
    runs = {rid: {lab: os.path.join(study, rel) for lab, rel in labs.items()} for rid, labs in doc['runs'].items()}
    if frames is None:
        frames = {}
        for r in doc['rigs']:
            p = os.path.join(study, 'predictions', f'{r["id"]}.jsonl')
            got = _read_predictions(p, _predictions_key(doc['model'], runs[r['id']]))
            if got is None:
                raise RC.CompareError(f'{r["id"]}: no predictions for its current runs (run `falcon rigs compare` '
                                      'again)')
            frames[r['id']] = got
    synthetic = False
    entry = str(doc['model'].get('entry') or '')
    if doc['model'].get('runtime') == 'python' and entry:
        try:
            import importlib
            mod, _, cls = entry.partition(':')
            synthetic = bool(getattr(getattr(importlib.import_module(mod), cls, None), 'synthetic', False))
        except (ImportError, ValueError):
            synthetic = False
    rep = RC.analyse(doc['rigs'], frames, runs, fps=float(doc.get('fps') or FPS), radius=radius, band_deg=band_deg,
                     thr=thr, n_boot=n_boot, seed=seed, block_s=block_s, pairs=pairs)
    rep.update(study=doc['name'], path=os.path.join(study, 'report.json'), model=doc['model'],
               synthetic_model=synthetic, world=doc['world'], seeds=doc['seeds'],
               scenarios={k: {'name': v.get('name'), 'about': v.get('about')} for k, v in doc['scenarios'].items()},
               rigs=[{'id': r['id'], 'name': r['name'], 'vehicle': r['vehicle'],
                      'vantage': {k: (r.get('vantage') or {}).get(k) for k in ('file', 'sha256', 'project', 'variant',
                                                                                'variant_uid', 'image_scale',
                                                                                'vehicle', 'notes')}
                      if r.get('vantage') else None} for r in doc['rigs']],
               notes=list(doc.get('notes') or []) + rep['notes'], limits=list(LIMITS))
    rep['provenance'] = PV.block({f'{rid}/{lab}/run.json': os.path.join(d, 'run.json')
                                  for rid, labs in runs.items() for lab, d in labs.items()}, seed,
                                 {'radius': radius, 'band_deg': band_deg, 'threshold': thr, 'n_boot': n_boot})
    write_json(os.path.join(study, 'report.json'), rep)
    with open(os.path.join(study, 'report.md'), 'w', encoding='utf-8') as fh:
        fh.write(RC.markdown(rep))
    return rep


def list_studies(workspace: str) -> list:
    """The workspace's rig studies, newest first: [{'id', 'dir', 'rigs', 'reported', 'mtime'}]."""
    base = os.path.join(workspace, STUDIES)
    out = []
    for n in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        d = os.path.join(base, n)
        if not os.path.isfile(os.path.join(d, 'study.json')):
            continue
        try:
            doc = read_json(os.path.join(d, 'study.json'))
        except (OSError, ValueError):
            continue
        rp = os.path.join(d, 'report.json')
        out.append({'id': n, 'dir': d, 'rigs': [r['name'] for r in doc.get('rigs', [])],
                    'reported': os.path.isfile(rp), 'mtime': os.path.getmtime(rp if os.path.isfile(rp) else d)})
    return sorted(out, key=lambda x: -x['mtime'])


# =========================================================================================== the command
def _pairs(vals):
    out = []
    for v in vals or []:
        a, _, b = v.partition(',')
        if not a or not b:
            raise RC.CompareError(f'--pair is A,B (two rig ids), not {v!r}')
        out.append((a, b))
    return out or None


def main_rigs(argv) -> int:
    """falcon rigs vantage|import|compare|report|note … (docs/rig-compare.md); exit codes as the CLI's."""
    import argparse
    ap = argparse.ArgumentParser(prog='falcon rigs', description='Vantage rig variants as Falcon rigs, compared by '
                                                                  'what a detector finds')
    sub = ap.add_subparsers(dest='what', required=True)
    v = sub.add_parser('vantage', help='what a .vantage file holds: variants, cameras, lenses, seams')
    v.add_argument('file')
    v.add_argument('--vehicle', default='auto')
    v.add_argument('--json', action='store_true')
    im = sub.add_parser('import', help='add Vantage variants to a project as rigs')
    im.add_argument('project')
    im.add_argument('file')
    im.add_argument('--variant', action='append', default=[], help='a variant\'s name, uid or number (default all)')
    im.add_argument('--vehicle', default='auto', help='a CARLA blueprint, or auto (the nearest by size)')
    im.add_argument('--scale', type=float, default=VR.IMAGE_SCALE, help='image scale against the module (0.05–1)')
    c = sub.add_parser('compare', help='record rigs on the same scenarios and seeds and compare a detector on them')
    c.add_argument('project')
    c.add_argument('--rig', action='append', default=[], help='a project rig id (two or more)')
    c.add_argument('--vantage', help='a .vantage file: its variants as rigs (not saved into the project)')
    c.add_argument('--variant', action='append', default=[])
    c.add_argument('--vehicle', default='auto')
    c.add_argument('--scale', type=float, default=VR.IMAGE_SCALE)
    c.add_argument('--model', required=True)
    c.add_argument('--scenario', action='append', default=[], help='default: ' + ', '.join(RC.DEFAULT_SCENARIOS))
    c.add_argument('--seeds', type=int, default=SEEDS, help='seeds 0 … N−1 per scenario')
    c.add_argument('--world', default='')
    c.add_argument('--name', default='')
    c.add_argument('--fps', type=float, default=FPS)
    c.add_argument('--seconds', type=float, default=0.0, help='each run\'s length (default: the scenario\'s)')
    c.add_argument('--every', type=int, default=1)
    c.add_argument('--fake-carla', default='')
    c.add_argument('--keep-server', action='store_true')
    for p in (c, sub.add_parser('report', help='score a recorded study again with other settings')):
        if p is not c:
            p.add_argument('study')
        p.add_argument('--pair', action='append', default=[], help='A,B: compare rig B against rig A')
        p.add_argument('--radius', type=float, default=RC.RADIUS_M)
        p.add_argument('--band', type=float, default=RC.BAND_DEG)
        p.add_argument('--threshold', type=float, default=RC.SCORE_THRESHOLD)
        p.add_argument('--boot', type=int, default=1000)
        p.add_argument('--seed', type=int, default=0)
    n = sub.add_parser('note', help='a Vantage design-log note on a study\'s comparison, written to a file')
    n.add_argument('study')
    n.add_argument('--out', '-o', required=True)
    n.add_argument('--pair', default='')
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return P.OK if e.code == 0 else P.FAILED
    try:
        if a.what == 'vantage':
            s = VR.summary(VR.read(a.file), a.vehicle)
            if a.json:
                print(json.dumps(s, indent=1))
                return P.OK
            _print_summary(s)
            return P.OK
        if a.what == 'import':
            return _import(a)
        if a.what == 'report':
            rep = write_report(a.study, n_boot=a.boot, seed=a.seed, pairs=_pairs(a.pair), radius=a.radius,
                               band_deg=a.band, thr=a.threshold)
            _print_report(rep)
            print(os.path.join(a.study, 'report.md'))
            return P.OK
        if a.what == 'note':
            rep = read_json(os.path.join(a.study, 'report.json'))
            pair = tuple(a.pair.split(',')) if a.pair else None
            note = RC.design_log(rep, pair)
            if os.path.splitext(a.out)[1].lower() == '.vantage':
                raise RC.CompareError('the note is written to its own file (.json), never into a .vantage project')
            write_json(a.out, note)
            for e in note['entries']:
                print(f'- {e["text"]}')
            print(a.out)
            return P.OK
        return _compare(a)
    except (RC.CompareError, VR.VantageError, OSError) as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED


def _print_summary(s):
    print(f'{s["name"]}  ({s["file"]})')
    veh = s['vehicle']
    print(f'  CARLA vehicle: {veh["blueprint"]} ({veh["how"]}'
          + (f', size error {veh["size_error"] * 100:.1f} %' if veh.get('size_error') is not None else '') + ')')
    for v in s['variants']:
        print(f'  {"*" if v["active"] else " "} {v["name"]}: {len(v["cameras"])} cameras'
              + (f', {len(v["sensors"])} other sensors' if v['sensors'] else '')
              + (f', {v["seams_set"]} seams set in Vantage' if v['seams_set'] else ''))
        for c in v['cameras']:
            print(f'      {c["tag"]:4s} x {c["x"]:+.3f} y {c["y"]:+.3f} z {c["z"]:.3f}  yaw {c["yaw"]:+.1f} pitch '
                  f'{c["pitch"]:+.1f} roll {c["roll"]:+.1f}  {c["module"]} / {c["lens"]}'
                  + (' (unit calibration)' if c['calibrated'] else ''))
        if v.get('seams'):
            print('      seams: ' + ', '.join(f'{x["a"]}|{x["b"]} {x["angle_deg"]:+.1f}°'
                                             + ('' if x['source'] == 'vantage' else '*') for x in v['seams'])
                  + '   (* estimated)')
    for n in s['notes']:
        print(f'  note: {n}')


def _import(a) -> int:
    from ..core.project import Edit, Project, ProjectError
    try:
        proj = Project.load(a.project)
    except (OSError, ProjectError) as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
    doc = VR.read(a.file)
    rigs = VR.rigs_for(doc, a.variant or None, a.vehicle, a.scale)
    new = VR.merge_into(proj.doc['rigs'], rigs)
    proj.apply(Edit('set', ['rigs'], new, text=f'Import {len(rigs)} rig(s) from {os.path.basename(doc["path"])}'))
    proj.save()
    for r in rigs:
        rid = next(x['id'] for x in new if (x.get('vantage') or {}).get('variant_uid') == r['vantage']['variant_uid']
                   and (x.get('vantage') or {}).get('file') == r['vantage']['file'])
        print(f'{rid}: {r["name"]} on {r["vehicle"]}, {len(r["sensors"])} sensors')
        for n in r['vantage']['notes']:
            print(f'  note: {n}')
    return P.OK


def _compare(a) -> int:
    from ..core.jobs import JobQueue, SubprocessLauncher
    from ..core.project import Project, ProjectError
    from ..core.settings import Settings
    from .service import CarlaService
    try:
        proj = Project.load(a.project)
    except (OSError, ProjectError) as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
    rigs = []
    for rid in a.rig:
        r = proj.item('rigs', rid)
        if r is None:
            raise RC.CompareError(f'no rig {rid!r} in the project')
        rigs.append(r)
    if a.vantage:
        rigs += VR.rigs_for(VR.read(a.vantage), a.variant or None, a.vehicle, a.scale)
    m = proj.item('models', a.model)
    if m is None:
        raise RC.CompareError(f'no model {a.model!r} in the project')
    em = P.Emitter()
    em.hello('rigs')
    svc = CarlaService(Settings(), JobQueue(SubprocessLauncher()), proj.workspace, fake_carla=a.fake_carla)
    try:
        st = RigStudy(svc, proj, rigs, m, tuple(a.scenario) or RC.DEFAULT_SCENARIOS, range(a.seeds), a.world, a.name,
                      a.fps, a.seconds or None, a.every)
        st.emit = em
        for n in st.notes:
            em.log(n)
        rep = st.run(lambda f, t: em.progress(f, t), n_boot=a.boot, seed=a.seed, pairs=_pairs(a.pair),
                     radius=a.radius, band_deg=a.band, thr=a.threshold)
    except (RC.CompareError, VR.VantageError, ValueError, RuntimeError, ProjectError) as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
    finally:
        if not a.keep_server:
            svc.shutdown()
    _print_report(rep)
    em.result('ok', summary={'study': st.name, 'dir': st.dir, 'worse': sum(len(p['worse']) for p in rep['pairs']),
                             'better': sum(len(p['better']) for p in rep['pairs'])})
    return P.OK


def _print_report(rep):
    for v in rep['variants'].values():
        t = {r['metric']: r for r in v['table']}
        print(f'{v["name"]}: ' + ', '.join(f'{k} {RC._f(t[k]["value"])}{RC._ci(t[k])}' for k in RC.HEADLINE
                                           if k in t))
    for p in rep['pairs']:
        print(f'{p["b"]} against {p["a"]}: worse {", ".join(p["worse"]) or "nothing"}; better '
              f'{", ".join(p["better"]) or "nothing"}')
