"""Deploying a registered ONNX detector to a Jetson AGX Orin as TensorRT engines, and gating its release on accuracy,
latency and energy measured on the device (docs/deploy.md; numpy at import, the rest inside the functions).

TensorRT engines are specific to the device and the TensorRT version that built them, so everything that touches an
engine happens on the device: Falcon copies the model and its tool there over SSH (hil.deploy's key-only SSH; the
address, user and key live only in the user's settings, never in a project, a result or a log) and runs
hil/engine_device.py there. Or on this machine as the **simulated device**: the same tool, the same files and steps,
with onnxruntime on the CPU standing in for TensorRT — labelled everywhere: not TensorRT, not the Orin, the pipeline
only. Its numbers are this machine's CPU's, and its gate verdicts are never release decisions.

Stages (`falcon deploy STAGE`, each reading and extending DIR/deploy.json):
  build     here: the evaluation frames (every n-th frame of each drive, as core.harness.collect chooses them) and
            the calibration frames (DEFAULT_CALIB_COUNT from other drives, spread evenly) are resized or letterboxed
            by core.models.geometry() and stored as uint8 (.npz); the bundle (the tool, falcon_models.py = a verbatim
            copy of core/models.py, the ONNX model, spec.json, the frames) goes to the device; INT8 calibration runs
            there (cached by a key over the model and the frames); one engine per precision (fp32, fp16, int8) and,
            with a DLA core, fp16/int8 on it with GPU fallback. Build logs, layer precisions and the workspace are
            brought back.
  profile   each engine timed on the device (median, p95, p99 over N runs after a warm-up), throughput, power on the
            INA3221 rails with an idle baseline before, energy per inference = mean power × median latency
            (estimated, and labelled so), memory, the power mode and clocks it ran under.
  accuracy  each engine over the evaluation frames on the device; the detections come back and are scored here by
            core.harness against the drives' ground truth, beside the host's ONNX model on the same frames
            (onnxruntime on the CPU: the reference); a paired bootstrap over drives gives each engine's AP difference
            and AP ratio to the reference with 95 % intervals.
  gate      a gate file (GATE_SCHEMA) evaluated per engine: pass / fail / incomplete, with the evidence of each
            criterion.
  report    the 'deploy' report (core.reports): the engines, their numbers, the gate.

The decoder is the host's: the device normalises each stored frame with falcon_models.normalise() and decodes with
falcon_models.decode_detections() — the same code as core.models (the copy's SHA-256 is recorded) — so an engine
whose outputs equal the reference's produces identical boxes (the simulated fp32 engine does, and the tests check it).
"""
from __future__ import annotations

import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np

SCHEMA = 'falcon.deploy/1'
GATE_SCHEMA = 'falcon.deploy-gate/1'
PRECISIONS = ('fp32', 'fp16', 'int8')
DEVICES = ('sim', 'orin')
DEFAULT_CALIB_COUNT = 500          # TensorRT's guidance: about 500 images suffice for INT8 calibration
DEFAULT_RUNS = 1000
DEFAULT_WARMUP_MS = 2000
DEFAULT_MIN_SECONDS = 10.0
DEFAULT_WORKSPACE_MIB = 4096
SIMULATED = ('SIMULATED DEVICE — not TensorRT, not the Orin: onnxruntime on this machine\'s CPU stands in for the '
             'engine; the pipeline only. No number here is the Orin\'s.')
DEFAULT_GATE = {
    'schema': GATE_SCHEMA,
    'name': 'release gate',
    'accuracy': {'metric': 'AP', 'min_relative': 0.98, 'rule': 'ci_lower', 'min_drives': 5, 'min_reference': 0.05},
    'latency': {'max_p95_ms': 33.3, 'measure': 'latency'},
    'energy': {'max_mj_per_frame': 1000.0, 'rails': 'sum'},
    'power_mode': None,
    'require_locked_clocks': False,
    'missing': 'incomplete',
}
_TOOL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'hil', 'engine_device.py')
_MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models.py')


class DeployError(Exception):
    pass


# ============================================================================================ the device
class LocalDevice:
    """The simulated device: a folder on this machine (DIR/simdevice) and this machine's Python running the tool."""
    simulated = True

    def __init__(self, root: str, python: str = ''):
        self.root = os.path.abspath(root)
        self.python = python or sys.executable
        os.makedirs(self.root, exist_ok=True)

    def describe(self) -> dict:
        return {'mode': 'sim', 'where': 'this machine (simulated device)', 'folder': self.root}

    def put(self, local: str, rel: str):
        dst = os.path.join(self.root, rel)
        if os.path.isdir(local):
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(local, dst)
        else:
            os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
            shutil.copyfile(local, dst)

    def get(self, rel: str, local: str) -> bool:
        src = os.path.join(self.root, rel)
        if not os.path.exists(src):
            return False
        os.makedirs(os.path.dirname(local) or '.', exist_ok=True)
        shutil.copyfile(src, local)
        return True

    def run(self, args: list, timeout: float) -> dict:
        try:
            r = subprocess.run([self.python, 'engine_device.py'] + [str(x) for x in args], cwd=self.root,
                               capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            raise DeployError(f'the tool did not finish within {timeout:.0f} s') from None
        return _result(r.returncode, r.stdout, r.stderr)


class SshDevice:
    """The Orin over SSH (hil.deploy: the user's key, BatchMode, accept-new host keys); its folder is
    <remote_dir>/deploy/<name> and every message is cleaned of the address and the user."""
    simulated = False

    def __init__(self, dev: dict, name: str):
        from ..hil import deploy as D
        self.D, self.dev = D, dev
        self._call(D.validate, dev)
        import re
        rd = str(dev.get('remote_dir') or '~/falcon-hil').rstrip('/')
        self.rel = f'{rd}/deploy/{re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip(".-") or "deploy"}'
        self.shell = D.remote_dir(dict(dev, remote_dir=self.rel))       # a shell word, ~ expanded by the shell
        self.scp_base = self.rel[2:] if self.rel.startswith('~/') else self.rel   # scp: relative to the home folder
        self.python = shlex.quote(str(dev.get('python') or 'python3'))

    def describe(self) -> dict:
        return {'mode': 'orin', 'where': 'from the user\'s settings', 'folder': self.rel}

    def _call(self, fn, *args):
        """hil.deploy's own errors (no address, no key, SSH refused, a time-out) as this module's."""
        try:
            return fn(*args)
        except self.D.DeployError as e:
            raise DeployError(str(e)) from None

    def _scp(self, a: str, b: str, recursive: bool):
        D = self.D
        cmd = [shutil.which('scp') or 'scp', '-P', str(int(self.dev.get('ssh_port') or 22))] + D._opts(self.dev) + \
            ['-q'] + (['-r'] if recursive else []) + [a, b]
        r = self._call(D.run, cmd, 3600.0)
        if r.returncode != 0:
            raise DeployError(f'copying failed: {D._clean(r.stderr or r.stdout, self.dev)}')

    def _remote(self, rel: str) -> str:
        return f'{self.dev["ssh_user"]}@{self.dev["address"]}:{self.scp_base}/{rel}'

    def put(self, local: str, rel: str):
        parent = os.path.dirname(rel)
        q = shlex.quote(rel)
        self._call(self.D.ssh, self.dev, f'mkdir -p {self.shell}/{shlex.quote(parent or ".")} && rm -rf '
                                         f'{self.shell}/{q}')
        self._scp(local, self._remote(rel), os.path.isdir(local))

    def get(self, rel: str, local: str) -> bool:
        os.makedirs(os.path.dirname(local) or '.', exist_ok=True)
        try:
            self._scp(self._remote(rel), local, False)
        except DeployError:
            return False
        return os.path.exists(local)

    def run(self, args: list, timeout: float) -> dict:
        D = self.D
        line = ' '.join(shlex.quote(str(x)) for x in args)
        r = self._call(D.run, D.ssh_command(self.dev, f'cd {self.shell} || exit 3; {self.python} engine_device.py '
                                                      f'{line}'), timeout)
        if r.returncode == 255:
            raise DeployError(f'SSH could not log in: {D._clean(r.stderr, self.dev)}')
        return _result(r.returncode, r.stdout, D._clean(r.stderr or '', self.dev))


def _result(code: int, out: str, err: str) -> dict:
    lines = [x for x in (out or '').splitlines() if x.strip()]
    try:
        doc = json.loads(lines[-1]) if lines else None
    except ValueError:
        doc = None
    if not isinstance(doc, dict):
        raise DeployError(f'the tool printed no result (exit {code}): {(err or out or "")[-600:]}')
    if code != 0 or doc.get('error'):
        raise DeployError(doc.get('error') or f'the tool failed (exit {code})')
    return doc


def make_device(doc: dict, out_dir: str, settings=None):
    """The device a deploy folder was built for: its simulated folder, or the Orin from the user's settings."""
    if doc['device']['mode'] == 'sim':
        return LocalDevice(os.path.join(out_dir, 'simdevice'))
    from .hil import device_settings
    from .settings import Settings
    return SshDevice(device_settings(settings or Settings()), os.path.basename(os.path.normpath(out_dir)))


# ============================================================================================ frames
def _onnx_io(path: str) -> dict:
    import onnxruntime as ort
    s = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    i = s.get_inputs()[0]
    return {'input_name': i.name, 'onnx_input_shape': [d if isinstance(d, int) else None for d in i.shape],
            'output_names': [o.name for o in s.get_outputs()]}


def model_spec(m: dict, onnx_path: str) -> dict:
    io = _onnx_io(onnx_path)
    inp = dict(m.get('input') or {})
    w, h = int(inp.get('width', 640)), int(inp.get('height', 640))
    shape = [1, 3, h, w] if inp.get('layout', 'NCHW') == 'NCHW' else [1, h, w, 3]
    dynamic = any(d is None for d in io['onnx_input_shape'])
    return {'input': inp, 'model': {k: m.get(k) for k in ('kind', 'output', 'classes', 'class_map')},
            'input_shape': shape, 'dynamic': dynamic,
            'shapes': f'{io["input_name"]}:{"x".join(map(str, shape))}' if dynamic else '', **io}


def _first_camera(run, camera):
    from .exporters import is_pinhole
    cams = [c for c in run.cams if is_pinhole(run, c)] or list(run.cams)
    cid = camera or (cams[0] if cams else None)
    if cid not in run.cams:
        raise DeployError(f'{run.dir}: no camera {cid!r}')
    return cid


def candidates(runs: list, camera: str | None, every: int) -> list:
    """(run, camera, record) of every n-th labelled frame of each drive — the frames core.harness.collect scores."""
    from .exporters import ExportError, Run
    out = []
    for r in runs:
        try:
            run = Run(r)
        except ExportError as e:
            raise DeployError(str(e)) from None
        cid = _first_camera(run, camera)
        out += [(run, cid, rec) for rec in run.camera_steps(cid, every) if run.file(rec, cid, 'objects')]
    return out


def write_frames(items: list, spec: dict, out_dir: str, role: str) -> dict:
    """The model-size uint8 frames (core.models.geometry: the host's own resize or letterbox) as .npz, and
    frames.json: image_id, drive, step, file, the transform back to image pixels, the source image."""
    import cv2
    from . import models as MD
    os.makedirs(out_dir, exist_ok=True)
    frames = []
    for i, (run, cid, rec) in enumerate(items):
        path = run.p(run.file(rec, cid, 'rgb'))
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise DeployError(f'cannot read {path}')
        x, back = MD.geometry(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), spec['input'])
        f = f'{i:06d}.npz'
        np.savez_compressed(os.path.join(out_dir, f), x=np.ascontiguousarray(x))
        frames.append({'i': i, 'image_id': f'{run.key}/{rec["step"]}', 'drive': run.key, 'step': rec['step'],
                       'camera': cid, 'file': f, 'image': path, 'back': {k: float(v) for k, v in back.items()}})
    doc = {'role': role, 'n': len(frames), 'frames': frames}
    with open(os.path.join(out_dir, 'frames.json'), 'w', encoding='utf-8') as fh:
        json.dump(doc, fh)
    return doc


def pick_calibration(items: list, count: int, exclude: set) -> list:
    """count frames spread evenly over the candidates (in drive order), none of them an evaluation frame."""
    pool = [it for it in items if f'{it[0].key}/{it[2]["step"]}' not in exclude]
    if count <= 0 or count >= len(pool):
        return pool
    idx = np.unique(np.linspace(0, len(pool) - 1, count).round().astype(int))
    return [pool[i] for i in idx]


# ============================================================================================ build
def engine_names(precisions, dla) -> list:
    names = [p for p in PRECISIONS if p in precisions]
    if dla is not None:
        names += [f'{p}-dla{int(dla)}' for p in ('fp16', 'int8') if p in precisions]
    return names


def _split(name: str) -> tuple:
    p, _, d = name.partition('-dla')
    return p, (int(d) if d else None)


def read_doc(out_dir: str) -> dict:
    from .provenance import read_json
    try:
        return read_json(os.path.join(out_dir, 'deploy.json'))
    except (OSError, ValueError) as e:
        raise DeployError(f'{out_dir} is not a deploy folder: {e}') from None


def save_doc(out_dir: str, doc: dict):
    from .harness import _clean
    from .provenance import write_json
    write_json(os.path.join(out_dir, 'deploy.json'), _clean(doc))


def build(m: dict, workspace: str, runs: list, out_dir: str, *, calib_runs=None, device: str = 'sim',
          precisions=PRECISIONS, dla=None, calib_count: int = DEFAULT_CALIB_COUNT, every: int = 1, camera=None,
          workspace_mib: int = DEFAULT_WORKSPACE_MIB, threads: int = 0, seed: int = 0, settings=None,
          backend: str = '', progress=None) -> dict:
    """The build stage (module doc) into out_dir; returns deploy.json's document."""
    from . import models as MD
    from . import provenance as PV
    from .paths import tilde
    if m.get('kind') != 'detector' or m.get('runtime') != 'onnx':
        raise DeployError('deployment takes a registered ONNX detector (kind detector, runtime onnx)')
    problems = MD.validate(m)
    if problems:
        raise DeployError('; '.join(problems))
    onnx_path = MD.resolve_path(m, workspace)
    if not os.path.isfile(onnx_path):
        raise DeployError(f'the model file is not here: {onnx_path} (download it first)')
    if device not in DEVICES:
        raise DeployError(f'device must be one of {DEVICES}')
    precisions = [p for p in PRECISIONS if p in precisions]
    if not precisions:
        raise DeployError(f'choose at least one precision of {PRECISIONS}')
    calib_runs = list(calib_runs or [])
    if 'int8' in precisions and not calib_runs:
        raise DeployError('INT8 needs calibration drives (--calib-run): drives other than the evaluation\'s')
    os.makedirs(out_dir, exist_ok=True)
    say = progress or (lambda f, t: None)
    say(0.02, 'preparing the frames')
    spec = model_spec(m, onnx_path)
    bundle = os.path.join(out_dir, 'bundle')
    shutil.rmtree(bundle, ignore_errors=True)
    os.makedirs(bundle)
    ev_items = candidates(runs, camera, every)
    if not ev_items:
        raise DeployError('the drives have no labelled frames of that camera')
    ev = write_frames(ev_items, spec, os.path.join(bundle, 'eval'), 'evaluation')
    eval_ids = {f['image_id'] for f in ev['frames']}
    notes = []
    calib = None
    if 'int8' in precisions:
        shared = {os.path.abspath(r) for r in calib_runs} & {os.path.abspath(r) for r in runs}
        if shared:
            notes.append(f'{len(shared)} calibration drive(s) are also evaluation drives: their evaluation frames '
                         'were left out of calibration, but neighbouring frames of the same drive are alike, so the '
                         'INT8 accuracy may be optimistic')
        picked = pick_calibration(candidates(calib_runs, camera, 1), calib_count, eval_ids)
        if not picked:
            raise DeployError('the calibration drives have no frames outside the evaluation set')
        if len(picked) < calib_count:
            notes.append(f'{len(picked)} calibration frames (fewer than the {calib_count} asked for)')
        calib = write_frames(picked, spec, os.path.join(bundle, 'calib'), 'calibration')
    shutil.copyfile(_TOOL, os.path.join(bundle, 'engine_device.py'))
    shutil.copyfile(_MODELS, os.path.join(bundle, 'falcon_models.py'))
    shutil.copyfile(onnx_path, os.path.join(bundle, 'model.onnx'))
    with open(os.path.join(bundle, 'spec.json'), 'w', encoding='utf-8') as fh:
        json.dump(spec, fh, indent=1)
    if device == 'sim':
        dev = LocalDevice(os.path.join(out_dir, 'simdevice'))
        backend = backend or 'onnxruntime'
    else:
        from .hil import device_settings
        from .settings import Settings
        dev = SshDevice(device_settings(settings or Settings()), os.path.basename(os.path.normpath(out_dir)))
        backend = backend or 'tensorrt'
    simulated = backend == 'onnxruntime'
    model_sha = PV.sha256_file(onnx_path)
    doc = {'schema': SCHEMA, 'simulated': simulated, 'label': SIMULATED if simulated else None,
           'model': {k: m.get(k) for k in ('id', 'name', 'kind', 'runtime', 'licence', 'classes', 'class_map',
                                            'input', 'output')} | {'path': tilde(onnx_path), 'sha256': model_sha},
           'device': dev.describe() | {'backend': backend}, 'notes': notes,
           'plan': {'precisions': precisions, 'dla_core': dla, 'engines': engine_names(precisions, dla),
                    'every': every, 'camera': camera, 'runs': [os.path.abspath(r) for r in runs],
                    'calib_runs': [os.path.abspath(r) for r in calib_runs], 'calib_count_asked': calib_count,
                    'calib_frames': calib['n'] if calib else 0, 'eval_frames': ev['n'],
                    'eval_drives': sorted({f['drive'] for f in ev['frames']}), 'workspace_mib': workspace_mib,
                    'threads': threads, 'spec': spec,
                    'falcon_models_sha256': PV.sha256_file(_MODELS), 'tool_sha256': PV.sha256_file(_TOOL)},
           'engines': {}, 'stages': {}}
    if dla is not None and 'fp32' in precisions:
        notes.append('the DLA runs FP16 and INT8 only: fp32 is built for the GPU alone')
    say(0.1, 'copying to the device')
    for f in ('engine_device.py', 'falcon_models.py', 'model.onnx', 'spec.json'):
        dev.put(os.path.join(bundle, f), f)
    dev.put(os.path.join(bundle, 'eval'), 'eval')
    if calib:
        dev.put(os.path.join(bundle, 'calib'), 'calib')
    common = ['--backend', backend] + (['--threads', str(threads)] if threads else []) + \
        (['--shapes', spec['shapes']] if spec['shapes'] else [])
    say(0.15, 'checking the device')
    doc['env'] = dev.run(['check'], 120.0)
    t_start = time.time()
    if calib:
        say(0.2, f'INT8 calibration on {calib["n"]} frames')
        key = PV.sha256_json({'model': model_sha, 'frames': [f['image_id'] for f in calib['frames']],
                              'spec': spec['input'], 'backend': backend})
        cache = 'cache/int8-qdq.onnx' if simulated else 'cache/int8.cache'     # outside calib/: a re-copy keeps it
        doc['calibration'] = dev.run(['calibrate'] + common + ['--onnx', 'model.onnx', '--frames', 'calib',
                                                              '--spec', 'spec.json', '--cache', cache, '--key', key,
                                                              '--workspace-mib', str(workspace_mib)], 7200.0)
    names = doc['plan']['engines']
    for k, name in enumerate(names):
        say(0.25 + 0.7 * k / len(names), f'building {name}')
        p, core = _split(name)
        args = ['build'] + common + ['--onnx', 'model.onnx', '--precision', p, '--out', f'engines/{name}',
                                     '--workspace-mib', str(workspace_mib)]
        if p == 'int8':
            args += ['--calib', doc['calibration']['cache']]
        if core is not None:
            args += ['--dla', str(core)]
        rec = {'name': name, 'precision': p, 'dla_core': core}
        try:
            rec['build'] = dev.run(args, 7200.0)
        except DeployError as e:
            rec['build'] = {'error': str(e)}
        for f in ('build.json', 'build.log', 'layers.json'):
            dev.get(f'engines/{name}/{f}', os.path.join(out_dir, 'engines', name, f))
        doc['engines'][name] = rec
    doc['stages']['build'] = {'finished': PV.utc_now(), 'seconds': time.time() - t_start}
    doc['provenance'] = PV.block({'model': onnx_path} | {f'run{i}/run.json': os.path.join(r, 'run.json')
                                                          for i, r in enumerate(runs)}
                                 | {f'calib{i}/run.json': os.path.join(r, 'run.json')
                                    for i, r in enumerate(calib_runs)}, seed,
                                 {'device': device, 'backend': backend, 'precisions': precisions, 'dla': dla,
                                  'every': every, 'camera': camera, 'calib_count': calib_count,
                                  'workspace_mib': workspace_mib})
    save_doc(out_dir, doc)
    say(1.0, 'built')
    return doc


def _built(doc: dict) -> list:
    return [n for n, e in doc['engines'].items() if not (e.get('build') or {}).get('error')]


# ============================================================================================ profile
def _stats(v) -> dict | None:
    if not v:
        return None
    a = np.asarray(v, float)
    return {'n': int(a.size), 'median': float(np.median(a)), 'p95': float(np.percentile(a, 95)),
            'p99': float(np.percentile(a, 99)), 'mean': float(a.mean()), 'std': float(a.std()),
            'min': float(a.min()), 'max': float(a.max())}


def _mean_rails(samples: list, window=None) -> dict | None:
    if window and all(window):
        samples = [s for s in samples if window[0] <= s['t_ns'] <= window[1]]
    if not samples:
        return None
    rails = sorted({k for s in samples for k in s['mw']})
    full = [s for s in samples if all(k in s['mw'] for k in rails)]
    return {'n_samples': len(samples), 'rails_mw': {k: float(np.mean([s['mw'][k] for s in samples if k in s['mw']]))
                                                     for k in rails},
            'sum_mw': float(np.mean([sum(s['mw'].values()) for s in full])) if full else None}


def summarise_profile(raw: dict) -> dict:
    """The statistics of a profile.json (the tool's raw values): latency, throughput, power, energy, memory."""
    lat, comp = _stats(raw.get('latency_ms')), _stats(raw.get('compute_ms'))
    out = {'latency_ms': lat, 'compute_ms': comp, 'latency_definition': raw.get('latency_definition'),
           'runs': (lat or {}).get('n'), 'warmup_ms': raw.get('warmup_ms'), 'simulated': bool(raw.get('simulated')),
           'power_mode': raw.get('power_mode'), 'clocks_locked': (raw.get('clocks') or {}).get('locked'),
           'temperatures_before_c': raw.get('temperatures_before_c'),
           'temperatures_after_c': raw.get('temperatures_after_c'), 'cpu_load_1min': raw.get('cpu_load_1min'),
           'threads': raw.get('threads'), 'providers': raw.get('providers')}
    w = raw.get('window_ns') or [None, None]
    if raw.get('throughput_qps_reported'):
        out['throughput_qps'] = raw['throughput_qps_reported']
        out['throughput_source'] = 'trtexec (its Throughput line)'
    elif lat and raw.get('wall_s'):
        out['throughput_qps'] = lat['n'] / raw['wall_s']
        out['throughput_source'] = 'timed runs / their wall time, one at a time'
    pw = {'source': raw.get('power_source'), 'interval_ms': raw.get('power_interval_ms'),
          'chip_update_interval_ms': raw.get('power_update_interval_ms')}
    run = _mean_rails(raw.get('samples') or [], w)
    idle = _mean_rails(raw.get('idle_samples') or [])
    if run is None:
        pw['available'] = False
        pw['why'] = ('no INA3221 rails on this machine (not a Jetson): power and energy are not measured'
                     if not raw.get('power_source') else 'no power sample fell in the timed window')
    else:
        pw.update(available=True, window=raw.get('window_definition') or 'the timed runs', run=run, idle=idle)
        if lat and run.get('sum_mw') is not None:
            pw['energy_mj_per_inference'] = run['sum_mw'] / 1000.0 * lat['median']
            pw['energy_definition'] = ('ESTIMATED: mean power over the timed window (sum of the reported rails) × '
                                       'the median latency')
            if idle and idle.get('sum_mw') is not None:
                pw['dynamic_energy_mj_per_inference'] = (run['sum_mw'] - idle['sum_mw']) / 1000.0 * lat['median']
            pw['rails_energy_mj'] = {k: v / 1000.0 * lat['median'] for k, v in run['rails_mw'].items()}
    out['power'] = pw
    mb, ml = raw.get('mem_available_kib_before'), raw.get('mem_available_kib_min')
    out['memory'] = {'max_rss_mib': (raw.get('max_rss_kib') or 0) / 1024.0 or None,
                     'mem_available_drop_mib': (mb - ml) / 1024.0 if mb and ml else None,
                     'note': 'MemAvailable before the run minus its lowest value during it (system-wide; on a Jetson '
                             'the GPU shares this memory, so it includes the engine\'s) and the tool\'s peak RSS',
                     'trtexec_lines': ((raw.get('log') or {}).get('memory_lines') or [])[:20]}
    return out


def profile(out_dir: str, *, runs: int = DEFAULT_RUNS, warmup_ms: int = DEFAULT_WARMUP_MS,
            min_seconds: float = DEFAULT_MIN_SECONDS, idle_s: float = 3.0, power_interval_ms: float = 20.0,
            settings=None, progress=None) -> dict:
    from . import provenance as PV
    doc = read_doc(out_dir)
    dev = make_device(doc, out_dir, settings)
    say = progress or (lambda f, t: None)
    spec = doc['plan']['spec']
    common = ['--backend', doc['device']['backend']] + \
        (['--threads', str(doc['plan']['threads'])] if doc['plan'].get('threads') else []) + \
        (['--shapes', spec['shapes']] if spec.get('shapes') else [])
    names = _built(doc)
    if not names:
        raise DeployError('no engine was built')
    t0 = time.time()
    for k, name in enumerate(names):
        say(k / len(names), f'profiling {name}')
        try:
            dev.run(['profile'] + common + ['--engine', f'engines/{name}', '--input', 'eval/000000.npz', '--spec',
                                            'spec.json', '--runs', str(runs), '--warmup-ms', str(warmup_ms),
                                            '--min-seconds', str(min_seconds), '--idle-s', str(idle_s),
                                            '--power-interval-ms', str(power_interval_ms)],
                    1800.0 + min_seconds * 2)
            local = os.path.join(out_dir, 'engines', name, 'profile.json')
            dev.get(f'engines/{name}/profile.json', local)
            dev.get(f'engines/{name}/profile.log', os.path.join(out_dir, 'engines', name, 'profile.log'))
            doc['engines'][name]['profile'] = summarise_profile(PV.read_json(local))
        except (DeployError, OSError, ValueError) as e:
            doc['engines'][name]['profile'] = {'error': str(e)}
    doc['stages']['profile'] = {'finished': PV.utc_now(), 'seconds': time.time() - t0, 'runs': runs,
                                'warmup_ms': warmup_ms, 'min_seconds': min_seconds, 'idle_s': idle_s,
                                'power_interval_ms': power_interval_ms}
    save_doc(out_dir, doc)
    say(1.0, 'profiled')
    return doc


# ============================================================================================ accuracy
class _CpuReference:
    """The host's ONNX model through onnxruntime on the CPU: core.models.Runner's preprocessing and decoding, the
    CPU provider only (a fixed reference whatever GPU this machine has)."""

    def __init__(self, m: dict, workspace: str):
        import onnxruntime as ort
        from . import models as MD
        self.m, self.MD = m, MD
        self.sess = ort.InferenceSession(MD.resolve_path(m, workspace), providers=['CPUExecutionProvider'])
        self.input_name = self.sess.get_inputs()[0].name
        self.providers = self.sess.get_providers()

    def predict(self, rgb, meta=None):
        x, back = self.MD.preprocess(rgb, self.m.get('input') or {})
        return self.MD.decode_detections(self.sess.run(None, {self.input_name: x}), self.m, back)


class _Replay:
    """Detections the device made, handed to core.harness.collect frame by frame (by run name and step)."""

    def __init__(self, dets: dict, providers: list):
        self.dets, self.providers = dets, providers

    def predict(self, rgb, meta=None):
        key = f'{meta["run"]}/{meta["step"]}'
        if key not in self.dets:
            raise DeployError(f'the device returned no detections for {key}')
        return self.dets[key]


def ratio_bootstrap(ref: list, eng: list, n_boot: int = 1000, seed: int = 0) -> dict:
    """Paired bootstrap (over drives; over frames with one drive) of the engine's AP, AP50 and AP75 relative to the
    reference's on the same frames: ratio and difference, both resampled together."""
    from . import metrics as M
    by = {f['image_id']: f for f in ref}
    items = [{'image_id': f['image_id'], 'drive': f['drive'], 'a': by[f['image_id']], 'b': f} for f in eng
             if f['image_id'] in by]
    if not items:
        raise DeployError('the engine and the reference share no frames')
    ra = {r['image_id']: r for r in M.coco_match([it['a'] for it in items])}
    rb = {r['image_id']: r for r in M.coco_match([it['b'] for it in items])}
    for it in items:
        it['ra'], it['rb'] = ra[it['image_id']], rb[it['image_id']]

    def fn(its):
        sa = M.coco_summarise([it['ra'] for it in its])
        sb = M.coco_summarise([it['rb'] for it in its])
        out = {}
        for k in ('AP', 'AP50', 'AP75'):
            a, b = sa.get(k), sb.get(k)
            ok = a is not None and b is not None and math.isfinite(a) and math.isfinite(b)
            out[f'{k}_ratio'] = b / a if ok and a > 0 else math.nan
            out[f'{k}_diff'] = b - a if ok else math.nan
        return out
    unit = 'drive' if len({it['drive'] for it in items}) > 1 else 'image'
    res = M.bootstrap(fn, items, n=n_boot, seed=seed, unit=unit)
    return {'unit': res['unit'], 'n_units': res['n_units'], 'n_frames': len(items), 'n_boot': n_boot, 'seed': seed,
            'metrics': res['metrics']}


def _identical(ref: list, eng: list) -> dict:
    by = {f['image_id']: f for f in ref}
    same = sum(1 for f in eng if f['image_id'] in by and f['det'] == by[f['image_id']]['det'])
    return {'frames': len(eng), 'identical_detections': same}


def accuracy(out_dir: str, m: dict, workspace: str, *, n_boot: int = 1000, seed: int = 0, settings=None,
             progress=None) -> dict:
    from . import harness as H
    from . import provenance as PV
    doc = read_doc(out_dir)
    dev = make_device(doc, out_dir, settings)
    say = progress or (lambda f, t: None)
    plan = doc['plan']
    spec = plan['spec']
    common = ['--backend', doc['device']['backend']] + (['--threads', str(plan['threads'])] if plan.get('threads')
                                                        else [])
    names = _built(doc)
    if not names:
        raise DeployError('no engine was built')
    t0 = time.time()
    say(0.02, 'the host\'s ONNX reference')
    ref_dir = os.path.join(out_dir, 'reference')
    os.makedirs(ref_dir, exist_ok=True)
    m = dict(m, kind='detector')
    ref = H.collect(plan['runs'], m, workspace, plan.get('camera'), plan['every'],
                    out_path=os.path.join(ref_dir, 'predictions.jsonl'), runner=_CpuReference(m, workspace))
    ref_frames = ref['frames']
    with open(os.path.join(out_dir, 'bundle', 'eval', 'frames.json'), encoding='utf-8') as fh:
        ev_ids = [f['image_id'] for f in json.load(fh)['frames']]
    if [f['image_id'] for f in ref_frames] != ev_ids:
        raise DeployError('the drives\' frames changed since the build (the evaluation frames no longer match)')
    ref_score = H.score(ref_frames, 'detector', n_boot, seed)
    doc['reference'] = {'providers': ref['providers'], 'what': 'the registered ONNX model on this machine\'s CPU '
                        '(onnxruntime), core.models\' preprocessing and decoder', 'frames': len(ref_frames),
                        'scores': {r['metric']: r for r in ref_score['table'] if r['metric'] in H.HEADLINE},
                        'note': ref_score.get('note'), 'unit': ref_score['unit'], 'n_units': ref_score['n_units']}
    for k, name in enumerate(names):
        say(0.1 + 0.85 * k / len(names), f'{name} on the device')
        rec = doc['engines'][name]
        try:
            info = dev.run(['infer'] + common + ['--engine', f'engines/{name}', '--frames', 'eval', '--spec',
                                                 'spec.json'], 7200.0)
            local = os.path.join(out_dir, 'engines', name, 'dets.jsonl')
            if not dev.get(f'engines/{name}/dets.jsonl', local):
                raise DeployError('the detections did not come back')
            dets = {}
            with open(local, encoding='utf-8') as fh:
                for line in fh:
                    if line.strip():
                        d = json.loads(line)
                        dets[d['image_id']] = d['det']
            os.makedirs(os.path.join(out_dir, 'accuracy', name), exist_ok=True)
            col = H.collect(plan['runs'], m, workspace, plan.get('camera'), plan['every'],
                            out_path=os.path.join(out_dir, 'accuracy', name, 'predictions.jsonl'),
                            runner=_Replay(dets, info.get('providers') or []))
            frames = col['frames']
            sc = H.score(frames, 'detector', n_boot, seed)
            cmp = H.compare(ref_frames, frames, 'detector', n_boot, seed)
            rec['accuracy'] = {'device_run': info, 'scores': {r['metric']: r for r in sc['table']
                                                              if r['metric'] in H.HEADLINE},
                               'note': sc.get('note'), 'unit': sc['unit'], 'n_units': sc['n_units'],
                               'vs_reference': ratio_bootstrap(ref_frames, frames, n_boot, seed),
                               'paired_diff': [r for r in cmp['rows'] if r['metric'] in H.HEADLINE],
                               'regressions': [x for x in cmp['regressions'] if x in H.HEADLINE],
                               'agreement': _identical(ref_frames, frames)}
        except (DeployError, H.HarnessError, OSError, ValueError) as e:
            rec['accuracy'] = {'error': str(e)}
    doc['stages']['accuracy'] = {'finished': PV.utc_now(), 'seconds': time.time() - t0, 'n_boot': n_boot,
                                 'seed': seed}
    save_doc(out_dir, doc)
    say(1.0, 'scored')
    return doc


# ============================================================================================ gate
def validate_gate(g: dict) -> list:
    out = []
    if g.get('schema') != GATE_SCHEMA:
        out.append(f'schema must be {GATE_SCHEMA!r}')
    a = g.get('accuracy')
    if a:
        if a.get('rule', 'ci_lower') not in ('ci_lower', 'point'):
            out.append('accuracy.rule is "ci_lower" or "point"')
        if a.get('metric', 'AP') not in ('AP', 'AP50', 'AP75'):
            out.append('accuracy.metric is AP, AP50 or AP75')
        if not isinstance(a.get('min_relative'), (int, float)):
            out.append('accuracy.min_relative must be a number (e.g. 0.98)')
    if g.get('latency') and g['latency'].get('measure', 'latency') not in ('latency', 'compute'):
        out.append('latency.measure is "latency" or "compute"')
    if g.get('missing', 'incomplete') not in ('incomplete', 'fail'):
        out.append('missing is "incomplete" or "fail"')
    return out


def _crit(status, value=None, limit=None, **evidence) -> dict:
    return {'status': status, 'value': value, 'limit': limit, **evidence}


def evaluate_gate(doc: dict, gate: dict) -> dict:
    """Each engine against the gate: {'engines': {name: {'verdict', 'criteria': {...}}}, 'release_decision'}.
    A criterion is pass, fail, not measured (no evidence: an error, no power rails, too few drives) or not set."""
    problems = validate_gate(gate)
    if problems:
        raise DeployError('the gate: ' + '; '.join(problems))
    out = {}
    for name, e in doc['engines'].items():
        c = {}
        a = gate.get('accuracy')
        if not a:
            c['accuracy'] = _crit('not set')
        else:
            k = a.get('metric', 'AP')
            vs = (e.get('accuracy') or {}).get('vs_reference')
            r = ((vs or {}).get('metrics') or {}).get(f'{k}_ratio') or {}
            need = int(a.get('min_drives') or 0)
            ref = (((doc.get('reference') or {}).get('scores') or {}).get(k) or {}).get('value')
            floor = float(a.get('min_reference') or 0.0)
            if vs and r.get('value') is not None and (ref is None or ref < floor):
                c['accuracy'] = _crit('not measured', r['value'], a['min_relative'], ci=[r['ci_low'], r['ci_high']],
                                      why=f'the reference scores {k} {ref if ref is None else round(ref, 4)}, below '
                                          f'the gate\'s min_reference {floor}: a ratio to so small a number says '
                                          'nothing')
            elif not vs or r.get('value') is None:
                why = (e.get('accuracy') or {}).get('error') or ('the reference scores 0 (no ratio)' if vs else
                                                                 'accuracy was not measured')
                c['accuracy'] = _crit('not measured', why=why, limit=a['min_relative'])
            elif (vs['unit'] == 'drive' and vs['n_units'] < need) or (vs['unit'] != 'drive' and need > 1):
                c['accuracy'] = _crit('not measured', r['value'], a['min_relative'], ci=[r['ci_low'], r['ci_high']],
                                      why=f'{vs["n_units"]} {vs["unit"]}(s): the gate asks for {need} drives or more '
                                          'for a trustworthy interval')
            else:
                v = r['ci_low'] if a.get('rule', 'ci_lower') == 'ci_lower' else r['value']
                c['accuracy'] = _crit('pass' if v is not None and v >= a['min_relative'] else 'fail', r['value'],
                                      a['min_relative'], ci=[r['ci_low'], r['ci_high']], rule=a.get('rule',
                                                                                                    'ci_lower'),
                                      metric=f'{k} relative to the reference', units=f'{vs["n_units"]} {vs["unit"]}s')
        p = e.get('profile') or {}
        lg = gate.get('latency')
        if not lg:
            c['latency'] = _crit('not set')
        else:
            st = p.get('compute_ms' if lg.get('measure') == 'compute' else 'latency_ms')
            if not st:
                c['latency'] = _crit('not measured', limit=lg['max_p95_ms'], why=p.get('error') or 'not profiled')
            else:
                c['latency'] = _crit('pass' if st['p95'] <= lg['max_p95_ms'] else 'fail', st['p95'], lg['max_p95_ms'],
                                     metric=f'p95 {lg.get("measure", "latency")} (ms)', n=st['n'])
        eg = gate.get('energy')
        if not eg:
            c['energy'] = _crit('not set')
        else:
            pw = p.get('power') or {}
            rails = eg.get('rails', 'sum')
            v = pw.get('energy_mj_per_inference') if rails == 'sum' else \
                (sum((pw.get('rails_energy_mj') or {}).get(x, math.nan) for x in rails) if pw.get('available')
                 else None)
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                c['energy'] = _crit('not measured', limit=eg['max_mj_per_frame'],
                                    why=pw.get('why') or p.get('error') or 'no power measurement')
            else:
                c['energy'] = _crit('pass' if v <= eg['max_mj_per_frame'] else 'fail', v, eg['max_mj_per_frame'],
                                    metric='energy per inference (mJ, ESTIMATED: mean power × median latency)',
                                    rails=rails)
        if gate.get('power_mode'):
            pm = (p.get('power_mode') or {}).get('name')
            c['power_mode'] = _crit('not measured' if not pm else ('pass' if pm == gate['power_mode'] else 'fail'),
                                    pm, gate['power_mode'])
        if gate.get('require_locked_clocks'):
            lk = p.get('clocks_locked')
            c['clocks'] = _crit('not measured' if lk is None else ('pass' if lk else 'fail'), lk, True)
        sts = [x['status'] for x in c.values()]
        if 'fail' in sts or ('not measured' in sts and gate.get('missing') == 'fail'):
            verdict = 'fail'
        elif 'not measured' in sts:
            verdict = 'incomplete'
        else:
            verdict = 'pass'
        out[name] = {'verdict': verdict, 'criteria': c}
    sim = bool(doc.get('simulated'))
    return {'gate': gate, 'engines': out, 'simulated': sim, 'release_decision': not sim,
            'note': ('SIMULATED DEVICE: these verdicts test the gate\'s plumbing on this machine\'s CPU numbers; they '
                     'are not release decisions') if sim else None}


def apply_gate(out_dir: str, gate: dict | None = None) -> dict:
    from . import provenance as PV
    doc = read_doc(out_dir)
    g = gate or DEFAULT_GATE
    res = evaluate_gate(doc, g)
    PV.write_json(os.path.join(out_dir, 'gate.json'), g)
    doc['gate'] = res | {'evaluated': PV.utc_now()}
    save_doc(out_dir, doc)
    return doc


# ============================================================================================ listing, report
def deploy_dir(ws: str) -> str:
    return os.path.join(ws, 'deploy')


def list_deployments(ws: str) -> list:
    d = deploy_dir(ws)
    try:
        names = sorted(os.listdir(d), reverse=True)
    except OSError:
        return []
    return [os.path.join(d, n) for n in names if os.path.isfile(os.path.join(d, n, 'deploy.json'))]


def engine_rows(doc: dict) -> list:
    """One row per engine for the page and the report: build, latency, throughput, power, energy, accuracy, verdict."""
    rows = []
    gate = (doc.get('gate') or {}).get('engines') or {}
    for name, e in doc['engines'].items():
        b, p, a = e.get('build') or {}, e.get('profile') or {}, e.get('accuracy') or {}
        lat = p.get('latency_ms') or {}
        pw = p.get('power') or {}
        ap = (a.get('scores') or {}).get('AP') or {}
        rel = (((a.get('vs_reference') or {}).get('metrics') or {}).get('AP_ratio')) or {}
        diff = next((r for r in a.get('paired_diff') or [] if r['metric'] == 'AP'), {})
        rows.append({'name': name, 'precision': e.get('precision'), 'dla_core': e.get('dla_core'),
                     'built': not b.get('error'), 'error': b.get('error') or p.get('error') or a.get('error'),
                     'engine_mib': (b.get('engine_bytes') or 0) / 2 ** 20 or None,
                     'layers': (b.get('layers') or {}).get('by_output_precision') or
                     (b.get('layers') or {}).get('initialiser_types'),
                     'dla_fallback': len(((b.get('log') or {}).get('dla_fallback')) or []) if e.get('dla_core')
                     is not None else None,
                     'median_ms': lat.get('median'), 'p95_ms': lat.get('p95'), 'p99_ms': lat.get('p99'),
                     'throughput_qps': p.get('throughput_qps'), 'power_w': ((pw.get('run') or {}).get('sum_mw') or 0)
                     / 1000.0 or None, 'energy_mj': pw.get('energy_mj_per_inference'),
                     'power_note': None if pw.get('available') else pw.get('why'),
                     'ap': ap.get('value'), 'ap_ci': [ap.get('ci_low'), ap.get('ci_high')],
                     'ap_diff': diff.get('diff'), 'ap_diff_ci': [diff.get('ci_low'), diff.get('ci_high')],
                     'ap_ratio': rel.get('value'), 'ap_ratio_ci': [rel.get('ci_low'), rel.get('ci_high')],
                     'verdict': (gate.get(name) or {}).get('verdict')})
    return rows


def report(src: str):
    """The 'deploy' report (core.reports): the device, the engines and their numbers, the accuracy against the
    reference with intervals, the gate's verdicts with their evidence."""
    from . import reports as R
    d = src if os.path.isdir(src) else os.path.dirname(src)
    pg = R.Page('deploy', 'Deployment', d)
    doc = pg.json(os.path.join(d, 'deploy.json'))
    m = doc.get('model') or {}
    pg.title = f'Deployment of {m.get("name")}' + (' (simulated device)' if doc.get('simulated') else '')
    if doc.get('simulated'):
        pg.note(doc.get('label') or SIMULATED)
    env, dv = doc.get('env') or {}, doc.get('device') or {}
    pm = env.get('power_mode') or {}
    plan = doc.get('plan') or {}
    pg.add(R.kv([('model', R.esc(f'{m.get("name")} ({m.get("licence")}); SHA-256 {m.get("sha256")}')),
                 ('device', R.esc(f'{dv.get("mode")} — {dv.get("where")}; backend {dv.get("backend")}')),
                 ('board', R.esc(env.get('model') or env.get('platform'))),
                 ('software', R.esc(f'L4T {env.get("l4t") or "—"}, CUDA {env.get("cuda") or "—"}, TensorRT '
                                    f'{env.get("tensorrt_python") or "—"} (trtexec {env.get("trtexec") or "none"}), '
                                    f'onnxruntime {env.get("onnxruntime") or "—"}, Python {env.get("python")}')),
                 ('power mode', R.esc(pm.get('name') or pm.get('id') or 'not recorded (not a Jetson)')),
                 ('clocks locked', R.esc({True: 'yes (min = max)', False: 'no', None: 'unknown'}[
                     (env.get('clocks') or {}).get('locked')])),
                 ('power source', R.esc((env.get('power') or {}).get('source') or 'none: power not measured')),
                 ('frames', R.esc(f'{plan.get("eval_frames")} evaluation frames from '
                                  f'{len(plan.get("eval_drives") or [])} drive(s), every {plan.get("every")}; '
                                  f'{plan.get("calib_frames")} calibration frames'))]))
    for n in doc.get('notes') or []:
        pg.note(n)
    rows = engine_rows(doc)
    pg.h2('Engines')
    lay = 'ONNX initialisers by type (simulated)' if doc.get('simulated') else 'layers by output precision'
    pg.add(R.table(['engine', 'size (MiB)', lay, 'median (ms)', 'p95 (ms)', 'p99 (ms)',
                    'throughput (qps)', 'power (W)', 'energy (mJ, est.)', 'error'],
                   [[R.esc(r['name']), R.num(r['engine_mib'], 2), R.esc(json.dumps(r['layers'] or {})),
                     R.num(r['median_ms'], 3), R.num(r['p95_ms'], 3), R.num(r['p99_ms'], 3),
                     R.num(r['throughput_qps'], 1), R.num(r['power_w'], 2), R.num(r['energy_mj'], 2),
                     R.esc(r['error'] or r['power_note'] or '')] for r in rows], numeric=(1, 3, 4, 5, 6, 7, 8)))
    for name, e in doc['engines'].items():
        fb = ((e.get('build') or {}).get('log') or {}).get('dla_fallback')
        if e.get('dla_core') is not None:
            pg.add(f'<h3>{R.esc(name)}: layers that fell back from the DLA to the GPU</h3>',
                   R.table(['layer', 'log line'], [[R.esc(x.get('layer') or '—'), f'<span class="mono">'
                                                    f'{R.esc(x["line"])}</span>'] for x in fb or []])
                   if fb else '<p class="muted">None reported in the build log.</p>')
    ref = doc.get('reference')
    if ref:
        pg.h2('Accuracy on the device against the host\'s ONNX model')
        pg.add(f'<p>{R.esc(ref["what"])}: AP {R.ci(ref["scores"].get("AP"))} on {ref["frames"]} frames; intervals '
               f'over {R.esc(ref["unit"])}s ({ref["n_units"]}).</p>')
        pg.note(ref.get('note'))
        pg.add(R.table(['engine', 'AP [95 %]', 'AP − reference [95 %, paired]', 'AP / reference [95 %, paired]',
                        'identical frames'],
                       [[R.esc(r['name']), _ci3(r['ap'], r['ap_ci']), _ci3(r['ap_diff'], r['ap_diff_ci'], True),
                         _ci3(r['ap_ratio'], r['ap_ratio_ci']),
                         R.esc(_agree(doc['engines'][r['name']]))] for r in rows], numeric=(1, 2, 3)))
        chart = [{'label': 'reference (host ONNX)', 'before': ref['scores'].get('AP')}] + \
            [{'label': r['name'], 'after': {'value': r['ap'], 'ci_low': r['ap_ci'][0], 'ci_high': r['ap_ci'][1]}}
             for r in rows if r['ap'] is not None]
        pg.add(R.legend('reference', 'engine'), R.interval_chart(chart, title='AP', names=('reference', 'engine')))
    g = doc.get('gate')
    if g:
        pg.h2(f'Release gate: {g["gate"].get("name")}')
        pg.note(g.get('note'))
        pg.add(f'<pre class="mono">{R.esc(json.dumps(g["gate"], indent=1))}</pre>')
        trs = []
        for name, v in g['engines'].items():
            for k, c in v['criteria'].items():
                cls = {'pass': 'good', 'fail': 'crit'}.get(c['status'], 'warn')
                ev = {x: y for x, y in c.items() if x not in ('status', 'value', 'limit')}
                trs.append([R.esc(name), f'<span class="{cls}">{R.esc(v["verdict"])}</span>', R.esc(k),
                            f'<span class="{cls}">{R.esc(c["status"])}</span>', R.num(c.get('value'))
                            if isinstance(c.get('value'), float) else R.esc(c.get('value')), R.esc(c.get('limit')),
                            R.esc(json.dumps(ev))])
        pg.add(R.table(['engine', 'verdict', 'criterion', 'status', 'value', 'limit', 'evidence'], trs))
    pg.provenance(doc.get('provenance'), 'the deployment')
    return pg


def _ci3(v, c, signed=False) -> str:
    from . import reports as R
    if v is None:
        return '—'
    if not c or c[0] is None:
        return R.num(v, 3, signed)
    return f'{R.num(v, 3, signed)} [{R.num(c[0], 3, signed)}, {R.num(c[1], 3, signed)}]'


def _agree(e: dict) -> str:
    a = (e.get('accuracy') or {}).get('agreement')
    return f'{a["identical_detections"]} of {a["frames"]}' if a else '—'


# ============================================================================================ the command
def _job_say(em):
    return lambda f, t: em.progress(f, t)


def main_deploy(argv) -> int:
    """`falcon deploy check|build|profile|accuracy|gate|report|all ...` (docs/deploy.md): protocol lines on stdout;
    0 ok, 2 failed."""
    import argparse
    from . import protocol as P
    from . import provenance as PV
    ap = argparse.ArgumentParser(prog='falcon deploy', description='TensorRT engines on the Jetson AGX Orin (or the '
                                                                   'simulated device), profiled and gated')
    sub = ap.add_subparsers(dest='what', required=True)
    ck = sub.add_parser('check', help='what the device has (TensorRT, trtexec, power rails, power mode)')
    ck.add_argument('--device', choices=DEVICES, default='sim')

    def build_args(p):
        p.add_argument('project')
        p.add_argument('--model', required=True)
        p.add_argument('--run', action='append', default=[], required=True, help='an evaluation drive')
        p.add_argument('--calib-run', action='append', default=[], help='a calibration drive (INT8)')
        p.add_argument('--out', '-o', required=True)
        p.add_argument('--device', choices=DEVICES, default='sim')
        p.add_argument('--precision', default=','.join(PRECISIONS))
        p.add_argument('--dla', type=int, default=None, help='also build fp16/int8 on this DLA core (0 or 1)')
        p.add_argument('--calib-count', type=int, default=DEFAULT_CALIB_COUNT)
        p.add_argument('--every', type=int, default=1)
        p.add_argument('--camera', default=None)
        p.add_argument('--workspace-mib', type=int, default=DEFAULT_WORKSPACE_MIB)
        p.add_argument('--threads', type=int, default=0, help='simulated device: onnxruntime threads (0: default)')
        p.add_argument('--seed', type=int, default=0)
        p.add_argument('--backend', default='', help=argparse.SUPPRESS)

    def profile_args(p):
        p.add_argument('--runs', type=int, default=DEFAULT_RUNS)
        p.add_argument('--warmup-ms', type=int, default=DEFAULT_WARMUP_MS)
        p.add_argument('--min-seconds', type=float, default=DEFAULT_MIN_SECONDS)
        p.add_argument('--idle-s', type=float, default=3.0)
        p.add_argument('--power-interval-ms', type=float, default=20.0)

    b = sub.add_parser('build', help='prepare, copy, calibrate and build the engines')
    build_args(b)
    pr = sub.add_parser('profile', help='latency, throughput, power and memory of each engine on the device')
    pr.add_argument('dir')
    profile_args(pr)
    ac = sub.add_parser('accuracy', help='each engine on the evaluation frames, scored against the host reference')
    ac.add_argument('dir')
    ac.add_argument('--boot', type=int, default=1000)
    ac.add_argument('--seed', type=int, default=0)
    gt = sub.add_parser('gate', help='evaluate a gate file per engine (or write the default one)')
    gt.add_argument('dir', nargs='?')
    gt.add_argument('--gate', default='', help='a gate JSON (default: the built-in DEFAULT_GATE)')
    gt.add_argument('--write-default', default='', metavar='FILE')
    rp = sub.add_parser('report', help='the deploy report (HTML)')
    rp.add_argument('dir')
    rp.add_argument('--out', default='', help='default: DIR/report.html')
    al = sub.add_parser('all', help='build, profile, accuracy, gate and report in one go')
    build_args(al)
    profile_args(al)
    al.add_argument('--boot', type=int, default=1000)
    al.add_argument('--gate', default='')
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return P.OK if e.code == 0 else P.FAILED
    em = P.Emitter()
    em.hello('deploy')
    try:
        if a.what == 'check':
            if a.device == 'sim':
                from ..hil import engine_device as T
                env = T.environment()
                env['simulated'] = SIMULATED
            else:
                from .hil import device_settings
                from .settings import Settings
                dev = SshDevice(device_settings(Settings()), 'check')
                dev.put(_TOOL, 'engine_device.py')
                dev.put(_MODELS, 'falcon_models.py')
                env = dev.run(['check'], 120.0)
            print(json.dumps(env, indent=1), flush=True)
            em.result('ok')
            return P.OK
        if a.what == 'gate' and a.write_default:
            PV.write_json(a.write_default, DEFAULT_GATE)
            em.result('ok', path=os.path.abspath(a.write_default))
            return P.OK
        gate = None
        if getattr(a, 'gate', ''):
            gate = PV.read_json(a.gate)
        if a.what in ('build', 'all'):
            from .paths import workspace_for
            from .project import Project, ProjectError
            try:
                proj = Project.load(a.project)
            except (OSError, ProjectError) as e:
                raise DeployError(str(e)) from None
            m = proj.item('models', a.model)
            if m is None:
                raise DeployError(f'no model {a.model!r} in the project')
            ws = workspace_for(a.project)
            prec = [x.strip() for x in a.precision.split(',') if x.strip()]
            bad = [x for x in prec if x not in PRECISIONS]
            if bad:
                raise DeployError(f'unknown precision(s) {bad}: choose from {PRECISIONS}')
            span = (0.0, 0.4) if a.what == 'all' else (0.0, 1.0)
            doc = build(m, ws, a.run, a.out, calib_runs=a.calib_run, device=a.device, precisions=prec, dla=a.dla,
                        calib_count=a.calib_count, every=a.every, camera=a.camera, workspace_mib=a.workspace_mib,
                        threads=a.threads, seed=a.seed, backend=a.backend,
                        progress=lambda f, t: em.progress(span[0] + (span[1] - span[0]) * f, t))
            for name, e in doc['engines'].items():
                em.log(f'{name}: ' + ('built' if not e['build'].get('error') else f'failed: {e["build"]["error"]}'))
            if a.what == 'build':
                em.result('ok', path=os.path.abspath(a.out), engines=_built(doc), simulated=doc['simulated'])
                return P.OK
            profile(a.out, runs=a.runs, warmup_ms=a.warmup_ms, min_seconds=a.min_seconds, idle_s=a.idle_s,
                    power_interval_ms=a.power_interval_ms, progress=lambda f, t: em.progress(0.4 + 0.25 * f, t))
            accuracy(a.out, m, ws, n_boot=a.boot, seed=a.seed,
                     progress=lambda f, t: em.progress(0.65 + 0.3 * f, t))
            doc = apply_gate(a.out, gate)
            from .reports import write_report
            write_report('deploy', a.out, os.path.join(a.out, 'report.html'))
            em.result('ok', path=os.path.abspath(a.out), simulated=doc['simulated'],
                      verdicts={n: v['verdict'] for n, v in doc['gate']['engines'].items()})
            return P.OK
        if a.what == 'profile':
            doc = profile(a.dir, runs=a.runs, warmup_ms=a.warmup_ms, min_seconds=a.min_seconds, idle_s=a.idle_s,
                          power_interval_ms=a.power_interval_ms, progress=_job_say(em))
            em.result('ok', summary={r['name']: {'median_ms': r['median_ms'], 'p95_ms': r['p95_ms'],
                                                 'energy_mj': r['energy_mj']} for r in engine_rows(doc)},
                      simulated=doc['simulated'])
            return P.OK
        if a.what == 'accuracy':
            proj_m, ws = _model_of(read_doc(a.dir), a.dir)
            doc = accuracy(a.dir, proj_m, ws, n_boot=a.boot, seed=a.seed, progress=_job_say(em))
            em.result('ok', summary={r['name']: {'AP': r['ap'], 'AP_ratio': r['ap_ratio']}
                                     for r in engine_rows(doc)}, simulated=doc['simulated'])
            return P.OK
        if a.what == 'gate':
            if not a.dir:
                raise DeployError('DIR (a deploy folder) or --write-default FILE')
            doc = apply_gate(a.dir, gate)
            for n, v in doc['gate']['engines'].items():
                em.log(f'{n}: {v["verdict"]} — ' + ', '.join(f'{k} {c["status"]}' for k, c in v['criteria'].items()))
            em.result('ok', verdicts={n: v['verdict'] for n, v in doc['gate']['engines'].items()},
                      simulated=doc['simulated'], release_decision=doc['gate']['release_decision'])
            return P.OK
        if a.what == 'report':
            from .reports import ReportError, write_report
            try:
                meta = write_report('deploy', a.dir, a.out or os.path.join(a.dir, 'report.html'))
            except ReportError as e:
                raise DeployError(str(e)) from None
            em.result('ok', path=meta['path'])
            return P.OK
    except (DeployError, OSError, ValueError, KeyError) as e:
        print(f'failed: {e}', flush=True)
        return P.FAILED
    return P.FAILED


def _model_of(doc: dict, out_dir: str) -> tuple:
    """The registration a deploy folder was built from (deploy.json keeps it) and a workspace whose models/ folder
    resolves its path: the accuracy stage needs the host's ONNX model, whose path and hash are recorded."""
    from . import provenance as PV
    m = dict(doc['model'])
    path = os.path.expanduser(m['path'])
    bundle = os.path.join(out_dir, 'bundle', 'model.onnx')
    if not os.path.isfile(path) or PV.sha256_file(path) != m['sha256']:
        path = bundle                                    # the copy the device got, byte for byte
    if not os.path.isfile(path) or PV.sha256_file(path) != m['sha256']:
        raise DeployError('the model file is gone or changed since the build (its SHA-256 differs)')
    m['path'] = os.path.abspath(path)
    return m, ''
