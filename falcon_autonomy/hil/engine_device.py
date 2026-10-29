"""Falcon's engine tool on the device (docs/deploy.md): one file for Python 3.10+ and numpy, copied by core.deploy over
SSH to a Jetson AGX Orin (JetPack 6: L4T r36, TensorRT 10, CUDA 12) and run there — or run on this machine as the
**simulated device**. Beside it goes falcon_models.py, a verbatim copy of falcon_autonomy/core/models.py (numpy only
at import): its normalise() turns the host-prepared frames into input tensors and its decode_detections() turns the
engine's outputs into boxes, so the device's detections follow exactly the rules the host's ONNX reference follows.

Backends (--backend):
  tensorrt     trtexec builds the engines (FP32, FP16, INT8; --useDLACore with GPU fallback) and times them; the
               TensorRT Python API (python3-libnvinfer, part of JetPack) runs the INT8 entropy calibration
               (IInt8EntropyCalibrator2) and the accuracy runs, with CUDA memory through libcudart by ctypes (no pycuda,
               no cuda-python). Without the Python API the accuracy run falls back to trtexec per frame
               (--loadInputs / --exportOutput, slow but dependency-free) and INT8 is refused: trtexec alone cannot
               calibrate (without a cache it uses placeholder scales). This backend was written against TensorRT's
               documented API and trtexec's documented flags; it has not run in Falcon's development (no Jetson and no
               TensorRT there) — the first real run is on your Orin.
  onnxruntime  the SIMULATED DEVICE: onnxruntime on the CPU stands in for the engine. Not TensorRT, not the Orin — it
               exercises the pipeline only. fp32 = the ONNX model as it is; fp16 = its float weights rounded to
               float16 and computed in float32 (TensorRT's FP16 kernels also compute in FP16: not reproduced); int8 =
               onnxruntime's static QDQ quantisation with entropy calibration on the same calibration frames.

Commands (each writes its JSON beside its outputs and prints it as the last line of stdout; exit 0, or 2 with
{"error": ...}):
  check       the device: Python, numpy, TensorRT (Python and trtexec), CUDA, L4T, board, power mode (nvpmodel),
              clocks (devfreq and cpufreq, min == max when locked by jetson_clocks), INA3221 rails, temperatures,
              onnxruntime
  calibrate   INT8 calibration from frames → a calibration cache (reused when its key matches)
  build       one engine (precision, DLA core) → build.json, build.log, layers.json
  profile     latency over N timed runs after a warm-up, throughput, power on the INA3221 rails, memory → profile.json
              (raw values; core.deploy computes the statistics)
  infer       the engine over the frames → dets.jsonl (detections in Falcon's classes, image pixels)

Power: the INA3221 monitors through sysfs — hwmon on L4T r35/r36 (…/ina3221/*/hwmon/hwmon*/inN_label, inN_input in mV,
currN_input in mA; power = V × I), the iio interface of L4T r32 (in_powerN_input in mW, rail_name_N) — read by a
thread every --power-interval-ms (default 20 ms); the chip's own update period (hwmon update_interval, ms) is
recorded, and samples faster than it repeat values. Without sysfs access, tegrastats --interval (its own rail figures,
mW) is the fallback. No rails at all (any non-Jetson machine): power is "not available", never estimated.
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import hashlib
import json
import os
import platform
import re
import resource
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
TOOL_VERSION = 1
BACKENDS = ('tensorrt', 'onnxruntime')
PRECISIONS = ('fp32', 'fp16', 'int8')
TRTEXEC_PATHS = ('/usr/src/tensorrt/bin/trtexec', '/usr/local/bin/trtexec', '/usr/bin/trtexec')
SIMULATED = ('SIMULATED DEVICE: onnxruntime on the CPU stands in for the engine — not TensorRT, not the Orin; the '
             'pipeline only')


class ToolError(Exception):
    pass


def models():
    try:
        import falcon_models
    except ImportError as e:
        raise ToolError(f'falcon_models.py (the copy of core/models.py) is not beside this script: {e}') from None
    return falcon_models


def read(path: str, default: str = '') -> str:
    try:
        with open(path, encoding='utf-8', errors='replace') as fh:
            return fh.read().strip().rstrip('\x00')
    except OSError:
        return default


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_json(path: str):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def write_json(path: str, doc):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(doc, fh, indent=1)
    os.replace(tmp, path)


def load_frame(path: str) -> np.ndarray:
    with np.load(path) as z:
        return z['x']


# ============================================================================================ the device
def find_trtexec(given: str = '') -> str:
    for p in ([given] if given else []) + [shutil.which('trtexec') or ''] + list(TRTEXEC_PATHS):
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return ''


def _run_text(cmd: list, timeout: float = 10.0) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        return (r.stdout or '') + (r.stderr or '')
    except (OSError, subprocess.SubprocessError):
        return ''


def power_mode() -> dict:
    """nvpmodel's mode (`nvpmodel -q`, else /var/lib/nvpmodel/status) — what the measurement ran under."""
    out = {'source': None, 'name': None, 'id': None, 'raw': None}
    text = _run_text(['nvpmodel', '-q']) if shutil.which('nvpmodel') else ''
    if text.strip():
        out.update(source='nvpmodel -q', raw=text.strip()[-400:])
        m = re.search(r'NV Power Mode:\s*(\S+)', text)
        if m:
            out['name'] = m.group(1)
        ids = re.findall(r'^\s*(\d+)\s*$', text, re.M)
        if ids:
            out['id'] = int(ids[-1])
        return out
    st = read('/var/lib/nvpmodel/status')
    if st:
        out.update(source='/var/lib/nvpmodel/status', raw=st[-200:])
        m = re.search(r'pmode:(\d+)', st)
        if m:
            out['id'] = int(m.group(1))
    return out


def clocks(root: str = '/sys') -> dict:
    """Current, minimum and maximum frequency of each devfreq device (the GPU, on an Orin) and each CPU core;
    jetson_clocks pins min to max, which is how 'locked' is judged here."""
    dev = {}
    for d in sorted(glob.glob(os.path.join(root, 'class/devfreq/*'))):
        cur, lo, hi = (read(os.path.join(d, f)) for f in ('cur_freq', 'min_freq', 'max_freq'))
        if cur:
            dev[os.path.basename(d)] = {'cur': _int(cur), 'min': _int(lo), 'max': _int(hi)}
    cpus = {}
    for d in sorted(glob.glob(os.path.join(root, 'devices/system/cpu/cpu[0-9]*/cpufreq'))):
        cur, lo, hi = (read(os.path.join(d, f)) for f in ('scaling_cur_freq', 'scaling_min_freq', 'scaling_max_freq'))
        if cur:
            cpus[os.path.basename(os.path.dirname(d))] = {'cur': _int(cur), 'min': _int(lo), 'max': _int(hi)}
    every = list(dev.values()) + list(cpus.values())
    locked = bool(every) and all(v['min'] is not None and v['min'] == v['max'] for v in every)
    return {'devfreq': dev, 'cpufreq': cpus, 'locked': locked if every else None}


def _int(s):
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def temperatures(root: str = '/sys') -> dict:
    out = {}
    for d in sorted(glob.glob(os.path.join(root, 'class/thermal/thermal_zone*'))):
        t = _int(read(os.path.join(d, 'temp')))
        if t is not None:
            out[read(os.path.join(d, 'type')) or os.path.basename(d)] = t / 1000.0
    return out


def mem_available_kib() -> int | None:
    for line in read('/proc/meminfo').splitlines():
        if line.startswith('MemAvailable:'):
            return _int(line.split()[1])
    return None


def environment(trtexec: str = '', power_root: str = '/sys') -> dict:
    env = {'tool_version': TOOL_VERSION, 'python': platform.python_version(), 'numpy': np.__version__,
           'machine': platform.machine(), 'platform': platform.platform(terse=True),
           'model': read('/proc/device-tree/model') or None, 'l4t': None, 'cuda': None, 'tensorrt_python': None,
           'trtexec': find_trtexec(trtexec) or None, 'onnxruntime': None, 'dla_cores': None,
           'power_mode': power_mode(), 'clocks': clocks(power_root), 'temperatures_c': temperatures(power_root),
           'mem_available_kib': mem_available_kib()}
    rel = read('/etc/nv_tegra_release')
    m = re.search(r'R(\d+)\s*\(release\),\s*REVISION:\s*([\d.]+)', rel)
    if m:
        env['l4t'] = f'{m.group(1)}.{m.group(2)}'
    cv = read('/usr/local/cuda/version.json')
    if cv:
        try:
            env['cuda'] = json.loads(cv)['cuda']['version']
        except (ValueError, KeyError):
            pass
    try:
        import tensorrt as trt
        env['tensorrt_python'] = trt.__version__
        try:
            env['dla_cores'] = int(trt.Builder(trt.Logger(trt.Logger.ERROR)).num_DLA_cores)
        except Exception:                                     # noqa: BLE001 — no GPU context: still report the rest
            pass
    except ImportError:
        pass
    try:
        import onnxruntime as ort
        env['onnxruntime'] = ort.__version__
    except ImportError:
        pass
    p = Power(power_root)
    env['power'] = {'source': p.source, 'rails': [r['label'] for r in p.rails], 'update_interval_ms': p.update_ms}
    return env


# ============================================================================================ power
class Power:
    """The INA3221 rails (module doc): discover(), read() → {label: mW}, and a sampling thread (start / stop)."""

    def __init__(self, root: str = '/sys', interval_ms: float = 20.0, tegrastats: str = ''):
        self.root, self.interval = root, max(float(interval_ms), 1.0) / 1000.0
        self.rails, self.source, self.update_ms = [], None, None
        self.samples: list = []
        self._stop = threading.Event()
        self._thread = None
        self._proc = None
        self._discover(tegrastats)

    def _discover(self, tegrastats):
        for hw in sorted(glob.glob(os.path.join(self.root, 'bus/i2c/drivers/ina3221/*/hwmon/hwmon*'))):
            for lab in sorted(glob.glob(os.path.join(hw, 'in*_label'))):
                k = re.search(r'in(\d+)_label$', lab).group(1)
                name = read(lab)
                v, c = os.path.join(hw, f'in{k}_input'), os.path.join(hw, f'curr{k}_input')
                if name and name.upper() != 'NC' and os.path.exists(v) and os.path.exists(c):
                    self.rails.append({'label': name, 'kind': 'hwmon', 'v': v, 'c': c})
            if self.update_ms is None:
                self.update_ms = _int(read(os.path.join(hw, 'update_interval'))) or None
        if self.rails:
            self.source = 'sysfs hwmon (ina3221): inN_input (mV) × currN_input (mA)'
            return
        for dev in sorted(glob.glob(os.path.join(self.root, 'bus/i2c/drivers/ina3221x/*/iio:device*'))):
            for pw in sorted(glob.glob(os.path.join(dev, 'in_power*_input'))):
                k = re.search(r'in_power(\d+)_input$', pw).group(1)
                name = read(os.path.join(dev, f'rail_name_{k}'))
                if name and name.upper() != 'NC':
                    self.rails.append({'label': name, 'kind': 'iio', 'p': pw})
        if self.rails:
            self.source = 'sysfs iio (ina3221x): in_powerN_input (mW)'
            return
        exe = tegrastats or shutil.which('tegrastats') or ''
        if exe and os.access(exe, os.X_OK):
            self.source = f'tegrastats --interval {int(self.interval * 1000)} (its rail figures, mW)'
            self.rails = [{'label': '(from tegrastats)', 'kind': 'tegrastats', 'exe': exe}]

    @property
    def available(self) -> bool:
        return bool(self.rails)

    def read(self) -> dict:
        out = {}
        for r in self.rails:
            if r['kind'] == 'hwmon':
                v, c = _int(read(r['v'])), _int(read(r['c']))
                if v is not None and c is not None:
                    out[r['label']] = v * c / 1000.0
            elif r['kind'] == 'iio':
                p = _int(read(r['p']))
                if p is not None:
                    out[r['label']] = float(p)
        return out

    def start(self):
        self.samples = []
        self._stop.clear()
        if not self.rails:
            return
        if self.rails[0]['kind'] == 'tegrastats':
            self._proc = subprocess.Popen([self.rails[0]['exe'], '--interval', str(int(self.interval * 1000))],
                                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
            self._thread = threading.Thread(target=self._tegrastats, daemon=True)
        else:
            self._thread = threading.Thread(target=self._sysfs, daemon=True)
        self._thread.start()

    def _sysfs(self):
        nxt = time.monotonic()
        while not self._stop.is_set():
            self.samples.append((time.monotonic_ns(), self.read()))
            nxt += self.interval
            self._stop.wait(max(nxt - time.monotonic(), 0.0))

    def _tegrastats(self):
        for line in self._proc.stdout:
            vals = {k: float(v) for k, v, _ in re.findall(r'(\w+) (\d+)mW/(\d+)mW', line)}
            if vals:
                self.samples.append((time.monotonic_ns(), vals))
            if self._stop.is_set():
                break

    def stop(self) -> list:
        self._stop.set()
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread is not None:
            self._thread.join(5)
        return [{'t_ns': t, 'mw': v} for t, v in self.samples]


# ============================================================================================ CUDA through ctypes
class Cuda:
    """The few libcudart calls an engine needs (cudaMalloc, cudaMemcpy, a stream), by ctypes."""
    H2D, D2H = 1, 2

    def __init__(self):
        names = ['libcudart.so', 'libcudart.so.12'] + sorted(glob.glob('/usr/local/cuda/lib64/libcudart.so*'))
        self.lib = None
        for n in names:
            try:
                self.lib = ctypes.CDLL(n)
                break
            except OSError:
                continue
        if self.lib is None:
            raise ToolError('libcudart (the CUDA runtime) was not found')
        L, vp = self.lib, ctypes.c_void_p
        L.cudaMalloc.argtypes = [ctypes.POINTER(vp), ctypes.c_size_t]
        L.cudaFree.argtypes = [vp]
        L.cudaMemcpy.argtypes = [vp, vp, ctypes.c_size_t, ctypes.c_int]
        L.cudaStreamCreate.argtypes = [ctypes.POINTER(vp)]
        L.cudaStreamSynchronize.argtypes = [vp]
        s = vp()
        self._ok(L.cudaStreamCreate(ctypes.byref(s)), 'cudaStreamCreate')
        self.stream = s.value or 0

    @staticmethod
    def _ok(rc, what):
        if rc != 0:
            raise ToolError(f'{what} failed (CUDA error {rc})')

    def malloc(self, n: int) -> int:
        p = ctypes.c_void_p()
        self._ok(self.lib.cudaMalloc(ctypes.byref(p), max(int(n), 1)), 'cudaMalloc')
        return p.value

    def free(self, p: int):
        self.lib.cudaFree(ctypes.c_void_p(p))

    def htod(self, dptr: int, arr: np.ndarray):
        arr = np.ascontiguousarray(arr)
        self._ok(self.lib.cudaMemcpy(ctypes.c_void_p(dptr), arr.ctypes.data_as(ctypes.c_void_p), arr.nbytes,
                                     self.H2D), 'cudaMemcpy to the device')

    def dtoh(self, arr: np.ndarray, dptr: int):
        self._ok(self.lib.cudaMemcpy(arr.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(dptr), arr.nbytes,
                                     self.D2H), 'cudaMemcpy from the device')

    def sync(self):
        self._ok(self.lib.cudaStreamSynchronize(ctypes.c_void_p(self.stream)), 'cudaStreamSynchronize')


# ============================================================================================ engines
class OrtEngine:
    """The simulated device's engine: an onnxruntime CPU session (module doc)."""

    def __init__(self, path: str, threads: int = 0):
        import onnxruntime as ort
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = int(threads)
        self.sess = ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
        self.input = self.sess.get_inputs()[0].name
        self.names = [o.name for o in self.sess.get_outputs()]
        self.providers = self.sess.get_providers()

    def run(self, x: np.ndarray, names: list | None = None) -> list:
        outs = self.sess.run(None, {self.input: x})
        if names:
            by = dict(zip(self.names, outs))
            return [by[n] for n in names]
        return outs


class TrtEngine:
    """A serialised TensorRT engine run through the Python API (tensor-name API: TensorRT 8.5+ and 10), CUDA memory
    by ctypes; outputs whose shape depends on the data (e.g. an NMS inside the model) get an IOutputAllocator."""

    def __init__(self, path: str, dla_core=None):
        import tensorrt as trt
        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, '')
        self.runtime = trt.Runtime(self.logger)
        if dla_core is not None:
            self.runtime.DLA_core = int(dla_core)
        with open(path, 'rb') as fh:
            self.engine = self.runtime.deserialize_cuda_engine(fh.read())
        if self.engine is None:
            raise ToolError('TensorRT could not deserialise the engine (built by another TensorRT version or device?)')
        self.ctx = self.engine.create_execution_context()
        self.cuda = Cuda()
        self.inputs, self.outputs = [], []
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            dt = np.dtype(trt.nptype(self.engine.get_tensor_dtype(n)))
            (self.inputs if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT else self.outputs).append((n, dt))
        self.buf = {}
        self.alloc = None
        self.providers = ['TensorRT ' + trt.__version__ + (f' DLA {dla_core}' if dla_core is not None else '')]

    def _buffer(self, name, nbytes):
        p, n = self.buf.get(name, (None, 0))
        if p is None or n < nbytes:
            if p is not None:
                self.cuda.free(p)
            p, n = self.cuda.malloc(nbytes), nbytes
            self.buf[name] = (p, n)
        return p

    def run(self, x: np.ndarray, names: list | None = None) -> list:
        trt = self.trt
        name, dt = self.inputs[0]
        x = np.ascontiguousarray(x.astype(dt))
        self.ctx.set_input_shape(name, tuple(x.shape))
        p = self._buffer(name, x.nbytes)
        self.cuda.htod(p, x)
        self.ctx.set_tensor_address(name, p)
        dynamic = []
        for n, odt in self.outputs:
            shape = tuple(self.ctx.get_tensor_shape(n))
            if any(d < 0 for d in shape):
                dynamic.append(n)
                continue
            self.ctx.set_tensor_address(n, self._buffer(n, int(np.prod(shape)) * odt.itemsize))
        if dynamic:
            if self.alloc is None:
                self.alloc = _output_allocator(trt, self.cuda)
            for n in dynamic:
                self.ctx.set_output_allocator(n, self.alloc)
        if not self.ctx.execute_async_v3(self.cuda.stream):
            raise ToolError('TensorRT execute_async_v3 failed')
        self.cuda.sync()
        outs = {}
        for n, odt in self.outputs:
            if n in dynamic:
                shape, ptr = self.alloc.shapes[n], self.alloc.ptr[n]
            else:
                shape, ptr = tuple(self.ctx.get_tensor_shape(n)), self.buf[n][0]
            a = np.empty(shape, odt)
            if a.size:
                self.cuda.dtoh(a, ptr)
            outs[n] = a
        order = names or [n for n, _ in self.outputs]
        return [outs[n] for n in order]


def _output_allocator(trt, cuda):
    class Alloc(trt.IOutputAllocator):
        def __init__(self):
            trt.IOutputAllocator.__init__(self)
            self.ptr, self.size, self.shapes = {}, {}, {}

        def reallocate_output(self, name, memory, size, alignment):
            if self.size.get(name, -1) < size:
                if name in self.ptr:
                    cuda.free(self.ptr[name])
                self.ptr[name], self.size[name] = cuda.malloc(max(int(size), 1)), int(size)
            return self.ptr[name]

        def reallocate_output_async(self, name, memory, size, alignment, stream=None):
            return self.reallocate_output(name, memory, size, alignment)

        def notify_shape(self, name, shape):
            self.shapes[name] = tuple(shape)
    return Alloc()


class TrtexecEngine:
    """The accuracy fallback without the TensorRT Python API: trtexec once per frame with --loadInputs and
    --exportOutput (the engine deserialised every time: slow, dependency-free)."""

    def __init__(self, path: str, exe: str, input_name: str, work: str, dla_core=None):
        self.path, self.exe, self.input, self.work, self.dla = path, exe, input_name, work, dla_core
        self.providers = ['trtexec per frame' + (f' DLA {dla_core}' if dla_core is not None else '')]

    def run(self, x: np.ndarray, names: list | None = None) -> list:
        inp, out = os.path.join(self.work, 'in.bin'), os.path.join(self.work, 'out.json')
        np.ascontiguousarray(x, np.float32).tofile(inp)
        cmd = [self.exe, f'--loadEngine={self.path}', f"--loadInputs='{self.input}':{inp}", f'--exportOutput={out}',
               '--iterations=1', '--warmUp=0', '--duration=0', '--avgRuns=1']
        if self.dla is not None:
            cmd.append(f'--useDLACore={self.dla}')
        r = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if r.returncode != 0:
            raise ToolError(f'trtexec failed on a frame: {(r.stdout + r.stderr)[-400:]}')
        by = {}
        for t in load_json(out):
            dims = [int(d) for d in str(t['dimensions']).split('x') if d != '']
            by[t['name']] = np.asarray(t['values'], np.float32).reshape(dims)
        return [by[n] for n in (names or list(by))]


def open_engine(a, spec: dict):
    meta = load_json(os.path.join(a.engine, 'build.json'))
    if a.backend == 'onnxruntime':
        return OrtEngine(os.path.join(a.engine, meta['engine_file']), a.threads), meta
    dla = meta.get('dla_core')
    path = os.path.join(a.engine, meta['engine_file'])
    try:
        import tensorrt  # noqa: F401
        return TrtEngine(path, dla), meta
    except ImportError:
        exe = find_trtexec(a.trtexec)
        if not exe:
            raise ToolError('neither the TensorRT Python API nor trtexec is available') from None
        return TrtexecEngine(path, exe, spec['input_name'], a.engine, dla), meta


# ============================================================================================ calibrate
def frame_files(d: str, count: int = 0) -> list:
    idx = load_json(os.path.join(d, 'frames.json'))
    files = [os.path.join(d, f['file']) for f in idx['frames']]
    return files[:count] if count else files


def cmd_calibrate(a) -> dict:
    spec = load_json(a.spec)
    files = frame_files(a.frames, a.count)
    if not files:
        raise ToolError('no calibration frames')
    keyfile = a.cache + '.key'
    os.makedirs(os.path.dirname(os.path.abspath(a.cache)), exist_ok=True)
    if a.key and os.path.isfile(a.cache) and read(keyfile) == a.key:
        return {'cache': a.cache, 'reused': True, 'frames': len(files), 'key': a.key, 'backend': a.backend}
    t0 = time.monotonic()
    if a.backend == 'onnxruntime':
        out = _ort_calibrate(a, spec, files)
    else:
        out = _trt_calibrate(a, spec, files)
    with open(keyfile, 'w', encoding='utf-8') as fh:
        fh.write(a.key or '')
    out.update(cache=a.cache, reused=False, frames=len(files), key=a.key, backend=a.backend,
               seconds=time.monotonic() - t0, sha256=sha256(a.cache))
    return out


def _ort_calibrate(a, spec, files) -> dict:
    """onnxruntime's static quantisation (QDQ, int8 activations and weights, entropy calibration); the 'cache' is
    the quantised model itself."""
    from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, \
        quantize_static
    FM = models()

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(files)

        def get_next(self):
            f = next(self.it, None)
            return None if f is None else {spec['input_name']: FM.normalise(load_frame(f), spec['input'])}
    quantize_static(a.onnx, a.cache, Reader(), quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8, calibrate_method=CalibrationMethod.Entropy,
                    extra_options={'CalibMaxIntermediateOutputs': 8})
    return {'kind': 'onnxruntime static QDQ model (entropy calibration)', 'method': 'entropy (onnxruntime)'}


def _trt_calibrate(a, spec, files) -> dict:
    """IInt8EntropyCalibrator2 over the frames while TensorRT builds an INT8 network; the calibration cache it writes
    is what trtexec --calib reads for every INT8 build (GPU or DLA)."""
    try:
        import tensorrt as trt
    except ImportError:
        raise ToolError('INT8 calibration needs the TensorRT Python API (sudo apt install python3-libnvinfer); trtexec '
                        'alone cannot calibrate') from None
    FM = models()
    cuda = Cuda()
    shape = tuple(spec['input_shape'])
    nbytes = int(np.prod(shape)) * 4

    class Calib(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            trt.IInt8EntropyCalibrator2.__init__(self)
            self.i, self.dptr = 0, cuda.malloc(nbytes)

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            if self.i >= len(files):
                return None
            x = FM.normalise(load_frame(files[self.i]), spec['input'])
            self.i += 1
            cuda.htod(self.dptr, np.ascontiguousarray(x, np.float32))
            return [int(self.dptr)]

        def read_calibration_cache(self):
            return None                                   # always calibrate here; reuse is decided by the key

        def write_calibration_cache(self, cache):
            with open(a.cache, 'wb') as fh:
                fh.write(bytes(cache))
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    major = int(trt.__version__.split('.')[0])
    flags = 0 if major >= 10 else 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    with open(a.onnx, 'rb') as fh:
        if not parser.parse(fh.read()):
            raise ToolError('TensorRT could not parse the ONNX model: ' +
                            '; '.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(a.workspace_mib) << 20)
    config.set_flag(trt.BuilderFlag.INT8)
    calib = Calib()
    config.int8_calibrator = calib
    inp = network.get_input(0)
    if any(d < 0 for d in inp.shape):
        prof = builder.create_optimization_profile()
        prof.set_shape(inp.name, shape, shape, shape)
        config.add_optimization_profile(prof)
        config.set_calibration_profile(prof)
    if builder.build_serialized_network(network, config) is None or not os.path.isfile(a.cache):
        raise ToolError('the INT8 calibration build failed (see the log)')
    cuda.free(calib.dptr)
    return {'kind': 'TensorRT calibration cache (IInt8EntropyCalibrator2)', 'method': 'entropy (TensorRT)',
            'tensorrt': trt.__version__}


# ============================================================================================ build
def parse_layers(path: str) -> dict:
    """trtexec --exportLayerInfo (with --profilingVerbosity=detailed): layers per output precision, and the DLA
    subgraphs (TensorRT names them ForeignNode[...])."""
    try:
        doc = load_json(path)
    except (OSError, ValueError):
        return {'available': False}
    layers = doc.get('Layers', []) if isinstance(doc, dict) else doc
    counts, dla, unknown = {}, [], 0
    for L in layers:
        if not isinstance(L, dict):
            unknown += 1
            continue
        name = str(L.get('Name', ''))
        if 'ForeignNode' in name or 'DLA' in str(L.get('LayerType', '')):
            dla.append(name[:200])
        outs = L.get('Outputs') or []
        fmt = str(outs[0].get('Format/Datatype', '')) if outs and isinstance(outs[0], dict) else ''
        m = re.search(r'\b(FP32|FP16|BF16|FP8|Int8|INT8|Int32|INT32|Int64|Bool|UInt8)\b', fmt)
        p = m.group(1).upper() if m else 'unknown'
        counts[p] = counts.get(p, 0) + 1
    return {'available': True, 'n_layers': len(layers), 'by_output_precision': counts, 'dla_subgraphs': dla,
            'names_only': unknown}


DLA_FALLBACK = re.compile(r'(not supported on DLA|Unsupported on DLA|falling back to GPU|'
                          r'Switching this layer\'s device type to GPU|cannot run on DLA)', re.I)


def parse_build_log(text: str) -> dict:
    """What a trtexec build log says that matters here; the wording differs between TensorRT versions, so the raw log
    is kept and only these patterns are read."""
    out = {'tensorrt': None, 'engine_built_s': None, 'memory_lines': [], 'dla_fallback': [], 'warnings': 0}
    m = re.search(r'TensorRT v(\d+)', text)
    if m:
        v = m.group(1)
        out['tensorrt'] = f'{int(v[:-4])}.{int(v[-4:-2])}.{int(v[-2:])}' if len(v) >= 5 else v
    m = re.search(r'Engine built in ([\d.]+) sec', text)
    if m:
        out['engine_built_s'] = float(m.group(1))
    for line in text.splitlines():
        if re.search(r'Total (Host|Device) Persistent Memory|Total Scratch Memory|Total Activation Memory|'
                     r'Engine size|Weights|MemUsageChange', line):
            if len(out['memory_lines']) < 60:
                out['memory_lines'].append(line.strip()[:300])
        if DLA_FALLBACK.search(line):
            q = re.search(r"[Ll]ayer\s+'?\"?([^'\"]+?)'?\"?\s*(?:\(|:|is\b)", line)
            out['dla_fallback'].append({'layer': q.group(1).strip() if q else None, 'line': line.strip()[:300]})
        if '[W]' in line:
            out['warnings'] += 1
    return out


def run_logged(cmd: list, log_path: str, markers=()) -> tuple:
    """Run cmd, its output line by line into log_path; the monotonic time at which each marker's first line was
    read. Returns (exit code, marker times, the log text)."""
    if shutil.which('stdbuf'):
        cmd = ['stdbuf', '-oL', '-eL'] + cmd
    seen = {}
    lines = []
    with open(log_path, 'w', encoding='utf-8') as log:
        log.write('$ ' + ' '.join(cmd) + '\n')
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                             stdin=subprocess.DEVNULL)
        for line in p.stdout:
            t = time.monotonic_ns()
            log.write(line)
            lines.append(line)
            for mk in markers:
                if mk not in seen and mk in line:
                    seen[mk] = t
        rc = p.wait()
    return rc, seen, ''.join(lines)


def cmd_build(a) -> dict:
    os.makedirs(a.out, exist_ok=True)
    if a.precision not in PRECISIONS:
        raise ToolError(f'precision must be one of {PRECISIONS}')
    if a.dla is not None and a.precision == 'fp32':
        raise ToolError('the DLA runs FP16 or INT8 only: choose fp16 or int8 with a DLA core')
    if a.precision == 'int8' and not (a.calib and os.path.isfile(a.calib)):
        raise ToolError('INT8 needs the calibration cache (run calibrate first): no placeholder scales')
    rec = {'precision': a.precision, 'dla_core': a.dla, 'backend': a.backend, 'workspace_mib': a.workspace_mib,
           'onnx_sha256': sha256(a.onnx), 'calibration': a.calib or None, 'tool_version': TOOL_VERSION}
    if a.backend == 'onnxruntime':
        rec.update(_ort_build(a))
    else:
        rec.update(_trt_build(a))
    rec['engine_sha256'] = sha256(os.path.join(a.out, rec['engine_file']))
    rec['engine_bytes'] = os.path.getsize(os.path.join(a.out, rec['engine_file']))
    write_json(os.path.join(a.out, 'build.json'), rec)
    return rec


def _ort_build(a) -> dict:
    """The simulated build: the precision's ONNX model (module doc), loaded once by onnxruntime to prove it runs."""
    import onnx
    from onnx import numpy_helper
    dst = os.path.join(a.out, 'engine.onnx')
    log = []
    t0 = time.monotonic()
    if a.precision == 'fp32':
        shutil.copyfile(a.onnx, dst)
        log.append('fp32: the ONNX model unchanged')
    elif a.precision == 'fp16':
        m = onnx.load(a.onnx)
        n = 0
        for init in m.graph.initializer:
            if init.data_type == onnx.TensorProto.FLOAT:
                w = numpy_helper.to_array(init)
                init.CopyFrom(numpy_helper.from_array(w.astype(np.float16).astype(np.float32), init.name))
                n += 1
        onnx.save(m, dst)
        log.append(f'fp16 (simulated): {n} float initialisers rounded to float16, computed in float32')
    else:
        shutil.copyfile(a.calib, dst)
        log.append('int8 (simulated): onnxruntime static QDQ model from the entropy calibration')
    eng = OrtEngine(dst, a.threads)
    secs = time.monotonic() - t0
    m = onnx.load(dst)
    ops, dtypes = {}, {}
    for nd in m.graph.node:
        ops[nd.op_type] = ops.get(nd.op_type, 0) + 1
    for init in m.graph.initializer:
        k = onnx.TensorProto.DataType.Name(init.data_type)
        dtypes[k] = dtypes.get(k, 0) + 1
    with open(os.path.join(a.out, 'build.log'), 'w', encoding='utf-8') as fh:
        fh.write(SIMULATED + '\n' + '\n'.join(log) + f'\nonnxruntime session created in {secs:.3f} s; providers '
                 f'{eng.providers}\n')
    layers = {'available': True, 'simulated': True, 'n_layers': len(m.graph.node), 'by_op_type': ops,
              'initialiser_types': dtypes, 'quantize_nodes': ops.get('QuantizeLinear', 0),
              'note': 'ONNX nodes of the simulated engine, not TensorRT layers'}
    write_json(os.path.join(a.out, 'layers.json'), layers)
    return {'engine_file': 'engine.onnx', 'simulated': True, 'label': SIMULATED, 'build_seconds': secs,
            'layers': layers, 'dla': 'not available on the simulated device', 'providers': eng.providers,
            'command': None, 'exit_code': 0}


def _trt_build(a) -> dict:
    exe = find_trtexec(a.trtexec)
    if not exe:
        raise ToolError('trtexec was not found (JetPack: /usr/src/tensorrt/bin/trtexec; or pass --trtexec)')
    engine = os.path.join(a.out, 'engine.plan')
    layers = os.path.join(a.out, 'layers.json')
    cmd = [exe, f'--onnx={a.onnx}', f'--saveEngine={engine}', f'--memPoolSize=workspace:{int(a.workspace_mib)}',
           f'--exportLayerInfo={layers}', '--profilingVerbosity=detailed', '--skipInference', '--verbose']
    if a.shapes:
        cmd.append(f'--shapes={a.shapes}')
    if a.precision == 'fp16':
        cmd.append('--fp16')
    elif a.precision == 'int8':
        cmd += ['--int8', '--fp16', f'--calib={a.calib}']
    if a.dla is not None:
        cmd += [f'--useDLACore={int(a.dla)}', '--allowGPUFallback']
    t0 = time.monotonic()
    rc, _, text = run_logged(cmd, os.path.join(a.out, 'build.log'))
    secs = time.monotonic() - t0
    parsed = parse_build_log(text)
    if rc != 0 or not os.path.isfile(engine):
        raise ToolError(f'trtexec failed to build the engine (exit {rc}); the last lines: {text[-600:]}')
    return {'engine_file': 'engine.plan', 'simulated': False, 'build_seconds': secs, 'command': cmd,
            'exit_code': rc, 'log': parsed, 'layers': parse_layers(layers), 'tensorrt': parsed['tensorrt']}


# ============================================================================================ profile
def cmd_profile(a) -> dict:
    spec = load_json(a.spec)
    meta = load_json(os.path.join(a.engine, 'build.json'))
    power = Power(a.power_root, a.power_interval_ms, a.tegrastats)
    rec = {'backend': a.backend, 'precision': meta['precision'], 'dla_core': meta.get('dla_core'),
           'runs_requested': a.runs, 'warmup_ms': a.warmup_ms, 'min_seconds': a.min_seconds,
           'power_source': power.source, 'power_interval_ms': a.power_interval_ms,
           'power_update_interval_ms': power.update_ms, 'power_mode': power_mode(), 'clocks': clocks(a.power_root),
           'temperatures_before_c': temperatures(a.power_root), 'idle_seconds': a.idle_s,
           'input_frame': os.path.basename(a.input)}
    if power.available and a.idle_s > 0:                 # the idle baseline, before the engine is loaded
        power.start()
        time.sleep(a.idle_s)
        rec['idle_samples'] = power.stop()
    mem0 = mem_available_kib()
    if a.backend == 'onnxruntime':
        rec.update(_ort_profile(a, spec, meta, power))
    else:
        rec.update(_trt_profile(a, spec, meta, power))
    rec['mem_available_kib_before'] = mem0
    rec['max_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rec['temperatures_after_c'] = temperatures(a.power_root)
    write_json(os.path.join(a.engine, 'profile.json'), rec)
    return {k: v for k, v in rec.items() if k not in ('latency_ms', 'compute_ms', 'samples', 'idle_samples')} | \
        {'n': len(rec.get('latency_ms') or [])}


def _ort_profile(a, spec, meta, power) -> dict:
    FM = models()
    eng = OrtEngine(os.path.join(a.engine, meta['engine_file']), a.threads)
    x = FM.normalise(load_frame(a.input), spec['input'])
    t_end = time.monotonic() + a.warmup_ms / 1000.0
    warm = 0
    while time.monotonic() < t_end:
        eng.run(x)
        warm += 1
    lat, low = [], []
    power.start()
    t0 = time.monotonic_ns()
    deadline = time.monotonic() + a.min_seconds
    while len(lat) < a.runs or time.monotonic() < deadline:
        s = time.perf_counter_ns()
        eng.run(x)
        lat.append((time.perf_counter_ns() - s) / 1e6)
        if len(lat) % 50 == 0:
            low.append(mem_available_kib())
    t1 = time.monotonic_ns()
    samples = power.stop()
    low = [v for v in low if v]
    return {'simulated': True, 'label': SIMULATED, 'latency_ms': lat, 'compute_ms': None, 'warmup_runs': warm,
            'window_ns': [t0, t1], 'wall_s': (t1 - t0) / 1e9, 'samples': samples, 'threads': a.threads,
            'providers': eng.providers, 'mem_available_kib_min': min(low) if low else None,
            'latency_definition': 'onnxruntime session.run() on this machine\'s CPU, the input already in memory',
            'throughput_qps_reported': None, 'cpu_load_1min': os.getloadavg()[0]}


def _trt_profile(a, spec, meta, power) -> dict:
    exe = find_trtexec(a.trtexec)
    if not exe:
        raise ToolError('trtexec was not found')
    FM = models()
    binf = os.path.join(a.engine, 'input.bin')
    np.ascontiguousarray(FM.normalise(load_frame(a.input), spec['input']), np.float32).tofile(binf)
    times = os.path.join(a.engine, 'times.json')
    cmd = [exe, f'--loadEngine={os.path.join(a.engine, meta["engine_file"])}', f'--iterations={int(a.runs)}',
           f'--warmUp={int(a.warmup_ms)}', f'--duration={int(a.min_seconds)}', '--avgRuns=1', '--useSpinWait',
           f"--loadInputs='{spec['input_name']}':{binf}", f'--exportTimes={times}']
    if meta.get('dla_core') is not None:
        cmd.append(f'--useDLACore={int(meta["dla_core"])}')
    if a.shapes:
        cmd.append(f'--shapes={a.shapes}')
    low = []
    stop = threading.Event()

    def watch_mem():
        while not stop.is_set():
            low.append(mem_available_kib())
            stop.wait(0.2)
    th = threading.Thread(target=watch_mem, daemon=True)
    th.start()
    power.start()
    rc, seen, text = run_logged(cmd, os.path.join(a.engine, 'profile.log'),
                                markers=('Warmup completed', 'Timing trace has'))
    samples = power.stop()
    stop.set()
    th.join(2)
    low = [v for v in low if v]
    if rc != 0:
        raise ToolError(f'trtexec failed to time the engine (exit {rc}): {text[-600:]}')
    try:
        tj = load_json(times)
    except (OSError, ValueError) as e:
        raise ToolError(f'trtexec wrote no timing file: {e}') from None
    lat = [float(t['latencyMs']) for t in tj if 'latencyMs' in t]
    comp = [float(t['computeMs']) for t in tj if 'computeMs' in t]
    m = re.search(r'Throughput:\s*([\d.]+)\s*qps', text)
    w = [seen.get('Warmup completed'), seen.get('Timing trace has')]
    os.remove(binf)
    return {'simulated': False, 'latency_ms': lat, 'compute_ms': comp or None, 'window_ns': w,
            'window_definition': 'from trtexec\'s "Warmup completed" line to its "Timing trace has" line, as read on '
                                 'the device', 'samples': samples, 'command': cmd, 'exit_code': rc,
            'throughput_qps_reported': float(m.group(1)) if m else None, 'log': parse_build_log(text),
            'mem_available_kib_min': min(low) if low else None,
            'latency_definition': 'trtexec latencyMs: H2D + GPU compute + D2H of one inference (computeMs: the GPU '
                                  'compute alone)', 'cpu_load_1min': os.getloadavg()[0]}


# ============================================================================================ infer
def cmd_infer(a) -> dict:
    spec = load_json(a.spec)
    FM = models()
    eng, meta = open_engine(a, spec)
    idx = load_json(os.path.join(a.frames, 'frames.json'))
    out_path = os.path.join(a.engine, 'dets.jsonl')
    names = spec.get('output_names') or None
    n, ms = 0, []
    with open(out_path, 'w', encoding='utf-8') as fh:
        for f in idx['frames']:
            x = FM.normalise(load_frame(os.path.join(a.frames, f['file'])), spec['input'])
            t0 = time.perf_counter_ns()
            outs = eng.run(x, names)
            ms.append((time.perf_counter_ns() - t0) / 1e6)
            dets = FM.decode_detections(outs, spec['model'], f['back'])
            fh.write(json.dumps({'image_id': f['image_id'], 'det': dets, 'ms': ms[-1]}, separators=(',', ':')) + '\n')
            n += 1
    return {'frames': n, 'backend': a.backend, 'precision': meta['precision'], 'dla_core': meta.get('dla_core'),
            'providers': list(eng.providers), 'median_ms': float(np.median(ms)) if ms else None,
            'simulated': a.backend == 'onnxruntime', 'dets_sha256': sha256(out_path)}


# ============================================================================================ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='engine_device.py', description='Falcon\'s engine tool on the device')
    sub = ap.add_subparsers(dest='cmd', required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--backend', choices=BACKENDS, default='tensorrt')
    common.add_argument('--trtexec', default='')
    common.add_argument('--threads', type=int, default=0, help='onnxruntime: intra-op threads (0: its default)')
    common.add_argument('--shapes', default='', help='trtexec --shapes for a model with dynamic input dimensions')
    c = sub.add_parser('check', parents=[common])
    c.add_argument('--power-root', default='/sys')
    c = sub.add_parser('calibrate', parents=[common])
    for k in ('--onnx', '--frames', '--spec', '--cache'):
        c.add_argument(k, required=True)
    c.add_argument('--count', type=int, default=0)
    c.add_argument('--key', default='')
    c.add_argument('--workspace-mib', type=int, default=4096)
    c = sub.add_parser('build', parents=[common])
    c.add_argument('--onnx', required=True)
    c.add_argument('--precision', required=True, choices=PRECISIONS)
    c.add_argument('--calib', default='')
    c.add_argument('--dla', type=int, default=None)
    c.add_argument('--workspace-mib', type=int, default=4096)
    c.add_argument('--out', required=True)
    c = sub.add_parser('profile', parents=[common])
    for k in ('--engine', '--input', '--spec'):
        c.add_argument(k, required=True)
    c.add_argument('--runs', type=int, default=1000)
    c.add_argument('--warmup-ms', type=int, default=2000)
    c.add_argument('--min-seconds', type=float, default=10.0)
    c.add_argument('--idle-s', type=float, default=3.0)
    c.add_argument('--power-interval-ms', type=float, default=20.0)
    c.add_argument('--power-root', default='/sys')
    c.add_argument('--tegrastats', default='')
    c = sub.add_parser('infer', parents=[common])
    for k in ('--engine', '--frames', '--spec'):
        c.add_argument(k, required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == 'check':
            out = environment(a.trtexec, a.power_root)
        elif a.cmd == 'calibrate':
            out = cmd_calibrate(a)
        elif a.cmd == 'build':
            out = cmd_build(a)
        elif a.cmd == 'profile':
            out = cmd_profile(a)
        else:
            out = cmd_infer(a)
    except (ToolError, OSError, ValueError, KeyError) as e:
        print(json.dumps({'error': f'{type(e).__name__}: {e}' if not isinstance(e, ToolError) else str(e)}),
              flush=True)
        return 2
    print(json.dumps(out, default=str), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
