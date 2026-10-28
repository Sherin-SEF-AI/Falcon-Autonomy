# PyInstaller spec for Falcon Autonomy (Linux x86_64): `packaging/build.sh` runs it.
# One folder bundle (dist/falcon/): the app, its CLI and the workers' sources as data files (real workers run in
# CARLA's own Python with these sources on PYTHONPATH; sim.server.falcon_source finds them in the bundle).
# CARLA itself is not bundled: the app finds it (Settings, the first-run wizard).
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, '..'))
PKG = os.path.join(ROOT, 'falcon_autonomy')


def sources():
    """Every file of the package (sources, schemas, fonts, icons) at its place under falcon_autonomy/."""
    out = []
    for d, dirs, files in os.walk(PKG):
        dirs[:] = [x for x in dirs if x != '__pycache__']
        for f in files:
            if f.endswith(('.pyc', '.pyo')):
                continue
            src = os.path.join(d, f)
            out.append((src, os.path.relpath(d, ROOT)))
    return out


hidden = (collect_submodules('falcon_autonomy') + collect_submodules('pyqtgraph') +
          ['OpenGL.platform.egl', 'OpenGL.platform.glx', 'jsonschema', 'scipy.linalg', 'scipy.stats',
           'scipy.optimize', 'mcap', 'onnxruntime', 'onnx', 'torchvision'])
datas = sources() + collect_data_files('pyqtgraph') + collect_data_files('jsonschema_specifications')
datas += [(os.path.join(ROOT, 'LICENSE'), '.'), (os.path.join(ROOT, 'README.md'), '.'),
          (os.path.join(ROOT, 'CHANGELOG.md'), '.'), (os.path.join(ROOT, 'docs'), 'docs')]

# Packages the system Python has that Falcon does not use (torch's optional compilers, data-science and web stacks):
# left out, or the bundle grows by gigabytes. torch's CUDA libraries (nvidia/*) stay: training on the GPU needs them.
EXCLUDES = ['tkinter', 'matplotlib', 'IPython', 'pytest', 'notebook', 'tensorflow', 'jax', 'triton', 'polars',
            '_polars_runtime_32', 'llvmlite', 'numba', 'transformers', 'tokenizers', 'hf_xet', 'huggingface_hub',
            'pandas', 'curl_cffi', 'uvloop', 'aiohttp', 'yt_dlp', 'websockets', 'requests',
            'torch.utils.tensorboard', 'sklearn', 'skimage',
            'brotli', 'mutagen', 'secretstorage', 'Cryptodome', 'zmq', 'tornado', 'nbformat']

a = Analysis([os.path.join(SPECPATH, 'entry.py')], pathex=[ROOT], binaries=[], datas=datas, hiddenimports=hidden,
             hookspath=[], runtime_hooks=[], excludes=EXCLUDES,
             noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='falcon', debug=False, strip=False, upx=False,
          console=True)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='falcon')
