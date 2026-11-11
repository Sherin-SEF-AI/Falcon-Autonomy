"""Test setup: a throw-away config folder, offscreen Qt, the fake carla, and the real-server marker.

The whole suite runs without a GPU or a CARLA server: workers run in this Python against tests/fake_carla. Tests
marked real_server run only with FALCON_REAL_CARLA=1 and a CARLA folder found (they start Falcon's own server on its
own port, never touching another). Run the suite memory-capped:

    systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 python3 -m pytest

Markers gated on the machine (ENV_MARKERS): each runs only with its variable set to 1 — ue5 (FALCON_REAL_UE5), vcan,
v4l2, cluster_real, cloud_paid, orin, nuscenes, models, cuda, network (FALCON_NETWORK). No test reaches the network:
an autouse fixture (_no_network) refuses socket.create_connection and http.client connections to anything but this
machine (loopback, localhost, ::1; Unix sockets are untouched) unless the test is marked network. Worker subprocesses
are outside it (they only talk to the fake carla and local sockets).

Temporary files go on disk, not the /tmp tmpfs: when TMPDIR is unset or on a tmpfs, it becomes /var/tmp/falcon-<uid>
(outside $HOME, so results that shorten home paths to ~ read the same as under /tmp), for this process, pytest's
tmp_path and every subprocess.
"""
import os
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAKE = os.path.join(ROOT, 'tests', 'fake_carla')


def _fs_type(path: str) -> str:
    """The filesystem type of the mount holding path ('' when /proc/self/mounts cannot be read)."""
    best, kind = '', ''
    try:
        real = os.path.realpath(path)
        with open('/proc/self/mounts', encoding='utf-8') as fh:
            for line in fh:
                f = line.split()
                if len(f) >= 3 and (real == f[1] or real.startswith(f[1].rstrip('/') + '/')) and len(f[1]) >= len(best):
                    best, kind = f[1], f[2]
    except OSError:
        return ''
    return kind


def _disk_tmp():
    """TMPDIR on disk: /var/tmp/falcon-<uid> when TMPDIR is unset or on a tmpfs (and /var/tmp is not)."""
    cur = os.environ.get('TMPDIR') or tempfile.gettempdir()
    if _fs_type(cur) not in ('tmpfs', 'ramfs'):
        return
    if not os.path.isdir('/var/tmp') or _fs_type('/var/tmp') in ('tmpfs', 'ramfs', ''):
        return
    d = f'/var/tmp/falcon-{os.getuid()}'
    try:
        os.makedirs(d, mode=0o700, exist_ok=True)
    except OSError:
        return
    os.environ['TMPDIR'] = d
    tempfile.tempdir = d


_disk_tmp()
_TMP = tempfile.mkdtemp(prefix='falcon-tests-')
# never the user's settings: everything Falcon writes to its config and cache folders goes to a throw-away place
os.environ['XDG_CONFIG_HOME'] = os.path.join(_TMP, 'config')
os.environ['XDG_CACHE_HOME'] = os.path.join(_TMP, 'cache')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('FALCON_REDUCE_MOTION', '1')       # no animations: what a test sees is settled (ui.motion)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def pytest_configure(config):
    config.addinivalue_line('markers', 'real_server: needs a real CARLA 0.9.16 server (FALCON_REAL_CARLA=1)')
    config.addinivalue_line('markers', 'carla_python: needs CARLA\'s own Python with the carla wheel')
    config.addinivalue_line('markers', 'gui: a Qt test (offscreen)')
    config.addinivalue_line('markers', 'ros2: needs Docker and the falcon-ros2-bridge:humble image (FALCON_ROS2=1)')
    for name, (var, what) in ENV_MARKERS.items():
        config.addinivalue_line('markers', f'{name}: {what} ({var}=1)')


# 4.0: markers for what only some machines have; a marked test runs only with its variable set to 1
ENV_MARKERS = {
    'ue5': ('FALCON_REAL_UE5', 'needs a real CARLA UE5 (0.10.x) installation'),
    'vcan': ('FALCON_VCAN', 'needs a vcan interface (set up once with sudo)'),
    'v4l2': ('FALCON_V4L2', 'needs a v4l2loopback device'),
    'cluster_real': ('FALCON_CLUSTER_REAL', 'needs real cluster nodes (LAN, Slurm or Kubernetes)'),
    'cloud_paid': ('FALCON_CLOUD_PAID', 'launches paid cloud machines (an approved, capped estimate)'),
    'orin': ('FALCON_ORIN', 'needs the Jetson AGX Orin'),
    'nuscenes': ('FALCON_NUSCENES', 'needs the nuScenes data on disk'),
    'models': ('FALCON_MODELS', 'needs downloaded model weights'),
    'cuda': ('FALCON_CUDA', 'needs a CUDA GPU'),
    'network': ('FALCON_NETWORK', 'reaches the network (otherwise refused by the _no_network fixture)'),
}


def _ros2_image() -> bool:
    """Whether Docker answers and the bridge image is built (asked only when FALCON_ROS2=1)."""
    import shutil
    import subprocess
    if not shutil.which('docker'):
        return False
    try:
        return subprocess.run(['docker', 'image', 'inspect', 'falcon-ros2-bridge:humble'], capture_output=True,
                              timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _carla_python():
    from falcon_autonomy.sim import server as S
    roots = S.candidate_roots()
    return S.client_python(roots[0]) if roots else ''


def pytest_collection_modifyitems(config, items):
    real = os.environ.get('FALCON_REAL_CARLA') == '1'
    ros2 = os.environ.get('FALCON_ROS2') == '1' and _ros2_image()
    py = _carla_python()
    for it in items:
        if 'real_server' in it.keywords and not real:
            it.add_marker(pytest.mark.skip(reason='real CARLA server tests need FALCON_REAL_CARLA=1'))
        if 'carla_python' in it.keywords and not py:
            it.add_marker(pytest.mark.skip(reason='CARLA\'s Python (a venv beside the CARLA folder) not found'))
        for name, (var, what) in ENV_MARKERS.items():
            if name in it.keywords and os.environ.get(var) != '1':
                it.add_marker(pytest.mark.skip(reason=f'{what}: set {var}=1'))
        if 'ros2' in it.keywords and not ros2:
            it.add_marker(pytest.mark.skip(reason='ROS 2 tests need FALCON_ROS2=1, Docker and the '
                                                  'falcon-ros2-bridge:humble image (falcon stack image)'))


class NetworkBlocked(ConnectionRefusedError):
    """A test reached for the network without @pytest.mark.network."""


def _local_host(host) -> bool:
    import ipaddress
    if host is None:
        return False
    h = str(host).strip('[]').split('%')[0].lower()
    if h in ('localhost', 'localhost.localdomain', 'ip6-localhost', ''):
        return True
    try:
        ip = ipaddress.ip_address(h)
    except ValueError:
        return False                           # a name: it would be resolved, so it may leave the machine
    return ip.is_loopback or ip.is_unspecified


@pytest.fixture(autouse=True)
def _no_network(request, monkeypatch):
    """socket.create_connection and http.client connections to anything but this machine are refused (NetworkBlocked,
    an OSError) unless the test is marked network."""
    if request.node.get_closest_marker('network') is not None:
        yield
        return
    import http.client
    import socket
    real_create = socket.create_connection

    def create_connection(address, *a, **k):
        if not _local_host(address[0]):
            raise NetworkBlocked(f'tests do not reach the network: {address[0]} (mark the test network)')
        return real_create(address, *a, **k)

    def guard(real):
        def connect(self):
            host = getattr(self, '_tunnel_host', None) or self.host
            if not (_local_host(self.host) and (host == self.host or _local_host(host))):
                raise NetworkBlocked(f'tests do not reach the network: {host} (mark the test network)')
            return real(self)
        return connect
    monkeypatch.setattr(socket, 'create_connection', create_connection)
    monkeypatch.setattr(http.client.HTTPConnection, 'connect', guard(http.client.HTTPConnection.connect))
    monkeypatch.setattr(http.client.HTTPSConnection, 'connect', guard(http.client.HTTPSConnection.connect))
    yield


@pytest.fixture
def fake_carla():
    return FAKE


@pytest.fixture
def carla_python():
    return _carla_python()


@pytest.fixture
def fake_module(monkeypatch):
    """`import carla` in this process gets the fake, reset."""
    monkeypatch.syspath_prepend(FAKE)
    sys.modules.pop('carla', None)
    import carla
    carla.reset()
    yield carla
    sys.modules.pop('carla', None)


@pytest.fixture
def settings(tmp_path):
    from falcon_autonomy.core.settings import Settings
    return Settings(str(tmp_path / 'settings.json'))


@pytest.fixture
def project(tmp_path):
    """A small project: the default world, a 160 × 90 camera with semantic labels, 3 traffic vehicles."""
    from falcon_autonomy.core.project import Project
    p = Project.new('Test')
    p.doc['rigs'][0]['sensors'][0].update(width=160, height=90)
    p.doc['traffic_profiles'][0]['vehicles'] = 3
    p.save(str(tmp_path / 'test.falcon.json'))
    return p


def worker_env(extra=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHON')}
    env['PYTHONPATH'] = os.pathsep.join([FAKE, ROOT])
    env['PYTHONUNBUFFERED'] = '1'
    env.update(extra or {})
    return env


def pytest_sessionfinish(session, exitstatus):
    """Before the interpreter exits: no background Qt task left running, no top-level window left alive, the event
    queue drained — Qt objects torn down while Python finalises its modules can crash the process at exit."""
    if 'PyQt6.QtWidgets' not in sys.modules:
        return
    from PyQt6.QtCore import QThreadPool
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        return
    QThreadPool.globalInstance().waitForDone(15000)
    for w in app.topLevelWidgets():
        w.close()
        w.deleteLater()
    for _ in range(5):
        app.processEvents()
