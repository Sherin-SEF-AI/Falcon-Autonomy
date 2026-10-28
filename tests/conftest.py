"""Test setup: a throw-away config folder, offscreen Qt, the fake carla, and the real-server marker.

The whole suite runs without a GPU or a CARLA server: workers run in this Python against tests/fake_carla. Tests
marked real_server run only with FALCON_REAL_CARLA=1 and a CARLA folder found (they start Falcon's own server on its
own port, never touching another). Run the suite memory-capped:

    systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 python3 -m pytest
"""
import os
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAKE = os.path.join(ROOT, 'tests', 'fake_carla')
_TMP = tempfile.mkdtemp(prefix='falcon-tests-')
# never the user's settings: everything Falcon writes to its config and cache folders goes to a throw-away place
os.environ['XDG_CONFIG_HOME'] = os.path.join(_TMP, 'config')
os.environ['XDG_CACHE_HOME'] = os.path.join(_TMP, 'cache')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def pytest_configure(config):
    config.addinivalue_line('markers', 'real_server: needs a real CARLA 0.9.16 server (FALCON_REAL_CARLA=1)')
    config.addinivalue_line('markers', 'carla_python: needs CARLA\'s own Python with the carla wheel')
    config.addinivalue_line('markers', 'gui: a Qt test (offscreen)')


def _carla_python():
    from falcon_autonomy.sim import server as S
    roots = S.candidate_roots()
    return S.client_python(roots[0]) if roots else ''


def pytest_collection_modifyitems(config, items):
    real = os.environ.get('FALCON_REAL_CARLA') == '1'
    py = _carla_python()
    for it in items:
        if 'real_server' in it.keywords and not real:
            it.add_marker(pytest.mark.skip(reason='real CARLA server tests need FALCON_REAL_CARLA=1'))
        if 'carla_python' in it.keywords and not py:
            it.add_marker(pytest.mark.skip(reason='CARLA\'s Python (a venv beside the CARLA folder) not found'))


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
