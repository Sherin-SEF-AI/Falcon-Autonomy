"""The Hardware page offscreen: the device kept in the user's settings only (never a password, never the project),
copying and checking the agent through stand-in ssh/scp programs, and a session against the fake carla with a
simulated device — the live waterfall, the final numbers and the report."""
import json
import os

import pytest

pytestmark = pytest.mark.gui

from test_gui import pump, qapp, window  # noqa: E402,F401  (fixtures)
from test_hil import fake_ssh  # noqa: E402,F401  (fixture)


@pytest.fixture
def page(window):
    from falcon_autonomy.ui.pages.hardware import SIZES, HardwarePage
    SIZES['192 × 108'] = (192, 108)
    p = HardwarePage(window)
    p.size.addItem('192 × 108')
    p.size.setCurrentText('192 × 108')
    p.resize(1300, 800)
    p.show()
    yield p
    p.timer.stop()
    p.close()
    p.deleteLater()


def test_device_settings_stay_in_the_users_settings(page, window, tmp_path):
    key = tmp_path / 'id_ed25519'
    key.write_text('x')
    page.remote_rb.setChecked(True)
    page.address.setText('orin.local')
    page.ssh_user.setText('nvidia')
    page.ssh_key.setText(str(key))
    page.save_device()
    dev = window.settings.get('hil_device')
    assert dev['address'] == 'orin.local' and dev['ssh_key'] == str(key) and dev['port'] == 7710
    text = open(window.settings.path).read()
    assert 'password' not in text.lower()
    assert 'orin.local' not in json.dumps(window.project.doc)
    assert page.check_btn.isEnabled() and page.start_agent_btn.isEnabled()
    page.sim_rb.setChecked(True)
    assert not page.check_btn.isEnabled()


def test_copy_and_check_through_ssh(page, qapp, fake_ssh):
    dev, home, log = fake_ssh
    page.remote_rb.setChecked(True)
    page.address.setText(dev['address'])
    page.ssh_user.setText(dev['ssh_user'])
    page.ssh_key.setText(dev['ssh_key'])
    page.dev_python.setText(dev['python'])
    page.check_device()
    assert pump(qapp, lambda: 'numpy' in page.dev_info.text(), 60), page.dev_info.text()
    assert (home / 'falcon-hil' / 'device_agent.py').is_file() and 'BatchMode=yes' in log.read_text()


def test_session_with_a_simulated_device_in_the_page(page, window, qapp):
    page.fps.setValue(10)
    page.seconds.setValue(3)
    page.camera.setCurrentIndex(page.camera.findData('raw-tcp'))
    page.cam_clock.setText('2, 30, 10')
    page.sim_clock.setText('-750, 20')
    page.start(lidar_overrides={'points_per_second': 20000, 'channels': 16})
    assert page.job is not None and page.stop_btn.isEnabled()
    assert pump(qapp, lambda: page.job.state in ('done', 'failed', 'cancelled'), 120), page.job.log[-10:]
    assert page.job.state == 'done', page.job.log[-10:]
    pump(qapp, timeout=0.3)
    assert any(m.get('t') == 'hil' for m in page.job.other)           # live numbers came during the session
    assert page.fall.rows and page.fall.e2e.get('median') is not None
    assert 'median' in page.stats.vals['End to end'].text() and 'on time' in page.stats.vals['Deadlines'].text()
    assert 'simulated' in page.stats.vals['Device'].text()
    assert os.path.isfile(page.report) and page.open_btn.isEnabled()
    page.fall.grab()                                                   # the waterfall paints
    run = json.load(open(os.path.join(page.job.spec.run_dir, 'run.json')))
    assert run['hil']['clock_offset_check']['within_uncertainty']
    assert run['hil']['sensor_clocks']['camera']['configured']['drift_ppm'] == 30.0
