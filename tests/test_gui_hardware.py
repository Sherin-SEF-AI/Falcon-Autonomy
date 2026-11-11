"""The Hardware page offscreen: the device kept in the user's settings only (never a password, never the project),
copying and checking the agent through stand-in ssh/scp programs, and a session against the fake carla with a
simulated device — the live waterfall, the final numbers and the report."""
import json
import os

import pytest

pytestmark = pytest.mark.gui

from test_gui import pump, qapp, slot_exceptions, window  # noqa: E402,F401  (fixtures)
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


def test_can_clock_and_v4l2_sections(page, window, monkeypatch):
    """The 4.0 sections: a V4L2 camera's and the CAN loop's config, the allow-list said beside the interface and
    enforced, a real adapter confirmed only with the statement, the clock source passed on."""
    from PyQt6.QtWidgets import QInputDialog
    from falcon_autonomy.hil import canbridge as B
    assert page.can_config() is None and page.v4l2_config() is None
    assert not page.can_if.isEnabled() and not page.v4l2_dev.isEnabled()
    page.camera.setCurrentIndex(page.camera.findData('v4l2'))
    assert page.v4l2_dev.isEnabled() and not page.encoder.isEnabled()
    page.v4l2_fmt.setCurrentText('NV12')
    assert page.v4l2_config()['format'] == 'NV12' and page.v4l2_config()['barcode']
    page.can_on.setChecked(True)
    assert page.can_if.isEnabled()
    page.can_if.setCurrentText('sim:gui')
    page.surface.setCurrentIndex(page.surface.findData('waterlogged'))
    page.mu.setValue(0.35)
    page.clock_src.setCurrentIndex(page.clock_src.findData('ntp-style'))
    cfg = page.build_config(False)
    assert cfg['can']['interface'] == 'sim:gui' and cfg['can']['chassis'] == {'surface': 'waterlogged', 'mu': 0.35}
    assert cfg['camera']['transport'] == 'v4l2' and cfg['camera']['v4l2']['format'] == 'NV12'
    assert cfg['clock_source'] == 'ntp-style'
    assert 'allowed' in page.can_status.text()
    monkeypatch.setattr(B, 'is_virtual', lambda name: False)
    window.settings.set('hil_can', {'interfaces_allowed': ['vcan*', 'can*']})
    page.can_if.setCurrentText('can0')
    assert 'refused' in page.can_status.text() and page.confirm_btn.isEnabled()
    with pytest.raises(ValueError, match='bench network, not a vehicle'):
        page.build_config(False)
    monkeypatch.setattr(QInputDialog, 'getText', staticmethod(lambda *a, **k: ('bench network, not a vehicle', True)))
    page.confirm_adapter()
    assert 'allowed' in page.can_status.text() and window.settings.get('hil_can')['confirmed']['can0']
    assert page.build_config(False)['can']['interface'] == 'can0'


def test_a_can_session_in_the_page(page, window, qapp):
    page.fps.setValue(10)
    page.seconds.setValue(3)
    page.camera.setCurrentIndex(page.camera.findData('raw-tcp'))
    page.can_on.setChecked(True)
    page.can_if.setCurrentText(f'sim:gui-{os.getpid()}')
    page.can_fd.setCurrentIndex(page.can_fd.findData('on'))
    page.start(lidar_overrides={'points_per_second': 20000, 'channels': 16})
    assert page.job is not None
    assert pump(qapp, lambda: page.job.state in ('done', 'failed', 'cancelled'), 120), page.job.log[-10:]
    assert page.job.state == 'done', page.job.log[-10:]
    pump(qapp, timeout=0.3)
    assert 'CAN frame to the simulation' in [r['label'] for r in page.fall.rows]
    can = page.stats.vals['CAN'].text()
    assert 'commands' in can and '(FD)' in can and 'ABORTED' not in can
    assert 'from ptp' in page.stats.vals['Clock offset'].text()          # auto: a simulated device's shared realtime
    page.fall.grab()


def test_settings_hardware_section(window, qapp, monkeypatch):
    from PyQt6.QtWidgets import QInputDialog
    from falcon_autonomy.ui.settings_hardware import section

    class Page:
        win = window
    sec = section(Page())
    st = window.settings
    sec.patterns.setText('vcan*, can1')
    sec.ptp_if.setText('enp1s0')
    sec.source.setCurrentIndex(sec.source.findData('ptp'))
    assert sec.apply(st) == ''
    assert st.get('hil_can')['interfaces_allowed'] == ['vcan*', 'can1']
    assert st.get('hil_ptp') == {'interface': 'enp1s0', 'clock_source': 'ptp'}
    sec.patterns.setText('vcan*; rm -rf /')
    assert 'Not an interface pattern' in sec.apply(st)
    sec.patterns.setText('vcan*')
    sec.ptp_if.setText('bad name')
    assert 'not an interface name' in sec.apply(st)
    answers = iter([('can2', True), ('bench network, not a vehicle', True)])
    monkeypatch.setattr(QInputDialog, 'getText', staticmethod(lambda *a, **k: next(answers)))
    sec.confirm()
    assert 'can2' in sec.confirmed.text() and 'can2' in sec.patterns.text()
    monkeypatch.setattr(QInputDialog, 'getItem', staticmethod(lambda *a, **k: ('can2', True)))
    sec.revoke()
    assert 'none' in sec.confirmed.text()
    sec.deleteLater()
