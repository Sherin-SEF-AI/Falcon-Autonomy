"""The cluster in the window (offscreen): the Jobs dock's Node column (hidden until a job is placed elsewhere),
Settings ▸ Compute (nodes, pools, cloud caps and keys), the placement combo, and the red banner for a cloud machine
that could not be terminated."""
import pytest

pytest.importorskip('PyQt6')

from test_gui import pump, qapp, slot_exceptions, window  # noqa: E402,F401  (fixtures)

from falcon_autonomy.core.jobs import JobSpec, Launcher  # noqa: E402
from falcon_autonomy.core.settings import Settings  # noqa: E402


class SlotLauncher(Launcher):
    """A cluster with one slot: the first job waits until free() is called, then runs on ws1/0."""

    def __init__(self):
        self.free = False

    def can_start(self, job):
        return '' if self.free else 'waiting for a free slot on the cluster (pool gpu)'

    def start(self, job, queue):
        job.node = 'ws1/0'

        class H:
            def cancel(self_):
                queue.on_exit(job, 3)

            def kill(self_):
                queue.on_exit(job, 3)

            def send(self_, cmd):
                return False
        return H()


def test_the_jobs_dock_shows_where_a_job_runs(window, qapp):
    dock = window.jobs_dock.widget()
    assert dock.tree.isColumnHidden(dock.NODE) and dock.COLS[dock.NODE] == 'Node'
    window.queue.remote = la = SlotLauncher()
    job = window.queue.submit(JobSpec('drive', 'remote drive', ['x'], placement='cluster:gpu', gid='g1'))
    pump(qapp, lambda: job.id in dock.items, 2)
    it = dock.items[job.id]
    assert not dock.tree.isColumnHidden(dock.NODE)
    assert job.state == 'waiting' and it.text(dock.NODE) == 'pool gpu' and 'job g1' in it.toolTip(dock.NODE)
    la.free = True
    window.queue.poll()
    pump(qapp, lambda: it.text(dock.NODE) == 'ws1/0', 2)
    assert job.state == 'running' and it.text(dock.NODE) == 'ws1/0'
    window.queue.cancel(job)
    pump(qapp, lambda: job.state == 'cancelled', 2)


class _Page:
    def __init__(self, win):
        self.win = win


def test_settings_compute(qapp, tmp_path, monkeypatch):
    from falcon_autonomy.assistant import keystore
    from falcon_autonomy.cluster import nodes as N
    from falcon_autonomy.ui.settings_compute import NodeDialog, section
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
    monkeypatch.setattr(keystore, '_keyring', lambda: None)
    st = Settings(str(tmp_path / 'settings.json'))

    class Win:
        settings = st
    sec = section(_Page(Win()))
    assert sec.table.rowCount() == 0 and sec.pools.rowCount() == 0
    key = tmp_path / 'id_ed25519'
    key.write_text('k')
    assert sec.add_node({'name': 'ws1', 'kind': 'ssh', 'host': 'ws1.lan', 'ssh_user': 'jo', 'ssh_key': str(key),
                         'slots': [0, 1], 'pool': 'gpu'})
    assert sec.add_node({'name': 'here', 'kind': 'local', 'slots': [3]})
    assert [sec.table.item(r, 0).text() for r in range(sec.table.rowCount())] == ['ws1', 'here']
    assert sec.table.item(0, 3).text() == '2' and sec.table.item(0, 4).text() == 'ws1.lan'
    pools = {sec.pools.item(r, 0).text(): r for r in range(sec.pools.rowCount())}
    assert set(pools) == {'gpu', 'default'}
    sec.pools.cellWidget(pools['gpu'], 2).setValue(1)
    sec.provider.setCurrentIndex(sec.provider.findData('lambda'))
    sec.image.setText('registry.example/falcon-worker:4.0')
    sec.per_burst.setValue(5.0)
    sec.per_month.setValue(4.0)
    assert 'monthly cap is below' in sec.apply(st)
    sec.per_month.setValue(40.0)
    assert sec.apply(st) == ''
    st.save()
    back = Settings(str(tmp_path / 'settings.json'))
    assert back.get('cloud')['provider'] == 'lambda' and back.get('cloud')['max_usd_per_burst'] == 5.0
    assert back.get('cloud')['image'] == 'registry.example/falcon-worker:4.0'
    assert N.load(back)['pools']['gpu']['max_running'] == 1 and len(N.load(back)['nodes']) == 2
    e, state = sec.keys['runpod']
    assert state.text() == 'not set'
    e.setText('rpa_abc123')
    sec.set_key('runpod')
    assert e.text() == '' and 'stored in' in state.text() and 'rpa_abc123' not in state.text()
    assert keystore.KeyStore(name='runpod_api_key').get() == 'rpa_abc123'
    assert 'rpa_abc123' not in open(tmp_path / 'settings.json').read()
    sec.remove_key('runpod')
    assert state.text() == 'not set'
    sec.table.selectRow(1)
    sec.remove_node()
    assert sec.table.rowCount() == 1
    d = NodeDialog()
    d.kind.setCurrentText('slurm')
    assert not d.workdir.isHidden() and d.slots.isHidden() and d.image.isHidden()
    d.name.setText('hpc')
    d.workdir.setText('relative/path')
    with pytest.raises(N.NodeError, match='workdir'):
        N.validate(d.values())


def test_the_placement_combo(qapp, tmp_path):
    from falcon_autonomy.ui.placement import PlacementCombo
    st = Settings(str(tmp_path / 'settings.json'))
    c = PlacementCombo(st)
    assert c.count() == 1 and not c.isEnabled() and c.placement() == 'local' and c.argv() == []
    st.set('cluster', {'nodes': [{'name': 'a', 'kind': 'local', 'slots': [0, 1], 'pool': 'gpu'},
                                 {'name': 'b', 'kind': 'local', 'slots': [2]}], 'pools': {}})
    c.reload()
    assert [c.itemData(i) for i in range(c.count())] == ['local', 'cluster', 'cluster:default', 'cluster:gpu']
    assert c.itemText(1) == 'Cluster (3 slots)' and c.isEnabled()
    c.setCurrentIndex(3)
    assert c.argv() == ['--on', 'cluster:gpu']
    c.reload()
    assert c.placement() == 'cluster:gpu'                    # the choice survives a reload


def test_the_red_banner_for_a_machine_not_terminated(window, qapp, tmp_path, monkeypatch):
    from falcon_autonomy.cluster import cloud as C
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
    b = window.cloud_banner
    b.refresh()
    assert b.isHidden()
    C.save_state({'burst': 'b1', 'provider': 'lambda', 'instance_id': 'i-9', 'state': 'terminate_failed',
                  'terminate_error': 'HTTP 500'})
    C._alerts()
    try:
        b.refresh()
        assert not b.isHidden() and 'could not be terminated' in b.text.text() and 'lambda i-9' in b.text.text()
    finally:
        C.save_state({'burst': 'b1', 'provider': 'lambda', 'instance_id': 'i-9', 'state': 'terminated'})
        C._alerts()
    b.refresh()
    assert b.isHidden() and C.alert() is None
