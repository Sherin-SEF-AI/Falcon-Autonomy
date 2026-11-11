"""The 4.0 scaffolding: the tests' network guard and temporary folder, settings an older file lacks, the keystore's
other secrets, the project's simulator backend and level-k traffic, study_service, the job spec's placement fields,
optional tabs, and the first cluster and formal tools."""
import http.client
import json
import os
import socket
import tempfile

import pytest

from test_gui import qapp  # noqa: F401 (the GUI tests' application)


# ------------------------------------------------------------------------------------------- tests' own guards
def test_network_refused_but_loopback_allowed():
    from conftest import NetworkBlocked
    with pytest.raises(NetworkBlocked):
        socket.create_connection(('192.0.2.1', 80), 0.2)
    with pytest.raises(NetworkBlocked):
        http.client.HTTPSConnection('example.com', timeout=0.2).connect()
    with pytest.raises(OSError):                  # refused by the guard, an OSError like any network failure
        http.client.HTTPConnection('example.com', timeout=0.2).connect()
    srv = socket.socket()
    srv.bind(('127.0.0.1', 0))
    srv.listen(8)
    try:
        port = srv.getsockname()[1]
        socket.create_connection(('127.0.0.1', port), 1.0).close()
        socket.create_connection(('localhost', port), 1.0).close()
        c = http.client.HTTPConnection('127.0.0.1', port, timeout=1.0)
        c.connect()
        c.close()
    finally:
        srv.close()


@pytest.mark.network
def test_network_marker_lets_a_test_through():
    assert socket.create_connection.__module__ == 'socket'             # the real one (and skipped without the variable)


def test_temporary_files_not_on_tmpfs(tmp_path):
    from conftest import _fs_type
    if _fs_type('/var/tmp') in ('tmpfs', 'ramfs', ''):
        pytest.skip('no on-disk /var/tmp on this machine')
    assert _fs_type(str(tmp_path)) not in ('tmpfs', 'ramfs') and _fs_type(tempfile.gettempdir()) not in ('tmpfs',)
    assert not str(tmp_path).startswith(os.path.expanduser('~') + os.sep)   # results shorten home paths to ~


# ------------------------------------------------------------------------------------------- settings and secrets
def test_settings_new_keys_and_an_older_file(tmp_path):
    from falcon_autonomy.core.settings import DEFAULTS, EXPLAIN, PRIVATE_SETTINGS, Settings
    for k in ('carla_roots', 'client_pythons', 'default_backend', 'profile', 'inference_precision', 'texture_pool_mb',
              'cluster', 'cloud', 'hil_can', 'hil_ptp'):
        assert k in DEFAULTS and k in EXPLAIN, k
    assert set(DEFAULTS) == set(EXPLAIN) and 'cluster' in PRIVATE_SETTINGS
    assert DEFAULTS['cloud'] == {'provider': '', 'max_usd_per_burst': 10.0, 'max_usd_month': 50.0, 'ttl_factor': 1.5}
    assert DEFAULTS['hil_can'] == {'interfaces_allowed': ['vcan*'], 'confirmed': {}}
    assert DEFAULTS['assistant']['provider'] == 'groq'
    from falcon_autonomy.assistant.service import EXTRA_DEFAULTS     # the provider's keys agree in both places
    assert all(DEFAULTS['assistant'][k] == v for k, v in EXTRA_DEFAULTS.items())
    p = tmp_path / 'settings.json'                # a 2.x file: no new keys, an assistant group without provider
    p.write_text(json.dumps({'rpc_port': 2320, 'assistant': {'enabled': True}, 'cloud': {'max_usd_month': 20}}))
    st = Settings(str(p))
    assert st.get('rpc_port') == 2320 and st.get('default_backend') == 'carla-0.9.16'
    assert st.get('assistant')['enabled'] is True and st.get('assistant')['provider'] == 'groq'
    assert st.get('cloud') == dict(DEFAULTS['cloud'], max_usd_month=20)


def test_keystore_other_secrets(tmp_path):
    from falcon_autonomy.assistant.keystore import KeyStore
    from falcon_autonomy.core.settings import Secrets
    sec = Secrets(str(tmp_path / 'secrets.json'))
    a, r = KeyStore(sec, use_keyring=False), KeyStore(sec, use_keyring=False, name='runpod_api_key')
    a.set('gsk_abc')
    r.set('rp_xyz')
    assert a.get() == 'gsk_abc' and r.get() == 'rp_xyz' and sec.get('runpod_api_key') == 'rp_xyz'
    with pytest.raises(ValueError):
        r.set('  ')
    r.clear()
    assert not r.has() and a.has()
    with pytest.raises(ValueError):
        KeyStore(sec, use_keyring=False, name='anything')


# ------------------------------------------------------------------------------------------- the project
def test_project_backend_and_levelk(tmp_path):
    from falcon_autonomy.core.project import (DEFAULT_BACKEND, Project, backend_edit, levelk_params, traffic_model,
                                              validate)
    p = Project.new('B')
    assert p.backend() == DEFAULT_BACKEND and backend_edit(p.doc, DEFAULT_BACKEND) is None
    p.apply(backend_edit(p.doc, 'carla-ue5'))
    assert p.doc['simulator'] == {'backend': 'carla-ue5'} and p.backend() == 'carla-ue5' and not validate(p.doc)
    p.apply(backend_edit(p.doc, DEFAULT_BACKEND))
    assert 'simulator' not in p.doc                               # the default is never written
    p.undo()
    assert p.backend() == 'carla-ue5'
    p.doc['simulator'] = {'backend': 'carla-0.10'}
    assert validate(p.doc)
    del p.doc['simulator']
    tp = p.doc['traffic_profiles'][0]
    assert traffic_model(tp) == 'tm' and levelk_params(tp) == {}
    tp['behaviour'].update(model='levelk', levelk={'car': {'k': 2, 'lambda': 1.5, 'alpha': 0.3}})
    assert not validate(p.doc) and traffic_model(tp) == 'levelk' and levelk_params(tp)['car']['k'] == 2
    tp['behaviour']['model'] = 'idm'
    assert validate(p.doc)


# ------------------------------------------------------------------------------------------- jobs and services
def test_study_service_and_job_spec_fields(tmp_path, fake_carla):
    from falcon_autonomy.core.jobs import JobSpec
    from falcon_autonomy.core.settings import Settings
    from falcon_autonomy.sim.service import CarlaService, study_service
    svc = study_service(Settings(str(tmp_path / 's.json')), str(tmp_path / 'ws'), fake_carla=fake_carla,
                        lock_dir=str(tmp_path / 'locks'))
    try:
        assert isinstance(svc, CarlaService) and svc.fake_carla == fake_carla and svc.workspace.endswith('ws')
        assert svc.queue.carla_lock is not None
    finally:
        svc.shutdown()
    with pytest.raises(ValueError, match='no cluster is set up'):        # placement works now: but no node here
        study_service(None, '', placement='cluster')
    with pytest.raises(ValueError, match='local, cluster or cluster:<pool>'):
        study_service(None, '', placement='elsewhere')
    s = JobSpec('drive', 'd', ['x'])
    assert (s.placement, s.gid, s.inputs, s.outputs, s.interactive) == ('local', '', (), 'all', False)
    s = JobSpec('gym', 'g', ['x'], meta={'interactive': True}, inputs=['a'], outputs=['run.json'])
    assert s.interactive and s.inputs == ('a',) and s.outputs == ('run.json',)


# ------------------------------------------------------------------------------------------- the CLI
def test_lazy_commands_not_in_this_build(capsys, monkeypatch):
    from falcon_autonomy import cli
    # a planted entry, so the test does not depend on which 4.0 workstreams have landed
    monkeypatch.setitem(cli.LAZY, 'levelk', ('core.no_such_module_40', 'main_levelk', 'a missing command'))
    assert cli.main(['levelk', 'check']) == 2 and 'failed: levelk is not in this build' in capsys.readouterr().err
    assert cli.run_lazy('dashcam ingest', 'core.no_such_module_40', 'main_ingest', ['ingest', 'x']) == 2
    assert 'failed: dashcam ingest is not in this build' in capsys.readouterr().err
    assert cli.run_lazy('x', 'core.rules', 'no_such_main', []) == 2
    assert cli.run_lazy('rules', 'core.rules', 'main_rules', ['list']) == 0


# ------------------------------------------------------------------------------------------- assistant tools
def test_cluster_nodes_and_formal_stl_check(tmp_path, monkeypatch):
    from falcon_autonomy.assistant import tools  # noqa: F401 (registers them)
    from falcon_autonomy.assistant.tool_api import ToolContext, call
    from falcon_autonomy.core import sysinfo
    from falcon_autonomy.core.settings import Settings
    st = Settings(str(tmp_path / 's.json'))
    st.set('cluster', {'nodes': [{'name': 'ws2', 'host': '10.0.0.7', 'ssh_user': 'me', 'ssh_key': '/k', 'slots': 2}]})
    monkeypatch.setattr(sysinfo, 'gpus', lambda timeout=5.0: [{'index': 0, 'name': 'GPU', 'memory_total_mb': 16000.0,
                                                               'memory_used_mb': 1.0, 'utilization_pct': 3.0,
                                                               'driver': 'x'}])
    ctx = ToolContext(settings=st)
    r = call(ctx, 'cluster_nodes', {})
    assert r['local']['gpus'][0]['memory_total_mb'] == 16000.0
    assert r['nodes'] == [{'name': 'ws2', 'kind': 'ssh', 'pool': 'default', 'slots': 2}]
    assert '10.0.0.7' not in json.dumps(r)
    r = call(ctx, 'formal_stl_check', {'formula': 'always[0,10] (lead_gap_m >= gap)', 'consts': {'gap': 5}})
    assert r['ok'] and r['signals'] == [{'name': 'lead_gap_m', 'unit': 'm'}] and r['horizon_s']['future'] == 10
    r = call(ctx, 'formal_stl_check', {'formula': 'always[0,10] (lead_gap_m >= '})
    assert not r['ok'] and 'column' in r['error']
    r = call(ctx, 'formal_stl_check', {'formula': 'eventually (warp_speed > 1)'})
    assert not r['ok'] and r['unknown'] == ['warp_speed'] and r.get('horizon_s') is None


# ------------------------------------------------------------------------------------------- optional tabs
def test_optional_tab_skips_what_is_missing(qapp):
    from PyQt6.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget
    from falcon_autonomy.ui import widgets as W
    tabs = QTabWidget()
    assert W.optional_tab(tabs, 'ui.no_such_tab', 'Tab', 'Missing', object()) is None and tabs.count() == 0
    assert W.optional_tab(tabs, 'ui.widgets', 'NoSuchClass', 'Missing') is None and tabs.count() == 0
    w = W.optional_tab(tabs, 'ui.widgets', 'label', 'A note')
    assert isinstance(w, QLabel) and tabs.tabText(0) == 'A note'
    host = QWidget()
    v = QVBoxLayout(host)
    v.addWidget(QLabel('first'))
    v.addStretch(1)
    W.optional_tab(v, 'ui.widgets', 'label', 'Section')
    assert isinstance(v.itemAt(1).widget(), W.Section) and v.itemAt(2).spacerItem() is not None
    from falcon_autonomy.ui.pages import settings as SP
    assert all(isinstance(w, W.Section) for w in SP.extra_sections(None))     # the 4.0 sections this build has
