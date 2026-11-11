"""Where a falsification study's and a failure-rate estimate's runs go, offscreen: Rare cases ▸ Falsify and the
failure-rate panel offer this machine, the cluster and each pool (only this machine, disabled, without a cluster),
queue the study with `--on`, and refuse a scenario made from the dashcam footage anywhere but here, with the reason
shown and nothing queued; `falcon falsify run` takes `--on` and refuses the same before anything is queued."""
import os
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.gui

from test_gui import pump, qapp, slot_exceptions, window  # noqa: E402,F401  (fixtures)

from conftest import FAKE  # noqa: E402
from falcon_autonomy.core import scenarios as SC  # noqa: E402
from falcon_autonomy.core.settings import Settings  # noqa: E402

NODES = {'nodes': [{'name': 'ws1', 'kind': 'local', 'pool': 'gpu', 'slots': [0, 1]},
                   {'name': 'ws2', 'kind': 'local', 'pool': 'cpu', 'slots': [2]}], 'pools': {}}


def _dashcam_scenario() -> dict:
    s = dict(SC.builtin('lead_brake'), id='scn-dashcam-1', name='Near miss from the dashcam')
    s['source'] = {'format': 'dashcam', 'clip': 'clip-0001.mp4'}
    return s


def test_the_falsify_tab_places_its_runs(qapp, tmp_path):
    from falcon_autonomy.ui.falsify_tab import FalsifyTab
    ws = str(tmp_path / 'ws')
    os.makedirs(ws)
    submitted = []

    class Q:
        def submit(self, spec):
            submitted.append(spec)
            return SimpleNamespace(state='queued', summary=lambda: 'queued')
    st = Settings(str(tmp_path / 'settings.json'))
    st.set('cluster', NODES)
    win = SimpleNamespace(project=SimpleNamespace(path=str(tmp_path / 'p.falcon.json'), dirty=False,
                                                  item=lambda section, ident: None),
                          workspace=lambda: ws, queue=Q(), fake_carla=FAKE, save=lambda: True, settings=st)
    scns = [(k, SC.builtin(k)) for k in ('cut_in', 'lead_brake')] + [('scn-dashcam-1', _dashcam_scenario())]
    page = SimpleNamespace(win=win, scenarios=lambda: scns)
    tab = FalsifyTab(page)
    combo = tab.placement
    assert [combo.itemData(i) for i in range(combo.count())] == ['local', 'cluster', 'cluster:cpu', 'cluster:gpu']
    assert combo.itemText(3) == 'Cluster pool gpu (2 slots)' and combo.isEnabled()
    tab.formula.setPlainText('G (pet_s >= 3.5)')
    tab.check()
    combo.setCurrentIndex(combo.findData('cluster:gpu'))
    assert tab.start_btn.isEnabled() and not tab.where.isVisibleTo(tab)
    tab.start()
    spec = submitted[-1]
    assert spec.argv[-2:] == ['--on', 'cluster:gpu'] and spec.title.endswith('on cluster:gpu')
    assert spec.argv[3:6] == ['falsify', 'run', win.project.path]
    tab.job = None
    # a scenario made from the dashcam footage: refused off this machine, with the reason; here it may run
    tab.scenario.setCurrentIndex(tab.scenario.findData('scn-dashcam-1'))
    assert tab.where.isVisibleTo(tab) and 'dashcam' in tab.where.text() and 'never leaves this machine' in \
        tab.where.text()
    assert not tab.start_btn.isEnabled()
    tab.start()
    assert len(submitted) == 1 and tab.kv.vals['State'].text().startswith('not queued: private')
    combo.setCurrentIndex(combo.findData('local'))
    assert not tab.where.isVisibleTo(tab) and tab.start_btn.isEnabled()
    tab.start()
    assert len(submitted) == 2 and '--on' not in submitted[-1].argv
    tab.shutdown()
    # a page without a window's settings: the user's (no cluster here) — only this machine, and disabled
    bare = FalsifyTab(SimpleNamespace(win=SimpleNamespace(project=SimpleNamespace(path='', dirty=False)),
                                      scenarios=lambda: scns))
    assert bare.placement.count() == 1 and not bare.placement.isEnabled() and bare.placement.argv() == []
    bare.shutdown()


def test_the_failure_rate_panel_places_its_runs(window, qapp, tmp_path, monkeypatch):
    window.settings.set('cluster', NODES)
    window.project.path = str(tmp_path / 'p.falcon.json')
    window.project.doc['scenarios'].append(_dashcam_scenario())
    window.save()
    rc = window.go('rarecases')
    rc.refresh()
    sec = rc.rates
    sec.placement.reload()
    submitted = []
    monkeypatch.setattr(window.queue, 'submit', lambda spec: submitted.append(spec) or SimpleNamespace(
        state='queued', summary=lambda: 'queued', spec=spec))
    rc.scenario.setCurrentIndex(rc.scenario.findData('lead_brake'))
    sec.placement.setCurrentIndex(sec.placement.findData('cluster'))
    assert sec.start_btn.isEnabled() and not sec.where.isVisibleTo(sec)
    sec.start()
    assert submitted and submitted[-1].argv[-2:] == ['--on', 'cluster'] and 'on cluster' in submitted[-1].title
    assert submitted[-1].argv[3:6] == ['failrate', 'run', window.project.path]
    sec.job = None
    rc.scenario.setCurrentIndex(rc.scenario.findData('scn-dashcam-1'))
    assert pump(qapp, lambda: sec.where.isVisibleTo(sec), 2)
    assert 'dashcam' in sec.where.label.text() and not sec.start_btn.isEnabled()
    sec.start()
    assert len(submitted) == 1 and sec.kv.vals['State'].text().startswith('not queued: private')
    sec.placement.setCurrentIndex(sec.placement.findData('local'))
    assert not sec.where.isVisibleTo(sec) and sec.start_btn.isEnabled()


def test_falsify_run_takes_on_and_refuses_dashcam_scenarios_off_this_machine(tmp_path, capsys, monkeypatch):
    from falcon_autonomy.core import protocol as P
    from falcon_autonomy.core.project import Project
    from falcon_autonomy.core.stlobjective import main_falsify
    from falcon_autonomy.sim.estimate import main_failrate
    from falcon_autonomy.sim.studio import main_study
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
    p = Project.new('T')
    p.doc['scenarios'].append(_dashcam_scenario())
    p.save(str(tmp_path / 'p.falcon.json'))
    common = ['--scenario', 'scn-dashcam-1', '--fake-carla', FAKE, '--on', 'cluster']
    assert main_falsify(['run', p.path, '--stl', 'G (pet_s >= 3.5)'] + common) == P.REFUSED
    assert main_failrate(['run', p.path] + common) == P.REFUSED
    assert main_study(['sweep', p.path, '--name', 's'] + common) == P.REFUSED
    out = capsys.readouterr().out
    assert out.count('private: a scenario made from an event in the dashcam footage') == 3
    assert not os.path.isdir(os.path.join(p.workspace, 'studies'))           # nothing was started
    assert main_falsify(['run', p.path, '--scenario', 'lead_brake', '--stl', 'G (pet_s >= 3.5)', '--on', 'cluster',
                         '--fake-carla', FAKE]) == P.REFUSED
    assert 'no cluster is set up' in capsys.readouterr().out
