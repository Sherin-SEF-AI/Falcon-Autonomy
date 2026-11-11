"""The Triage page and the team workspace in the window, offscreen: making the workspace a team one (its catalogue
moves to this machine), the queue (a new item, assign, investigate, fix, comment), failure modes clustered off the
GUI thread and a cluster turned into a triage item, the bisect's command, a run's comments and owner in Data ▸
Browse, the project's save refused over another member's save (and kept when the user chooses), Settings ▸ Team."""
import json
import os
import subprocess

import pytest

from conftest import FAKE
from test_failclusters import make_workspace
from test_gui import pump, qapp, slot_exceptions, window  # noqa: F401  (fixtures)

pytestmark = pytest.mark.gui


def test_triage_page_end_to_end(window, qapp, tmp_path, monkeypatch):
    from falcon_autonomy.core import team as TM
    from falcon_autonomy.core.paths import cache_dir
    from falcon_autonomy.ui import tasks
    monkeypatch.setenv('FALCON_USER', 'asha')
    page = window.go('triage')
    assert page is window.stack.currentWidget() and page.title == 'Triage'
    ws = window.workspace()
    assert page.make_team.isEnabled()
    page.init_team()
    assert TM.is_team(ws) and not page.make_team.isEnabled()
    assert window.service.catalog.path.startswith(cache_dir())
    assert 'team workspace' in page.team_info.vals['Workspace'].text()
    # the queue
    it = page.new_item(title='Ego brakes late at night')
    assert it['id'] == 'T-1' and page.items.topLevelItemCount() == 1 and page.items.currentItem() is not None
    assert not page.b_fixed.isEnabled() and page.b_inv.isEnabled()
    page._event('assign')
    assert TM.get_item(ws, 'T-1')['assignee'] == 'asha'
    page._event('investigating')
    assert page.b_fixed.isEnabled() and not page.b_verified.isEnabled()
    page._event('fixed')
    assert TM.get_item(ws, 'T-1')['state'] == 'fixed' and page.b_verified.isEnabled() and page.b_reopen.isEnabled()
    assert page.history.count() == 4 and 'investigating → fixed' in page.history.item(3).text()
    page.comments.text.setText('fixed by raising the brake gain')
    assert page.comments.add()['object'] == 'triage:T-1' and page.comments.list.count() == 1
    page.state.setCurrentIndex(page.state.findData('verified'))
    assert page.items.topLevelItemCount() == 0
    page.state.setCurrentIndex(0)
    # failure modes, clustered off the GUI thread
    make_workspace(ws)
    page.build_clusters()
    assert pump(qapp, lambda: page.cl_tree.topLevelItemCount() == 2, timeout=20)
    tasks.wait_all()
    assert '19 failures in 2 clusters' in page.cluster_status.text()
    page.cl_tree.setCurrentItem(page.cl_tree.topLevelItem(0))
    assert page.cl_detail.topLevelItem(0).childCount() >= 2 and page.b_triage_cluster.isEnabled()
    assert not page.b_name.isEnabled()                                   # the assistant is off
    t2 = page.triage_cluster()
    assert t2['id'] == 'T-2' and t2['links'][0].startswith('cluster:C-') and \
        any(x.startswith('run:studies/') for x in t2['links'])
    # the bisect's command
    repo = tmp_path / 'stack'
    repo.mkdir()
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    page.repo.setText(str(repo))
    page.good.setText('v1')
    page.repeats.setValue(3)
    argv = page.bisect_argv('x')
    assert argv[3:] == ['bisect', 'run', str(repo), '--config', 'falcon-ci.yml', '--good', 'v1', '--bad', 'HEAD',
                        '--out', os.path.join(ws, 'bisect', 'x'), '--repeats', '3', '--fake-carla', FAKE]
    os.makedirs(os.path.join(ws, 'bisect', 'b1'))
    with open(os.path.join(ws, 'bisect', 'b1', 'bisect.json'), 'w') as fh:
        json.dump({'regressed': ['rt'], 'bisect': {'rt': {'status': 'found', 'first_bad': {'short': 'abc'}}}}, fh)
    with open(os.path.join(ws, 'bisect', 'b1', 'summary.md'), 'w') as fh:
        fh.write('# Regression bisect\n')
    page.refresh()
    assert page.bisects.topLevelItemCount() == 1 and 'rt: abc' in page.bisects.topLevelItem(0).text(2)
    page.bisects.setCurrentItem(page.bisects.topLevelItem(0))
    assert page.bisect_text.toPlainText().startswith('# Regression bisect')


def test_run_comments_owner_and_a_conflicting_save(window, qapp, monkeypatch):
    from falcon_autonomy.core import team as TM
    from falcon_autonomy.core.project import Project
    monkeypatch.setenv('FALCON_USER', 'asha')
    ws = window.workspace()
    TM.init(ws, 't', TM.identity(window.settings))
    run = os.path.join(ws, 'runs', 'r1')
    os.makedirs(run)
    with open(os.path.join(run, 'run.json'), 'w') as fh:
        json.dump({'status': 'ok', 'name': 'r1', 'started': '2026-11-01T00:00:00Z'}, fh)
    TM.record_owner(ws, run, {'user': 'ravi', 'host': 'ws-2'})
    data = window.go('data')
    data.show_run(run)
    br = data.browse
    assert br.comments.ref == 'run:runs/r1' and br.info.vals['Recorded by'].text() == 'ravi on ws-2'
    br.comments.text.setText('the lead car is hidden by the bus')
    br.comments.add()
    assert TM.comments(ws, 'run:runs/r1')[0]['user'] == 'asha'
    # another member saves the project after this window opened it
    TM.remember(window.project)
    other = Project.load(window.project.path)
    TM.remember(other)
    other.doc['name'] = 'saved by ravi'
    TM.save_project(other, {'user': 'ravi', 'host': 'ws-2'})
    window.project.doc['name'] = 'saved by asha'
    window.project.dirty = True
    asked = []
    monkeypatch.setattr('falcon_autonomy.ui.team.save_conflict', lambda win, e: asked.append(str(e)) or False)
    assert window.save() is False and 'ravi' in asked[0]
    assert Project.load(window.project.path).doc['name'] == 'saved by ravi'
    monkeypatch.setattr('falcon_autonomy.ui.team.save_conflict', lambda win, e: True)
    assert window.save() is True and Project.load(window.project.path).doc['name'] == 'saved by asha'
    # Settings ▸ Team
    st = window.go('settings')
    st.user_name.setText('  Asha  K ')
    st.apply()
    assert window.settings.get('user_name') == 'Asha K' and TM.identity(window.settings)['user'] == 'Asha K'
