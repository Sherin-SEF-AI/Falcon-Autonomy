"""The Replay page offscreen (ui.pages.replay): import a log through the page, see its placement and trajectories,
edit changes, replay the original and the changed log as jobs against the fake carla, and compare them."""
import json
import os

import numpy as np
import pytest

pytestmark = pytest.mark.gui

from test_gui import pump, qapp, slot_exceptions, window  # noqa: E402,F401


def _straight_log(path):
    from falcon_autonomy.core.logreplay import Log, Track
    n = 81
    t = np.arange(n) * 0.1
    ego = Track('ego', 'car', t, -40 + 8 * t, np.zeros(n), np.zeros(n), np.zeros(n))
    lead = Track('lead', 'car', t, -15 + 7 * t, np.zeros(n), np.full(n, 0.75), np.zeros(n), (4.5, 1.9, 1.5))
    Log(ego, [lead], {'name': 'straight', 'map': {'kind': 'carla', 'name': 'Town10HD_Opt'},
                      'source': {'format': 'falcon'}}).save(path)


def test_replay_page_import_changes_runs_and_compare(window, qapp, tmp_path):
    from falcon_autonomy.core.jobs import FINAL
    from falcon_autonomy.ui.pages.replay import ReplayPage
    page = ReplayPage(window)
    window.stack.addWidget(page)
    window.stack.setCurrentWidget(page)
    src = tmp_path / 'src_run'
    _straight_log(str(tmp_path / 'straight'))
    # an import through the page: a ROS 2 bag (written by core.mcap_bag), its town given
    from falcon_autonomy.core import mcap_bag as B
    bag = B.BagWriter(str(src))
    bag.add_topic('/tf', 'tf2_msgs/msg/TFMessage')
    for k in range(60):
        ns = 1_700_000_000_000_000_000 + k * 100_000_000
        bag.write('/tf', ns, {'transforms': [B.transform_stamped(np.array([[1, 0, 0, -40 + 0.8 * k], [0, 1, 0, 0],
                                                                           [0, 0, 1, 0], [0, 0, 0, 1.0]]), ns,
                                                                 'map', 'base_link')]})
    bag.close()
    page.path.setText(str(src))
    page.town.setText('Town10HD_Opt')
    page.do_import()
    assert pump(qapp, lambda: page.log is not None and page.log.meta['source']['format'] == 'ros2', 20)
    assert page.logs.count() == 1 and page.info.vals['Map'].text().startswith('carla Town10HD_Opt')
    assert page.plan is not None and page.plan['placement']['mode'] == 'matched'
    # the straight log with an actor, chosen from the workspace's logs
    os.makedirs(page.logs_dir(), exist_ok=True)
    os.replace(str(tmp_path / 'straight'), os.path.join(page.logs_dir(), 'straight'))
    page.refresh()
    page.logs.setCurrentIndex(page.logs.findText('straight'))
    assert page.log.meta['name'] == 'straight' and page.actor.count() == 2
    assert page.match.vals['Placement'].text() == 'matched'
    page.check_map()
    assert pump(qapp, lambda: 'measuring' not in page.match.vals['Road match'].text(), 20)
    assert 'on the server' in page.match.vals['Road match'].text()        # no lanes offline for a town
    # changes: a pedestrian from the form, a refused one, a weather change
    page.op.setCurrentIndex(page.op.findData('add_pedestrian'))
    page.value.setValue(4.0)
    page.add_from_form()
    assert page.changes == [{'op': 'add_pedestrian', 'id': 'pedestrian_1', 'at_ego_s': 4.0, 'from': 'right'}]
    assert not page.add_change({'op': 'remove', 'actor': 'nobody'}) and 'refused' in page.note.text()
    assert page.add_change({'op': 'weather', 'preset': 'HardRainNoon'})
    assert any(a['id'] == 'pedestrian_1' for a in page.plan['actors'])
    page.slider.setValue(40)
    page.view.grab()                                                       # paints without error
    page.change_list.setCurrentItem(page.change_list.topLevelItem(1))
    page.remove_selected()
    assert [c['op'] for c in page.changes] == ['add_pedestrian']
    # the original and the changed replay as jobs (the fake carla), then the comparison
    page.run('original')
    page.run('changed')
    assert pump(qapp, lambda: all(j.state in FINAL for j, _ in page.jobs.values()), 120)
    assert all(j.state == 'done' or j.exit_code == 0 for j, _ in page.jobs.values()), \
        [(j.state, j.exit_code, j.log[-5:]) for j, _ in page.jobs.values()]
    page.do_compare()
    assert pump(qapp, lambda: page.comparison is not None, 30)
    rows = {page.table.topLevelItem(i).text(0): page.table.topLevelItem(i).text(3)
            for i in range(page.table.topLevelItemCount())}
    assert rows['Collision (0/1)'] in ('0.000', '1.000') and 'deterministic' in page.note.text()
    _, changed_dir = page.jobs['changed']
    rp = json.load(open(os.path.join(changed_dir, 'replay.json')))
    assert rp['changes'][0]['op'] == 'add_pedestrian' and rp['map_match']['verdict'] == 'match'
    window.stack.removeWidget(page)
    page.deleteLater()
