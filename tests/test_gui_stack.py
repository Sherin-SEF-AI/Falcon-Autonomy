"""The Stack testing page offscreen: the topics of the rig and the control in use, a baseline suite run as a job
against the fake carla, its results table and summary, a replay, and the safety-case report from an ODD."""
import json
import os

import pytest

from test_gui import pump, qapp, window  # noqa: F401  (fixtures)

pytestmark = pytest.mark.gui


def test_stack_page_suite_results_and_report(window, qapp, tmp_path):
    from falcon_autonomy.core.jobs import FINAL
    from falcon_autonomy.ui.pages.stack import StackPage
    proj = window.project
    proj.doc['rigs'][0]['sensors'][0].update(width=48, height=32, labels=[])
    proj.save(str(tmp_path / 'gui.falcon.json'))
    page = StackPage(window)
    try:
        topics = [page.topics.topLevelItem(i).text(1) for i in range(page.topics.topLevelItemCount())]
        assert '/falcon/ego/front/image' in topics and topics[-1] == '/control/command/control_cmd'
        page.control.setCurrentIndex(page.control.findData('ackermann'))
        assert page.topics.topLevelItem(page.topics.topLevelItemCount() - 1).text(1) == '/falcon/ego/ackermann_cmd'
        # a baseline suite: the lane follower on one scenario, two seeds
        page.name.setText('gui-baseline')
        page.driver.setCurrentIndex(page.driver.findData('follower'))
        for i in range(page.scn_list.count()):
            it = page.scn_list.item(i)
            it.setCheckState(Qt.CheckState.Checked if it.data(256) == 'lead_brake' else Qt.CheckState.Unchecked)
        page.seeds.setText('0, 1')
        doc = page.suite_doc()
        assert doc['entries'] == [{'scenario': 'lead_brake', 'seeds': [0, 1]}] and doc['stack']['driver'] == 'follower'
        page.run_suite()
        job = next(j for j, k in page.jobs.items() if k == 'suite')
        assert pump(qapp, lambda: job.state in FINAL, timeout=300), job.summary()
        assert job.state == 'done', job.log[-10:]
        pump(qapp, timeout=0.3)
        assert page.results.topLevelItemCount() == 2
        assert page.summary.vals['Runs'].text().startswith('2 finished')
        assert json.loads(page.summary_json())['overall']['runs'] == 2
        # replay the first run
        page.results.setCurrentItem(page.results.topLevelItem(0))
        page.replay()
        rj = next(j for j, k in page.jobs.items() if k == 'replay')
        assert pump(qapp, lambda: rj.state in FINAL, timeout=300) and rj.state == 'done', rj.log[-10:]
        run = page.selected_run()['run']
        rep = json.load(open(run.rstrip('/') + '-replay/replay.json'))
        assert rep['max_m'] == 0.0
        # the safety case from the example ODD
        page.new_odd()
        assert os.path.isfile(page.odd.text())
        page.build_report()
        bj = next(j for j, k in page.jobs.items() if k == 'report')
        assert pump(qapp, lambda: bj.state in FINAL, timeout=120) and bj.state == 'done', bj.log[-10:]
        html = open(os.path.join(page.report_dir(), 'safety-case.html')).read()
        assert 'Not a certification claim' in html and 'Lead vehicle brakes' in html
    finally:
        page.timer.stop()
        page.deleteLater()


from PyQt6.QtCore import Qt  # noqa: E402
