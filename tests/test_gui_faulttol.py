"""Stack testing ▸ Fault tolerance offscreen: a campaign's table (mono numerals, the meter against a requirement, the
status as a word), the requirement set from the tab and kept in the settings, the response curve of the selected
cell with its hover text, and a closed-loop campaign started from the tab against the fake carla."""
import pytest

from test_gui import pump, qapp, slot_exceptions, window  # noqa: F401  (fixtures)

pytestmark = pytest.mark.gui


def _campaign(ws):
    from falcon_autonomy.core import faulttol as FT
    cam = {'id': 'cam', 'type': 'camera.pinhole'}
    lid = {'id': 'lid', 'type': 'lidar'}
    cells = []
    for s, f, fn, thr in ((cam, 'cam_yaw', lambda x: 20.0 * x, 5.0), (lid, 'lidar_sector', lambda x: x / 3.6, 10.0),
                          (cam, 'exposure', lambda x: 0.0, 5.0)):
        c = FT._cell(s, f, 'recall_drop', thr, 'points lost')
        c.update(FT.tolerance(fn, thr, 0.0, FT.MODELS[f]['max']))
        cells.append(c)
    cells.append(FT._refused(cam, 'noise', 'noise needs a detector (--model)'))
    FT.write(FT.result(cells, 'offline', 'planted'), FT.campaign_dir(ws, 'planted'))
    return cells


def test_the_table_the_requirement_and_the_curve(window, qapp, tmp_path):
    from falcon_autonomy.ui.faulttol_tab import FaultTolTab
    from falcon_autonomy.ui.pages.stack import StackPage
    from falcon_autonomy.ui.theme import THEME
    window.project.save(str(tmp_path / 'gui.falcon.json'))
    _campaign(window.workspace())
    page = StackPage(window)
    try:
        tab = page.faulttol
        assert isinstance(tab, FaultTolTab)
        tab.refresh()
        assert tab.campaign.currentData() == 'planted' and tab.table.topLevelItemCount() == 4
        rows = {(tab.table.topLevelItem(i).text(0), tab.table.topLevelItem(i).data(0, 256)['fault']):
                tab.table.topLevelItem(i) for i in range(4)}
        yaw = rows[('cam', 'cam_yaw')]
        assert yaw.text(2).endswith(' deg') and 0.23 <= float(yaw.text(2).split()[0]) <= 0.25
        assert yaw.font(2).family() == 'IBM Plex Mono' and yaw.text(6) == 'no requirement'
        assert rows[('cam', 'exposure')].text(3) == '[3, —]'                  # no failure up to the largest tried
        assert rows[('cam', 'noise')].text(6) == 'not measured' and '--model' in rows[('cam', 'noise')].toolTip(6)
        assert tab.fragile.value.text() == 'cam' and 'tested range' in tab.fragile.sub.text()
        assert 'Derived — cam: extrinsic yaw must hold within' in tab.derived_text()
        assert tab.derived.vals['cam'].text().startswith('extrinsic yaw must hold within')
        # a requirement the yaw tolerance does not meet: the word, its colour, the meter, the settings
        tab.table.setCurrentItem(yaw)
        assert tab.req.isEnabled() and tab.req.suffix() == ' deg'
        tab.req.setValue(0.5)
        assert window.settings.get('faulttol_requirements') == {'cam_yaw': 0.5}
        yaw = tab.table.currentItem()
        assert yaw.text(6) == 'not met' and yaw.text(4) == '0.5'
        assert yaw.foreground(6).color().name() == THEME.status('critical')
        assert tab.table.itemWidget(yaw, 5).frac == 1.0 and tab.req_stat.value.text() == '0 of 1'
        assert tab.fragile.sub.text().endswith('of the requirement')
        tab.req.setValue(0.1)
        yaw = tab.table.currentItem()
        assert yaw.text(6) == 'met' and yaw.foreground(6).color().name() == THEME.status('good')
        assert 0.40 <= tab.table.itemWidget(yaw, 5).frac < 0.44 and tab.req_stat.value.text() == '1 of 1'
        # the curve of the selected cell: its points, the threshold, the hover text
        tab.curve.resize(520, 200)
        tab.curve.show()
        tab.curve.grab()
        cell = tab.curve.cell
        assert cell['fault'] == 'cam_yaw' and len(tab.curve._pts) == len(cell['curve']) >= 8
        last = tab.curve._pts[-1]
        assert last[4] and tab.curve.tip(last) == '5 deg: 100 (points lost) — fails'
        assert 'passes' in tab.curve.tip(tab.curve._pts[0])
        tab.req.setValue(0.0)                                              # cleared
        assert window.settings.get('faulttol_requirements') == {} and tab.table.currentItem().text(6) == \
            'no requirement'
        tab.table.setCurrentItem(rows_item(tab, 'noise'))
        tab.curve.grab()
        assert tab.curve._pts == []                                        # a refused cell: its reason, no points
    finally:
        page.timer.stop()
        page.deleteLater()


def rows_item(tab, fault):
    return next(tab.table.topLevelItem(i) for i in range(tab.table.topLevelItemCount())
                if tab.table.topLevelItem(i).data(0, 256)['fault'] == fault)


def test_a_closed_loop_campaign_from_the_tab(window, qapp, tmp_path):
    from falcon_autonomy.core.jobs import FINAL
    from falcon_autonomy.ui.pages.stack import StackPage
    from test_stack import stack_rig
    stack_rig(window.project)
    window.project.doc['rigs'][0]['sensors'] = [s for s in window.project.doc['rigs'][0]['sensors']
                                                if s['type'] in ('gnss', 'imu')]
    window.project.save(str(tmp_path / 'gui.falcon.json'))
    page = StackPage(window)
    try:
        tab = page.faulttol
        tab.start()
        assert 'name' in tab.state.text() and tab.job is None             # no name: nothing queued
        tab.name.setText('gui-loop')
        tab.scenario.setCurrentIndex(tab.scenario.findData('lead_brake'))
        argv = tab.argv()
        assert argv[:2] == ['faulttol', 'run'] and '--closed-loop' in argv and '--fake-carla' in argv
        tab.start()
        assert pump(qapp, lambda: tab.job.state in FINAL, timeout=240), tab.job.summary()
        assert tab.job.state == 'done', tab.job.log[-10:]
        tab.poll()
        assert tab.campaign.currentData() == 'gui-loop' and tab.table.topLevelItemCount() == 1
        it = tab.table.topLevelItem(0)
        assert it.text(0) == 'gnss' and it.data(0, 256)['status'] == 'bracketed' and tab.doc['mode'] == 'closed_loop'
    finally:
        page.timer.stop()
        page.deleteLater()
