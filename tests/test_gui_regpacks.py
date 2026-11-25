"""Reports ▸ Regulation packs offscreen: the notice, the pack list, a pack's cases with their clauses and the mark on
the ones that use an unverified value, a pack run from the tab against the fake carla with its results as words and
colours, the checklist, and the exported report."""
import os

import pytest

from test_gui import pump, qapp, slot_exceptions, window  # noqa: F401  (fixtures)

pytestmark = pytest.mark.gui


def _rows(tab):
    return [tab.cases.topLevelItem(i) for i in range(tab.cases.topLevelItemCount())]


def test_the_packs_a_run_and_the_report(window, qapp, tmp_path):
    from falcon_autonomy.core import regpacks as RP
    from falcon_autonomy.core import reports as R
    from falcon_autonomy.core.jobs import FINAL
    from falcon_autonomy.ui.pages.reports import ReportsPage
    from falcon_autonomy.ui.regpack_tab import UNVERIFIED_MARK, RegPackTab
    from falcon_autonomy.ui.theme import THEME
    window.project.doc['rigs'][0]['sensors'][0].update(width=48, height=32, labels=[])
    window.project.save(str(tmp_path / 'gui.falcon.json'))
    page = ReportsPage(window)
    try:
        tab = page.regpacks
        assert isinstance(tab, RegPackTab) and page.gsn is not None
        assert 'not a type-approval' in tab.notice.label.text() and 'not a qualified tool' in tab.notice.label.text()
        assert page.kind.findData('regpack') >= 0
        assert [tab.packs.topLevelItem(i).data(0, 256) for i in range(4)] == list(RP.PACKS)
        aeb = tab.packs.topLevelItem(1)
        assert aeb.text(1) == '110' and aeb.text(2) == '104' and aeb.text(3) == '3'
        assert aeb.foreground(3).color().name() == THEME.status('warning')
        # the pack itself, not run: R157's cut-ins use a value read from a figure
        assert tab.pack_id() == 'r157' and len(_rows(tab)) == 22 and tab.verdict.value.text() == 'not run'
        cut = next(r for r in _rows(tab) if r.text(0).startswith('Cut-in: ego 60, other 40'))
        assert cut.text(1) == 'Annex 5, 4.4; 5.2.5.2; Annex 4 Appendix 3' and cut.text(2) == 'not run'
        assert cut.text(3) == UNVERIFIED_MARK and 'dy0' in cut.toolTip(3)
        assert _rows(tab)[1].text(3) == '' and tab.unver.value.text() == '2'
        vals = {tab.values.topLevelItem(i).text(0): tab.values.topLevelItem(i)
                for i in range(tab.values.topLevelItemCount())}
        assert set(vals) == {'pm_dy0', 'pm_vy_max'} and vals['pm_dy0'].text(3) == 'read from a figure'
        assert 'read from a chart or figure' in vals['pm_dy0'].toolTip(3) and not tab.export_btn.isEnabled()
        assert len(tab.flag_widgets) == 2 and 'Annex 4 Appendix 3, Figure 6' in tab.flag_widgets['pm_dy0'].label.text()
        # Euro NCAP lane support: nothing runs, every case says why
        tab.packs.setCurrentItem(tab.packs.topLevelItem(2))
        assert not tab.run_btn.isEnabled() and all(r.text(2) == 'refused' for r in _rows(tab))
        assert 'steering disturbance' in _rows(tab)[0].text(4)
        # run R157's obstacle cases on the fake carla
        tab.packs.setCurrentItem(tab.packs.topLevelItem(0))
        tab.start()
        assert tab.job is None and 'name' in tab.state.text()
        tab.name.setText('gui-r157')
        tab.group.setCurrentIndex(tab.group.findData('obstacle'))
        assert tab.argv()[:4] == ['regpack', 'run', window.project.path, 'r157'] and '--group' in tab.argv()
        tab.start()
        assert pump(qapp, lambda: tab.job.state in FINAL, timeout=240), tab.job.summary()
        assert tab.job.state == 'done', tab.job.log[-10:]
        tab.poll()
        assert tab.result.currentData() == 'gui-r157' and len(_rows(tab)) == 5
        for r in _rows(tab):
            assert r.text(2) in ('passed', 'failed') and r.text(1).startswith('Annex 5, 4.2.2')
            tone = 'good' if r.text(2) == 'passed' else 'critical'
            assert r.foreground(2).color().name() == THEME.status(tone)
        assert tab.verdict.value.text() in ('incomplete', 'failed') and 'PRE-CHECK' in tab.verdict.caption.text()
        # the report: written into the reports folder, listed by the page, disclaimer inside
        out = tab.export_report()
        assert os.path.dirname(out) == R.reports_dir(window.workspace())
        assert 'not a type-approval' in open(out).read() and R.read_meta(out)['kind'] == 'regpack'
        assert any(page.list.topLevelItem(i).data(0, 256) == out for i in range(page.list.topLevelItemCount()))
        # the checklist: statuses as words, nothing passes
        tab.packs.setCurrentItem(tab.packs.topLevelItem(3))
        words = {r.text(2) for r in _rows(tab)}
        assert len(_rows(tab)) == 33 and words == {'covered', 'partial', 'gap'}
        assert tab.result.currentData() == '' and tab.verdict.value.text() == 'not run'
    finally:
        page.deleteLater()
