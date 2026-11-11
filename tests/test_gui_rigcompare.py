"""Rigs ▸ Vantage variants offscreen: the import dialog adds the chosen variants as rigs in one undoable edit (the
.vantage file untouched), the comparison dialog builds `falcon rigs compare` and queues it, and a study's results
show every rig's numbers, the paired verdicts and write the Vantage design-log note to its own file."""
import hashlib
import json
import os

import pytest
from PyQt6.QtCore import Qt

pytestmark = pytest.mark.gui

from test_gui import pump, qapp, slot_exceptions, window  # noqa: E402,F401  (fixtures)
from test_rigcompare import SAMPLE, _synthetic_study  # noqa: E402


@pytest.fixture
def page(window):
    from falcon_autonomy.ui.pages.rigs import RigsPage
    p = RigsPage(window)
    p.resize(1300, 800)
    p.show()
    yield p
    p.close()
    p.deleteLater()


def test_import_compare_and_results(page, window, qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QDialog
    from falcon_autonomy.core import rigcompare as RC
    from falcon_autonomy.core.provenance import write_json
    from falcon_autonomy.ui import rigcompare as U
    before = hashlib.sha256(open(SAMPLE, 'rb').read()).hexdigest()
    seen = {}

    def run_import(dlg):
        seen['variants'] = dlg.variants.count()
        for i in range(dlg.variants.count()):                     # the two 4-cam variants only
            it = dlg.variants.item(i)
            it.setCheckState(Qt.CheckState.Checked if i in (1, 2) else Qt.CheckState.Unchecked)
        dlg.scale.setValue(0.1)
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(U.VantageImportDialog, 'exec', run_import)
    n0 = len(window.project.doc['rigs'])
    rigs = U.import_vantage(page, SAMPLE)
    assert seen['variants'] == 5 and len(rigs) == 2
    doc = window.project.doc
    assert len(doc['rigs']) == n0 + 2 and doc['rigs'][-1]['vantage']['variant'] == '4-cam, mirror cameras 5° off'
    assert doc['rigs'][-1]['sensors'][0]['width'] == 192
    window.undo.undo()
    assert len(window.project.doc['rigs']) == n0                  # one edit, one undo
    window.undo.redo()
    assert hashlib.sha256(open(SAMPLE, 'rb').read()).hexdigest() == before
    # the comparison dialog: the Vantage rigs checked, a detector needed
    dlg = U.RigCompareDialog(window)
    assert U.checked(dlg.rigs) == [r['id'] for r in window.project.doc['rigs'][-2:]]
    assert 'Register a detector' in dlg.problem()
    window.project.doc['models'] = [{'id': 'syn', 'name': 'Synthetic', 'kind': 'detector', 'runtime': 'python',
                                     'entry': 'falcon_autonomy.plugins.synthetic_detector:SyntheticDetector',
                                     'licence': 'GPL-3.0-or-later'}]
    dlg = U.RigCompareDialog(window)
    assert dlg.problem() == ''
    argv = dlg.argv()
    assert argv[:2] == ['rigs', 'compare'] and argv.count('--rig') == 2 and '--model' in argv
    assert argv.count('--scenario') == 2 and '--fake-carla' in argv
    # the results of a (synthetic) study in the workspace
    rigs_, frames, runs = _synthetic_study(tmp_path / 'syn', weak=(45.0, 0.2))
    rep = RC.analyse(rigs_, frames, runs, fps=10, n_boot=100, ap=False)
    rep.update(study='syn', model={'id': 'm'}, synthetic_model=True, rigs=[], limits=['a limit'])
    d = os.path.join(window.workspace(), 'rigstudies', 'syn')
    os.makedirs(d)
    write_json(os.path.join(d, 'study.json'), {'name': 'syn', 'rigs': rigs_})
    write_json(os.path.join(d, 'report.json'), rep)
    res = U.RigStudyDialog(d)
    assert res.table.topLevelItemCount() > 10 and res.table.columnCount() == 3
    verdicts = [res.pairs[0].topLevelItem(i).text(5) for i in range(res.pairs[0].topLevelItemCount())]
    assert 'worse' in verdicts
    from PyQt6.QtWidgets import QMessageBox
    warned = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *a, **k: warned.append(a[2]))
    assert res.export_note(str(tmp_path / 'x.vantage')) == '' and 'never into' in warned[0]
    out = res.export_note(str(tmp_path / 'note.json'))
    assert json.load(open(out))['schema'] == 'vantage.design_log/1'
    res.deleteLater()
    pump(qapp, timeout=0.1)
