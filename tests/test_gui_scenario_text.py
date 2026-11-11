"""Scenarios ▸ Describe a scenario (offscreen Qt, the fake carla's queue): Falcon's draft offline, its preview and
claims, Add and run (an undoable insert and a queued run), and a finished run checked into the claims table."""
import os

import pytest

from test_gui import pump, qapp, slot_exceptions, window  # noqa: F401 (the GUI tests' fixtures)
from test_scenario_text import synthetic_run

pytestmark = pytest.mark.gui


def test_describe_draft_add_run_and_check(window, qapp, tmp_path):  # noqa: F811
    import numpy as np
    page = window.go('scenarios')
    sec = page.describe
    window.project.save(str(tmp_path / 'p.falcon.json'))
    assert not sec.run_btn.isEnabled()
    assert sec.write(wait=True) is None and 'Describe' in sec.status.text()
    sec.text.setPlainText('a car brakes hard ahead at night in the rain')
    rec = sec.write(wait=True)                         # the tests' assistant is off: Falcon's draft
    assert rec['status'] == 'valid' and rec['mode'] == 'phrases'
    assert 'Falcon\'s draft' in sec.status.text() and '"lead"' in sec.preview.toPlainText()
    assert sec.claims.topLevelItemCount() == 4 and sec.claims.topLevelItem(0).text(2) == 'not run yet'
    assert sec.run_btn.isEnabled()
    n = len(window.project.doc['scenarios'])
    job = sec.add_and_run()
    assert job is not None and job.spec.kind == 'drive' and len(window.project.doc['scenarios']) == n + 1
    assert window.project.doc['scenarios'][-1]['source']['description'] == rec['description']
    window.queue.cancel(job)
    window.undo.undo()                                  # the insert is one undoable edit
    assert len(window.project.doc['scenarios']) == n
    # a run of it (synthetic): checked into the table
    t = np.arange(160) * 0.05
    v = np.where(t < 3.0, 8.0, np.maximum(0.0, 8.0 - 6.0 * (t - 3.0)))
    lead = (30.0 + np.concatenate([[0.0], np.cumsum(v[:-1] * 0.05)]), np.zeros(160), 0.0, v)
    d = synthetic_run(str(tmp_path), 'r', {'lead': lead}, {'lead': 'car'},
                      weather={'sun_altitude_angle': -30.0, 'precipitation': 60.0})
    rep = sec.check(d)
    assert rep['counts'] == {'met': 4, 'not_met': 0, 'not_checkable': 0}
    assert [sec.claims.topLevelItem(i).text(2) for i in range(4)] == ['met'] * 4
    assert 'every claim met' in sec.status.text()
    assert sec.check(str(tmp_path)) is None and 'Cannot check' in sec.status.text()
    assert os.path.isdir(os.path.join(window.workspace(), 'scenario-text'))
