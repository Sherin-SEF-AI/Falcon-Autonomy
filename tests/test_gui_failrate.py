"""Rare cases ▸ Failure rate and Operating-domain coverage offscreen: an estimate shown with its interval, per km when
an encounter rate is given, refused with its reasons; the estimate queued as the command line; the ODD grid with its
statuses, an entry's cells with their estimate, and the gaps filled — scenarios added (undoable) and searches queued."""
import json
import os

import numpy as np
import pytest

pytestmark = pytest.mark.gui

from test_gui import pump, qapp, slot_exceptions, window  # noqa: E402,F401  (fixtures)
from test_oddcoverage import _cfg, _run, _study  # noqa: E402

from falcon_autonomy.core import failrate as F  # noqa: E402


def _estimate(ws, name, est, by_cell=None):
    d = os.path.join(ws, 'studies', name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'failure-rate.json'), 'w') as fh:
        json.dump({'schema': F.SCHEMA, 'scenario': 'lead_brake', 'estimate': est, 'done': True,
                   'stop_reason': 'the failure threshold was reached', 'runs': {'total': 700, 'lost': 0},
                   'stages': [{}, {}], 'plain_mc': {'runs': 100, 'failures': 0}, 'mc_runs_for_same_rel_se': 4e7,
                   'judge': {'about': 'the objective ≥ 0.5'}, 'by_cell': by_cell or {}}, fh)


def test_failure_rate_and_coverage_sections(window, qapp, tmp_path):
    window.project.path = str(tmp_path / 'p.falcon.json')
    window.save()
    window.service.set_workspace(window.workspace())
    ws = window.workspace()
    rows = [(_run(ws, f'studies/s1/runs/r{i}', _cfg(window.project, 'lead_brake',
                                                    {'speed': 30.0, 'gap': 10 + i, 'decel': 5.0, 'brake_at': 3.0})),
             0.9 if i == 0 else 0.1) for i in range(4)]
    _study(ws, 's1', rows)
    est = F.summary(np.full(400, 1e-5), np.arange(400) < 100)
    cell = 'clear/straight/car/20-40/day'
    _estimate(ws, 'rate-lead_brake', est, {cell: {'runs': 300, 'failures': 80, 'p': 3e-6, 'ci95': [2e-6, 4e-6],
                                                   'status': 'ok', 'study': 'rate-lead_brake'}})
    rc = window.go('rarecases')
    rc.scenario.setCurrentIndex(rc.scenario.findData('lead_brake'))
    sec = rc.rates
    assert sec.study() == 'rate-lead_brake'
    sec.load()
    assert sec.tiles['p'].value.text() == '2.5e-06' and '95 %' in sec.tiles['p'].sub.text()
    assert sec.tiles['runs'].value.text() == '700' and not sec.notice.isVisibleTo(sec)
    sec.per.setCurrentIndex(sec.per.findData('km'))
    sec.encounters.setValue(0.1)
    sec.load()
    assert sec.tiles['rate'].value.text() == '2.5' and '10^7 km' in sec.tiles['rate'].unit.text()
    assert sec.encounters.isVisibleTo(sec) and 'about 40,000,000 runs' in sec.kv.vals['Plain Monte Carlo'].text()
    # refused: its reasons, no number
    _estimate(ws, 'rate-lead_brake-checks', F.summary(np.ones(300), np.zeros(300), defensive=0.1))
    sec.judge.setCurrentIndex(sec.judge.findData('checks'))
    assert sec.tiles['p'].value.text() == '≤ 0.1' and sec.notice.isVisibleTo(sec)
    assert sec.notice.label.text().startswith('Refused: no run failed')
    # queue it: the command line the CLI takes
    sec.judge.setCurrentIndex(sec.judge.findData('objective'))
    sec.stage.setValue(10)
    sec.final.setValue(10)
    sec.stages.setValue(1)
    sec.start()
    job = sec.job
    assert job is not None and job.spec.argv[3:6] == ['failrate', 'run', window.project.path]
    assert '--exposure-per' in job.spec.argv and '--encounters' in job.spec.argv
    window.queue.cancel(job)
    assert pump(qapp, lambda: job.state in ('cancelled', 'failed', 'done'), 60)
    # coverage: the grid, an entry's cells with the estimate, the gaps filled
    cov = rc.coverage
    cov.refresh()
    assert cov.tiles['cells'].value.text() == '6,240' and cov.tiles['too_few'].value.text() == '1'
    assert cov.grid.rowCount() == 6 and cov.grid.columnCount() == 4
    cov.grid.setCurrentCell(0, 0)                              # clear × day
    assert cov.grid.item(0, 0).text().startswith('4 runs · 1 failed')
    assert cov.cells.topLevelItemCount() == 1 and cov.cells.topLevelItem(0).text(6).startswith('3e-06')
    cov.rows.setCurrentIndex(cov.rows.findData('actor'))
    cov.cols.setCurrentIndex(cov.cols.findData('speed'))
    assert cov.grid.rowCount() == 13 and cov.grid.columnCount() == 5
    cov.grid.resize(600, 400)
    assert not cov.grid.grab().isNull()
    n0 = len(window.project.doc['scenarios'])
    cov.n_fill.setValue(2)
    cov.fill()
    assert len(window.project.doc['scenarios']) == n0 + 2 and len(cov.jobs) == 2
    assert all(j.spec.argv[3] == 'search' and 'bayes-pof' in j.spec.argv for j in cov.jobs)
    assert 'scenarios added' in cov.fill_state.vals['Filling'].text()
    for j in cov.jobs:
        window.queue.cancel(j)
    assert pump(qapp, lambda: all(j.state in ('cancelled', 'failed', 'done') for j in cov.jobs), 60)
    window.undo.undo()                                         # the scenarios came as undoable edits
    assert len(window.project.doc['scenarios']) == n0 + 1
