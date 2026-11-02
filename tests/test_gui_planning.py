"""The planner benchmark on Scenarios and "Make a scenario from this event" on Replay (offscreen Qt, the fake carla's
queue; synthetic data only)."""
import json
import os
import shutil

import pytest

from test_gui import pump, qapp, slot_exceptions, window  # noqa: F401 (the GUI tests' fixtures)
from test_incident import XODR, recon  # noqa: F401

pytestmark = pytest.mark.gui


def _results(bdir):
    from falcon_autonomy.core import planbench as PB
    os.makedirs(bdir, exist_ok=True)

    def row(nc, progress):
        return {'terms': {'nc': {'score': nc}, 'dac': {'score': 1.0}, 'ddc': {'score': 1.0}, 'ttc': {'score': 1.0},
                          'comfort': {'score': 1.0}}, 'progress_m': progress}
    res = PB.aggregate({'follower': {'r0': row(1.0, 50.0), 'r1': row(1.0, 40.0)},
                        'none': {'r0': row(0.0, 60.0), 'r1': row(1.0, 40.0)}}, 'follower', n_boot=50)
    with open(os.path.join(bdir, 'results.json'), 'w', encoding='utf-8') as fh:
        json.dump(res, fh)


def test_scenarios_page_queues_and_shows_the_benchmark(window, qapp, tmp_path):  # noqa: F811
    page = window.go('scenarios')
    b = page.bench
    b.planners.setText('follower, none')
    b.name.setText('gui-bench')
    b.route_set.setCurrentIndex(b.route_set.findData('falcon-pdm-mini'))
    window.project.save(str(tmp_path / 'p.falcon.json'))
    argv = b.argv()
    assert argv[:2] == ['bench', 'run'] and argv[argv.index('--set') + 1] == 'falcon-pdm-mini'
    assert argv.count('--planner') == 2
    job = b.run()
    assert job is not None and job.spec.kind == 'bench' and 'bench' in job.spec.argv
    window.queue.cancel(job)
    _results(os.path.join(window.workspace(), 'benchmarks', 'gui-bench'))
    res = b.show()
    assert res is not None and b.table.topLevelItemCount() == 2
    assert b.table.topLevelItem(0).text(0) == 'follower'
    assert 'none − follower' in b.status.text()
    b.name.setText('../bad')
    assert b.run() is None and 'letters' in b.status.text()


def test_replay_page_makes_a_scenario_from_an_event(window, qapp, tmp_path, recon):  # noqa: F811
    rec, det = recon
    p = window.project
    p.save(str(tmp_path / 'p.falcon.json'))
    wdir = os.path.join(p.workspace, 'worlds', 'w-test')
    os.makedirs(wdir, exist_ok=True)
    with open(os.path.join(wdir, 'world.xodr'), 'w', encoding='utf-8') as fh:
        fh.write(XODR)
    p.doc['worlds'].append({'id': 'w-test', 'name': 'Synthetic road', 'kind': 'osm', 'xodr': 'worlds/w-test/world.xodr',
                            'osm': {'bbox': [-0.01, -0.01, 0.01, 0.01], 'centre': [0.0, 0.0], 'handedness': 'RHT'}})
    p.save()
    panel = window.go('replay').recon
    panel.refresh_worlds()
    assert panel.make_incident() is None and 'reconstruct' in panel.note.text()
    panel.rec_dir = os.path.join(p.workspace, 'recon', 'dashcam-synthetic')
    shutil.copytree(rec, panel.rec_dir)
    panel.dash_world.setCurrentIndex(panel.dash_world.findData('w-test'))
    panel.detections = det
    job = panel.make_incident()
    assert job is not None and job.spec.kind == 'incident' and '--detections' in job.spec.argv
    assert pump(qapp, lambda: job.state in ('done', 'failed', 'cancelled'), timeout=120), job.state
    assert job.state == 'done', job.log
    assert pump(qapp, lambda: panel.search_btn.isEnabled(), timeout=5)
    assert 'car' in panel.info.vals['Incident'].text() and 'pedestrian' in panel.info.vals['Incident'].text()
    s = panel.search_incident()
    assert s is not None and 'variations.json' in ' '.join(s.spec.argv)
    window.queue.cancel(s)
