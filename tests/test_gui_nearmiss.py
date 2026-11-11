"""The Replay page's Near-misses section offscreen (ui.dashcam_ingest): a synthetic clip ingested (as in
test_dashcam_nearmiss), the ranked list, the clip player, Make scenario as a job that writes the OpenSCENARIO file on
an imported road, Queue variations as a local study job, and Ingest queued as a resumable job."""
import datetime as dt
import os
import shutil

import pytest

pytestmark = [pytest.mark.gui,
              pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='needs ffmpeg')]
pytest.importorskip('scipy')

from test_dashcam_nearmiss import IST, SETTINGS, UTC0, FakeDepth, make_clip  # noqa: E402
from test_gui import pump, qapp, slot_exceptions, window  # noqa: E402,F401


@pytest.fixture
def ingested(tmp_path):
    from falcon_autonomy.core import dashcam_ingest as I
    from falcon_autonomy.core.dashcam_detect import FileDetector
    folder = str(tmp_path / 'footage')
    clip, side, depths = make_clip(os.path.join(folder, 'Movie_F'), (UTC0 + dt.timedelta(hours=2)).astimezone(IST))
    I.ingest(folder, dict(SETTINGS, detector=f'file:{side}'), detector=FileDetector(side), depth=FakeDepth(depths))
    return folder


def test_nearmiss_section(window, qapp, tmp_path, ingested):
    from test_incident import XODR
    window.project.path = str(tmp_path / 'proj.falcon.json')
    wdir = os.path.join(window.workspace(), 'worlds', 'w-test')
    os.makedirs(wdir, exist_ok=True)
    with open(os.path.join(wdir, 'world.xodr'), 'w') as fh:
        fh.write(XODR)
    window.project.doc['worlds'].append({'id': 'w-test', 'name': 'Synthetic road', 'kind': 'osm',
                                         'xodr': 'worlds/w-test/world.xodr',
                                         'osm': {'bbox': [-0.01, -0.01, 0.01, 0.01], 'centre': [0.0, 0.0],
                                                 'handedness': 'RHT'}})
    window.save()
    page = window.go('replay')
    tab = page.nearmiss
    assert tab is not None
    tab.refresh_worlds()
    assert 'Apache-2.0' in tab.models.text() and 'refused' in tab.models.text()
    tab.folder.setText(ingested)
    assert tab.refresh() >= 3
    assert tab.tree.topLevelItem(0).text(0) == '1' and tab.tree.topLevelItem(0).text(1) == 'car'
    assert 'events ranked' in tab.status.text()
    assert any('heuristic' in tab.tree.topLevelItem(i).text(1) for i in range(tab.tree.topLevelItemCount()))
    tab.tree.setCurrentItem(tab.tree.topLevelItem(0))
    assert tab.player.cap is not None and tab.player.slider.maximum() > 10
    assert tab.player.show_frame(5) and tab.player.view.pixmap() is not None
    tab.player.toggle()
    assert pump(qapp, lambda: tab.player.slider.value() > 2, 5)
    tab.player.toggle()
    assert tab.scenario_btn.isEnabled()
    job = tab.make_scenario()
    assert job is not None and pump(qapp, lambda: job.state in ('done', 'failed', 'cancelled'), 120)
    assert job.state == 'done', job.log
    assert pump(qapp, lambda: tab.vary_btn.isEnabled(), 5)
    from falcon_autonomy.core import incident as INC
    inc_dir = INC.incident_dir(window.workspace(), tab.incident)
    assert os.path.isfile(os.path.join(inc_dir, 'incident.xosc'))
    vj = tab.queue_variations()
    assert vj.spec.kind == 'study' and '--scenario' in vj.spec.argv and vj.spec.meta['source']['format'] == 'dashcam'
    window.queue.cancel(vj)
    tab.device.setCurrentIndex(tab.device.findData('cpu'))
    ij = tab.ingest()
    assert ij.spec.kind == 'dashcam-ingest' and not ij.spec.needs_gpu and ij.spec.resumable
    assert ij.spec.argv[ij.spec.argv.index('ingest') + 1] == ingested
    window.queue.cancel(ij)
    assert pump(qapp, lambda: ij.state in ('done', 'failed', 'cancelled'), 60)
