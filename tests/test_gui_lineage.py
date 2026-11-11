"""The Models page's Lineage tab offscreen (ui.lineage_tab): the workspace's models and dataset versions with their
Merkle roots, verifying a version, a proof refused without a model, indexing runs as jobs, the store's summary."""
import json
import os
import sys

import pytest

pytestmark = pytest.mark.gui
pytest.importorskip('pyarrow')

from test_gui import pump, qapp, slot_exceptions, window  # noqa: E402,F401
from test_lineage import make_run  # noqa: E402


def test_lineage_tab_lists_verifies_and_indexes(window, qapp, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'blake3', None)
    from falcon_autonomy.core import datasets as D
    window.project.path = str(tmp_path / 'proj.falcon.json')
    window.save()
    ws = window.workspace()
    a = make_run(ws, 'a')
    b = make_run(ws, 'b', seed=1)
    v = D.make_version({'id': 'ds', 'name': 'Set', 'seed': 0}, [a], {'train': ['runs/a']}, 'drive', 1, ws)
    vp = os.path.join(ws, 'datasets', 'ds', 'versions', 'v001.json')
    os.makedirs(os.path.dirname(vp))
    with open(vp, 'w') as fh:
        json.dump(v, fh)
    page = window.go('models')
    tab = page.lineage
    assert tab is not None and page.tabs.indexOf(tab) >= 0
    tab.refresh()
    assert tab.tree.topLevelItemCount() == 1
    it = tab.tree.topLevelItem(0)
    assert it.text(0) == 'dataset version' and it.text(1) == 'Set v1' and it.text(2).startswith('s256:')
    assert it.toolTip(2) == v['lineage']['merkle_root'] and it.text(4) == '8'
    tab.tree.setCurrentItem(it)
    tab.verify()
    assert pump(qapp, lambda: 'Check' in tab.info.vals, 20)
    assert tab.info.vals['Check'].text() == 'verified'
    assert tab.prove('x.jpg') is None and 'trained model' in tab.note.text()
    n = len(window.queue.jobs)
    assert tab.index_runs() == 1                                    # b: a was indexed by its version
    assert [j.spec.kind for j in window.queue.jobs[n:]] == ['lineage-index']
    window.queue.cancel(window.queue.jobs[-1])
    tab.show_store()
    assert pump(qapp, lambda: 'Store' in tab.info.vals, 20)
    assert 'would remove 0' in tab.info.vals['Store'].text()
    del b
