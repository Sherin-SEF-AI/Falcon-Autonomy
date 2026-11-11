"""Where a study's runs go: a combo of this machine, the whole cluster and each of its pools (docs/cluster.md). A
dialog that queues a study puts it beside its other options and passes placement() on as `--on`; with no cluster set
up it offers only this machine and says so in its tooltip."""
from __future__ import annotations

from PyQt6.QtWidgets import QComboBox


class PlacementCombo(QComboBox):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.reload()

    def reload(self):
        from ..cluster import nodes as N
        keep = self.placement() if self.count() else 'local'
        self.clear()
        self.addItem('This machine', 'local')
        cfg = N.load(self.settings)
        if cfg['nodes']:
            pools = N.pools(cfg)
            total = sum(p['slots'] for p in pools.values())
            self.addItem(f'Cluster ({total} slot{"s" * (total != 1)})', 'cluster')
            for name, p in sorted(pools.items()):
                self.addItem(f'Cluster pool {name} ({p["slots"]} slot{"s" * (p["slots"] != 1)})', f'cluster:{name}')
            self.setToolTip('Where the runs go; the study itself runs here (Settings ▸ Compute lists the nodes)')
        else:
            self.setToolTip('No cluster is set up: add nodes in Settings ▸ Compute to run elsewhere')
        i = self.findData(keep)
        self.setCurrentIndex(i if i >= 0 else 0)
        self.setEnabled(self.count() > 1)

    def placement(self) -> str:
        return self.currentData() or 'local'

    def argv(self) -> list:
        """['--on', WHERE] for a study's command line, or [] on this machine."""
        p = self.placement()
        return [] if p == 'local' else ['--on', p]
