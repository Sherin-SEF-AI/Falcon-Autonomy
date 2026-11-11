"""Where a study's runs go: a combo of this machine, the whole cluster and each of its pools (docs/cluster.md). A
dialog that queues a study puts it beside its other options and passes placement() on as `--on`; with no cluster set
up it offers only this machine and says so in its tooltip. refusal(scenario, world) says why a study of a
dashcam-derived scenario or world cannot go to the placement chosen (cluster.bundle.placement_refusal: it never leaves
this machine) — the panel shows it and does not queue the study. settings: the window's (a page without a window: the
user's)."""
from __future__ import annotations

from PyQt6.QtWidgets import QComboBox


class PlacementCombo(QComboBox):
    def __init__(self, settings=None, parent=None):
        super().__init__(parent)
        if settings is None:
            from ..core.settings import Settings
            settings = Settings()
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

    def refusal(self, scenario: dict | None, world: dict | None = None) -> str:
        """'' when a study of this scenario (on this world) may run where chosen, else why not (module doc)."""
        from ..cluster.bundle import placement_refusal
        return placement_refusal(self.placement(), scenario, world)
