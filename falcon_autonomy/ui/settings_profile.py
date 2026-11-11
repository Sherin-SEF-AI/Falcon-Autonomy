"""Settings ▸ Performance: the performance profile (core.profiles: standard, or low_vram for a GPU of about 8 GB), the
inference precision and the server's texture pool.

Choosing a profile and pressing Apply writes its values (the server's quality, memory cap and texture pool, the render
budget, the inference precision) into the settings, keeping what it replaced to restore with standard; the fields
above then show them. A section of the Settings page (its SECTIONS hook): section(page) builds it, apply(settings)
writes it unsaved ('' ok, else why not).
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QSpinBox

from ..core import profiles as PR
from .widgets import Form, Section, label


class ProfileSection(Section):
    def __init__(self, page):
        super().__init__('Performance', 'low_vram fits Falcon to a GPU with about 8 GB of video memory: CARLA\'s Low '
                                        'quality, a 512 MB texture pool, a 96 Mpx render budget, FP16 inference where '
                                        'accuracy holds, a 10 GB server cap and smaller reconstructions. Not yet '
                                        'measured on an 8 GB card.')
        self.page = page
        st = page.win.settings if page is not None and getattr(page, 'win', None) is not None else None
        self.settings = st
        f = Form()
        self.profile = QComboBox()
        for name, p in PR.PROFILES.items():
            self.profile.addItem(name, name)
            self.profile.setItemData(self.profile.count() - 1, p['about'], Qt.ItemDataRole.ToolTipRole)
        cur = PR.current(st) if st is not None else 'standard'
        self.profile.setCurrentIndex(max(self.profile.findData(cur), 0))
        self._row(f, 'Profile', self.profile, 'Apply writes its values into the settings; standard puts yours back')
        self.precision = QComboBox()
        for p, text in (('fp32', 'FP32'), ('fp16', 'FP16'), ('auto', 'Auto (FP16 when AP holds within 0.5 points)')):
            self.precision.addItem(text, p)
        prec = (st.get('inference_precision', 'fp32') if st is not None else 'fp32') or 'fp32'
        self.precision.setCurrentIndex(max(self.precision.findData(prec), 0))
        self._row(f, 'Inference precision', self.precision, 'model evaluations (falcon evaluate)')
        self.pool = QSpinBox()
        self.pool.setRange(0, 16384)
        self.pool.setSingleStep(128)
        self.pool.setSuffix(' MB')
        self.pool.setSpecialValueText('CARLA\'s own')
        self.pool.setValue(int((st.get('texture_pool_mb', 0) if st is not None else 0) or 0))
        self._row(f, 'Texture pool', self.pool, 'Unreal\'s texture streaming pool (r.Streaming.PoolSize); from the '
                  'next server start')
        self.state = label(self.state_text(), wrap=True)
        self.state.setProperty('muted', True)
        self._row(f, '', self.state)
        self.profile.currentIndexChanged.connect(self._profile_changed)
        self.add(f)

    def _row(self, form, text, field, note=''):
        if self.page is not None and hasattr(self.page, '_row'):
            return self.page._row(form, text, field, note)
        return form.row(text, field, note)

    def state_text(self) -> str:
        if self.settings is None:
            return ''
        eff = PR.effective(self.settings)
        out = f'In force: {eff["profile"]} — {eff["about"]}.'
        if eff.get('perception'):
            c = eff['perception']
            out += (f' Near-miss mining: batches of {c["batch"]}, {c["detector"]}, depth at {c["depth_input_px"]} px, '
                    f'{c["gpu_memory_fraction"]:.0%} of the GPU.')
        if eff['differs']:
            out += ' Changed since: ' + ', '.join(f'{k} {d["setting"]} (the profile\'s {d["profile"]})'
                                                 for k, d in eff['differs'].items()) + '.'
        return out

    def _profile_changed(self, _i):
        vals = PR.PROFILES[self.profile.currentData()]['values']
        if 'inference_precision' in vals:
            self.precision.setCurrentIndex(max(self.precision.findData(vals['inference_precision']), 0))
        if 'texture_pool_mb' in vals:
            self.pool.setValue(int(vals['texture_pool_mb']))

    def apply(self, settings) -> str:
        """Switch the profile when it changed (its values written), then the precision and pool as shown."""
        name = self.profile.currentData()
        try:
            if name != PR.current(settings):
                PR.set_profile(settings, name, save=False)   # saved with the page's other settings
                self._sync_page(settings)
            settings.set('inference_precision', self.precision.currentData(), save=False)
            settings.set('texture_pool_mb', int(self.pool.value()), save=False)
        except (PR.ProfileError, KeyError, OSError) as e:
            return f'Performance: {e}'
        self.state.setText(self.state_text())
        return ''

    def _sync_page(self, settings):
        """The page's own fields for the values a profile writes (quality, memory cap, render budget), so its Apply
        does not write the old ones back."""
        p = self.page
        if p is None:
            return
        for attr, key, kind in (('quality', 'server_quality', 'text'), ('mem', 'server_memory_gb', 'value'),
                                ('budget', 'render_budget_mpx', 'value')):
            w = getattr(p, attr, None)
            if w is None:
                continue
            if kind == 'text':
                w.setCurrentText(str(settings.get(key)))
            else:
                w.setValue(float(settings.get(key)))


def section(page) -> ProfileSection:
    return ProfileSection(page)
