"""Settings ▸ Simulator: the CARLA backend runs use, and each backend's folder and Python (sim.backends).

CARLA 0.9.16's folder and Python stay in the CARLA section above (carla_root, client_python, as before); this section
sets the default backend (a project's simulator.backend overrides it) and the other backends' folders and Pythons
(carla_roots, client_pythons), and shows what each backend can do. A section of the Settings page (its SECTIONS hook):
section(page) builds it, its apply(settings) checks and writes the values unsaved ('' ok, else why not).
"""
from __future__ import annotations

from PyQt6.QtWidgets import QComboBox, QFileDialog, QHBoxLayout, QLineEdit

from ..core.paths import tilde
from ..sim import backends as B
from ..sim import server as S
from . import tasks
from .widgets import Form, Section, button, label


class BackendSection(Section):
    def __init__(self, page):
        super().__init__('Simulator', 'Which CARLA runs your drives and scenarios. CARLA 0.9.16 is the default and '
                                      'what every figure in Falcon was measured on; CARLA 0.10 (Unreal Engine 5) is an '
                                      'adapter not yet run against a real 0.10 server. A project can name its own '
                                      '(its simulator), which overrides the default here.')
        self.page = page
        st = page.win.settings if page is not None and getattr(page, 'win', None) is not None else None
        self.settings = st
        f = Form()
        self.default = QComboBox()
        for b in B.BACKENDS.values():
            self.default.addItem(b.title, b.id)
        self.default.setCurrentIndex(max(self.default.findData(B.default_backend(st)), 0))
        self._row(f, 'Default backend', self.default, 'a project\'s simulator overrides it')
        roots = dict((st.get('carla_roots', {}) if st is not None else {}) or {})
        pys = dict((st.get('client_pythons', {}) if st is not None else {}) or {})
        self.roots, self.pys, self.status = {}, {}, {}
        for b in B.BACKENDS.values():
            if b.id == B.DEFAULT:
                self._row(f, b.title, label('folder and Python: the CARLA section above', muted=True))
                continue
            e = QLineEdit(roots.get(b.id, ''))
            found = S.candidate_roots(backend=b.id)
            e.setPlaceholderText(tilde(found[0]) + ' (found)' if found else f'holds {b.binary}')
            rb = QHBoxLayout()
            rb.addWidget(e, 1)
            rb.addWidget(button('Browse…', lambda _=False, bid=b.id: self.browse(bid)))
            self._row(f, f'{b.title} folder', rb, f'holds {b.binary}')
            p = QLineEdit(pys.get(b.id, ''))
            p.setPlaceholderText('a venv beside the folder')
            pb = QHBoxLayout()
            pb.addWidget(p, 1)
            pb.addWidget(button('Check', lambda _=False, bid=b.id: self.check(bid)))
            self._row(f, f'{b.title} Python', pb, f'the venv with its client: {b.wheel}')
            s = label('', muted=True, wrap=True)
            self._row(f, '', s)
            self.roots[b.id], self.pys[b.id], self.status[b.id] = e, p, s
        self.table = label(self.capability_text(), wrap=True)    # in full (a long muted label folds behind 'more')
        self.table.setProperty('muted', True)
        self._row(f, 'What each can do', self.table)
        self.add(f)

    def _row(self, form, text, field, note=''):
        if self.page is not None and hasattr(self.page, '_row'):
            return self.page._row(form, text, field, note)        # the command palette finds the row
        return form.row(text, field, note)

    @staticmethod
    def capability_text() -> str:
        lines = []
        for b in B.BACKENDS.values():
            caps = B.capabilities(b.id)
            have = B.maps(b.id)
            lines.append(f'{b.title}: maps {", ".join(have) if have else "as the server lists"}; '
                         + ', '.join(f'{B.CAP_TEXT[c]} {"yes" if v else "no"}' for c, v in caps.items())
                         + ('' if b.measured else ' (assumed until `falcon backend probe`)'))
        return '\n'.join(lines)

    def browse(self, bid: str):
        d = QFileDialog.getExistingDirectory(self, f'{B.get(bid).title} folder', self.roots[bid].text())
        if d:
            self.roots[bid].setText(d)

    def check(self, bid: str):
        root = self.roots[bid].text().strip() or B.root_for(self.settings, bid)
        py = self.pys[bid].text().strip() or S.client_python(root)
        self.status[bid].setText('checking…')
        tasks.run(self, lambda: S.check_client(py, backend=bid), lambda r: self.status[bid].setText(
            ('✓ ' if r['ok'] else '⚠ ') + r['text']))

    def values(self) -> tuple:
        """(default backend, carla_roots, client_pythons) as entered."""
        roots = {k: e.text().strip() for k, e in self.roots.items() if e.text().strip()}
        pys = {k: e.text().strip() for k, e in self.pys.items() if e.text().strip()}
        return self.default.currentData(), roots, pys

    def apply(self, settings) -> str:
        """Check and write the section's values (unsaved): '' when done, else why not."""
        default, roots, pys = self.values()
        for bid, root in roots.items():
            found = B.detect_root(root)
            if found != bid:
                return (f'{root} is a {B.get(found).title} folder, not {B.get(bid).title}.' if found else
                        f'{root} is not a {B.get(bid).title} folder ({B.get(bid).binary} is missing).')
        keep_r = {k: v for k, v in (settings.get('carla_roots', {}) or {}).items() if k not in self.roots}
        keep_p = {k: v for k, v in (settings.get('client_pythons', {}) or {}).items() if k not in self.pys}
        settings.set('default_backend', default, save=False)
        settings.set('carla_roots', dict(keep_r, **roots), save=False)
        settings.set('client_pythons', dict(keep_p, **pys), save=False)
        return ''


def section(page) -> BackendSection:
    return BackendSection(page)
